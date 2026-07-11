"""
BISINDO v10 — Flask API untuk app Flutter SnapSign
==================================================
Menjalankan model RGB v10 (MobileNetV2 + TSM + CBAM) sebagai backend yang
COCOK dengan detection_service.dart (protokol app_skeleton.py):

  Siklus per-client : prepare(2s) -> collect(5s) -> hold(3s) -> prepare ...
  POST /predict      : { "frame": "<base64 JPEG>", "client_id": "..." }
                       -> { phase, remaining, phase_duration, label, confidence,
                            top3, pose_ok, hands, frames, hand_frames }
  GET  /health       : cek hidup
  POST /reset        : reset siklus client

Pipeline tiap frame: decode JPEG -> (opsional normalisasi cahaya) -> segmentasi
selfie -> simpan (saat collect). Akhir collect: trim + 16 frame margin 0.15 ->
model -> softmax -> top3. SAMA dengan realtime_v10.py (fungsi di-reuse).

Jalankan (HP & PC satu WiFi):
    python server_v10.py --checkpoint best_stage2_v10.pth --port 8000
    # kalau cahaya HP beda jauh:
    python server_v10.py --light clahe+gray
Lalu set AppConstants.apiBaseUrl = http://<IP-PC>:8000  (cek IP: ipconfig)
"""

import os, time, base64, threading, argparse, traceback
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
except Exception:
    CORS = None
import mediapipe as mp

import realtime_v10 as R   # model + segmentasi + sampling (reuse, tidak jalankan main)

# ── Konfigurasi siklus (samakan dgn app_skeleton agar UI Flutter pas) ────────
PREPARE_SEC = 2.0
COLLECT_SEC = 5.0
HOLD_SEC    = 3.0

def _phase_duration(state):
    return {"prepare": PREPARE_SEC, "collect": COLLECT_SEC, "hold": HOLD_SEC}.get(state, 0.0)


# ── Segmentasi + deteksi tangan (thread-safe via lock; MediaPipe tdk thread-safe) ─
_mp_lock = threading.Lock()
_selfie  = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
_hands   = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                    model_complexity=0, min_detection_confidence=0.5)
_dilate_k = np.ones((R.MASK_DILATE, R.MASK_DILATE), np.uint8)


def center_crop_square(bgr):
    """Crop tengah frame jadi square sebelum resize 224×224.
    Agar frame portrait (mobile) dan landscape (webcam) punya proporsi sama."""
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return bgr[y0:y0+s, x0:x0+s]


def segment_frame(bgr, mode='blur'):
    """Selfie segmentation + refine. Center-crop ke square dulu agar proporsi
    konsisten antara frame portrait (mobile) dan landscape (webcam)."""
    bgr = center_crop_square(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with _mp_lock:
        mask = _selfie.process(rgb).segmentation_mask
    binm = cv2.dilate((mask > 0.3).astype(np.uint8), _dilate_k, iterations=1)
    mask = np.clip(np.maximum(mask, binm.astype(np.float32) * 0.8), 0.0, 1.0)
    seg = R.composite(bgr, mask, mode)
    return seg, float(mask.mean())


def count_hands(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with _mp_lock:
        res = _hands.process(rgb)
    return len(res.multi_hand_landmarks) if res.multi_hand_landmarks else 0


# ── State machine per-client ─────────────────────────────────────────────────
class ClientState:
    def __init__(self):
        self.state = "prepare"; self.phase_start = time.time()
        self.frames = []        # frame tersegmentasi 224 (saat collect)
        self.hand_frames = 0
        self.label = ""; self.confidence = 0.0; self.top3 = []
        self.lock = threading.Lock()

    def reset(self):
        self.state = "collect"; self.phase_start = time.time()
        self.frames = []; self.hand_frames = 0
        self.label = ""; self.confidence = 0.0; self.top3 = []


_clients = defaultdict(ClientState)
_MODEL = None; _CLASSES = None; _DEVICE = None
_LIGHT = 'none'; _MODE = 'blur'
# Un-mirror frame dari app: HANYA untuk APK lama yang masih flipHorizontal.
# Jangan aktifkan bersamaan dengan app yang sudah diperbaiki (double flip!).
_UNMIRROR = False
# Grad-CAM overlay di montage /preview (mode demo/sidang). Menambah ~1-2 detik
# per prediksi karena butuh forward+backward ekstra -> default OFF.
_GRADCAM = False


# Montage terakhir disimpan di memory — tidak ada file/database
_last_montage_jpg: bytes = b''
_last_montage_lock = threading.Lock()

CONF_THRESHOLD = 50.0   # % minimum agar prediksi dianggap valid


def gradcam_tiles(seg16, x, target_idx, layer='cbam'):
    """Grad-CAM (Selvaraju dkk., 2017) per frame untuk montage /preview.

    layer='features' -> ukur SEBELUM CBAM (backbone mentah)
    layer='cbam'      -> ukur SETELAH CBAM (default; ini yg mencerminkan
                          efek nyata channel+spatial attention CBAM)

    Butuh forward+backward ekstra di luar inference_mode.
    """
    acts = {}

    def _hook(_m, _i, out):
        acts['feat'] = out

    target_module = _MODEL.features if layer == 'features' else _MODEL.cbam
    h = target_module.register_forward_hook(_hook)
    try:
        with torch.enable_grad():
            logits = _MODEL(x)                     # (1, num_classes)
            score = logits[0, int(target_idx)]
            feat = acts['feat']                    # (T, C, 7, 7)
            grads = torch.autograd.grad(score, feat)[0]
    finally:
        h.remove()

    w = grads.mean(dim=(2, 3), keepdim=True)       # bobot per-channel (GAP gradien)
    cam = torch.relu((w * feat).sum(dim=1))        # (T, 7, 7)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)  # normalisasi global antar frame

    tiles = []
    for f, c in zip(seg16, cam):
        heat = cv2.applyColorMap(
            (cv2.resize(c, (f.shape[1], f.shape[0])) * 255).astype(np.uint8),
            cv2.COLORMAP_JET)
        tiles.append(cv2.addWeighted(f, 0.55, heat, 0.45, 0))
    return tiles


def predict_clip(frames224):
    """list frame BGR 224 tersegmentasi -> (label, conf_pct, top3[[label,frac]])."""
    global _last_montage_jpg
    if len(frames224) < 4:
        print(f"  [debug] frames terkumpul = {len(frames224)} (< 4) -> app kirim frame terlalu sedikit")
        return "Tidak terdeteksi", 0.0, []
    src, peak = R.trim_active(frames224)
    print(f"  [debug] frames = {len(frames224)} | motion peak = {peak:.2f} (gate 0.15)")
    if peak < 0.15:                         # gate dilonggarkan (HP CPU/frame sedikit)
        return "Tidak terdeteksi", 0.0, []
    f16 = R.sample_clip_eval(src, R.NUM_FRAMES, 0.15)
    x = R.frames_to_tensor(f16, _DEVICE)
    with torch.inference_mode():
        probs = F.softmax(_MODEL(x), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1][:3]
    top3  = [[_CLASSES[i], round(float(probs[i]), 4)] for i in order]
    label = _CLASSES[order[0]]
    conf  = float(probs[order[0]]) * 100.0
    if conf < CONF_THRESHOLD:
        return "Belum dikenali", conf, top3

    # ── Montage 16 frame di memory (buka /preview di browser) ────────────
    try:
        header = (f"{label} ({conf:.1f}%)  "
                  f"raw={len(frames224)}fr trim={len(src)}fr peak={peak:.2f}")
        mont = R.make_montage(f16, header=header)
        ok, buf = cv2.imencode('.jpg', mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            with _last_montage_lock:
                _last_montage_jpg = buf.tobytes()
    except Exception:
        pass

    return label, conf, top3


app = Flask(__name__)
if CORS:
    CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "best_stage2_v10 (RGB MobileNetV2+TSM+CBAM)",
                    "type": "rgb-video", "classes": R.NUM_CLASSES,
                    "num_frames": R.NUM_FRAMES, "collect_sec": COLLECT_SEC,
                    "hold_sec": HOLD_SEC, "device": str(_DEVICE), "timestamp": time.time()})


@app.route("/labels", methods=["GET"])
def labels():
    return jsonify({"class_names": _CLASSES})


@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()
    try:
        body = request.get_json(force=True, silent=True) or {}
        fb = body.get("frame", "")
        client_id = body.get("client_id", request.remote_addr)
        if not fb:
            return jsonify({"error": "Field 'frame' wajib (base64 JPEG)"}), 400
        if "," in fb:
            fb = fb.split(",", 1)[1]
        jpeg = base64.b64decode(fb)
        arr = np.frombuffer(jpeg, np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({"error": "gagal decode JPEG"}), 422
        if _UNMIRROR:
            bgr = cv2.flip(bgr, 1)
    except Exception as e:
        return jsonify({"error": f"request tidak valid: {e}"}), 400

    try:
        if _LIGHT != 'none':
            bgr = R.normalize_lighting(bgr, _LIGHT)
        seg, coverage = segment_frame(bgr, _MODE)
        n_hands = count_hands(bgr)
        pose_ok = coverage > 0.04        # ada orang di frame
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"gagal proses gambar: {e}"}), 422

    cs = _clients[client_id]
    with cs.lock:
        now = time.time(); elapsed = now - cs.phase_start

        if cs.state == "prepare":
            if elapsed >= PREPARE_SEC:
                cs.state = "collect"; cs.phase_start = now
                cs.frames = []; cs.hand_frames = 0
        elif cs.state == "collect":
            cs.frames.append(cv2.resize(seg, R.IMG_SIZE))
            if n_hands > 0:
                cs.hand_frames += 1
            if elapsed >= COLLECT_SEC:
                label, conf, top3 = predict_clip(cs.frames)
                cs.label, cs.confidence, cs.top3 = label, conf, top3
                cs.state = "hold"; cs.phase_start = now; cs.frames = []
                print(f"[{client_id}] -> {label} ({conf:.1f}%)  top3={top3}")
        else:  # hold
            if elapsed >= HOLD_SEC:
                cs.state = "prepare"; cs.phase_start = now
                cs.frames = []; cs.hand_frames = 0

        phase = cs.state
        pd = _phase_duration(phase)
        remaining = max(0.0, pd - (now - cs.phase_start))
        resp = {
            "phase": phase, "remaining": round(remaining, 1), "phase_duration": pd,
            "label": cs.label, "confidence": round(cs.confidence, 2), "top3": cs.top3,
            "pose_ok": pose_ok, "hands": n_hands,
            "frames": len(cs.frames), "hand_frames": cs.hand_frames,
            "elapsed_ms": round((time.time() - start) * 1000, 1),
        }
    return jsonify(resp)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Terima SATU klip (banyak frame sekaligus) dari HP -> prediksi.
    Body: { "frames": ["<b64 jpeg>", ...], "client_id": "..." }
    App membuffer frame lokal saat rekam lalu kirim semua di sini (frame padat,
    tanpa bottleneck kirim per-frame)."""
    global _last_montage_jpg
    body = request.get_json(force=True, silent=True) or {}
    frames_b64 = body.get("frames", [])
    full = []
    for fb in frames_b64:
        if isinstance(fb, str) and "," in fb:
            fb = fb.split(",", 1)[1]
        try:
            jpeg = base64.b64decode(fb)
            bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            bgr = None
        if bgr is not None:
            if _UNMIRROR:
                bgr = cv2.flip(bgr, 1)
            full.append(bgr)

    n = len(full)
    if n < 4:
        return jsonify({"label": "Tidak terdeteksi", "confidence": 0.0,
                        "top3": [], "frames": n})

    small = [cv2.resize(f, R.IMG_SIZE) for f in full]
    peak = float(R.motion_energy(small).max())
    if peak < 0.15:
        print(f"[batch] frames={n} peak={peak:.2f} -> diam")
        return jsonify({"label": "Tidak terdeteksi", "confidence": 0.0,
                        "top3": [], "frames": n, "peak": round(peak, 2)})

    # sampel 16 indeks merata tanpa margin (realtime: gestur dimulai sejak awal window)
    import time as _t
    margin = 0.0
    start = int(n * margin); end = int(n * (1 - margin))
    step = max(end - start, 1) / float(R.NUM_FRAMES)
    idx16 = [min(int(start + i * step + step / 2), n - 1) for i in range(R.NUM_FRAMES)]
    t0 = _t.perf_counter()
    seg16 = []
    for i in idx16:
        s, _ = segment_frame(full[i], _MODE)
        seg16.append(cv2.resize(s, R.IMG_SIZE))
    t1 = _t.perf_counter()

    x = R.frames_to_tensor(seg16, _DEVICE)
    with torch.inference_mode():
        probs = F.softmax(_MODEL(x), dim=1)[0].cpu().numpy()
    t2 = _t.perf_counter()
    print(f"  [timing] segmentasi={1000*(t1-t0):.0f}ms | inferensi={1000*(t2-t1):.0f}ms")
    order = probs.argsort()[::-1][:3]
    top3 = [[_CLASSES[i], round(float(probs[i]), 4)] for i in order]
    label = _CLASSES[order[0]]; conf = float(probs[order[0]]) * 100.0
    print(f"[batch] frames={n} peak={peak:.1f} -> {label} ({conf:.1f}%)  top3={top3}")

    # montage 16 frame utk debug via /preview — disimpan juga saat gagal (<50%)
    try:
        header = f"{label} ({conf:.1f}%)  batch={n}fr peak={peak:.1f}"
        tiles = seg16
        if _GRADCAM:
            t_cam = _t.perf_counter()
            tiles = gradcam_tiles(seg16, x, order[0])
            header += f"  [Grad-CAM {1000*(_t.perf_counter()-t_cam):.0f}ms]"
        mont = R.make_montage(tiles, header=header)
        ok, buf = cv2.imencode('.jpg', mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            with _last_montage_lock:
                _last_montage_jpg = buf.tobytes()
    except Exception:
        pass

    if conf < CONF_THRESHOLD:
        return jsonify({"label": "Belum dikenali", "confidence": round(conf, 2),
                        "top3": top3, "frames": n, "peak": round(peak, 2)})
    return jsonify({"label": label, "confidence": round(conf, 2),
                    "top3": top3, "frames": n, "peak": round(peak, 2)})


@app.route("/preview.jpg", methods=["GET"])
def preview_jpg():
    """Gambar montage mentah (dipolling oleh halaman /preview)."""
    from flask import Response
    with _last_montage_lock:
        data = _last_montage_jpg
    if not data:
        return Response(status=204)
    return Response(data, mimetype='image/jpeg',
                    headers={'Cache-Control': 'no-store'})


@app.route("/preview", methods=["GET"])
def preview():
    """Tampilkan 16 frame terakhir yang dikirim ke model — auto refresh."""
    from flask import send_file
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "preview.html"))


@app.route("/reset", methods=["POST"])
def reset_cycle():
    body = request.get_json(force=True, silent=True) or {}
    client_id = body.get("client_id", request.remote_addr)
    if client_id in _clients:
        with _clients[client_id].lock:
            _clients[client_id].reset()
    return jsonify({"status": "siklus direset", "client_id": client_id})


# ── Uji model lewat UPLOAD VIDEO (halaman web sederhana) ─────────────────────
def _predict_video_file(path):
    """Baca video -> ambil 16 frame (margin 0.15) -> segmentasi -> prediksi."""
    global _last_montage_jpg
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 4:
        cap.release()
        return "Tidak terdeteksi", 0.0, [], total
    margin = 0.15
    start = int(total * margin); end = int(total * (1 - margin))
    step = max(end - start, 1) / float(R.NUM_FRAMES)
    seg16 = []
    for i in range(R.NUM_FRAMES):
        idx = min(int(start + i * step + step / 2), total - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if not ret:
            continue
        s, _ = segment_frame(f, _MODE)
        seg16.append(cv2.resize(s, R.IMG_SIZE))
    cap.release()
    if len(seg16) < 4:
        return "Tidak terdeteksi", 0.0, [], total

    x = R.frames_to_tensor(seg16, _DEVICE)
    with torch.inference_mode():
        probs = F.softmax(_MODEL(x), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1][:3]
    top3 = [[_CLASSES[i], round(float(probs[i]), 4)] for i in order]
    label = _CLASSES[order[0]]; conf = float(probs[order[0]]) * 100.0

    try:
        mont = R.make_montage(seg16, header=f"{label} ({conf:.1f}%)")
        ok, buf = cv2.imencode('.jpg', mont, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            with _last_montage_lock:
                _last_montage_jpg = buf.tobytes()
    except Exception:
        pass

    if conf < CONF_THRESHOLD:
        return "Belum dikenali", conf, top3, total
    return label, conf, top3, total


_UPLOAD_HTML = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SnapSign - Uji Model</title>
<style>
  *{box-sizing:border-box}
  body{font-family:Segoe UI,Arial,sans-serif;background:#f4f6f9;margin:0;padding:20px;color:#1b2733}
  .card{max-width:600px;margin:0 auto;background:#fff;border-radius:16px;padding:24px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:#777;font-size:13px;margin-bottom:16px}
  .tabs{display:flex;gap:8px;margin-bottom:18px}
  .tab{flex:1;padding:10px;border:2px solid #e0e6ef;border-radius:10px;background:#f8faff;cursor:pointer;font-size:14px;font-weight:600;color:#555;text-align:center;transition:.2s}
  .tab.active{border-color:#2d7ef7;background:#eef4ff;color:#2d7ef7}
  .section{display:none}.section.active{display:block}
  input[type=file]{width:100%;padding:10px;border:2px dashed #b9c4d0;border-radius:10px;background:#fafcff;font-size:13px}
  video{width:100%;border-radius:10px;margin-top:10px;background:#000;max-height:300px}
  .btn{margin-top:12px;width:100%;padding:12px;border:0;border-radius:10px;background:#2d7ef7;color:#fff;font-size:15px;font-weight:600;cursor:pointer;transition:.2s}
  .btn:disabled{background:#9bb8e8;cursor:default}
  .btn.danger{background:#e74c3c}
  .btn.green{background:#27ae60}
  .status{margin-top:10px;text-align:center;font-size:13px;color:#555;min-height:20px}
  .countdown{font-size:40px;font-weight:700;color:#2d7ef7;text-align:center;margin:8px 0}
  .res{margin-top:16px;padding:16px;border-radius:12px;background:#f0f5ff;display:none}
  .res-label{font-size:28px;font-weight:700;color:#1b5e20}
  .res-conf{color:#555;margin:4px 0 12px;font-size:14px}
  .row{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px}
  .bar{flex:1;height:10px;background:#e3e8ef;border-radius:6px;overflow:hidden}
  .bar>i{display:block;height:100%;background:#2d7ef7;border-radius:6px;transition:width .4s}
  .muted{color:#888;font-size:12px;margin-top:10px}
</style></head><body>
<div class="card">
  <h1>SnapSign &mdash; Uji Model</h1>
  <div class="sub">Uji model klasifikasi isyarat BISINDO secara langsung.</div>

  <div class="tabs">
    <div class="tab active" onclick="switchTab('upload')">&#128190; Upload Video</div>
    <div class="tab" onclick="switchTab('webcam')">&#127910; Webcam Langsung</div>
  </div>

  <!-- TAB UPLOAD -->
  <div class="section active" id="sec-upload">
    <input id="file" type="file" accept="video/*">
    <video id="prev-upload" controls style="display:none"></video>
    <button class="btn" id="btn-upload" disabled>Uji Video</button>
  </div>

  <!-- TAB WEBCAM -->
  <div class="section" id="sec-webcam">
    <video id="prev-webcam" autoplay muted playsinline style="display:none"></video>
    <div class="countdown" id="cd" style="display:none">3</div>
    <div class="status" id="wc-status">Klik tombol untuk mulai.</div>
    <button class="btn green" id="btn-start-wc">&#9654; Mulai Rekam (3 detik)</button>
    <button class="btn danger" id="btn-stop-wc" style="display:none">&#9632; Batal</button>
  </div>

  <!-- HASIL (shared) -->
  <div class="res" id="res">
    <div class="res-label" id="lbl">-</div>
    <div class="res-conf" id="cf"></div>
    <div id="top3"></div>
    <div class="muted" id="meta"></div>
  </div>
</div>

<script>
// ── Shared helpers ─────────────────────────────────────────────────────────
const $=id=>document.getElementById(id);
let activeTab='upload';

function switchTab(t){
  activeTab=t;
  document.querySelectorAll('.tab').forEach((el,i)=>el.classList.toggle('active',['upload','webcam'][i]===t));
  document.querySelectorAll('.section').forEach((el,i)=>el.classList.toggle('active',['upload','webcam'][i]===t));
  $('res').style.display='none';
  if(t==='webcam') initWebcam(); else stopWebcam();
}

function showResult(j){
  $('lbl').textContent=j.label||'-';
  $('cf').textContent='Confidence: '+(j.confidence!=null?j.confidence.toFixed(1)+'%':'-');
  $('top3').innerHTML='';
  (j.top3||[]).forEach(t=>{
    const p=Math.round(t[1]*100);
    $('top3').innerHTML+=`<div class="row"><span style="width:96px">${t[0]}</span><span class="bar"><i style="width:${p}%"></i></span><span style="width:32px;text-align:right">${p}%</span></div>`;
  });
  $('meta').textContent='Frame dikirim: '+(j.frames||'-')+'  |  16 frame disampling ke model';
  $('res').style.display='block';
}

// ── Tab UPLOAD ─────────────────────────────────────────────────────────────
$('file').onchange=()=>{
  const f=$('file').files[0];
  if(f){ $('prev-upload').src=URL.createObjectURL(f); $('prev-upload').style.display='block'; $('btn-upload').disabled=false; $('res').style.display='none'; }
};
$('btn-upload').onclick=async()=>{
  $('btn-upload').disabled=true; $('btn-upload').textContent='Memproses...';
  const fd=new FormData(); fd.append('video',$('file').files[0]);
  try{
    const r=await fetch('/predict_video',{method:'POST',body:fd});
    showResult(await r.json());
  }catch(e){$('lbl').textContent='Error';$('cf').textContent=String(e);$('res').style.display='block';}
  $('btn-upload').disabled=false; $('btn-upload').textContent='Uji Video';
};

// ── Tab WEBCAM ─────────────────────────────────────────────────────────────
let stream=null, capturing=false, frames=[], captureTimer=null, cdInterval=null;
const RECORD_SEC=3, FPS_TARGET=20;

async function initWebcam(){
  if(stream) return;
  try{
    stream=await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'},audio:false});
    $('prev-webcam').srcObject=stream; $('prev-webcam').style.display='block';
    $('wc-status').textContent='Kamera siap. Klik Mulai Rekam.';
  }catch(e){$('wc-status').textContent='Kamera tidak bisa diakses: '+e.message;}
}

function stopWebcam(){
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;}
  $('prev-webcam').style.display='none';
  stopCapture();
}

function stopCapture(){
  capturing=false;
  clearInterval(captureTimer); clearInterval(cdInterval);
  $('cd').style.display='none';
  $('btn-start-wc').style.display='block';
  $('btn-stop-wc').style.display='none';
}

$('btn-start-wc').onclick=()=>{
  if(!stream){initWebcam().then(startCapture);return;}
  startCapture();
};

function startCapture(){
  if(!stream){$('wc-status').textContent='Kamera belum siap.';return;}
  frames=[]; capturing=true; $('res').style.display='none';
  $('btn-start-wc').style.display='none';
  $('btn-stop-wc').style.display='block';
  $('cd').style.display='block';

  const canvas=document.createElement('canvas');
  const interval=Math.round(1000/FPS_TARGET);
  let remaining=RECORD_SEC;

  $('cd').textContent=remaining;
  $('wc-status').textContent='Merekam gestur... lakukan isyarat sekarang!';

  cdInterval=setInterval(()=>{
    remaining=Math.max(0,remaining-1);
    $('cd').textContent=remaining;
  },1000);

  captureTimer=setInterval(()=>{
    if(!capturing) return;
    const v=$('prev-webcam');
    canvas.width=v.videoWidth||640; canvas.height=v.videoHeight||480;
    canvas.getContext('2d').drawImage(v,0,0);
    frames.push(canvas.toDataURL('image/jpeg',0.75).split(',')[1]);
  },interval);

  setTimeout(()=>{
    stopCapture();
    sendFrames();
  },RECORD_SEC*1000);
}

$('btn-stop-wc').onclick=stopCapture;

async function sendFrames(){
  if(frames.length<4){$('wc-status').textContent='Frame tidak cukup, coba lagi.';return;}
  $('wc-status').textContent=`Memproses ${frames.length} frame...`;
  try{
    const r=await fetch('/predict_batch',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({frames,client_id:'web_webcam'})
    });
    const j=await r.json();
    showResult(j);
    $('wc-status').textContent='Selesai. Klik Mulai Rekam untuk gestur berikutnya.';
  }catch(e){$('wc-status').textContent='Error: '+String(e);}
  $('btn-start-wc').style.display='block';
}
</script></body></html>"""


@app.route("/", methods=["GET"])
def upload_page():
    from flask import Response
    return Response(_UPLOAD_HTML, mimetype='text/html')


@app.route("/predict_video", methods=["POST"])
def predict_video():
    import tempfile
    if "video" not in request.files:
        return jsonify({"error": "field 'video' wajib"}), 400
    f = request.files["video"]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = tmp.name; tmp.close()
    try:
        f.save(tmp_path)
        label, conf, top3, n = _predict_video_file(tmp_path)
        print(f"[video] {f.filename} ({n} frame) -> {label} ({conf:.1f}%)")
        return jsonify({"label": label, "confidence": round(conf, 2),
                        "top3": top3, "frames": n})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import socket
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_stage2_v10.pth")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--light", choices=['none', 'clahe', 'gray', 'clahe+gray'], default='none')
    ap.add_argument("--mode", choices=['blur', 'black'], default='blur')
    ap.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument("--unmirror", action="store_true",
                    help="flip balik frame dari app (HANYA untuk APK lama yang masih flipHorizontal)")
    ap.add_argument("--gradcam", action="store_true",
                    help="overlay Grad-CAM di montage /preview (mode demo, +1-2 detik/prediksi)")
    args = ap.parse_args()
    _UNMIRROR = args.unmirror
    _GRADCAM = args.gradcam

    _DEVICE = torch.device(args.device)
    if _DEVICE.type == 'cpu':
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    _MODEL, _CLASSES = R.load_model(args.checkpoint, _DEVICE)
    _LIGHT, _MODE = args.light, args.mode

    try:
        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "127.0.0.1"
    print("=" * 55)
    print("  SnapSign Flask API — v10 RGB (MobileNetV2+TSM+CBAM)")
    print("=" * 55)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Device     : {_DEVICE} | light={_LIGHT} | unmirror={'ON' if _UNMIRROR else 'off'} | gradcam={'ON' if _GRADCAM else 'off'}")
    print(f"  Endpoint   : http://{ip}:{args.port}")
    print(f"  Siklus     : prepare {PREPARE_SEC}s -> collect {COLLECT_SEC}s -> hold {HOLD_SEC}s")
    print(f"  >> Set AppConstants.apiBaseUrl = http://{ip}:{args.port}")
    print("  Pastikan HP & PC satu WiFi.  Ctrl+C untuk stop.")
    print("=" * 55)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
