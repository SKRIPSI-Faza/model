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


# Montage terakhir disimpan di memory — tidak ada file/database
_last_montage_jpg: bytes = b''
_last_montage_lock = threading.Lock()

CONF_THRESHOLD = 22.0   # % minimum agar prediksi dianggap valid

def predict_clip(frames224):
    """list frame BGR 224 tersegmentasi -> (label, conf_pct, top3[[label,frac]])."""
    global _last_montage_jpg
    if len(frames224) < 4:
        return "Tidak terdeteksi", 0.0, []
    src, peak = R.trim_active(frames224)
    if peak < 0.5:                          # hampir tidak ada gerakan
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


@app.route("/preview", methods=["GET"])
def preview():
    """Tampilkan 16 frame terakhir yang dikirim ke model. Buka di browser."""
    with _last_montage_lock:
        data = _last_montage_jpg
    if not data:
        return "Belum ada prediksi.", 200, {'Content-Type': 'text/plain'}
    from flask import Response
    return Response(data, mimetype='image/jpeg')


@app.route("/reset", methods=["POST"])
def reset_cycle():
    body = request.get_json(force=True, silent=True) or {}
    client_id = body.get("client_id", request.remote_addr)
    if client_id in _clients:
        with _clients[client_id].lock:
            _clients[client_id].reset()
    return jsonify({"status": "siklus direset", "client_id": client_id})


if __name__ == "__main__":
    import socket
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="best_stage2_v10.pth")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--light", choices=['none', 'clahe', 'gray', 'clahe+gray'], default='none')
    ap.add_argument("--mode", choices=['blur', 'black'], default='blur')
    ap.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

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
    print(f"  Device     : {_DEVICE} | light={_LIGHT}")
    print(f"  Endpoint   : http://{ip}:{args.port}")
    print(f"  Siklus     : prepare {PREPARE_SEC}s -> collect {COLLECT_SEC}s -> hold {HOLD_SEC}s")
    print(f"  >> Set AppConstants.apiBaseUrl = http://{ip}:{args.port}")
    print("  Pastikan HP & PC satu WiFi.  Ctrl+C untuk stop.")
    print("=" * 55)
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
