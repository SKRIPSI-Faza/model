"""
BISINDO 24-Kelas v10 — Real-time Webcam (siklus rekam, mulus di CPU)
====================================================================
Versi BENAR (menggantikan versi sliding-window yang patah-patah). Mengikuti
arsitektur realtime_test.py v9 yang terbukti:
  - SEGMENTASI background identik training (selfie segmentation + smoothing) →
    input ke model sama dengan data latih (WLBISINDO_seg).
  - INFERENSI di THREAD terpisah → loop kamera tidak freeze (mulus walau CPU).
  - SIKLUS: BERSIAP(countdown) → REKAM 3 dtk → PROSES(auto-trim + 16 frame
    margin 0.15 + voting) → HASIL. Cocok karena model dilatih atas klip rapi.
  - "BELUM DIKENALI" eksplisit: kalau confidence < threshold ATAU tidak ada
    cukup gerakan saat merekam.

Berdiri sendiri (tidak meng-import train_v10 / preprocess_segment).

Jalankan:
    python realtime_v10.py --checkpoint best_stage2_v10.pth --camera 0
    # kalau cahaya kamera beda jauh dari data latih:
    python realtime_v10.py --light clahe+gray --threshold 0.6

Kontrol: q/ESC = keluar | s = toggle tampilan (kamera <-> yang dilihat model)
"""

import os, sys, time, argparse, threading, queue
from collections import deque

try:
    sys.stdout.reconfigure(encoding='utf-8')   # konsol Windows
except Exception:
    pass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.mobilenetv2 import InvertedResidual
import torchvision.transforms.functional as TF
from PIL import Image
import mediapipe as mp

# ─── Konfigurasi (sama dengan training v10)
EXPLICIT_CLASSES = [
    'air', 'bagaimana', 'belajar', 'berangkat', 'cari',
    'datang', 'dengar', 'dimana', 'hijau', 'merah',
    'kapan', 'keluarga', 'kuning', 'lagi', 'maaf',
    'makan', 'mengapa', 'motor', 'rumah', 'saya',
    'siapa', 'teman', 'terimakasih', 'tuli'
]
NUM_CLASSES   = 24
NUM_FRAMES    = 16
IMG_SIZE      = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Segmentasi (identik preprocess_segment.py v9)
MASK_ALPHA  = 0.6
MASK_DILATE = 9
BLUR_KERNEL = 41


# ─── Model: MobileNetV2 + TSM + CBAM 
class TemporalShift(nn.Module):
    def __init__(self, n_segment=16, n_div=8):
        super().__init__()
        self.n_segment = n_segment; self.fold_div = n_div

    def forward(self, x):
        BT, C, H, W = x.size()
        B = BT // self.n_segment
        x = x.view(B, self.n_segment, C, H, W)
        fold = C // self.fold_div
        out = torch.zeros_like(x)
        out[:, 1:,    :fold]     = x[:, :-1, :fold]
        out[:, :-1, fold:2*fold] = x[:, 1:,  fold:2*fold]
        out[:, :,   2*fold:]     = x[:, :,   2*fold:]
        return out.view(BT, C, H, W)


def inject_tsm(model, n_segment=16, n_div=8):
    for block in model.features:
        if isinstance(block, InvertedResidual):
            orig = block.conv[0]
            block.conv[0] = nn.Sequential(TemporalShift(n_segment, n_div), orig)
    return model


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1); self.mx = nn.AdaptiveMaxPool2d(1)
        self.fc  = nn.Sequential(nn.Linear(ch, ch//r, bias=False), nn.ReLU(),
                                  nn.Linear(ch//r, ch, bias=False))
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        a = self.fc(self.avg(x).view(B, C))
        m = self.fc(self.mx(x).view(B, C))
        return x * self.sig(a + m).view(B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx  = x.max(dim=1, keepdim=True).values
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.channel_att = ChannelAttention(ch, r)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class BISINDOClassifier(nn.Module):
    def __init__(self, num_classes, num_frames=16, use_cbam=True,
                 grid=4, reduce_dim=256, dropout=0.5):
        super().__init__()
        self.num_frames = num_frames
        backbone = models.mobilenet_v2(weights=None)     # bobot diisi state_dict
        backbone = inject_tsm(backbone, n_segment=num_frames, n_div=8)
        self.features = backbone.features
        feat_dim      = backbone.last_channel
        self.cbam   = CBAM(feat_dim) if use_cbam else nn.Identity()
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_dim, reduce_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_dim), nn.ReLU(inplace=True))
        self.pool        = nn.AdaptiveAvgPool2d((grid, grid))
        self.spatial_dim = reduce_dim * grid * grid
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(self.spatial_dim, num_classes))

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.features(x); x = self.cbam(x); x = self.reduce(x)
        x = self.pool(x).view(B, T, -1).mean(dim=1)
        return self.classifier(x)


# ─── Segmentasi real-time (identik preprocess_segment.process_video) 
def composite(frame, mask, mode):
    m3 = np.repeat(mask[:, :, None], 3, axis=2)
    bg = np.zeros_like(frame) if mode == 'black' \
        else cv2.GaussianBlur(frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
    out = frame.astype(np.float32) * m3 + bg.astype(np.float32) * (1.0 - m3)
    return out.astype(np.uint8)


class RealtimeSegmenter:
    def __init__(self, mode='blur'):
        self.mode = mode
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
        self.k = np.ones((MASK_DILATE, MASK_DILATE), np.uint8)
        self.prev = None

    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mask = self.seg.process(rgb).segmentation_mask
        binm = cv2.dilate((mask > 0.3).astype(np.uint8), self.k, iterations=1)
        mask = np.clip(np.maximum(mask, binm.astype(np.float32) * 0.8), 0.0, 1.0)
        if self.prev is not None and self.prev.shape == mask.shape:
            mask = MASK_ALPHA * mask + (1.0 - MASK_ALPHA) * self.prev
        self.prev = mask
        return composite(frame_bgr, mask, self.mode)

    def close(self):
        self.seg.close()


# Normalisasi cahaya (opsional, atasi domain gap kamera)
def _gray_world(img):
    f = img.astype(np.float32)
    avg = f.reshape(-1, 3).mean(axis=0)
    return np.clip(f * (avg.mean() / (avg + 1e-6)), 0, 255).astype(np.uint8)


def normalize_lighting(frame, method):
    if method == 'none':
        return frame
    if method in ('clahe', 'clahe+gray'):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        frame = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    if method in ('gray', 'clahe+gray'):
        frame = _gray_world(frame)
    return frame


# Preprocessing klip (identik jalur eval training)
def frames_to_tensor(frames_bgr, device):
    out = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t = TF.normalize(TF.to_tensor(Image.fromarray(rgb)), IMAGENET_MEAN, IMAGENET_STD)
        out.append(t)
    return torch.stack(out).unsqueeze(0).to(device)


def sample_clip_eval(frames, num_frames, margin=0.15):
    total = max(len(frames), 1)
    start = int(total * margin); end = int(total * (1.0 - margin))
    seg = max(end - start, 1) / float(num_frames)
    out = []
    for i in range(num_frames):
        s = int(start + i * seg); e = int(start + (i + 1) * seg)
        out.append(frames[min(s + (e - s) // 2, total - 1)])
    return out


def motion_energy(frames):
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    m = [0.0] + [float(cv2.absdiff(grays[i], grays[i-1]).mean()) for i in range(1, len(grays))]
    return np.asarray(m, dtype=np.float32)


def trim_active(frames, thr_ratio=0.30, pad=2, min_len=10):
    if len(frames) < min_len:
        return frames, 0.0
    m = motion_energy(frames)
    peak = float(m.max())
    if peak < 1e-3:
        return frames, peak
    base = float(np.percentile(m, 20))
    active = np.where(m >= base + thr_ratio * (peak - base))[0]
    if active.size < 2:
        return frames, peak
    s = max(int(active[0]) - pad, 0); e = min(int(active[-1]) + pad, len(frames) - 1)
    if e - s + 1 < min_len:
        return frames, peak
    return frames[s:e + 1], peak


def make_montage(frames16, header='', cols=4, tile=200):
    """Susun 16 frame (yang dilihat model) jadi grid 4x4 + nomor urut.
    Inilah validasi visual: apakah 16 frame benar memuat isyarat."""
    tiles = []
    for i, f in enumerate(frames16):
        t = cv2.resize(f, (tile, tile)).copy()
        cv2.putText(t, str(i + 1), (6, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)
        tiles.append(t)
    while len(tiles) % cols != 0:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r * cols:(r + 1) * cols])
            for r in range(len(tiles) // cols)]
    grid = np.vstack(rows)
    if header:
        bar = np.zeros((36, grid.shape[1], 3), np.uint8)
        cv2.putText(bar, header, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 230, 0), 2)
        grid = np.vstack([bar, grid])
    return grid


def save_montage(frames16, label, conf, motion, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time()*1000)%1000:03d}'
    img = make_montage(frames16, f'{label} {conf*100:.0f}%  motion={motion:.2f}')
    path = os.path.join(save_dir, f'{ts}_{label}_{conf*100:.0f}.png')
    cv2.imwrite(path, img)
    print(f'[SAVE] {path}')


def load_model(path, device):
    model = BISINDOClassifier(NUM_CLASSES, NUM_FRAMES).to(device)
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state, classes = ckpt['model_state_dict'], ckpt.get('classes', EXPLICIT_CLASSES)
    else:
        state, classes = ckpt, EXPLICIT_CLASSES
    model.load_state_dict(state); model.eval()
    return model, classes


# Main: siklus rekam + thread inferensi 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='best_stage2_v10.pth')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--mode', choices=['blur', 'black'], default='blur')
    ap.add_argument('--light', choices=['none', 'clahe', 'gray', 'clahe+gray'], default='none')
    ap.add_argument('--threshold', type=float, default=0.55,
                    help='di bawah ini → "belum dikenali"')
    ap.add_argument('--min-motion', type=float, default=0.8,
                    help='energi gerak minimum saat rekam; di bawah ini → "belum dikenali". '
                         'Set 0 untuk MATIKAN gate gerak (samakan dgn perilaku v9)')
    ap.add_argument('--prep-sec', type=float, default=2.0)
    ap.add_argument('--record-sec', type=float, default=3.0)
    ap.add_argument('--show-sec', type=float, default=1.5)
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--vote', type=int, default=3)
    ap.add_argument('--save-dir', default=None,
                    help='simpan montage 16-frame yang dilihat model tiap siklus (validasi)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--flip-input', action='store_true',
                    help='balik horizontal frame ke MODEL (uji apakah arah tanganmu '
                         'kebalik dari data latih)')
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f'[ERROR] checkpoint tidak ada: {args.checkpoint}'); sys.exit(1)

    device = torch.device(args.device)
    if device.type == 'cpu':
        torch.set_num_threads(max(1, os.cpu_count() or 1))
    print(f'Device: {device} | Checkpoint: {args.checkpoint}')
    model, classes = load_model(args.checkpoint, device)
    print(f'Kelas: {len(classes)} | Siklus: BERSIAP {args.prep_sec:.0f}s → '
          f'REKAM {args.record_sec:.0f}s → PROSES → HASIL')

    segmenter = RealtimeSegmenter(mode=args.mode)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[ERROR] kamera {args.camera} tidak bisa dibuka'); segmenter.close(); sys.exit(1)

    lock = threading.Lock(); stop_event = threading.Event()
    clip_q = queue.Queue(maxsize=1)
    shared = {'label': '...', 'conf': 0.0, 'motion': 0.0, 'reason': '',
              'top3': [], 'frames16': None, 'processing': False}
    prob_hist = deque(maxlen=max(args.vote, 1))

    def worker():
        while not stop_event.is_set():
            try:
                clip = clip_q.get(timeout=0.2)
            except queue.Empty:
                continue
            src, peak = trim_active(clip)
            # Sampel 16 frame DULU → selalu bisa divalidasi (walau hasilnya "diam")
            f16 = sample_clip_eval(src, NUM_FRAMES, args.margin)
            with lock:
                shared['frames16'] = list(f16)
            # Gate gerak (matikan dgn --min-motion 0)
            if args.min_motion > 0 and peak < args.min_motion:
                print(f'[CYCLE] motion={peak:5.2f}  -> DIAM (belum dikenali)')
                with lock:
                    shared.update(label='belum dikenali', conf=0.0, motion=peak,
                                  reason=f'diam (motion {peak:.2f}<{args.min_motion})',
                                  processing=False)
                if args.save_dir:
                    save_montage(f16, 'belum_dikenali', 0.0, peak, args.save_dir)
                prob_hist.clear(); continue
            x = frames_to_tensor(f16, device)
            with torch.inference_mode():
                probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
            # Reset voting bila isyarat BERGANTI → keputusan tidak nyangkut di
            # isyarat lama; voting hanya menstabilkan saat isyarat yg sama diulang.
            cur_top = int(probs.argmax())
            if prob_hist and int(np.mean(prob_hist, axis=0).argmax()) != cur_top:
                prob_hist.clear()
            prob_hist.append(probs)
            voted = np.mean(prob_hist, axis=0)
            i = int(voted.argmax()); conf = float(voted[i])
            top3 = [(classes[j], float(voted[j])) for j in voted.argsort()[::-1][:3]]
            print(f'[CYCLE] motion={peak:5.2f}  conf={conf*100:4.1f}%  | ' +
                  '  '.join(f'{l}:{c*100:.0f}%' for l, c in top3))
            with lock:
                shared['top3'] = top3
                if conf >= args.threshold:
                    shared.update(label=classes[i], conf=conf, motion=peak,
                                  reason='', processing=False)
                else:
                    shared.update(label='belum dikenali', conf=conf, motion=peak,
                                  reason=f'ragu (conf {conf*100:.0f}%<{args.threshold*100:.0f}%)',
                                  processing=False)
            if args.save_dir:
                lbl = classes[i] if conf >= args.threshold else 'ragu'
                save_montage(f16, lbl, conf, peak, args.save_dir)

    th = threading.Thread(target=worker, daemon=True); th.start()

    state = 'PREP'; record = []; prep_start = time.time()
    rec_start = show_until = 0.0; last_seg = None; showing_seg = False
    montage_img = None
    fps, t_prev = 0.0, time.time()

    print('\n[Jalan] q=keluar  s=toggle tampilan\n')
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.flip_input:                  # uji arah tangan (mirror) ke model
                frame = cv2.flip(frame, 1)
            frame = normalize_lighting(frame, args.light)
            seg = segmenter(frame); last_seg = seg
            now = time.time()

            if state == 'PREP':
                if now - prep_start >= args.prep_sec:
                    state, rec_start, record = 'REC', now, []
            elif state == 'REC':
                record.append(cv2.resize(seg, IMG_SIZE))
                if now - rec_start >= args.record_sec:
                    with lock: shared['processing'] = True
                    try:
                        clip_q.put_nowait(list(record)); state = 'PROSES'
                    except queue.Full:
                        with lock: shared['processing'] = False
                        rec_start = now
                    record = []
            elif state == 'PROSES':
                with lock:
                    done = not shared['processing']
                    f16 = shared['frames16']; lbl = shared['label']; cf = shared['conf']
                if done:
                    state, show_until = 'SHOW', now + args.show_sec
                    if f16:                       # bangun montage 16-frame sekali
                        mg = make_montage(f16, f'{lbl}  {cf*100:.0f}%')
                        sc = 640.0 / mg.shape[1]
                        montage_img = cv2.resize(mg, (640, int(mg.shape[0]*sc)))
            elif state == 'SHOW':
                if now >= show_until: state, prep_start = 'PREP', now

            inst = 1.0 / max(now - t_prev, 1e-6)
            fps = 0.9 * fps + 0.1 * inst if fps else inst; t_prev = now

            with lock:
                label, conf = shared['label'], shared['conf']
                reason, motion = shared['reason'], shared['motion']
                top3 = list(shared['top3'])

            base = last_seg if showing_seg else frame
            canvas = cv2.flip(base, 1)            # cermin TAMPILAN saja
            h, w = canvas.shape[:2]
            cv2.rectangle(canvas, (0, 0), (w, 95), (0, 0, 0), -1)

            if state == 'PREP':
                rem = max(args.prep_sec - (now - prep_start), 0.0)
                cv2.putText(canvas, 'BERSIAP... isyarat saat REKAM', (14, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 235), 2)
                if args.prep_sec > 0:
                    big = str(int(rem) + 1)
                    (tw, t_h), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 5, 8)
                    cv2.putText(canvas, big, ((w-tw)//2, (h+t_h)//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 180, 235), 8)
            elif state == 'REC':
                cv2.putText(canvas, f'REKAM {max(args.record_sec-(now-rec_start),0):0.1f}s',
                            (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 235), 2)
                cv2.rectangle(canvas, (0, 0), (int(w*(now-rec_start)/args.record_sec), 6),
                              (0, 0, 235), -1)
            elif state == 'PROSES':
                cv2.putText(canvas, 'MEMPROSES...', (14, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 235), 2)
            else:  # SHOW
                unknown = (label == 'belum dikenali')
                col = (0, 0, 235) if unknown else (0, 230, 0)
                txt = 'belum dikenali' if unknown else f'{label}  {conf*100:.0f}%'
                cv2.putText(canvas, txt, (14, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 3)
                # alasan kalau "belum dikenali" (diam / ragu) + nilai motion utk kalibrasi
                info = reason if reason else f'motion={motion:.2f}'
                cv2.putText(canvas, info, (14, 88),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                # Top-3 kandidat (berguna saat ragu: jawaban benar sering di sini)
                for k, (l, c) in enumerate(top3):
                    cv2.putText(canvas, f'{k+1}. {l} {c*100:.0f}%', (14, 120 + k*22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 230, 0) if k == 0 else (180, 180, 180), 1)

            view = 'SEG (dilihat model)' if showing_seg else 'KAMERA'
            cv2.putText(canvas, f'{fps:4.1f} FPS | {view} | [s] toggle [q] keluar',
                        (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1)

            cv2.imshow('BISINDO v10 — Real-time', canvas)
            # Jendela validasi: 16 frame yang DILIHAT model (tampil saat HASIL)
            if state == 'SHOW' and montage_img is not None:
                cv2.imshow('16 frame yang dilihat model', montage_img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            if key == ord('s'): showing_seg = not showing_seg
    finally:
        stop_event.set(); th.join(timeout=2.0)
        cap.release(); cv2.destroyAllWindows(); segmenter.close()
        print('[Selesai]')


if __name__ == '__main__':
    main()
