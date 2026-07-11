"""
Uji model SnapSign langsung dari webcam laptop.
Tekan SPASI untuk mulai rekam 3 detik, ESC untuk keluar.
Jalankan: python uji_webcam.py --checkpoint best_finetune_hp.pth
"""
import argparse, os, sys, time
import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realtime_v10 as R
from server_v10 import segment_frame, BISINDOClassifier, inject_tsm

RECORD_SEC  = 3
PREP_SEC    = 2
HOLD_SEC    = 3
FONT        = cv2.FONT_HERSHEY_SIMPLEX


def load_model(ckpt_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = BISINDOClassifier(num_classes=R.NUM_CLASSES, num_frames=R.NUM_FRAMES).to(device)
    ckpt   = torch.load(ckpt_path, map_location=device)
    state  = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    classes = ckpt.get('classes', R.CLASSES)
    print(f"[model] {os.path.basename(ckpt_path)} | {len(classes)} kelas | {device}")
    return model, classes, device


def predict(frames_bgr, model, classes, device):
    seg16 = []
    indices = np.linspace(0, len(frames_bgr) - 1, R.NUM_FRAMES, dtype=int)
    for i in indices:
        s, _ = segment_frame(frames_bgr[i], 'none')
        seg16.append(cv2.resize(s, R.IMG_SIZE))
    x = R.frames_to_tensor(seg16, device)
    with torch.inference_mode():
        probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
    order = probs.argsort()[::-1]
    top3  = [(classes[i], float(probs[i]) * 100) for i in order[:3]]
    return top3


def overlay_text(frame, text, y, scale=0.9, color=(255,255,255), thick=2, bg=(0,0,0)):
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.rectangle(frame, (10, y - h - 6), (10 + w + 8, y + 6), bg, -1)
    cv2.putText(frame, text, (14, y), FONT, scale, color, thick, cv2.LINE_AA)


def draw_topbar(frame, phase, remaining, top3, conf_threshold=15.0):
    h, w = frame.shape[:2]
    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (30, 30, 30), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    if phase == 'prepare':
        msg   = f"Bersiap... {remaining:.1f}s"
        color = (0, 200, 255)
    elif phase == 'collect':
        msg   = f"REKAM: {remaining:.1f}s"
        color = (0, 255, 100)
    elif phase == 'hold':
        msg   = f"Hasil ditampilkan... {remaining:.1f}s"
        color = (200, 200, 200)
    else:
        msg   = "Tekan SPASI untuk mulai | ESC keluar"
        color = (180, 180, 180)

    cv2.putText(frame, msg, (14, 34), FONT, 0.85, color, 2, cv2.LINE_AA)

    if phase == 'hold' and top3:
        label, conf = top3[0]
        bar_y = bar_h + 10
        col   = (0, 220, 80) if conf >= conf_threshold else (0, 140, 255)
        overlay_text(frame, f"{label}  {conf:.1f}%", bar_y + 28, scale=1.1, color=col, bg=(20,20,20))
        for k, (lb, cn) in enumerate(top3[1:3]):
            overlay_text(frame, f"  {k+2}. {lb}  {cn:.1f}%", bar_y + 60 + k * 28,
                         scale=0.65, color=(200,200,200), bg=(20,20,20))


def run(ckpt_path):
    model, classes, device = load_model(ckpt_path)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa dibuka.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    phase        = 'idle'
    phase_start  = 0.0
    frames_buf   = []
    top3_result  = []

    print("Tekan SPASI untuk mulai rekam, ESC untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        now   = time.time()

        if phase == 'idle':
            draw_topbar(frame, 'idle', 0, [])

        elif phase == 'prepare':
            elapsed   = now - phase_start
            remaining = max(0.0, PREP_SEC - elapsed)
            draw_topbar(frame, 'prepare', remaining, [])
            # Countdown besar di tengah
            num = str(int(remaining) + 1)
            (tw, th), _ = cv2.getTextSize(num, FONT, 4.0, 6)
            cx = (frame.shape[1] - tw) // 2
            cy = (frame.shape[0] + th) // 2
            cv2.putText(frame, num, (cx, cy), FONT, 4.0, (0,200,255), 6, cv2.LINE_AA)
            if elapsed >= PREP_SEC:
                phase       = 'collect'
                phase_start = now
                frames_buf  = []

        elif phase == 'collect':
            elapsed   = now - phase_start
            remaining = max(0.0, RECORD_SEC - elapsed)
            frames_buf.append(frame.copy())
            draw_topbar(frame, 'collect', remaining, [])
            # Lingkaran rekam
            cv2.circle(frame, (frame.shape[1] - 24, 24), 9, (0, 0, 220), -1)
            if elapsed >= RECORD_SEC:
                phase      = 'processing'
                phase_start = now

        elif phase == 'processing':
            draw_topbar(frame, 'idle', 0, [])
            overlay_text(frame, "Memproses...", frame.shape[0] // 2,
                         scale=1.0, color=(255,255,0), bg=(30,30,30))
            cv2.imshow('SnapSign - Uji Webcam', frame)
            cv2.waitKey(1)
            top3_result = predict(frames_buf, model, classes, device)
            print(f"\n[hasil] {top3_result[0][0]}  {top3_result[0][1]:.1f}%")
            for lb, cn in top3_result[1:]:
                print(f"        {lb}  {cn:.1f}%")
            phase       = 'hold'
            phase_start = now
            continue

        elif phase == 'hold':
            elapsed   = now - phase_start
            remaining = max(0.0, HOLD_SEC - elapsed)
            draw_topbar(frame, 'hold', remaining, top3_result)
            if elapsed >= HOLD_SEC:
                phase = 'idle'

        cv2.imshow('SnapSign - Uji Webcam', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:   # ESC
            break
        if key == 32 and phase == 'idle':   # SPASI
            phase       = 'prepare'
            phase_start = now
            top3_result = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default='best_finetune_hp.pth')
    args = ap.parse_args()
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.checkpoint)
    run(ckpt)
