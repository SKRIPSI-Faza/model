"""
BISINDO v10 — Evaluasi Top-1 & Top-3 Accuracy
==============================================
Mengukur akurasi pada folder video berlabel. Berguna untuk laporan:
melaporkan Top-1 DAN Top-3 menunjukkan "model hampir benar" walau Top-1
turun pada signer baru (subject-dependent).

Struktur folder --data:
    data/
      air/      *.mp4
      makan/    *.mp4
      ...        (nama folder = nama kelas, lihat EXPLICIT_CLASSES)

Jalankan:
    # data sudah tersegmentasi (mis. test set WLBISINDO_seg)
    python eval_topk.py --data path/test_seg

    # data MENTAH (perlu segmentasi seperti training)
    python eval_topk.py --data path/test_raw --segment
"""

import os, argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2

import realtime_v10 as R   # reuse model + pipeline (tidak menjalankan main)


def predict_video(path, model, device, segment, margin, trim):
    seg = R.RealtimeSegmenter('blur') if segment else None
    cap = cv2.VideoCapture(path); frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if seg is not None:
            f = seg(f)
        frames.append(cv2.resize(f, R.IMG_SIZE))
    cap.release()
    if seg:
        seg.close()
    if not frames:
        return None
    src = R.trim_active(frames)[0] if trim else frames
    f16 = R.sample_clip_eval(src, R.NUM_FRAMES, margin)
    x = R.frames_to_tensor(f16, device)
    with torch.inference_mode():
        return F.softmax(model(x), dim=1)[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='folder berisi subfolder per-kelas')
    ap.add_argument('--checkpoint', default='best_stage2_v10.pth')
    ap.add_argument('--segment', action='store_true', help='segmentasi dulu (data mentah)')
    ap.add_argument('--margin', type=float, default=0.15)
    ap.add_argument('--trim', action='store_true', help='auto-trim idle (default: tidak)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    device = torch.device(args.device)
    model, classes = R.load_model(args.checkpoint, device)
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    print(f'Device: {device} | kelas: {len(classes)}')

    VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv')
    n, top1, top3 = 0, 0, 0
    per_cls = {}   # kelas -> [benar_top1, total]
    salah = []     # daftar prediksi salah utk ditelaah

    for cls in sorted(os.listdir(args.data)):
        cdir = os.path.join(args.data, cls)
        if not os.path.isdir(cdir) or cls not in cls_to_idx:
            continue
        y = cls_to_idx[cls]
        per_cls.setdefault(cls, [0, 0])
        for v in sorted(os.listdir(cdir)):
            if not v.lower().endswith(VIDEO_EXT):
                continue
            probs = predict_video(os.path.join(cdir, v), model, device,
                                   args.segment, args.margin, args.trim)
            if probs is None:
                continue
            order = probs.argsort()[::-1]
            n += 1; per_cls[cls][1] += 1
            if order[0] == y:
                top1 += 1; top3 += 1; per_cls[cls][0] += 1
            elif y in order[:3]:
                top3 += 1
                salah.append((cls, classes[order[0]], probs[order[0]], '(benar di top-3)'))
            else:
                salah.append((cls, classes[order[0]], probs[order[0]], ''))

    if n == 0:
        print('[ERROR] tidak ada video cocok. Pastikan nama folder = nama kelas.')
        return
    print(f'\n=== HASIL ({n} video) ===')
    print(f'  Top-1 Accuracy : {top1/n*100:.2f}%')
    print(f'  Top-3 Accuracy : {top3/n*100:.2f}%')
    print('\n--- Per kelas (Top-1) ---')
    for c, (ok, tot) in sorted(per_cls.items()):
        if tot:
            print(f'  {c:14s} {ok}/{tot}  ({ok/tot*100:.0f}%)')
    if salah:
        print('\n--- Prediksi salah (Top-1) ---')
        for true_c, pred_c, p, note in salah[:30]:
            print(f'  {true_c:14s} -> {pred_c:14s} {p*100:.0f}%  {note}')


if __name__ == '__main__':
    main()
