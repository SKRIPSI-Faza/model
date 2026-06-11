"""
BISINDO v10 — Fine-tune ke Signer Sendiri
==========================================
Melatih ulang SINGKAT dari best_stage2_v10.pth memakai klip hasil rekam_kata.py
agar model mengenali gaya isyaratmu → confidence realtime naik tajam.

Strategi (anti lupa / catastrophic forgetting pada data kecil):
  - Backbone DIBEKUKAN, hanya CBAM + reduce + classifier dilatih (LR kecil).
  - Klip disegmentasi SEKALI lalu di-cache (hemat waktu antar-epoch).
  - Early stopping pada val.

Jalankan (butuh klip di data_custom/ dari rekam_kata.py):
    python finetune_v10.py --data data_custom --epochs 15

Output: best_finetune_v10.pth  → pakai di realtime:
    python realtime_v10.py --checkpoint best_finetune_v10.pth --camera 0
"""

import os, sys, argparse, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import realtime_v10 as R   # reuse model + segmentasi + sampling

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv')


def load_clips(data_dir, classes):
    """Kumpulkan path + label; klip diseg+sampel SEKALI lalu di-cache (uint8)."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    seg = R.RealtimeSegmenter('blur')
    X, y = [], []
    for cls in sorted(os.listdir(data_dir)):
        cdir = os.path.join(data_dir, cls)
        if not os.path.isdir(cdir) or cls not in cls_to_idx:
            continue
        for v in sorted(os.listdir(cdir)):
            if not v.lower().endswith(VIDEO_EXT):
                continue
            cap = cv2.VideoCapture(os.path.join(cdir, v)); frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(seg(f), R.IMG_SIZE))
            cap.release()
            if not frames:
                continue
            src = R.trim_active(frames)[0]
            f16 = R.sample_clip_eval(src, R.NUM_FRAMES, 0.15)   # 16 frame BGR uint8
            X.append(np.stack(f16)); y.append(cls_to_idx[cls])
    seg.close()
    return X, y


def augment_photometric(frame_bgr):
    """Jitter cahaya ringan (variasi kondisi kamera). Konsisten dgn semangat v10."""
    b = random.uniform(0.7, 1.3)
    f = np.clip(frame_bgr.astype(np.float32) * b, 0, 255).astype(np.uint8)
    return f


class ClipDataset(Dataset):
    def __init__(self, X, y, train=False):
        self.X = X; self.y = y; self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        frames = self.X[idx]                       # (16, H, W, 3) uint8 BGR
        out = []
        for f in frames:
            if self.train:
                f = augment_photometric(f)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - np.array(R.IMAGENET_MEAN)) / np.array(R.IMAGENET_STD)
            out.append(torch.from_numpy(rgb).permute(2, 0, 1).float())
        return torch.stack(out), self.y[idx]       # (16,3,H,W), label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_custom')
    ap.add_argument('--checkpoint', default='best_stage2_v10.pth')
    ap.add_argument('--out', default='best_finetune_v10.pth')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        print(f'[ERROR] folder data tidak ada: {args.data} (jalankan rekam_kata.py dulu)')
        sys.exit(1)

    device = torch.device(args.device)
    model, classes = R.load_model(args.checkpoint, device)
    print(f'Device: {device} | memuat: {args.checkpoint}')

    print('Memproses & menyegmentasi klip (sekali)...')
    X, y = load_clips(args.data, classes)
    if len(X) < 4:
        print(f'[ERROR] klip terlalu sedikit ({len(X)}). Rekam lebih banyak.')
        sys.exit(1)
    print(f'Total {len(X)} klip dari {len(set(y))} kelas.')

    # split train/val (stratified sederhana)
    idx = list(range(len(X))); random.shuffle(idx)
    n_val = max(1, int(len(idx) * args.val_frac))
    val_idx, tr_idx = set(idx[:n_val]), idx[n_val:]
    tr = ClipDataset([X[i] for i in tr_idx], [y[i] for i in tr_idx], train=True)
    va = ClipDataset([X[i] for i in val_idx], [y[i] for i in val_idx], train=False)
    tl = DataLoader(tr, args.batch, shuffle=True)
    vl = DataLoader(va, args.batch, shuffle=False)

    # Bekukan backbone; latih hanya cbam + reduce + classifier
    for p in model.features.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'Parameter dilatih: {sum(p.numel() for p in trainable):,}')

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-3)
    best_val, best_state = 0.0, None

    for ep in range(1, args.epochs + 1):
        model.train(); tr_correct = tr_n = 0; tr_loss = 0.0
        for xb, yb in tl:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            opt.zero_grad()
            out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (out.argmax(1) == yb).sum().item(); tr_n += xb.size(0)

        model.eval(); va_correct = va_n = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), torch.tensor(yb).to(device)
                out = model(xb)
                va_correct += (out.argmax(1) == yb).sum().item(); va_n += xb.size(0)
        va_acc = va_correct / max(va_n, 1)
        print(f'Ep {ep:02d} | train_loss={tr_loss/max(tr_n,1):.3f} '
              f'train_acc={tr_correct/max(tr_n,1):.3f} | val_acc={va_acc:.3f}')
        if va_acc >= best_val:
            best_val = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        torch.save(best_state, args.out)
        print(f'\n[SELESAI] val_acc terbaik {best_val:.3f} → disimpan: {args.out}')
        print(f'Pakai: python realtime_v10.py --checkpoint {args.out} --camera 0')


if __name__ == '__main__':
    main()
