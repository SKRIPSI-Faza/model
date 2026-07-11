"""
finetune_hp.py — Fine-tune adaptasi signer dengan preprocessing IDENTIK SERVER
==============================================================================
Untuk demo di APLIKASI HP (jalur server_v10.py). Bedanya dengan finetune_v10.py:

  finetune_v10.py : segmentasi TANPA center_crop  -> cocok untuk webcam (realtime_v10)
  finetune_hp.py  : center_crop_square DULU + segmentasi -> IDENTIK server_v10.segment_frame
                    (server HP melakukan center_crop pada frame dari kamera HP)

Jadi model hasil script ini geometrinya COCOK dengan apa yang server kirim ke
model saat app HP dipakai.

Klip latih sebaiknya DIREKAM LEWAT HP (kamera & framing sama dgn inference).
Susun: data_custom/<kata>/<apa saja>.mp4   (nama folder = nama kelas)

Jalankan:
    python finetune_hp.py --data data_custom --epochs 20
Output: best_finetune_hp.pth
    python server_v10.py --checkpoint best_finetune_hp.pth --port 8000
"""

import os, sys, argparse, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import mediapipe as mp

import realtime_v10 as R   # reuse model + composite + trim + sampling + konstanta

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv')

# ── Segmentasi IDENTIK server_v10.segment_frame (center_crop + selfie) ───────
_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
_dilate = np.ones((R.MASK_DILATE, R.MASK_DILATE), np.uint8)


def center_crop_square(bgr):
    h, w = bgr.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return bgr[y0:y0 + s, x0:x0 + s]


def segment_frame(bgr, mode='blur'):
    bgr = center_crop_square(bgr)                  # <-- kunci: sama dgn server HP
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = _selfie.process(rgb).segmentation_mask
    binm = cv2.dilate((mask > 0.3).astype(np.uint8), _dilate, iterations=1)
    mask = np.clip(np.maximum(mask, binm.astype(np.float32) * 0.8), 0.0, 1.0)
    return R.composite(bgr, mask, mode)


def load_clips(data_dir, classes, allowed=None):
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    cls_list = [c for c in sorted(os.listdir(data_dir))
                if os.path.isdir(os.path.join(data_dir, c))
                and c in cls_to_idx
                and (not allowed or c in allowed)]
    total_vids = sum(
        sum(1 for v in os.listdir(os.path.join(data_dir, c)) if v.lower().endswith(VIDEO_EXT))
        for c in cls_list)
    done = 0
    for cls in cls_list:
        cdir = os.path.join(data_dir, cls)
        vids = [v for v in sorted(os.listdir(cdir)) if v.lower().endswith(VIDEO_EXT)]
        print(f'  [{cls}] {len(vids)} video...', flush=True)
        for v in vids:
            cap = cv2.VideoCapture(os.path.join(cdir, v)); frames = []
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                frames.append(cv2.resize(segment_frame(f), R.IMG_SIZE))
            cap.release()
            done += 1
            print(f'    {done}/{total_vids} {v}', end='\r', flush=True)
            if not frames:
                continue
            src = R.trim_active(frames)[0]
            f16 = R.sample_clip_eval(src, R.NUM_FRAMES, 0.15)
            X.append(np.stack(f16)); y.append(cls_to_idx[cls])
        print()
    return X, y


def augment_photometric(frame_bgr):
    b = random.uniform(0.7, 1.3)
    return np.clip(frame_bgr.astype(np.float32) * b, 0, 255).astype(np.uint8)


class ClipDataset(Dataset):
    def __init__(self, X, y, train=False):
        self.X = X; self.y = y; self.train = train

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        out = []
        for f in self.X[idx]:
            if self.train:
                f = augment_photometric(f)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - np.array(R.IMAGENET_MEAN)) / np.array(R.IMAGENET_STD)
            out.append(torch.from_numpy(rgb).permute(2, 0, 1).float())
        return torch.stack(out), self.y[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data_custom')
    ap.add_argument('--checkpoint', default='best_stage2_v10.pth')
    ap.add_argument('--out', default='best_finetune_hp.pth')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--test-frac', type=float, default=0.15)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--words', default=None, help='subset kata dipisah koma (default semua)')
    args = ap.parse_args()

    if not os.path.isdir(args.data):
        print(f'[ERROR] folder data tidak ada: {args.data}'); sys.exit(1)

    device = torch.device(args.device)
    model, classes = R.load_model(args.checkpoint, device)
    print(f'Device: {device} | memuat: {args.checkpoint}')
    print('Preprocessing: center_crop + selfie segmentation (IDENTIK server HP)')

    allowed = set(w.strip() for w in args.words.split(',')) if args.words else None
    if allowed:
        print(f'Subset kata: {sorted(allowed)}')
    print('Memproses & menyegmentasi klip (sekali)...')
    X, y = load_clips(args.data, classes, allowed)
    _selfie.close()
    if len(X) < 4:
        print(f'[ERROR] klip terlalu sedikit ({len(X)}). Rekam lebih banyak.'); sys.exit(1)
    print(f'Total {len(X)} klip dari {len(set(y))} kelas.')

    # Stratified split — tiap kelas dapat porsi test/val yang sama
    from collections import defaultdict
    cls_buckets = defaultdict(list)
    for i, label in enumerate(y):
        cls_buckets[label].append(i)

    test_idx, val_idx, tr_idx = [], [], []
    for label_idx in sorted(cls_buckets):
        bucket = cls_buckets[label_idx][:]
        random.shuffle(bucket)
        n_test = max(1, int(len(bucket) * args.test_frac))
        n_val  = max(1, int(len(bucket) * args.val_frac))
        test_idx.extend(bucket[:n_test])
        val_idx.extend(bucket[n_test:n_test + n_val])
        tr_idx.extend(bucket[n_test + n_val:])
    print(f'Split (stratified): train={len(tr_idx)} | val={len(val_idx)} | test={len(test_idx)}')
    tr = ClipDataset([X[i] for i in tr_idx], [y[i] for i in tr_idx], train=True)
    va = ClipDataset([X[i] for i in val_idx], [y[i] for i in val_idx], train=False)
    te = ClipDataset([X[i] for i in test_idx], [y[i] for i in test_idx], train=False)
    tl = DataLoader(tr, args.batch, shuffle=True)
    vl = DataLoader(va, args.batch, shuffle=False)
    tel = DataLoader(te, args.batch, shuffle=False)

    # Bekukan seluruh backbone dulu
    for p in model.features.parameters():
        p.requires_grad = False
    # Buka 4 blok terakhir MobileNetV2 (blok 15-18) agar adaptasi ke signer baru
    for p in model.features[15:].parameters():
        p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'Parameter dilatih: {sum(p.numel() for p in trainable):,}')

    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-3)
    best_val, best_state = 0.0, None
    no_improve = 0
    early_stop_patience = 4
    n_train_batch = len(tl)
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    import time, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t_total_start = time.time()

    for ep in range(1, args.epochs + 1):
        model.train(); tr_correct = tr_n = 0; tr_loss = 0.0
        t_ep = time.time()
        for i, (xb, yb) in enumerate(tl, 1):
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            opt.zero_grad()
            out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step()
            tr_loss += loss.item() * xb.size(0)
            tr_correct += (out.argmax(1) == yb).sum().item(); tr_n += xb.size(0)
            print(f'  Ep {ep:02d}/{args.epochs} batch {i}/{n_train_batch} '
                  f'loss={tr_loss/max(tr_n,1):.3f}', end='\r', flush=True)

        model.eval(); va_correct = va_n = 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(device), torch.tensor(yb).to(device)
                out = model(xb)
                va_correct += (out.argmax(1) == yb).sum().item(); va_n += xb.size(0)
        va_acc = va_correct / max(va_n, 1)
        tr_acc = tr_correct / max(tr_n, 1)
        ep_sec = time.time() - t_ep
        eta_menit = int((args.epochs - ep) * ep_sec / 60)
        flag = ' ← BEST' if va_acc >= best_val else ''
        print(f'Ep {ep:02d}/{args.epochs} | loss={tr_loss/max(tr_n,1):.3f} '
              f'train={tr_acc:.3f} val={va_acc:.3f} '
              f'| {ep_sec:.0f}s/ep ETA~{eta_menit}m{flag}')
        history['train_loss'].append(tr_loss / max(tr_n, 1))
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        if va_acc >= best_val:
            best_val = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, args.out)
            print(f'  [Saved] {args.out}')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= early_stop_patience:
                print(f'\n[Early Stop] Val acc tidak membaik selama {early_stop_patience} epoch → berhenti.')
                break

    # ── Evaluasi test set (independen, tidak dipakai saat training) ──────────
    model.load_state_dict(best_state)
    model.eval(); te_correct = te_n = 0
    with torch.no_grad():
        for xb, yb in tel:
            xb, yb = xb.to(device), torch.tensor(yb).to(device)
            out = model(xb)
            te_correct += (out.argmax(1) == yb).sum().item(); te_n += xb.size(0)
    test_acc = te_correct / max(te_n, 1)

    total_menit = int((time.time() - t_total_start) / 60)
    print(f'\n[SELESAI] {args.epochs} epoch dalam {total_menit} menit')
    print(f'Val_acc terbaik : {best_val:.4f}')
    print(f'Test_acc        : {test_acc:.4f}  ← angka ini yang dilaporkan ke skripsi')
    print(f'Checkpoint      : {args.out}')
    print(f'Jalankan server : python server_v10.py --checkpoint {args.out} --port 8000')

    # ── Grafik training history ───────────────────────────────────────────────
    graph_path = args.out.replace('.pth', '_history.png')
    epochs_range = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Fine-tuning History  |  val_acc_best={best_val:.4f}  test_acc={test_acc:.4f}',
                 fontsize=12, fontweight='bold')

    axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', color='steelblue')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, history['train_acc'], label='Train Acc', color='steelblue')
    axes[1].plot(epochs_range, history['val_acc'],   label='Val Acc',   color='orange')
    axes[1].axhline(y=test_acc, color='red', linestyle='--',
                    label=f'Test Acc = {test_acc:.4f}')
    axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(graph_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Grafik tersimpan : {graph_path}')


if __name__ == '__main__':
    main()
