"""
eval_finetune_v2.py — Confusion matrix untuk best_finetune_v2.pth
Merekonstruksi split 80/20 IDENTIK dengan finetune_v10.py (SEED=42, val_frac=0.2,
shuffle global non-stratified) lalu evaluasi HANYA pada subset validasi
(20% data yang tidak pernah dilihat model saat training).

Checkpoint ini adalah yang benar-benar dipakai aplikasi (run_24kata.ps1),
BEDA metode dengan best_finetune_hp.pth (lihat catatan di eval_finetune.py).

Output: penulisan/gambar/confusion_matrix_finetune_v2.png
Jalankan: python backend/eval_finetune_v2.py
"""
import os, sys, random
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realtime_v10 as R

SEED     = 42
VAL_FRAC = 0.2
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_custom')
CKPT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_finetune_v2.pth')
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'penulisan', 'gambar')
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def load_clips(classes):
    """IDENTIK finetune_v10.py::load_clips — segmentasi RealtimeSegmenter('blur'),
    TANPA center_crop (beda dgn finetune_hp.py/eval_finetune.py)."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    seg = R.RealtimeSegmenter('blur')
    X, y, paths = [], [], []
    for cls in sorted(os.listdir(DATA_DIR)):
        cdir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cdir) or cls not in cls_to_idx:
            continue
        vids = sorted(v for v in os.listdir(cdir) if v.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')))
        for v in vids:
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
            f16 = R.sample_clip_eval(src, R.NUM_FRAMES, 0.15)
            X.append(np.stack(f16)); y.append(cls_to_idx[cls]); paths.append(v)
    seg.close()
    return X, y, paths


def to_tensor(frames_bgr):
    out = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.array(R.IMAGENET_MEAN)) / np.array(R.IMAGENET_STD)
        out.append(torch.from_numpy(rgb).permute(2, 0, 1).float())
    return torch.stack(out)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device} | Checkpoint: {CKPT}')

    model, classes = R.load_model(CKPT, device)
    model.eval()
    print(f'Kelas ({len(classes)}): {classes}')

    print('\nMemproses klip (segmentasi, TANPA center_crop -> identik finetune_v10.py)...')
    X, y, paths = load_clips(classes)
    print(f'Total klip: {len(X)}')

    # Split IDENTIK finetune_v10.py: shuffle GLOBAL (bukan stratified), SEED=42
    idx = list(range(len(X))); random.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_FRAC))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f'Val (held-out, TIDAK stratified -> cek distribusi kelas di bawah): {len(val_idx)} klip')

    from collections import Counter
    val_dist = Counter(y[i] for i in val_idx)
    print('Distribusi kelas di val set:',
          {classes[k]: v for k, v in sorted(val_dist.items())})

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in val_idx:
            x = to_tensor(X[i]).unsqueeze(0).to(device)
            pred = model(x).argmax(1).item()
            all_preds.append(pred)
            all_labels.append(y[i])

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f'\nAkurasi pada val set (held-out): {acc*100:.2f}% '
          f'({sum(p==l for p,l in zip(all_preds,all_labels))}/{len(all_labels)})')

    n_cls = len(classes)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1

    print(f'\n{"Kelas":<20} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
    print('-' * 60)
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        sup  = cm[i, :].sum()
        print(f'{cls:<20} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {sup:>10}')

    labels_id = [c.capitalize().replace('terimakasih', 'Terima Kasih') for c in classes]
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_id, yticklabels=labels_id,
                linewidths=0.3, linecolor='#cccccc', ax=ax)
    ax.set_xlabel('Prediksi', fontsize=13)
    ax.set_ylabel('Label Aktual', fontsize=13)
    ax.set_title(f'Confusion Matrix — Model Fine-tune (best_finetune_v2.pth)\n'
                 f'Val set (held-out, non-stratified): {len(val_idx)} klip | Akurasi: {acc*100:.2f}%',
                 fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'confusion_matrix_finetune_v2.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[OK] {out_path}')
