"""
eval_threshold.py — Analisis error-reject tradeoff (Chow, 1970) untuk menentukan
CONF_THRESHOLD berbasis data, bukan angka bulat 50% tanpa dasar.

Split IDENTIK eval_finetune_v2.py (SEED=42, val_frac=0.2, non-stratified,
sesuai finetune_v10.py -> best_finetune_v2.pth).

Untuk tiap ambang confidence t (0%..95%):
  - Coverage = % sampel yang confidence-nya >= t (sistem berani menjawab)
  - Risk     = % SALAH di antara sampel yang diterima (bukan "Belum dikenali")
  - Akurasi pada sampel diterima = 1 - Risk

Output:
  penulisan/gambar/error_reject_tradeoff.png
  penulisan/gambar/tabel_threshold_analysis.txt
Jalankan: python backend/eval_threshold.py
"""
import os, sys, random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    """IDENTIK finetune_v10.py / eval_finetune_v2.py -> RealtimeSegmenter, tanpa center_crop."""
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    seg = R.RealtimeSegmenter('blur')
    X, y = [], []
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
            X.append(np.stack(f16)); y.append(cls_to_idx[cls])
    seg.close()
    return X, y


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

    print('Memproses klip (identik pipeline eval_finetune_v2.py)...')
    X, y = load_clips(classes)
    print(f'Total klip: {len(X)}')

    # Split IDENTIK finetune_v10.py -> hanya evaluasi di val set (held-out)
    idx = list(range(len(X))); random.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_FRAC))
    val_idx = idx[:n_val]
    print(f'Val set (held-out): {len(val_idx)} klip\n')

    confidences, corrects = [], []
    with torch.no_grad():
        for i in val_idx:
            x = to_tensor(X[i]).unsqueeze(0).to(device)
            probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
            pred = int(probs.argmax())
            confidences.append(float(probs[pred]))
            corrects.append(pred == y[i])

    confidences = np.array(confidences)
    corrects = np.array(corrects)
    print(f'Akurasi top-1 (tanpa threshold, seluruh val set): '
          f'{corrects.mean()*100:.2f}% ({corrects.sum()}/{len(corrects)})\n')

    # ── Sweep threshold: Chow's error-reject tradeoff ─────────────────────────
    thresholds = np.arange(0.0, 1.00, 0.05)
    rows = []
    for t in thresholds:
        accepted = confidences >= t
        n_accept = accepted.sum()
        coverage = n_accept / len(confidences)
        if n_accept > 0:
            acc_accepted = corrects[accepted].mean()
            risk = 1 - acc_accepted
        else:
            acc_accepted, risk = float('nan'), float('nan')
        rows.append((t, coverage, acc_accepted, risk))

    print(f'{"Threshold":>10} {"Coverage":>10} {"Akurasi(diterima)":>18} {"Risk":>8}')
    print('-' * 50)
    lines = [f'{"Threshold":>10} {"Coverage":>10} {"Akurasi(diterima)":>18} {"Risk":>8}', '-' * 50]
    for t, cov, acc, risk in rows:
        line = f'{t*100:>9.0f}% {cov*100:>9.1f}% {acc*100 if not np.isnan(acc) else float("nan"):>17.1f}% {risk*100 if not np.isnan(risk) else float("nan"):>7.1f}%'
        print(line)
        lines.append(line)

    txt_path = os.path.join(OUT_DIR, 'tabel_threshold_analysis.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n[OK] {txt_path}')

    # ── Grafik error-reject tradeoff ───────────────────────────────────────────
    covs  = [r[1]*100 for r in rows]
    accs  = [r[2]*100 for r in rows]
    risks = [r[3]*100 for r in rows]
    ts    = [r[0]*100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(ts, covs, 'o-', color='steelblue', label='Coverage (% dijawab)')
    ax1.plot(ts, accs, 's-', color='seagreen', label='Akurasi pada sampel diterima')
    ax1.axvline(50, color='red', linestyle='--', alpha=0.6, label='Threshold saat ini (50%)')
    ax1.set_xlabel('Confidence Threshold (%)', fontsize=12)
    ax1.set_ylabel('Persentase (%)', fontsize=12)
    ax1.set_title("Error-Reject Tradeoff (Chow, 1970)\nbest_finetune_v2.pth pada val set held-out",
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 105)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'error_reject_tradeoff.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] {out_path}')
