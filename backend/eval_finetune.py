"""
eval_finetune.py — Generate confusion matrix dari model fine-tune
Mereproduksi split IDENTIK dengan finetune_hp.py (SEED=42, test_frac=0.15)
Output: penulisan/gambar/confusion_matrix_finetune.png
Jalankan: python backend/eval_finetune.py
"""
import os, sys, random
import numpy as np
import cv2
import torch
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import realtime_v10 as R

SEED       = 42
TEST_FRAC  = 0.15
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_custom')
CKPT       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_finetune_hp.pth')
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'penulisan', 'gambar')
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

_selfie = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
_dilate = np.ones((R.MASK_DILATE, R.MASK_DILATE), np.uint8)

def center_crop_square(bgr):
    h, w = bgr.shape[:2]
    s = min(h, w)
    return bgr[(h-s)//2:(h-s)//2+s, (w-s)//2:(w-s)//2+s]

def segment_frame(bgr):
    bgr = center_crop_square(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = _selfie.process(rgb).segmentation_mask
    binm = cv2.dilate((mask > 0.3).astype(np.uint8), _dilate, iterations=1)
    mask = np.clip(np.maximum(mask, binm.astype(np.float32) * 0.8), 0.0, 1.0)
    return R.composite(bgr, mask, 'blur')

def load_clips(classes):
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    X, y = [], []
    for cls in sorted(os.listdir(DATA_DIR)):
        cdir = os.path.join(DATA_DIR, cls)
        if not os.path.isdir(cdir) or cls not in cls_to_idx:
            continue
        vids = sorted([v for v in os.listdir(cdir) if v.lower().endswith(('.mp4','.avi','.mov','.mkv'))])
        print(f'  [{cls}] {len(vids)} video...')
        for v in vids:
            cap = cv2.VideoCapture(os.path.join(cdir, v))
            frames = []
            while True:
                ret, f = cap.read()
                if not ret: break
                frames.append(cv2.resize(segment_frame(f), R.IMG_SIZE))
            cap.release()
            if not frames: continue
            src = R.trim_active(frames)[0]
            f16 = R.sample_clip_eval(src, R.NUM_FRAMES, 0.15)
            X.append(np.stack(f16)); y.append(cls_to_idx[cls])
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
    print(f'Device: {device}')
    print(f'Checkpoint: {CKPT}')

    model, classes = R.load_model(CKPT, device)
    model.eval()
    print(f'Kelas ({len(classes)}): {classes}')

    print('\nMemproses klip (segmentasi)...')
    X, y = load_clips(classes)
    _selfie.close()
    print(f'Total klip: {len(X)}')

    # Stratified split — identik dengan finetune_hp.py yang sudah diperbaiki
    from collections import defaultdict
    cls_buckets = defaultdict(list)
    for i, label in enumerate(y):
        cls_buckets[label].append(i)

    # Sanity check: pastikan tiap kelas benar-benar punya klip sebelum split
    for ci, cls in enumerate(classes):
        n = len(cls_buckets.get(ci, []))
        if n == 0:
            print(f'  [PERINGATAN] kelas "{cls}" tidak punya klip valid di X,y!')
    print(f'Klip per kelas (setelah preprocessing): '
          f'{[len(cls_buckets.get(i, [])) for i in range(len(classes))]}')

    test_idx = []
    for label_idx in sorted(cls_buckets):
        bucket = cls_buckets[label_idx][:]
        random.shuffle(bucket)
        n_test = max(1, int(len(bucket) * TEST_FRAC))
        test_idx.extend(bucket[:n_test])
    print(f'Test set (stratified): {len(test_idx)} klip (~{TEST_FRAC*100:.0f}% per kelas)')

    # Evaluasi test set
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in test_idx:
            x = to_tensor(X[i]).unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(1).item()
            all_preds.append(pred)
            all_labels.append(y[i])

    # Akurasi
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f'\nTest accuracy: {acc*100:.2f}% ({sum(p==l for p,l in zip(all_preds,all_labels))}/{len(all_labels)})')

    # Classification report manual
    n_cls = len(classes)
    print(f'\n{"Kelas":<20} {"Precision":>10} {"Recall":>10} {"F1":>10} {"Support":>10}')
    print('-' * 60)
    cm = np.zeros((n_cls, n_cls), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t][p] += 1
    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        sup  = cm[i, :].sum()
        print(f'{cls:<20} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {sup:>10}')
    print('-' * 60)
    macro_p = np.mean([cm[i,i]/(cm[:,i].sum()) if cm[:,i].sum()>0 else 0 for i in range(n_cls)])
    macro_r = np.mean([cm[i,i]/(cm[i,:].sum()) if cm[i,:].sum()>0 else 0 for i in range(n_cls)])
    print(f'{"macro avg":<20} {macro_p:>10.3f} {macro_r:>10.3f}')
    labels_id = [c.capitalize().replace('terimakasih','Terima Kasih') for c in classes]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_id, yticklabels=labels_id,
                linewidths=0.3, linecolor='#cccccc', ax=ax)
    ax.set_xlabel('Prediksi', fontsize=13)
    ax.set_ylabel('Label Aktual', fontsize=13)
    ax.set_title(f'Confusion Matrix — Model Fine-tune (best_finetune_hp.pth)\n'
                 f'Test set: {len(test_idx)} klip | Akurasi: {acc*100:.2f}%',
                 fontsize=13, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, 'confusion_matrix_finetune.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n[OK] {out_path}')
