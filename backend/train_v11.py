"""
BISINDO 24-Kelas v11 — Training (MobileNetV2 + TSM + CBAM)
=========================================================
FOKUS v11: menutup DOMAIN GAP training vs inference yang membuat model "menebak"
di real-time (test v10 = 97% tapi kamera HP ~tebakan). Akar masalahnya BUKAN
arsitektur (test tinggi) melainkan input inference != input training.

Perubahan vs v10 (3 hal, satu kelompok variabel: ALIGNMENT + robustness):
  1. center_crop_square di extract_frames  ── PENTING.
     v10 me-resize paksa frame penuh -> 224 (gepeng), padahal server_v10.py
     melakukan center_crop_square DULU baru resize. v11 menyamakan geometri
     training dgn inference (skala/posisi tangan kini cocok).
  2. Augmentasi domain-gap (konsisten per-video): Gaussian blur ringan,
     kompresi JPEG, dan random erasing -> meniru kualitas kamera HP + base64
     JPEG + segmentasi MediaPipe yang tepinya bocor / memotong.
  3. Kalibrasi threshold otomatis dari probabilitas test -> angka berbasis data
     untuk CONF_THRESHOLD di server (v10 hardcoded 22%).

TIDAK diubah (agar checkpoint kompatibel dgn realtime_v10.py / server_v10.py):
  arsitektur (TSM placement, CBAM, reduce 256 + grid 4x4), split, seed, epoch.
  Penyelarasan TSM ke paper Zhang 2022 adalah eksperimen ablasi TERPISAH, bukan
  penyebab kegagalan real-time (training & inference pakai TSM yang sama).

Nama file OUTPUT sengaja TANPA "v11" (agar tidak muncul di judul confusion
matrix); hanya folder (SAVE_DIR) yang boleh memuat v11.

Jalankan:
    python train_v11.py

Output: best_stage1.pth, best_stage2.pth, bisindo_24_checkpoint.pth,
        training_history.png, confusion_matrix_test.png

PENTING setelah training: agar geometri tetap cocok, pastikan SEMUA jalur
inference melakukan center_crop_square sebelum resize:
  - server_v10.py    : SUDAH (segment_frame -> center_crop_square).
  - realtime_v10.py  : BELUM -> tambahkan center_crop di RealtimeSegmenter.
"""

import os, re, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models.mobilenetv2 import InvertedResidual
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

# ─── Path & Konfigurasi (via argparse agar portabel) ────────────────────────
import argparse as _ap
_parser = _ap.ArgumentParser(add_help=False)
_parser.add_argument('--data1',    default=None,  help='Folder data Kaggle (wajib)')
_parser.add_argument('--data2',    default=None,  help='Folder data custom/signer sendiri (opsional)')
_parser.add_argument('--out-dir',  default='v11_out', help='Folder output')
_parser.add_argument('--epochs1',  type=int, default=12)
_parser.add_argument('--epochs2',  type=int, default=45)
_parser.add_argument('--workers',  type=int, default=0,
                     help='DataLoader num_workers (0=safe di Windows)')
_args, _ = _parser.parse_known_args()

if _args.data1 is None:
    print('[ERROR] Wajib isi --data1 (folder data Kaggle)')
    print('Contoh: python train_v11.py --data1 D:/kaggle_data --data2 D:/data_custom')
    import sys; sys.exit(1)

DATASET_PATH = _args.data1
DATASET_PATH2 = _args.data2   # boleh None
SAVE_DIR     = _args.out_dir
EPOCH_STAGE1 = _args.epochs1
EPOCH_STAGE2 = _args.epochs2
NUM_WORKERS  = _args.workers
os.makedirs(SAVE_DIR, exist_ok=True)

CKPT_S1 = os.path.join(SAVE_DIR, 'best_stage1.pth')
CKPT_S2 = os.path.join(SAVE_DIR, 'best_stage2.pth')

EXPLICIT_CLASSES = [
    'air', 'bagaimana', 'belajar', 'berangkat', 'cari',
    'datang', 'dengar', 'dimana', 'hijau', 'merah',
    'kapan', 'keluarga', 'kuning', 'lagi', 'maaf',
    'makan', 'mengapa', 'motor', 'rumah', 'saya',
    'siapa', 'teman', 'terimakasih', 'tuli'
]

NUM_CLASSES           = 24
NUM_FRAMES            = 16
IMG_SIZE              = (224, 224)
MARGIN                = 0.15
BATCH_SIZE            = 8
EPOCH_STAGE1          = 12
EPOCH_STAGE2          = 45
LR_STAGE1             = 1e-3
LR_STAGE2             = 1e-4
WEIGHT_DECAY          = 1e-3
EARLY_STOP_PATIENCE   = 7
LR_SCHEDULER_FACTOR   = 0.5
LR_SCHEDULER_PATIENCE = 3
LABEL_SMOOTHING       = 0.1
DROPOUT               = 0.5
NUM_WORKERS           = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device  : {device}')
print(f'Data1   : {DATASET_PATH}')
print(f'Data2   : {DATASET_PATH2 or "(tidak ada)"}')
print(f'Out dir : {SAVE_DIR}')


# ─── Data Loading + Random Stratified Split (70:15:15) ──────────────────────
def parse_signer(filename):
    name = os.path.basename(filename).lower()
    m = re.search(r'signer(\d+)', name)
    if m: return int(m.group(1))
    m = re.search(r'(?:^|[^a-z])s(\d+)[_\-]', name)
    if m: return int(m.group(1))
    return -1


def load_dataset(dataset_path, classes):
    video_files, labels, signers = [], [], []
    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            print(f'  [WARN] folder kelas tidak ada: {class_name}')
            continue
        for vname in sorted(os.listdir(class_path)):
            if vname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_files.append(os.path.join(class_path, vname))
                labels.append(label)
                signers.append(parse_signer(vname))
    print(f'Loaded {len(video_files)} video dari {len(classes)} kelas')
    return video_files, labels, signers


# ─── [FIX 1] Center crop square — IDENTIK server_v10.py.segment_frame ────────
# Menyamakan geometri training dgn inference: frame portrait (HP) & landscape
# (dataset) dipotong tengah jadi kotak SEBELUM resize, bukan di-resize gepeng.
def center_crop_square(frame):
    h, w = frame.shape[:2]
    s = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return frame[y0:y0 + s, x0:x0 + s]


# ─── Frame Extraction (margin 0.15) + center crop ───────────────────────────
def extract_frames(video_path, num_frames, img_size=(224, 224),
                   is_train=False, margin=MARGIN):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)] * num_frames

    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    start_frame = int(total * margin)
    end_frame   = int(total * (1.0 - margin))
    valid_total = max(end_frame - start_frame, 1)
    seg_size    = valid_total / float(num_frames)

    indices = []
    for i in range(num_frames):
        s = int(start_frame + i * seg_size)
        e = int(start_frame + (i + 1) * seg_size)
        if is_train and e > s:
            idx = np.random.randint(s, e)
        else:
            idx = s + (e - s) // 2
        indices.append(min(idx, total - 1))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = center_crop_square(frame)         # [FIX 1] samakan dgn inference
            frame = cv2.resize(frame, img_size)
            frames.append(frame)
        else:
            frames.append(frames[-1] if frames else
                          np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8))
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


# ─── [FIX 2] Augmentasi domain-gap (kualitas kamera HP) ──────────────────────
def _jpeg_compress(bgr, quality):
    ok, enc = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else bgr


# ─── Augmentasi konsisten per-video ─────────────────────────────────────────
# SPASIAL & FOTOMETRIK: identik v10. BARU v11: blur + JPEG + erasing (domain-gap).
# Semua nilai acak dipilih SEKALI per video lalu dipakai utk semua frame.
class ConsistentVideoAugment:
    def __init__(self, train=True, img_size=224):
        self.train = train
        self.img_size = img_size

    def __call__(self, frames_bgr):
        T = len(frames_bgr)
        W = self.img_size

        if self.train:
            # Spasial (v9/v10)
            angle     = random.uniform(-7, 7)
            translate = (random.uniform(-0.07, 0.07) * W,
                         random.uniform(-0.07, 0.07) * W)
            scale     = random.uniform(0.88, 1.12)
            # Fotometrik (v10)
            b = random.uniform(0.6, 1.4)
            c = random.uniform(0.6, 1.4)
            s = random.uniform(0.8, 1.2)
            h = random.uniform(-0.08, 0.08)
            g = random.uniform(0.7, 1.5)
            # Domain-gap (v11) — meniru kamera HP + JPEG + segmentasi bocor
            do_blur = random.random() < 0.30
            blur_k  = random.choice([3, 5])
            do_jpeg = random.random() < 0.40
            jpeg_q  = random.randint(40, 80)
            do_erase = random.random() < 0.25
            ey, ex = random.uniform(0.0, 0.7), random.uniform(0.0, 0.7)
            eh, ew = random.uniform(0.10, 0.30), random.uniform(0.10, 0.30)

        out = []
        for t in range(T):
            f = frames_bgr[t]
            if self.train:
                if do_blur:
                    f = cv2.GaussianBlur(f, (blur_k, blur_k), 0)
                if do_jpeg:
                    f = _jpeg_compress(f, jpeg_q)
            rgb_np = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb_np)
            if self.train:
                pil = TF.affine(pil, angle=angle, translate=list(translate),
                                scale=scale, shear=[0.0, 0.0])
                pil = TF.adjust_brightness(pil, b)
                pil = TF.adjust_contrast(pil, c)
                pil = TF.adjust_saturation(pil, s)
                pil = TF.adjust_hue(pil, h)
                pil = TF.adjust_gamma(pil, g)
            tens = TF.to_tensor(pil)
            if self.train and do_erase:                # oklusi kotak konsisten
                Ht, Wt = tens.shape[1], tens.shape[2]
                y0, x0 = int(ey * Ht), int(ex * Wt)
                hh, ww = int(eh * Ht), int(ew * Wt)
                tens[:, y0:y0 + hh, x0:x0 + ww] = 0.0
            tens = TF.normalize(tens, IMAGENET_MEAN, IMAGENET_STD)
            out.append(tens)
        return torch.stack(out)   # (T, 3, H, W)


class VideoDataset(Dataset):
    def __init__(self, video_files, labels, signers, augmentor,
                 num_frames=16, is_train=False):
        self.video_files = video_files
        self.labels      = labels
        self.signers     = signers
        self.augmentor   = augmentor
        self.num_frames  = num_frames
        self.is_train    = is_train

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        frames = extract_frames(self.video_files[idx], self.num_frames,
                                is_train=self.is_train)
        video = self.augmentor(frames)
        return video, self.labels[idx], self.signers[idx]


# ─── Model: MobileNetV2 + TSM + CBAM (IDENTIK v10 → checkpoint kompatibel) ────
class TemporalShift(nn.Module):
    def __init__(self, n_segment=16, n_div=8):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div  = n_div

    def forward(self, x):
        BT, C, H, W = x.size()
        B = BT // self.n_segment
        x = x.view(B, self.n_segment, C, H, W)
        fold = C // self.fold_div
        out = torch.zeros_like(x)
        out[:, 1:,    :fold]      = x[:, :-1, :fold]
        out[:, :-1, fold:2*fold]  = x[:, 1:,  fold:2*fold]
        out[:, :,   2*fold:]      = x[:, :,   2*fold:]
        return out.view(BT, C, H, W)


def inject_tsm(model, n_segment=16, n_div=8):
    count = 0
    for block in model.features:
        if isinstance(block, InvertedResidual):
            orig = block.conv[0]
            block.conv[0] = nn.Sequential(TemporalShift(n_segment, n_div), orig)
            count += 1
    print(f'  TSM injected ke {count} InvertedResidual blocks')
    return model


class ChannelAttention(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.mx  = nn.AdaptiveMaxPool2d(1)
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
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        backbone = inject_tsm(backbone, n_segment=num_frames, n_div=8)
        self.features = backbone.features
        feat_dim      = backbone.last_channel   # 1280

        self.cbam   = CBAM(feat_dim) if use_cbam else nn.Identity()
        self.reduce = nn.Sequential(
            nn.Conv2d(feat_dim, reduce_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True))
        self.pool        = nn.AdaptiveAvgPool2d((grid, grid))
        self.spatial_dim = reduce_dim * grid * grid

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.spatial_dim, num_classes))

    def forward(self, x):   # (B, T, 3, H, W)
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.cbam(x)
        x = self.reduce(x)
        x = self.pool(x).view(B, T, -1)
        x = x.mean(dim=1)
        return self.classifier(x)


def freeze_backbone(model):
    for p in model.features.parameters():
        p.requires_grad = False
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  [Stage 1] Backbone frozen — trainable: {n:,}')


def unfreeze_partial(model, from_layer=14):
    for p in model.features.parameters():
        p.requires_grad = False
    for layer in list(model.features.children())[from_layer:]:
        for p in layer.parameters():
            p.requires_grad = True
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  [Stage 2] Unfreeze from layer {from_layer} — trainable: {n:,}')


# ─── Training utilities ─────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=7, min_delta=1e-4, path='best.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = float('inf')
        self.counter = 0
        self.stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
            print(f'    [v] Val loss {val_loss:.4f} membaik — checkpoint disimpan')
        else:
            self.counter += 1
            print(f'    [-] Tidak ada peningkatan ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.stop = True
                model.load_state_dict(torch.load(self.path, map_location=device))
                print(f'    [x] Early stop. Bobot terbaik dimuat.')


def run_epoch(model, loader, criterion, optimizer=None, is_train=True, desc=''):
    model.train() if is_train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    per_signer_correct = defaultdict(int)
    per_signer_total   = defaultdict(int)

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for inputs, labels_batch, signers_batch in tqdm(loader, desc=desc, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            if is_train:
                optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_batch)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                correct = (preds == labels_batch)
                total_loss    += loss.item() * inputs.size(0)
                total_correct += correct.sum().item()
                total_n       += inputs.size(0)
                for sig, ok in zip(signers_batch.tolist(), correct.cpu().tolist()):
                    per_signer_total[sig]   += 1
                    per_signer_correct[sig] += int(ok)

    per_signer_acc = {s: per_signer_correct[s] / per_signer_total[s]
                      for s in per_signer_total}
    return total_loss / total_n, total_correct / total_n, per_signer_acc


def train_model(model, criterion, train_loader, val_loader, num_epochs,
                optimizer, scheduler=None, early_stopping=None, stage_name=''):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'val_per_signer': [], 'lr': []}
    for ep in range(1, num_epochs + 1):
        t_loss, t_acc, _ = run_epoch(model, train_loader, criterion, optimizer,
                                     is_train=True, desc=f'{stage_name} E{ep:02d} Train')
        v_loss, v_acc, v_sig = run_epoch(model, val_loader, criterion, None,
                                         is_train=False, desc=f'{stage_name} E{ep:02d} Val')
        lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(t_loss); history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss);   history['val_acc'].append(v_acc)
        history['val_per_signer'].append(v_sig); history['lr'].append(lr)

        v_sig_str = ' | '.join(f's{s}:{a:.2f}' for s, a in sorted(v_sig.items()))
        print(f'[{stage_name}] Ep {ep:02d} | Train L={t_loss:.4f} A={t_acc:.4f} | '
              f'Val L={v_loss:.4f} A={v_acc:.4f} | {v_sig_str} | LR={lr:.2e}')

        if scheduler:
            scheduler.step(v_loss)
        if early_stopping:
            early_stopping(v_loss, model)
            if early_stopping.stop:
                print(f'  Stopped at epoch {ep}/{num_epochs}')
                break
    return history


def plot_history(h1, h2, save_path):
    s1, s2 = len(h1['train_loss']), len(h2['train_loss'])
    if s1 + s2 == 0:
        print('[Plot] Tidak ada data.'); return
    eps = list(range(1, s1 + s2 + 1))
    tl = h1['train_loss'] + h2['train_loss']; vl = h1['val_loss'] + h2['val_loss']
    ta = h1['train_acc']  + h2['train_acc'];  va = h1['val_acc']  + h2['val_acc']
    fig, (axL, axA) = plt.subplots(1, 2, figsize=(14, 5))
    axL.plot(eps, tl, label='Train'); axL.plot(eps, vl, label='Val')
    axL.axvline(s1 + 0.5, color='red', ls='--', label='S1->S2')
    axL.set(title='Loss', xlabel='Epoch'); axL.legend(); axL.grid(alpha=0.3)
    axA.plot(eps, ta, label='Train'); axA.plot(eps, va, label='Val')
    axA.axvline(s1 + 0.5, color='red', ls='--')
    axA.set(title='Accuracy', xlabel='Epoch'); axA.legend(); axA.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=140, bbox_inches='tight')
    print(f'[Plot] Saved: {save_path}')


# ─── [FIX 3] Kalibrasi threshold confidence dari probabilitas test ───────────
def calibrate_threshold(all_probs, all_preds, all_labels, target_prec=0.95):
    maxp    = all_probs.max(axis=1) * 100.0          # confidence (%) tiap sampel
    correct = (np.array(all_preds) == np.array(all_labels))
    print('\n=== KALIBRASI THRESHOLD (confidence %) ===')
    print('  thr  | coverage | acc_diterima')
    best_thr = None
    for thr in range(10, 95, 5):
        acc_mask = maxp >= thr
        if acc_mask.sum() == 0:
            continue
        coverage = acc_mask.mean()
        prec     = correct[acc_mask].mean()
        print(f'  {thr:3d}% |  {coverage*100:5.1f}% |   {prec*100:5.1f}%')
        if best_thr is None and prec >= target_prec:
            best_thr = thr
    if best_thr is not None:
        cov = (maxp >= best_thr).mean()
        print(f'\n  >> Rekomendasi CONF_THRESHOLD = {best_thr}%  '
              f'(akurasi diterima >= {target_prec*100:.0f}%, coverage {cov*100:.1f}%)')
        print(f'     Ganti di server_v10.py:  CONF_THRESHOLD = {best_thr}.0')
    else:
        print('\n  >> Tidak ada threshold yang capai target presisi; pakai default.')
    return best_thr


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    # 1. Load + gabung data1 (Kaggle) dan data2 (custom) → split 70:15:15
    video_files, labels, signers = load_dataset(DATASET_PATH, EXPLICIT_CLASSES)
    if len(video_files) == 0:
        print('[ERROR] Tidak ada video di --data1.')
        return
    print(f'Data1: {len(video_files)} video')

    if DATASET_PATH2 and os.path.isdir(DATASET_PATH2):
        vf2, lb2, sg2 = load_dataset(DATASET_PATH2, EXPLICIT_CLASSES)
        # Tandai signer custom dengan ID unik (9999) agar terlacak
        sg2 = [9999] * len(vf2)
        video_files += vf2; labels += lb2; signers += sg2
        print(f'Data2: {len(vf2)} video → Total gabungan: {len(video_files)} video')
    else:
        print(f'Data1 saja: {len(video_files)} video')

    train_idx, temp_idx = train_test_split(
        range(len(video_files)), test_size=0.30, random_state=SEED, stratify=labels)
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=temp_labels)

    def pick(idxs): return ([video_files[i] for i in idxs],
                            [labels[i] for i in idxs],
                            [signers[i] for i in idxs])
    train_vf, train_lb, train_sg = pick(train_idx)
    val_vf,   val_lb,   val_sg   = pick(val_idx)
    test_vf,  test_lb,  test_sg  = pick(test_idx)
    print(f'Split → Train {len(train_vf)} | Val {len(val_vf)} | Test {len(test_vf)}')

    assert not (set(train_vf) & set(val_vf)) and not (set(train_vf) & set(test_vf)) \
        and not (set(val_vf) & set(test_vf)), 'LEAKAGE: video overlap antar split'

    train_aug = ConsistentVideoAugment(train=True)
    eval_aug  = ConsistentVideoAugment(train=False)
    train_ds = VideoDataset(train_vf, train_lb, train_sg, train_aug, NUM_FRAMES, is_train=True)
    val_ds   = VideoDataset(val_vf,   val_lb,   val_sg,   eval_aug,  NUM_FRAMES, is_train=False)
    test_ds  = VideoDataset(test_vf,  test_lb,  test_sg,  eval_aug,  NUM_FRAMES, is_train=False)

    _pw = NUM_WORKERS > 0
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=_pw)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=_pw)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=_pw)

    # 2. Build model
    print('\nBuilding model v11...')
    model = BISINDOClassifier(NUM_CLASSES, NUM_FRAMES, dropout=DROPOUT).to(device)
    print(f'  Total parameter: {sum(p.numel() for p in model.parameters()):,}')
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    # 3. Stage 1 — frozen backbone
    print('\n' + '=' * 60 + '\nSTAGE 1 — FROZEN BACKBONE\n' + '=' * 60)
    freeze_backbone(model)
    opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_STAGE1, weight_decay=WEIGHT_DECAY)
    sch1 = optim.lr_scheduler.ReduceLROnPlateau(opt1, mode='min',
                       factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
    es1 = EarlyStopping(EARLY_STOP_PATIENCE, path=CKPT_S1)
    h1 = train_model(model, criterion, train_loader, val_loader, EPOCH_STAGE1,
                     opt1, sch1, es1, stage_name='S1')

    # 4. Stage 2 — fine-tuning
    print('\n' + '=' * 60 + '\nSTAGE 2 — FINE-TUNING\n' + '=' * 60)
    unfreeze_partial(model, from_layer=14)
    opt2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR_STAGE2, weight_decay=WEIGHT_DECAY * 5)
    sch2 = optim.lr_scheduler.ReduceLROnPlateau(opt2, mode='min',
                       factor=LR_SCHEDULER_FACTOR, patience=LR_SCHEDULER_PATIENCE)
    es2 = EarlyStopping(EARLY_STOP_PATIENCE, path=CKPT_S2)
    h2 = train_model(model, criterion, train_loader, val_loader, EPOCH_STAGE2,
                     opt2, sch2, es2, stage_name='S2')

    plot_history(h1, h2, os.path.join(SAVE_DIR, 'training_history.png'))

    # 5. Final test evaluation
    print('\n' + '=' * 60 + '\nFINAL TEST EVALUATION\n' + '=' * 60)
    for ckpt in [CKPT_S2, CKPT_S1]:
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location=device))
            print(f'Bobot dimuat: {ckpt}'); break
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for inputs, lb, _ in tqdm(test_loader, desc='Test'):
            inputs = inputs.to(device)
            probs = F.softmax(model(inputs), dim=1)
            all_probs.append(probs.cpu())
            all_preds.extend(probs.argmax(1).cpu().tolist())
            all_labels.extend(lb.tolist())
    all_probs = torch.cat(all_probs).numpy()

    test_acc    = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1    = f1_score(all_labels, all_preds, average='macro')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'\n--- METRICS ---\n  Top-1 Acc : {test_acc:.4f}\n'
          f'  Macro F1  : {macro_f1:.4f}\n  Weighted F1: {weighted_f1:.4f}')
    print('\n=== CLASSIFICATION REPORT ===')
    print(classification_report(all_labels, all_preds,
                                target_names=EXPLICIT_CLASSES, digits=4))

    best_thr = calibrate_threshold(all_probs, all_preds, all_labels)

    # 6. Confusion matrix (judul TANPA versi)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EXPLICIT_CLASSES, yticklabels=EXPLICIT_CLASSES,
                linewidths=0.5, linecolor='gray')
    plt.title(f'Confusion Matrix — Test (Acc={test_acc:.4f}, F1={macro_f1:.4f})')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(SAVE_DIR, 'confusion_matrix_test.png')
    plt.savefig(cm_path, dpi=140, bbox_inches='tight')
    print(f'[CM] Saved: {cm_path}')

    # 7. Save checkpoint + metadata
    ckpt_path = os.path.join(SAVE_DIR, 'bisindo_24_checkpoint.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': EXPLICIT_CLASSES, 'num_frames': NUM_FRAMES,
        'num_classes': NUM_CLASSES,
        'history_stage1': h1, 'history_stage2': h2,
        'test_metrics': {'accuracy': float(test_acc),
                         'macro_f1': float(macro_f1),
                         'weighted_f1': float(weighted_f1)},
        'recommended_conf_threshold': best_thr,
        'config': {'dropout': DROPOUT, 'margin': MARGIN,
                   'augment': 'spatial=v9, photometric=v10, domain-gap=v11 (blur+jpeg+erase)',
                   'preprocess': 'center_crop_square (align dgn inference)',
                   'dataset': 'selfie-segmented'},
    }, ckpt_path)
    print(f'[Checkpoint] Saved: {ckpt_path}')
    print('\n[SELESAI] Salin best_stage2.pth ke folder app untuk inference.')
    print('  PENTING: pastikan realtime_v10.py juga center_crop_square sebelum resize.')


if __name__ == '__main__':
    main()
