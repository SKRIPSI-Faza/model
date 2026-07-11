"""
Visualisasi hasil preprocessing untuk Bab 4 skripsi.
Menghasilkan:
  1. contoh_16frame.png        — 16 frame hasil center sampling
  2. contoh_augmentasi.png     — original vs 7 jenis augmentasi (per baris)
  3. contoh_segmentasi.png     — frame asli vs frame tersegmentasi
Simpan ke: penulisan/gambar/
"""

import cv2
import numpy as np
import random
import os
import sys
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = os.path.dirname(SCRIPT_DIR)
OUT_DIR     = os.path.join(ROOT_DIR, 'penulisan', 'gambar')
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_DIR = r'D:\WLBISINDO_raw\WLBISINDO_raw'
SAMPLE_WORD = 'saya'
SAMPLE_FILE = 'signer0_label9_sample1.mp4'
VIDEO_PATH  = os.path.join(DATASET_DIR, SAMPLE_WORD, SAMPLE_FILE)

MARGIN      = 0.15
NUM_FRAMES  = 16
IMG_SIZE    = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ── helpers ───────────────────────────────────────────────────────────────────

def extract_frames_center(video_path, num_frames=16, margin=MARGIN):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
    total     = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    start_f   = int(total * margin)
    end_f     = int(total * (1.0 - margin))
    valid_tot = max(end_f - start_f, 1)
    seg_size  = valid_tot / float(num_frames)
    indices   = []
    for i in range(num_frames):
        s = int(start_f + i * seg_size)
        e = int(start_f + (i + 1) * seg_size)
        indices.append(min(s + (e - s) // 2, total - 1))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def bgr_to_pil(bgr):
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def pil_to_np(pil):
    return np.array(pil)


# ══════════════════════════════════════════════════════════════════════════════
# 1. GAMBAR: 16 frame center sampling
# ══════════════════════════════════════════════════════════════════════════════

def buat_16frame():
    frames = extract_frames_center(VIDEO_PATH)
    fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
    fig.patch.set_facecolor('#1a1a2e')
    for idx, (ax, frame) in enumerate(zip(axes.flat, frames)):
        ax.imshow(frame)
        ax.set_title(f'Frame {idx+1}', color='white', fontsize=9, pad=4)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#4a9eff')
            spine.set_linewidth(1.5)
    fig.suptitle(f'Contoh Ekstraksi 16 Frame — Kata "{SAMPLE_WORD.capitalize()}" (Center Sampling, margin=15%)',
                 color='white', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout(pad=0.6)
    out = os.path.join(OUT_DIR, 'contoh_16frame_preprocessing.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ══════════════════════════════════════════════════════════════════════════════
# 2. GAMBAR: Augmentasi — original vs tiap jenis
# ══════════════════════════════════════════════════════════════════════════════

def buat_augmentasi():
    # Ambil satu frame representatif (frame ke-8, tengah klip)
    frames = extract_frames_center(VIDEO_PATH)
    base_np = frames[7]           # numpy RGB
    base_pil = Image.fromarray(base_np)

    random.seed(42)
    aug_params = {
        'angle'    : random.uniform(-7, 7),
        'translate': (random.uniform(-0.07, 0.07) * 224, random.uniform(-0.07, 0.07) * 224),
        'scale'    : random.uniform(0.88, 1.12),
        'b'        : random.uniform(0.6, 1.4),
        'c'        : random.uniform(0.6, 1.4),
        's'        : random.uniform(0.8, 1.2),
        'h'        : random.uniform(-0.08, 0.08),
        'g'        : random.uniform(0.7, 1.5),
    }

    def apply_aug(pil, kind):
        p = aug_params
        if kind == 'rotation':
            return TF.affine(pil, angle=p['angle'], translate=[0,0], scale=1.0, shear=[0,0])
        elif kind == 'scale':
            return TF.affine(pil, angle=0, translate=[0,0], scale=p['scale'], shear=[0,0])
        elif kind == 'translation':
            return TF.affine(pil, angle=0, translate=list(p['translate']), scale=1.0, shear=[0,0])
        elif kind == 'brightness':
            return TF.adjust_brightness(pil, p['b'])
        elif kind == 'contrast':
            return TF.adjust_contrast(pil, p['c'])
        elif kind == 'saturation':
            return TF.adjust_saturation(pil, p['s'])
        elif kind == 'hue':
            return TF.adjust_hue(pil, p['h'])
        elif kind == 'gamma':
            return TF.adjust_gamma(pil, p['g'])

    kinds = ['rotation', 'scale', 'translation', 'brightness', 'contrast', 'saturation', 'hue', 'gamma']
    labels_id = {
        'rotation'   : f'Rotasi\n({aug_params["angle"]:.1f}°)',
        'scale'      : f'Skala\n({aug_params["scale"]:.2f}×)',
        'translation': f'Translasi\n({aug_params["translate"][0]:.1f}, {aug_params["translate"][1]:.1f}) px',
        'brightness' : f'Brightness\n({aug_params["b"]:.2f})',
        'contrast'   : f'Contrast\n({aug_params["c"]:.2f})',
        'saturation' : f'Saturation\n({aug_params["s"]:.2f})',
        'hue'        : f'Hue\n({aug_params["h"]:.3f})',
        'gamma'      : f'Gamma\n({aug_params["g"]:.2f})',
    }

    ncols = len(kinds) + 1  # +1 untuk original
    fig, axes = plt.subplots(1, ncols, figsize=(2.4 * ncols, 3.8))
    fig.patch.set_facecolor('#1a1a2e')

    # Original
    axes[0].imshow(base_np)
    axes[0].set_title('Original', color='#ffdd57', fontsize=9, fontweight='bold', pad=4)
    axes[0].axis('off')
    for sp in axes[0].spines.values():
        sp.set_edgecolor('#ffdd57'); sp.set_linewidth(2)

    for ax, kind in zip(axes[1:], kinds):
        aug_pil = apply_aug(base_pil, kind)
        ax.imshow(np.array(aug_pil))
        ax.set_title(labels_id[kind], color='#7ec8e3', fontsize=8.5, pad=4)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor('#4a9eff'); sp.set_linewidth(1.2)

    fig.suptitle('Contoh Hasil Augmentasi Data (Frame ke-8, Kata "Saya")',
                 color='white', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(pad=0.5)
    out = os.path.join(OUT_DIR, 'contoh_augmentasi.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')
    print(f'     Parameter acak (seed=42):')
    for k, v in aug_params.items():
        print(f'       {k:10s} = {v}')


# ══════════════════════════════════════════════════════════════════════════════
# 3. GAMBAR: Jitter vs Center sampling (2 baris x 16 frame)
# ══════════════════════════════════════════════════════════════════════════════

def extract_frames_jitter(video_path, num_frames=16, margin=MARGIN, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [np.zeros((224, 224, 3), dtype=np.uint8)] * num_frames
    total     = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    start_f   = int(total * margin)
    end_f     = int(total * (1.0 - margin))
    valid_tot = max(end_f - start_f, 1)
    seg_size  = valid_tot / float(num_frames)
    indices   = []
    for i in range(num_frames):
        s = int(start_f + i * seg_size)
        e = int(start_f + (i + 1) * seg_size)
        idx = np.random.randint(s, e) if e > s else s
        indices.append(min(idx, total - 1))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, IMG_SIZE)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames[:num_frames]


def buat_sampling_comparison():
    center_frames = extract_frames_center(VIDEO_PATH)
    jitter_frames = extract_frames_jitter(VIDEO_PATH)

    fig, axes = plt.subplots(2, 16, figsize=(28, 4.5))
    fig.patch.set_facecolor('#1a1a2e')

    row_labels = ['Center\nSampling', 'Jitter\nSampling']
    row_colors = ['#57c9a0', '#ff9a57']

    for row_idx, (frames, label, color) in enumerate(zip(
            [center_frames, jitter_frames], row_labels, row_colors)):
        for col_idx, (ax, frame) in enumerate(zip(axes[row_idx], frames)):
            ax.imshow(frame)
            ax.set_title(f'{col_idx+1}', color='white', fontsize=7, pad=2)
            ax.axis('off')
            for sp in ax.spines.values():
                sp.set_edgecolor(color); sp.set_linewidth(1.5)

    # Row label di sisi kiri
    for row_idx, (label, color) in enumerate(zip(row_labels, row_colors)):
        fig.text(0.005, 0.75 - row_idx * 0.5, label,
                 va='center', ha='left', color=color,
                 fontsize=10, fontweight='bold', rotation=0)

    fig.suptitle(f'Perbandingan Center Sampling vs Jitter Sampling — Kata "{SAMPLE_WORD.capitalize()}"',
                 color='white', fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout(pad=0.4)
    out = os.path.join(OUT_DIR, 'contoh_sampling_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ══════════════════════════════════════════════════════════════════════════════
# 4. GAMBAR: Normalisasi — sebelum vs sesudah (visualisasi distribusi piksel)
# ══════════════════════════════════════════════════════════════════════════════

def buat_normalisasi():
    frames = extract_frames_center(VIDEO_PATH)
    frame_np = frames[7].astype(np.float32) / 255.0  # [0,1]

    mean = np.array(IMAGENET_MEAN)
    std  = np.array(IMAGENET_STD)
    frame_norm = (frame_np - mean) / std  # normalized

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#1a1a2e')

    for ax in axes:
        ax.set_facecolor('#2a2a3e')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        for sp in ax.spines.values():
            sp.set_edgecolor('#555')

    colors_ch = ['#e74c3c', '#2ecc71', '#3498db']
    ch_names  = ['R (merah)', 'G (hijau)', 'B (biru)']

    for ch, (c, name) in enumerate(zip(colors_ch, ch_names)):
        axes[0].hist(frame_np[:, :, ch].flatten(), bins=60, color=c,
                     alpha=0.65, label=name, density=True)
    axes[0].set_title('Sebelum Normalisasi (nilai piksel [0, 1])',
                      color='white', fontsize=11, pad=8)
    axes[0].set_xlabel('Nilai Piksel'); axes[0].set_ylabel('Densitas')
    axes[0].legend(facecolor='#2a2a3e', labelcolor='white', fontsize=9)
    axes[0].set_xlim(0, 1)

    for ch, (c, name) in enumerate(zip(colors_ch, ch_names)):
        axes[1].hist(frame_norm[:, :, ch].flatten(), bins=60, color=c,
                     alpha=0.65, label=name, density=True)
    axes[1].set_title('Setelah Normalisasi ImageNet\n(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])',
                      color='white', fontsize=11, pad=8)
    axes[1].set_xlabel('Nilai Piksel Ternormalisasi'); axes[1].set_ylabel('Densitas')
    axes[1].legend(facecolor='#2a2a3e', labelcolor='white', fontsize=9)

    fig.suptitle('Distribusi Nilai Piksel: Sebelum vs Sesudah Normalisasi',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=1.2)
    out = os.path.join(OUT_DIR, 'contoh_normalisasi.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f'[OK] {out}')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Video  : {VIDEO_PATH}')
    print(f'Output : {OUT_DIR}\n')

    print('1/4 Membuat contoh 16 frame...')
    buat_16frame()

    print('2/4 Membuat contoh augmentasi...')
    buat_augmentasi()

    print('3/4 Membuat perbandingan sampling...')
    buat_sampling_comparison()

    print('4/4 Membuat visualisasi normalisasi...')
    buat_normalisasi()

    print('\nSelesai. Cek folder: penulisan/gambar/')
