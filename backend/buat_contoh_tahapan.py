"""
Visualisasi hasil setiap tahapan preprocessing untuk Bab 4.
Menghasilkan 5 gambar (1 per tahapan) + 1 gambar rangkuman pipeline.
Output -> penulisan/gambar/
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── config ────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR     = os.path.join(ROOT_DIR, 'penulisan', 'gambar')
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_DIR = r'D:\WLBISINDO_raw\WLBISINDO_raw'
WORD        = 'saya'
FILE        = 'signer0_label9_sample1.mp4'
VIDEO_PATH  = os.path.join(DATASET_DIR, WORD, FILE)

MARGIN      = 0.15
NUM_FRAMES  = 16
IMG_SIZE    = (224, 224)
BLUR_KERNEL = 41
MASK_DILATE = 9
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

BG_DARK = '#1a1a2e'

# ── helpers ───────────────────────────────────────────────────────────────────

def read_raw_frame(video_path, frame_idx=None):
    """Baca satu frame mentah dari video (BGR, ukuran asli)."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = total // 2 if frame_idx is None else frame_idx
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def extract_16frames_center(video_path):
    cap = cv2.VideoCapture(video_path)
    total     = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    start_f   = int(total * MARGIN)
    end_f     = int(total * (1.0 - MARGIN))
    valid_tot = max(end_f - start_f, 1)
    seg_size  = valid_tot / float(NUM_FRAMES)
    indices   = [min(int(start_f + i*seg_size) + (int((i+1)*seg_size - i*seg_size))//2,
                     total-1) for i in range(NUM_FRAMES)]
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if ret
                      else (frames[-1] if frames else np.zeros((480,640,3),dtype=np.uint8)))
    cap.release()
    return frames


def do_segmentation(bgr_frame):
    seg_mp = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)
    k      = np.ones((MASK_DILATE, MASK_DILATE), np.uint8)
    rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mask   = seg_mp.process(rgb).segmentation_mask
    binm   = cv2.dilate((mask > 0.3).astype(np.uint8), k, iterations=1)
    mask   = np.clip(np.maximum(mask, binm.astype(np.float32)*0.8), 0.0, 1.0)
    bg     = cv2.GaussianBlur(bgr_frame, (BLUR_KERNEL, BLUR_KERNEL), 0)
    m3     = mask[:,:,None]
    out    = (bgr_frame.astype(np.float32)*m3 + bg.astype(np.float32)*(1-m3)).astype(np.uint8)
    seg_mp.close()
    return out, (mask*255).astype(np.uint8)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[OK] {path}')


# ══════════════════════════════════════════════════════════════════════════════
# TAHAP 1 — Segmentasi Background
# ══════════════════════════════════════════════════════════════════════════════
def tahap1_segmentasi():
    bgr = read_raw_frame(VIDEO_PATH)
    seg, mask_gray = do_segmentation(bgr)

    orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    seg_rgb  = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    blur_rgb = cv2.cvtColor(cv2.GaussianBlur(bgr,(BLUR_KERNEL,BLUR_KERNEL),0),cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor(BG_DARK)

    imgs   = [orig_rgb, mask_gray, blur_rgb, seg_rgb]
    titles = ['(a) Frame Asli (BGR→RGB)',
              '(b) Masker Segmentasi\n(MediaPipe Selfie Seg.)',
              f'(c) Background Setelah\nGaussian Blur ({BLUR_KERNEL}×{BLUR_KERNEL})',
              '(d) Hasil Segmentasi\n(Alpha Compositing)']
    cmaps  = [None, 'gray', None, None]
    colors = ['#ffdd57','#a8e6cf','#7ec8e3','#ff9a57']

    for ax, img, title, cmap, col in zip(axes, imgs, titles, cmaps, colors):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=col, fontsize=10, pad=6)
        ax.axis('off')
        h, w = img.shape[:2]
        ax.text(0.01, 0.01, f'{w}×{h}px', transform=ax.transAxes,
                color='white', fontsize=8, va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#00000088'))
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(1.5)

    fig.suptitle('Tahap 1 — Segmentasi Background (Kata: "Saya")',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=0.8)
    save(fig, 'tahap1_segmentasi.png')


# ══════════════════════════════════════════════════════════════════════════════
# TAHAP 2 — Ekstraksi 16 Frame
# ══════════════════════════════════════════════════════════════════════════════
def tahap2_ekstraksi():
    frames = extract_16frames_center(VIDEO_PATH)
    cap    = cv2.VideoCapture(VIDEO_PATH)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    w_vid  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.5))
    fig.patch.set_facecolor(BG_DARK)

    for i, (ax, frame) in enumerate(zip(axes.flat, frames)):
        ax.imshow(frame)
        ax.set_title(f'Frame {i+1}', color='white', fontsize=8, pad=3)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor('#4a9eff'); sp.set_linewidth(1.2)

    info = (f'Video asli: {w_vid}×{h_vid}px | {total} frame | {fps:.0f} fps  ·  '
            f'Margin = {MARGIN*100:.0f}%  ·  Metode: Center Sampling')
    fig.text(0.5, 0.01, info, ha='center', color='#aaaaaa', fontsize=9)
    fig.suptitle('Tahap 2 — Ekstraksi 16 Frame dari Video (Center Sampling, margin=15%)',
                 color='white', fontsize=12, fontweight='bold')
    plt.tight_layout(pad=0.5, rect=[0,0.04,1,1])
    save(fig, 'tahap2_ekstraksi.png')


# ══════════════════════════════════════════════════════════════════════════════
# TAHAP 3 — Resize 224×224
# ══════════════════════════════════════════════════════════════════════════════
def tahap3_resize():
    bgr   = read_raw_frame(VIDEO_PATH)
    orig  = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.cvtColor(cv2.resize(bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor(BG_DARK)

    data = [(orig,    f'(a) Sebelum Resize\n({orig.shape[1]}×{orig.shape[0]} px)', '#ffdd57'),
            (resized, f'(b) Setelah Resize\n({IMG_SIZE[0]}×{IMG_SIZE[1]} px)',     '#57c9a0')]

    for ax, (img, title, col) in zip(axes, data):
        ax.imshow(img)
        ax.set_title(title, color=col, fontsize=12, pad=8)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(2)
        # grid overlay di gambar resize
        if img.shape[0] == 224:
            step = 224//4
            for v in range(0, 224, step):
                ax.axvline(v, color='white', alpha=0.15, lw=0.5)
                ax.axhline(v, color='white', alpha=0.15, lw=0.5)

    fig.suptitle('Tahap 3 — Resize ke 224×224 Piksel',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=1.0)
    save(fig, 'tahap3_resize.png')


# ══════════════════════════════════════════════════════════════════════════════
# TAHAP 4 — Konversi BGR → RGB
# ══════════════════════════════════════════════════════════════════════════════
def tahap4_colorspace():
    bgr     = cv2.resize(read_raw_frame(VIDEO_PATH), IMG_SIZE)
    rgb     = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # split channels
    b_ch, g_ch, r_ch = cv2.split(bgr)   # OpenCV BGR order
    r_ch2, g_ch2, b_ch2 = cv2.split(rgb)  # correct RGB order

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(BG_DARK)
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.25)

    # Row 0: BGR
    ax_bgr = fig.add_subplot(gs[0, 0])
    ax_bgr.imshow(rgb)   # shown as RGB so colours look "wrong" → intentional to show swap
    ax_bgr.set_title('Frame BGR\n(tampil salah → B↔R swap)', color='#ffdd57', fontsize=9)
    ax_bgr.axis('off')
    for sp in ax_bgr.spines.values(): sp.set_edgecolor('#ffdd57'); sp.set_linewidth(1.5)

    ch_titles_bgr = ['Channel B (tampil\nsebagai merah)', 'Channel G', 'Channel R (tampil\nsebagai biru)']
    ch_data_bgr   = [b_ch, g_ch, r_ch]
    ch_colors_bgr = ['#5b9bd5','#57c9a0','#e74c3c']
    for col, (title, data, col_c) in enumerate(zip(ch_titles_bgr, ch_data_bgr, ch_colors_bgr)):
        ax = fig.add_subplot(gs[0, col+1])
        ax.imshow(data, cmap='gray')
        ax.set_title(title, color=col_c, fontsize=8.5)
        ax.axis('off')

    # Row 1: RGB
    ax_rgb = fig.add_subplot(gs[1, 0])
    ax_rgb.imshow(rgb)
    ax_rgb.set_title('Frame RGB\n(warna benar)', color='#57c9a0', fontsize=9)
    ax_rgb.axis('off')
    for sp in ax_rgb.spines.values(): sp.set_edgecolor('#57c9a0'); sp.set_linewidth(1.5)

    ch_titles_rgb = ['Channel R\n(merah)', 'Channel G\n(hijau)', 'Channel B\n(biru)']
    ch_data_rgb   = [r_ch2, g_ch2, b_ch2]
    ch_colors_rgb = ['#e74c3c','#57c9a0','#5b9bd5']
    for col, (title, data, col_c) in enumerate(zip(ch_titles_rgb, ch_data_rgb, ch_colors_rgb)):
        ax = fig.add_subplot(gs[1, col+1])
        ax.imshow(data, cmap='gray')
        ax.set_title(title, color=col_c, fontsize=8.5)
        ax.axis('off')

    # arrow
    fig.text(0.5, 0.52, '▼  cv2.cvtColor(BGR → RGB)', ha='center',
             color='white', fontsize=10, fontweight='bold')

    fig.suptitle('Tahap 4 — Konversi Ruang Warna BGR → RGB',
                 color='white', fontsize=13, fontweight='bold')
    save(fig, 'tahap4_colorspace.png')


# ══════════════════════════════════════════════════════════════════════════════
# TAHAP 5 — Normalisasi ImageNet
# ══════════════════════════════════════════════════════════════════════════════
def tahap5_normalisasi():
    bgr   = cv2.resize(read_raw_frame(VIDEO_PATH), IMG_SIZE)
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    norm  = (rgb - IMAGENET_MEAN) / IMAGENET_STD

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(BG_DARK)
    for ax in axes:
        ax.set_facecolor('#2a2a3e')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#555')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')

    ch_names = ['R', 'G', 'B']
    colors   = ['#e74c3c','#2ecc71','#3498db']

    # histogram sebelum
    for ch, (name, col) in enumerate(zip(ch_names, colors)):
        axes[0].hist(rgb[:,:,ch].flatten(), bins=60, color=col,
                     alpha=0.7, label=name, density=True)
    axes[0].set_title('Sebelum Normalisasi\n(nilai [0, 1])', color='#ffdd57', fontsize=11)
    axes[0].set_xlabel('Nilai Piksel'); axes[0].set_ylabel('Densitas')
    axes[0].legend(facecolor='#2a2a3e', labelcolor='white')
    axes[0].set_xlim(0, 1)

    # histogram sesudah
    for ch, (name, col) in enumerate(zip(ch_names, colors)):
        axes[1].hist(norm[:,:,ch].flatten(), bins=60, color=col,
                     alpha=0.7, label=name, density=True)
    axes[1].set_title('Setelah Normalisasi ImageNet\nmean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]',
                      color='#57c9a0', fontsize=10)
    axes[1].set_xlabel('Nilai Piksel Ternormalisasi'); axes[1].set_ylabel('Densitas')
    axes[1].legend(facecolor='#2a2a3e', labelcolor='white')

    # statistik tabel
    axes[2].axis('off')
    axes[2].set_facecolor('#2a2a3e')
    rows = [['', 'Min', 'Max', 'Mean', 'Std']]
    for ch, name in enumerate(ch_names):
        before = rgb[:,:,ch].flatten()
        after  = norm[:,:,ch].flatten()
        rows.append([f'Sebelum {name}',
                     f'{before.min():.3f}', f'{before.max():.3f}',
                     f'{before.mean():.3f}', f'{before.std():.3f}'])
        rows.append([f'Sesudah {name}',
                     f'{after.min():.3f}', f'{after.max():.3f}',
                     f'{after.mean():.3f}', f'{after.std():.3f}'])
    tbl = axes[2].table(cellText=rows[1:], colLabels=rows[0],
                        loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a2e' if r % 2 == 0 else '#2a2a3e')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#444')
    axes[2].set_title('Statistik Nilai Piksel per Channel',
                      color='white', fontsize=11, pad=20)

    fig.suptitle('Tahap 5 — Normalisasi Nilai Piksel (ImageNet Normalization)',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=1.0)
    save(fig, 'tahap5_normalisasi.png')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'Video  : {VIDEO_PATH}')
    print(f'Output : {OUT_DIR}\n')
    print('Tahap 1 — Segmentasi Background (MediaPipe)...')
    tahap1_segmentasi()
    print('Tahap 2 — Ekstraksi 16 Frame...')
    tahap2_ekstraksi()
    print('Tahap 3 — Resize 224×224...')
    tahap3_resize()
    print('Tahap 4 - Konversi BGR ke RGB...')
    tahap4_colorspace()
    print('Tahap 5 - Normalisasi ImageNet...')
    tahap5_normalisasi()
    print('\nSelesai.')
