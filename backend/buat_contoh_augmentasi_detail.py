"""
Visualisasi hasil augmentasi per jenis sesuai Tabel 3.7.
Output: penulisan/gambar/
  - augmentasi_spasial.png     (original + rotasi + skala + translasi)
  - augmentasi_fotometrik.png  (original + brightness + contrast + saturation + hue + gamma)
  - augmentasi_all.png         (rangkuman semua 8 jenis dalam 1 gambar)
"""
import cv2, numpy as np, os, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms.functional as TF
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT_DIR, 'penulisan', 'gambar')
os.makedirs(OUT_DIR, exist_ok=True)

VIDEO_PATH = r'D:\WLBISINDO_raw\WLBISINDO_raw\saya\signer0_label9_sample1.mp4'
IMG_SIZE   = (224, 224)
BG         = '#1a1a2e'
random.seed(99)
np.random.seed(99)

# ── ambil frame tengah video ──────────────────────────────────────────────────
def get_frame():
    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, f = cap.read(); cap.release()
    f = cv2.resize(f, IMG_SIZE)
    return Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

def pil2np(p): return np.array(p)

def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[OK] {p}')

def style_ax(ax, title, color, border=1.8):
    ax.set_title(title, color=color, fontsize=9.5, pad=5, fontweight='bold')
    ax.axis('off')
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(border)

# ── nilai demo (di dalam rentang tabel, dipilih agar kontras terlihat) ────────
DEMO = {
    'rotasi'     : 7.0,          # maks ±7°
    'skala_kecil': 0.88,         # min skala
    'skala_besar': 1.12,         # maks skala
    'trans_x'    : 0.07*224,     # +7% ke kanan
    'trans_y'    : -0.07*224,    # -7% ke atas
    'bright_gelap': 0.6,         # gelap
    'bright_terang': 1.4,        # terang
    'contrast_rendah': 0.6,
    'contrast_tinggi': 1.4,
    'sat_rendah' : 0.8,
    'sat_tinggi' : 1.2,
    'hue_neg'    : -0.08,
    'hue_pos'    : 0.08,
    'gamma_gelap': 0.7,
    'gamma_terang': 1.5,
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. AUGMENTASI SPASIAL
# ══════════════════════════════════════════════════════════════════════════════
def spasial():
    orig = get_frame()
    rot_neg  = TF.affine(orig, angle=-7.0, translate=[0,0], scale=1.0, shear=[0,0])
    rot_pos  = TF.affine(orig, angle=+7.0, translate=[0,0], scale=1.0, shear=[0,0])
    sk_kecil = TF.affine(orig, angle=0,    translate=[0,0], scale=0.88, shear=[0,0])
    sk_besar = TF.affine(orig, angle=0,    translate=[0,0], scale=1.12, shear=[0,0])
    tr_kanan = TF.affine(orig, angle=0, translate=[int(0.07*224),0],  scale=1.0, shear=[0,0])
    tr_atas  = TF.affine(orig, angle=0, translate=[0,int(-0.07*224)], scale=1.0, shear=[0,0])

    # 2 baris x 4 kolom (7 gambar + 1 kosong di pojok kanan bawah)
    items = [
        (orig,     'Original',               '#ffffff'),
        (rot_neg,  'Rotasi -7°',             '#ff9a57'),
        (rot_pos,  'Rotasi +7°',             '#ff9a57'),
        (sk_kecil, 'Skala 0.88x (zoom out)', '#57c9a0'),
        (sk_besar, 'Skala 1.12x (zoom in)',  '#57c9a0'),
        (tr_kanan, 'Translasi +7% (kanan)',  '#7ec8e3'),
        (tr_atas,  'Translasi -7% (atas)',   '#7ec8e3'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(13, 7))
    fig.patch.set_facecolor(BG)
    for idx, (ax, (img, title, col)) in enumerate(zip(axes.flat, items)):
        ax.imshow(pil2np(img))
        style_ax(ax, title, col)
    axes.flat[-1].axis('off')   # panel kosong

    patches = [
        mpatches.Patch(color='#ff9a57', label='Rotasi (±7°)'),
        mpatches.Patch(color='#57c9a0', label='Skala (0.88-1.12)'),
        mpatches.Patch(color='#7ec8e3', label='Translasi (±7%)'),
    ]
    fig.legend(handles=patches, loc='lower right', ncol=1,
               facecolor='#2a2a3e', labelcolor='white', fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.99, 0.08))
    fig.suptitle('Augmentasi Spasial — Rotasi, Skala, Translasi\n(Kata: "Saya")',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=0.8)
    save(fig, 'augmentasi_spasial.png')


# ══════════════════════════════════════════════════════════════════════════════
# 2. AUGMENTASI FOTOMETRIK
# ══════════════════════════════════════════════════════════════════════════════
def fotometrik():
    orig = get_frame()

    items = [
        (orig,                             'Original',                '#ffffff'),
        (TF.adjust_brightness(orig, 0.6),  'Brightness 0.6 (gelap)',  '#ffdd57'),
        (TF.adjust_brightness(orig, 1.4),  'Brightness 1.4 (terang)', '#ffdd57'),
        (TF.adjust_contrast(orig, 0.6),    'Contrast 0.6 (rendah)',   '#ff9a57'),
        (TF.adjust_contrast(orig, 1.4),    'Contrast 1.4 (tinggi)',   '#ff9a57'),
        (TF.adjust_saturation(orig, 0.8),  'Saturation 0.8 (pudar)',  '#57c9a0'),
        (TF.adjust_saturation(orig, 1.2),  'Saturation 1.2 (jenuh)',  '#57c9a0'),
        (TF.adjust_hue(orig, -0.08),       'Hue -0.08',               '#a8b8ff'),
        (TF.adjust_hue(orig, +0.08),       'Hue +0.08',               '#a8b8ff'),
        (TF.adjust_gamma(orig, 0.7),       'Gamma 0.7 (cerah)',       '#ff7eb3'),
        (TF.adjust_gamma(orig, 1.5),       'Gamma 1.5 (gelap)',       '#ff7eb3'),
    ]

    # 3 baris x 4 kolom = 12 slot, 11 gambar + 1 kosong
    fig, axes = plt.subplots(3, 4, figsize=(13, 10.5))
    fig.patch.set_facecolor(BG)
    for ax, (img, title, col) in zip(axes.flat, items):
        ax.imshow(pil2np(img))
        style_ax(ax, title, col)
    axes.flat[-1].axis('off')

    patches = [
        mpatches.Patch(color='#ffdd57', label='Brightness (0.6-1.4)'),
        mpatches.Patch(color='#ff9a57', label='Contrast (0.6-1.4)'),
        mpatches.Patch(color='#57c9a0', label='Saturation (0.8-1.2)'),
        mpatches.Patch(color='#a8b8ff', label='Hue (+-0.08)'),
        mpatches.Patch(color='#ff7eb3', label='Gamma (0.7-1.5)'),
    ]
    fig.legend(handles=patches, loc='lower right', ncol=1,
               facecolor='#2a2a3e', labelcolor='white', fontsize=10,
               framealpha=0.9, bbox_to_anchor=(0.99, 0.06))
    fig.suptitle('Augmentasi Fotometrik — Brightness, Contrast, Saturation, Hue, Gamma\n(Kata: "Saya")',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=0.8)
    save(fig, 'augmentasi_fotometrik.png')


# ══════════════════════════════════════════════════════════════════════════════
# 3. RANGKUMAN — 1 nilai representatif per jenis (9 panel: original + 8)
# ══════════════════════════════════════════════════════════════════════════════
def rangkuman():
    orig = get_frame()
    items = [
        (orig,                                    'Original',       '#ffffff',  '—'),
        (TF.affine(orig,angle=6,translate=[0,0],scale=1.0,shear=[0,0]),
                                                  'Rotasi',         '#ff9a57',  '+6°'),
        (TF.affine(orig,angle=0,translate=[0,0],scale=0.90,shear=[0,0]),
                                                  'Skala',          '#57c9a0',  '0.90×'),
        (TF.affine(orig,angle=0,translate=[12,10],scale=1.0,shear=[0,0]),
                                                  'Translasi',      '#7ec8e3',  '+12, +10 px'),
        (TF.adjust_brightness(orig, 0.65),        'Brightness',     '#ffdd57',  '0.65'),
        (TF.adjust_contrast(orig, 1.35),          'Contrast',       '#ff9a57',  '1.35'),
        (TF.adjust_saturation(orig, 0.82),        'Saturation',     '#57c9a0',  '0.82'),
        (TF.adjust_hue(orig, -0.07),              'Hue',            '#a8b8ff',  '-0.07'),
        (TF.adjust_gamma(orig, 1.45),             'Gamma',          '#ff7eb3',  '1.45'),
    ]

    # 3 baris x 3 kolom = 9 panel pas
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor(BG)
    for ax, (img, title, col, val) in zip(axes.flat, items):
        ax.imshow(pil2np(img))
        ax.set_title(f'{title}\n({val})', color=col, fontsize=10,
                     pad=5, fontweight='bold')
        ax.axis('off')
        border = 2.5 if title == 'Original' else 1.5
        for sp in ax.spines.values():
            sp.set_edgecolor(col); sp.set_linewidth(border)

    # label kelompok di sisi kiri
    fig.text(0.01, 0.83, 'SPASIAL', va='center', ha='left',
             color='#aaaaaa', fontsize=9, style='italic', rotation=90)
    fig.text(0.01, 0.50, 'FOTOMETRIK', va='center', ha='left',
             color='#aaaaaa', fontsize=9, style='italic', rotation=90)

    fig.suptitle('Contoh Hasil Augmentasi Data — 8 Jenis Transformasi\n(Kata: "Saya")',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=0.8)
    save(fig, 'augmentasi_rangkuman.png')


if __name__ == '__main__':
    print('1/3 Augmentasi Spasial...')
    spasial()
    print('2/3 Augmentasi Fotometrik...')
    fotometrik()
    print('3/3 Rangkuman semua jenis...')
    rangkuman()
    print('\nSelesai.')
