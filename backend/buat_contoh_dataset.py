"""
Generate gambar untuk Bab 4.2:
  1. grid_kelas.png        — 1 frame representatif dari setiap 24 kelas
  2. struktur_folder.png   — visualisasi pohon folder dataset
"""
import cv2, numpy as np, os, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT_DIR, 'penulisan', 'gambar')
os.makedirs(OUT_DIR, exist_ok=True)

DATASET  = r'D:\WLBISINDO_raw\WLBISINDO_raw'
BG       = '#1a1a2e'
random.seed(7)

CLASSES = ['air','bagaimana','belajar','berangkat','cari','datang',
           'dengar','dimana','hijau','kapan','keluarga','kuning',
           'lagi','maaf','makan','mengapa','merah','motor','rumah',
           'saya','siapa','teman','terimakasih','tuli']

LABELS_ID = {
    'air':'Air','bagaimana':'Bagaimana','belajar':'Belajar',
    'berangkat':'Berangkat','cari':'Cari','datang':'Datang',
    'dengar':'Dengar','dimana':'Di Mana','hijau':'Hijau',
    'kapan':'Kapan','keluarga':'Keluarga','kuning':'Kuning',
    'lagi':'Lagi','maaf':'Maaf','makan':'Makan',
    'mengapa':'Mengapa','merah':'Merah','motor':'Motor',
    'rumah':'Rumah','saya':'Saya','siapa':'Siapa',
    'teman':'Teman','terimakasih':'Terima Kasih','tuli':'Tuli',
}

def get_frame(cls, idx=0):
    folder = os.path.join(DATASET, cls)
    vids   = sorted([f for f in os.listdir(folder) if f.endswith('.mp4')])
    path   = os.path.join(folder, vids[idx % len(vids)])
    cap    = cv2.VideoCapture(path)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, f = cap.read(); cap.release()
    if ret:
        f = cv2.resize(f, (224, 224))
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return np.zeros((224, 224, 3), dtype=np.uint8)

def save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'[OK] {p}')

# ── 1. Grid 24 kelas ──────────────────────────────────────────────────────────
def grid_kelas():
    fig, axes = plt.subplots(4, 6, figsize=(16, 11))
    fig.patch.set_facecolor(BG)

    for i, (ax, cls) in enumerate(zip(axes.flat, CLASSES)):
        frame = get_frame(cls)
        ax.imshow(frame)
        ax.set_title(f'{i+1}. {LABELS_ID[cls]}',
                     color='white', fontsize=9, fontweight='bold', pad=4)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor('#4a9eff'); sp.set_linewidth(1.2)
        # label jumlah video
        ax.text(0.98, 0.02, '50 video', transform=ax.transAxes,
                color='#aaaaaa', fontsize=7, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#00000099'))

    fig.suptitle('Contoh Frame Representatif — 24 Kelas Kosakata BISINDO\n(WLBISINDO Dataset, frame tengah per video)',
                 color='white', fontsize=13, fontweight='bold')
    plt.tight_layout(pad=0.6, rect=[0, 0, 1, 0.95])
    save(fig, 'dataset_grid_kelas.png')

# ── 2. Struktur folder ────────────────────────────────────────────────────────
def struktur_folder():
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis('off')

    # tree data: (level, label, note, color)
    tree = [
        (0, 'WLBISINDO_raw/',         '24 folder kelas', '#ffdd57'),
        (1, 'air/',                   '50 video (.mp4)',  '#7ec8e3'),
        (2, 'signer0_label0_sample1.mp4', '',            '#aaaaaa'),
        (2, 'signer0_label0_sample2.mp4', '',            '#aaaaaa'),
        (2, '... (48 video lainnya)', '',                 '#666666'),
        (1, 'bagaimana/',             '50 video (.mp4)',  '#7ec8e3'),
        (2, 'signer0_label1_sample1.mp4', '',            '#aaaaaa'),
        (2, '...',                    '',                 '#666666'),
        (1, 'belajar/',               '50 video (.mp4)',  '#7ec8e3'),
        (2, '...',                    '',                 '#666666'),
        (1, '... (21 folder lainnya)','masing-masing 50 video', '#57c9a0'),
        (1, 'tuli/',                  '50 video (.mp4)',  '#7ec8e3'),
        (2, '...',                    '',                 '#666666'),
    ]

    y      = 0.96
    dy     = 0.068
    indent = 0.06

    for level, label, note, color in tree:
        x = 0.04 + level * indent
        # connector lines
        if level > 0:
            ax.plot([x - indent + 0.015, x - 0.005], [y + dy*0.4, y + dy*0.4],
                    color='#444', lw=1.2)
        # icon
        icon = '📁' if label.endswith('/') else '🎬' if label.endswith('.mp4') else '•'
        ax.text(x, y, f'{icon}  {label}', transform=ax.transAxes,
                color=color, fontsize=10 if level < 2 else 9,
                fontweight='bold' if level == 0 else 'normal',
                va='top', fontfamily='monospace')
        if note:
            ax.text(x + 0.38, y, f'← {note}', transform=ax.transAxes,
                    color='#888888', fontsize=8.5, va='top', style='italic')
        y -= dy

    # stats box
    stats = (
        'Statistik Dataset\n'
        '─────────────────\n'
        'Total kelas    : 24\n'
        'Video per kelas: 50\n'
        'Total video    : 1.200\n'
        'Signer         : 5 orang\n'
        'Format         : .mp4\n'
        'Resolusi asli  : 1280×720 px'
    )
    ax.text(0.62, 0.92, stats, transform=ax.transAxes,
            color='white', fontsize=10, va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6',
                      facecolor='#2a2a3e', edgecolor='#4a9eff', lw=1.5))

    ax.set_title('Struktur Folder Dataset WLBISINDO',
                 color='white', fontsize=13, fontweight='bold', pad=14)
    save(fig, 'dataset_struktur_folder.png')

if __name__ == '__main__':
    print('1/2 Grid 24 kelas...')
    grid_kelas()
    print('2/2 Struktur folder...')
    struktur_folder()
    print('\nSelesai.')
