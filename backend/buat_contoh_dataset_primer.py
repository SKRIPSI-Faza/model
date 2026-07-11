"""
Generate gambar & statistik DATASET PRIMER (data_custom, hasil rekam_kata.py):
  1. grid_kelas_primer.png          — 1 frame representatif dari setiap 24 kelas
  2. statistik_dataset_primer.txt   — tabel statistik per kelas (utk tabel BAB III/IV)
"""
import cv2, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT_DIR, 'penulisan', 'gambar')
os.makedirs(OUT_DIR, exist_ok=True)

DATASET = os.path.join(ROOT_DIR, 'data_custom')

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
VIDEO_EXT = ('.mp4', '.avi', '.mov', '.mkv')


def clip_info(path):
    cap = cv2.VideoCapture(path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = n / fps if fps > 0 else 0.0
    return w, h, fps, n, dur


def mid_frame(path):
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, f = cap.read(); cap.release()
    if ret:
        return cv2.cvtColor(cv2.resize(f, (224, 224)), cv2.COLOR_BGR2RGB)
    return np.zeros((224, 224, 3), dtype=np.uint8)


classes = sorted(c for c in os.listdir(DATASET)
                 if os.path.isdir(os.path.join(DATASET, c)))

# ── 1. Grid 4x6: satu frame tengah per kelas ─────────────────────────────────
fig, axes = plt.subplots(4, 6, figsize=(15, 11))
fig.suptitle('Contoh Frame Dataset Primer (24 Kelas Kosakata BISINDO)',
             fontsize=15, fontweight='bold')
for ax, cls in zip(axes.flat, classes):
    cdir = os.path.join(DATASET, cls)
    vids = sorted(v for v in os.listdir(cdir) if v.lower().endswith(VIDEO_EXT))
    ax.imshow(mid_frame(os.path.join(cdir, vids[0])))
    ax.set_title(LABELS_ID.get(cls, cls), fontsize=11, fontweight='bold')
    ax.axis('off')
plt.tight_layout(rect=[0, 0, 1, 0.965])
grid_path = os.path.join(OUT_DIR, 'grid_kelas_primer.png')
plt.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'[OK] {grid_path}')

# ── 2. Statistik per kelas ───────────────────────────────────────────────────
rows, total_clips, total_dur = [], 0, 0.0
res_set, fps_set = set(), set()
for cls in classes:
    cdir = os.path.join(DATASET, cls)
    vids = sorted(v for v in os.listdir(cdir) if v.lower().endswith(VIDEO_EXT))
    durs = []
    for v in vids:
        w, h, fps, n, dur = clip_info(os.path.join(cdir, v))
        durs.append(dur)
        res_set.add(f'{w}x{h}')
        fps_set.add(round(fps))
    total_clips += len(vids); total_dur += sum(durs)
    rows.append((LABELS_ID.get(cls, cls), len(vids),
                 min(durs), max(durs), sum(durs) / len(durs)))

lines = []
lines.append('STATISTIK DATASET PRIMER (data_custom)')
lines.append('=' * 62)
lines.append(f'{"Kelas":<16}{"Jumlah Klip":>12}{"Durasi Min":>11}'
             f'{"Durasi Max":>11}{"Rata-rata":>11}')
lines.append('-' * 62)
for name, n, dmin, dmax, davg in rows:
    lines.append(f'{name:<16}{n:>12}{dmin:>10.1f}s{dmax:>10.1f}s{davg:>10.1f}s')
lines.append('-' * 62)
lines.append(f'{"TOTAL":<16}{total_clips:>12}{"":>11}{"":>11}{total_dur:>9.1f}s')
lines.append('')
lines.append(f'Resolusi   : {", ".join(sorted(res_set))}')
lines.append(f'Frame rate : {", ".join(str(f) for f in sorted(fps_set))} fps')
lines.append(f'Jumlah kelas: {len(classes)} | Total durasi: {total_dur/60:.1f} menit')

txt_path = os.path.join(OUT_DIR, 'statistik_dataset_primer.txt')
with open(txt_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f'[OK] {txt_path}')
print()
print('\n'.join(lines))
