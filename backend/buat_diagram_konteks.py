"""
Diagram Konteks (DFD Level 0) SnapSign BISINDO.
Menghasilkan: penulisan/gambar/diagram_konteks.png
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

os.makedirs(r'd:\v10_lighting\penulisan\gambar', exist_ok=True)
os.chdir(r'd:\v10_lighting\penulisan\gambar')
plt.rcParams['font.family'] = 'DejaVu Sans'

C = dict(
    sys='#AED6F1', ext='#D5E8D4', edge='#2C3E50',
    txt='#17202A', arr='#2C3E50', flow='#922B21',
)


def entity_box(ax, cx, cy, w, h, label, sublabel=''):
    ax.add_patch(mpatches.Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        fc=C['ext'], ec=C['edge'], lw=2.0, zorder=4))
    ax.text(cx, cy + (0.18 if sublabel else 0), label,
            ha='center', va='center', fontsize=11,
            fontweight='bold', color=C['txt'], zorder=6)
    if sublabel:
        ax.text(cx, cy - 0.28, sublabel, ha='center', va='center',
                fontsize=8.5, color='#555', zorder=6)


def system_circle(ax, cx, cy, r, label, sublabel=''):
    circ = plt.Circle((cx, cy), r, fc=C['sys'], ec=C['edge'], lw=2.5, zorder=4)
    ax.add_patch(circ)
    ax.text(cx, cy + (0.2 if sublabel else 0), label,
            ha='center', va='center', fontsize=12,
            fontweight='bold', color=C['txt'], zorder=6)
    if sublabel:
        ax.text(cx, cy - 0.35, sublabel, ha='center', va='center',
                fontsize=9, color='#444', style='italic', zorder=6)


def flow_arrow(ax, x1, y1, x2, y2, label, label_side='top', color=C['flow']):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arr'],
                                lw=1.8, mutation_scale=18), zorder=5)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dy = 0.22 if label_side == 'top' else -0.22
    dx = 0.0
    if abs(x2 - x1) < 0.1:
        dx = 0.5 if label_side == 'right' else -0.5
        dy = 0.0
    ax.text(mx + dx, my + dy, label, ha='center', va='center',
            fontsize=8.5, color=color, fontweight='bold',
            bbox=dict(fc='white', ec='none', pad=1.5), zorder=7)


fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis('off')
fig.patch.set_facecolor('white')

ax.text(6.5, 6.65, 'Diagram Konteks Sistem SnapSign BISINDO',
        ha='center', fontsize=14, fontweight='bold', color=C['txt'])
ax.text(6.5, 6.3, 'DFD Level 0',
        ha='center', fontsize=10, color='#666', style='italic')

# ── Entitas eksternal ──────────────────────────────────────────────────────
entity_box(ax, 1.5, 3.5, 2.2, 1.2, 'Pengguna', '(User)')
entity_box(ax, 11.5, 3.5, 2.2, 1.2, 'Server Model', '(Flask / CPU)')

# ── Sistem (lingkaran tengah) ──────────────────────────────────────────────
system_circle(ax, 6.5, 3.5, 1.85, 'Sistem SnapSign', 'Aplikasi Mobile Flutter')

# ── Aliran data: Pengguna → Sistem ────────────────────────────────────────
flow_arrow(ax, 2.62, 3.85, 4.65, 3.85,
           'Gestur Isyarat (video kamera)', label_side='top')

# ── Aliran data: Sistem → Pengguna ────────────────────────────────────────
flow_arrow(ax, 4.65, 3.15, 2.62, 3.15,
           'Label Kosakata + Confidence', label_side='top')

# ── Aliran data: Sistem → Server ──────────────────────────────────────────
flow_arrow(ax, 8.35, 3.85, 10.38, 3.85,
           'Klip Frame JPEG (base64, HTTP POST)', label_side='top')

# ── Aliran data: Server → Sistem ──────────────────────────────────────────
flow_arrow(ax, 10.38, 3.15, 8.35, 3.15,
           'Hasil Prediksi JSON\n(label, confidence, top-3)', label_side='top')

# ── Aliran data tambahan: Pengguna → Sistem (lokal) ───────────────────────
flow_arrow(ax, 2.62, 4.1, 4.65, 4.6,
           'Buka Kamus / Panduan / Pengaturan', label_side='top')
flow_arrow(ax, 4.65, 4.4, 2.62, 3.9,
           'Konten Lokal (GIF, teks)', label_side='top')

# ── Legenda ───────────────────────────────────────────────────────────────
leg_x, leg_y = 0.4, 1.4
ax.add_patch(mpatches.Rectangle((leg_x, leg_y - 0.7), 3.8, 1.2,
             fc='#fafafa', ec='#ccc', lw=1.0, zorder=3))
ax.text(leg_x + 1.9, leg_y + 0.28, 'Keterangan', ha='center',
        fontsize=9, fontweight='bold', color='#333', zorder=5)
ax.add_patch(mpatches.Rectangle((leg_x + 0.15, leg_y - 0.18), 0.5, 0.3,
             fc=C['ext'], ec=C['edge'], lw=1.3, zorder=4))
ax.text(leg_x + 0.78, leg_y - 0.03, 'Entitas Eksternal',
        fontsize=8.5, va='center', color='#333', zorder=5)
ax.add_patch(plt.Circle((leg_x + 0.4, leg_y - 0.5), 0.18,
             fc=C['sys'], ec=C['edge'], lw=1.3, zorder=4))
ax.text(leg_x + 0.78, leg_y - 0.5, 'Sistem (proses utama)',
        fontsize=8.5, va='center', color='#333', zorder=5)

plt.tight_layout(pad=0.5)
fig.savefig('diagram_konteks.png', dpi=160, bbox_inches='tight')
plt.close(fig)
print('[OK] diagram_konteks.png -> d:\\v10_lighting\\penulisan\\gambar\\')
