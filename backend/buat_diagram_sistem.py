"""
Diagram Arsitektur Sistem & Mekanisme Deteksi SnapSign BISINDO.
Menghasilkan 2 PNG untuk Bab 3:
  - arsitektur_sistem.png   : pola client (Flutter) - server (Flask API)
  - mekanisme_deteksi.png   : alur deteksi aplikasi <-> API (swimlane)
Hanya menu Deteksi yang memakai API; Kamus/Panduan/Pengaturan berjalan lokal.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon

os.makedirs(r'd:\v10_lighting\penulisan\gambar', exist_ok=True)
os.chdir(r'd:\v10_lighting\penulisan\gambar')
plt.rcParams['font.family'] = 'DejaVu Sans'

C = dict(
    app='#AED6F1', srv='#A9DFBF', local='#E5E7E9', io='#FAD7A0',
    dec='#F9E79F', term='#D5D8DC', edge='#2C3E50', txt='#17202A', arr='#2C3E50',
)


def box(ax, cx, cy, w, h, lbl, fc, fs=9, bold=False, rounded=True):
    if rounded:
        p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                           boxstyle="round,pad=0.02,rounding_size=0.12",
                           fc=fc, ec=C['edge'], lw=1.3, zorder=4)
    else:
        p = mpatches.Rectangle((cx - w / 2, cy - h / 2), w, h,
                               fc=fc, ec=C['edge'], lw=1.3, zorder=4)
    ax.add_patch(p)
    ax.text(cx, cy, lbl, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=C['txt'],
            multialignment='center', zorder=6)


def diamond(ax, cx, cy, w, h, lbl, fs=8):
    vx = [cx, cx + w / 2, cx, cx - w / 2]
    vy = [cy + h / 2, cy, cy - h / 2, cy]
    ax.add_patch(Polygon(list(zip(vx, vy)), closed=True,
                         fc=C['dec'], ec=C['edge'], lw=1.3, zorder=4))
    ax.text(cx, cy, lbl, ha='center', va='center', fontsize=fs, zorder=6)


def container(ax, x, y, w, h, title, fc):
    ax.add_patch(mpatches.Rectangle((x, y), w, h, fc=fc, ec=C['edge'],
                                    lw=1.6, zorder=2, alpha=0.35))
    ax.text(x + w / 2, y + h - 0.28, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color=C['txt'], zorder=6)


def arrow(ax, x1, y1, x2, y2, lbl='', dashed=False, fs=8, lpos='mid'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C['arr'], lw=1.4,
                                mutation_scale=14,
                                linestyle='--' if dashed else '-'), zorder=5)
    if lbl:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + 0.15, my, lbl, fontsize=fs, color='#922B21',
                ha='left', va='center', zorder=7)


# ══════════════════════════════════════════════════════════════════════════
# 1. ARSITEKTUR SISTEM
# ══════════════════════════════════════════════════════════════════════════
def make_arsitektur():
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(5, 11.6, 'Arsitektur Sistem SnapSign (Client - Server)',
            ha='center', fontsize=13, fontweight='bold')

    # Pengguna
    ax.add_patch(Ellipse((5, 10.7), 2.4, 0.7, fc=C['term'], ec=C['edge'], lw=1.5, zorder=4))
    ax.text(5, 10.7, 'Pengguna', ha='center', va='center', fontsize=10, fontweight='bold', zorder=6)

    # Aplikasi Mobile
    container(ax, 0.8, 7.6, 8.4, 2.4, 'APLIKASI MOBILE (Flutter)', C['app'])
    box(ax, 5, 9.1, 7.4, 0.7, 'Menu Lokal (tanpa API):\nKamus, Panduan, Pengaturan', C['local'], fs=9)
    box(ax, 5, 8.1, 5.0, 0.7, 'Menu Deteksi Kosakata\n(memerlukan API)', '#FFFFFF', fs=9, bold=True)

    # Server API
    container(ax, 0.8, 1.6, 8.4, 3.6, 'SERVER API (Flask)', C['srv'])
    box(ax, 5, 4.4, 7.6, 0.65, 'Endpoint:  /predict_batch   &   /health', '#FFFFFF', fs=9)
    box(ax, 5, 3.4, 7.6, 0.8, 'Preprocessing:\nSegmentasi + Sampling 16 Frame + Normalisasi', '#FFFFFF', fs=9)
    box(ax, 5, 2.4, 7.6, 0.65, 'Model:  MobileNetV2 + TSM + CBAM', '#FFFFFF', fs=9, bold=True)

    # Panah
    arrow(ax, 5, 10.35, 5, 10.0)                                   # user -> app
    arrow(ax, 4.0, 7.75, 4.0, 5.2, 'HTTP POST\n(base64 JPEG,\nWiFi)')   # deteksi -> server
    arrow(ax, 6.2, 5.2, 6.2, 7.75, 'Response JSON\n(label,\nconfidence)', dashed=True)  # server -> app

    ax.text(5, 0.9, 'Catatan: hanya menu Deteksi yang memakai API; menu lain berjalan lokal.',
            ha='center', fontsize=8, style='italic', color='#555555')

    plt.tight_layout()
    fig.savefig('arsitektur_sistem.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('[OK] arsitektur_sistem.png')


# ══════════════════════════════════════════════════════════════════════════
# 2. MEKANISME DETEKSI (swimlane)
# ══════════════════════════════════════════════════════════════════════════
def make_mekanisme():
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 12); ax.set_ylim(0, 16); ax.axis('off')
    fig.patch.set_facecolor('white')

    ax.text(6, 15.5, 'Mekanisme Deteksi Isyarat (Aplikasi - API)',
            ha='center', fontsize=13, fontweight='bold')

    # Lane
    container(ax, 0.4, 1.0, 5.2, 13.8, 'APLIKASI MOBILE', '#D6EAF8')
    container(ax, 6.4, 1.0, 5.2, 13.8, 'SERVER API (Flask)', '#D5F5E3')

    AL, SL = 3.0, 9.0  # x-tengah lane app & server

    # App
    box(ax, AL, 14.0, 4.4, 0.7, 'Buka Menu Deteksi', C['app'])
    box(ax, AL, 12.9, 4.4, 0.7, 'Fase Persiapan (2 detik)', C['io'])
    box(ax, AL, 11.7, 4.4, 0.85, 'Rekam Gestur 3 detik\n(buffer frame di perangkat)', C['app'])
    box(ax, AL, 10.4, 4.4, 0.85, 'Konversi frame ke JPEG\nKirim klip ke server', C['app'])
    box(ax, AL, 3.3, 4.4, 0.95, "Tampilkan hasil:\nlabel + confidence\n(atau 'Belum dikenali')", C['app'])

    # Server
    box(ax, SL, 9.4, 4.4, 0.7, 'Terima klip (/predict_batch)', C['srv'])
    box(ax, SL, 8.3, 4.4, 0.7, 'Segmentasi background tiap frame', C['srv'])
    box(ax, SL, 7.2, 4.4, 0.7, 'Sampling 16 frame + Normalisasi', C['srv'])
    box(ax, SL, 6.0, 4.4, 0.85, 'Inferensi Model (Softmax)\nMobileNetV2 + TSM + CBAM', C['srv'])
    diamond(ax, SL, 4.6, 3.0, 1.3, 'Confidence\n>= threshold ?')

    # Panah app
    arrow(ax, AL, 13.65, AL, 13.25)
    arrow(ax, AL, 12.55, AL, 12.15)
    arrow(ax, AL, 11.27, AL, 10.85)
    # POST app -> server
    arrow(ax, AL + 2.2, 10.4, SL - 2.2, 9.4, 'HTTP POST\n(base64)')
    # Panah server
    arrow(ax, SL, 9.05, SL, 8.65)
    arrow(ax, SL, 7.95, SL, 7.55)
    arrow(ax, SL, 6.85, SL, 6.45)
    arrow(ax, SL, 5.55, SL, 5.25)
    # Response server -> app
    arrow(ax, SL - 1.6, 4.6, AL + 2.2, 3.5, 'Response JSON\n(label, confidence)', dashed=True)

    plt.tight_layout()
    fig.savefig('mekanisme_deteksi.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('[OK] mekanisme_deteksi.png')


if __name__ == '__main__':
    make_arsitektur()
    make_mekanisme()
    print('Selesai -> d:\\v10_lighting\\penulisan\\gambar\\')
