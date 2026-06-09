"""
BISINDO v10 — Rekam Klip Isyarat Sendiri (untuk fine-tuning)
============================================================
Merekam klip isyaratmu per kata → tersimpan rapi untuk finetune_v10.py.
Direkam MENTAH (tanpa segmentasi); finetune_v10.py yang menyegmentasi
agar identik pipeline training.

Jalankan:
    python rekam_kata.py --signer 99 --samples 12 --sec 3
    # rekam kata tertentu saja:
    python rekam_kata.py --words air,makan,saya --samples 10

Kontrol:
    SPACE : mulai merekam 1 sampel (ada countdown 3-2-1)
    n     : lewati ke kata berikutnya
    b     : kembali ke kata sebelumnya
    q/ESC : keluar
Output: data_custom/<kata>/signer<ID>_<kata>_sampleN.mp4
"""

import os, sys, time, argparse
import cv2

EXPLICIT_CLASSES = [
    'air', 'bagaimana', 'belajar', 'berangkat', 'cari',
    'datang', 'dengar', 'dimana', 'hijau', 'merah',
    'kapan', 'keluarga', 'kuning', 'lagi', 'maaf',
    'makan', 'mengapa', 'motor', 'rumah', 'saya',
    'siapa', 'teman', 'terimakasih', 'tuli'
]


def count_existing(out, word, signer):
    d = os.path.join(out, word)
    if not os.path.isdir(d):
        return 0
    pre = f'signer{signer}_{word}_sample'
    return sum(1 for f in os.listdir(d) if f.startswith(pre))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data_custom')
    ap.add_argument('--signer', type=int, default=99)
    ap.add_argument('--samples', type=int, default=12, help='target sampel per kata')
    ap.add_argument('--sec', type=float, default=3.0, help='durasi rekam per sampel')
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--words', default=None, help='daftar kata dipisah koma (default semua)')
    args = ap.parse_args()

    words = args.words.split(',') if args.words else EXPLICIT_CLASSES
    words = [w.strip() for w in words if w.strip() in EXPLICIT_CLASSES]
    if not words:
        print('[ERROR] tidak ada kata valid.'); return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[ERROR] kamera {args.camera} tidak bisa dibuka.'); return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    wi = 0
    state = 'IDLE'          # IDLE -> COUNTDOWN -> REC
    cd_start = rec_start = 0.0
    writer = None
    print('SPACE=rekam  n=kata berikut  b=sebelumnya  q=keluar')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        word = words[wi]
        have = count_existing(args.out, word, args.signer)
        disp = cv2.flip(frame, 1)
        now = time.time()

        # transisi state
        big = None
        if state == 'COUNTDOWN':
            rem = 3 - (now - cd_start)
            if rem <= 0:
                state = 'REC'; rec_start = now
                os.makedirs(os.path.join(args.out, word), exist_ok=True)
                path = os.path.join(args.out, word,
                                    f'signer{args.signer}_{word}_sample{have+1}.mp4')
                writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
                cur_path = path
            else:
                big = str(int(rem) + 1)
        elif state == 'REC':
            writer.write(frame)            # simpan frame ASLI (tanpa mirror)
            if now - rec_start >= args.sec:
                writer.release(); writer = None
                state = 'IDLE'
                print(f'[OK] {cur_path}')

        # overlay
        cv2.rectangle(disp, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(disp, f'KATA: {word}  ({have}/{args.samples})  [{wi+1}/{len(words)}]',
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 0), 2)
        if state == 'IDLE':
            msg, col = 'SPACE=rekam  n=berikut  b=sebelum  q=keluar', (200, 200, 200)
        elif state == 'COUNTDOWN':
            msg, col = 'BERSIAP...', (0, 180, 235)
        else:
            msg, col = f'REKAM {max(args.sec-(now-rec_start),0):.1f}s', (0, 0, 235)
            cv2.rectangle(disp, (0, 0), (int(w*(now-rec_start)/args.sec), 6), (0, 0, 235), -1)
        cv2.putText(disp, msg, (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if big:
            (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 5, 8)
            cv2.putText(disp, big, ((w-tw)//2, (h+th)//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 180, 235), 8)

        cv2.imshow('Rekam Isyarat', disp)
        key = cv2.waitKey(1) & 0xFF
        if state == 'IDLE':
            if key == ord(' '):
                state = 'COUNTDOWN'; cd_start = now
            elif key == ord('n'):
                wi = (wi + 1) % len(words)
            elif key == ord('b'):
                wi = (wi - 1) % len(words)
            elif key in (ord('q'), 27):
                break

    if writer:
        writer.release()
    cap.release(); cv2.destroyAllWindows()
    print('[Selesai] klip tersimpan di', args.out)


if __name__ == '__main__':
    main()
