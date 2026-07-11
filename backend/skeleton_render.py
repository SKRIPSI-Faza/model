"""
skeleton_render.py — MediaPipe Pose + Hands -> gambar skeleton TEBAL-BERWARNA,
dinormalisasi terhadap BAHU (invarian orang & kamera).

Pakai detektor TANGAN khusus (mp.solutions.hands) yang jauh lebih kuat daripada
hand-detector bawaan Holistic, + Pose untuk badan-atas & titik KEPALA/TELINGA
(acuan isyarat di telinga/kepala). Dipakai IDENTIK di training & inference.

LIHAT DULU (tanpa retrain):
    # webcam
    python skeleton_render.py --camera 0
    # satu video (lebih terkontrol — bisa diputar ulang)
    python skeleton_render.py --video "path\ke\video.mp4"
    # simpan hasil berdampingan jadi mp4
    python skeleton_render.py --video "path\ke\video.mp4" --save out.mp4
    # deteksi lemah? naikkan akurasi:
    python skeleton_render.py --video v.mp4 --complexity 2 --min-det 0.3

Kontrol jendela: q = keluar, SPASI = pause/lanjut (mode video).
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp

mp_pose          = mp.solutions.pose
mp_hands         = mp.solutions.hands
HAND_CONNECTIONS = mp_hands.HAND_CONNECTIONS

# Pose badan-atas + kepala. Indeks: 0=hidung, 7/8=telinga, 11/12=bahu,
# 13/14=siku, 15/16=pergelangan.
POSE_CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                    (11, 0), (12, 0), (0, 7), (0, 8)]
POSE_POINTS      = [0, 7, 8, 11, 12, 13, 14, 15, 16]

COL_POSE  = (200, 200, 200)   # badan/kepala: abu
COL_LHAND = (0, 200, 255)     # tangan kiri: oranye
COL_RHAND = (255, 150, 0)     # tangan kanan: biru

CANVAS_SCALE = 0.25           # lebar bahu -> 25% kanvas
ORIGIN_Y     = 0.45           # pusat bahu di 45% tinggi kanvas


class SkeletonRenderer:
    """render(frame_bgr) -> (skeleton BGR size×size, pose_ok, n_hands)."""

    def __init__(self, size=224, complexity=1, min_det=0.5, min_trk=0.5,
                 static=False):
        self.size = size
        self.pose = mp_pose.Pose(
            static_image_mode=static, model_complexity=min(complexity, 2),
            smooth_landmarks=not static,
            min_detection_confidence=min_det, min_tracking_confidence=min_trk)
        self.hands = mp_hands.Hands(
            static_image_mode=static, max_num_hands=2,
            model_complexity=min(complexity, 1),
            min_detection_confidence=min_det, min_tracking_confidence=min_trk)

    def _px(self, lm, w, h):
        return np.array([lm.x * w, lm.y * h], dtype=np.float32)

    def render(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pres = self.pose.process(rgb)
        hres = self.hands.process(rgb)
        canvas = np.zeros((self.size, self.size, 3), np.uint8)

        if pres.pose_landmarks is None:
            return canvas, False, 0

        pl = pres.pose_landmarks.landmark
        l_sh, r_sh = self._px(pl[11], w, h), self._px(pl[12], w, h)
        center = (l_sh + r_sh) / 2.0
        sw = float(np.linalg.norm(l_sh - r_sh))
        if sw < 1e-3:
            return canvas, False, 0

        scale = CANVAS_SCALE * self.size / sw
        ox, oy = self.size * 0.5, self.size * ORIGIN_Y

        def to_canvas(px):
            return (int(round(ox + (px[0] - center[0]) * scale)),
                    int(round(oy + (px[1] - center[1]) * scale)))

        th  = max(2, self.size // 80)
        rad = max(2, self.size // 90)

        # ── Pose badan-atas + kepala ──
        pose_px = {i: self._px(pl[i], w, h) for i in POSE_POINTS}
        for a, b in POSE_CONNECTIONS:
            cv2.line(canvas, to_canvas(pose_px[a]), to_canvas(pose_px[b]), COL_POSE, th)
        for i in POSE_POINTS:
            cv2.circle(canvas, to_canvas(pose_px[i]), rad, COL_POSE, -1)

        # ── Tangan (detektor khusus) ──
        n_hands = 0
        if hres.multi_hand_landmarks:
            handed = hres.multi_handedness or []
            for k, hand_lm in enumerate(hres.multi_hand_landmarks):
                label = (handed[k].classification[0].label
                         if k < len(handed) else 'Right')
                col = COL_LHAND if label == 'Left' else COL_RHAND
                pts = [self._px(lm, w, h) for lm in hand_lm.landmark]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(canvas, to_canvas(pts[a]), to_canvas(pts[b]), col, th)
                for px in pts:
                    cv2.circle(canvas, to_canvas(px), rad, col, -1)
                n_hands += 1

        return canvas, True, n_hands

    def close(self):
        self.pose.close(); self.hands.close()


def _overlay_stats(view, pose_ok, n_hands, rate1, rate2):
    txt = f'pose:{"OK" if pose_ok else "-"}  tangan:{n_hands}'
    col = (0, 230, 0) if n_hands >= 1 else (0, 0, 235)
    cv2.putText(view, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    cv2.putText(view, f'>=1 tangan: {rate1:.0f}%  | 2 tangan: {rate2:.0f}%',
                (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (190, 190, 190), 1)


def run(source, is_video, size, complexity, min_det, save_path):
    rend = SkeletonRenderer(size=size, complexity=complexity, min_det=min_det)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'[ERROR] tak bisa buka: {source}'); rend.close(); return

    writer = None
    if save_path:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (size * 2, size))

    n_tot = n_h1 = n_h2 = 0
    paused = False
    print('[Jalan] q=keluar' + ('  SPASI=pause' if is_video else ''))
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                skel, pose_ok, n_hands = rend.render(frame)
                n_tot += 1
                n_h1 += int(n_hands >= 1); n_h2 += int(n_hands >= 2)
                rate1 = n_h1 / n_tot * 100; rate2 = n_h2 / n_tot * 100

                cam = cv2.resize(frame, (size, size))
                view = skel.copy()
                _overlay_stats(view, pose_ok, n_hands, rate1, rate2)
                combo = np.hstack([cam, view])
                if writer:
                    writer.write(combo)
                cv2.imshow('Sumber (kiri) | Skeleton yang dilihat model (kanan)', combo)

            key = cv2.waitKey(15 if is_video else 1) & 0xFF
            if key == ord('q'):
                break
            if is_video and key == ord(' '):
                paused = not paused
    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows(); rend.close()
        if n_tot:
            print(f'\n[Ringkas] {n_tot} frame | >=1 tangan: {n_h1/n_tot*100:.0f}% '
                  f'| 2 tangan: {n_h2/n_tot*100:.0f}%')
            print('  Stabil & >80% (1 tangan) -> skeleton layak, lanjut retrain.')
            print('  Sering 0 tangan / kedip  -> coba --complexity 2 --min-det 0.3;')
            print('                              kalau tetap buruk -> pakai HYBRID.')
        if save_path:
            print(f'[Saved] {save_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--video', default=None, help='path video (override --camera)')
    ap.add_argument('--save', default=None, help='simpan hasil berdampingan ke mp4')
    ap.add_argument('--size', type=int, default=224)
    ap.add_argument('--complexity', type=int, default=1, help='1=cepat, 2=akurat')
    ap.add_argument('--min-det', type=float, default=0.5,
                    help='turunkan (mis. 0.3) kalau tangan sering tak terdeteksi')
    args = ap.parse_args()

    if args.video:
        run(args.video, True, args.size, args.complexity, args.min_det, args.save)
    else:
        run(args.camera, False, args.size, args.complexity, args.min_det, args.save)


if __name__ == '__main__':
    main()
