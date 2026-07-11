import sys, cv2, numpy as np
sys.path.insert(0, r'd:\v10_lighting\backend')
from skeleton_render import SkeletonRenderer

VIDEO = r'D:\v10_lighting\signer1_label27_sample1.mp4'
COMPLEX = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MINDET  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

rend = SkeletonRenderer(size=224, complexity=COMPLEX, min_det=MINDET, static=False)
cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f'video: {w}x{h}, {total} frames, {fps:.1f} fps  (complexity={COMPLEX}, min_det={MINDET})')

pairs = []
n = ph1 = ph2 = pp = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    skel, pose_ok, nh = rend.render(frame)
    n += 1; pp += int(pose_ok); ph1 += int(nh >= 1); ph2 += int(nh >= 2)
    pairs.append((cv2.resize(frame, (224, 224)), skel, nh))
cap.release(); rend.close()

if n == 0:
    print('[ERROR] tidak ada frame terbaca'); sys.exit(1)
print(f'frames: {n} | pose: {pp/n*100:.0f}% | >=1 tangan: {ph1/n*100:.0f}% | 2 tangan: {ph2/n*100:.0f}%')

margin = 0.15
start = int(n * margin); end = int(n * (1 - margin)); seg = max(end - start, 1) / 16
idxs = [min(int(start + i * seg + seg / 2), n - 1) for i in range(16)]
tiles = []
for i in idxs:
    cam, skel, nh = pairs[i]
    combo = np.hstack([cam, skel])
    cv2.putText(combo, f'f{i} h{nh}', (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    tiles.append(combo)
rows = [np.hstack(tiles[r*4:(r+1)*4]) for r in range(4)]
grid = np.vstack(rows)
out = r'D:\v10_lighting\backend\_skel_probe.png'
cv2.imwrite(out, grid)
print('saved', out)
