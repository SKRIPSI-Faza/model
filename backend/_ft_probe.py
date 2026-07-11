import sys, os, cv2, numpy as np, torch
import torch.nn.functional as F
sys.path.insert(0, r'd:\v10_lighting\backend')
import realtime_v10 as R
import finetune_hp as FT

DATA = r'd:\v10_lighting\data_custom'
CKPT = r'd:\v10_lighting\backend\best_finetune_hp.pth'
MAXC = 5  # klip per kelas (biar cepat)

device = torch.device('cpu')
model, classes = R.load_model(CKPT, device)
cls_to_idx = {c: i for i, c in enumerate(classes)}

results, confus = {}, {}
for cls in sorted(os.listdir(DATA)):
    cdir = os.path.join(DATA, cls)
    if not os.path.isdir(cdir) or cls not in cls_to_idx:
        continue
    vids = [v for v in sorted(os.listdir(cdir)) if v.lower().endswith('.mp4')][:MAXC]
    for v in vids:
        cap = cv2.VideoCapture(os.path.join(cdir, v)); frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(cv2.resize(FT.segment_frame(f), R.IMG_SIZE))
        cap.release()
        if not frames:
            continue
        src = R.trim_active(frames)[0]
        f16 = R.sample_clip_eval(src, R.NUM_FRAMES, 0.15)
        x = R.frames_to_tensor(f16, device)
        with torch.inference_mode():
            probs = F.softmax(model(x), dim=1)[0].cpu().numpy()
        pred = classes[int(probs.argmax())]
        conf = float(probs.max()) * 100
        results.setdefault(cls, [0, 0]); results[cls][1] += 1
        if pred == cls:
            results[cls][0] += 1
        else:
            confus.setdefault(cls, []).append(f"{pred}({conf:.0f}%)")
FT._selfie.close()

print("\n=== Akurasi per-kata (best_finetune_hp pada klip rekamanmu) ===")
tot_c = tot_n = 0
for c in sorted(results):
    ok, n = results[c]; tot_c += ok; tot_n += n
    mark = "OK " if ok == n else " ! "
    cs = ("  salah->" + ", ".join(confus.get(c, [])[:3])) if ok < n else ""
    print(f"  {mark}{c:14s} {ok}/{n}{cs}")
print(f"\nTOTAL: {tot_c}/{tot_n} ({tot_c/max(tot_n,1)*100:.0f}%)")
