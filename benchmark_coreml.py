"""
benchmark_coreml.py – Evaluate ArcFace R50 CoreML (.mlpackage) pipeline

Setup (run once):
    python prepare_benchmark.py     # splits test3/gallery → benchmark/
    python convert_to_coreml.py     # creates models/arcface_r50.mlpackage

Then:
    python benchmark_coreml.py      # runs evaluation, prints metrics

Model  : models/arcface_r50.mlpackage  (CoreML mlpackage, macOS only)
Detect : MTCNN (facenet_pytorch)
Align  : umeyama 5-landmark → 112×112
Embed  : ArcFace R50 CoreML → 512-d

NOTE: CoreML inference requires macOS. On Linux this script will raise an error
      when calling model.predict(). Use benchmark.py (ONNX) on Linux instead.
"""

import os
import sys
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR      = os.path.join(SCRIPT_DIR, "benchmark")
GALLERY_DIR    = os.path.join(BENCH_DIR, "gallery")
KNOWN_DIR      = os.path.join(BENCH_DIR, "known")
STRANGE_DIR    = os.path.join(BENCH_DIR, "strange")
COREML_PATH    = os.path.join(SCRIPT_DIR, "models", "arcface_r50.mlpackage")

THRESHOLD = 0.50   # cosine threshold (same as benchmark.py ArcFace default)

# ── ArcFace 5-landmark reference (112×112) ────────────────────────────────────
ARCFACE_REF_112 = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)


# ─────────────────────────────────────────────────────────────────────────────
# CoreML model loader
# ─────────────────────────────────────────────────────────────────────────────
def load_coreml(model_path: str):
    try:
        import coremltools as ct
    except ImportError:
        print("[ERROR] coremltools not installed.  pip install coremltools")
        sys.exit(1)
    if not os.path.exists(model_path):
        print(f"[ERROR] Missing: {model_path}")
        print("  Convert with:  python convert_to_coreml.py")
        sys.exit(1)
    model = ct.models.MLModel(model_path)
    spec  = model.get_spec()
    input_name  = spec.description.input[0].name
    output_name = spec.description.output[0].name
    print(f"  CoreML loaded  |  input='{input_name}'  output='{output_name}'")
    return model, input_name, output_name


# ─────────────────────────────────────────────────────────────────────────────
# 5-landmark umeyama alignment
# ─────────────────────────────────────────────────────────────────────────────
def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src, dst = src.astype(np.float64), dst.astype(np.float64)
    n = src.shape[0]
    sm, dm = src.mean(0), dst.mean(0)
    sd, dd = src - sm, dst - dm
    A = (dd.T @ sd) / n
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1; R = U @ Vt
    var = np.sum(sd ** 2) / n
    s = np.sum(S) / (var + 1e-12)
    t = dm - s * (R @ sm)
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = s * R; M[:, 2] = t
    return M.astype(np.float32)


def align_face(img_rgb: np.ndarray, lm5: np.ndarray) -> np.ndarray:
    M       = _umeyama(lm5, ARCFACE_REF_112)
    aligned = cv2.warpAffine(img_rgb, M, (112, 112),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0))
    return aligned.astype(np.uint8)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# CoreML inference
# ─────────────────────────────────────────────────────────────────────────────
def run_coreml(model, input_name: str, output_name: str,
               face_rgb_112: np.ndarray) -> np.ndarray:
    """Pass PIL Image (preprocessing baked-in) → 512-d L2-normalised embedding."""
    pil = Image.fromarray(face_rgb_112.astype(np.uint8))
    out = model.predict({input_name: pil})
    emb = np.array(out[output_name]).flatten().astype(np.float32)
    emb /= np.linalg.norm(emb) + 1e-12
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Image → embedding
# ─────────────────────────────────────────────────────────────────────────────
def extract_embedding(img_path, mtcnn, model, input_name, output_name
                      ) -> Tuple[Optional[np.ndarray], float]:
    t0 = time.perf_counter()
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  [ERR] {os.path.basename(img_path)}: cv2 read failed")
        return None, 0.0
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil     = Image.fromarray(img_rgb)
    boxes, probs, points = mtcnn.detect(pil, landmarks=True)

    if boxes is None:
        H, W = img_rgb.shape[:2]
        if max(H, W) > 640:
            sc = 640 / max(H, W)
            small = cv2.resize(img_rgb, (int(W*sc), int(H*sc)))
            b2, p2, pts2 = mtcnn.detect(Image.fromarray(small), landmarks=True)
            if b2 is not None:
                boxes, probs, points = b2, p2, pts2 / sc

    if boxes is None or points is None:
        face_112 = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    else:
        best     = int(np.argmax([p if p is not None else -1 for p in probs]))
        lm5      = points[best].astype(np.float32)
        face_112 = align_face(img_rgb, lm5)

    emb = run_coreml(model, input_name, output_name, face_112)
    return emb, (time.perf_counter() - t0) * 1000


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers  (identical to benchmark.py)
# ─────────────────────────────────────────────────────────────────────────────
def compute_far_frr(sk, ss, thr):
    fa  = sum(1 for s in ss if s >= thr)
    fr  = sum(1 for s in sk if s <  thr)
    return fa / max(len(ss), 1), fr / max(len(sk), 1)


def find_eer(sk, ss):
    best_t, best_e, best_f, best_r = 0.5, 1.0, 1.0, 1.0
    for thr in np.linspace(min(sk+ss), max(sk+ss), 500):
        far, frr = compute_far_frr(sk, ss, float(thr))
        if abs(far-frr) < best_e:
            best_e, best_t = abs(far-frr), float(thr)
            best_f, best_r = far, frr
    return best_t, best_f, best_r


def compute_auc(sk, ss):
    pts = []
    for thr in np.linspace(-0.1, 1.0, 300):
        far, frr = compute_far_frr(sk, ss, float(thr))
        pts.append((far, 1.0-frr))
    pts.sort()
    return float(np.trapz([p[1] for p in pts], [p[0] for p in pts]))


# ─────────────────────────────────────────────────────────────────────────────
# Embed a directory
# ─────────────────────────────────────────────────────────────────────────────
def embed_dir(d, label, mtcnn, model, in_name, out_name):
    files = sorted(f for f in os.listdir(d)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    names, embs, times = [], [], []
    ok = fail = 0
    for fname in files:
        emb, ms = extract_embedding(
            os.path.join(d, fname), mtcnn, model, in_name, out_name)
        if emb is not None:
            names.append(fname); embs.append(emb); times.append(ms); ok += 1
        else:
            fail += 1
    print(f"  [{label}] {ok} ok / {fail} fail  |  avg {np.mean(times):.1f} ms / img")
    return names, embs, times


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for d, n in [(GALLERY_DIR, "gallery"), (KNOWN_DIR, "known"), (STRANGE_DIR, "strange")]:
        if not os.path.isdir(d):
            print(f"[ERROR] benchmark/{n}/ not found. Run prepare_benchmark.py first.")
            sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    print("Loading models …")
    model, in_name, out_name = load_coreml(COREML_PATH)
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20,
                  thresholds=[0.6, 0.7, 0.7], factor=0.709,
                  post_process=False, keep_all=True, device=device)

    print("\nEmbedding images …")
    g_names, g_embs, g_t = embed_dir(GALLERY_DIR, "gallery", mtcnn, model, in_name, out_name)
    k_names, k_embs, k_t = embed_dir(KNOWN_DIR,   "known",   mtcnn, model, in_name, out_name)
    s_names, s_embs, s_t = embed_dir(STRANGE_DIR, "strange", mtcnn, model, in_name, out_name)

    if not g_embs:
        print("[ERROR] no gallery embeddings"); sys.exit(1)

    all_times = g_t + k_t + s_t
    print(f"\n  Total          : {len(all_times)} images")
    print(f"  Inference time : avg {np.mean(all_times):.1f} ms"
          f"  | min {np.min(all_times):.1f} ms"
          f"  | max {np.max(all_times):.1f} ms"
          f"  | p95 {np.percentile(all_times, 95):.1f} ms")

    gallery_matrix = l2_normalize(np.stack(g_embs))
    g_pids = [f.split("_")[0] for f in g_names]

    def pid(fname): return fname.split("_")[0]

    # ── score known ───────────────────────────────────────────────────────────
    known_results = []
    for fname, emb in zip(k_names, k_embs):
        pn  = l2_normalize(emb.reshape(1, -1))
        sim = (pn @ gallery_matrix.T).flatten()
        bi  = int(np.argmax(sim))
        known_results.append((fname, float(sim[bi]), pid(g_names[bi]),
                               pid(fname) == pid(g_names[bi])))
    sk = [r[1] for r in known_results]

    # ── score strange ─────────────────────────────────────────────────────────
    ss = []
    for emb in s_embs:
        pn = l2_normalize(emb.reshape(1, -1))
        ss.append(float(np.max((pn @ gallery_matrix.T).flatten())))

    # ── metrics ───────────────────────────────────────────────────────────────
    eer_t, eer_f, eer_r = find_eer(sk, ss)
    auc = compute_auc(sk, ss)
    far0_t = float(max(ss)) + 1e-5
    _, far0_r = compute_far_frr(sk, ss, far0_t)

    def acc(thr):
        ac = sum(1 for _, s, _, ok in known_results if s >= thr and ok)
        at = sum(1 for _, s, _, _  in known_results if s >= thr)
        return ac / max(at, 1)

    test_thrs = sorted(set(round(t, 4) for t in
                           [0.30, 0.35, 0.40, 0.45, THRESHOLD, 0.55, eer_t, far0_t]))

    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS  –  ArcFace R50 CoreML  (umeyama 5-landmark align)")
    print("=" * 70)
    print(f"  Gallery  : {len(g_embs):3d} images  ({len(set(g_pids))} persons)")
    print(f"  Known    : {len(k_embs):3d} images  (should be accepted)")
    print(f"  Strange  : {len(s_embs):3d} images  (should be rejected)")
    print(f"  CoreML   : {COREML_PATH}\n")
    print(f"  Score distribution:")
    print(f"    known   → mean={np.mean(sk):.4f}  std={np.std(sk):.4f}  "
          f"min={np.min(sk):.4f}  max={np.max(sk):.4f}")
    print(f"    strange → mean={np.mean(ss):.4f}  std={np.std(ss):.4f}  "
          f"min={np.min(ss):.4f}  max={np.max(ss):.4f}")
    print(f"\n  AUC (ROC) : {auc:.4f}  (1.0 = perfect)")
    print(f"\n  Threshold sweep  (FAR=0 target = {far0_t:.4f}):")
    print(f"  {'Threshold':>12}  {'FAR':>7}  {'FRR':>7}  {'TAR':>7}  {'ACC(known)':>11}  note")
    print(f"  {'-'*68}")
    for thr in test_thrs:
        far, frr = compute_far_frr(sk, ss, thr)
        notes = []
        if abs(thr - THRESHOLD) < 1e-4: notes.append("← default")
        if abs(thr - eer_t)     < 1e-4: notes.append("← EER")
        if abs(thr - far0_t)    < 1e-4: notes.append("← FAR=0")
        print(f"  {thr:>12.4f}  {far:>7.4f}  {frr:>7.4f}  {1-frr:>7.4f}  {acc(thr):>11.4f}  {'  '.join(notes)}")

    print(f"\n  ★ EER threshold   : {eer_t:.4f}  (EER≈{(eer_f+eer_r)/2:.4f}  "
          f"FAR={eer_f:.4f}  FRR={eer_r:.4f}  ACC={acc(eer_t):.4f})")
    print(f"  ★ FAR=0 threshold : {far0_t:.4f}  "
          f"(FAR=0.0000  FRR={far0_r:.4f}  TAR={1-far0_r:.4f}  ACC={acc(far0_t):.4f})")
    print(f"  ★ Inference time  : {np.mean(all_times):.1f} ms avg / image")
    print("=" * 70)

    # ── detailed known ────────────────────────────────────────────────────────
    thr_show = far0_t
    print(f"\nDetailed scores (known probes) @ threshold={thr_show:.4f}:")
    print(f"  {'File':<32} {'MaxSim':>8}  {'MatchedID':<14} {'IDok':>5}  Decision")
    print(f"  {'-'*72}")
    for fname, score, mpid, ok in known_results:
        accepted = score >= thr_show
        id_ok    = "✓" if ok else "✗"
        if accepted and ok:      dec = "✓ accept (correct ID)"
        elif accepted and not ok: dec = "✓ accept (WRONG ID!) ✗"
        else:                     dec = "✗ reject"
        print(f"  {fname:<32} {score:>8.4f}   {mpid:<14} {id_ok:>5}   {dec}")

    print(f"\nDetailed scores (strange probes) @ threshold={thr_show:.4f}:")
    print(f"  {'File':<32} {'MaxSim':>8}  Decision")
    print(f"  {'-'*58}")
    for fname, score in zip(s_names, ss):
        dec = "✗ REJECT" if score < thr_show else "✓ FALSE ACCEPT ✗"
        print(f"  {fname:<32} {score:>8.4f}   {dec}")


if __name__ == "__main__":
    main()
