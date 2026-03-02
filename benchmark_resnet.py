"""
benchmark_resnet.py – Evaluate InceptionResnetV1 (VGGFace2) face recognition pipeline

Setup (run once first):
    python prepare_benchmark.py    # splits test3/gallery → benchmark/{gallery,known,strange}

Then:
    python benchmark_resnet.py     # runs evaluation, prints metrics

Model : facenet_pytorch.InceptionResnetV1 (VGGFace2 pretrained, 512-d)
Detect: facenet_pytorch.MTCNN  (post_process=True → already normalised for InceptionResnetV1)

Metrics:
  - Inference time per image (ms)
  - FAR  (False Acceptance Rate)  = strangers accepted / total strangers
  - FRR  (False Rejection Rate)   = known rejected    / total known
  - TAR  (True Acceptance Rate)   = 1 - FRR
  - ACC  (Accuracy among known)   = correct-ID matches / accepted known
  - EER  optimal threshold
  - FAR=0 threshold
  - AUC  (Area Under ROC Curve)
"""

import os
import sys
import time
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR   = os.path.join(SCRIPT_DIR, "benchmark")
GALLERY_DIR = os.path.join(BENCH_DIR, "gallery")
KNOWN_DIR   = os.path.join(BENCH_DIR, "known")
STRANGE_DIR = os.path.join(BENCH_DIR, "strange")

# Default threshold tuned for InceptionResnetV1 / VGGFace2
DEFAULT_THRESHOLD = 0.74


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


@torch.no_grad()
def extract_embedding(
    img_path: str,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], float]:
    """
    1. Load image (PIL RGB)
    2. MTCNN → 160×160 aligned + normalised face tensor (post_process=True)
    3. InceptionResnetV1 → 512-d L2-normalised embedding

    Returns (embedding, inference_ms). embedding=None on failure.
    """
    t0 = time.perf_counter()
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as ex:
        print(f"  [ERR] {os.path.basename(img_path)}: {ex}")
        return None, 0.0

    face_tensors = mtcnn(img)   # [N,3,160,160] normalised, or None

    if face_tensors is None:
        # fallback: downscale to 640 px
        W, H = img.size
        m = max(W, H)
        if m > 640:
            scale = 640 / m
            img_small = img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR)
            face_tensors = mtcnn(img_small)

    if face_tensors is None:
        print(f"  [WARN] No face: {os.path.basename(img_path)}")
        return None, 0.0

    # pick highest-confidence face
    _, probs = mtcnn.detect(img)
    if probs is not None and len(probs) > 0:
        best = int(np.argmax([p if p is not None else -1 for p in probs]))
    else:
        best = 0

    if face_tensors.ndim == 3:
        face = face_tensors.unsqueeze(0)
    else:
        best = min(best, face_tensors.shape[0] - 1)
        face = face_tensors[best].unsqueeze(0)   # [1,3,160,160]

    face = face.to(device)
    emb  = resnet(face)[0].detach().cpu().numpy().astype(np.float32)   # (512,)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return emb, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_far_frr(scores_known, scores_strange, threshold):
    fa  = sum(1 for s in scores_strange if s >= threshold)
    fr  = sum(1 for s in scores_known   if s <  threshold)
    far = fa / max(len(scores_strange), 1)
    frr = fr / max(len(scores_known),   1)
    return far, frr


def find_eer_threshold(scores_known, scores_strange):
    best_thresh, best_eer, best_far, best_frr = 0.5, 1.0, 1.0, 1.0
    all_scores = scores_known + scores_strange
    for thr in np.linspace(min(all_scores), max(all_scores), 500):
        far, frr = compute_far_frr(scores_known, scores_strange, float(thr))
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer, best_thresh = eer, float(thr)
            best_far, best_frr   = far, frr
    return best_thresh, best_far, best_frr


def compute_auc(scores_known, scores_strange):
    points = []
    for thr in np.linspace(-0.1, 1.0, 300):
        far, frr = compute_far_frr(scores_known, scores_strange, float(thr))
        points.append((far, 1.0 - frr))
    points.sort()
    fars = [p[0] for p in points]
    tars = [p[1] for p in points]
    return float(np.trapz(tars, fars))


# ─────────────────────────────────────────────────────────────────────────────
# Load embeddings for a directory
# ─────────────────────────────────────────────────────────────────────────────
def embed_directory(
    dir_path: str,
    label: str,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: torch.device,
) -> Tuple[List[str], List[np.ndarray], List[float]]:
    files = sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    names, embs, times = [], [], []
    ok = fail = 0
    for fname in files:
        emb, ms = extract_embedding(os.path.join(dir_path, fname), mtcnn, resnet, device)
        if emb is not None:
            names.append(fname)
            embs.append(emb)
            times.append(ms)
            ok += 1
        else:
            fail += 1
    print(f"  [{label}] {ok} ok / {fail} fail  |  avg {np.mean(times):.1f} ms / img")
    return names, embs, times


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    for d, name in [(GALLERY_DIR, "gallery"), (KNOWN_DIR, "known"), (STRANGE_DIR, "strange")]:
        if not os.path.isdir(d):
            print(f"[ERROR] benchmark/{name}/ not found. Run prepare_benchmark.py first.")
            sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── models ────────────────────────────────────────────────────────────────
    print("Loading MTCNN + InceptionResnetV1 (VGGFace2) …")
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=True,   # normalises output for InceptionResnetV1
        keep_all=True, device=device,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # ── embed all sets ────────────────────────────────────────────────────────
    print("\nEmbedding images …")
    g_names, g_embs, g_times = embed_directory(GALLERY_DIR, "gallery", mtcnn, resnet, device)
    k_names, k_embs, k_times = embed_directory(KNOWN_DIR,   "known",   mtcnn, resnet, device)
    s_names, s_embs, s_times = embed_directory(STRANGE_DIR, "strange", mtcnn, resnet, device)

    if not g_embs:
        print("[ERROR] no gallery embeddings"); sys.exit(1)

    all_times = g_times + k_times + s_times
    print(f"\n  Total images embedded : {len(all_times)}")
    print(f"  Inference time        : avg {np.mean(all_times):.1f} ms"
          f"  | min {np.min(all_times):.1f} ms"
          f"  | max {np.max(all_times):.1f} ms"
          f"  | p95 {np.percentile(all_times, 95):.1f} ms")

    # ── build gallery matrix ──────────────────────────────────────────────────
    gallery_matrix = l2_normalize(np.stack(g_embs))   # (G, 512)

    def pid(fname):
        return fname.split("_")[0]

    g_pids = [pid(f) for f in g_names]

    # ── score known probes ────────────────────────────────────────────────────
    known_results = []
    for fname, emb in zip(k_names, k_embs):
        probe_norm = l2_normalize(emb.reshape(1, -1))
        sims       = (probe_norm @ gallery_matrix.T).flatten()
        best_idx   = int(np.argmax(sims))
        max_sim    = float(sims[best_idx])
        best_gname = g_names[best_idx]
        is_correct = (pid(fname) == pid(best_gname))
        known_results.append((fname, max_sim, pid(best_gname), is_correct))

    scores_known = [r[1] for r in known_results]

    # ── score strange probes ──────────────────────────────────────────────────
    scores_strange = []
    for fname, emb in zip(s_names, s_embs):
        probe_norm = l2_normalize(emb.reshape(1, -1))
        sims       = (probe_norm @ gallery_matrix.T).flatten()
        scores_strange.append(float(np.max(sims)))

    # ── thresholds ────────────────────────────────────────────────────────────
    eer_thresh, eer_far, eer_frr = find_eer_threshold(scores_known, scores_strange)
    auc = compute_auc(scores_known, scores_strange)

    far0_thresh = float(max(scores_strange)) + 1e-5
    _, far0_frr = compute_far_frr(scores_known, scores_strange, far0_thresh)

    def compute_acc_known(thr: float) -> float:
        accepted_correct = sum(
            1 for _, sim, _, is_correct in known_results
            if sim >= thr and is_correct
        )
        accepted_total = sum(1 for _, sim, _, _ in known_results if sim >= thr)
        return accepted_correct / max(accepted_total, 1)

    test_thresholds = [0.60, 0.65, 0.70, DEFAULT_THRESHOLD, 0.78, 0.82, eer_thresh, far0_thresh]
    test_thresholds = sorted(set(round(t, 4) for t in test_thresholds))

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS  –  InceptionResnetV1 VGGFace2  (MTCNN align)")
    print("=" * 70)
    print(f"  Gallery  : {len(g_embs):3d} images  ({len(set(g_pids))} persons)")
    print(f"  Known    : {len(k_embs):3d} images  (should be accepted)")
    print(f"  Strange  : {len(s_embs):3d} images  (should be rejected)\n")

    print("  Score distribution:")
    print(f"    known   → mean={np.mean(scores_known):.4f}  std={np.std(scores_known):.4f}  "
          f"min={np.min(scores_known):.4f}  max={np.max(scores_known):.4f}")
    print(f"    strange → mean={np.mean(scores_strange):.4f}  std={np.std(scores_strange):.4f}  "
          f"min={np.min(scores_strange):.4f}  max={np.max(scores_strange):.4f}")
    print(f"\n  AUC (ROC) : {auc:.4f}  (1.0 = perfect)")

    print(f"\n  Threshold sweep  (FAR=0 target = {far0_thresh:.4f}):")
    print(f"  {'Threshold':>12}  {'FAR':>7}  {'FRR':>7}  {'TAR':>7}  {'ACC(known)':>11}  note")
    print(f"  {'-'*68}")
    for thr in test_thresholds:
        far, frr = compute_far_frr(scores_known, scores_strange, thr)
        tar = 1.0 - frr
        acc = compute_acc_known(thr)
        notes = []
        if abs(thr - DEFAULT_THRESHOLD) < 1e-4: notes.append("← default")
        if abs(thr - eer_thresh)       < 1e-4:  notes.append("← EER")
        if abs(thr - far0_thresh)      < 1e-4:  notes.append("← FAR=0")
        print(f"  {thr:>12.4f}  {far:>7.4f}  {frr:>7.4f}  {tar:>7.4f}  {acc:>11.4f}  {'  '.join(notes)}")

    print(f"\n  ★ Default threshold : {DEFAULT_THRESHOLD:.4f}  "
          f"(FAR={compute_far_frr(scores_known, scores_strange, DEFAULT_THRESHOLD)[0]:.4f}  "
          f"FRR={compute_far_frr(scores_known, scores_strange, DEFAULT_THRESHOLD)[1]:.4f}  "
          f"ACC={compute_acc_known(DEFAULT_THRESHOLD):.4f})")
    print(f"  ★ EER threshold     : {eer_thresh:.4f}  "
          f"(EER≈{((eer_far+eer_frr)/2):.4f}  FAR={eer_far:.4f}  FRR={eer_frr:.4f}  "
          f"ACC={compute_acc_known(eer_thresh):.4f})")
    print(f"  ★ FAR=0 threshold   : {far0_thresh:.4f}  "
          f"(FAR=0.0000  FRR={far0_frr:.4f}  TAR={1-far0_frr:.4f}  "
          f"ACC={compute_acc_known(far0_thresh):.4f})")
    print(f"  ★ Inference time    : {np.mean(all_times):.1f} ms avg per image")
    print("=" * 70)

    # ── per-probe details (using default threshold) ───────────────────────────
    thr_show = DEFAULT_THRESHOLD

    print(f"\nDetailed scores (known probes) @ threshold={thr_show:.4f}:")
    print(f"  {'File':<32} {'MaxSim':>8}  {'MatchedID':<14} {'IDok':>5}  Decision")
    print(f"  {'-'*72}")
    for fname, score, matched_pid, is_correct in known_results:
        accepted = score >= thr_show
        id_ok    = "✓" if is_correct else "✗"
        if accepted and is_correct:      decision = "✓ accept (correct ID)"
        elif accepted and not is_correct: decision = "✓ accept (WRONG ID!) ✗"
        else:                             decision = "✗ reject"
        print(f"  {fname:<32} {score:>8.4f}   {matched_pid:<14} {id_ok:>5}   {decision}")

    print(f"\nDetailed scores (strange probes) @ threshold={thr_show:.4f}:")
    print(f"  {'File':<32} {'MaxSim':>8}  Decision")
    print(f"  {'-'*58}")
    for fname, score in zip(s_names, scores_strange):
        decision = "✗ REJECT" if score < thr_show else "✓ FALSE ACCEPT ✗"
        print(f"  {fname:<32} {score:>8.4f}   {decision}")


if __name__ == "__main__":
    main()
