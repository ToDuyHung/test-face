"""
benchmark.py – Evaluate ArcFace face recognition pipeline

Setup (run once first):
    python prepare_benchmark.py    # splits test3/gallery → benchmark/{gallery,known,strange}

Then:
    python benchmark.py            # runs evaluation, prints metrics

Metrics:
  - Inference time per image (ms)
  - FAR  (False Acceptance Rate)  = strangers accepted / total strangers
  - FRR  (False Rejection Rate)   = known rejected    / total known
  - TAR  (True Acceptance Rate)   = 1 - FRR           (correctly identified known)
  - Optimal threshold via EER     (Equal Error Rate point)
  - AUC  (Area Under ROC Curve)
"""

import os
import sys
import time
from typing import Optional, List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from facenet_pytorch import MTCNN

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR   = os.path.join(SCRIPT_DIR, "benchmark")
GALLERY_DIR = os.path.join(BENCH_DIR, "gallery")
KNOWN_DIR   = os.path.join(BENCH_DIR, "known")
STRANGE_DIR = os.path.join(BENCH_DIR, "strange")

ARCFACE_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "w600k_r50.onnx")

# ── ArcFace reference landmarks ───────────────────────────────────────────────
ARCFACE_REF_112 = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_arcface_session() -> ort.InferenceSession:
    if not os.path.exists(ARCFACE_MODEL_PATH):
        print("[ERROR] ArcFace model not found. Run test_pytorch.py first to download it.")
        sys.exit(1)
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(ARCFACE_MODEL_PATH, providers=providers)
    print(f"  ArcFace loaded  |  provider: {sess.get_providers()[0]}")
    return sess


def arcface_preprocess(face_rgb_112: np.ndarray) -> np.ndarray:
    img = face_rgb_112.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0)


def run_arcface(sess: ort.InferenceSession, face_np: np.ndarray) -> np.ndarray:
    inp = {sess.get_inputs()[0].name: arcface_preprocess(face_np)}
    emb = sess.run(None, inp)[0][0]
    emb = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype(np.float32)


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
        U[:, -1] *= -1
        R = U @ Vt
    var = np.sum(sd ** 2) / n
    scale = np.sum(S) / (var + 1e-12)
    t = dm - scale * (R @ sm)
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2]  = t
    return M.astype(np.float32)


def align_face(img_rgb: np.ndarray, lm5: np.ndarray, out_size: int = 112) -> np.ndarray:
    ref = ARCFACE_REF_112 * (out_size / 112.0)
    M   = _umeyama(lm5, ref)
    aligned = cv2.warpAffine(
        img_rgb, M, (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned.astype(np.uint8)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Image → embedding
# ─────────────────────────────────────────────────────────────────────────────
def extract_embedding(
    img_path: str,
    mtcnn: MTCNN,
    arc_sess: ort.InferenceSession,
) -> Tuple[Optional[np.ndarray], float]:
    """Returns (embedding, inference_ms). embedding=None on failure."""
    t0 = time.perf_counter()
    try:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2 read failed")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None, 0.0

    pil_img = Image.fromarray(img_rgb)
    boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)

    if boxes is None:
        H, W = img_rgb.shape[:2]
        if max(H, W) > 640:
            scale = 640 / max(H, W)
            small = cv2.resize(img_rgb, (int(W * scale), int(H * scale)))
            b2, p2, pts2 = mtcnn.detect(Image.fromarray(small), landmarks=True)
            if b2 is not None:
                boxes, probs, points = b2, p2, pts2 / scale

    if boxes is None or points is None:
        # plain resize fallback
        face_112 = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    else:
        best     = int(np.argmax([p if p is not None else -1 for p in probs]))
        lm5      = points[best].astype(np.float32)
        face_112 = align_face(img_rgb, lm5, out_size=112)

    emb = run_arcface(arc_sess, face_112.astype(np.float32))
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return emb, elapsed_ms


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────
def compute_far_frr(scores_known, scores_strange, threshold):
    """
    known   → should be ACCEPTED (max_sim >= threshold)
    strange → should be REJECTED  (max_sim <  threshold)
    """
    fa  = sum(1 for s in scores_strange if s >= threshold)   # false accepts
    fr  = sum(1 for s in scores_known   if s <  threshold)   # false rejects
    far = fa / max(len(scores_strange), 1)
    frr = fr / max(len(scores_known), 1)
    return far, frr


def find_eer_threshold(scores_known, scores_strange):
    """Sweep thresholds, find EER point (FAR ≈ FRR)."""
    # collect all unique scores as candidate thresholds
    all_scores = sorted(set(scores_known + scores_strange))
    best_thresh, best_eer, best_far, best_frr = 0.5, 1.0, 1.0, 1.0
    for thr in np.linspace(min(all_scores), max(all_scores), 500):
        far, frr = compute_far_frr(scores_known, scores_strange, float(thr))
        eer = abs(far - frr)
        if eer < best_eer:
            best_eer, best_thresh = eer, float(thr)
            best_far, best_frr   = far, frr
    return best_thresh, best_far, best_frr


def compute_auc(scores_known, scores_strange):
    """Area under ROC (TAR vs FAR) via trapezoidal rule."""
    points = []
    for thr in np.linspace(-0.1, 1.0, 300):
        far, frr = compute_far_frr(scores_known, scores_strange, float(thr))
        tar = 1.0 - frr
        points.append((far, tar))
    points.sort()
    fars  = [p[0] for p in points]
    tars  = [p[1] for p in points]
    return float(np.trapz(tars, fars))


# ─────────────────────────────────────────────────────────────────────────────
# Load embeddings for a directory
# ─────────────────────────────────────────────────────────────────────────────
def embed_directory(
    dir_path: str,
    label: str,
    mtcnn: MTCNN,
    arc_sess: ort.InferenceSession,
) -> Tuple[List[str], List[np.ndarray], List[float]]:
    files = sorted(
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    names, embs, times = [], [], []
    ok = fail = 0
    for fname in files:
        emb, ms = extract_embedding(os.path.join(dir_path, fname), mtcnn, arc_sess)
        if emb is not None:
            names.append(fname)
            embs.append(emb)
            times.append(ms)
            ok += 1
        else:
            fail += 1
    print(f"  [{label}] {ok} ok / {fail} fail  |  "
          f"avg {np.mean(times):.1f} ms / img")
    return names, embs, times


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── checks ────────────────────────────────────────────────────────────────
    for d, name in [(GALLERY_DIR, "gallery"), (KNOWN_DIR, "known"), (STRANGE_DIR, "strange")]:
        if not os.path.isdir(d):
            print(f"[ERROR] benchmark/{name}/ not found. Run prepare_benchmark.py first.")
            sys.exit(1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ── models ────────────────────────────────────────────────────────────────
    print("Loading models …")
    arc_sess = load_arcface_session()
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=False, keep_all=True, device=device,
    )

    # ── embed all sets ────────────────────────────────────────────────────────
    print("\nEmbedding images …")
    g_names, g_embs, g_times = embed_directory(GALLERY_DIR, "gallery", mtcnn, arc_sess)
    k_names, k_embs, k_times = embed_directory(KNOWN_DIR,   "known",   mtcnn, arc_sess)
    s_names, s_embs, s_times = embed_directory(STRANGE_DIR, "strange", mtcnn, arc_sess)

    if not g_embs:
        print("[ERROR] no gallery embeddings"); sys.exit(1)

    all_times = g_times + k_times + s_times
    print(f"\n  Total images embedded : {len(all_times)}")
    print(f"  Inference time        : avg {np.mean(all_times):.1f} ms  "
          f"| min {np.min(all_times):.1f} ms  "
          f"| max {np.max(all_times):.1f} ms  "
          f"| p95 {np.percentile(all_times, 95):.1f} ms")

    # ── build gallery matrix ──────────────────────────────────────────────────
    gallery_matrix = l2_normalize(np.stack(g_embs))   # (G, 512)

    # Extract person ID from filename: FID70014724_front.jpg → FID70014724
    def pid(fname):
        return fname.split("_")[0]

    g_pids = [pid(f) for f in g_names]

    # ── score known probes ─────────────────────────────────────────────────────
    # Each entry: (max_sim, best_gallery_pid, is_correct_id)
    known_results = []
    for fname, emb in zip(k_names, k_embs):
        probe_norm  = l2_normalize(emb.reshape(1, -1))
        sims        = (probe_norm @ gallery_matrix.T).flatten()
        best_idx    = int(np.argmax(sims))
        max_sim     = float(sims[best_idx])
        best_gname  = g_names[best_idx]
        is_correct  = (pid(fname) == pid(best_gname))   # True if matched right person
        known_results.append((fname, max_sim, pid(best_gname), is_correct))

    scores_known = [r[1] for r in known_results]

    # ── score strange probes ──────────────────────────────────────────────────
    scores_strange = []
    for fname, emb in zip(s_names, s_embs):
        probe_norm = l2_normalize(emb.reshape(1, -1))
        sims       = (probe_norm @ gallery_matrix.T).flatten()
        max_sim    = float(np.max(sims))
        scores_strange.append(max_sim)

    # ── find optimal threshold (EER) ─────────────────────────────────────────
    eer_thresh, eer_far, eer_frr = find_eer_threshold(scores_known, scores_strange)
    auc = compute_auc(scores_known, scores_strange)

    # ── FAR=0 threshold: smallest thr where ALL strangers are rejected ────────
    far0_thresh = float(max(scores_strange)) + 1e-5
    _, far0_frr = compute_far_frr(scores_known, scores_strange, far0_thresh)

    def compute_acc_known(thr: float) -> float:
        """
        ACC = correct identity matches / total known probes above threshold.
        A known probe is 'correct' if:  accepted AND matched the right gallery person.
        """
        accepted_correct = sum(
            1 for _, sim, _, is_correct in known_results
            if sim >= thr and is_correct
        )
        accepted_total = sum(1 for _, sim, _, _ in known_results if sim >= thr)
        return accepted_correct / max(accepted_total, 1)

    # ── metrics at different thresholds ──────────────────────────────────────
    test_thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, eer_thresh, far0_thresh]
    test_thresholds = sorted(set(round(t, 4) for t in test_thresholds))

    # ── print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS  –  ArcFace R50  (umeyama 5-landmark align)")
    print("=" * 70)
    print(f"  Gallery  : {len(g_embs):3d} images  ({len(set(g_pids))} persons)")
    print(f"  Known    : {len(k_embs):3d} images  (should be accepted)")
    print(f"  Strange  : {len(s_embs):3d} images  (should be rejected)\n")

    print(f"  Score distribution:")
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
        if abs(thr - eer_thresh) < 1e-4:  notes.append("← EER")
        if abs(thr - far0_thresh) < 1e-4: notes.append("← FAR=0")
        print(f"  {thr:>12.4f}  {far:>7.4f}  {frr:>7.4f}  {tar:>7.4f}  {acc:>11.4f}  {'  '.join(notes)}")

    print(f"\n  ★ EER threshold     : {eer_thresh:.4f}  "
          f"(EER≈{((eer_far+eer_frr)/2):.4f}  FAR={eer_far:.4f}  FRR={eer_frr:.4f}  "
          f"ACC={compute_acc_known(eer_thresh):.4f})")
    print(f"  ★ FAR=0 threshold   : {far0_thresh:.4f}  "
          f"(FAR=0.0000  FRR={far0_frr:.4f}  TAR={1-far0_frr:.4f}  "
          f"ACC={compute_acc_known(far0_thresh):.4f})")
    print(f"  ★ Inference time    : {np.mean(all_times):.1f} ms avg per image")
    print("=" * 70)

    # ── per-probe score dump ──────────────────────────────────────────────────
    # Use FAR=0 threshold for the Decision column
    thr_show = far0_thresh

    print(f"\nDetailed scores (known probes) @ threshold={thr_show:.4f}:")
    print(f"  {'File':<32} {'MaxSim':>8}  {'MatchedID':<14} {'IDok':>5}  Decision")
    print(f"  {'-'*72}")
    for fname, score, matched_pid, is_correct in known_results:
        accepted  = score >= thr_show
        id_ok     = "✓" if is_correct else "✗"
        if accepted and is_correct:     decision = "✓ accept (correct ID)"
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
