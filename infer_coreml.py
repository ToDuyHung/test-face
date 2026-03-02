"""
infer_coreml.py – ArcFace R50 inference via CoreML .mlpackage

Pipeline (same as benchmark.py):
  1. MTCNN detect face → 5-point landmarks
  2. Umeyama 5-landmark affine warp → 112×112 RGB
  3. ArcFace CoreML  → 512-d embedding
  4. L2-normalise → cosine similarity

Requirements:
    pip install coremltools facenet-pytorch torch opencv-python Pillow
    macOS with Apple Silicon or Intel Mac  (CoreML = macOS only)

Usage:
    # Single image
    python infer_coreml.py --image path/to/face.jpg

    # Gallery vs probes (same directory structure as benchmark.py)
    python infer_coreml.py --gallery benchmark/gallery --probes benchmark/known
    python infer_coreml.py --gallery benchmark/gallery --probes benchmark/strange

    # Custom mlpackage path
    python infer_coreml.py --model models/arcface_r50.mlpackage --image face.jpg
"""

import argparse
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
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "models", "arcface_r50.mlpackage")

THRESHOLD = 0.50   # cosine threshold: same person ≥ 0.50

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
# Load CoreML model
# ─────────────────────────────────────────────────────────────────────────────
def load_coreml_model(model_path: str):
    try:
        import coremltools as ct
    except ImportError:
        print("[ERROR] coremltools not installed.  Run:  pip install coremltools")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"[ERROR] CoreML model not found: {model_path}")
        print("  Convert first:  python convert_to_coreml.py")
        sys.exit(1)

    import coremltools as ct
    model = ct.models.MLModel(model_path)
    print(f"  CoreML model loaded: {model_path}")
    return model


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


def align_face_112(img_rgb: np.ndarray, lm5: np.ndarray) -> np.ndarray:
    """Affine-warp img_rgb (uint8 RGB) using MTCNN 5-point landmarks → 112×112."""
    M       = _umeyama(lm5, ARCFACE_REF_112)
    aligned = cv2.warpAffine(
        img_rgb, M, (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# CoreML inference helpers
# ─────────────────────────────────────────────────────────────────────────────
def run_coreml(model, face_rgb_112: np.ndarray) -> np.ndarray:
    """
    Run ArcFace CoreML on a 112×112 RGB uint8 face.
    If the model has ImageType input, pass a PIL.Image.
    Otherwise pre-normalise and pass float array directly.
    """
    pil_face = Image.fromarray(face_rgb_112.astype(np.uint8))

    # Try ImageType input (preprocessing baked-in)
    try:
        spec = model.get_spec()
        input_name = spec.description.input[0].name
        out = model.predict({input_name: pil_face})
        output_name = spec.description.output[0].name
        emb = np.array(out[output_name]).flatten().astype(np.float32)
    except Exception:
        # Fallback: manual normalisation → float32 array
        arr = face_rgb_112.astype(np.float32)
        arr = (arr / 127.5) - 1.0
        arr = arr.transpose(2, 0, 1)          # HWC → CHW
        arr = np.expand_dims(arr, 0)          # → (1,3,112,112)
        out = model.predict({"input": arr})
        emb = np.array(list(out.values())[0]).flatten().astype(np.float32)

    emb /= np.linalg.norm(emb) + 1e-12
    return emb


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline: image path → embedding
# ─────────────────────────────────────────────────────────────────────────────
def extract_embedding(
    img_path: str,
    mtcnn: MTCNN,
    coreml_model,
) -> Tuple[Optional[np.ndarray], float]:
    """Returns (L2-normalised 512-d embedding, inference_ms) or (None, 0.0)."""
    t0 = time.perf_counter()

    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"  [ERR] cv2 read failed: {img_path}")
        return None, 0.0
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

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
        print(f"  [WARN] MTCNN miss → plain resize: {os.path.basename(img_path)}")
        face_112 = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
    else:
        best     = int(np.argmax([p if p is not None else -1 for p in probs]))
        lm5      = points[best].astype(np.float32)
        face_112 = align_face_112(img_rgb, lm5)

    emb = run_coreml(coreml_model, face_112)
    return emb, (time.perf_counter() - t0) * 1000


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Modes
# ─────────────────────────────────────────────────────────────────────────────
def mode_single_image(img_path: str, mtcnn: MTCNN, model, verbose: bool = True):
    """Embed a single image and print the embedding statistics."""
    print(f"\nImage: {img_path}")
    emb, ms = extract_embedding(img_path, mtcnn, model)
    if emb is None:
        print("[FAIL] No embedding extracted.")
        return None
    print(f"  ✓ Embedding extracted  ({ms:.1f} ms)")
    print(f"    shape : {emb.shape}")
    print(f"    norm  : {np.linalg.norm(emb):.6f}  (should be ≈1.0)")
    print(f"    mean  : {emb.mean():.4f}  std: {emb.std():.4f}")
    return emb


def mode_gallery_probe(
    gallery_dir: str,
    probe_dir: str,
    mtcnn: MTCNN,
    model,
    threshold: float = THRESHOLD,
):
    """Build gallery, score probes, print similarity table."""
    def load_dir(d, label):
        files = sorted(f for f in os.listdir(d)
                       if f.lower().endswith((".jpg", ".jpeg", ".png")))
        names, embs, times = [], [], []
        ok = fail = 0
        for fname in files:
            emb, ms = extract_embedding(os.path.join(d, fname), mtcnn, model)
            if emb is not None:
                names.append(fname); embs.append(emb); times.append(ms); ok += 1
            else:
                fail += 1
        print(f"  [{label}] {ok} ok / {fail} fail  |  avg {np.mean(times):.1f} ms/img")
        return names, embs, times

    print("\nEmbedding gallery …")
    g_names, g_embs, g_times = load_dir(gallery_dir, "gallery")
    print("Embedding probes …")
    p_names, p_embs, p_times = load_dir(probe_dir,   "probes")

    if not g_embs:
        print("[ERROR] no gallery embeddings"); return

    gallery_matrix = l2_normalize(np.stack(g_embs))
    all_times = g_times + p_times

    print(f"\n  Avg inference time : {np.mean(all_times):.1f} ms / image")
    print(f"  Threshold          : {threshold}")
    print(f"\n{'='*90}")
    print(f"  {'Probe':<28}  {'Best match (gallery)':<28}  {'Max':>7}  {'Mean':>7}  Decision")
    print(f"  {'-'*86}")

    for fname, emb in zip(p_names, p_embs):
        probe_norm = l2_normalize(emb.reshape(1, -1))
        sims       = (probe_norm @ gallery_matrix.T).flatten()
        best_idx   = int(np.argmax(sims))
        max_sim    = float(sims[best_idx])
        mean_sim   = float(np.mean(sims))
        best_name  = g_names[best_idx]
        decision   = "✓ IN gallery" if max_sim >= threshold else "✗ NOT IN gallery"
        print(f"  {fname:<28}  {best_name:<28}  {max_sim:>7.4f}   {mean_sim:>7.4f}   {decision}")

    print(f"{'='*90}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="ArcFace R50 CoreML inference",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model",     default=MODEL_PATH,
                        help=f"Path to .mlpackage  (default: {MODEL_PATH})")
    parser.add_argument("--image",     default=None,
                        help="Single image path  → print embedding stats")
    parser.add_argument("--gallery",   default=None,
                        help="Gallery directory  (requires --probes)")
    parser.add_argument("--probes",    default=None,
                        help="Probes directory   (requires --gallery)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"Cosine threshold  (default={THRESHOLD})")
    args = parser.parse_args()

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading models …")
    device = torch.device("cpu")   # MTCNN on CPU; CoreML handles its own device
    mtcnn = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709,
        post_process=False, keep_all=True, device=device,
    )
    coreml_model = load_coreml_model(args.model)

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.image:
        mode_single_image(args.image, mtcnn, coreml_model)

    elif args.gallery and args.probes:
        mode_gallery_probe(args.gallery, args.probes, mtcnn, coreml_model, args.threshold)

    else:
        # Default demo: benchmark/gallery vs benchmark/known
        benchmark_dir = os.path.join(SCRIPT_DIR, "benchmark")
        gallery_dir   = os.path.join(benchmark_dir, "gallery")
        probes_dir    = os.path.join(benchmark_dir, "known")

        if os.path.isdir(gallery_dir) and os.path.isdir(probes_dir):
            print(f"\nNo --image / --gallery specified. Using default:")
            print(f"  Gallery : {gallery_dir}")
            print(f"  Probes  : {probes_dir}")
            mode_gallery_probe(gallery_dir, probes_dir, mtcnn, coreml_model, args.threshold)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
