"""
test_pytorch.py – MTCNN (5-landmark alignment) + ArcFace R50 ONNX embedding

Detection + alignment : facenet_pytorch.MTCNN via detect(landmarks=True)
                        → umeyama 5-point affine warp → 160×160
Embedding             : ArcFace R50 ONNX  (InsightFace buffalo_l, 512-d)
                        Auto-downloaded from GitHub releases on first run.

Gallery:  test3/gallery/
Probes:   test3/probes/

Why ArcFace beats InceptionResnetV1 (VGGFace2)?
  - ArcFace additive margin loss → tighter intra-class, larger inter-class gap
  - Cosine threshold: same-person ~>0.35, different-person ~<0.25 (much cleaner
    separation than VGGFace2's 0.70-0.82 overlapping range)
"""

import io
import os
import sys
import zipfile
import urllib.request
from typing import Optional, List

import cv2
import numpy as np
from PIL import Image
import torch
import onnxruntime as ort
from facenet_pytorch import MTCNN

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(SCRIPT_DIR, "test3", "gallery")
PROBES_DIR  = os.path.join(SCRIPT_DIR, "test3", "probes")
# PROBES_DIR  = os.path.join(SCRIPT_DIR, "test3", "online")

GALLERY_ID  = "1"

# ── ArcFace model ─────────────────────────────────────────────────────────────
MODELS_DIR         = os.path.join(SCRIPT_DIR, "models")
ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, "w600k_r50.onnx")

# buffalo_l.zip on GitHub releases (no auth required)
BUFFALO_L_ZIP_URL = (
    "https://github.com/deepinsight/insightface"
    "/releases/download/v0.7/buffalo_l.zip"
)

# ── ArcFace reference landmarks (112×112 canonical coordinates) ───────────────
ARCFACE_REF_112 = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]],
    dtype=np.float32,
)


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────
def download_arcface() -> None:
    """Download buffalo_l.zip and extract w600k_r50.onnx if needed."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(ARCFACE_MODEL_PATH):
        return

    print(f"Downloading buffalo_l.zip from GitHub releases …")

    def _progress(count, block_size, total_size):
        pct = min(count * block_size / total_size * 100, 100) if total_size > 0 else 0
        mb  = count * block_size / 1024 / 1024
        print(f"\r  {pct:.1f}%  ({mb:.1f} MB)", end="", flush=True)

    zip_path = os.path.join(MODELS_DIR, "buffalo_l.zip")
    urllib.request.urlretrieve(BUFFALO_L_ZIP_URL, zip_path, _progress)
    print()

    print("  Extracting w600k_r50.onnx …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        target = "w600k_r50.onnx"
        # handle sub-directory inside zip  (e.g. buffalo_l/w600k_r50.onnx)
        names = [n for n in zf.namelist() if n.endswith(target)]
        if not names:
            raise RuntimeError(f"w600k_r50.onnx not found in zip. Contents: {zf.namelist()}")
        with zf.open(names[0]) as src, open(ARCFACE_MODEL_PATH, "wb") as dst:
            dst.write(src.read())

    os.remove(zip_path)
    print(f"  ✓ Saved → {ARCFACE_MODEL_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# ArcFace helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_arcface_session() -> ort.InferenceSession:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(ARCFACE_MODEL_PATH, providers=providers)
    print(f"  ArcFace loaded  |  provider: {sess.get_providers()[0]}")
    return sess


def arcface_preprocess(face_rgb_112: np.ndarray) -> np.ndarray:
    """RGB uint8 112×112 → float32 (1,3,112,112) normalised to [-1,1]."""
    img = face_rgb_112.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = img.transpose(2, 0, 1)        # HWC → CHW
    return np.expand_dims(img, 0)       # (1,3,112,112)


def run_arcface(sess: ort.InferenceSession, face_np: np.ndarray) -> np.ndarray:
    """Run ArcFace ONNX → (512,) L2-normalised embedding."""
    inp  = {sess.get_inputs()[0].name: arcface_preprocess(face_np)}
    emb  = sess.run(None, inp)[0][0]           # (512,)
    emb  = emb / (np.linalg.norm(emb) + 1e-12)
    return emb.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 5-landmark umeyama alignment  (same logic as onnx_baseline_v3)
# ─────────────────────────────────────────────────────────────────────────────
def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src, dst = src.astype(np.float64), dst.astype(np.float64)
    n = src.shape[0]
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    sd, dd = src - src_mean, dst - dst_mean
    A = (dd.T @ sd) / n
    U, S, Vt = np.linalg.svd(A)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    var = np.sum(sd ** 2) / n
    scale = np.sum(S) / (var + 1e-12)
    t = dst_mean - scale * (R @ src_mean)
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2]  = t
    return M.astype(np.float32)


def align_face(img_rgb: np.ndarray, lm5: np.ndarray, out_size: int = 112) -> np.ndarray:
    """Affine-warp img_rgb using MTCNN 5-point landmarks → out_size × out_size."""
    ref = ARCFACE_REF_112 * (out_size / 112.0)
    M   = _umeyama(lm5, ref)
    aligned = cv2.warpAffine(
        img_rgb, M, (out_size, out_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    return aligned.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Common helper
# ─────────────────────────────────────────────────────────────────────────────
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ─────────────────────────────────────────────────────────────────────────────
# Main embedding extractor
# ─────────────────────────────────────────────────────────────────────────────
def extract_embedding(
    img_path: str,
    mtcnn: MTCNN,
    arc_sess: ort.InferenceSession,
) -> Optional[np.ndarray]:
    """
    1. Load image (RGB uint8)
    2. MTCNN.detect(landmarks=True) → 5-point landmarks
    3. umeyama affine warp → 112×112 aligned face
    4. ArcFace ONNX → 512-d L2-normalised embedding

    Fallback: if MTCNN misses, try downscaling to 640px then use plain resize.
    """
    try:
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("cv2 read failed")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as ex:
        print(f"  [ERR] {os.path.basename(img_path)}: {ex}")
        return None

    print(f"  [DEBUG] {os.path.basename(img_path)} – size: {img_rgb.shape[1]}×{img_rgb.shape[0]}")

    # ── detect with MTCNN ────────────────────────────────────────────────────
    pil_img = Image.fromarray(img_rgb)
    boxes, probs, points = mtcnn.detect(pil_img, landmarks=True)

    if boxes is None:
        # Try downscale to help MTCNN
        H, W = img_rgb.shape[:2]
        if max(H, W) > 640:
            scale = 640 / max(H, W)
            small = cv2.resize(img_rgb, (int(W * scale), int(H * scale)))
            b2, p2, pts2 = mtcnn.detect(Image.fromarray(small), landmarks=True)
            if b2 is not None:
                boxes, probs, points = b2, p2, pts2 / scale

    if boxes is None or points is None:
        # Final fallback: treat whole image as face
        print(f"  [WARN] MTCNN miss → plain resize: {os.path.basename(img_path)}")
        face_112 = cv2.resize(img_rgb, (112, 112), interpolation=cv2.INTER_LINEAR)
        return run_arcface(arc_sess, face_112.astype(np.float32))

    # ── pick best face + align ────────────────────────────────────────────────
    best  = int(np.argmax([p if p is not None else -1 for p in probs]))
    lm5   = points[best].astype(np.float32)   # (5, 2)
    face_112 = align_face(img_rgb, lm5, out_size=112)

    return run_arcface(arc_sess, face_112.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── models ────────────────────────────────────────────────────────────────
    download_arcface()
    print("Loading models …")

    arc_sess = load_arcface_session()

    # MTCNN — detection + landmark only; alignment done manually via umeyama
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        keep_all=True,
        device=device,
    )

    # ── gallery ───────────────────────────────────────────────────────────────
    gallery_files = sorted(
        f for f in os.listdir(GALLERY_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"\n─── Gallery (ID={GALLERY_ID}) ───────────────────────────────────────")
    print(f"Found {len(gallery_files)} gallery images")

    gallery_embs:  List[np.ndarray] = []
    gallery_names: List[str]        = []

    for fname in gallery_files:
        emb = extract_embedding(os.path.join(GALLERY_DIR, fname), mtcnn, arc_sess)
        if emb is not None:
            gallery_embs.append(emb)
            gallery_names.append(fname)
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname}  (failed)")

    if not gallery_embs:
        print("[ERROR] No gallery embeddings. Exiting.")
        sys.exit(1)

    gallery_matrix = l2_normalize(np.stack(gallery_embs))   # (N, 512)
    print(f"\nGallery ready: {len(gallery_embs)} / {len(gallery_files)}")

    # ── probes ────────────────────────────────────────────────────────────────
    probe_files = sorted(
        f for f in os.listdir(PROBES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"\n─── Probes ──────────────────────────────────────────────────────────")
    print(f"Found {len(probe_files)} probe images: {probe_files}\n")

    THRESHOLD = 0.5

    results = []
    for fname in probe_files:
        emb = extract_embedding(os.path.join(PROBES_DIR, fname), mtcnn, arc_sess)
        if emb is None:
            print(f"  {fname}: [WARN] failed – skipping")
            continue

        probe_norm = l2_normalize(emb.reshape(1, -1))
        sims       = (probe_norm @ gallery_matrix.T).flatten()
        best_idx   = int(np.argmax(sims))
        max_sim    = float(sims[best_idx])
        # mean_sim   = float(np.mean(sims))
        std_sim    = float(np.std(sims))
        # z_score    = (max_sim - mean_sim) / (std_sim + 1e-9)
        best_name  = gallery_names[best_idx]
        decision   = "✓ IN" if max_sim >= THRESHOLD else "✗ NOT IN"

        results.append((fname, best_name, max_sim, decision))

    # ── table ─────────────────────────────────────────────────────────────────
    print(f"\nModel : ArcFace R50 (w600k_r50.onnx) | Alignment: umeyama 5-landmark")
    print(f"Threshold : cosine ≥ {THRESHOLD:.2f}  →  IN gallery")
    print("=" * 98)
    print(f"  {'Probe':<22} {'Best match (gallery)':<28} {'Max':>7}  {'Decision'}")
    print("-" * 98)
    for fname, best_name, max_sim, decision in results:
        print(
            f"  {fname:<22} {best_name:<28} "
            f"{max_sim:>7.4f}   {decision} gallery"
        )
    print("=" * 98)
    print(f"  Threshold={THRESHOLD}  |  Adjust based on FAR/FRR requirements")


if __name__ == "__main__":
    main()
