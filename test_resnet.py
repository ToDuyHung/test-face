"""
test_pytorch.py – pure-PyTorch gallery/probe cosine similarity test

Same purpose as test.py but uses facenet_pytorch natively (no ONNX overrides):
  - Detection + alignment: facenet_pytorch.MTCNN  (PyTorch, no custom ONNX)
  - Embedding:             facenet_pytorch.InceptionResnetV1  (VGGFace2, 512-d)

Gallery:  test3/gallery/   (all images = one person)
Probes:   test3/probes/

Output: cosine-similarity table identical to test.py
"""

import os
import sys
from typing import Optional, List

import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# ── resolve paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))

GALLERY_DIR = os.path.join(SCRIPT_DIR, "test3", "gallery")
PROBES_DIR  = os.path.join(SCRIPT_DIR, "test3", "probes")
# PROBES_DIR  = os.path.join(SCRIPT_DIR, "test3", "online")

GALLERY_ID  = "1"    # all gallery images belong to person ID = 1


# ── helpers ───────────────────────────────────────────────────────────────────
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


# ── embedding extraction ──────────────────────────────────────────────────────
@torch.no_grad()
def extract_embedding(
    img_path: str,
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: torch.device,
) -> Optional[np.ndarray]:
    """
    Detect face with MTCNN, embed with InceptionResnetV1.
    Returns L2-normalised 512-d numpy array, or None if no face detected.

    MTCNN is initialised with post_process=True, so mtcnn(img) returns a
    [3,160,160] float tensor already normalised: (x - 127.5) / 128.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as ex:
        print(f"  [ERR] load failed: {img_path}: {ex}")
        return None

    # keep_all=True so we can pick the face with highest confidence
    face_tensors = mtcnn(img)   # [N,3,160,160] or None

    if face_tensors is None:
        # ── fallback: downscale to 640 px on longer side ─────────────────
        W, H = img.size
        max_side = 640
        m = max(W, H)
        if m > max_side:
            scale = max_side / m
            img_small = img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR)
            face_tensors = mtcnn(img_small)

    if face_tensors is None:
        print(f"  [WARN] No face detected: {os.path.basename(img_path)}")
        return None

    # pick best face: highest MTCNN prob (use detect() for probs, reuse crop)
    _, probs = mtcnn.detect(img)
    if probs is not None and len(probs) > 0:
        best = int(np.argmax([p if p is not None else -1 for p in probs]))
    else:
        best = 0

    if face_tensors.ndim == 3:          # single face: [3,H,W]
        face = face_tensors.unsqueeze(0)
    else:                               # multiple faces: [N,3,H,W]
        best = min(best, face_tensors.shape[0] - 1)
        face = face_tensors[best].unsqueeze(0)   # [1,3,H,W]

    face = face.to(device)
    emb  = resnet(face)[0].detach().cpu().numpy().astype(np.float32)  # (512,)
    return emb


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── load models ───────────────────────────────────────────────────────────
    print("Loading MTCNN + InceptionResnetV1 (VGGFace2) …")
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,   # normalise output for InceptionResnetV1
        keep_all=True,
        device=device,
    )
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    # ── gallery ───────────────────────────────────────────────────────────────
    gallery_files = sorted(
        f for f in os.listdir(GALLERY_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"\n─── Gallery (ID={GALLERY_ID}) ──────────────────────────────────────")
    print(f"Found {len(gallery_files)} gallery images: {gallery_files}")

    gallery_embs:  List[np.ndarray] = []
    gallery_names: List[str]        = []

    for fname in gallery_files:
        fpath = os.path.join(GALLERY_DIR, fname)
        emb   = extract_embedding(fpath, mtcnn, resnet, device)
        if emb is not None:
            gallery_embs.append(emb)
            gallery_names.append(fname)
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname}  (no face)")

    if len(gallery_embs) == 0:
        print("[ERROR] No gallery embeddings extracted. Exiting.")
        sys.exit(1)

    gallery_matrix = l2_normalize(np.stack(gallery_embs, axis=0))  # (N, 512)
    print(f"\nGallery embeddings ready: {gallery_matrix.shape[0]} / {len(gallery_files)}")

    # ── probes ────────────────────────────────────────────────────────────────
    probe_files = sorted(
        f for f in os.listdir(PROBES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"\n─── Probes ─────────────────────────────────────────────────────────")
    print(f"Found {len(probe_files)} probe images: {probe_files}")
    print()

    results = []
    for fname in probe_files:
        fpath = os.path.join(PROBES_DIR, fname)
        emb   = extract_embedding(fpath, mtcnn, resnet, device)

        if emb is None:
            print(f"  {fname}: [WARN] no face detected – skipping")
            continue

        probe_norm = l2_normalize(emb.reshape(1, -1))          # (1, 512)
        sims       = (probe_norm @ gallery_matrix.T).flatten()  # (N,)
        best_idx   = int(np.argmax(sims))
        max_sim    = float(sims[best_idx])
        mean_sim   = float(np.mean(sims))
        best_gname = gallery_names[best_idx]

        results.append((fname, best_gname, max_sim, mean_sim))

    # ── pretty-print results ──────────────────────────────────────────────────
    print("=" * 68)
    print(f"{'Probe':<22} {'Best match (gallery)':<22} {'Max sim':>8}  {'Mean sim':>8}")
    print("-" * 68)
    for fname, best_gname, max_sim, mean_sim in results:
        print(f"  {fname:<20} {best_gname:<22} {max_sim:>8.4f}   {mean_sim:>8.4f}")
    print("=" * 68)


if __name__ == "__main__":
    main()
