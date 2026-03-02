"""
convert_to_coreml.py – ArcFace R50 ONNX → CoreML .mlpackage

Route: ONNX → PyTorch (via onnx2torch) → TorchScript trace → CoreML

coremltools v7+ dropped direct ONNX support; we bridge through PyTorch.

Requirements:
    pip install coremltools onnx onnx2torch

Input  : models/w600k_r50.onnx
Output : models/arcface_r50.mlpackage

Preprocessing baked into the CoreML model:
    pixel_normalised = (pixel_uint8 / 127.5) - 1.0   → range [-1, 1]
    Input layout     : NCHW float32  (1, 3, 112, 112)

Output : 512-d float embedding (NOT L2-normalised – normalise before cosine match)

Usage:
    python convert_to_coreml.py
    python convert_to_coreml.py --precision float16   # smaller/faster on Apple Silicon
    python convert_to_coreml.py --onnx models/w600k_r50.onnx --output models/arcface_r50.mlpackage
"""

import argparse
import os
import sys

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH   = os.path.join(SCRIPT_DIR, "models", "w600k_r50.onnx")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "models", "arcface_r50.mlpackage")

INPUT_H = INPUT_W = 112


def check_imports():
    missing = []
    for pkg, install in [("coremltools", "coremltools"),
                          ("onnx",        "onnx"),
                          ("onnx2torch",  "onnx2torch")]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(install)
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"  Install with:  pip install {' '.join(missing)}")
        sys.exit(1)


def convert(onnx_path: str, output_path: str, precision: str = "float32") -> None:
    import onnx
    import onnx2torch
    import torch
    import coremltools as ct

    if not os.path.exists(onnx_path):
        print(f"[ERROR] ONNX model not found: {onnx_path}")
        print("  Download it by running:  python test_pytorch.py")
        sys.exit(1)

    # ── Step 1: ONNX → PyTorch ────────────────────────────────────────────────
    print(f"Step 1/3  Loading ONNX …  ({onnx_path})")
    onnx_model  = onnx.load(onnx_path)
    torch_model = onnx2torch.convert(onnx_model)
    torch_model.eval()

    # ── Step 2: TorchScript trace ─────────────────────────────────────────────
    print("Step 2/3  TorchScript tracing …")
    dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, dtype=torch.float32)
    with torch.no_grad():
        traced = torch.jit.trace(torch_model, dummy)

    # ── Step 3: TorchScript → CoreML ─────────────────────────────────────────
    print(f"Step 3/3  Converting to CoreML (precision={precision}) …")
    compute_precision = (
        ct.precision.FLOAT16 if precision == "float16" else ct.precision.FLOAT32
    )

    # Bake normalisation into the CoreML model:
    #   normalised = (raw_uint8_pixel / 127.5) - 1.0
    #   scale = 1/127.5,  bias (per channel) = -1.0
    image_input = ct.ImageType(
        name="input",
        shape=(1, 3, INPUT_H, INPUT_W),
        scale=1.0 / 127.5,
        bias=[-1.0, -1.0, -1.0],
        color_layout=ct.colorlayout.RGB,
        channel_first=True,
    )

    mlmodel = ct.convert(
        traced,
        inputs=[image_input],
        compute_precision=compute_precision,
        minimum_deployment_target=ct.target.iOS16,
        convert_to="mlprogram",
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    mlmodel.short_description = "ArcFace R50 face embedding (InsightFace buffalo_l)"
    mlmodel.author            = "deepinsight (converted via onnx2torch + coremltools)"
    mlmodel.version           = "1.0"
    mlmodel.input_description["input"] = (
        "Aligned face crop, RGB uint8, 112×112. "
        "Preprocessing (÷127.5 - 1) is baked into the model."
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    mlmodel.save(output_path)

    size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(output_path)
        for f in files
    ) / 1024 / 1024

    print(f"\n✓ Saved → {output_path}  ({size_mb:.1f} MB)")
    print(f"\nModel spec:")
    print(f"  Input     : RGB image 112×112  (preprocessing baked-in)")
    print(f"  Output    : 512-d float embedding  (L2-normalise before cosine match)")
    print(f"  Precision : {precision}")
    print(f"\nRun inference:  python infer_coreml.py --image <path>")


def main():
    parser = argparse.ArgumentParser(
        description="Convert ArcFace R50 ONNX to CoreML .mlpackage  "
                    "(via onnx2torch → TorchScript → coremltools)"
    )
    parser.add_argument("--onnx",      default=ONNX_PATH,   help="Source .onnx path")
    parser.add_argument("--output",    default=OUTPUT_PATH,  help="Output .mlpackage path")
    parser.add_argument("--precision", default="float32",    choices=["float32", "float16"],
                        help="float16 = smaller/faster on Apple Silicon (default: float32)")
    args = parser.parse_args()

    check_imports()
    convert(args.onnx, args.output, args.precision)


if __name__ == "__main__":
    main()
