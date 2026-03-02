"""
prepare_benchmark.py – Split test3/gallery into 3 benchmark sets

Source  : test3/gallery/   (67 persons × 5 poses each = 335 images)
          Filename pattern: FID{ID}_{pose}.jpg  (poses: front, bottom, left, right, top)

Output  : benchmark/
          ├── gallery/    – 1 enrolled pose per person  (front)       → 47 persons
          ├── known/      – remaining 4 poses per enrolled person      → 47 × 4 = 188 images
          └── strange/    – all poses of un-enrolled persons           → 20 × 5 = 100 images

Split   : 70 % enrolled (47 persons) / 30 % strangers (20 persons)  — deterministic seed=42

Run once, then run benchmark.py to evaluate.
"""

import os
import re
import shutil
import random
from collections import defaultdict

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR  = os.path.join(SCRIPT_DIR, "test3", "gallery")
BENCH_DIR   = os.path.join(SCRIPT_DIR, "benchmark")

GALLERY_DIR = os.path.join(BENCH_DIR, "gallery")
KNOWN_DIR   = os.path.join(BENCH_DIR, "known")
STRANGE_DIR = os.path.join(BENCH_DIR, "strange")

# ── config ────────────────────────────────────────────────────────────────────
GALLERY_POSE    = "front"       # pose used to enroll in gallery
ENROLLED_RATIO  = 0.70          # fraction of persons to enroll
SEED            = 42


def parse_gallery(source_dir: str) -> dict[str, list[str]]:
    """Return {person_id: [filenames]} from source directory."""
    person_files: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(r"^(FID\d+)_(\w+)\.(jpg|jpeg|png)$", re.IGNORECASE)
    for fname in sorted(os.listdir(source_dir)):
        m = pattern.match(fname)
        if m:
            person_files[m.group(1)].append(fname)
    return dict(person_files)


def main():
    # ── recreate output dirs ──────────────────────────────────────────────────
    for d in [GALLERY_DIR, KNOWN_DIR, STRANGE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # ── parse source ──────────────────────────────────────────────────────────
    person_files = parse_gallery(SOURCE_DIR)
    all_ids      = sorted(person_files.keys())
    n_total      = len(all_ids)
    n_enrolled   = max(1, int(round(n_total * ENROLLED_RATIO)))
    n_strangers  = n_total - n_enrolled

    rng = random.Random(SEED)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    enrolled_ids = set(shuffled[:n_enrolled])
    stranger_ids = set(shuffled[n_enrolled:])

    print(f"Total persons     : {n_total}")
    print(f"Enrolled (gallery): {n_enrolled}")
    print(f"Strangers         : {n_strangers}")

    g_count = k_count = s_count = 0

    # ── enrolled: split front → gallery, rest → known ─────────────────────────
    for pid in sorted(enrolled_ids):
        for fname in person_files[pid]:
            pose = fname.split("_")[1].split(".")[0].lower()  # e.g. "front"
            src  = os.path.join(SOURCE_DIR, fname)
            if pose == GALLERY_POSE:
                shutil.copy2(src, os.path.join(GALLERY_DIR, fname))
                g_count += 1
            else:
                shutil.copy2(src, os.path.join(KNOWN_DIR, fname))
                k_count += 1

    # ── strangers: all poses → strange ───────────────────────────────────────
    for pid in sorted(stranger_ids):
        for fname in person_files[pid]:
            shutil.copy2(
                os.path.join(SOURCE_DIR, fname),
                os.path.join(STRANGE_DIR, fname),
            )
            s_count += 1

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\nbenchmark/")
    print(f"  gallery/ : {g_count:3d} images   ({n_enrolled} persons × 1 pose '{GALLERY_POSE}')")
    print(f"  known/   : {k_count:3d} images   ({n_enrolled} persons × up-to-4 other poses)")
    print(f"  strange/ : {s_count:3d} images   ({n_strangers} persons × 5 poses)")
    print(f"\nDone. Now run:  python benchmark.py")


if __name__ == "__main__":
    main()
