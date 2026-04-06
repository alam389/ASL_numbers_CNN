"""Import the ASL-HG classes (digits 0-9 and letters A-Z) into data/ and Test_Data/.

Download "ASL_HG_36000" from https://data.mendeley.com/datasets/j4y5w2c8w9/1, then:

    python prepare_hg.py path/to/ASL_Raw_Images.zip          # or the outer dataset zip
    python prepare_hg.py path/to/zip --replace                # clear data/ and Test_Data/ first
    python prepare_hg.py path/to/zip --replace --digits-only  # only 0-9

Use the RAW images, not the processed ones (those are already background-segmented).
A random --test-fraction of the new images goes to Test_Data/ (never trained on), so test.py
measures accuracy on real backgrounds.
"""
import argparse
import io
import os
import random
import re
import shutil
import zipfile

import settings

DIGITS = {str(i) for i in range(10)}
LETTERS = {chr(c) for c in range(ord("A"), ord("Z") + 1)}


def _open_zips(path):
    """Yield the zip plus any zips nested inside it (the dataset ships zips inside a zip)."""
    z = zipfile.ZipFile(path)
    yield z
    for name in z.namelist():
        # skip the background-segmented "processed" zip without inflating it
        if name.lower().endswith(".zip") and "process" not in name.lower():
            yield from _open_zips(io.BytesIO(z.read(name)))


def _class_of(member, classes):
    """Class folder for a member path like '.../Raw/7/img.jpg', else None."""
    parts = [p for p in re.split(r"[\\/]", member) if p]
    if (len(parts) < 2 or not member.lower().endswith(settings.IMAGE_EXTENSIONS)
            or "__MACOSX" in parts or "process" in member.lower()):
        return None
    folder = parts[-2]
    return folder if folder in classes else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip_path")
    ap.add_argument("--replace", action="store_true", help="remove existing images from data/ and Test_Data/ first")
    ap.add_argument("--digits-only", action="store_true", help="skip the A-Z folders")
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    classes = DIGITS if args.digits_only else DIGITS | LETTERS
    found = {}
    for z in _open_zips(args.zip_path):
        for m in z.namelist():
            d = _class_of(m, classes)
            if d is not None:
                found.setdefault(d, []).append((z, m))
    if set(found) != classes:
        raise SystemExit(f"Expected class folders {sorted(classes)}, found {sorted(found)}. "
                         "Is this the raw-images zip? Check the folder names inside it.")

    if args.replace:
        for d in (settings.DATA_DIR, settings.EVAL_DIR):
            if os.path.isdir(d):
                shutil.rmtree(d)
    rng = random.Random(args.seed)
    n_total = n_test_total = 0
    for d, items in sorted(found.items()):
        rng.shuffle(items)
        n_test = int(len(items) * args.test_fraction)
        for split_dir in (settings.EVAL_DIR, settings.DATA_DIR):
            os.makedirs(os.path.join(split_dir, d), exist_ok=True)
        for i, (z, m) in enumerate(items):
            out_dir = os.path.join(settings.EVAL_DIR if i < n_test else settings.DATA_DIR, d)
            ext = os.path.splitext(m)[1].lower()
            with open(os.path.join(out_dir, f"hg_{d}_{i:05d}{ext}"), "wb") as f:
                f.write(z.read(m))
        n_total += len(items)
        n_test_total += n_test
        print(f"class {d}: {len(items)} images ({n_test} to test)")
    print(f"Wrote {n_total - n_test_total} to {settings.DATA_DIR}/ and {n_test_total} to {settings.EVAL_DIR}/")
    print("Crop caches rebuild automatically on the next training run.")


if __name__ == "__main__":
    main()
