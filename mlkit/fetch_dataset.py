"""Download the real-screenshot UI dataset into datasets/.

Source: https://huggingface.co/datasets/YashJain/UI-Elements-Detection-Dataset
        Apache-2.0, ~900MB, 691 annotated screenshots of real websites.

Only the split folders are pulled. The repo also carries a yolo_dataset/ copy with
pre-annotated preview images, which is the same data drawn on twice over.

Usage:  python mlkit/fetch_dataset.py
"""
import argparse
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(REPO_ROOT, "datasets", "ui-elements-hf")
REPO_ID = "YashJain/UI-Elements-Detection-Dataset"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=DEFAULT_DEST)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    os.makedirs(args.dest, exist_ok=True)
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=args.dest,
        allow_patterns=["train/**", "val/**", "test/**", "dataset.yaml", "README.md"],
    )
    print("\ndownloaded to %s" % path)
    for split in ("train", "val", "test"):
        d = os.path.join(args.dest, split, "images")
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print("  %-6s %4d images" % (split, n))


if __name__ == "__main__":
    main()
