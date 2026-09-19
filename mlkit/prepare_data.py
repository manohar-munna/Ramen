"""Merge the real-screenshot dataset and the generated pages into one training set.

Neither source is used raw. The HuggingFace labels come from the DOM, which means they
carry the DOM's shape rather than the page's: an anchor wrapping a button yields two
boxes on the same pixels, a handler bound to an empty element yields a box on nothing,
and a class like `clickable` lands on top of whatever it decorates. Those are cleaned
here rather than left for the model to average out.

The generated pages contribute only the structural classes. Letting them also supervise
buttons and links would drown a few hundred real examples under thousands of synthetic
ones and teach the generator's idiom straight back to the model.

Usage:  python mlkit/prepare_data.py
"""
import argparse
import os
import shutil
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import CLASSES, GEN_CLASSES, hf_id_to_ours  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_DIR = os.path.join(REPO_ROOT, "datasets", "ui-elements-hf")
GEN_DIR = os.path.join(REPO_ROOT, "datasets", "generated-pages")
OUT_DIR = os.path.join(REPO_ROOT, "datasets", "prepared")

MIN_SIDE = 0.004        # a box thinner than this on a 1920px page is not an element
MIN_AREA = 0.00004
DUP_IOU = 0.80          # same class, this much overlap: one annotation seen twice


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / (a[3] * a[4] + b[3] * b[4] - inter)


def clean(boxes):
    """Drop degenerate boxes and collapse the same annotation seen twice."""
    kept = []
    for b in sorted(boxes, key=lambda b: -(b[3] * b[4])):     # largest first
        if b[3] < MIN_SIDE or b[4] < MIN_SIDE or b[3] * b[4] < MIN_AREA:
            continue
        if any(o[0] == b[0] and iou(o, b) > DUP_IOU for o in kept):
            continue
        kept.append(b)
    return kept


def read_label(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) >= 5:
                out.append([int(float(p[0]))] + [float(v) for v in p[1:5]])
    return out


def write_label(path, boxes):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for c, cx, cy, w, h in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n" % (c, cx, cy, w, h))


def ingest(src_root, split_map, out_root, remap, keep, tally, prefix):
    """Copy one source into the merged set, remapping and filtering as it goes."""
    n_img = 0
    for src_split, dst_split in split_map.items():
        # The two sources nest split and kind the opposite way round: the downloaded
        # set is train/images, the generated one images/train. Accept either.
        img_dir = os.path.join(src_root, src_split, "images")
        lbl_dir = os.path.join(src_root, src_split, "labels")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(src_root, "images", src_split)
            lbl_dir = os.path.join(src_root, "labels", src_split)
        if not os.path.isdir(img_dir):
            continue
        out_img = os.path.join(out_root, "images", dst_split)
        out_lbl = os.path.join(out_root, "labels", dst_split)
        os.makedirs(out_img, exist_ok=True)
        os.makedirs(out_lbl, exist_ok=True)

        for fn in sorted(os.listdir(img_dir)):
            stem, ext = os.path.splitext(fn)
            lp = os.path.join(lbl_dir, stem + ".txt")
            if ext.lower() not in (".png", ".jpg", ".jpeg") or not os.path.exists(lp):
                continue
            boxes = []
            for c, cx, cy, w, h in read_label(lp):
                ours = remap(c)
                if ours is None or CLASSES[ours] not in keep:
                    continue
                boxes.append([ours, cx, cy, w, h])
            boxes = clean(boxes)
            if not boxes:
                continue
            dst_stem = "%s_%s" % (prefix, stem)
            shutil.copyfile(os.path.join(img_dir, fn), os.path.join(out_img, dst_stem + ext))
            write_label(os.path.join(out_lbl, dst_stem + ".txt"), boxes)
            for b in boxes:
                tally[CLASSES[b[0]]] += 1
            n_img += 1
    return n_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default=HF_DIR)
    ap.add_argument("--generated", default=GEN_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--with-generated", action="store_true",
                    help="also train on generated pages (see note in main)")
    args = ap.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    tally = Counter()

    n_hf = 0
    if os.path.isdir(args.hf):
        # Its own test split becomes more validation: a held-out real page is worth more
        # than another hundred training rows.
        n_hf = ingest(args.hf, {"train": "train", "val": "val", "test": "val"},
                      args.out, hf_id_to_ours,
                      keep={"button", "link", "input", "icon", "image", "text"},
                      tally=tally, prefix="real")
        print("real screenshots : %d images" % n_hf)
    else:
        print("real screenshots : MISSING -- run mlkit/fetch_dataset.py")

    # Generated pages are off by default, and the reason is worth recording. Trained
    # alongside the real screenshots they scored 0.995 on their own validation split and
    # produced, on a real page, not one heading, card or image at any confidence down to
    # 0.08. The two domains are trivially separable, so the model learned to tell them
    # apart and applied a different prior to each: find structure on a generated page,
    # find only controls on a real one. Mixing them bought nothing and cost a shortcut.
    # Pass --with-generated to reproduce that experiment.
    n_gen = 0
    if args.with_generated and os.path.isdir(args.generated):
        n_gen = ingest(args.generated, {"train": "train", "val": "val"},
                       args.out, lambda c: c,          # already our ids
                       keep=set(GEN_CLASSES), tally=tally, prefix="gen")
        print("generated pages  : %d images" % n_gen)
    elif args.with_generated:
        print("generated pages  : MISSING -- run mlkit/generate_pages.py")
    else:
        print("generated pages  : skipped (--with-generated to include)")

    if not (n_hf or n_gen):
        raise SystemExit("nothing to prepare")

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\n\nnames:\n"
                % args.out.replace("\\", "/"))
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))

    print("\nboxes per class:")
    for c in CLASSES:
        print("  %-8s %6d" % (c, tally[c]))
    print("\nprepared -> %s" % args.out)


if __name__ == "__main__":
    main()
