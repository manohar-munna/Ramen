"""Draw YOLO boxes onto an image so labels and predictions can be checked by eye.

Usage:  python mlkit/draw_boxes.py <image> [<labels.txt>] [-o out.png]
        With no label file, looks for the matching one under labels/<split>/.
"""
import argparse
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import CLASSES  # noqa: E402

COLOURS = [
    (60, 76, 231), (231, 145, 60), (60, 200, 120), (200, 60, 200), (40, 180, 230),
    (120, 120, 250), (30, 160, 60), (230, 90, 160), (90, 90, 90), (0, 190, 220),
]


def draw(image_path, label_path, out_path, conf_col=False):
    img = cv2.imread(image_path)
    if img is None:
        raise SystemExit("cannot read %s" % image_path)
    h, w = img.shape[:2]
    overlay = img.copy()
    rows = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
            conf = float(parts[5]) if conf_col and len(parts) > 5 else None
            x0 = int((cx - bw / 2) * w); y0 = int((cy - bh / 2) * h)
            x1 = int((cx + bw / 2) * w); y1 = int((cy + bh / 2) * h)
            rows.append((cid, x0, y0, x1, y1, conf))

    # Large boxes first so small ones stay legible on top.
    rows.sort(key=lambda r: -((r[3] - r[1]) * (r[4] - r[2])))
    for cid, x0, y0, x1, y1, conf in rows:
        col = COLOURS[cid % len(COLOURS)]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), col, 2)
        name = CLASSES[cid] if cid < len(CLASSES) else str(cid)
        label = name if conf is None else "%s %.2f" % (name, conf)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y0 - 4 if y0 > th + 6 else y1 + th + 4
        cv2.rectangle(overlay, (x0, ty - th - 3), (x0 + tw + 6, ty + 3), col, -1)
        cv2.putText(overlay, label, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, 0.85, img, 0.15, 0)
    cv2.imwrite(out_path, out)
    return len(rows), out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("labels", nargs="?")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--conf", action="store_true", help="labels carry a 6th confidence column")
    args = ap.parse_args()

    labels = args.labels
    if labels is None:
        # Accept either separator: a path typed with forward slashes on Windows still
        # has to find its label file.
        norm = args.image.replace(chr(92), "/")
        labels = norm.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    out = args.out or (os.path.splitext(args.image)[0] + "_boxes.png")
    n, path = draw(args.image, labels, out, conf_col=args.conf)
    print("%d boxes -> %s" % (n, path))


if __name__ == "__main__":
    main()
