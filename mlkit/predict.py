"""Run the trained detector on an image and draw what it found.

The point is to look at the result, not to read a metric. mAP can sit high while the
detector misses every button on the one page that matters, so this writes an annotated
image and prints what was detected, and the judgement is made by eye.

Usage:  python mlkit/predict.py <image> [--weights path] [--conf 0.25]
"""
import argparse
import os
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from draw_boxes import draw  # noqa: E402
from taxonomy import CLASSES  # noqa: E402

DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "scratch", "runs", "ui-detector", "weights", "best.pt")


def detect(image_path, weights, conf, imgsz):
    from ultralytics import YOLO
    model = YOLO(weights)
    res = model.predict(image_path, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for b in res.boxes:
        cid = int(b.cls.item())
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
        out.append({"cls": cid, "name": CLASSES[cid] if cid < len(CLASSES) else str(cid),
                    "conf": float(b.conf.item()), "bbox": [x0, y0, x1, y1]})
    return out, res.orig_shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        raise SystemExit("no weights at %s -- run mlkit/train.py first" % args.weights)

    dets, (h, w) = detect(args.image, args.weights, args.conf, args.imgsz)
    print("%d detections on %dx%d  (conf >= %.2f)" % (len(dets), w, h, args.conf))
    for name, n in Counter(d["name"] for d in dets).most_common():
        confs = [d["conf"] for d in dets if d["name"] == name]
        print("  %-8s %3d   conf %.2f-%.2f" % (name, n, min(confs), max(confs)))

    scratch = os.path.join(REPO_ROOT, "scratch")
    os.makedirs(scratch, exist_ok=True)
    tmp = os.path.join(scratch, "_pred.txt")
    with open(tmp, "w", encoding="utf-8") as f:
        for d in dets:
            x0, y0, x1, y1 = d["bbox"]
            f.write("%d %.6f %.6f %.6f %.6f %.4f\n" % (
                d["cls"], ((x0 + x1) / 2) / w, ((y0 + y1) / 2) / h,
                (x1 - x0) / w, (y1 - y0) / h, d["conf"]))

    out = args.out or os.path.join(
        scratch, os.path.splitext(os.path.basename(args.image))[0] + "_pred.png")
    n, path = draw(args.image, tmp, out, conf_col=True)
    print("\nannotated -> %s" % path)


if __name__ == "__main__":
    main()
