"""Does the detector actually improve the reconstruction? Look, then score.

Runs the engine alone and the engine with the detector fused in, on the pages that have
ground truth, and reports what the detector changed. The annotated images it writes are
the point; the counts are there to say where to look.

Usage:  python mlkit/compare_hybrid.py [image ...]
"""
import argparse
import glob
import json
import os
import sys

import cv2

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import ImageReconstructor  # noqa: E402
from detect import UIDetector, fuse  # noqa: E402

TRUTH_DIR = os.path.join(REPO_ROOT, "benchmarks", "ground_truth")
OUT_DIR = os.path.join(REPO_ROOT, "scratch", "hybrid")
ACCEPT = {"a": {"a"}, "button": {"button"}, "input": {"input"}, "card": {"div"}}


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)


def score(page, spec):
    hit = 0
    for comp in spec["components"]:
        best, best_iou = None, 0.0
        for e in page.elements:
            v = iou(comp["bbox"], e.bbox)
            if v > best_iou:
                best, best_iou = e, v
        if best is None or best_iou < 0.35:
            continue
        want = comp["tag"]
        ok = (best.role == "card") if want == "card" else ((best.tag or "") in ACCEPT.get(want, {want}))
        hit += 1 if ok else 0
    return hit, len(spec["components"])


def annotate(img_path, page, out_path):
    img = cv2.imread(img_path)
    cols = {"button": (60, 76, 231), "a": (231, 145, 60), "input": (60, 200, 120)}
    for e in page.elements:
        if e.role not in ("button", "link", "input", "navlink", "listitem", "badge"):
            continue
        tag = e.tag or ""
        col = cols.get(tag, (150, 150, 150))
        x0, y0, x1, y1 = [int(v) for v in e.bbox]
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 2)
        lab = "%s %.2f" % (tag, e.confidence or 0)
        (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = y0 - 4 if y0 > th + 6 else y1 + th + 4
        cv2.rectangle(img, (x0, ty - th - 3), (x0 + tw + 6, ty + 3), col, -1)
        cv2.putText(img, lab, (x0 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    targets = args.images or sorted(glob.glob(os.path.join(REPO_ROOT, "benchmarks", "images", "*.png")))
    det = UIDetector(conf=args.conf)
    if not det.available:
        raise SystemExit("no weights at %s -- run mlkit/train.py" % det.weights)

    for img_path in targets:
        name = os.path.splitext(os.path.basename(img_path))[0]
        page = ImageReconstructor.reconstruct_image(img_path)
        truth_path = os.path.join(TRUTH_DIR, name + ".json")
        spec = json.load(open(truth_path, encoding="utf-8")) if os.path.exists(truth_path) else None

        before = score(page, spec) if spec else None
        annotate(img_path, page, os.path.join(OUT_DIR, name + "_engine.png"))

        report = fuse(page.elements, det.detect(img_path))
        after = score(page, spec) if spec else None
        annotate(img_path, page, os.path.join(OUT_DIR, name + "_hybrid.png"))

        print("\n%s" % name)
        if spec:
            print("  components correct: %d/%d  ->  %d/%d" % (before[0], before[1], after[0], after[1]))
        print("  detector confirmed %d, re-labelled %d, saw %d the engine did not"
              % (len(report["confirmed"]), len(report["changed"]), len(report["unmatched"])))
        for eid, was, now, c in report["changed"][:6]:
            print("      %-12s %s -> %s (%.2f)" % (eid, was, now, c))
        print("  images -> %s_{engine,hybrid}.png" % os.path.join(OUT_DIR, name))


if __name__ == "__main__":
    main()
