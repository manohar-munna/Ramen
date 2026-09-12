"""Component-level scoreboard for the image -> HTML path.

SSIM measures whether the page *looks* right; it says nothing about whether a button is
a button. This scores the reconstruction against a hand-read list of what the source
image actually contains, which is the number worth optimising.

Usage:  python tools/audit_components.py [<image> ...]
        (defaults to every ground-truth file in tools/ground_truth/)
"""
import glob
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import ImageReconstructor  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUTH_DIR = os.path.join(REPO_ROOT, "benchmarks", "ground_truth")
MATCH_IOU = 0.35            # below this, nothing meaningful was produced there

# A ground-truth tag is satisfied by any of these emitted tags.
ACCEPT = {
    "a": {"a"},
    "button": {"button"},
    "input": {"input"},
    "card": {"div"},         # a card is a div; the role is what distinguishes it
}


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / float(union) if union > 0 else 0.0


def audit(truth_path, verbose=True):
    spec = json.load(open(truth_path, encoding="utf-8"))
    img = os.path.join(REPO_ROOT, spec["image"])
    if not os.path.exists(img):
        print("SKIP (image missing): %s" % spec["image"])
        return None

    page = ImageReconstructor.reconstruct_image(img)
    if verbose:
        print("\n%s  (%dx%d)" % (spec["image"], page.width, page.height))
        print("  emitted: %s" % dict(Counter(e.type for e in page.elements)))
        print("  roles  : %s\n" % dict(Counter(e.role for e in page.elements).most_common(8)))

    hit = wrong = miss = 0
    for comp in spec["components"]:
        box, want = comp["bbox"], comp["tag"]
        best, best_iou = None, 0.0
        for e in page.elements:
            v = iou(box, e.bbox)
            if v > best_iou:
                best, best_iou = e, v

        if best is None or best_iou < MATCH_IOU:
            miss += 1
            mark, got, role = "MISS", "NOTHING", "-"
        else:
            got = best.tag or "-"
            role = best.role or "-"
            ok = got in ACCEPT.get(want, {want})
            if want == "card":
                ok = role == "card"
            if ok:
                hit += 1
                mark = "ok"
            else:
                wrong += 1
                mark = "WRONG"
        if verbose:
            print("  %-5s %-18s want=<%-6s> got=<%-7s> role=%-11s iou=%.2f"
                  % (mark, comp["name"], want, got, role, best_iou))

    total = len(spec["components"])
    area = float(page.width * page.height)
    art_frags = [e for e in page.elements if e.role == "artwork"]
    art = sum((e.bbox[2]-e.bbox[0])*(e.bbox[3]-e.bbox[1]) for e in art_frags)
    result = {
        "image": spec["image"], "correct": hit, "wrong": wrong, "missing": miss,
        "total": total, "raster_pct": art / area * 100.0, "fragments": len(art_frags),
    }
    if verbose:
        print("\n  correct %d/%d (%.0f%%)   wrong %d   missing %d"
              % (hit, total, hit / total * 100.0, wrong, miss))
        print("  raster artwork covers %.0f%% of the page in %d fragments"
              % (result["raster_pct"], result["fragments"]))
    return result


def main():
    args = sys.argv[1:]
    if args:
        paths = [os.path.join(TRUTH_DIR, os.path.splitext(os.path.basename(a))[0] + ".json")
                 for a in args]
    else:
        paths = sorted(glob.glob(os.path.join(TRUTH_DIR, "*.json")))
    if not paths:
        print("No ground-truth files in %s" % TRUTH_DIR)
        return

    results = [r for r in (audit(p) for p in paths if os.path.exists(p)) if r]
    if len(results) > 1:
        c = sum(r["correct"] for r in results)
        t = sum(r["total"] for r in results)
        print("\nOVERALL correct %d/%d (%.0f%%)" % (c, t, c / t * 100.0))


if __name__ == "__main__":
    main()
