"""Combine the detector's opinion with the engine's geometry.

The two are good at different halves of the job, and the split is measured rather than
assumed.

The detector decides *what an interactive thing is*. On held-out real pages it reaches
mAP50 0.47 on buttons and 0.41 on links, where the hand-written scoring in the engine was
working from rules like "a filled rounded box with a centred label" and missing anything
that did not look like that.

The engine decides *where everything is*. It measures a box to the pixel along with its
fill, corner radius, border and shadow; a detector's box is approximate by construction
and would throw all of that away.

So a detection does not replace an engine surface -- it re-labels one, keeping the
engine's geometry, and carries a confidence combining both opinions. A detection with no
surface under it becomes a candidate in its own right, because the engine missing a
control entirely is the failure this is meant to fix.
"""
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import CLASSES  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "models", "ui-detector.pt")

# Only the classes the detector has real supervision for. Everything else it was trained
# on came from generated pages and does not survive contact with a real screenshot.
TRUSTED = {"button", "link", "input"}

# How far a detection may sit from a surface and still be talking about it. Detector
# boxes are loose, so this is deliberately more forgiving than an equality test.
MATCH_IOU = 0.45


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


class UIDetector:
    """Lazily-loaded wrapper so importing this module costs nothing without weights."""

    def __init__(self, weights: Optional[str] = None, conf: float = 0.25, imgsz: int = 960):
        self.weights = weights or DEFAULT_WEIGHTS
        self.conf = conf
        self.imgsz = imgsz
        self._model = None

    @property
    def available(self) -> bool:
        return os.path.exists(self.weights)

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, image) -> List[Dict[str, Any]]:
        """Boxes in page pixels, filtered to the classes the detector actually knows."""
        if not self.available:
            return []
        res = self._load().predict(image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        out = []
        for b in res.boxes:
            cid = int(b.cls.item())
            name = CLASSES[cid] if cid < len(CLASSES) else None
            if name not in TRUSTED:
                continue
            x0, y0, x1, y1 = [float(v) for v in b.xyxy[0].tolist()]
            out.append({"role": name, "conf": float(b.conf.item()), "bbox": [x0, y0, x1, y1]})
        return out


def fuse(elements, detections, iou_threshold: float = MATCH_IOU):
    """Re-label engine elements from detections, in place, and report what changed.

    Geometry is never taken from a detection. Where the two agree the confidence rises;
    where they disagree the detector wins on interactive classes, because that is the
    half it was measured to be better at. Anything the detector saw and the engine did
    not is returned separately rather than silently invented as an element -- the caller
    decides whether a box with no measured geometry is worth emitting.
    """
    changed, confirmed, unmatched = [], [], []
    claimed = set()
    TAG = {"button": "button", "link": "a", "input": "input"}

    for det in sorted(detections, key=lambda d: -d["conf"]):
        # A control is not always a filled box. A nav link is bare text on the page, so
        # the engine has measured its glyphs and nothing else; matching only against
        # surfaces threw away every detection of one. Text elements are candidates too,
        # and their geometry is the most precise thing on the page.
        best, best_iou = None, 0.0
        for el in elements:
            if id(el) in claimed or el.type not in ("rect", "text"):
                continue
            if el.type == "text" and det["role"] == "input":
                continue                      # a field is a box, not a run of glyphs
            v = _iou(det["bbox"], el.bbox)
            if el.type == "text":
                # A detector box around a link is drawn generously around its text, so
                # containment counts for more here than a strict overlap.
                bx = el.bbox
                inside = (bx[0] >= det["bbox"][0] - 6 and bx[1] >= det["bbox"][1] - 6
                          and bx[2] <= det["bbox"][2] + 6 and bx[3] <= det["bbox"][3] + 6)
                if inside:
                    v = max(v, 0.6)
            if v > best_iou:
                best, best_iou = el, v
        if best is None or best_iou < iou_threshold:
            unmatched.append(det)
            continue

        claimed.add(id(best))
        prior = float(best.confidence or 0.5)
        already = (best.role == det["role"]
                   or (det["role"] == "link" and best.role in ("navlink", "listitem"))
                   or (det["role"] == "button" and best.role in ("badge", "icon-button")))
        if already:
            # Two independent methods agreeing is worth more than either alone.
            best.confidence = round(min(0.99, prior + (1.0 - prior) * det["conf"]), 3)
            confirmed.append((best.id, det["role"], best.confidence))
        else:
            was = best.role
            best.role = det["role"]
            best.tag = TAG[det["role"]]
            best.confidence = round(det["conf"], 3)
            changed.append((best.id, was, det["role"], best.confidence))

    return {"changed": changed, "confirmed": confirmed, "unmatched": unmatched}
