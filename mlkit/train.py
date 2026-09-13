"""Train the UI component detector.

Starts from COCO-pretrained YOLO nano weights -- roughly 2.6M parameters, a few
milliseconds per page on GPU and still practical on CPU. Not a VLM and not a language
model: a plain convolutional detector that returns boxes, classes and confidences.

The detector is not meant to replace the reconstruction engine. The engine measures
geometry from pixels far more precisely than a detector ever will -- exact fills, corner
radii, shadows, type metrics. What the engine is weakest at is deciding *what a thing
is*, and that is what this supplies, as a second opinion carrying its own confidence.

Usage:  python mlkit/train.py --epochs 40
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(REPO_ROOT, "datasets", "prepared", "data.yaml"))
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="ui-detector")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit("no dataset at %s -- run mlkit/prepare_data.py first" % args.data)

    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=os.path.join(REPO_ROOT, "scratch", "runs"),
        name=args.name,
        exist_ok=True,
        # Pages are axis-aligned and read left-to-right: a mirrored or rotated page is
        # not a page, and colour jitter would teach the detector to ignore the very
        # contrast that separates a filled control from its ground.
        fliplr=0.0, flipud=0.0, degrees=0.0, shear=0.0, perspective=0.0,
        mosaic=0.4, scale=0.35, translate=0.06, hsv_h=0.0, hsv_s=0.25, hsv_v=0.25,
        erasing=0.0,
        patience=20,
        # Decoding 500 full-size PNGs every epoch made the GPU wait on the disk: the
        # first run spent twelve minutes an epoch at a batch this small. Caching the
        # decoded images turns the run compute-bound, where it belongs.
        # Real screenshots are 1920x1080; caching those decoded in RAM needs ~7GB, so
        # they go to disk as .npy instead. The cost being avoided is PNG decode, which
        # is what left the GPU idle for twelve minutes an epoch on the first run.
        cache="disk", workers=8,
        verbose=True,
    )
    best = os.path.join(REPO_ROOT, "scratch", "runs", args.name, "weights", "best.pt")
    print("\nbest weights: %s" % best)


if __name__ == "__main__":
    main()
