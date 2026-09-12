"""Honest fidelity evaluation for the image -> HTML path.

Runs ImageReconstructor on a source image, renders the resulting HTML in
headless Chrome at 1:1 pixel scale, and reports SSIM / pixel-diff against the
original. Writes a [original | reconstruction | heatmap] composite for eyeballing.

Usage:  python tools/eval_image.py <image> [<image> ...]
"""
import glob
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageChops
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import ImageReconstructor, HTMLRenderer, DocumentData  # noqa: E402

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# scratch/ is gitignored -- renders and composites are build output, not source.
OUT_DIR = os.path.join(REPO_ROOT, "scratch", "image_eval")


def find_chrome():
    for p in CHROME_CANDIDATES:
        if os.path.exists(p):
            return p
    raise RuntimeError("Chrome not found; checked: %s" % CHROME_CANDIDATES)


def evaluate(image_path, chrome):
    name = os.path.splitext(os.path.basename(image_path))[0].replace(" ", "_")
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Reconstruct
    page = ImageReconstructor.reconstruct_image(
        image_path, asset_dir=os.path.join(OUT_DIR, name + "_assets")
    )
    doc = DocumentData(title=name, pageCount=1, pages=[page])

    counts = {}
    for el in page.elements:
        counts[el.type] = counts.get(el.type, 0) + 1

    # 2. Render HTML with zero page chrome so the page starts at (0,0)
    # interactive=False: the demo snackbar and Inspect button are chrome for verifying
    # the reconstruction, not part of it, and would be scored as differences.
    html = HTMLRenderer.render_document(doc, editable=False, title=name, interactive=False)
    html = html.replace("padding: 24px;", "padding: 0;").replace("gap: 24px;", "gap: 0;")
    html_path = os.path.abspath(os.path.join(OUT_DIR, name + ".html"))
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 3. Screenshot at 1:1 (image px == CSS px for the image path)
    shot_path = os.path.abspath(os.path.join(OUT_DIR, name + "_recon.png"))
    subprocess.run([
        chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
        "--force-device-scale-factor=1",
        "--window-size=%d,%d" % (int(page.width), int(page.height)),
        "--screenshot=" + shot_path,
        "file:///" + html_path.replace("\\", "/"),
    ], check=True, capture_output=True)

    # 4. Compare
    im_orig = Image.open(image_path).convert("RGB")
    im_recon = Image.open(shot_path).convert("RGB")
    w = min(im_orig.width, im_recon.width)
    h = min(im_orig.height, im_recon.height)
    im_orig = im_orig.crop((0, 0, w, h))
    im_recon = im_recon.crop((0, 0, w, h))

    a = np.array(im_orig.convert("L"))
    b = np.array(im_recon.convert("L"))
    score, _ = ssim(a, b, full=True)

    diff = np.array(ImageChops.difference(im_orig, im_recon))
    gray_diff = diff.mean(axis=2)
    pix_diff_pct = float((gray_diff > 30).sum()) / (w * h) * 100.0

    # 5. Composite [original | recon | heatmap]
    heat = np.zeros((h, w, 3), dtype=np.uint8)
    heat[:, :, 2] = 60
    mask = gray_diff > 15
    heat[mask, 0] = np.clip(gray_diff[mask] * 2, 0, 255)
    heat[mask, 1] = np.clip(gray_diff[mask] * 3, 0, 255)
    comp = Image.new("RGB", (w * 3 + 40, h), (255, 255, 255))
    comp.paste(im_orig, (0, 0))
    comp.paste(im_recon, (w + 20, 0))
    comp.paste(Image.fromarray(heat), (w * 2 + 40, 0))
    comp_path = os.path.join(OUT_DIR, name + "_comparison.png")
    comp.save(comp_path)

    return {
        "name": name, "size": "%dx%d" % (w, h), "ssim": score * 100.0,
        "pix_diff": pix_diff_pct, "elements": len(page.elements),
        "counts": counts, "comparison": comp_path, "recon": shot_path,
    }


def main():
    targets = sys.argv[1:] or sorted(
        glob.glob(os.path.join(REPO_ROOT, "benchmarks", "images", "*.png")))
    chrome = find_chrome()
    results = []
    for t in targets:
        if not os.path.exists(t):
            print("SKIP (missing): %s" % t)
            continue
        r = evaluate(t, chrome)
        results.append(r)
        print("\n%s  [%s]" % (r["name"], r["size"]))
        print("  SSIM              : %.2f%%" % r["ssim"])
        print("  pixels differing  : %.2f%%" % r["pix_diff"])
        print("  elements emitted  : %d  %s" % (r["elements"], r["counts"]))
        print("  composite         : %s" % r["comparison"])
    if len(results) > 1:
        print("\nMean SSIM across %d images: %.2f%%"
              % (len(results), np.mean([r["ssim"] for r in results])))


if __name__ == "__main__":
    main()
