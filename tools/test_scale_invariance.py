"""Check the engine reads the page, not the screenshot's dimensions.

Every pixel threshold in the engine describes a feature of a rendered page -- how wide
an antialiased rim is, how large a component must be to matter, how far a shadow
reaches. If those are left absolute the engine only works near one capture size. This
resamples each reference and reports whether the reconstruction holds up.

Usage:  python tools/test_scale_invariance.py [factor ...]   (default 0.6 1.0 1.6)
"""
import glob
import os
import sys

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from engine import ImageReconstructor, HTMLRenderer, DocumentData  # noqa: E402
from _chrome import find_chrome, screenshot  # noqa: E402

OUT = os.path.join(REPO_ROOT, "scratch", "scale_test")


def score(image_path, chrome, tag):
    page = ImageReconstructor.reconstruct_image(image_path)
    doc = DocumentData(title=tag, pageCount=1, pages=[page])
    html = HTMLRenderer.render_document(doc, editable=False, title=tag, interactive=False)
    html = html.replace("padding: 24px;", "padding: 0;").replace("gap: 24px;", "gap: 0;")
    hp = os.path.abspath(os.path.join(OUT, tag + ".html"))
    open(hp, "w", encoding="utf-8").write(html)
    # Distinct from the resampled source: writing both to one path made the
    # screenshot overwrite its own input, and every page scored a perfect 100%.
    sp = os.path.abspath(os.path.join(OUT, tag + "_recon.png"))
    screenshot(chrome, hp, sp, page.width, page.height)
    a = Image.open(image_path).convert("L")
    b = Image.open(sp).convert("L")
    w, h = min(a.width, b.width), min(a.height, b.height)
    s, _ = ssim(np.array(a.crop((0, 0, w, h))), np.array(b.crop((0, 0, w, h))), full=True)
    counts = {}
    for e in page.elements:
        counts[e.type] = counts.get(e.type, 0) + 1
    return s * 100.0, counts


def main():
    factors = [float(a) for a in sys.argv[1:]] or [0.6, 1.0, 1.6]
    os.makedirs(OUT, exist_ok=True)
    chrome = find_chrome()
    rows = []
    for src in sorted(glob.glob(os.path.join(REPO_ROOT, "benchmarks", "images", "*.png"))):
        name = os.path.splitext(os.path.basename(src))[0]
        base = Image.open(src).convert("RGB")
        line = []
        for f in factors:
            if abs(f - 1.0) < 1e-6:
                path = src
            else:
                path = os.path.join(OUT, "%s@%.2f_src.png" % (name, f))
                base.resize((max(1, int(base.width * f)), max(1, int(base.height * f))),
                            Image.LANCZOS).save(path)
            s, counts = score(path, chrome, "%s@%.2f" % (name, f))
            line.append((f, s, counts.get("rect", 0), counts.get("text", 0)))
        rows.append((name, line))
        print("\n%s" % name)
        for f, s, r, t in line:
            print("   x%.2f  SSIM %6.2f%%   rect %3d  text %3d" % (f, s, r, t))
        spread = max(s for _, s, _, _ in line) - min(s for _, s, _, _ in line)
        print("   spread across scales: %.2f points" % spread)

    print("\n%-26s %s" % ("page", "  ".join("x%.2f" % f for f in factors)))
    worst = 0.0
    for name, line in rows:
        print("%-26s %s" % (name[:26], "  ".join("%5.1f" % s for _, s, _, _ in line)))
        worst = max(worst, max(s for _, s, _, _ in line) - min(s for _, s, _, _ in line))
    print("\nlargest spread across scales on any page: %.2f points" % worst)


if __name__ == "__main__":
    main()
