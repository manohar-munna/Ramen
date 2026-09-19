"""Generate labelled training pages for the UI component detector.

These are not synthetic images standing in for real ones. The input domain is
"screenshot of a rendered web page", and that is exactly what this produces: real HTML
laid out by a real browser and captured the same way a user's screenshot is. What makes
it useful as training data is that the labels come from the DOM, so every box is exact
and free, where hand-labelling real screenshots costs minutes each and is never as
precise.

The risk this does carry is distribution, not realism: pages are only as varied as the
generator makes them, so a model trained here can learn the generator's habits. That is
why layout, palette, type, spacing, radius and density are all randomised, and why the
held-out check is always a real screenshot the generator never saw.

Usage:  python mlkit/generate_dataset.py --train 400 --val 60
"""
import argparse
import base64
import io
import json
import os
import re
import subprocess
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taxonomy import CLASSES, CLASS_ID  # noqa: E402

FONTS = [
    "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif",
    "Georgia, 'Times New Roman', serif",
    "'Trebuchet MS', Verdana, sans-serif",
    "'Courier New', ui-monospace, monospace",
    "Tahoma, Geneva, sans-serif",
]
WORDS = ("Build Ship Scale Design Launch Grow Connect Secure Automate Deliver Simple Fast "
         "Modern Team Cloud Data Flow Studio Works Labs Craft Bright Clear Sharp Prime "
         "Nimbus Vertex Apex Lumen Atlas Orbit Nova Quill Ember Harbor Meridian").split()
NAV_WORDS = ["Product", "Pricing", "Docs", "Company", "Blog", "About", "Features",
             "Solutions", "Support", "Careers", "Contact", "Resources", "Platform"]
CTA_WORDS = ["Get started", "Try it free", "Book a demo", "Sign up", "Start free",
             "Learn more", "Contact sales", "Download", "Join now", "See pricing"]


def find_chrome():
    for p in CHROME_CANDIDATES:
        if os.path.exists(p):
            return p
    raise RuntimeError("Chrome not found; checked: %s" % CHROME_CANDIDATES)


def phrase(rng, n):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def noise_image(rng, w, h):
    """A stand-in photograph.

    Full-spectrum noise was the obvious thing to reach for and the wrong one: real
    photographs are not rainbows. Their colour sits in a narrow band, varies smoothly
    across the frame, carries a light gradient and a little grain. A model trained on
    confetti would learn that "image" means saturated chaos and miss every muted
    photograph it was meant to find.
    """
    base = rng.random(3) * 0.5 + 0.18                      # one dominant hue, not all of them
    shift = (rng.random(3) - 0.5) * 0.42                   # a second tone to drift towards
    field = np.zeros((h, w), np.float32)
    amp = 1.0
    for octave in (2, 4, 8, 16):
        small = rng.random((octave, octave)).astype(np.float32)
        up = np.asarray(Image.fromarray((small * 255).astype(np.uint8))
                        .resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
        field += up * amp
        amp *= 0.5
    field = (field - field.min()) / max(float(np.ptp(field)), 1e-6)
    field *= np.linspace(1.12, 0.72, h, dtype=np.float32)[:, None]   # light falls off
    rgb = (base[None, None, :] + shift[None, None, :] * field[:, :, None]) * 255.0
    rgb += (rng.random((h, w, 1)).astype(np.float32) - 0.5) * 12.0   # sensor grain
    buf = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buf, format="JPEG", quality=74)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def palette(rng):
    hue = int(rng.integers(0, 360))
    dark = bool(rng.random() < 0.38)
    if dark:
        return {
            "bg": "hsl(%d, %d%%, %d%%)" % (hue, rng.integers(8, 28), rng.integers(8, 16)),
            "surface": "hsl(%d, %d%%, %d%%)" % (hue, rng.integers(10, 30), rng.integers(17, 26)),
            "fg": "hsl(%d, 12%%, 95%%)" % hue,
            "muted": "hsl(%d, 10%%, 68%%)" % hue,
            "accent": "hsl(%d, %d%%, %d%%)" % ((hue + rng.integers(120, 240)) % 360,
                                               rng.integers(60, 92), rng.integers(50, 64)),
            "on_accent": "#0b1020",
            "border": "hsl(%d, 14%%, 32%%)" % hue,
        }
    return {
        "bg": "hsl(%d, %d%%, %d%%)" % (hue, rng.integers(0, 30), rng.integers(95, 100)),
        "surface": "hsl(%d, %d%%, %d%%)" % (hue, rng.integers(10, 40), rng.integers(93, 98)),
        "fg": "hsl(%d, 22%%, 12%%)" % hue,
        "muted": "hsl(%d, 12%%, 42%%)" % hue,
        "accent": "hsl(%d, %d%%, %d%%)" % ((hue + rng.integers(100, 260)) % 360,
                                           rng.integers(58, 88), rng.integers(38, 54)),
        "on_accent": "#ffffff",
        "border": "hsl(%d, 16%%, 86%%)" % hue,
    }


def build_page(rng, width, height):
    p = palette(rng)
    font = rng.choice(FONTS)
    # Real hero sections are frequently built over a photograph, with the controls
    # sitting on top of it. A first model trained only on flat grounds found no button
    # at all on such a page: it had never been asked to separate a control from
    # anything but a plain fill. Roughly half of these pages now have a photographic
    # ground, with translucent panels and copy laid over it.
    photo_ground = bool(rng.random() < 0.45)
    if photo_ground:
        p["fg"] = "#ffffff" if rng.random() < 0.75 else "hsl(0,0%,10%)"
        p["muted"] = "rgba(255,255,255,0.82)" if p["fg"] == "#ffffff" else "rgba(0,0,0,0.66)"
        p["surface"] = ("rgba(%d,%d,%d,%.2f)"
                        % (rng.integers(0, 60), rng.integers(0, 60), rng.integers(0, 70),
                           0.45 + rng.random() * 0.45)
                        if rng.random() < 0.6 else
                        "rgba(255,255,255,%.2f)" % (0.55 + rng.random() * 0.4))
        p["border"] = "rgba(255,255,255,0.28)"
    radius = int(rng.integers(0, 20))
    pill = int(rng.integers(0, 3)) == 0
    btn_radius = 999 if pill else radius
    pad_x = int(rng.integers(28, 72))
    gap = int(rng.integers(14, 40))

    def btn(label, primary=True):
        if primary:
            style = ("background:%s;color:%s;border:0;" % (p["accent"], p["on_accent"]))
        elif rng.random() < 0.5:
            style = ("background:transparent;color:%s;border:2px solid %s;"
                     % (p["fg"], p["border"]))
        else:
            style = ("background:%s;color:%s;border:0;" % (p["surface"], p["fg"]))
        return ('<button data-cls="button" style="%spadding:%dpx %dpx;border-radius:%dpx;'
                'font:600 %dpx %s;cursor:pointer;">%s</button>'
                % (style, rng.integers(9, 18), rng.integers(16, 34), btn_radius,
                   rng.integers(13, 18), font, label))

    parts = []

    # --- top bar -----------------------------------------------------------------
    if rng.random() < 0.9:
        links = "".join(
            '<a data-cls="link" href="#" style="color:%s;text-decoration:none;font:%dpx %s;">%s</a>'
            % (p["muted"], rng.integers(13, 17), font, rng.choice(NAV_WORDS))
            for _ in range(int(rng.integers(2, 6))))
        brand = ('<span data-cls="heading" style="color:%s;font:700 %dpx %s;">%s</span>'
                 % (p["fg"], rng.integers(17, 24), font, rng.choice(WORDS)))
        icon = ('<span data-cls="icon" style="display:inline-block;width:%dpx;height:%dpx;'
                'border-radius:%dpx;background:%s;"></span>'
                % (rng.integers(20, 34), rng.integers(20, 34),
                   999 if rng.random() < 0.6 else 6, p["accent"]))
        right = btn(rng.choice(CTA_WORDS)) if rng.random() < 0.85 else ""
        parts.append(
            '<div data-cls="nav" style="display:flex;align-items:center;gap:%dpx;'
            'padding:%dpx %dpx;background:%s;border-bottom:1px solid %s;">'
            '%s%s<div style="display:flex;gap:%dpx;margin-left:auto;align-items:center;">%s%s</div>'
            '</div>'
            % (gap, rng.integers(14, 26), pad_x,
               "transparent" if (photo_ground and rng.random() < 0.65) else p["surface"],
               "transparent" if photo_ground else p["border"],
               icon, brand, gap, links, right))

    # --- hero --------------------------------------------------------------------
    split = rng.random() < 0.55
    hero_bits = []
    if rng.random() < 0.45:
        hero_bits.append(
            '<span data-cls="badge" style="display:inline-block;background:%s;color:%s;'
            'padding:5px 12px;border-radius:999px;font:600 %dpx %s;">%s</span>'
            % (p["surface"], p["muted"], rng.integers(11, 14), font, phrase(rng, 2)))
    # Display type on real hero sections runs far larger than a default heading, and a
    # model that has only seen 34-68px treats a 120px headline as something else.
    head_px = int(rng.integers(30, 132) if rng.random() < 0.45 else rng.integers(30, 70))
    hero_bits.append(
        '<h1 data-cls="heading" style="color:%s;font:700 %dpx/1.05 %s;margin:0;">%s</h1>'
        % (p["fg"], head_px, font, phrase(rng, int(rng.integers(1, 4)))))
    hero_bits.append(
        '<p data-cls="text" style="color:%s;font:%dpx/1.55 %s;margin:0;max-width:%dpx;">%s</p>'
        % (p["muted"], rng.integers(15, 21), font, rng.integers(320, 560),
           phrase(rng, int(rng.integers(10, 22)))))
    ctas = btn(rng.choice(CTA_WORDS))
    if rng.random() < 0.7:
        ctas += btn(rng.choice(CTA_WORDS), primary=False)
    if rng.random() < 0.3:
        ctas += ('<input data-cls="input" placeholder="%s" style="padding:%dpx 14px;'
                 'border-radius:%dpx;border:1px solid %s;background:%s;color:%s;'
                 'font:%dpx %s;width:%dpx;">'
                 % (phrase(rng, 2), rng.integers(10, 16), radius, p["border"],
                    p["surface"], p["muted"], rng.integers(13, 17), font,
                    rng.integers(180, 320)))
    hero_bits.append('<div style="display:flex;gap:%dpx;flex-wrap:wrap;align-items:center;">%s</div>'
                     % (gap, ctas))
    hero_text = ('<div style="display:flex;flex-direction:column;gap:%dpx;%s">%s</div>'
                 % (gap, "" if split else "align-items:center;text-align:center;",
                    "".join(hero_bits)))

    if split:
        vis = ('<img data-cls="image" src="%s" style="width:100%%;height:%dpx;'
               'border-radius:%dpx;object-fit:cover;display:block;">'
               % (noise_image(rng, 320, 240), rng.integers(220, 380), radius))
        if rng.random() < 0.45:
            vis = ('<div data-cls="card" style="background:%s;border:1px solid %s;'
                   'border-radius:%dpx;padding:%dpx;display:flex;flex-direction:column;gap:12px;">'
                   '%s<span data-cls="heading" style="color:%s;font:700 %dpx %s;">%s</span>'
                   '<span data-cls="text" style="color:%s;font:%dpx %s;">%s</span>%s</div>'
                   % (p["surface"], p["border"], radius, rng.integers(16, 28), vis,
                      p["fg"], rng.integers(17, 24), font, phrase(rng, 2),
                      p["muted"], rng.integers(13, 16), font, phrase(rng, 8),
                      btn(rng.choice(CTA_WORDS))))
        parts.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:%dpx;'
                     'align-items:center;padding:%dpx %dpx;">%s%s</div>'
                     % (gap * 2, rng.integers(40, 90), pad_x, hero_text, vis))
    else:
        parts.append('<div style="padding:%dpx %dpx;">%s</div>'
                     % (rng.integers(50, 100), pad_x, hero_text))

    # --- feature or card row ------------------------------------------------------
    if rng.random() < 0.75:
        n = int(rng.integers(2, 5))
        cards = []
        for _ in range(n):
            inner = ('<span data-cls="icon" style="display:inline-block;width:%dpx;height:%dpx;'
                     'border-radius:%dpx;background:%s;"></span>'
                     % (rng.integers(26, 44), rng.integers(26, 44),
                        999 if rng.random() < 0.5 else 8, p["accent"]))
            inner += ('<span data-cls="heading" style="color:%s;font:700 %dpx %s;">%s</span>'
                      % (p["fg"], rng.integers(15, 21), font, phrase(rng, 2)))
            inner += ('<span data-cls="text" style="color:%s;font:%dpx/1.5 %s;">%s</span>'
                      % (p["muted"], rng.integers(12, 16), font, phrase(rng, 9)))
            if rng.random() < 0.35:
                inner += ('<a data-cls="link" href="#" style="color:%s;font:600 %dpx %s;">%s</a>'
                          % (p["accent"], rng.integers(13, 16), font, rng.choice(CTA_WORDS)))
            boxed = rng.random() < 0.7
            cards.append('<div %s style="display:flex;flex-direction:column;gap:10px;%s">%s</div>'
                         % ('data-cls="card"' if boxed else "",
                            ("background:%s;border:1px solid %s;border-radius:%dpx;padding:%dpx;"
                             % (p["surface"], p["border"], radius, rng.integers(16, 30)))
                            if boxed else "", inner))
        parts.append('<div style="display:grid;grid-template-columns:repeat(%d,1fr);gap:%dpx;'
                     'padding:%dpx %dpx;">%s</div>'
                     % (n, gap, rng.integers(20, 60), pad_x, "".join(cards)))

    body = "".join(parts)
    if photo_ground:
        ground = ("background-image:url(%s);background-size:cover;background-position:center;"
                  % noise_image(rng, 420, 300))
        if rng.random() < 0.7:                       # the scrim real designs put over a photo
            ground = ("background-image:linear-gradient(rgba(0,0,0,%.2f),rgba(0,0,0,%.2f)),url(%s);"
                      "background-size:cover;background-position:center;"
                      % (rng.random() * 0.45, 0.15 + rng.random() * 0.5,
                         noise_image(rng, 420, 300)))
    else:
        ground = "background:%s;" % p["bg"]

    return ("<!doctype html><html><head><meta charset='utf-8'><style>"
            "*{box-sizing:border-box}html,body{margin:0;padding:0;}"
            "body{%s;font-family:%s;width:%dpx;min-height:%dpx;}"
            "</style></head><body>%s"
            "<script id='labels' type='application/json'></script><script>"
            "(function(){var o=[];document.querySelectorAll('[data-cls]').forEach(function(e){"
            "var r=e.getBoundingClientRect();"
            "if(r.width>4&&r.height>4&&r.left<%d&&r.top<%d)"
            "o.push({c:e.dataset.cls,x:r.left,y:r.top,w:r.width,h:r.height});});"
            "document.getElementById('labels').textContent=JSON.stringify(o);})();"
            "</script></body></html>"
            % (ground, font, width, height, body, width, height))


def render(chrome, html_path, png_path, width, height):
    """Screenshot and DOM geometry from one identical layout."""
    flags = [chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
             "--force-device-scale-factor=1", "--virtual-time-budget=3000",
             "--window-size=%d,%d" % (width, height)]
    url = "file:///" + os.path.abspath(html_path).replace("\\", "/")
    subprocess.run(flags + ["--screenshot=" + os.path.abspath(png_path), url],
                   check=True, capture_output=True)
    dom = subprocess.run(flags + ["--dump-dom", url],
                         check=True, capture_output=True).stdout.decode("utf-8", "replace")
    m = re.search(r"<script id=\"labels\" type=\"application/json\">(.*?)</script>", dom, re.S)
    return json.loads(m.group(1)) if m else []


def write_split(chrome, out_root, split, count, seed):
    img_dir = os.path.join(out_root, "images", split)
    lbl_dir = os.path.join(out_root, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    tmp = os.path.join(out_root, "_build")
    os.makedirs(tmp, exist_ok=True)

    kept = 0
    for i in range(count):
        rng = np.random.default_rng(seed + i)
        width = int(rng.choice([1280, 1366, 1440, 1200, 1536]))
        height = int(rng.choice([720, 800, 860, 900]))
        html = build_page(rng, width, height)
        hp = os.path.join(tmp, "page.html")
        with open(hp, "w", encoding="utf-8") as f:
            f.write(html)

        name = "%s_%05d" % (split, i)
        png = os.path.join(img_dir, name + ".png")
        try:
            boxes = render(chrome, hp, png, width, height)
        except subprocess.CalledProcessError:
            continue
        if not boxes or not os.path.exists(png):
            continue

        lines = []
        for b in boxes:
            cid = CLASS_ID.get(b["c"])
            if cid is None:
                continue
            x0 = max(0.0, b["x"]); y0 = max(0.0, b["y"])
            x1 = min(float(width), b["x"] + b["w"]); y1 = min(float(height), b["y"] + b["h"])
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            lines.append("%d %.6f %.6f %.6f %.6f" % (
                cid, ((x0 + x1) / 2) / width, ((y0 + y1) / 2) / height,
                (x1 - x0) / width, (y1 - y0) / height))
        if not lines:
            os.remove(png)
            continue
        with open(os.path.join(lbl_dir, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        kept += 1
        if kept % 25 == 0:
            print("  %s: %d/%d" % (split, kept, count), flush=True)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--val", type=int, default=60)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "scratch", "uidata"))
    args = ap.parse_args()

    chrome = find_chrome()
    os.makedirs(args.out, exist_ok=True)
    n_tr = write_split(chrome, args.out, "train", args.train, seed=1000)
    n_va = write_split(chrome, args.out, "val", args.val, seed=90000)

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8", newline="\n") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\n\nnames:\n"
                % args.out.replace("\\", "/"))
        for i, c in enumerate(CLASSES):
            f.write("  %d: %s\n" % (i, c))
    print("\ntrain %d  val %d  ->  %s" % (n_tr, n_va, args.out))


if __name__ == "__main__":
    main()
