"""Runs the enhancement pass over every reference image and scores what came back.

Resumable on purpose. Reconstruction costs a minute or two a page and the API has a
daily allowance, so anything already on disk is reused and a quota refusal stops the run
cleanly with a report of what did finish rather than losing the lot.

    python tools/enhance_all.py [-m MODEL] [--only NAME] [--reconstruct-only]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.getcwd())

import enhancer                                          # noqa: E402
from engine import DocumentData, HTMLRenderer, ImageReconstructor   # noqa: E402
from _chrome import find_chrome, screenshot                          # noqa: E402

OUT = os.path.join("scratch", "enhance_all")


def reconstruct(path, name):
    """Document JSON for a page, built once and cached."""
    doc_path = os.path.join(OUT, name + ".doc.json")
    if os.path.isfile(doc_path):
        with open(doc_path, encoding="utf-8") as fh:
            return DocumentData.model_validate(json.load(fh))
    page = ImageReconstructor.reconstruct_image(path)
    doc = DocumentData(title=name, pageCount=1, pages=[page])
    with open(doc_path, "w", encoding="utf-8") as fh:
        json.dump(doc.model_dump(), fh)
    return doc


def render(html_path, png_path, width=1400, height=2400):
    """Screenshot with animations settled, so a staggered fade is not read as a fault."""
    with open(html_path, encoding="utf-8") as fh:
        html = fh.read()
    settled = html.replace(
        "</head>",
        "<style>*,*::before,*::after{animation:none!important;"
        "transition:none!important;opacity:1!important}</style></head>", 1)
    tmp = html_path.replace(".html", ".settled.html")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(settled)
    try:
        screenshot(find_chrome(), tmp, png_path, width, height,
                   extra=["--virtual-time-budget=8000"])
        return True
    except Exception:
        return False


def count(pattern, s):
    return len(re.findall(pattern, s, re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model")
    ap.add_argument("--only")
    ap.add_argument("--reconstruct-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    images = sorted(glob.glob(os.path.join("benchmarks", "images", "*.png")))
    if args.only:
        images = [f for f in images if args.only in os.path.basename(f)]

    rows = []
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        print("=" * 74)
        print(name, flush=True)
        t0 = time.time()
        try:
            doc = reconstruct(path, name)
        except Exception as e:
            print("  reconstruction failed: %s" % e, flush=True)
            continue

        before = sum(1 for p in doc.pages for e in p.elements if e.type == "image")
        flat = enhancer.flatten_document(doc)
        after = sum(1 for p in flat.pages for e in p.elements if e.type == "image")
        faithful = HTMLRenderer.render_document(flat, editable=False, interactive=False)
        faithful_path = os.path.join(OUT, name + ".flat.html")
        with open(faithful_path, "w", encoding="utf-8") as fh:
            fh.write(faithful)
        print("  reconstructed in %.0fs; rasters %d -> %d" %
              (time.time() - t0, before, after), flush=True)
        if args.reconstruct_only:
            continue

        out_path = os.path.join(OUT, name + ".enhanced.html")
        if os.path.isfile(out_path):
            print("  already enhanced, reusing", flush=True)
            with open(out_path, encoding="utf-8") as fh:
                enhanced = fh.read()
            result = {"model": "(cached)", "assets_total": after,
                      "assets_missing": [], "assets_missing_initially": [],
                      "assets_recovered": 0, "repair_rounds": 0, "notes": []}
        else:
            t1 = time.time()
            try:
                result = enhancer.enhance_html(
                    faithful, model=args.model, reference=path,
                    on_retry=lambda a, t, c, d: print(
                        "    %s from the model (%d/%d); waiting %.0fs" % (c, a, t, d),
                        flush=True))
            except enhancer.EnhancementError as e:
                msg = str(e).splitlines()[0]
                print("  ENHANCEMENT FAILED: %s" % msg, flush=True)
                if "429" in msg:
                    print("\nDaily allowance reached. Re-run later to continue; "
                          "everything finished so far is on disk.", flush=True)
                    break
                continue
            enhanced = str(result["html"])
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(enhanced)
            print("  enhanced by %s in %.0fs" % (result["model"], time.time() - t1),
                  flush=True)

        for note in result.get("notes") or []:
            print("    %s" % note, flush=True)

        render(out_path, os.path.join(OUT, name + ".png"))

        audit = subprocess.run(
            [sys.executable, os.path.join("tools", "audit_enhanced.py"),
             faithful_path, out_path],
            capture_output=True, text=True)
        text_ok = "no visible text was lost" in audit.stdout
        img_ok = "every embedded image survived" in audit.stdout
        lost_words = 0
        m = re.search(r"FAIL\s+(\d+) word occurrence", audit.stdout)
        if m:
            lost_words = int(m.group(1))

        rows.append({
            "name": name,
            "rasters": "%d->%d" % (before, after),
            "abs_before": count(r"position\s*:\s*absolute", faithful),
            "abs_after": count(r"position\s*:\s*absolute", enhanced),
            "flex": count(r"display\s*:\s*(flex|grid)", enhanced),
            "trans": count(r"\btransition\s*:", enhanced),
            "semantic": count(r"<(header|nav|main|section|footer|article|aside)\b", enhanced),
            "imgs": "%d/%d" % (result["assets_total"] - len(result["assets_missing"]),
                               result["assets_total"]),
            "img_ok": img_ok,
            "text_ok": text_ok,
            "lost_words": lost_words,
            "dropped": len(result.get("assets_missing_initially") or []),
            "recovered": result.get("assets_recovered", 0),
        })
        print("  images %s, text %s" % (
            "all kept" if img_ok else "SOME LOST",
            "intact" if text_ok else "%d word(s) lost" % lost_words), flush=True)

    print()
    print("=" * 96)
    print("%-24s %9s %12s %6s %6s %5s %8s %s" % (
        "page", "rasters", "abs pos", "flex", "trans", "sem", "images", "text"))
    print("-" * 96)
    for r in rows:
        print("%-24s %9s %5d ->%4d %6d %6d %5d %8s %s" % (
            r["name"][:24], r["rasters"], r["abs_before"], r["abs_after"],
            r["flex"], r["trans"], r["semantic"], r["imgs"],
            "ok" if r["text_ok"] else "-%d words" % r["lost_words"]))
    if rows:
        print("-" * 96)
        print("%d pages | images intact on %d | text intact on %d | dropped %d, "
              "recovered %d by repair" % (
                  len(rows), sum(1 for r in rows if r["img_ok"]),
                  sum(1 for r in rows if r["text_ok"]),
                  sum(r["dropped"] for r in rows),
                  sum(r["recovered"] for r in rows)))
    print("output in", OUT)


if __name__ == "__main__":
    main()
