"""Checks what the enhancement pass actually did to a page.

The enhanced HTML is the one artefact in this repo that nothing measures: a model
followed instructions and we take delivery. That is fine for layout, which has no ground
truth, but not for content -- text going missing or images being dropped is a silent
failure that looks like success. This compares the rewrite against the reconstruction it
came from and reports what changed.

    python tools/audit_enhanced.py <original.html> <enhanced.html>
"""
import html as html_mod
import os
import re
import sys
from collections import Counter

_TAG = re.compile(r"<[^>]+>")
_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_DATA_URI = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=]+)")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


_HEAD = re.compile(r"<head[^>]*>.*?</head>", re.S | re.I)


def visible_words(html: str) -> Counter:
    """Every word a reader would see in the page, as a multiset.

    The <head> is cut first. It carries the <title>, which is not on the page and which
    a rewrite is right to improve -- "site-stanford" becoming "Stanford University" is a
    better title, not lost content. Counting it reported a word lost on five of the
    eleven references and hid how clean those runs actually were.
    """
    body = _HEAD.sub(" ", html)
    body = _STYLE.sub(" ", body)
    body = _TAG.sub(" ", body)
    # html.unescape rather than a handful of replacements: the reconstruction emits
    # numeric entities such as &#x27;, and a partial decoder reads that as the word
    # "x27", then reports it missing from any rewrite that used a real apostrophe.
    body = html_mod.unescape(body)
    return Counter(w.lower() for w in _WORD.findall(body))


def asset_digests(html: str) -> Counter:
    """Identity of each embedded image, by a cheap fingerprint of its payload."""
    return Counter(len(m) and (m[:24] + m[-24:]) for m in _DATA_URI.findall(html))


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip())
        return 2
    original, enhanced = argv[1], argv[2]
    for p in (original, enhanced):
        if not os.path.isfile(p):
            print("no such file: %s" % p)
            return 2
    a = open(original, encoding="utf-8").read()
    b = open(enhanced, encoding="utf-8").read()

    wa, wb = visible_words(a), visible_words(b)
    lost = wa - wb
    added = wb - wa
    aa, ab = asset_digests(a), asset_digests(b)

    print("%-26s %12s %12s" % ("", "reconstruction", "enhanced"))
    print("%-26s %12s %12s" % ("size", f"{len(a):,}", f"{len(b):,}"))
    print("%-26s %12d %12d" % ("distinct words", len(wa), len(wb)))
    print("%-26s %12d %12d" % ("total words", sum(wa.values()), sum(wb.values())))
    print("%-26s %12d %12d" % ("embedded images", sum(aa.values()), sum(ab.values())))
    print("%-26s %12d %12d" % ("distinct images", len(aa), len(ab)))

    # Layout signals: what the rewrite was actually asked to change.
    def count(pattern, s):
        return len(re.findall(pattern, s, re.I))

    print()
    print("%-26s %12s %12s" % ("", "before", "after"))
    for label, pattern in (
        ("position:absolute", r"position\s*:\s*absolute"),
        ("inline style attrs", r"\sstyle\s*="),
        ("display:flex|grid", r"display\s*:\s*(flex|grid)"),
        ("transition rules", r"\btransition\s*:"),
        ("hover/focus rules", r":(hover|focus)"),
        ("@media queries", r"@media"),
        ("prefers-reduced-motion", r"prefers-reduced-motion"),
        ("semantic landmarks", r"<(header|nav|main|section|footer|article|aside)\b"),
        ("headings h1-h6", r"<h[1-6]\b"),
        ("buttons", r"<button\b"),
        ("links", r"<a\b"),
    ):
        print("%-26s %12d %12d" % (label, count(pattern, a), count(pattern, b)))

    print()
    missing_imgs = sum((aa - ab).values())
    if missing_imgs:
        print("FAIL  %d embedded image(s) lost" % missing_imgs)
    else:
        print("ok    every embedded image survived")

    if lost:
        shown = ", ".join("%s x%d" % (w, n) for w, n in lost.most_common(12))
        print("FAIL  %d word occurrence(s) lost: %s" % (sum(lost.values()), shown))
    else:
        print("ok    no visible text was lost")
    if added:
        shown = ", ".join("%s x%d" % (w, n) for w, n in added.most_common(12))
        print("note  %d word occurrence(s) added: %s" % (sum(added.values()), shown))

    return 1 if (missing_imgs or lost) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
