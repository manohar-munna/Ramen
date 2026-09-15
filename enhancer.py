"""Optional post-pass: hand the reconstructed HTML to a language model to be rebuilt.

This is deliberately *outside* the reconstruction engine and never runs as part of it.
The engine's job is to be faithful and deterministic: what it emits is measured against
the original and is reproducible from the pixels alone. What it emits is also, honestly,
a pile of absolutely-positioned divs, because faithfulness is the only thing it is
optimising. Turning that into a document a person would want to edit -- flowing layout,
semantic tags, hover states, transitions -- is a different job with no ground truth, and
a language model is a reasonable tool for it.

Keeping the two apart matters. The reconstruction stays verifiable; this pass is a
convenience on top, and its output is stored alongside the faithful version rather than
replacing it.

The one thing that makes this practical: roughly 95% of a reconstructed page by weight is
inline base64 PNG. Sending that to a model would cost 380k tokens for a single page and
tell it nothing -- it cannot see the pixels and has no reason to want them. Swapping each
data URI for a short marker takes the same page to 16k tokens, and the markers are put
back afterwards, so the images are never at the mercy of the model reproducing a megabyte
of base64 exactly.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

# Distinctive enough that a model will carry it through verbatim, and short enough that
# it costs nothing. Deliberately not a URL: nothing should try to fetch it.
_PLACEHOLDER = "RAMEN_ASSET_%d"
_PLACEHOLDER_RE = re.compile(r"RAMEN_ASSET_(\d+)")
_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")

DEFAULT_MODEL = "gemini-2.5-pro"
_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent")


class EnhancementError(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


# ---------------------------------------------------------------- asset handling

def strip_assets(html: str) -> Tuple[str, List[str]]:
    """Replaces every inline data URI with a short marker.

    Identical assets share one marker, which matters more than it sounds: a page that
    repeats the same icon eight times would otherwise pay for it eight times.
    """
    assets: List[str] = []
    index: Dict[str, int] = {}

    def swap(match: re.Match) -> str:
        uri = "".join(match.group(0).split())     # data URIs may carry stray whitespace
        if uri not in index:
            index[uri] = len(assets)
            assets.append(uri)
        return _PLACEHOLDER % index[uri]

    return _DATA_URI_RE.sub(swap, html), assets


def restore_assets(html: str, assets: List[str]) -> Tuple[str, List[int]]:
    """Puts the data URIs back. Returns the HTML and any markers the model lost.

    A dropped marker is worth reporting rather than hiding: it means the model deleted an
    image, and the caller should be able to say so instead of silently shipping a page
    with content missing.
    """
    seen = {int(m) for m in _PLACEHOLDER_RE.findall(html)}
    missing = [i for i in range(len(assets)) if i not in seen]

    def swap(match: re.Match) -> str:
        i = int(match.group(1))
        return assets[i] if 0 <= i < len(assets) else match.group(0)

    return _PLACEHOLDER_RE.sub(swap, html), missing


def _unfence(text: str) -> str:
    """Models wrap code in fences however often you ask them not to."""
    text = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\n(.*?)\n?```$", text, re.S)
    if fence:
        return fence.group(1).strip()
    return text


# ---------------------------------------------------------------- the instruction

PROMPT = """\
You are rewriting one HTML file. It was produced by a computer-vision pipeline that
reconstructs a web page from a screenshot, so it is accurate but badly built: almost
every element is `position: absolute` at measured pixel coordinates, containers are
anonymous divs, and nothing reflows.

Rewrite it as the page a competent front-end developer would have written to produce
that same design.

MUST NOT CHANGE
- Every piece of visible text, character for character. Do not reword, fix spelling,
  translate, shorten, or add text of your own.
- Every `RAMEN_ASSET_<n>` marker. These stand in for images. Keep each one exactly as
  written, in an `src` attribute, and keep all of them -- do not drop, merge, renumber
  or invent markers. There are %(n_assets)d of them, numbered 0 to %(max_asset)d.
- The colours, font sizes, weights and spacing, near enough that the page still looks
  like the same design.

MUST DO
- Replace absolute positioning with real layout: flexbox and grid, normal document
  flow, sensible containers. The page must reflow when the window is resized.
- Use semantic elements: header, nav, main, section, footer, h1-h6, p, ul/li, button,
  a. Where the input already uses a meaningful tag, keep that meaning.
- Put the CSS in one `<style>` block, organised, using CSS custom properties for the
  palette. No inline `style` attributes except where genuinely per-element.
- Make it responsive: it should be usable down to 480px wide.
- Add the polish the original screenshot could not capture: hover and focus states on
  every interactive element, smooth `transition` on colour/transform/shadow changes,
  and tasteful entrance animations. Keep them subtle and fast (150-300ms). Respect
  `@media (prefers-reduced-motion: reduce)` by disabling them.
- Keep images responsive: `max-width: 100%%`, `height: auto`, and `object-fit` where an
  image fills a box.

OUTPUT
Return the complete HTML document and nothing else. No explanation, no commentary, no
markdown fences. Start with `<!DOCTYPE html>`.

Here is the file:

%(html)s
"""


def build_prompt(skeleton: str, n_assets: int, extra: Optional[str] = None) -> str:
    prompt = PROMPT % {
        "html": skeleton,
        "n_assets": n_assets,
        "max_asset": max(n_assets - 1, 0),
    }
    if extra:
        prompt += "\n\nAdditional instructions from the user, which take precedence:\n"
        prompt += extra.strip() + "\n"
    return prompt


# ---------------------------------------------------------------- configuration

def load_dotenv(path: Optional[str] = None) -> bool:
    """Reads a .env file into os.environ without pulling in a dependency for it.

    Values already in the environment win, so an operator can override the file without
    editing it. Returns whether a file was found.
    """
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.startswith("export "):
                line = line[len("export "):]
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value
    return True


def is_configured() -> bool:
    """Whether an enhancement could run, without revealing anything about the key."""
    load_dotenv()
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


# ---------------------------------------------------------------- the call

def _api_key(explicit: Optional[str] = None) -> str:
    load_dotenv()
    key = explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise EnhancementError(
            "No API key. Put GEMINI_API_KEY=... in a .env file at the project root, "
            "or set it in the environment.")
    return key


def call_gemini(prompt: str, api_key: Optional[str] = None,
                model: Optional[str] = None, timeout: int = 300) -> str:
    """Sends one prompt and returns the text of the reply."""
    model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            # Low but not zero: layout decisions benefit from a little freedom, and
            # nothing here needs to be reproducible -- the faithful version already is.
            "temperature": 0.35,
            "maxOutputTokens": 65536,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        _ENDPOINT.format(model=model),
        data=body,
        headers={"Content-Type": "application/json",
                 "x-goog-api-key": _api_key(api_key)},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        raise EnhancementError(f"Gemini returned {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise EnhancementError(f"Could not reach Gemini: {e.reason}") from e

    candidates = payload.get("candidates") or []
    if not candidates:
        blocked = (payload.get("promptFeedback") or {}).get("blockReason")
        raise EnhancementError(f"Gemini returned no candidates"
                               + (f" (blocked: {blocked})" if blocked else ""))
    parts = (candidates[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    if not text:
        reason = candidates[0].get("finishReason")
        raise EnhancementError(
            "Gemini returned an empty reply"
            + (f" (finishReason: {reason})" if reason else "")
            + (". The page may be too large for one response." if reason == "MAX_TOKENS"
               else ""))
    return text


# ---------------------------------------------------------------- the whole pass

def enhance_html(html: str, api_key: Optional[str] = None, model: Optional[str] = None,
                 extra: Optional[str] = None, timeout: int = 300) -> Dict[str, object]:
    """Rewrites a reconstructed page. Returns the HTML and what happened to it.

    Never raises for a merely disappointing result -- a model that drops an image still
    produced something worth looking at -- but does report the loss so the caller can.
    """
    skeleton, assets = strip_assets(html)
    prompt = build_prompt(skeleton, len(assets), extra)

    reply = _unfence(call_gemini(prompt, api_key=api_key, model=model, timeout=timeout))
    if "<" not in reply:
        raise EnhancementError("Gemini's reply does not look like HTML: "
                               + reply[:200])

    enhanced, missing = restore_assets(reply, assets)
    return {
        "html": enhanced,
        "model": model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
        "assets_total": len(assets),
        "assets_missing": missing,
        "sent_chars": len(prompt),
        "original_chars": len(html),
        "enhanced_chars": len(enhanced),
    }


# ---------------------------------------------------------------- command line

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys
    import time

    ap = argparse.ArgumentParser(
        description="Rewrite a reconstruction into a laid-out, animated page.")
    ap.add_argument("source", help="a screenshot/PDF to reconstruct first, or an .html file")
    ap.add_argument("-o", "--out", help="where to write the result "
                                        "(default: <source>.enhanced.html)")
    ap.add_argument("-m", "--model", help=f"default: {DEFAULT_MODEL}")
    ap.add_argument("-i", "--instructions", help="extra direction for the rewrite")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompt and report its size without calling anything")
    args = ap.parse_args(argv)

    if args.source.lower().endswith((".html", ".htm")):
        with open(args.source, encoding="utf-8") as fh:
            html = fh.read()
    else:
        from engine import DocumentData, HTMLRenderer, ImageReconstructor
        page = ImageReconstructor.reconstruct_image(args.source)
        html = HTMLRenderer.render_document(
            DocumentData(title=os.path.basename(args.source), pageCount=1, pages=[page]),
            editable=False, interactive=False)

    if args.dry_run:
        skeleton, assets = strip_assets(html)
        prompt = build_prompt(skeleton, len(assets), args.instructions)
        print(f"page      {len(html):>9,} chars")
        print(f"prompt    {len(prompt):>9,} chars  (~{len(prompt)//4:,} tokens)")
        print(f"assets    {len(assets):>9,} held back, "
              f"{100.0 * (len(html) - len(skeleton)) / max(len(html), 1):.1f}% of the page")
        print(f"key       {'present' if is_configured() else 'MISSING - see .env.example'}")
        return 0

    out = args.out or (os.path.splitext(args.source)[0] + ".enhanced.html")
    started = time.time()
    try:
        result = enhance_html(html, model=args.model, extra=args.instructions)
    except EnhancementError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(str(result["html"]))
    print(f"{result['model']} in {time.time() - started:.0f}s -> {out}")
    print(f"  {result['original_chars']:,} chars in, {result['enhanced_chars']:,} out; "
          f"sent {result['sent_chars']:,}")
    missing = result["assets_missing"]
    if missing:
        print(f"  WARNING: the model dropped {len(missing)} image(s): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
