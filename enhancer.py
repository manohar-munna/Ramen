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

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

# Distinctive enough that a model will carry it through verbatim, and short enough that
# it costs nothing. Deliberately not a URL: nothing should try to fetch it.
_PLACEHOLDER = "RAMEN_ASSET_%d"
_PLACEHOLDER_RE = re.compile(r"RAMEN_ASSET_(\d+)")
_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")

# An alias rather than a pinned version, deliberately. A pinned name goes stale without
# warning: gemini-2.5-pro was still listed by the models endpoint while returning 404 to
# any key created after it was retired. Flash rather than pro because pro is not in the
# free tier -- a new key gets 429 on every pro model and works fine on flash, and being
# usable out of the box matters more here than the last few points of quality.
DEFAULT_MODEL = "gemini-3-flash-preview"
# Availability is genuinely unreliable: gemini-2.5-pro is still listed by the models
# endpoint while returning 404 to any key made after it was retired, pro models 429 on
# the free tier, and gemini-flash-latest returned 503 to a full page four times running
# while answering a one-line prompt instantly. So try several rather than trusting one.
FALLBACK_MODELS = ("gemini-3-flash-preview", "gemini-flash-latest", "gemini-3.5-flash")
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


def describe_assets(assets: List[str], max_colours: int = 3) -> List[str]:
    """One line per held-back image: its size and the colours in it.

    Stripping the images also strips the page's colour identity, which is a problem
    nobody notices until the result comes back grey. On the logistics page the red is
    entirely in the photographs of the containers; every flat surface really is a grey or
    a near-white, so a model shown only the skeleton has no way to know the page is red,
    and duly picks a palette of #1a1a1a and #888888.

    Sending a few dominant colours and the dimensions costs a couple of dozen tokens and
    gives back both the palette and enough shape information to size the image sensibly.
    """
    try:
        from PIL import Image
    except ImportError:
        return []

    lines = []
    for i, uri in enumerate(assets):
        head, _, payload = uri.partition(",")
        try:
            raw = base64.b64decode(payload)
            im = Image.open(io.BytesIO(raw))
            w, h = im.size
            rgb = im.convert("RGBA")
            # Ignore transparent pixels: a carved-out backdrop is mostly nothing, and
            # averaging the nothing in turns every colour towards the same grey.
            small = rgb.resize((min(w, 64), min(h, 64)))
            pixels = [p for p in small.getdata() if p[3] > 128]
            if not pixels:
                lines.append(f"{_PLACEHOLDER % i}  {w}x{h}  (fully transparent)")
                continue
            quant = Image.new("RGB", (len(pixels), 1))
            quant.putdata([p[:3] for p in pixels])
            quant = quant.quantize(colors=max_colours, method=Image.Quantize.FASTOCTREE)
            pal = quant.getpalette() or []
            counts = sorted(quant.getcolors() or [], reverse=True)
            names = []
            for _, idx in counts[:max_colours]:
                r, g, b = pal[idx * 3:idx * 3 + 3]
                names.append("#%02x%02x%02x" % (r, g, b))
            lines.append(f"{_PLACEHOLDER % i}  {w}x{h}  {' '.join(names)}")
        except Exception:
            lines.append(f"{_PLACEHOLDER % i}  (unreadable)")
    return lines


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

THE IMAGES YOU CANNOT SEE
Each marker below is followed by its pixel size and its most common colours. The page's
real colour identity is usually in these rather than in the CSS, because photographs
carry it and flat surfaces do not. Build the palette from them as well as from the
stylesheet -- do not return a grey page because the skeleton looked grey. Use the sizes
to give each image a sensible aspect ratio.

%(assets)s

OUTPUT
Return the complete HTML document and nothing else. No explanation, no commentary, no
markdown fences. Start with `<!DOCTYPE html>`.

Here is the file:

%(html)s
"""


def build_prompt(skeleton: str, n_assets: int, extra: Optional[str] = None,
                 asset_lines: Optional[List[str]] = None) -> str:
    prompt = PROMPT % {
        "html": skeleton,
        "n_assets": n_assets,
        "max_asset": max(n_assets - 1, 0),
        "assets": "\n".join(asset_lines or []) or "(none)",
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


def list_models(api_key: Optional[str] = None, timeout: int = 60) -> List[str]:
    """Model names this key can actually pass to generateContent.

    Worth having as a first-class command: the error you get for a retired model names a
    replacement that may also not be available to you, and guessing from documentation is
    how this broke in the first place.
    """
    req = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200",
        headers={"x-goog-api-key": _api_key(api_key)})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise EnhancementError(f"Could not list models: {e.code}") from e
    except urllib.error.URLError as e:
        raise EnhancementError(f"Could not reach Gemini: {e.reason}") from e
    return sorted(
        m["name"].replace("models/", "")
        for m in payload.get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", []))


# 503 means the model is busy and 429 can mean a per-minute ceiling rather than an
# exhausted plan. Both are worth waiting out rather than handing back to the caller: a
# page takes a couple of minutes to reconstruct, and losing that to a transient spike is
# a poor trade for the few seconds a retry costs.
RETRY_STATUSES = (429, 500, 502, 503, 504)
RETRY_ATTEMPTS = 4
RETRY_BACKOFF = 6.0     # seconds, doubling


def call_gemini(prompt: str, api_key: Optional[str] = None,
                model: Optional[str] = None, timeout: int = 300,
                attempts: int = RETRY_ATTEMPTS, on_retry=None) -> str:
    """Sends one prompt and returns the text of the reply, retrying transient failures."""
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

    key = _api_key(api_key)
    for attempt in range(1, max(1, attempts) + 1):
        try:
            payload = _post(model, body, key, timeout)
            break
        except _Transient as t:
            if attempt >= attempts:
                raise t.error
            delay = RETRY_BACKOFF * (2 ** (attempt - 1))
            if on_retry:
                on_retry(attempt, attempts, t.code, delay)
            time.sleep(delay)

    candidates = payload.get("candidates") or []
    if not candidates:
        blocked = (payload.get("promptFeedback") or {}).get("blockReason")
        raise EnhancementError("Gemini returned no candidates"
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


class _Transient(Exception):
    def __init__(self, code, error):
        self.code, self.error = code, error


def _post(model: str, body: bytes, key: str, timeout: int) -> dict:
    req = urllib.request.Request(
        _ENDPOINT.format(model=model),
        data=body,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:600]
        # These two are worth naming, because the raw message sends people the wrong way.
        sep = chr(10) + chr(10)
        if e.code == 404 and "model" in detail.lower():
            hint = (sep + f"The model '{model}' is not available to this key. Models are "
                    "retired without being delisted, so a name can appear valid and still "
                    "404. Run `python enhancer.py --list-models` to see what this key can "
                    "actually call, then set GEMINI_MODEL in .env.")
        elif e.code == 429:
            hint = (sep + "This is a quota limit, not a bad key. Pro models are not in the "
                    "free tier: a new key gets 429 on every one of them and works on flash. "
                    "Either leave GEMINI_MODEL unset (it defaults to flash) or enable "
                    "billing for pro access.")
        else:
            hint = ""
        err = EnhancementError(f"Gemini returned {e.code}: {detail}{hint}")
        if e.code in RETRY_STATUSES:
            raise _Transient(e.code, err) from e
        raise err from e
    except urllib.error.URLError as e:
        raise EnhancementError(f"Could not reach Gemini: {e.reason}") from e


# ---------------------------------------------------------------- the whole pass

def enhance_html(html: str, api_key: Optional[str] = None, model: Optional[str] = None,
                 extra: Optional[str] = None, timeout: int = 300,
                 on_retry=None) -> Dict[str, object]:
    """Rewrites a reconstructed page. Returns the HTML and what happened to it.

    Never raises for a merely disappointing result -- a model that drops an image still
    produced something worth looking at -- but does report the loss so the caller can.
    """
    skeleton, assets = strip_assets(html)
    prompt = build_prompt(skeleton, len(assets), extra, describe_assets(assets))

    reply = _unfence(call_gemini(prompt, api_key=api_key, model=model, timeout=timeout,
                                 on_retry=on_retry))
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

    ap = argparse.ArgumentParser(
        description="Rewrite a reconstruction into a laid-out, animated page.")
    ap.add_argument("source", nargs="?",
                    help="a screenshot/PDF to reconstruct first, or an .html file")
    ap.add_argument("-o", "--out", help="where to write the result "
                                        "(default: <source>.enhanced.html)")
    ap.add_argument("-m", "--model", help=f"default: {DEFAULT_MODEL}")
    ap.add_argument("-i", "--instructions", help="extra direction for the rewrite")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompt and report its size without calling anything")
    ap.add_argument("--list-models", action="store_true",
                    help="ask the API which models this key can call, and exit")
    args = ap.parse_args(argv)

    if args.list_models:
        for name in list_models():
            print(name)
        return 0

    if not args.source:
        ap.error("a source is required unless --list-models is given")

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
        prompt = build_prompt(skeleton, len(assets), args.instructions,
                              describe_assets(assets))
        print(f"page      {len(html):>9,} chars")
        print(f"prompt    {len(prompt):>9,} chars  (~{len(prompt)//4:,} tokens)")
        print(f"assets    {len(assets):>9,} held back, "
              f"{100.0 * (len(html) - len(skeleton)) / max(len(html), 1):.1f}% of the page")
        print(f"key       {'present' if is_configured() else 'MISSING - see .env.example'}")
        return 0

    out = args.out or (os.path.splitext(args.source)[0] + ".enhanced.html")
    started = time.time()

    def note(attempt, total, code, delay):
        print(f"  {code} from the model (attempt {attempt}/{total}); "
              f"retrying in {delay:.0f}s", flush=True)

    try:
        result = enhance_html(html, model=args.model, extra=args.instructions,
                              on_retry=note)
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
