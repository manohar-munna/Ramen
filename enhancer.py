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
import html
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

# Distinctive enough that a model will carry it through verbatim, and short enough that
# it costs nothing. Deliberately not a URL: nothing should try to fetch it.
_PLACEHOLDER = "RAMEN_ASSET_%d"
_PLACEHOLDER_RE = re.compile(r"RAMEN_ASSET_(\d+)")
_DATA_URI_RE = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\s]+")
_BLANK_PIXEL = ("data:image/gif;base64,"
                "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")

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
# Tried in turn when the requested one has nothing left on any key. Ordered by how good
# the page comes out, not by speed: the flash models first, then the lite ones, which
# are weaker but hold a separate allowance. That last part is the whole point -- with
# every flash model spent for the day a run used to fail outright, while four lite
# models sat there answering in a second. gemini-flash-latest is deliberately absent,
# having returned 503 to a full page four times running while answering a one-line
# prompt instantly.
FALLBACK_MODELS = (
    "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash-preview",
    "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-2.5-flash-lite",
)
_ENDPOINT = ("https://generativelanguage.googleapis.com/v1beta/models/"
             "{model}:generateContent")


# The model that last answered, so callers can report what actually did the work
# rather than what was asked for.
_LAST_MODEL = [""]


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


def missing_markers(html: str, n_assets: int) -> List[int]:
    """Indices of assets whose marker does not appear in `html`."""
    seen = {int(m) for m in _PLACEHOLDER_RE.findall(html)}
    return [i for i in range(n_assets) if i not in seen]


def marker_context(skeleton: str, index: int, span: int = 220) -> str:
    """The text around a marker in the original skeleton, as a one-line hint.

    Telling the model an image is missing is not much use on its own; telling it what
    the image sat next to is, because that is the question it has to answer to put it
    back in the right place.
    """
    m = re.search(r"RAMEN_ASSET_%d\b" % index, skeleton)
    if not m:
        return ""
    lo = max(0, m.start() - span)
    hi = min(len(skeleton), m.end() + span)
    return " ".join(skeleton[lo:hi].split())


def restore_assets(html: str, assets: List[str]) -> Tuple[str, List[int]]:
    """Puts the data URIs back. Returns the HTML and any markers the model lost.

    A dropped marker is worth reporting rather than hiding: it means the model deleted an
    image, and the caller should be able to say so instead of silently shipping a page
    with content missing.
    """
    missing = missing_markers(html, len(assets))

    def swap(match: re.Match) -> str:
        i = int(match.group(1))
        if 0 <= i < len(assets):
            return assets[i]
        # A marker outside the range it was given is one the model made up. Left as it
        # was written it becomes a relative URL, and the page then asks the server for
        # /RAMEN_ASSET_23 -- hundreds of 404s and a broken image for each. A transparent
        # pixel is the honest substitute: there was never an image behind it.
        return _BLANK_PIXEL

    return _PLACEHOLDER_RE.sub(swap, html), missing


_FONT_FACE_RE = re.compile(r"@font-face\s*\{[^}]*\}", re.I)


def strip_font_faces(html: str) -> str:
    """The page without its embedded typeface.

    The engine carries its font inside the page as base64 so the reconstruction looks
    the same everywhere. Sent to the model, that is 235,000 characters -- about 58,800
    tokens, three times the whole of the page's actual markup -- describing nothing it
    can use. It comes out before the page goes anywhere and goes back in on the way out.
    """
    return _FONT_FACE_RE.sub("", html)


def at_origin(html: str) -> str:
    """The engine's page on its own, at the top left, exactly the size it was measured.

    The export is made for reading in a browser: the page sits on a grey desk with 24px
    of padding round it and a drop shadow. Rendered at the screenshot's size, that puts
    every element 24px right of and below where it belongs, on a grey frame -- the page
    scored 56% against its own screenshot for no reason but that, and the model was
    shown the shifted version as the geometry to keep. The evaluation tool had always
    stripped the padding; this path never did.
    """
    override = ("<style>html,body{margin:0!important;padding:0!important;"
                "background:transparent!important}body{gap:0!important}"
                ".pdf-page{box-shadow:none!important;margin:0!important}</style>")
    at = html.lower().find("</head>")
    return (html[:at] + override + html[at:]) if at >= 0 else (override + html)


def with_fonts(html: str) -> str:
    """Puts the engine's embedded typeface back into a page that has lost it."""
    try:
        from engine import embedded_font_css
        css = embedded_font_css()
    except Exception:
        return html
    if not css or "@font-face" in html:
        return html
    tag = "<style>%s</style>" % css
    at = html.lower().find("</head>")
    return (html[:at] + tag + html[at:]) if at >= 0 else (tag + html)


_ROLE_RE = re.compile(r'data-role="([^"]+)"')

# Describe what the engine actually recorded, not what it might imply. "backdrop" here
# means only "a photographic or gradient region kept as pixels" -- calling it a section
# background in the prompt read as an instruction, and a hero illustration came back
# tiled behind the entire page with the text unreadable on top of it. The skeleton
# already carries each image's measured position, so the model can see where a picture
# sat; it only needs to know which markers are photographs and which are small graphics.
_ROLE_NOTES = {
    "backdrop": "photograph or gradient, kept as pixels",
    "artwork": "small graphic or icon",
    "illustration": "graphic or 3D illustration crop from screenshot",
    "logo": "logo or brand mark crop from screenshot",
    "icon": "icon crop from screenshot",
    "vector": "vector graphic / shape crop from screenshot",
    "graphic": "graphic element crop from screenshot",
    "full_screenshot": "full original page screenshot / background reference",
}


def asset_roles(html: str, assets: List[str]) -> Dict[int, str]:
    """Maps each stripped asset back to the data-role of the element that carried it."""
    # The role sits on the wrapping div and the data URI on an <img> inside it, so the
    # two are never in the same tag. Walk the document instead: each role claims the
    # assets that appear before the next one does.
    by_uri = {uri: i for i, uri in enumerate(assets)}
    roles: Dict[int, str] = {}
    marks = [(m.start(), m.group(1)) for m in _ROLE_RE.finditer(html)]
    for m in _DATA_URI_RE.finditer(html):
        i = by_uri.get("".join(m.group(0).split()))
        if i is None or i in roles:
            continue
        owner = ""
        for pos, role in marks:
            if pos > m.start():
                break
            owner = role
        if owner:
            roles[i] = owner
    return roles


def describe_assets(assets: List[str], max_colours: int = 3,
                    roles: Optional[Dict[int, str]] = None,
                    boxes: Optional[Dict[int, Tuple[int, int, int, int]]] = None) -> List[str]:
    """One line per held-back image: its size, colours, and position."""
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
            note = _ROLE_NOTES.get((roles or {}).get(i, ""), "")
            loc = ""
            if boxes and i in boxes:
                bx0, by0, bx1, by1 = boxes[i]
                loc = f"  at (x={bx0}, y={by0})"
            lines.append(f"{_PLACEHOLDER % i}  {w}x{h}  {' '.join(names)}{loc}"
                         + (f"  -- {note}" if note else ""))
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



# ---------------------------------------------------------------- flattening rasters

# A reconstruction's artwork is deliberately fragmentary: the residual pass cuts out
# exactly the pixels CSS could not explain, which on a photographic page means dozens of
# small crops that only add up to a picture because each one is pinned to an exact
# coordinate. That is correct for a faithful render and useless for a rewrite -- the
# moment those fragments enter normal flow they scatter, and a logistics page came back
# with its brand mark floating over a shipping container and a caption three times its
# proper size.
#
# So compose them first. Fragments that overlap, or sit close enough to be reading as one
# picture, are painted onto a single canvas in paint order and handed over as one image.
# The model then receives a handful of real pictures with real aspect ratios instead of a
# pile of jigsaw pieces, and laying those out is a job it can actually do.
FLATTEN_GAP = 12.0          # px at reference width; fragments nearer than this join up
FLATTEN_MIN_GROUP = 2       # a lone fragment is left exactly as it was


def _bbox_of(elem) -> Tuple[float, float, float, float]:
    b = list(elem.bbox or [0, 0, 0, 0]) + [0, 0, 0, 0]
    return float(b[0]), float(b[1]), float(b[2]), float(b[3])


def _group_fragments(boxes: List[Tuple[float, float, float, float]],
                     gap: float, blocked=None) -> List[List[int]]:
    """Union-find over boxes that touch once grown by `gap`.

    `blocked(i, j)` may veto a pair. Two fragments can only be composed onto one canvas
    if nothing paints between them, so the caller uses it to keep the page's layering.
    """
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(boxes)):
        ax0, ay0, ax1, ay1 = boxes[i]
        for j in range(i + 1, len(boxes)):
            bx0, by0, bx1, by1 = boxes[j]
            if (ax0 - gap < bx1 and bx0 - gap < ax1
                    and ay0 - gap < by1 and by0 - gap < ay1):
                if blocked and blocked(i, j):
                    continue
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _reparent_is_safe(elements, host_id: str, moved_index: int,
                      x0: float, y0: float, x1: float, y1: float) -> bool:
    """Whether nesting an element inside `host_id` leaves the page looking the same.

    Nesting is not free. A child paints with its parent, so moving an element earlier in
    the document moves it earlier in the paint order, and anything that used to sit
    between the two now covers it. Measured before this existed: five of the eleven
    reference pages rendered differently after flattening, by as much as 237 levels on a
    channel, and the splitting step was most of it.
    """
    host_at = None
    for i, e in enumerate(elements):
        if e.id == host_id:
            host_at = i
            break
    if host_at is None or host_at > moved_index:
        return False                      # the host paints after it; nesting reorders
    for k in range(host_at + 1, moved_index):
        other = elements[k]
        if other.type == "rect" and (other.box is None
                                     or not other.box.backgroundColor):
            continue                      # paints nothing
        if other.type == "text":
            continue                      # text is emitted above everything regardless
        ox0, oy0, ox1, oy1 = _bbox_of(other)
        if ox0 < x1 and x0 < ox1 and oy0 < y1 and y0 < oy1:
            return False                  # this would end up on top of the moved element
    return True


def _innermost_container(elements, x0: float, y0: float, x1: float, y1: float,
                         pad: float = 2.0, moved_index: Optional[int] = None
                         ) -> Optional[str]:
    """Id of the smallest painted box that encloses this rectangle, if any."""
    best_id, best_area = None, None
    for e in elements:
        if e.type != "rect":
            continue
        bx0, by0, bx1, by1 = _bbox_of(e)
        if (bx0 - pad <= x0 and by0 - pad <= y0
                and bx1 + pad >= x1 and by1 + pad >= y1):
            area = (bx1 - bx0) * (by1 - by0)
            # A box the same size as the picture is the picture's own frame, not a
            # section that holds it; either is fine, but prefer the tighter one.
            if area > 0 and (best_area is None or area < best_area):
                if moved_index is not None and not _reparent_is_safe(
                        elements, e.id, moved_index, x0, y0, x1, y1):
                    continue
                best_id, best_area = e.id, area
    return best_id


def _split_across_containers(elements, elem, min_share: float = 0.06):
    """Cuts an unparentable raster along the top-level boxes it crosses.

    A photograph that spans two columns has nothing that contains it, so it can only be
    emitted at the document root -- where a picture the width of the page is read as the
    page's background and put behind all the content. Usually it is not one picture at
    all: the region detector joined two photographs that happened to sit in the same
    horizontal band. Cutting it back along the containers it crosses gives each column
    its own image, each nested where it belongs.

    Returns replacement elements, or None to leave it alone.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    x0, y0, x1, y1 = _bbox_of(elem)
    w, h = int(round(x1 - x0)), int(round(y1 - y0))
    if w < 4 or h < 4:
        return None

    at = None
    for i, e in enumerate(elements):
        if e is elem or e.id == elem.id:
            at = i
            break
    if at is None:
        return None

    tops = [e for e in elements
            if e.type == "rect" and not getattr(e, "parentId", None)]
    pieces = []
    for host in tops:
        hx0, hy0, hx1, hy1 = _bbox_of(host)
        ix0, iy0 = max(x0, hx0), max(y0, hy0)
        ix1, iy1 = min(x1, hx1), min(y1, hy1)
        if ix1 - ix0 < 4 or iy1 - iy0 < 4:
            continue
        share = ((ix1 - ix0) * (iy1 - iy0)) / max((x1 - x0) * (y1 - y0), 1.0)
        if share < min_share:
            continue                     # a clipped corner, not a column of the picture
        if not _reparent_is_safe(elements, host.id, at, ix0, iy0, ix1, iy1):
            return None                  # cutting it up here would change the render
        pieces.append((host.id, ix0, iy0, ix1, iy1))

    if len(pieces) < 2:
        return None                      # nothing gained by cutting it up

    try:
        raw = base64.b64decode(elem.src.partition(",")[2])
        im = Image.open(io.BytesIO(raw)).convert("RGBA")
    except Exception:
        return None
    if im.size != (w, h):
        im = im.resize((w, h), Image.LANCZOS)

    out = []
    for n, (host_id, ix0, iy0, ix1, iy1) in enumerate(pieces, start=1):
        crop = im.crop((int(round(ix0 - x0)), int(round(iy0 - y0)),
                        int(round(ix1 - x0)), int(round(iy1 - y0))))
        bbox = crop.getbbox()
        if not bbox:
            continue                     # this column of the picture is empty
        crop = crop.crop(bbox)
        px0, py0 = ix0 + bbox[0], iy0 + bbox[1]
        cw, ch = crop.size
        if cw < 4 or ch < 4:
            continue
        buf = io.BytesIO()
        crop.save(buf, format="PNG", optimize=True)
        out.append(elem.model_copy(update={
            "id": f"{elem.id}-{n}",
            "bbox": [px0, py0, px0 + cw, py0 + ch],
            "src": "data:image/png;base64,"
                   + base64.b64encode(buf.getvalue()).decode("ascii"),
            "naturalWidth": float(cw),
            "naturalHeight": float(ch),
            "parentId": host_id,
        }))
    return out if len(out) >= 2 else None


def flatten_page_rasters(page, gap: float = FLATTEN_GAP):
    """Returns a copy of `page` whose overlapping image fragments are composed.

    Paint order is the element order, which the engine has already sorted by z-index, so
    compositing in that order reproduces what a browser would draw.
    """
    try:
        from PIL import Image
    except ImportError:
        return page

    elements = list(page.elements or [])
    idx = [i for i, e in enumerate(elements)
           if e.type == "image" and e.src and e.src.startswith("data:image")]
    if len(idx) < FLATTEN_MIN_GROUP:
        return page

    scale = max(float(page.width or 1400.0) / 1400.0, 0.5)
    boxes = [_bbox_of(elements[i]) for i in idx]

    # Merge on proximity, but never across the top-level containers of the page. A group
    # that spans two of them has no element that contains it, so the composite has to be
    # emitted at the document root -- and a root-level image 1235x836 on a 1297x1056 page
    # is, quite reasonably, read as the page's background and put behind everything.
    #
    # Grouping by immediate parent instead was tried and is worse: 17 images rather than
    # 7, colliding in flow exactly as the raw fragments had, three of them dropped. The
    # top-level ancestor is the boundary that matters, because it is the one that decides
    # whether anything can hold the result.
    by_id = {e.id: e for e in elements}

    _PAGE = object()          # one shared key, not one per homeless fragment

    def root_of(elem):
        seen = set()
        first = True
        while True:
            pid = getattr(elem, "parentId", None)
            if not pid or pid in seen or pid not in by_id:
                # A fragment with no container is not inside anything, so pairing it
                # with another of the same kind crosses no boundary. Returning its own
                # id here instead gave every one of them a group to itself: on the
                # woodnest page 99 of 107 rasters are unparented, 166 pairs of them sit
                # within the merge gap, and not one merge happened.
                return _PAGE if first else elem.id
            seen.add(pid)
            elem = by_id[pid]
            first = False

    groups: List[List[int]] = []
    by_root: Dict[object, List[int]] = {}
    for k, i in enumerate(idx):
        by_root.setdefault(root_of(elements[i]), []).append(k)
    # Compositing puts every fragment of a group at one position in the paint order, so
    # it is only faithful while nothing else is drawn between them. Where something is --
    # a card painted over one crop and under the next -- merging silently reorders the
    # page. Checked across the references rather than assumed: before this, five of the
    # eleven rendered differently after flattening, by up to 237 levels on a channel.
    def paints_between(a: int, b: int) -> bool:
        lo, hi = sorted((idx[a], idx[b]))
        ux0 = min(boxes[a][0], boxes[b][0])
        uy0 = min(boxes[a][1], boxes[b][1])
        ux1 = max(boxes[a][2], boxes[b][2])
        uy1 = max(boxes[a][3], boxes[b][3])
        for k in range(lo + 1, hi):
            other = elements[k]
            if other.type == "image":
                continue          # another fragment: it joins the group or stays in order
            if other.type == "rect" and (other.box is None
                                         or not other.box.backgroundColor):
                continue          # paints nothing, so it cannot come between them
            ox0, oy0, ox1, oy1 = _bbox_of(other)
            if ox0 < ux1 and ux0 < ox1 and oy0 < uy1 and uy0 < oy1:
                return True
        return False

    for members in by_root.values():
        sub = [boxes[m] for m in members]
        veto = lambda a, b: paints_between(members[a], members[b])   # noqa: E731
        for grp in _group_fragments(sub, gap * scale, blocked=veto):
            groups.append([members[g] for g in grp])

    replacements: Dict[int, object] = {}
    drop: set = set()
    made = 0

    for members in groups:
        if len(members) < FLATTEN_MIN_GROUP:
            continue
        members.sort()                                  # paint order
        gx0 = min(boxes[m][0] for m in members)
        gy0 = min(boxes[m][1] for m in members)
        gx1 = max(boxes[m][2] for m in members)
        gy1 = max(boxes[m][3] for m in members)
        gw, gh = int(round(gx1 - gx0)), int(round(gy1 - gy0))
        if gw < 2 or gh < 2 or gw * gh > 40_000_000:    # a canvas nobody wants to hold
            continue

        canvas = Image.new("RGBA", (gw, gh), (0, 0, 0, 0))
        painted = 0
        for m in members:
            e = elements[idx[m]]
            x0, y0, x1, y1 = boxes[m]
            w, h = int(round(x1 - x0)), int(round(y1 - y0))
            if w < 1 or h < 1:
                continue
            try:
                raw = base64.b64decode(e.src.partition(",")[2])
                im = Image.open(io.BytesIO(raw)).convert("RGBA")
            except Exception:
                continue
            # The renderer stretches each crop to its box, so match that here rather
            # than pasting at natural size: several of them are not the same.
            if im.size != (w, h):
                im = im.resize((w, h), Image.LANCZOS)
            canvas.alpha_composite(im, (int(round(x0 - gx0)), int(round(y0 - gy0))))
            painted += 1

        if painted < FLATTEN_MIN_GROUP:
            continue

        # Trim fully transparent margins, so the box the model sees is the picture.
        bbox = canvas.getbbox()
        if bbox and bbox != (0, 0, gw, gh):
            canvas = canvas.crop(bbox)
            gx0, gy0 = gx0 + bbox[0], gy0 + bbox[1]
            gw, gh = canvas.size
        if gw < 2 or gh < 2:
            continue

        buf = io.BytesIO()
        canvas.save(buf, format="PNG", optimize=True)
        uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

        keep = idx[members[0]]
        host = elements[keep]
        # The composite covers more ground than any one fragment did, so inheriting a
        # fragment's parent can leave it hanging outside the box that holds it. Ask which
        # element actually contains the new rectangle, smallest first: that is the section
        # it belongs to, and nesting it there is what stops it becoming a page backdrop.
        merged = host.model_copy(update={
            "bbox": [gx0, gy0, gx0 + gw, gy0 + gh],
            "src": uri,
            "naturalWidth": float(gw),
            "naturalHeight": float(gh),
            "parentId": _innermost_container(elements, gx0, gy0, gx0 + gw, gy0 + gh,
                                             moved_index=keep),
        })
        replacements[keep] = merged
        for m in members[1:]:
            drop.add(idx[m])
        made += 1

    out = [replacements.get(i, e) for i, e in enumerate(elements) if i not in drop]

    # Anything still homeless spans more than one top-level box; cut it along them.
    split: List[object] = []
    changed = False
    for e in out:
        if (e.type == "image" and e.src and not getattr(e, "parentId", None)
                and e.src.startswith("data:image")):
            parts = _split_across_containers(out, e)
            if parts:
                split.extend(parts)
                changed = True
                continue
        split.append(e)

    if not made and not changed:
        return page
    return page.model_copy(update={"elements": split})


_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "msedge", "edge",
)


def _find_chrome() -> Optional[str]:
    import shutil
    for c in _CHROME_CANDIDATES:
        if os.path.isfile(c):
            return c
        found = shutil.which(c)
        if found:
            return found
    return None


def renders_identically(before, after, tolerance: int = 2) -> Optional[bool]:
    """Whether two pages draw the same thing. None when it cannot be checked.

    Flattening ought to be invisible -- compositing is what the browser was doing
    anyway -- but it is not invisible by construction. The renderer leaves a container
    on `z-index: auto` exactly when it has children, so merging or re-parenting changes
    who has children and silently restacks things a long way from the edit. Analytic
    guards for that were tried and were both too strict and too loose in the same run:
    they refused a page that was provably fine and passed one that was 114 levels out.
    Rendering both and looking is the only answer that is actually about the pixels.
    """
    chrome = _find_chrome()
    if not chrome:
        return None
    try:
        import subprocess
        import tempfile
        import numpy as np
        from PIL import Image
        from engine import DocumentData, HTMLRenderer
    except Exception:
        return None

    w = int(float(before.width or 1400))
    h = int(float(before.height or 1000))
    shots = []
    with tempfile.TemporaryDirectory() as tmp:
        for tag, page in (("a", before), ("b", after)):
            html = HTMLRenderer.render_document(
                DocumentData(title=tag, pageCount=1, pages=[page]),
                editable=False, interactive=False)
            html = html.replace("padding: 24px;", "padding: 0;").replace(
                "gap: 24px;", "gap: 0;")
            hp = os.path.join(tmp, tag + ".html")
            with open(hp, "w", encoding="utf-8") as fh:
                fh.write(html)
            sp = os.path.join(tmp, tag + ".png")
            try:
                subprocess.run([chrome, "--headless", "--disable-gpu",
                                "--hide-scrollbars", "--force-device-scale-factor=1",
                                "--window-size=%d,%d" % (w, h),
                                "--virtual-time-budget=5000",
                                "--screenshot=" + sp, "file:///" + hp.replace("\\", "/")],
                               check=True, capture_output=True, timeout=180)
                shots.append(np.asarray(Image.open(sp).convert("RGB"), dtype=np.int16))
            except Exception:
                return None
    if len(shots) != 2:
        return None
    a, b = shots
    hh, ww = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    return int(np.abs(a[:hh, :ww] - b[:hh, :ww]).max()) <= tolerance


def render_html(html: str, width: int = 1400, height: int = 2400,
                timeout: int = 180) -> Optional[Tuple[str, str]]:
    """Screenshots a page and returns it as (mime, base64), or None if it cannot.

    Entrance animations are settled first. Without that, a page that fades its sections
    in comes back half transparent and every comparison reads it as a fault -- which it
    did, and cost an afternoon before the cause was obvious.
    """
    chrome = _find_chrome()
    if not chrome:
        return None
    # Animations are run to their end rather than switched off. `animation:none` leaves
    # a fade-in at its first frame, invisible, so this used to force `opacity:1` on
    # every element as well -- which made anything meant to be translucent opaque, both
    # for the score and in the picture the model is asked to critique. A zero-length
    # animation lands on its last frame, which is what a person actually sees.
    settled = html.replace(
        "</head>",
        "<style>*,*::before,*::after{animation-duration:0s!important;"
        "animation-delay:0s!important;transition:none!important}</style></head>", 1)
    try:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            hp = os.path.join(tmp, "page.html")
            with open(hp, "w", encoding="utf-8") as fh:
                fh.write(settled)
            sp = os.path.join(tmp, "page.png")
            subprocess.run([chrome, "--headless", "--disable-gpu", "--hide-scrollbars",
                            "--force-device-scale-factor=1",
                            "--window-size=%d,%d" % (width, height),
                            "--virtual-time-budget=8000",
                            "--screenshot=" + sp, "file:///" + hp.replace("\\", "/")],
                           check=True, capture_output=True, timeout=timeout)
            with open(sp, "rb") as fh:
                return "image/png", base64.b64encode(fh.read()).decode("ascii")
    except Exception:
        return None


def _shot_dims(shot: Optional[Tuple[str, str]]) -> Tuple[int, int]:
    """Pixel size of the reference screenshot (width, height), or (1400, 1000) default."""
    if not shot:
        return 1400, 1000
    try:
        from PIL import Image
        raw = base64.b64decode(shot[1])
        img = Image.open(io.BytesIO(raw))
        return img.size
    except Exception:
        return 1400, 1000


def calculate_ssim_score(shot: Tuple[str, str], render: Tuple[str, str]) -> float:
    """How closely a render matches the screenshot, 0 to 100, in colour.

    Structural similarity on each colour channel, averaged. It was grayscale SSIM, which
    cannot see colour at all: a headline drawn black instead of purple scored about the
    same, so a round that corrected a colour and moved something by a pixel read as a
    regression and was thrown away. It also went through FidelityChecker, which builds,
    blends and PNG-encodes a heatmap on every call -- twice a round -- to be read for
    one number.
    """
    if not shot or not render:
        return 0.0
    try:
        import cv2
        import numpy as np
        from skimage.metrics import structural_similarity as _ssim
        a = cv2.imdecode(np.frombuffer(base64.b64decode(shot[1]), np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.frombuffer(base64.b64decode(render[1]), np.uint8),
                         cv2.IMREAD_COLOR)
        if a is None or b is None:
            return 0.0
        if a.shape[:2] != b.shape[:2]:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        return float(_ssim(a, b, channel_axis=2)) * 100.0
    except Exception as e:
        _LOG.append("ssim: comparison failed (%s)" % e)
    return 0.0


_CONTROL_RE = re.compile(r"<(button|input|select|textarea)[\s>/]", re.I)

# How far a version's text may move from the page's, in letter trigrams. A correction
# swaps letters -- about as many lost as gained -- while a dropped line only loses them
# and invented prose only adds them. So the net change, one way or the other, is held
# tight, and the gross change, which corrections do produce, only loosely. A single
# percentage could not do both: 5% let a whole sentence of body copy go, and anything
# tight enough to catch that refused a page with a lot of misread text put right.
CONTENT_NET_FLOOR = 36          # trigrams, or...
CONTENT_NET_SHARE = 0.02        # ...this share of the page, whichever is larger
CONTENT_GROSS_FLOOR = 60
CONTENT_GROSS_SHARE = 0.15


def _count_controls(html: str) -> int:
    """Real interactive elements. A rewrite that turns them all into <a> has lost."""
    return len(_CONTROL_RE.findall(html))


class _TextNodes(HTMLParser):
    """The text a reader would see, one chunk per text node."""

    _HIDDEN = ("head", "script", "style", "title", "template")

    def __init__(self):
        super().__init__()
        self.chunks: List[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._HIDDEN:
            self._depth += 1

    def handle_endtag(self, tag):
        if tag in self._HIDDEN and self._depth:
            self._depth -= 1

    def handle_data(self, data):
        if not self._depth and data.strip():
            self.chunks.append(data)


def _text_trigrams(html: str) -> Counter:
    """Letter trigrams of the visible text, taken per text node.

    Per node, so that moving a block elsewhere does not count as changing its words.
    Letters and digits only, so that splitting a run-together word or re-spacing it
    costs nothing: "yourbuyersread." and "your buyers read." differ by two trigrams.
    """
    parser = _TextNodes()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    grams: Counter = Counter()
    for chunk in parser.chunks:
        t = "".join(ch for ch in chunk.lower() if ch.isalnum())
        if 0 < len(t) < 3:
            grams[t] += 1
        for i in range(len(t) - 2):
            grams[t[i:i + 3]] += 1
    return grams


def keeps_the_content(before: str, after: str) -> Optional[str]:
    """Why a version should be refused, or None if it may be accepted.

    Versions are judged on how their render compares with the screenshot, and a picture
    is a weak judge of whether the words survived: a paragraph replaced by a flat panel
    of the same colour scores about the same. This is the other vote.

    It used to count whole words, which refused exactly the corrections the model is
    asked to make -- "yourbuyersread." written as three words and "Linkedln" as
    "LinkedIn" twice is five invented words by that measure, over a limit of four.
    Trigrams per text node see those as the small edits they are, and still see a lost
    paragraph as hundreds.
    """
    a, b = _text_trigrams(before), _text_trigrams(after)
    total = sum(a.values())
    lost = sum((a - b).values())
    gained = sum((b - a).values())
    net = max(CONTENT_NET_FLOOR, int(total * CONTENT_NET_SHARE))
    gross = max(CONTENT_GROSS_FLOOR, int(total * CONTENT_GROSS_SHARE))
    if lost - gained > net:
        return "it dropped text (%d of the page's %d letter groups are gone)" % (lost, total)
    if gained - lost > net:
        return "it invented text (%d letter groups that are not on the page)" % gained
    if lost + gained > gross:
        return ("it rewrote the text (%d letter groups changed, of %d)"
                % (lost + gained, total))
    if _count_controls(after) < _count_controls(before):
        return "it turned %d control(s) into something else" % (
            _count_controls(before) - _count_controls(after))
    return None


def flatten_document(doc, verify: bool = True):
    """`flatten_page_rasters` across every page, keeping only what renders the same.

    A page that cannot be flattened without changing how it looks is handed over
    unflattened. That is worse input for the rewrite -- more fragments to place -- but
    it is honest input, and a picture quietly restacked is worse than a fiddly one.
    """
    pages = []
    for page in (doc.pages or []):
        flat = flatten_page_rasters(page)
        if flat is page or not verify:
            pages.append(flat)
            continue
        same = renders_identically(page, flat)
        if same is False:
            _LOG.append("flatten: %d fragments left alone on one page; composing them "
                        "would have changed the render"
                        % sum(1 for e in page.elements if e.type == "image"))
            pages.append(page)
        else:
            pages.append(flat)
    return doc.model_copy(update={"pages": pages})

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
- Sections must not overlap or be stacked on top of one another. Where the input has
  one element inside the bounds of another, that is containment: nest it, rather than
  emitting two blocks that collide. Read each element's measured `left`/`top`/`width`/
  `height` to work out what sat inside what.
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


REFINE_PROMPT = """\
You rewrote a page. Two screenshots are attached, in this order:

1. THE TARGET -- the original design this is supposed to look like.
2. YOUR RESULT -- your own rewrite, rendered in a browser just now.

Compare them and fix the differences. Work on what a person would notice first:
sections in the wrong order or overlapping, blocks that should sit side by side and
are stacked (or the reverse), spacing and alignment that do not match, type that is
much too large or too small, an image at the wrong size or in the wrong place, and
anything from the target that is missing or clearly out of position.

Rules unchanged from before:
- Every piece of visible text stays exactly as it is in the HTML below. Do not reword,
  translate, correct spelling, or add text of your own -- including words the target
  shows cut off behind something, which are genuinely cut off in the design.
- Every `RAMEN_ASSET_<n>` marker must survive, in an `src` attribute. There are
  %(n_assets)d of them.
- Keep the layout flowing: flexbox and grid, no absolute positioning, responsive down
  to 480px, hover and focus states, and transitions.

If a difference comes from the reconstruction rather than from your layout -- a colour
the HTML simply does not contain, a piece of artwork the engine never extracted -- leave
it alone. You cannot invent what was not given to you.

Return the complete corrected HTML document and nothing else. No explanation, no
markdown fences. Start with `<!DOCTYPE html>`.

Your current document:

%(html)s
"""


def build_refine_prompt(current: str, n_assets: int) -> str:
    return REFINE_PROMPT % {"html": current, "n_assets": n_assets}


REPAIR_PROMPT = """\
You rewrote an HTML page and some images were lost: their `RAMEN_ASSET_<n>` markers are
not in your output, so those pictures are gone from the page.

Do NOT rewrite the document. Just say where each missing image belongs.

For each marker below, reply with one line:

    RAMEN_ASSET_<n> ||| <anchor>

where `<anchor>` is a short run of text copied EXACTLY from the document below, between
40 and 120 characters, that appears exactly once. The image will be inserted immediately
after that anchor. Choose an anchor that puts the picture where it belongs -- inside the
right section, next to the content it goes with.

Reply with nothing but those lines. No explanation, no markdown fences.

Missing images:

%(missing)s

The document:

%(html)s
"""


_LOG: List[str] = []          # diagnostics the caller may want to print
_REPAIR_LINE = re.compile(r"(RAMEN_ASSET_(\d+))\s*\|\|\|\s*(.+?)\s*$", re.M)


def build_repair_prompt(current: str, missing: List[int], skeleton: str,
                        asset_lines: Optional[List[str]] = None) -> str:
    by_index = {}
    for line in (asset_lines or []):
        parts = line.split()
        if parts:
            m = _PLACEHOLDER_RE.match(parts[0])
            if m:
                by_index[int(m.group(1))] = line

    notes = []
    for i in missing:
        notes.append("- " + (by_index.get(i) or (_PLACEHOLDER % i)))
        ctx = marker_context(skeleton, i)
        if ctx:
            notes.append("  it sat here in the original: %s" % ctx)
    return REPAIR_PROMPT % {"missing": "\n".join(notes), "html": current}


def apply_repair(current: str, reply: str, missing: List[int]) -> Tuple[str, List[int]]:
    """Inserts each missing marker after the anchor the model named.

    The model decides *where*; the insertion itself is done here. Asking it to hand back
    a corrected copy of the whole document meant regenerating seventy thousand characters
    to add four image tags, and it simply did not -- the same four came back missing
    every time. An anchor is a dozen tokens and can be checked before it is trusted.
    """
    placed, rejected = [], []
    for whole, index, anchor in _REPAIR_LINE.findall(reply):
        i = int(index)
        if i not in missing:
            continue
        anchor = anchor.strip().strip('"').strip("'")
        hits = current.count(anchor)
        if len(anchor) < 8 or hits != 1:
            # Not found or ambiguous. Worth counting rather than swallowing: an anchor
            # that never matches means the model is quoting text it did not write.
            rejected.append((i, len(anchor), hits))
            continue
        tag = f'<img src="{whole}" alt="" loading="lazy">'
        at = current.index(anchor) + len(anchor)
        current = current[:at] + tag + current[at:]
        placed.append(i)
    if rejected:
        _LOG.append("repair: rejected %d anchor(s): %s"
                    % (len(rejected), ", ".join(
                        "asset %d (len %d, %d matches)" % r for r in rejected)))
    return current, [i for i in missing if i not in placed]


REFERENCE_NOTE = """\

THE SCREENSHOT
A screenshot of the page this HTML was reconstructed from is attached. It is the
authority on what the page is meant to look like, and the HTML below is only a
machine's approximation of it -- where the two disagree, believe the screenshot.

Use it to work out what the HTML cannot tell you: which blocks are one component,
what is a header or a card or a footer, what the reading order is, which things are
aligned with each other, and where the real margins and rhythm are. Several of the
HTML's elements are fragments of one visual thing; the screenshot is how you can tell.

It does not license inventing anything. Text still comes from the HTML, character for
character, including words the screenshot shows cut off behind something -- those are
genuinely cut off in the design and must stay that way.
"""


def build_prompt(skeleton: str, n_assets: int, extra: Optional[str] = None,
                 asset_lines: Optional[List[str]] = None,
                 has_reference: bool = False) -> str:
    prompt = PROMPT % {
        "html": skeleton,
        "n_assets": n_assets,
        "max_asset": max(n_assets - 1, 0),
        "assets": "\n".join(asset_lines or []) or "(none)",
    }
    if has_reference:
        head, sep, tail = prompt.partition("OUTPUT\n")
        prompt = head + REFERENCE_NOTE + "\n" + sep + tail
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

def redact_keys(text: str) -> str:
    """Blanks anything shaped like a credential.

    The API quotes the key back inside its own error text -- "Consumer
    'api_key:AIzaSy...' has been suspended" -- so any message printed, logged or shown
    to a user carries a live secret unless it is taken out first. Found by reading the
    output of the key checker, which had just put two of them on screen.
    """
    out = []
    i = 0
    while i < len(text):
        if text.startswith("AIza", i) or text.startswith("AQ.", i):
            j = i
            while j < len(text) and (text[j].isalnum() or text[j] in "._-"):
                j += 1
            if j - i >= 20:
                out.append(text[i:i + 6] + "...[redacted]")
                i = j
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def short_reason(err: Exception) -> str:
    """A one-line reason fit for a status bar.

    The API answers with a JSON body, so slicing the first characters of the exception
    yields 'Gemini returned 429: {' -- technically the error and of no use to anyone.
    Parse it and take the message.
    """
    text = str(err)

    # The body is clipped to 600 characters upstream, so it is usually invalid JSON by
    # the time it gets here -- which made json.loads fail and the whole thing fall back
    # to "Gemini returned 429: {", the exact output this function exists to prevent.
    # Read the fields out of the text directly and do not depend on it parsing.
    def field(name):
        key = '"%s"' % name
        at = text.find(key)
        if at < 0:
            return ""
        at = text.find(":", at + len(key))
        if at < 0:
            return ""
        rest = text[at + 1:].lstrip()
        if rest.startswith('"'):
            out, i = [], 1
            while i < len(rest) and rest[i] != '"':
                if rest[i] == "\\" and i + 1 < len(rest):
                    out.append(" " if rest[i + 1] == "n" else rest[i + 1])
                    i += 2
                    continue
                out.append(rest[i])
                i += 1
            return "".join(out).strip()
        return rest.split(",")[0].split("}")[0].strip()

    msg = field("message")
    code = field("code")
    low = msg.lower()
    if code == "429" or "quota" in low or "rate limit" in low:
        per_day = "per day" in low or "free_tier_requests" in low
        return ("this model's free-tier allowance is used up"
                + (" for today" if per_day else " for the moment"))
    if code == "503" or "high demand" in low:
        return "the model is busy"
    if msg:
        return redact_keys(msg)[:160]
    return redact_keys(text.splitlines()[0])[:160]


# What has already been found not to work, remembered for the life of the process so a
# run does not spend forty seconds rediscovering it on every call. Keyed by (key, model)
# because the free tier counts per model: the same credential can be spent on one and
# fine on the next, which is exactly what happened -- one key with nothing left for
# gemini-3-flash and a full allowance on gemini-3.5-flash.
_SPENT: Dict[Tuple[str, str], Tuple[float, str]] = {}
# A rejected credential is rejected everywhere, so it is remembered without a model.
_DEAD_KEYS: Dict[str, str] = {}
# How long a spent mark is believed. The allowance it describes resets daily and this
# server is expected to stay up across the reset, so a mark kept for the life of the
# process eventually describes a quota that came back hours ago.
SPENT_TTL = 3600.0


def api_keys(explicit: Optional[str] = None) -> List[str]:
    """Every credential available, in the order they should be tried."""
    if explicit:
        return [explicit]
    load_dotenv()
    out: List[str] = []
    for source in ("GEMINI_API_KEYS", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        for part in re.split(r"[,\s]+", os.environ.get(source, "") or ""):
            part = part.strip()
            if part and part not in out:
                out.append(part)
    return out


def live_keys(model: str = "", explicit: Optional[str] = None) -> List[str]:
    """Keys that might still work for this model, in order."""
    keys = [k for k in api_keys(explicit) if k not in _DEAD_KEYS]
    now = time.time()
    fresh = [k for k in keys
             if now - _SPENT.get((k, model), (0.0, ""))[0] > SPENT_TTL]
    # If everything looks spent, try them anyway rather than refusing outright: a
    # per-minute ceiling clears on its own and the marks may simply be stale.
    return fresh or keys or api_keys(explicit)


def fresh_keys(model: str = "", explicit: Optional[str] = None) -> List[str]:
    """Keys that are neither rejected nor known to have nothing left on this model."""
    now = time.time()
    return [k for k in api_keys(explicit) if k not in _DEAD_KEYS
            and now - _SPENT.get((k, model), (0.0, ""))[0] > SPENT_TTL]


def _worth_asking(model_name: str, names: List[str], explicit: Optional[str]) -> bool:
    """Whether to try this model, or go straight to the next one.

    A model with nothing left on any key is skipped while there is another after it.
    The refinement loop makes a call per step, and each one used to start at the
    requested model and be refused -- a guaranteed 429, every call, from a model
    already known to be spent -- before walking on to the one that would answer.
    """
    return model_name == names[-1] or bool(fresh_keys(model_name, explicit))


def mark_spent(key: str, model: str, reason: str) -> None:
    """Records that this key has nothing left for this model, and when."""
    _SPENT[(key, model)] = (time.time(), reason)


def mark_dead(key: str, reason: str) -> None:
    """Records that this credential is not accepted at all, for any model."""
    _DEAD_KEYS[key] = reason


def models_to_try(model: Optional[str] = None) -> List[str]:
    """The requested model first, then the fallbacks, without repeats."""
    first = model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL
    out = [first]
    for m in FALLBACK_MODELS:
        if m not in out:
            out.append(m)
    return out


def key_label(key: str) -> str:
    """Enough of a key to tell two apart in a log, and no more."""
    return key[:6] + "..." + key[-4:] if len(key) > 12 else "key"


def _daily_allowance_gone(detail: str) -> bool:
    """Whether a 429 is the day's allowance rather than a ceiling on the minute."""
    return "per day" in detail or "_requests, limit:" in detail


def _key_verdict(code: int, detail: str) -> str:
    """"dead" if the credential is refused outright, "spent" if it is used up.

    A 429 is not one thing, and the body says which. The day's allowance being gone
    is worth remembering: nothing brings it back before the reset, and the run should
    move to another key. A ceiling on the minute clears in seconds, and treating that
    as spent abandoned a working key over a limit that had already lifted. Empty means
    neither -- retry this one rather than giving up on it.
    """
    if code in (401, 403):
        return "dead"
    if code == 429 and _daily_allowance_gone(detail):
        return "spent"
    return ""


def _api_key(explicit: Optional[str] = None) -> str:
    keys = live_keys("", explicit)
    key = keys[0] if keys else None
    if not key:
        raise EnhancementError(
            "No API key. Put GEMINI_API_KEYS=key1,key2,... in a .env file at the "
            "project root, or set GEMINI_API_KEY in the environment.")
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

# How many times to go back and ask for images the rewrite lost.
REPAIR_ATTEMPTS = 2

# A match good enough that another round is not worth a request. It is an early exit,
# not the goal: the loop stops when it stops improving, because a bar it cannot reach
# -- which 95% grayscale SSIM was, for every page, when the engine's own pixel-placed
# reconstruction averages 91% -- turns "until it matches" into "every round, every time".
TARGET_ACCURACY_SCORE = 98.0

# How many times to render the result and ask the model to close the gap with the
# original.
REFINE_ROUNDS = 5



def call_gemini(prompt: str, api_key: Optional[str] = None,
                model: Optional[str] = None, timeout: int = 300,
                attempts: int = RETRY_ATTEMPTS, on_retry=None,
                image: Optional[Tuple[str, str]] = None,
                images: Optional[List[Tuple[str, str]]] = None) -> str:
    """Sends one prompt, optionally with an image, and returns the text of the reply.

    `image` is (mime type, base64 payload). A screenshot costs about a thousand tokens
    against a skeleton of sixteen thousand, which is cheap for the only description of
    the page that is not second-hand.
    """
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for shot in ([image] if image else []) + list(images or []):
        if shot:
            parts.append({"inline_data": {"mime_type": shot[0], "data": shot[1]}})
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {
            # Low but not zero: layout decisions benefit from a little freedom, and
            # nothing here needs to be reproducible -- the faithful version already is.
            "temperature": 0.35,
            "maxOutputTokens": 65536,
        },
    }).encode("utf-8")

    # Walk the credentials and the models before walking the clock. A spent allowance
    # is not a transient failure -- waiting forty seconds and asking again only wastes
    # the time -- so a 429 moves to the next key, and when every key is spent for this
    # model it moves to the next model. Only a busy server earns a wait. Before this,
    # one exhausted model ended the run outright while the same key had a full allowance
    # on the next one along.
    payload = None
    last_error: Optional[EnhancementError] = None
    used_model = None
    names = models_to_try(model)
    for model_name in names:
        if not _worth_asking(model_name, names, api_key):
            continue
        keys = live_keys(model_name, api_key)
        for key in keys:
            give_up_on_key = False
            for attempt in range(1, max(1, attempts) + 1):
                try:
                    payload = _post(model_name, body, key, timeout)
                    used_model = model_name
                    break
                except _KeyProblem as kp:
                    last_error = kp.error
                    if kp.verdict == "dead":
                        mark_dead(key, kp.reason)
                    else:
                        mark_spent(key, model_name, kp.reason)
                    if on_retry:
                        on_retry(0, 0, "%s on %s: %s" % (
                            key_label(key), model_name, kp.reason), 0)
                    give_up_on_key = True
                    break
                except _Transient as t:
                    last_error = t.error
                    if attempt >= attempts:
                        give_up_on_key = True
                        break
                    # The server knows better than a doubling guess when it says so.
                    delay = t.retry_after or (RETRY_BACKOFF * (2 ** (attempt - 1)))
                    if on_retry:
                        on_retry(attempt, attempts, t.code or "a timeout", delay)
                    time.sleep(delay)
            if payload is not None or not give_up_on_key:
                break
        if payload is not None:
            break
    if payload is None:
        raise last_error or EnhancementError("No key or model could answer.")
    if used_model:
        _LAST_MODEL[0] = used_model

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
    def __init__(self, code, error, retry_after=None):
        self.code, self.error, self.retry_after = code, error, retry_after


class _KeyProblem(Exception):
    """This credential will not work here; another one, or another model, might."""

    def __init__(self, error, reason, verdict="spent"):
        self.error, self.reason, self.verdict = error, reason, verdict


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
            if _daily_allowance_gone(detail):
                hint = (sep + "The free tier's daily request allowance for this model is "
                        "gone; no amount of retrying will help until it resets. Try "
                        "another model (python enhancer.py --list-models) or enable "
                        "billing.")
            else:
                hint = (sep + "A rate limit rather than a bad key. Pro models are also "
                        "outside the free tier, so a new key gets 429 on every one of "
                        "them and works on flash.")
        else:
            hint = ""
        err = EnhancementError(
            f"Gemini returned {e.code}: {redact_keys(detail)}{hint}")
        verdict = _key_verdict(e.code, detail)
        if verdict:
            raise _KeyProblem(err, short_reason(err), verdict) from e
        # A 429 that survived the verdict above is a ceiling on the minute, which is
        # exactly what backing off is for. The daily kind never reaches here.
        if e.code in RETRY_STATUSES:
            wait = None
            m = re.search(r'"retryDelay"\s*:\s*"([\d.]+)s"', detail)
            if m:
                try:
                    wait = float(m.group(1))
                except ValueError:
                    wait = None
            raise _Transient(e.code, err, wait) from e
        raise err from e
    except urllib.error.URLError as e:
        # A dropped or refused connection is worth another try, same as a 503.
        raise _Transient(0, EnhancementError(
            f"Could not reach Gemini: {e.reason}")) from e
    except (TimeoutError, OSError) as e:
        # A read timeout is neither HTTPError nor URLError, so it escaped as a raw
        # traceback and killed the run outright. Generating a whole page legitimately
        # takes minutes, and waiting too long for one is the most ordinary transient
        # failure there is.
        raise _Transient(0, EnhancementError(
            f"Gemini timed out or the connection failed: {e}")) from e


# ---------------------------------------------------------------- the whole pass

def as_inline_image(source) -> Optional[Tuple[str, str]]:
    """(mime, base64) from a path, raw bytes, or a data URI. None if there is nothing."""
    if not source:
        return None
    if isinstance(source, tuple):
        return source
    if isinstance(source, bytes):
        return "image/png", base64.b64encode(source).decode("ascii")
    text = str(source)
    if text.startswith("data:"):
        head, _, payload = text.partition(",")
        mime = head[5:].split(";")[0] or "image/png"
        return mime, "".join(payload.split())
    if os.path.isfile(text):
        ext = os.path.splitext(text)[1].lower()
        mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp"}.get(ext, "image/png")
        with open(text, "rb") as fh:
            return mime, base64.b64encode(fh.read()).decode("ascii")
    return None


def reference_from_document(doc) -> Optional[Tuple[str, str]]:
    """The original screenshot, which every reconstructed page already carries."""
    for page in (getattr(doc, "pages", None) or []):
        got = as_inline_image(getattr(page, "originalImageSrc", None))
        if got:
            return got
    return None


def enhance_html(html: str, api_key: Optional[str] = None, model: Optional[str] = None,
                 extra: Optional[str] = None, timeout: int = 300,
                 on_retry=None, reference=None,
                 refine: int = REFINE_ROUNDS, on_round=None,
                 target_score: float = TARGET_ACCURACY_SCORE) -> Dict[str, object]:
    """Rewrites a reconstructed page. Returns the HTML and what happened to it.

    Never raises for a merely disappointing result -- a model that drops an image still
    produced something worth looking at -- but does report the loss so the caller can.
    """
    shot = as_inline_image(reference)
    # The page's own images, where the engine measured them. The embedded typeface comes
    # out too: it is 58,800 tokens of base64 and goes back in on the way out.
    skeleton, assets = strip_assets(html)
    skeleton = strip_font_faces(skeleton)
    asset_lines = describe_assets(assets, roles=asset_roles(html, assets))

    prompt = build_prompt(skeleton, len(assets), extra, asset_lines, has_reference=bool(shot))

    reply = _unfence(call_gemini(prompt, api_key=api_key, model=model, timeout=timeout,
                                 on_retry=on_retry, image=shot))
    if "<" not in reply:
        raise EnhancementError("Gemini's reply does not look like HTML: "
                               + reply[:200])

    # A dropped marker is a piece of the page nobody can see any more, and re-rolling the
    # whole rewrite to fix it throws away a good layout to chase an image. Ask for the
    # missing ones specifically instead, telling it what each one sat next to -- that is
    # the question it has to answer to put them back in the right place. Keep whichever
    # attempt lost the least, so a repair that makes things worse costs nothing.
    del _LOG[:]
    lost = missing_markers(reply, len(assets))
    lost_initially = list(lost)
    repairs = 0
    while lost and repairs < REPAIR_ATTEMPTS:
        repairs += 1
        try:
            answer = call_gemini(
                build_repair_prompt(reply, lost, skeleton, asset_lines),
                api_key=api_key, model=model, timeout=timeout, on_retry=on_retry)
        except EnhancementError:
            break                       # the first result is still worth returning
        fixed, still = apply_repair(reply, answer, lost)
        if len(still) >= len(lost):
            break                       # no anchor was usable; keep what we had
        reply, lost = fixed, still

    # ---- look at the result and close the gap -------------------------------------
    rounds_done = 0
    best_reply = reply
    best_score = 0.0
    w, h = _shot_dims(shot) if shot else (1400, 1000)

    for round_idx in range(max(0, refine)):
        if not shot:
            break                    # nothing to compare against
        current_html, _ = restore_assets(reply, assets)
        render = render_html(with_fonts(current_html), width=w, height=h)
        if not render:
            _LOG.append("refine: skipped, no headless browser available")
            break

        score = calculate_ssim_score(shot, render)
        if score > best_score:
            best_score = score
            best_reply = reply

        if on_round:
            on_round(rounds_done + 1, max(0, refine))

        if score >= target_score:
            _LOG.append("refine: reached target visual fidelity %.1f%% >= %.1f%%" % (score, target_score))
            break

        try:
            better = _unfence(call_gemini(
                build_refine_prompt(reply, len(assets)),
                api_key=api_key, model=model, timeout=timeout, on_retry=on_retry,
                images=[shot, render]))
        except EnhancementError as e:
            _LOG.append("refine: round %d failed (%s)" % (rounds_done + 1,
                                                          str(e).splitlines()[0][:80]))
            break
        if "<" not in better:
            continue

        test_html, _ = restore_assets(better, assets)
        test_render = render_html(with_fonts(test_html), width=w, height=h)
        if test_render:
            test_score = calculate_ssim_score(shot, test_render)
            # Judged against what the engine measured, not against the last accepted
            # round, so small losses cannot accumulate a round at a time.
            refused = keeps_the_content(skeleton, better)
            if refused:
                # The picture may well have improved; the page is still worse.
                _LOG.append("refine: round %d rejected, %s" % (rounds_done + 1, refused))
            # Only an improvement replaces the best. Accepting anything within a point
            # and a half of it let the working copy sink a little every round while
            # every later round went on correcting the worse page.
            elif test_score > best_score + MIN_GAIN:
                reply = better
                score = test_score
                if score > best_score:
                    best_score = score
                    best_reply = reply
                if score >= target_score:
                    rounds_done += 1
                    break
        elif not keeps_the_content(skeleton, better):
            reply = better

        rounds_done += 1

    enhanced, missing = restore_assets(best_reply, assets)
    enhanced = with_fonts(enhanced)
    return {
        "html": enhanced,
        "refine_rounds": rounds_done,
        "model": _LAST_MODEL[0] or model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
        "score": best_score,
        "target": target_score,
        "assets_total": len(assets),
        "assets_missing": missing,
        "assets_missing_initially": lost_initially,
        "assets_recovered": len(lost_initially) - len(missing),
        "repair_rounds": repairs,
        "saw_reference": bool(shot),
        "notes": list(_LOG),
        "sent_chars": len(prompt),
        "original_chars": len(html),
        "enhanced_chars": len(enhanced),
    }



# ============================================================================
# The Enhance button: recreate the page so it looks like the screenshot
# ============================================================================
#
# The target is the screenshot, judged by eye. The last version of this started the
# model from the engine's reconstruction and made the engine's page the floor, which
# is why its result looked like the engine: the engine's stand-in typeface, its
# artwork cut into fragments, the smudges where text was erased from backgrounds.
#
# Now the page is recreated from the screenshot itself, with three things settled
# before the model writes a line:
#
# - What is a picture. One call names the photographs, mockups, avatars and logos, and
#   each one is cropped straight out of the original screenshot and placed where it
#   sits -- identical by construction, text and all. A tilted laptop screen full of
#   UI cannot be rebuilt as live text in perspective; a crop of it is exact. The
#   boxes the model gives are rough, so each is snapped to the picture's own edges.
# - What typeface. The same call names the families, and they are fetched from Google
#   Fonts and embedded, so the page is drawn in the right face everywhere. The engine
#   set everything in Plus Jakarta Sans; a page set in Inter came back in the wrong
#   letterforms however well it was sized.
# - Where the text goes. The engine's measured positions are re-solved in the chosen
#   typeface and given to the model as CSS, so a line lands where the screenshot has it.
#   The words are for the model to read from the screenshot; the OCR only places them.

_FONT_DIR_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "fonts")
_LOCAL_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_LOCAL_FACES = {"inter": "Inter.ttf", "plus jakarta sans": "PlusJakartaSans.ttf"}
# A browser's user agent gets woff2, which is small; a plain one gets TTF, which PIL can
# measure. Google serves both from the same API.
_UA_BROWSER = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_UA_PLAIN = "Wget/1.21"
_WEIGHT_NAMES = {100: "Thin", 200: "ExtraLight", 300: "Light", 400: "Regular",
                 500: "Medium", 600: "SemiBold", 700: "Bold", 800: "ExtraBold",
                 900: "Black"}
_FAMILY_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 \-]{0,48}$")


def _slug(family: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", family.lower()).strip("-") or "font"


def _get(url: str, ua: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _axes(family: str) -> Dict[str, Tuple[float, float]]:
    """The variable axes Google Fonts has for a family: {"opsz": (14, 32), ...}.

    Read from Google's catalogue once and kept as a small table, since the catalogue is
    2.7MB. An empty dict when it cannot be had, and the family is fetched as before.
    """
    table_path = os.path.join(_FONT_DIR_CACHE, "axes.json")
    table: Optional[Dict[str, Any]] = None
    if os.path.exists(table_path):
        try:
            with open(table_path, encoding="utf-8") as fh:
                table = json.load(fh)
        except Exception:
            table = None
    if table is None:
        try:
            meta = json.loads(_get("https://fonts.google.com/metadata/fonts", _UA_BROWSER, 60))
            table = {f["family"].lower(): {a["tag"]: [a["min"], a["max"]] for a in f.get("axes") or []}
                     for f in meta.get("familyMetadataList") or []}
            os.makedirs(_FONT_DIR_CACHE, exist_ok=True)
            with open(table_path, "w", encoding="utf-8") as fh:
                json.dump(table, fh)
        except Exception:
            return {}
    return {k: (float(v[0]), float(v[1])) for k, v in (table.get(family.lower()) or {}).items()}


def _gf_css(family: str, axis: str, ua: str) -> str:
    fam = urllib.parse.quote(family).replace("%20", "+")
    url = "https://fonts.googleapis.com/css2?family=%s%s&display=block" % (fam, axis)
    return _get(url, ua).decode("utf-8", "replace")


def _as_weight(w: Any) -> int:
    try:
        return max(100, min(900, int(round(int(str(w)) / 100.0)) * 100))
    except (TypeError, ValueError):
        return 700 if str(w).lower() in ("bold", "bolder") else 400


def measure_face(family: str, weight: int):
    """A PIL font of this family at this weight, at 100px, for measuring text.

    Fetched from Google Fonts once and cached. When that is not possible -- offline, or
    a family Google does not have -- a shipped face is used if it is the same family,
    and otherwise None, and the caller falls back to the engine's own measurements.
    """
    from PIL import ImageFont
    weight = _as_weight(weight)
    os.makedirs(_FONT_DIR_CACHE, exist_ok=True)
    order = [weight] + sorted({400, 500, 600, 700, 300, 800} - {weight},
                              key=lambda x: abs(x - weight))
    for w in order:
        path = os.path.join(_FONT_DIR_CACHE, "%s-%d.ttf" % (_slug(family), w))
        if not os.path.exists(path):
            try:
                css = _gf_css(family, ":wght@%d" % w, _UA_PLAIN)
                url = re.search(r"url\((https://[^)]+)\)", css)
                if not url:
                    continue
                with open(path, "wb") as fh:
                    fh.write(_get(url.group(1), _UA_PLAIN))
            except Exception:
                continue
        try:
            return ImageFont.truetype(path, 100)
        except Exception:
            continue
    local = _LOCAL_FACES.get(family.strip().lower())
    if local and os.path.exists(os.path.join(_LOCAL_FONT_DIR, local)):
        font = ImageFont.truetype(os.path.join(_LOCAL_FONT_DIR, local), 100)
        try:
            font.set_variation_by_name(_WEIGHT_NAMES.get(weight, "Regular"))
        except Exception:
            pass
        return font
    return None


def embedded_family_css(families: List[str], weights: List[int]) -> str:
    """@font-face rules for these families with the font files inlined.

    Only the latin and latin-extended subsets, which is 30-60KB a family as woff2 --
    Inter's full TTF is 876KB. Cached per family, so a page is fetched for once.
    """
    os.makedirs(_FONT_DIR_CACHE, exist_ok=True)
    wanted = sorted({_as_weight(w) for w in weights} | {400, 700})
    rules: List[str] = []
    for family in families:
        # With its optical sizes, where it has them. Inter drawn at 76px from its text
        # cut came out 28px wider than the screenshot's headline and looser in every
        # letter; the page is set in its display cut, which the browser picks by itself
        # from the font size once the font carries the axis.
        axes = _axes(family)
        tries = [":wght@100..900", ":wght@" + ";".join(str(w) for w in wanted),
                 ":wght@400;700", ""]
        if "opsz" in axes and "wght" in axes:
            lo, hi = axes["opsz"]
            wlo, whi = axes["wght"]
            tries.insert(0, ":opsz,wght@%g..%g,%g..%g" % (lo, hi, wlo, whi))
        cache = os.path.join(_FONT_DIR_CACHE, _slug(family) + "-axes.css")
        css = None
        if os.path.exists(cache):
            with open(cache, encoding="utf-8") as fh:
                css = fh.read()
        if css is None:
            for axis in tries:
                try:
                    raw = _gf_css(family, axis, _UA_BROWSER)
                    break
                except Exception:
                    raw = None
            if raw:
                keep = []
                for name, block in re.findall(r"/\*\s*([\w-]+)\s*\*/\s*(@font-face\s*\{[^}]*\})", raw):
                    if name not in ("latin", "latin-ext"):
                        continue
                    for url in re.findall(r"url\((https://[^)]+)\)", block):
                        try:
                            data = base64.b64encode(_get(url, _UA_BROWSER)).decode("ascii")
                        except Exception:
                            block = ""
                            break
                        block = block.replace(url, "data:font/woff2;base64," + data)
                    if block:
                        keep.append(block)
                css = "\n".join(keep)
                if css:
                    with open(cache, "w", encoding="utf-8") as fh:
                        fh.write(css)
        if not css:
            # Offline: a shipped face if it is the same family, whole.
            local = _LOCAL_FACES.get(family.strip().lower())
            path = os.path.join(_LOCAL_FONT_DIR, local) if local else None
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    css = ("@font-face{font-family:'%s';src:url(data:font/ttf;base64,%s) "
                           "format('truetype');font-weight:100 900;font-display:block}"
                           % (family, base64.b64encode(fh.read()).decode("ascii")))
        if css:
            rules.append(css)
    return "\n".join(rules)


_FONT_LINK_RE = re.compile(r"<link[^>]+fonts\.(?:googleapis|gstatic)\.com[^>]*>", re.I)
_FONT_IMPORT_RE = re.compile(r"@import\s+url\([^)]*fonts\.googleapis\.com[^)]*\)\s*;?", re.I)


def apply_fonts(html: str, css: str) -> str:
    """The page with its typefaces embedded, and any link to fetch them removed."""
    html = _FONT_LINK_RE.sub("", html)
    html = _FONT_IMPORT_RE.sub("", html)
    if not css:
        return html
    tag = '<style id="ramen-fonts">%s</style>' % css
    at = html.lower().find("</head>")
    return (html[:at] + tag + html[at:]) if at >= 0 else (tag + html)


ANALYSIS_PROMPT = """\
The attached screenshot is a web page, %(width)d x %(height)d px.

1. FONTS. Name the typeface families the page is set in, as Google Fonts family names:
   the exact face if it is on Google Fonts, otherwise the closest one that is. Say which
   is used for headings and which for body text (they may be the same).

2. PICTURES. List every region that is a picture rather than something built from text
   and CSS boxes: photographs, illustrations, screenshots and mockups of products or
   devices (with all the text inside them), avatars and profile photos, logos and brand
   marks including wordmarks, and icons that are not simple shapes. Do not list plain
   text, buttons, cards, panels, backgrounds or gradients -- those are built in HTML.
   Give each a box that encloses the whole picture and nothing else.

Reply with JSON only, in this shape:
{"fonts": {"heading": "Family Name", "body": "Family Name"},
 "pictures": [{"label": "short name", "box_2d": [ymin, xmin, ymax, xmax]}]}
box_2d is normalised to 0-1000 on each axis.
"""


def analyse_screenshot(shot: Tuple[str, str], width: int, height: int,
                       api_key: Optional[str] = None, model: Optional[str] = None,
                       timeout: int = 150, attempts: int = 2) -> Dict[str, Any]:
    """The page's typefaces and its pictures, as the model reads them."""
    reply = call_gemini(ANALYSIS_PROMPT % {"width": width, "height": height},
                        api_key=api_key, model=model, timeout=timeout,
                        attempts=attempts, image=shot)
    text = _unfence(reply)
    m = re.search(r"\{.*\}", text, re.S)
    data = json.loads(m.group(0) if m else text)
    fonts = data.get("fonts") or {}
    clean = {}
    for role in ("heading", "body"):
        name = str(fonts.get(role) or "").strip().strip("'\"")
        if _FAMILY_OK.match(name):
            clean[role] = name
    pictures = [p for p in (data.get("pictures") or [])
                if isinstance(p, dict) and isinstance(p.get("box_2d"), list)
                and len(p["box_2d"]) == 4]
    return {"fonts": clean, "pictures": pictures}


# How far outside the model's rough box a picture's own edges may be looked for: a share
# of the box, and never more than a few pixels. A box can shrink onto its picture freely,
# but growth is what goes wrong -- on a page with a glow behind it, everything differs
# from the ground, and a laptop's box grew 108px into the nav and swallowed it.
SNAP_MARGIN = 0.12
SNAP_MAX_GROWTH = 14


def snap_to_picture(img, x0: int, y0: int, x1: int, y1: int) -> Tuple[int, int, int, int]:
    """Moves a rough box onto the edges of the picture it was drawn round.

    A picture is whatever stands out from the ground around it, taken as the marks that
    reach into the middle of the box -- so a neighbouring heading, which the box only
    grazes, is not pulled in, and a logo the box cut short is completed.
    """
    import cv2
    import numpy as np
    h, w = img.shape[:2]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    mx = max(4, min(SNAP_MAX_GROWTH, int(bw * SNAP_MARGIN)))
    my = max(4, min(SNAP_MAX_GROWTH, int(bh * SNAP_MARGIN)))
    ex0, ey0, ex1, ey1 = max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)
    win = img[ey0:ey1, ex0:ex1]
    if win.size == 0 or win.shape[0] < 4 or win.shape[1] < 4:
        return x0, y0, x1, y1
    border = np.concatenate([win[0, :], win[-1, :], win[:, 0], win[:, -1]], axis=0)
    dist = np.linalg.norm(win.astype(np.float32)
                          - np.median(border, axis=0).astype(np.float32), axis=2)
    content = (dist > 22.0).astype(np.uint8)
    content = cv2.dilate(content, np.ones((5, 5), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(content, 8)
    if n <= 1:
        return x0, y0, x1, y1
    cx0, cy0 = (x0 - ex0) + bw // 5, (y0 - ey0) + bh // 5
    cx1, cy1 = (x1 - ex0) - bw // 5, (y1 - ey0) - bh // 5
    core = lab[max(0, cy0):max(cy0 + 1, cy1), max(0, cx0):max(cx0 + 1, cx1)]
    touched = set(np.unique(core).tolist()) - {0}
    if not touched:
        return x0, y0, x1, y1
    sx0 = min(int(stats[i, cv2.CC_STAT_LEFT]) for i in touched)
    sy0 = min(int(stats[i, cv2.CC_STAT_TOP]) for i in touched)
    sx1 = max(int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]) for i in touched)
    sy1 = max(int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]) for i in touched)
    # The dilation added two pixels a side; take them back.
    return (ex0 + max(0, sx0 + 2), ey0 + max(0, sy0 + 2),
            ex0 + min(win.shape[1], sx1 - 2), ey0 + min(win.shape[0], sy1 - 2))


def picture_crops(shot: Tuple[str, str], pictures: List[Dict[str, Any]],
                  text_boxes: Optional[List[Tuple[float, float, float, float]]] = None):
    """Crops of the original for each picture: (data uris, prompt lines, pixel boxes).

    A snapped box that takes in a line of text its rough box did not hold goes back to
    the rough box: that line is the page's own text, and cropping it into a picture
    would make it uneditable and hide it from everything that checks the text.
    """
    import numpy as np
    from PIL import Image
    im = Image.open(io.BytesIO(base64.b64decode(shot[1]))).convert("RGB")
    w, h = im.size
    arr = np.asarray(im)
    boxes = []
    for p in pictures:
        try:
            ymin, xmin, ymax, xmax = [float(v) for v in p["box_2d"]]
        except (TypeError, ValueError):
            continue
        rx0, rx1 = sorted((int(xmin * w / 1000.0), int(xmax * w / 1000.0)))
        ry0, ry1 = sorted((int(ymin * h / 1000.0), int(ymax * h / 1000.0)))
        x0, y0, x1, y1 = snap_to_picture(arr, rx0, ry0, rx1, ry1)
        if text_boxes:
            rough = [(rx0, ry0, rx1, ry1)]
            snapped = [(x0, y0, x1, y1)]
            if any(_inside(t, snapped) and not _inside(t, rough) for t in text_boxes):
                x0, y0, x1, y1 = max(0, rx0), max(0, ry0), min(w, rx1), min(h, ry1)
        if text_boxes:
            x0, y0, x1, y1 = clear_of_text((x0, y0, x1, y1), text_boxes)
        if min(x1 - x0, y1 - y0) < 8:
            continue
        if (x1 - x0) >= w * 0.95 and (y1 - y0) >= h * 0.95:
            continue                      # the whole page is not a picture on it
        boxes.append((x0, y0, x1, y1, str(p.get("label") or "picture")[:40]))

    # A picture wholly inside a bigger one is already in its crop.
    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    kept: List[Tuple[int, int, int, int, str]] = []
    for b in boxes:
        area = (b[2] - b[0]) * (b[3] - b[1])
        inside = False
        for k in kept:
            ix = max(0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0, min(b[3], k[3]) - max(b[1], k[1]))
            if ix * iy >= area * 0.9:
                inside = True
                break
        if not inside:
            kept.append(b)
    kept.sort(key=lambda b: (b[1], b[0]))

    assets, lines, pix = [], [], []
    for i, (x0, y0, x1, y1, label) in enumerate(kept):
        buf = io.BytesIO()
        im.crop((x0, y0, x1, y1)).save(buf, format="PNG", optimize=True)
        assets.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"))
        lines.append("- %s: left:%dpx; top:%dpx; width:%dpx; height:%dpx"
                     % (label, x0, y0, x1 - x0, y1 - y0))
        pix.append((x0, y0, x1, y1))
    return assets, lines, pix


def clear_of_text(box: Tuple[int, int, int, int],
                  text_boxes: List[Tuple[float, float, float, float]], pad: int = 3
                  ) -> Tuple[int, int, int, int]:
    """The box, cut back so it does not reach into any line of the page's own text.

    A picture sits above the page, so where its box reaches over a line of text that
    line is drawn twice -- the crop's copy over the page's -- a few pixels apart. The
    laptop's box on 8xbrand started at x=438, inside the headline and the paragraph,
    and the page read "pcsts" and "prodluct". A line whose middle is in the box is the
    picture's own and stays; one that only overlaps it is cut out, on whichever side
    loses the least of the picture.
    """
    x0, y0, x1, y1 = box
    for _ in range(4):
        changed = False
        for t in text_boxes:
            tx0, ty0, tx1, ty1 = t
            if tx1 <= x0 or tx0 >= x1 or ty1 <= y0 or ty0 >= y1:
                continue
            if _inside(t, [(x0, y0, x1, y1)]):
                continue
            cuts = [(int(min(x1, tx1 + pad)), y0, x1, y1), (x0, y0, int(max(x0, tx0 - pad)), y1),
                    (x0, int(min(y1, ty1 + pad)), x1, y1), (x0, y0, x1, int(max(y0, ty0 - pad)))]
            x0, y0, x1, y1 = max(cuts, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
            changed = True
        if not changed:
            break
    return int(x0), int(y0), int(x1), int(y1)


def _crop_uri(im, box) -> str:
    buf = io.BytesIO()
    im.crop(tuple(int(v) for v in box)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# What makes a region of the screenshot a picture nobody placed: detail in the
# screenshot, none in the page drawn from it.
UNPLACED_EDGE = 40
UNPLACED_SCREEN_DETAIL = 0.10
UNPLACED_PAGE_DETAIL = 0.03
UNPLACED_MIN_SIDE = 14
UNPLACED_LIMIT = 12


def unplaced_pictures(shot: Tuple[str, str], render: Tuple[str, str],
                      text_boxes: List[Tuple[float, float, float, float]],
                      pictures: List[Tuple[int, int, int, int]]
                      ) -> List[Tuple[int, int, int, int]]:
    """Regions with something drawn in the screenshot and nothing in the page.

    The call that finds the pictures misses some, and not the same ones twice: one run
    of 8xbrand found the row of avatars and the next did not, and the page had a blank
    where they belong. The page drawn from the screenshot shows where: there is detail
    in the screenshot and flat background in the page. Regions that are mostly a line
    of text are left alone -- that is text the page put somewhere else, not a picture.
    """
    import cv2
    import numpy as np
    a = cv2.imdecode(np.frombuffer(base64.b64decode(shot[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    b = cv2.imdecode(np.frombuffer(base64.b64decode(render[1]), np.uint8), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None:
        return []
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
    k = np.ones((3, 3), np.uint8)
    detail_a = cv2.blur((cv2.morphologyEx(a, cv2.MORPH_GRADIENT, k) > UNPLACED_EDGE).astype(np.float32), (15, 15))
    detail_b = cv2.blur((cv2.morphologyEx(b, cv2.MORPH_GRADIENT, k) > UNPLACED_EDGE).astype(np.float32), (15, 15))
    apart = cv2.blur(np.abs(a.astype(np.float32) - b.astype(np.float32)), (9, 9))
    mask = ((detail_a > UNPLACED_SCREEN_DETAIL) & (detail_b < UNPLACED_PAGE_DETAIL)
            & (apart > 12)).astype(np.uint8) * 255
    for x0, y0, x1, y1 in pictures:
        mask[max(0, y0):y1, max(0, x0):x1] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    found = []
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if bw < UNPLACED_MIN_SIDE or bh < UNPLACED_MIN_SIDE or area < 250:
            continue
        if bw * bh > w * h * 0.25:
            continue
        box = (max(0, x - 3), max(0, y - 3), min(w, x + bw + 3), min(h, y + bh + 3))
        covered = 0.0
        for tx0, ty0, tx1, ty1 in text_boxes:
            covered += (max(0.0, min(box[2], tx1) - max(box[0], tx0))
                        * max(0.0, min(box[3], ty1) - max(box[1], ty0)))
        if covered > 0.3 * (box[2] - box[0]) * (box[3] - box[1]):
            continue
        box = clear_of_text(box, text_boxes)
        if min(box[2] - box[0], box[3] - box[1]) < UNPLACED_MIN_SIDE:
            continue
        found.append(box)
    found.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    return found[:UNPLACED_LIMIT]


def _inside(box, pictures) -> bool:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    return any(p[0] <= cx <= p[2] and p[1] <= cy <= p[3] for p in pictures)


# The page's background, read off the screenshot at an eighth of its size. Fine enough
# to carry a glow or a gradient exactly, coarse enough that nothing sharp survives.
PLATE_SCALE = 8
# Surfaces smaller than this share of the page -- buttons, badges, inputs -- are cleared
# from the plate so the HTML's own versions of them do not sit on a blurred copy.
PLATE_SURFACE_SHARE = 0.15
# How far a cell has to differ from the wide median around it to be cleared as an object.
PLATE_OBJECT_CONTRAST = 18


def background_plate(shot: Tuple[str, str], clear: List[Tuple[float, float, float, float]]
                     ) -> str:
    """The screenshot's background alone -- its colour, glows and gradients -- as an image.

    Every picture is cut from the original with the background it sits on, so on a
    page whose background is anything but flat, each crop showed as a rectangle of a
    slightly different colour: the checks kept reporting "a white rectangle" and "a
    purple square" round things that were in exactly the right place, and asked for
    them to be moved. A background written in CSS cannot match a photographed glow.
    One measured off the same screenshot does, so the crops meet it with no seam.

    Text, pictures and small surfaces are cleared first and the gaps filled from what
    surrounds them, so the plate holds only the ground they sit on.
    """
    import cv2
    import numpy as np
    from PIL import Image
    im = np.asarray(Image.open(io.BytesIO(base64.b64decode(shot[1]))).convert("RGB"))
    h, w = im.shape[:2]
    sw, sh = max(8, w // PLATE_SCALE), max(8, h // PLATE_SCALE)
    small = cv2.resize(im, (sw, sh), interpolation=cv2.INTER_AREA)
    mask = np.zeros((sh, sw), np.uint8)
    for x0, y0, x1, y1 in clear:
        a0 = max(0, int((x0 - 4) * sw / w) - 1)
        b0 = max(0, int((y0 - 4) * sh / h) - 1)
        a1 = min(sw, int(np.ceil((x1 + 4) * sw / w)) + 1)
        b1 = min(sh, int(np.ceil((y1 + 4) * sh / h)) + 1)
        mask[b0:b1, a0:a1] = 255
    # Anything else that stands out from a wide median of its surroundings is an object,
    # not ground. The engine does not make a surface for every button -- the dark
    # "Start a campaign" pills had none -- and a plate that kept them put a blurred dark
    # smear under the HTML's own buttons.
    wide = cv2.medianBlur(small, 15)
    stands_out = (np.abs(small.astype(np.int16) - wide.astype(np.int16)).max(axis=2)
                  > PLATE_OBJECT_CONTRAST).astype(np.uint8) * 255
    mask = cv2.bitwise_or(mask, cv2.dilate(stands_out, np.ones((3, 3), np.uint8)))
    if mask.mean() < 250:
        small = cv2.inpaint(small, mask, 4, cv2.INPAINT_TELEA)
    small = cv2.GaussianBlur(small, (0, 0), 1.2)
    buf = io.BytesIO()
    Image.fromarray(small).save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_MARKER_IMG_RE = re.compile(r"<img\b[^>]*RAMEN_ASSET_\d+[^>]*>", re.I)


def picture_layer(assets: List[str], boxes: List[Tuple[int, int, int, int]],
                  plate: str, width: int, height: int) -> str:
    """The background plate and every picture, placed exactly, as markup for the page.

    Placed by this code, not by the model. Left to the model, the pictures moved: the
    check misjudged where they were -- it reported logos cut off below the bottom of an
    836px page -- and the correction obeyed, shifting a crop that was already exact and
    taking the page from 91.6% to 64.6%. What was measured is not the model's to move.
    """
    imgs = "".join(
        '<img src="%s" alt="" style="position:absolute;left:%dpx;top:%dpx;'
        'width:%dpx;height:%dpx;display:block;max-width:none">'
        % (uri, x0, y0, x1 - x0, y1 - y0)
        for uri, (x0, y0, x1, y1) in zip(assets, boxes))
    return ('<style id="ramen-plate">html{background:url(%s) 0 0/%dpx %dpx no-repeat !important}'
            'body{background:transparent !important}</style>'
            '<div id="ramen-pictures" aria-hidden="true" style="position:absolute;left:0;top:0;'
            'width:%dpx;height:%dpx;z-index:40;pointer-events:none">%s</div>'
            % (plate, width, height, width, height, imgs))


def lock_pictures(page_html: str, layer: str) -> str:
    """The model's page with the pictures and background put where they were measured.

    Any image the model wrote for a picture is removed, and the measured layer is
    added in its place, above the page and at the page's origin.
    """
    page_html = _MARKER_IMG_RE.sub("", page_html)
    style_at = layer.index("</style>") + len("</style>")
    style, pictures = layer[:style_at], layer[style_at:]
    head = page_html.lower().find("</head>")
    if head >= 0:
        page_html = page_html[:head] + style + page_html[head:]
    else:
        page_html = style + page_html
    body = page_html.lower().rfind("</body>")
    if body >= 0:
        return page_html[:body] + pictures + page_html[body:]
    return page_html + pictures


# A line of display text is set in the heading face from this size up.
HEADING_MIN_SIZE = 24.0


# Weights a line is tried at, to find the one whose ink matches the screenshot's.
WEIGHT_CANDIDATES = (300, 400, 500, 600, 700, 800, 900)
# How closely the best weight's letters must match the screenshot's to be believed.
WEIGHT_MIN_MATCH = 0.5
# How much better than regular another weight must match, on text below display size.
WEIGHT_SMALL_MARGIN = 0.03
# How far the size a small line's width gives may be from the size its height gives, to
# be tried as well.
SIZE_BY_WIDTH_SPREAD = 0.15
# Sizes tried between the two, the last of them the untracked one.
SIZE_STEPS = 3
# Lines closer in colour than this, in RGB, are taken for one style.
STYLE_COLOUR_DISTANCE = 40
# Rows drawn in one screenshot, so a long list stays inside what the browser will capture.
CHROME_ROWS_PER_SHOT = 120
# The least contrast between a line's ink and its ground for its ink to be weighed.
WEIGHT_MIN_CONTRAST = 40.0
# A gap in a display line's ink this wide, as a share of its size, is a space: 8xbrand's
# headline, tracked tight, has word gaps of 0.13-0.16em and letter gaps under 0.06em.
WORD_GAP_SHARE = 0.1


def _chrome_ink(rows: List[Tuple], font_css: str) -> Optional[List[Optional[Tuple]]]:
    """Ink of each (text, family, weight, size[, letter-spacing]) as the browser draws it.

    Returns, for each row, its ink box relative to where its line box starts (with
    line-height 1) and the ink itself, 0-1 a pixel. The browser, not a font library,
    because the browser is what draws the page: it picks the optical size by itself,
    kerns, and falls back the same way when a font is missing.
    """
    if len(rows) > CHROME_ROWS_PER_SHOT:
        out: List[Optional[Tuple]] = []
        for at in range(0, len(rows), CHROME_ROWS_PER_SHOT):
            part = _chrome_ink(rows[at:at + CHROME_ROWS_PER_SHOT], font_css)
            if part is None:
                return None
            out.extend(part)
        return out
    import numpy as np
    from PIL import Image
    pad = 40
    tops, y = [], 0
    rows = [tuple(r) + (0.0,) * (5 - len(r)) for r in rows]
    for text, family, weight, size, _ in rows:
        tops.append(y + int(size * 0.5) + 10)
        y += int(size * 2.2) + 20
    width = int(min(6000, max(400, max(r[3] * 0.8 * (len(r[0]) + 2) for r in rows) + 2 * pad)))
    divs = "".join(
        '<div style="position:absolute;left:%dpx;top:%dpx;font:%d %.2fpx/1 \'%s\';'
        'letter-spacing:%.2fpx;white-space:pre;color:#000">%s</div>'
        % (pad, top, weight, size, family, track, html.escape(text))
        for (text, family, weight, size, track), top in zip(rows, tops))
    page = ("<!DOCTYPE html><html><head><style>%s\nhtml,body{margin:0;background:#fff}"
            "</style></head><body>%s</body></html>" % (font_css, divs))
    shot = render_html(page, width=width, height=y + 20)
    if not shot:
        return None
    g = 255.0 - np.asarray(Image.open(io.BytesIO(base64.b64decode(shot[1]))).convert("L"),
                           dtype=np.float32)
    out = []
    for (text, family, weight, size, _), top in zip(rows, tops):
        b0 = max(0, top - int(size * 0.5) - 8)
        band = g[b0:top + int(size * 1.7) + 8]
        on = band > 60
        if not on.any():
            out.append(None)
            continue
        ys = np.where(on.any(axis=1))[0]
        xs = np.where(on.any(axis=0))[0]
        out.append((float(xs[0] - pad), float(ys[0] + b0 - top),
                    float(xs[-1] + 1 - pad), float(ys[-1] + 1 + b0 - top),
                    band[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1] / 255.0))
    return out


def _darkness(arr, box, inks: List[Tuple[float, Tuple[int, int, int]]], pad: int = 1):
    """How much of each pixel in the box is ink rather than ground, 0-1, or None.

    Ground is read round the box; the ink is the line's measured colour, per column
    where the line changes colour part way.
    """
    import numpy as np
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    ring = np.concatenate([
        arr[max(0, y0 - 4):y0, x0:x1].reshape(-1, 3), arr[y1:y1 + 4, x0:x1].reshape(-1, 3),
        arr[y0:y1, max(0, x0 - 4):x0].reshape(-1, 3), arr[y0:y1, x1:x1 + 4].reshape(-1, 3)])
    if not len(ring):
        return None
    ground = np.median(ring.astype(np.float32), axis=0)
    crop = arr[y0:y1, x0:x1].astype(np.float32)
    cols = np.zeros((x1 - x0, 3), np.float32)
    for i in range(x1 - x0):
        at = i / float(max(1, x1 - x0 - 1))
        ink = inks[0][1]
        for frac, colour in inks:
            if at >= frac:
                ink = colour
        cols[i] = ink
    d = cols - ground
    reach = (d * d).sum(axis=1)
    if float(np.sqrt(reach.min())) < WEIGHT_MIN_CONTRAST:
        return None
    t = ((crop - ground) * d[None, :, :]).sum(axis=2) / reach[None, :]
    return np.clip(t, 0.0, 1.0)


def _match(screen, drawn) -> Tuple[float, int, int]:
    """How alike two inks are in shape, -1 to 1, whatever their colour or contrast.

    The ink's amount cannot choose a weight: the screenshot's text is softened by its
    compression and its colour is measured a little light, and together they made
    every line read two or three weights heavier than it is. The shape of the letters
    does not depend on either -- normalised correlation, with the drawn line softened
    as much as the screenshot is. The drawn line is laid on the screenshot's box as it
    is, not stretched to it, so a line drawn at the wrong size matches badly, and the
    best of a pixel's shift either way is taken. Returns (score, dy, dx).
    """
    import cv2
    import numpy as np
    if screen is None or drawn is None or drawn.size == 0:
        return -1.0, 0, 0
    h, w = screen.shape[:2]
    d = cv2.GaussianBlur(drawn.astype(np.float32), (0, 0), 1.0)
    a = screen.astype(np.float32) - float(screen.mean())
    aa = float((a * a).sum())
    best = (-1.0, 0, 0)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            canvas = np.zeros((h, w), np.float32)
            y0, x0 = max(0, dy), max(0, dx)
            sy0, sx0 = max(0, -dy), max(0, -dx)
            ch = min(h - y0, d.shape[0] - sy0)
            cw = min(w - x0, d.shape[1] - sx0)
            if ch <= 0 or cw <= 0:
                continue
            canvas[y0:y0 + ch, x0:x0 + cw] = d[sy0:sy0 + ch, sx0:sx0 + cw]
            b = canvas - float(canvas.mean())
            den = float(np.sqrt(aa * float((b * b).sum())))
            if den > 0:
                score = float((a * b).sum() / den)
                if score > best[0]:
                    best = (score, dy, dx)
    return best


# How much of a pixel the drawn letters must cover for the screenshot's pixel there to
# count as all ink, and how many such pixels a colour is read from.
FULL_COVER = 0.95
FULL_COVER_MIN_PIXELS = 6


def _fit_colour(arr, box, drawn, dy: int, dx: int,
                spans: List[Tuple[float, float]]) -> List[Optional[str]]:
    """The colour each part of a line is set in, fitted to the screenshot's own pixels.

    Where the letters drawn in the browser cover a pixel completely, the screenshot's
    pixel there is all ink, and their median is its colour. Sampling the darkest pixels
    of the screenshot alone read small grey text lighter than it is -- few pixels of a
    16px stroke are all ink, and which ones is a guess without the letters to go by --
    and fitting ground-plus-coverage read it black, because the screenshot was drawn
    heavier than the browser draws the same weight (its text carries 1.1-1.7 times the
    ink), so every pixel looked like more ink than the coverage allowed.
    """
    import cv2
    import numpy as np
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    bh, bw = y1 - y0, x1 - x0
    if bh < 2 or bw < 2 or drawn is None:
        return [None] * len(spans)
    ring = np.concatenate([
        arr[max(0, y0 - 4):y0, x0:x1].reshape(-1, 3), arr[y1:y1 + 4, x0:x1].reshape(-1, 3),
        arr[y0:y1, max(0, x0 - 4):x0].reshape(-1, 3), arr[y0:y1, x1:x1 + 4].reshape(-1, 3)])
    if not len(ring):
        return [None] * len(spans)
    ground = np.median(ring.astype(np.float32), axis=0)
    d = cv2.GaussianBlur(drawn.astype(np.float32), (0, 0), 1.0)
    cover = np.zeros((bh, bw), np.float32)
    ty, tx = max(0, dy), max(0, dx)
    sy, sx = max(0, -dy), max(0, -dx)
    ch, cw = min(bh - ty, d.shape[0] - sy), min(bw - tx, d.shape[1] - sx)
    if ch <= 0 or cw <= 0:
        return [None] * len(spans)
    cover[ty:ty + ch, tx:tx + cw] = d[sy:sy + ch, sx:sx + cw]
    pix = arr[y0:y1, x0:x1].astype(np.float32)
    full = np.zeros((bh, bw), np.float32)
    raw = drawn.astype(np.float32)
    full[ty:ty + ch, tx:tx + cw] = raw[sy:sy + ch, sx:sx + cw]
    out: List[Optional[str]] = []
    for a, b in spans:
        c0, c1 = int(a * bw), max(int(a * bw) + 1, int(b * bw))
        core = full[:, c0:c1] >= FULL_COVER
        if int(core.sum()) < FULL_COVER_MIN_PIXELS:
            out.append(None)
            continue
        # Of those, the ones furthest from the ground: a pixel a stroke's edge half
        # covers in the screenshot can be fully covered in the browser's drawing.
        inside = pix[:, c0:c1][core]
        far = np.linalg.norm(inside - ground, axis=1)
        ink = np.median(inside[far >= np.percentile(far, 60)], axis=0)
        out.append("#%02x%02x%02x" % tuple(int(round(v)) for v in np.clip(ink, 0, 255)))
    return out


def _hex_rgb(c: str) -> Tuple[int, int, int]:
    c = (c or "#000000").lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    try:
        return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    except ValueError:
        return 0, 0, 0


def _respace(text: str, dark, size: float, face) -> str:
    """The line with the spaces its ink shows put back where the OCR lost them.

    PaddleOCR reads a large headline as "yourbuyersread.", and every measurement taken
    of that string is off by two spaces' width -- the tracking came out loose enough to
    read wrong, and the model, correcting the words, had no way to correct the width.
    The ink has the gaps: a run of empty columns a space wide is a space.
    """
    import numpy as np
    if dark is None or len(text) < 4 or size < HEADING_MIN_SIZE or face is None:
        return text
    empty = dark.max(axis=0) < 0.2
    gaps, run = [], 0
    for i, e in enumerate(empty):
        if e:
            run += 1
        else:
            if run and i - run > 0 and run >= WORD_GAP_SHARE * size:
                gaps.append((i - run / 2.0) / float(len(empty)))
            run = 0
    if not gaps:
        return text
    try:
        total = float(face.getlength(text)) or 1.0
        edges = [float(face.getlength(text[:i])) / total for i in range(len(text) + 1)]
    except Exception:
        edges = [i / float(len(text)) for i in range(len(text) + 1)]
    inserts = []
    for f in gaps:
        i = min(range(1, len(text)), key=lambda j: abs(edges[j] - f))
        near = text[max(0, i - 2):i + 2]
        if " " not in near:
            inserts.append(i)
    for i in sorted(set(inserts), reverse=True):
        text = text[:i] + " " + text[i:]
    return text


def _colour_runs(text: str, stops, face) -> List[Tuple[str, str]]:
    """The line cut into the parts it is set in, each with its colour.

    A line that changes colour was described as "colours #05060e from 0% of its width,
    then #6871c5 from 29%", and the model painted it with a gradient clipped to the
    text. That clips to the line's box, and with the tight line-height a headline is
    set at, the tails of every y and p were cut off. Named parts are written as spans.
    """
    try:
        total = float(face.getlength(text)) if face is not None else 0.0
    except Exception:
        total = 0.0
    edges = ([float(face.getlength(text[:i])) / total for i in range(len(text) + 1)]
             if total else [i / float(max(1, len(text))) for i in range(len(text) + 1)])
    cuts = []
    for frac, _ in list(stops)[1:]:
        i = min(range(1, len(text)), key=lambda j: abs(edges[j] - float(frac)))
        for d in (0, 1, -1, 2, -2):          # a colour changes between words
            j = i + d
            if 0 < j < len(text) and (text[j - 1] == " " or text[j] == " "):
                i = j if text[j - 1] == " " else j + 1
                break
        cuts.append(max(1, min(len(text) - 1, i)))
    parts, at = [], 0
    for cut, (_, colour) in zip(cuts + [len(text)], stops):
        if cut > at:
            parts.append((text[at:cut], colour))
            at = cut
    return parts


def _pil_css(text: str, box, family: str, weight: int, size_hint: float, st) -> str:
    """The line's CSS from a font library's measurements, when the browser is not there."""
    x0, y0, x1, y1 = box
    face = measure_face(family, weight)
    if face is not None:
        try:
            ink = face.getbbox(text)
            ascent, descent = face.getmetrics()
            ink_w, ink_h = float(ink[2] - ink[0]), float(ink[3] - ink[1])
            if ink_h > 0 and ink_w > 0:
                scale = (y1 - y0) / ink_h
                size = 100.0 * scale
                track = ((x1 - x0) - ink_w * scale) / max(len(text) - 1, 1)
                track = max(-0.12 * size, min(0.25 * size, track))
                return ("left:%.1fpx; top:%.1fpx; font:%d %.2fpx/1 '%s'; letter-spacing:%.2fpx"
                        % (x0 - ink[0] * scale, y0 - ink[1] * scale, weight, size, family, track))
        except Exception:
            pass
    return ("left:%.1fpx; top:%.1fpx; font:%d %.2fpx/1 '%s'; letter-spacing:%.2fpx"
            % (x0, y0, weight, size_hint, family, float(st.letterSpacing or 0.0)))


def text_hints(page, pictures, fonts: Dict[str, str], shot: Optional[Tuple[str, str]] = None,
               font_css: str = "") -> Tuple[List[str], str, List[int]]:
    """CSS for every line of text not inside a picture, solved in the page's own fonts.

    The engine measured where each line's ink is. Size, weight and tracking are solved
    again here by drawing each line in the browser, in the fonts the page is given, and
    matching its ink to the screenshot's: the size from the ink's height, the weight
    from how much ink there is, the tracking from its width. The engine's own numbers
    were solved in its stand-in face and read the headline as 800 when it is 700.

    Returns the prompt lines, a reference page of the same text for the content check,
    and the weights used.
    """
    import numpy as np
    from PIL import Image
    heading = fonts.get("heading") or fonts.get("body") or "Inter"
    body = fonts.get("body") or heading
    arr = (np.asarray(Image.open(io.BytesIO(base64.b64decode(shot[1]))).convert("RGB"))
           if shot else None)
    runs = [e for e in (page.elements or [])
            if e.type == "text" and (e.text or "").strip() and e.style is not None]
    runs.sort(key=lambda e: (round(float(e.bbox[1]) / 8.0), float(e.bbox[0])))
    items: List[Dict[str, Any]] = []
    for e in runs:
        box = tuple(float(v) for v in e.bbox)
        if _inside(box, pictures):
            continue
        st = e.style
        size_hint = float(st.fontSize or 14.0)
        family = heading if size_hint >= HEADING_MIN_SIZE else body
        stops = ([(float(at), c) for at, c in st.colorStops]
                 if st.colorStops and len(st.colorStops) >= 2 else [(0.0, st.color or "#000000")])
        inks = [(f, _hex_rgb(c)) for f, c in stops]
        dark = _darkness(arr, box, inks, pad=0) if arr is not None else None
        weight = _as_weight(st.fontWeight)
        text = _respace(" ".join((e.text or "").split()), dark, size_hint,
                        measure_face(family, weight))
        items.append({"text": text, "box": box, "family": family, "weight": weight,
                      "size": size_hint, "stops": stops, "dark": dark, "st": st, "css": None})

    # ---- in the browser: each line drawn at every weight, at the size its ink's height
    # ---- gives with the tracking its width then needs -- and, for small text, also at
    # ---- the size its width gives untracked -- and the one shaped most like it kept
    measured = False
    if items and font_css and arr is not None:
        first = _chrome_ink([(i["text"], i["family"], i["weight"], i["size"]) for i in items], font_css)
        if first:
            for i, ink in zip(items, first):
                if ink and ink[3] > ink[1]:
                    i["size"] *= (i["box"][3] - i["box"][1]) / (ink[3] - ink[1])
            n = len(WEIGHT_CANDIDATES)
            plain = _chrome_ink([(i["text"], i["family"], w, i["size"])
                                 for i in items for w in WEIGHT_CANDIDATES], font_css)
            if plain:
                for k, i in enumerate(items):
                    bw = i["box"][2] - i["box"][0]
                    gaps = max(len(i["text"]) - 1, 1)
                    i["cands"] = []
                    for w, ink in zip(WEIGHT_CANDIDATES, plain[k * n:(k + 1) * n]):
                        if not ink or ink[2] <= ink[0]:
                            continue
                        track = (bw - (ink[2] - ink[0])) / gaps
                        track = max(-0.12 * i["size"], min(0.25 * i["size"], track))
                        i["cands"].append((w, i["size"], track))
                        # A 16px line's ink is 16px tall, and a pixel of that is 6% of
                        # its size: small text is more often right at the size its width
                        # gives, untracked, than tracked to fit a size a pixel out.
                        by_width = i["size"] * bw / (ink[2] - ink[0])
                        if i["size"] < HEADING_MIN_SIZE and abs(by_width / i["size"] - 1.0) <= SIZE_BY_WIDTH_SPREAD:
                            for step in range(1, SIZE_STEPS + 1):
                                size = i["size"] + (by_width - i["size"]) * step / SIZE_STEPS
                                natural = (ink[2] - ink[0]) * size / i["size"]
                                i["cands"].append((w, size, (bw - natural) / gaps))
                drawn = _chrome_ink([(i["text"], i["family"], w, size, track)
                                     for i in items for (w, size, track) in i["cands"]], font_css)
                if drawn:
                    measured = True
                    at = 0
                    for i in items:
                        tried = drawn[at:at + len(i["cands"])]
                        at += len(i["cands"])
                        scored = []
                        for cand, ink in zip(i["cands"], tried):
                            if ink:
                                m = _match(i["dark"], ink[4]) if i["dark"] is not None else (0.0, 0, 0)
                                scored.append((m[0], cand, ink, m))
                        if not scored:
                            continue
                        best = max(scored, key=lambda t: t[0])
                        # Text of a dozen pixels is too coarse to tell neighbouring
                        # weights apart reliably: "For creators" matched 300 at 0.877
                        # and 400 at 0.862. Regular stands unless another clearly wins.
                        if i["size"] < HEADING_MIN_SIZE:
                            regular = [t for t in scored if t[1][0] == 400]
                            if regular:
                                r = max(regular, key=lambda t: t[0])
                                if best[0] - r[0] < WEIGHT_SMALL_MARGIN:
                                    best = r
                        if i["dark"] is not None and best[0] < WEIGHT_MIN_MATCH:
                            best = next((t for t in scored if t[1][0] == i["weight"]), best)
                        i["weight"], i["size"], i["track"] = best[1]
                        i["ink"], i["match"] = best[2], best[0]
                        i["fit"] = best[3]
                        i["by_weight"] = {}
                        for t in sorted(scored, key=lambda t: t[0]):
                            i["by_weight"][t[1][0]] = t

    # Lines solved one at a time come out a few percent apart even when they are one
    # headline, because each line's ink height depends on which letters it happens to
    # contain. Near sizes are one size, and display lines of one size one weight.
    order = sorted(range(len(items)), key=lambda k: items[k]["size"])
    groups: List[List[int]] = []
    for k in order:
        if groups and items[k]["size"] <= items[groups[-1][0]]["size"] * 1.07:
            groups[-1].append(k)
        else:
            groups.append([k])
    for g in groups:
        # Lines of one colour in a group are one style: they take the weight their
        # matches favour, each at its own best size for that weight.
        styles: List[List[int]] = []
        for k in g:
            ink = _hex_rgb(items[k]["stops"][0][1])
            for st_ in styles:
                ref_ink = _hex_rgb(items[st_[0]]["stops"][0][1])
                if sum((p - q) ** 2 for p, q in zip(ink, ref_ink)) ** 0.5 < STYLE_COLOUR_DISTANCE:
                    st_.append(k)
                    break
            else:
                styles.append([k])
        for st_ in styles:
            if len(st_) < 2:
                continue
            votes: Counter = Counter()
            for k in st_:
                votes[items[k]["weight"]] += items[k].get("match", 0.0)
            top = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
            for k in st_:
                t = (items[k].get("by_weight") or {}).get(top)
                if items[k]["weight"] != top and t:
                    items[k]["weight"], items[k]["size"], items[k]["track"] = t[1]
                    items[k]["ink"], items[k]["match"], items[k]["fit"] = t[2], t[0], t[3]
        # One size for a style: the one its best-matched line was found at. The median
        # of the whole group took the paragraph to the size of a button's label.
        for st_ in styles:
            centre = items[max(st_, key=lambda k: items[k].get("match", 0.0))]["size"]
            for k in st_:
                items[k]["scale"] = centre / items[k]["size"]
                items[k]["size"] = centre

    lines, ref, weights = [], [], []
    for i in items:
        x0, y0, x1, y1 = i["box"]
        ink = i.get("ink") if measured else None
        if ink:
            k = i.get("scale", 1.0)
            ix0, iy0, ix1 = ink[0] * k, ink[1] * k, ink[2] * k
            size = i["size"]
            track = ((x1 - x0) - (ix1 - ix0)) / max(len(i["text"]) - 1, 1) + i["track"] * k
            track = max(-0.12 * size, min(0.25 * size, track))
            css = ("left:%.1fpx; top:%.1fpx; font:%d %.2fpx/1 '%s'; letter-spacing:%.2fpx"
                   % (x0 - ix0, y0 - iy0, i["weight"], size, i["family"], track))
        else:
            css = _pil_css(i["text"], i["box"], i["family"], i["weight"], i["size"], i["st"])
        weights.append(i["weight"])
        if ink and i.get("fit") and arr is not None:
            bounds = [f for f, _ in i["stops"]] + [1.0]
            fitted = _fit_colour(arr, i["box"], ink[4], i["fit"][1], i["fit"][2],
                                 list(zip(bounds[:-1], bounds[1:])))
            i["stops"] = [(f, c2 or c) for (f, c), c2 in zip(i["stops"], fitted)]
        if len(i["stops"]) >= 2:
            parts = _colour_runs(i["text"], i["stops"], measure_face(i["family"], i["weight"]))
            colour = ("in parts, one <span> each (not a gradient or background-clip): "
                      + ", then ".join('"%s" color:%s' % (t.replace('"', "'"), c) for t, c in parts))
        else:
            colour = "color:%s" % i["stops"][0][1]
        lines.append('- "%s" -- %s; %s; white-space:nowrap  (measured width %dpx)'
                     % (i["text"].replace('"', "'"), css, colour, round(x1 - x0)))
        ref.append("<p>%s</p>" % html.escape(i["text"]))
    return lines, "<html><body>%s</body></html>" % "".join(ref), weights


GENERATE_PROMPT = """\
Recreate the attached screenshot as one HTML page that looks exactly like it, at exactly
%(width)d x %(height)d px. The two will be put side by side, and it should not be possible
to tell which is the screenshot.

CANVAS
`html, body { margin: 0; padding: 0 }` and one root element exactly %(width)d x %(height)d px
with `position: relative; overflow: hidden`. Nothing may depend on the window size: no
media queries, no percentage or viewport units for layout.

PICTURES AND BACKGROUND -- ALREADY DONE
These regions of the screenshot are pictures, and they are cut from it and placed on the
page for you, exactly where they are listed. So is the page's background -- its colour,
gradients and glows, measured off the screenshot. Do not write <img> tags for pictures, do
not give html, body or the root element a background, and do not draw page-wide gradients.
Leave the picture regions empty: they already hold everything drawn inside them, text
included.
%(assets)s

TEXT
All other text is written as real HTML text. Each line below was measured off the
screenshot; the CSS puts its letters where the screenshot has them, so position each line
absolutely with that CSS. The words were read by OCR and contain mistakes -- words run
together, l and I confused, fragments garbled -- so read every line in the screenshot and
write what it actually says. Add no text that is not in the screenshot and leave none out.
Where you correct a line -- put back a space the OCR lost, say -- keep it the measured
width by adjusting its letter-spacing, so its letters still land where they are.
%(texts)s

TYPEFACES
Headings are set in '%(heading)s' and body text in '%(body)s'. Use exactly those
font-family names. The fonts are supplied with the page, so do not link to or import any.

EVERYTHING ELSE
Cards, panels, buttons, inputs, badges, borders, radii, shadows and dividers are built in
CSS, matching the screenshot's colours, sizes and positions exactly. Buttons and links are
real <button> and <a> elements, with their labels positioned as measured above. Nothing
animates.

Return the complete HTML document and nothing else -- no explanation, no markdown fences.
Start with <!DOCTYPE html>.
"""


def build_generate_prompt(asset_lines: List[str], text_lines: List[str],
                          fonts: Dict[str, str], width: int = 1400,
                          height: int = 900) -> str:
    heading = fonts.get("heading") or fonts.get("body") or "Inter"
    return GENERATE_PROMPT % {
        "assets": "\n".join(asset_lines) or "(there are none)",
        "texts": "\n".join(text_lines) or "(there is none)",
        "heading": heading,
        "body": fonts.get("body") or heading,
        "width": width,
        "height": height,
    }


# The critique is told what is exact, so that it does not report it. It used to judge
# everything, and it cannot measure: it put logos at y=865 on an 836px page and a sun
# icon at x=855 when it sits at 1360, and asked for crops that were already exact to be
# moved -- which the correction did, and the page fell from 91.6% to 64.6%.
CRITIQUE_PROMPT = """\
Two screenshots are attached, both %(width)d x %(height)d px:
1. THE TARGET -- the page to reproduce.
2. THE ATTEMPT -- the current HTML, rendered.

Some things in THE ATTEMPT are exact already and must not be reported: the pictures
(photographs, mockups, device screens, avatars, logos, icons), which are cut from THE
TARGET and placed where they are in it; the page's background colours and gradients,
taken from it too; and where each line of text starts.

Compare everything else. Buttons, inputs, cards, panels, badges and dividers: is each one
there, and are its colour, border, corner radius, shadow and size right? The text: does
it read the same, and are its colour, weight and size right? Name each thing by what it
is and what it says -- the "Book a call" button, the line "your buyers read." -- not by
coordinates, and say what it should be instead.

At most 8 lines, each starting with "- ", the most visible first. If there is nothing
worth changing, reply with exactly: - nothing worth changing
"""


FIX_PROMPT = """\
Two screenshots are attached, both %(width)d x %(height)d px:
1. THE TARGET -- the page to reproduce.
2. THE ATTEMPT -- the HTML below, rendered.

These differences were found between them:
%(issues)s

Correct them in the HTML below so it renders like THE TARGET. Change only what those
differences call for:
- Do not move any line of text. Keep its left, top, font-size and letter-spacing as they
  are, unless a difference is specifically about that line.
- Do not add images and do not give the page a background: the pictures and the page's
  background are added separately, exactly where they belong.
- Keep the fixed %(width)d x %(height)d px canvas, the same font families, and no animation.

Return the complete corrected document and nothing else -- no explanation, no markdown
fences. Start with <!DOCTYPE html>.

%(html)s
"""


def build_critique_prompt(width: int = 1400, height: int = 900) -> str:
    return CRITIQUE_PROMPT % {"width": width, "height": height}


def build_fix_prompt(current: str, issues: List[str],
                     width: int = 1400, height: int = 900) -> str:
    return FIX_PROMPT % {
        "issues": "\n".join("- " + i for i in issues),
        "html": current,
        "width": width,
        "height": height,
    }


def parse_critique(reply: str) -> List[str]:
    """The lines of a critique, cleaned. Empty when it found nothing worth changing."""
    out = []
    for line in _unfence(reply).splitlines():
        line = line.strip()
        if not line.startswith("-"):
            continue
        text = line.lstrip("- ").strip()
        if not text or text.lower().startswith("nothing worth changing"):
            continue
        out.append(text)
    return out[:8]


def stream_gemini(prompt: str, images: Optional[List[Tuple[str, str]]] = None,
                  api_key: Optional[str] = None, model: Optional[str] = None,
                  timeout: int = 600, attempts: int = RETRY_ATTEMPTS):
    """Yields ("chunk", text) as the model writes, and ("status", text) while waiting.

    Server-sent events rather than one blocking call, because a page takes a minute or
    two to write and watching it appear is most of the point.
    """
    parts: List[Dict[str, Any]] = [{"text": prompt}]
    for shot in (images or []):
        if shot:
            parts.append({"inline_data": {"mime_type": shot[0], "data": shot[1]}})
    body = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 65536},
    }).encode("utf-8")

    def stream_url(name):
        return (_ENDPOINT.format(model=name).replace(":generateContent",
                                                     ":streamGenerateContent")
                + "?alt=sse")
    # Opening the stream gets the same treatment as any other call: walk the keys, then
    # the models, and wait only for a busy server. It had none of that, and a single 503
    # -- the most common answer this API gives -- ended the run with a traceback. Only
    # the connection is retried; once chunks have reached the caller there is no
    # starting again without repeating what it has already shown.
    resp = None
    last = None
    chosen = None
    names = models_to_try(model)
    for model_name in names:
        if not _worth_asking(model_name, names, api_key):
            continue
        keys = live_keys(model_name, api_key)
        if not keys:
            continue
        for key in keys:
            give_up_on_key = False
            for attempt in range(1, max(1, attempts) + 1):
                req = urllib.request.Request(
                    stream_url(model_name), data=body,
                    headers={"Content-Type": "application/json",
                             "x-goog-api-key": key},
                    method="POST")
                try:
                    resp = urllib.request.urlopen(req, timeout=timeout)
                    chosen = model_name
                    break
                except urllib.error.HTTPError as e:
                    detail = redact_keys(e.read().decode("utf-8", "replace")[:600])
                    last = EnhancementError(
                        f"Gemini returned {e.code}: {detail}")
                    verdict = _key_verdict(e.code, detail)
                    if verdict:
                        if verdict == "dead":
                            mark_dead(key, short_reason(last))
                        else:
                            mark_spent(key, model_name, short_reason(last))
                        yield ("status", "%s on %s: %s. Trying the next one..."
                               % (key_label(key), model_name, short_reason(last)))
                        give_up_on_key = True
                        break
                    if e.code not in RETRY_STATUSES or attempt >= attempts:
                        give_up_on_key = True
                        break
                    wait = None
                    marker = '"retryDelay"'
                    at = detail.find(marker)
                    if at >= 0:
                        for piece in detail[at + len(marker):].split('"'):
                            body_num = piece[:-1] if piece.endswith("s") else ""
                            if body_num.replace(".", "", 1).isdigit():
                                wait = float(body_num)
                                break
                    delay = wait or (RETRY_BACKOFF * (2 ** (attempt - 1)))
                    yield ("status",
                           "%s is busy (%d). Waiting %.0fs, attempt %d of %d..."
                           % (model_name, e.code, delay, attempt + 1, attempts))
                    time.sleep(delay)
                except (urllib.error.URLError, TimeoutError, OSError) as e:
                    last = EnhancementError(f"Could not reach Gemini: {e}")
                    if attempt >= attempts:
                        give_up_on_key = True
                        break
                    delay = RETRY_BACKOFF * (2 ** (attempt - 1))
                    yield ("status", "Connection failed. Waiting %.0fs, attempt %d of %d..."
                           % (delay, attempt + 1, attempts))
                    time.sleep(delay)
            if resp is not None or not give_up_on_key:
                break
        if resp is not None:
            break
    if resp is None:
        raise last or EnhancementError("No key or model could open the stream.")
    if chosen:
        _LAST_MODEL[0] = chosen
        if chosen != (model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL):
            yield ("status", "Using %s instead." % chosen)

    # What the stream saw, so that "no HTML came back" can say why. Without this the
    # failure was a bare message with nothing after the colon and no way to tell a
    # safety block from a token limit from a model that only thought and never wrote.
    seen = {"events": 0, "finish": "", "blocked": "", "thoughts": 0, "chars": 0}
    with resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            seen["events"] += 1
            feedback = chunk.get("promptFeedback") or {}
            if feedback.get("blockReason"):
                seen["blocked"] = str(feedback["blockReason"])
            for cand in chunk.get("candidates", []):
                if cand.get("finishReason"):
                    seen["finish"] = str(cand["finishReason"])
                for part in (cand.get("content") or {}).get("parts", []) or []:
                    text = part.get("text")
                    if not text:
                        continue
                    # A thinking model streams its reasoning as ordinary text parts
                    # flagged as thoughts. Letting those through would paste the
                    # model's deliberation into the page.
                    if part.get("thought"):
                        seen["thoughts"] += len(text)
                        continue
                    seen["chars"] += len(text)
                    yield ("chunk", text)

    if not seen["chars"]:
        why = []
        if seen["blocked"]:
            why.append("the request was blocked (%s)" % seen["blocked"])
        if seen["finish"] and seen["finish"] != "STOP":
            why.append("it stopped early (%s)" % seen["finish"])
        if seen["thoughts"]:
            why.append("it produced %d characters of reasoning and no page"
                       % seen["thoughts"])
        if not seen["events"]:
            why.append("the stream carried no events at all")
        raise EnhancementError(
            "%s wrote nothing: %s." % (chosen or "The model",
                                       "; ".join(why) or "no reason was given"))


# The loop's budget: one call to read the screenshot, one to write the page, and two a
# round after that. Before the first rewrite of this path it was thirty-two a click.
VERIFY_ROUNDS = 3
MAX_VERIFY_ROUNDS = 8
# What a version has to add to the best so far to replace it, in colour-SSIM points.
# Below this the difference is the noise of re-rendering, not an improvement.
MIN_GAIN = 0.3
# Rounds in a row that add nothing before the loop stops asking.
STALL_LIMIT = 2
# No new round starts after this many seconds of refinement.
REFINE_TIME_BUDGET = 600.0
GENERATE_TIMEOUT = 300
CRITIQUE_TIMEOUT = 150
LOOP_ATTEMPTS = 2


def generate_from_document(doc, api_key: Optional[str] = None,
                           model: Optional[str] = None, timeout: int = GENERATE_TIMEOUT,
                           verify: int = VERIFY_ROUNDS,
                           target_score: float = TARGET_ACCURACY_SCORE,
                           page_index: int = 0):
    """Recreates one page so it looks like its screenshot.

    Yields:
    - ("phase", label)       what stage it is in, for the progress bar's heading
    - ("status", text)       what it is doing right now
    - ("layer", html)        the background and pictures, placed, for the live preview
    - ("fonts", css)         the embedded typefaces, for the live preview
    - ("chunk", text)        the page as it is written
    - ("score", {...})       a version was measured
    - ("issue", text)        a difference the check found
    - ("revision", {"html"}) a better version
    - ("done", {...})        the best version, and how it got there
    """
    from engine import DocumentData, Exporter

    pages = list(getattr(doc, "pages", None) or [])
    if not pages:
        raise EnhancementError("This document has no pages.")
    page = pages[max(0, min(int(page_index or 0), len(pages) - 1))]
    shot = as_inline_image(getattr(page, "originalImageSrc", None))
    if not shot:
        raise EnhancementError("This page has no original screenshot to work from.")
    w, h = _shot_dims(shot)
    rounds_cap = max(0, min(int(verify), MAX_VERIFY_ROUNDS))
    calls = 0

    # ---- the engine's page, measured only so the result can be compared with it -----
    yield "phase", "Measuring the page"
    one = DocumentData(title=getattr(doc, "title", None) or "page", pageCount=1, pages=[page])
    engine_page = with_fonts(at_origin(Exporter.export_standalone_html(one, interactive=False)))
    engine_render = render_html(engine_page, width=w, height=h)
    engine_score = calculate_ssim_score(shot, engine_render) if engine_render else 0.0

    # ---- what the screenshot is made of: typefaces and pictures -------------------
    yield "phase", "Reading the screenshot"
    yield "status", "Identifying the typefaces and the pictures in the screenshot..."
    calls += 1
    try:
        found = analyse_screenshot(shot, w, h, api_key=api_key, model=model,
                                   timeout=CRITIQUE_TIMEOUT, attempts=LOOP_ATTEMPTS)
    except (EnhancementError, ValueError) as e:
        yield "status", "The screenshot could not be read (%s); carrying on without it." % (
            short_reason(e) if isinstance(e, EnhancementError) else "no JSON came back")
        found = {"fonts": {}, "pictures": []}
    fonts = found["fonts"] or {}
    if not fonts:
        # The engine's own choice, rather than a guess.
        fam = next((e.style.fontFamily for e in page.elements
                    if e.type == "text" and e.style is not None), "") or ""
        first = fam.split(",")[0].strip().strip("'\"") or "Inter"
        fonts = {"heading": first, "body": first}
    gen_model = _LAST_MODEL[0] or model or DEFAULT_MODEL

    text_boxes = [tuple(float(v) for v in e.bbox) for e in page.elements
                  if e.type == "text" and (e.text or "").strip()]
    assets, asset_lines, picture_boxes = picture_crops(shot, found["pictures"], text_boxes)
    surfaces = [tuple(float(v) for v in e.bbox) for e in page.elements if e.type == "rect"
                and (e.bbox[2] - e.bbox[0]) * (e.bbox[3] - e.bbox[1]) < w * h * PLATE_SURFACE_SHARE]
    plate = background_plate(shot, text_boxes + [tuple(b) for b in picture_boxes] + surfaces)
    layer = picture_layer(assets, picture_boxes, plate, w, h)
    yield "layer", layer
    yield "status", ("Set in %s%s; %d picture(s) cut from the screenshot."
                     % (fonts.get("heading"),
                        "" if fonts.get("body") == fonts.get("heading")
                        else " and " + str(fonts.get("body")), len(assets)))

    families = sorted({f for f in (fonts.get("heading"), fonts.get("body")) if f})
    font_css = embedded_family_css(families, list(WEIGHT_CANDIDATES))
    yield "status", "Measuring each line of text in %s..." % " and ".join(families)
    text_lines, reference, weights = text_hints(page, picture_boxes, fonts, shot, font_css)
    if not font_css:
        yield "status", "The typefaces could not be fetched; the page will use a fallback."
    yield "fonts", font_css

    def complete(raw: str) -> Tuple[str, List[int]]:
        return apply_fonts(lock_pictures(raw, layer), font_css), []

    def measure(raw: str) -> Tuple[Optional[Tuple[str, str]], float]:
        rendered = render_html(complete(raw)[0], width=w, height=h)
        return rendered, (calculate_ssim_score(shot, rendered) if rendered else 0.0)

    versions: List[Dict[str, Any]] = []

    def better(a: Dict[str, Any], b: Optional[Dict[str, Any]]) -> bool:
        """Whether version a should replace b as the one to show."""
        if b is None:
            return True
        if bool(a["refused"]) != bool(b["refused"]):
            return not a["refused"]       # keeping the page's text comes first
        return a["score"] > b["score"] + MIN_GAIN

    def take(raw: str, source: str, used_model: str) -> Dict[str, Any]:
        rendered, score = measure(raw)
        v = {"raw": raw, "render": rendered, "score": score, "source": source,
             "model": used_model, "refused": keeps_the_content(reference, raw) or ""}
        versions.append(v)
        return v

    # ---- the page, written from the screenshot and streamed -----------------------
    yield "phase", "Writing the page"
    yield "status", "Writing the page from the screenshot..."
    prompt = build_generate_prompt(asset_lines, text_lines, fonts, width=w, height=h)
    buf: List[str] = []
    tried: List[str] = []
    for candidate in models_to_try(gen_model):
        if candidate in tried:
            continue
        tried.append(candidate)
        buf = []
        calls += 1
        try:
            for kind, piece in stream_gemini(prompt, images=[shot], api_key=api_key,
                                             model=candidate, timeout=timeout):
                if kind == "status":
                    yield "status", piece
                    continue
                buf.append(piece)
                yield "chunk", piece
        except EnhancementError as e:
            if buf:
                raise
            remaining = [m for m in models_to_try(gen_model) if m not in tried]
            if not remaining:
                raise
            yield "status", "%s. Trying %s..." % (short_reason(e), remaining[0])
            continue
        if buf:
            break
    gen_model = _LAST_MODEL[0] or gen_model
    raw = _unfence("".join(buf))
    if "<" not in raw:
        raise EnhancementError("%s did not return HTML. It said: %s"
                               % (gen_model, raw.strip()[:200] or "(nothing)"))
    if "</html>" not in raw.lower():
        yield "status", "The page came back cut short; measuring what there is."

    best = take(raw, "rewrite", gen_model)
    # Pictures the first look missed show as detail in the screenshot and bare ground in
    # the page. They are cut and placed like the rest, and the page measured again.
    if best["render"]:
        extra = unplaced_pictures(shot, best["render"], text_boxes, picture_boxes)
        if extra:
            from PIL import Image
            im = Image.open(io.BytesIO(base64.b64decode(shot[1]))).convert("RGB")
            assets = assets + [_crop_uri(im, b) for b in extra]
            picture_boxes = list(picture_boxes) + extra
            layer = picture_layer(assets, picture_boxes, plate, w, h)
            yield "layer", layer
            yield "status", "%d more picture(s) found in the screenshot and placed." % len(extra)
            versions.remove(best)
            best = take(raw, "rewrite", gen_model)
    if best["render"]:
        yield "score", {"score": round(best["score"], 1), "target": target_score,
                        "round": 0, "max_rounds": rounds_cap, "label": "rewrite"}
    yield "revision", {"html": complete(best["raw"])[0]}
    if best["refused"]:
        yield "status", "The page lost some of the text: %s." % best["refused"]

    # ---- look at it beside the screenshot, and correct it -------------------------
    all_issues: List[str] = []
    rounds_done = 0
    stalls = 0
    started = time.time()
    for rnd in range(1, rounds_cap + 1):
        if not best["render"]:
            break
        if best["score"] >= target_score and not best["refused"]:
            yield "status", "%.1f%% is close enough to stop." % best["score"]
            break
        if time.time() - started > REFINE_TIME_BUDGET:
            yield "status", "Out of time for corrections; keeping the best version."
            break

        yield "phase", "Checking, round %d of %d" % (rnd, rounds_cap)
        yield "status", "Comparing the page with the screenshot..."
        calls += 1
        try:
            critique = call_gemini(build_critique_prompt(w, h), api_key=api_key,
                                   model=gen_model, timeout=CRITIQUE_TIMEOUT,
                                   attempts=LOOP_ATTEMPTS, images=[shot, best["render"]])
        except EnhancementError as e:
            yield "status", "The check could not be made: %s." % short_reason(e)
            break
        issues = parse_critique(critique)
        if best["refused"]:
            issues = ["Some of the screenshot's text is missing or wrong: %s." % best["refused"]] + issues
        if not issues:
            yield "status", "The check found nothing worth changing."
            break
        for issue in issues:
            yield "issue", issue

        yield "phase", "Fixing, round %d of %d" % (rnd, rounds_cap)
        yield "status", "Correcting %d difference(s)..." % len(issues)
        calls += 1
        try:
            fixed = _unfence(call_gemini(
                build_fix_prompt(best["raw"], issues, w, h),
                api_key=api_key, model=gen_model, timeout=timeout,
                attempts=LOOP_ATTEMPTS, images=[shot, best["render"]]))
        except EnhancementError as e:
            yield "status", "The correction could not be made: %s." % short_reason(e)
            stalls += 1
            if stalls >= STALL_LIMIT:
                break
            continue
        if "<" not in fixed:
            stalls += 1
            yield "status", "Round %d returned no HTML." % rnd
            if stalls >= STALL_LIMIT:
                break
            continue

        v = take(fixed, "round %d" % rnd, _LAST_MODEL[0] or gen_model)
        if v["render"]:
            yield "score", {"score": round(v["score"], 1), "target": target_score,
                            "round": rnd, "max_rounds": rounds_cap, "label": "round %d" % rnd}
        if better(v, best):
            best = v
            stalls = 0
            rounds_done += 1
            all_issues.extend(issues)
            yield "status", "Round %d kept: %.1f%%." % (rnd, best["score"])
            yield "revision", {"html": complete(best["raw"])[0]}
        else:
            stalls += 1
            why = v["refused"] or ("%.1f%% does not beat %.1f%%" % (v["score"], best["score"]))
            yield "status", "Round %d was not kept: %s." % (rnd, why)
            if stalls >= STALL_LIMIT:
                yield "status", "Stopping: %d rounds in a row added nothing." % STALL_LIMIT
                break

    final, missing = complete(best["raw"])
    yield "done", {
        "html": final,
        "model": best["model"],
        "source": best["source"],
        "engine_score": round(engine_score, 1),
        "score": round(best["score"], 1),
        "target": target_score,
        "fonts": fonts,
        "pictures": len(assets),
        "assets_total": len(assets),
        "assets_missing": missing,
        "verify_rounds": rounds_done,
        "issues": all_issues,
        "text_check": best["refused"] or "kept",
        "calls": calls,
        "chars": len(final),
    }


# ---------------------------------------------------------------- command line

def _main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Rewrite a reconstruction into a laid-out, animated page.")
    ap.add_argument("source", nargs="?",
                    help="a screenshot or PDF to reconstruct first, a .json document "
                         "export, or an already-rendered .html file")
    ap.add_argument("-o", "--out", help="where to write the result "
                                        "(default: <source>.enhanced.html)")
    ap.add_argument("-m", "--model", help=f"default: {DEFAULT_MODEL}")
    ap.add_argument("-i", "--instructions", help="extra direction for the rewrite")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the prompt and report its size without calling anything")
    ap.add_argument("--list-models", action="store_true",
                    help="ask the API which models this key can call, and exit")
    ap.add_argument("--no-flatten", action="store_true",
                    help="send the raster fragments as-is instead of composing them")
    ap.add_argument("-r", "--reference",
                    help="screenshot of the intended result (defaults to the one the "
                         "document already carries)")
    ap.add_argument("--no-reference", action="store_true",
                    help="do not send a screenshot, only the HTML")
    ap.add_argument("--refine", type=int, default=REFINE_ROUNDS, metavar="N",
                    help="render the result and ask the model to close the gap with the "
                         "original, N times (default %d)" % REFINE_ROUNDS)
    args = ap.parse_args(argv)

    if args.list_models:
        for name in list_models():
            print(name)
        return 0

    if not args.source:
        ap.error("a source is required unless --list-models is given")

    lower = args.source.lower()
    reference = args.reference
    if lower.endswith((".html", ".htm")):
        with open(args.source, encoding="utf-8") as fh:
            html = fh.read()
        if not args.no_flatten:
            print("note: raster flattening needs the document model, so it is skipped "
                  "for .html input. Pass the image, or a .json export, to get it.")
    else:
        from engine import DocumentData, HTMLRenderer, ImageReconstructor
        if lower.endswith(".json"):
            import json as _json
            with open(args.source, encoding="utf-8") as fh:
                doc = DocumentData.model_validate(_json.load(fh))
        else:
            page = ImageReconstructor.reconstruct_image(args.source)
            doc = DocumentData(title=os.path.basename(args.source), pageCount=1,
                               pages=[page])
        # The reconstruction carries the screenshot it was built from, so the reference
        # costs nothing to find and is the only unmediated description of the page.
        if reference is None and not lower.endswith(".json"):
            reference = args.source
        if reference is None:
            ref = reference_from_document(doc)
            if ref:
                reference = ref
        if not args.no_flatten:
            before = sum(1 for p in doc.pages for e in p.elements if e.type == "image")
            doc = flatten_document(doc)
            after = sum(1 for p in doc.pages for e in p.elements if e.type == "image")
            if after != before:
                print(f"flattened {before} raster fragments into {after} images")
        html = HTMLRenderer.render_document(doc, editable=False, interactive=False)

    if args.no_reference:
        reference = None

    if args.dry_run:
        skeleton, assets = strip_assets(html)
        prompt = build_prompt(skeleton, len(assets), args.instructions,
                              describe_assets(assets, roles=asset_roles(html, assets)),
                              has_reference=bool(as_inline_image(reference)))
        print(f"page      {len(html):>9,} chars")
        print(f"prompt    {len(prompt):>9,} chars  (~{len(prompt)//4:,} tokens)")
        print(f"assets    {len(assets):>9,} held back, "
              f"{100.0 * (len(html) - len(skeleton)) / max(len(html), 1):.1f}% of the page")
        shot = as_inline_image(reference)
        print(f"reference {'attached, ~1k tokens' if shot else 'none'}")
        print(f"key       {'present' if is_configured() else 'MISSING - see .env.example'}")
        return 0

    out = args.out or (os.path.splitext(args.source)[0] + ".enhanced.html")
    started = time.time()

    def note(attempt, total, code, delay):
        print(f"  {code} from the model (attempt {attempt}/{total}); "
              f"retrying in {delay:.0f}s", flush=True)

    try:
        result = enhance_html(
            html, model=args.model, extra=args.instructions, on_retry=note,
            reference=reference, refine=args.refine,
            on_round=lambda i, n: print("  comparing the render against the original "
                                        "(round %d/%d)" % (i, n), flush=True))
    except EnhancementError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(str(result["html"]))
    print(f"{result['model']} in {time.time() - started:.0f}s -> {out}"
          + ("  (with the screenshot)" if result.get("saw_reference") else ""))
    print(f"  {result['original_chars']:,} chars in, {result['enhanced_chars']:,} out; "
          f"sent {result['sent_chars']:,}")
    refined = result.get("refine_rounds") or 0
    if refined:
        print(f"  refined over {refined} visual round(s)")
    rounds = result.get("repair_rounds") or 0
    first = result.get("assets_missing_initially") or []
    if first:
        print(f"  the rewrite dropped {len(first)} image(s): {first}")
    if rounds:
        print(f"  recovered {result.get('assets_recovered', 0)} of them "
              f"over {rounds} repair round(s)")
    for note in result.get("notes") or []:
        print(f"  {note}")
    missing = result["assets_missing"]
    if missing:
        print(f"  WARNING: {len(missing)} image(s) still missing: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
