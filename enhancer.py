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
    settled = html.replace(
        "</head>",
        "<style>*,*::before,*::after{animation:none!important;"
        "transition:none!important;opacity:1!important}</style></head>", 1)
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
    """Computes SSIM visual match percentage (0.0 to 100.0) between shot and render."""
    if not shot or not render:
        return 0.0
    try:
        shot_bytes = base64.b64decode(shot[1])
        render_bytes = base64.b64decode(render[1])
        from engine import FidelityChecker
        res = FidelityChecker.compare_images(shot_bytes, render_bytes)
        if "similarityPercent" in res:
            return float(res["similarityPercent"])
    except Exception as e:
        logger.warning("Failed to calculate SSIM score: %s", e)
    return 0.0


_WORDS_RE = re.compile(r"[^\W_]+", re.UNICODE)
_TAGS_RE = re.compile(r"<[^>]+>")
_HEADSTYLE_RE = re.compile(r"<(head|script|style)[^>]*>.*?</\1>", re.S | re.I)


_CONTROL_RE = re.compile(r"<(button|input|select|textarea)[\s>/]", re.I)

# A refinement may tidy wording it inherited, but wholesale new text means it has
# started transcribing what it sees in the render instead of reading the HTML.
REFINE_MAX_WORD_GAIN = 4


def _count_controls(html: str) -> int:
    """Real interactive elements. A rewrite that turns them all into <a> has lost."""
    return len(_CONTROL_RE.findall(html))


def _visible_words(html: str):
    """Multiset of words a reader would see. Used to refuse a lossy refinement."""
    from collections import Counter
    import html as _h
    body = _HEADSTYLE_RE.sub(" ", html)
    body = _TAGS_RE.sub(" ", body)
    return Counter(w.lower() for w in _WORDS_RE.findall(_h.unescape(body)))


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

# Target visual match percentage (SSIM) before exiting the refinement loop.
TARGET_ACCURACY_SCORE = 95.0

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
    for model_name in models_to_try(model):
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
    skeleton, assets = strip_assets(html)
    asset_lines = describe_assets(assets, roles=asset_roles(html, assets))
    shot = as_inline_image(reference)
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
        render = render_html(current_html, width=w, height=h)
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
        test_render = render_html(test_html, width=w, height=h)
        if test_render:
            test_score = calculate_ssim_score(shot, test_render)
            if test_score >= score - 1.5:
                reply = better
                score = test_score
                if score > best_score:
                    best_score = score
                    best_reply = reply
                if score >= target_score:
                    rounds_done += 1
                    break
        else:
            reply = better

        rounds_done += 1

    enhanced, missing = restore_assets(best_reply, assets)
    return {
        "html": enhanced,
        "refine_rounds": rounds_done,
        "model": model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODEL,
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
# Building the page from the screenshot, streamed
# ============================================================================
#
# A different job from `enhance_html`, and worth having beside it. That one hands the
# model the reconstruction and asks for it to be rebuilt, which anchors the result to
# whatever the engine made of the page -- including its mistakes. This asks for the page
# to be written from the screenshot, the way screenshot-to-code does it, and streams the
# answer so the page can be watched assembling itself.
#
# What it keeps from Ramen rather than from that project: the images are real. Where
# screenshot-to-code fills in placehold.co boxes and describes what ought to go in them,
# the markers here resolve to the artwork the engine actually cut out of the screenshot.
# The text is real too -- OCR measured it, so the model is given the strings instead of
# being asked to read them off a picture, which is where transcription errors come from.

GENERATE_PROMPT = """\
You are an expert front-end developer. You are given a screenshot of a web page and you
write a single self-contained HTML file that looks exactly like it.

- Match the screenshot closely: background colours, text colour, font size and weight,
  spacing, alignment, borders, radii, shadows.
- Write the FULL code. Never write a comment in place of content -- no "<!-- repeat for
  each item -->", no "<!-- other nav links here -->". If the screenshot shows nine cards,
  write nine cards.
- Use semantic HTML: header, nav, main, section, footer, h1-h6, p, ul/li, button, a.
- Lay it out with flexbox and grid so it reflows. Make it usable down to 480px.
- Add hover and focus states on everything interactive, and short transitions
  (150-300ms). Disable them under `@media (prefers-reduced-motion: reduce)`.
- Put all CSS in one <style> block in the head. Use CSS custom properties for the
  palette. Plain CSS, no framework, no CDN, no external requests of any kind.

THE TEXT
Use these strings, exactly as written, for the page's text. They were measured from the
screenshot, so they are more reliable than reading the picture -- including where a word
looks wrong: several are genuinely cut off behind something in the design, and must stay
cut off. Do not correct, translate, complete or invent any of them.

%(texts)s

THE IMAGES
Every picture in this page has already been extracted for you. Use these markers as the
`src` of an `<img>` -- write the marker exactly, it is replaced with the real image
afterwards. Each line gives the marker, the size it was in the original, its main
colours, and what kind of thing it is. Use all %(n_assets)d of them, and do not invent
any others or link to any external image.

%(assets)s

Return the complete HTML document and nothing else. No explanation, no markdown fences.
Start with `<!DOCTYPE html>`.
"""


# Set from looking at the crops rather than guessing. Laid out as a contact sheet, the
# genuine rubbish on the Reddit page is unmistakable and all of one kind: 178x3, 148x4,
# 19x5, 44x7 -- antialiasing rims, every one under eight pixels on its short side.
# Above that line sit real things, including a 13x17 share icon and a 14x11 plus icon
# that an earlier threshold of 20 discarded, after which the model quite correctly
# reported the icons as missing and could do nothing about it.
ASSET_MIN_SIDE = 8
ASSET_MIN_AREA = 100


def _asset_dims(assets: List[str]) -> List[Optional[Tuple[int, int]]]:
    """Pixel size of each data URI, or None where it cannot be read."""
    try:
        from PIL import Image
    except ImportError:
        return [None] * len(assets)
    out: List[Optional[Tuple[int, int]]] = []
    for uri in assets:
        try:
            raw = base64.b64decode(uri.partition(",")[2])
            out.append(Image.open(io.BytesIO(raw)).size)
        except Exception:
            out.append(None)
    return out


def page_texts(doc) -> List[str]:
    """Every string the engine read, in reading order, deduplicated."""
    out, seen = [], set()
    for page in (getattr(doc, "pages", None) or []):
        rows = []
        for e in (page.elements or []):
            text = (getattr(e, "text", None) or "").strip()
            if text and e.type in ("text", "rect"):
                b = e.bbox or [0, 0, 0, 0]
                rows.append((round(float(b[1]) / 12.0), float(b[0]), text))
        for _, _, text in sorted(rows):
            key = text.lower()
            if key not in seen:
                seen.add(key)
                out.append(text)
    return out


def build_generate_prompt(texts: List[str], asset_lines: List[str]) -> str:
    return GENERATE_PROMPT % {
        "texts": "\n".join("- " + t.replace("\n", " ") for t in texts) or "(none)",
        "assets": "\n".join(asset_lines) or "(none)",
        "n_assets": len(asset_lines),
    }


CRITIQUE_PROMPT = """\
Two screenshots are attached:

1. THE TARGET -- the original design to reproduce.
2. THE ATTEMPT -- the current HTML rendered in a browser just now.

List what is wrong with the attempt compared to the target. One short line each, at most ten, most serious first.
Look for:
- Missing sections, logos, brand marks, partner logos, or text that are in the target but absent in the attempt.
- Misaligned, squished, or overlapping navigation items, headers, cards, or mockups.
- Spacing, padding, and alignment that do not match.
- Image assets (RAMEN_ASSET_<n>) at the wrong size, missing, or in the wrong place.
- Typography (font size, weight, line height, color) that does not match.

Reply with the lines and nothing else, each starting with "- ". If the attempt already matches the target at 95%+ fidelity, reply with the single line "- nothing worth changing".
"""


FIX_PROMPT = """\
Two screenshots are attached:
1. THE TARGET -- the original design to reproduce.
2. THE ATTEMPT -- the current HTML rendered in a browser.

Below is the current HTML document and a list of problems found by comparing the attempt against the target:

PROBLEMS
%(issues)s

Fix all of those problems and return the corrected document so the rendered page matches THE TARGET (target fidelity >= 95%%).
- Accurately add any missing sections, partner brand logos, badges, and labels mentioned in the problems.
- Fix any misaligned, squished, or overlapping navigation items, cards, or mockups using proper flexbox/grid layout and CSS.
- Image sources are `RAMEN_ASSET_<n>` markers, numbered 0 to %(max_asset)d. %(unused_note)s
- Keep the layout flowing and responsive down to 480px, with hover/focus states.

Return the complete document and nothing else. No explanation, no markdown fences.
Start with `<!DOCTYPE html>`.

%(html)s
"""


def build_critique_prompt() -> str:
    return CRITIQUE_PROMPT


def build_fix_prompt(current: str, issues: List[str], n_assets: int) -> str:
    unused = missing_markers(current, n_assets)
    if unused:
        note = ("Markers %s are not used anywhere, so those pictures are invisible: "
                "place them where they belong." % ", ".join(str(i) for i in unused[:12]))
    else:
        note = "All of them are in use; keep it that way."
    return FIX_PROMPT % {
        "issues": "\n".join("- " + i for i in issues) or "- layout does not match",
        "html": current,
        "max_asset": max(n_assets - 1, 0),
        "unused_note": note,
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
    return out[:10]


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
    for model_name in models_to_try(model):
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


# How many times to render the finished page, show it to the model beside the original,
# and let it correct itself. Each round is one request and about a minute.
VERIFY_ROUNDS = 15


def generate_from_document(doc, api_key: Optional[str] = None,
                           model: Optional[str] = None, timeout: int = 600,
                           verify: int = VERIFY_ROUNDS,
                           target_score: float = TARGET_ACCURACY_SCORE):
    """Streams a page written from the screenshot, then restores the real images.

    Yields:
    - ("status", text)
    - ("chunk", text)
    - ("issue", text)
    - ("score", {"score": float, "target": float, "round": int, "max_rounds": int})
    - ("revision", {"html": html})
    - ("done", result_dict)
    """
    from engine import DocumentData, HTMLRenderer

    shot = reference_from_document(doc)
    if not shot:
        raise EnhancementError(
            "This document has no original screenshot to work from.")

    # Preparation takes half a minute before a single character arrives, because
    # composing the images renders the page twice to check the composite is faithful.
    # Saying so beats a silent stare: the first version showed "Reading the
    # screenshot..." for thirty seconds and looked hung.
    yield "status", "Composing the extracted images..."

    # The engine's own render is only used to harvest the assets; the model never sees it.
    flat = flatten_document(doc)
    html = HTMLRenderer.render_document(
        DocumentData(title=getattr(doc, "title", "page") or "page",
                     pageCount=len(flat.pages or []), pages=list(flat.pages or [])),
        editable=False, interactive=False)
    _, all_assets = strip_assets(html)
    roles = asset_roles(html, all_assets)

    # Only offer pictures worth placing. The residual pass emits every scrap CSS could
    # not explain, and on the Reddit page 21 of its 31 crops are slivers -- 19x5, 178x3,
    # 148x4 -- which are antialiasing rims, not content. Handed all of them, the model
    # reasonably ignores the noise, and the run then reports "15 of 31 images missing"
    # as though half the page were lost. Dropping them shortens the prompt, leaves fewer
    # markers to misplace, and makes the count mean something.
    keep = [i for i, dims in enumerate(_asset_dims(all_assets))
            if dims and min(dims) >= ASSET_MIN_SIDE and dims[0] * dims[1] >= ASSET_MIN_AREA]
    assets = [all_assets[i] for i in keep]
    asset_lines = describe_assets(assets,
                                  roles={n: roles.get(i, "") for n, i in enumerate(keep)})
    dropped = len(all_assets) - len(assets)
    if dropped:
        yield "status", ("Set aside %d fragment(s) too small to place; offering %d image(s)."
                         % (dropped, len(assets)))
    prompt = build_generate_prompt(page_texts(flat), asset_lines)

    yield "status", ("Sending the screenshot, %d image(s) and %d line(s) of text..."
                     % (len(assets), len(page_texts(flat))))
    yield "assets", assets

    # A model that writes nothing is not a dead end while others remain. It happens --
    # a lite model spending its whole budget on reasoning, a stop before the first
    # token -- and it used to end the run with an error that had nothing after the
    # colon. Nothing has been shown to the caller at that point, so starting again on
    # the next model costs only the request.
    buf = []
    tried: List[str] = []
    for candidate in models_to_try(model):
        if candidate in tried:
            continue
        tried.append(candidate)
        buf = []
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
                raise                      # partly written; cannot start over cleanly
            remaining = [m for m in models_to_try(model) if m not in tried]
            if not remaining:
                raise
            yield "status", "%s. Trying %s..." % (short_reason(e), remaining[0])
            continue
        if buf:
            break
    if not buf:
        raise EnhancementError("No model produced a page.")

    raw = _unfence("".join(buf))
    if "<" not in raw:
        used = _LAST_MODEL[0] or model or DEFAULT_MODEL
        detail = raw.strip()[:200] or "(nothing at all)"
        raise EnhancementError(
            "%s did not return HTML. It said: %s" % (used, detail))

    # ---- look at the result and correct it ----------------------------------------
    #
    # One shot at a whole page from a screenshot gets the shape roughly right and the
    # details wrong, and the model has no way to know which is which: it never sees
    # what it built. Rendering the page and handing it back beside the original is the
    # correction it cannot otherwise make. Each round is announced, and so is every
    # problem it reports finding, because a minute of silence reads as a hang.
    all_issues: List[str] = []
    rounds_done = 0
    w, h = _shot_dims(shot)
    best_score = 0.0

    MAX_ROUNDS = max(15, verify)
    TARGET_SCORE = target_score or 95.0

    for attempt in range(1, MAX_ROUNDS + 1):
        current, _ = restore_assets(raw, assets)
        yield "status", "Rendering HTML screenshot for visual verification (round %d of %d)..." % (
            attempt, MAX_ROUNDS)
        render = render_html(current, width=w, height=h)
        if not render:
            yield "status", "No headless browser available, so skipping the checks."
            break

        score = calculate_ssim_score(shot, render)
        if score > best_score:
            best_score = score

        yield "score", {
            "score": round(score, 1),
            "target": TARGET_SCORE,
            "round": attempt,
            "max_rounds": MAX_ROUNDS
        }

        # Exit condition: if score >= target_score (95%+), we are done!
        if score >= TARGET_SCORE:
            yield "status", ("Match target achieved: %.1f%% >= %.1f%%!" % (score, TARGET_SCORE))
            break

        yield "status", ("Comparing attempt against target (current match: %.1f%%, target: ≥ %.1f%%)..." % (score, TARGET_SCORE))
        try:
            critique = call_gemini(build_critique_prompt(), api_key=api_key,
                                   model=model, timeout=timeout,
                                   images=[shot, render])
        except EnhancementError as e:
            yield "status", "Check %d critique call failed: %s." % (
                attempt, short_reason(e))
            break

        issues = parse_critique(critique)
        for issue in issues:
            yield "issue", issue
        unused = missing_markers(raw, len(assets))
        if not issues and not unused and score >= TARGET_SCORE:
            yield "status", "Check %d found nothing worth changing." % attempt
            break

        yield "status", "Applying %d fix(es)%s (targeting ≥ %.1f%% match)..." % (
            len(issues), " and placing %d unused image(s)" % len(unused) if unused else "", TARGET_SCORE)
        try:
            fixed = _unfence(call_gemini(
                build_fix_prompt(raw, issues, len(assets)), api_key=api_key,
                model=model, timeout=timeout,
                images=[shot, render]))
        except EnhancementError as e:
            yield "status", "Could not apply check %d - %s." % (
                attempt, short_reason(e))
            continue

        if not fixed or "<" not in fixed or len(fixed) < 100:
            yield "status", "Nothing valid came back to apply from check %d; retrying..." % attempt
            continue

        # Render the fixed page and check if it improved fidelity
        fixed_current, _ = restore_assets(fixed, assets)
        fixed_render = render_html(fixed_current, width=w, height=h)
        if fixed_render:
            fixed_score = calculate_ssim_score(shot, fixed_render)
            # Accept if it improves or stays roughly the same while addressing structural issues
            if fixed_score >= score - 2.0:
                raw = fixed
                rounds_done += 1
                all_issues.extend(issues)
                score = fixed_score
                if score > best_score:
                    best_score = score
                yield "status", ("Applied round %d: %d fix(es) (new match: %.1f%%)." % (
                    attempt, len(issues), score))
                rev_html, _ = restore_assets(raw, assets)
                yield "revision", {"html": rev_html}
                if score >= TARGET_SCORE:
                    yield "status", ("Match target achieved: %.1f%% >= %.1f%%!" % (score, TARGET_SCORE))
                    break
            else:
                yield "status", ("Round %d fix reduced match score (%.1f%% vs %.1f%%); continuing refinement..." % (
                    attempt, fixed_score, score))
        else:
            raw = fixed
            rounds_done += 1
            all_issues.extend(issues)
            rev_html, _ = restore_assets(raw, assets)
            yield "revision", {"html": rev_html}

    final, missing = restore_assets(raw, assets)
    yield "done", {
        "html": final,
        "model": _LAST_MODEL[0] or model or DEFAULT_MODEL,
        "assets_total": len(assets),
        "assets_missing": missing,
        "verify_rounds": rounds_done,
        "score": round(best_score, 1),
        "target": target_score,
        "issues": all_issues,
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
