import os
import sys
import site
import io
import re
import html
import uuid
import base64
import json
import zipfile
import logging
from collections import Counter
from typing import List, Optional, Dict, Any, Literal, Tuple
from pydantic import BaseModel, Field
from PIL import Image
import numpy as np
import cv2
import pymupdf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

# Windows DLL and Paddle CPU execution configuration
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

if sys.platform == 'win32':
    for p in site.getsitepackages():
        tlib = os.path.join(p, 'torch', 'lib')
        if os.path.exists(tlib) and hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(tlib)
            except Exception:
                pass

# Patch Paddle static runner config resolver to disable PIR & mkldnn on Windows
try:
    import paddlex.inference.models.runners.paddle_static.runner as psr
    _orig_static_resolve = psr.resolve_paddle_static_engine_config
    def _safe_static_resolve(model_name, engine_config):
        cfg = _orig_static_resolve(model_name, engine_config)
        cfg['enable_new_ir'] = False
        cfg['run_mode'] = 'paddle'
        return cfg
    psr.resolve_paddle_static_engine_config = _safe_static_resolve
except Exception:
    pass

logger = logging.getLogger("RamenEngine")

# ==============================================================================
# 1. DATA SCHEMA
# ==============================================================================

class TextStyle(BaseModel):
    fontFamily: str = 'sans-serif'
    fontSize: float = 12.0
    fontWeight: str = 'normal'
    fontStyle: str = 'normal'
    color: str = '#000000'
    textAlign: str = 'left'
    lineHeight: float = 1.0
    letterSpacing: Optional[float] = 0.0
    wordSpacing: Optional[float] = 0.0
    scaleX: Optional[float] = 1.0
    # Where the lettering changes colour along the run, as (share of the width, hex).
    # A line set in two colours -- "your buyers read.", black then accent -- cannot be
    # expressed by `color` alone, and splitting the text at the change needs to know
    # which character the change falls on, which the pixels do not say: the letters
    # merge, so there is no mapping from marks to characters. The position of the
    # change is known exactly though, so it is kept as a position.
    colorStops: Optional[List[Tuple[float, str]]] = None
    backgroundColor: Optional[str] = None
    opacity: float = 1.0
    # Left inset, used when an element's box was widened to take in a leading icon so
    # that the text still starts where it was measured rather than on top of the icon.
    paddingLeft: float = 0.0

class TextSpan(BaseModel):
    text: str
    bbox: List[float] = Field(default_factory=list)  # [x1, y1, x2, y2]
    style: TextStyle = Field(default_factory=TextStyle)

class BoxStyle(BaseModel):
    """CSS box painting for a reconstructed surface (a real div, not a raster)."""
    backgroundColor: Optional[str] = None
    backgroundImage: Optional[str] = None       # e.g. linear-gradient(...)
    borderColor: Optional[str] = None
    borderWidth: float = 0.0
    borderStyle: str = 'solid'
    borderRadius: Optional[str] = None          # e.g. '26px'
    boxShadow: Optional[str] = None

class DocumentElement(BaseModel):
    id: str = Field(default_factory=lambda: f'elem-{uuid.uuid4().hex[:8]}')
    type: Literal['text', 'image', 'table', 'vector', 'formula', 'custom', 'rect']
    bbox: List[float]  # [x1, y1, x2, y2] in points/pixels
    zIndex: int = 1
    rotation: float = 0.0
    opacity: float = 1.0

    # Semantic reconstruction. `tag` is the HTML element actually emitted; `role`
    # labels what the analyzer believes it is, for the editor and for debugging.
    # `confidence` records how strong the evidence was, and `fallbackSrc` keeps the
    # original pixels so a wrong promotion can always be downgraded to an image.
    tag: Optional[str] = None
    role: Optional[str] = None
    box: Optional[BoxStyle] = None
    parentId: Optional[str] = None
    confidence: Optional[float] = None
    fallbackSrc: Optional[str] = None

    # Text fields
    text: Optional[str] = None
    style: Optional[TextStyle] = None
    spans: Optional[List[TextSpan]] = None

    # Image fields
    src: Optional[str] = None
    assetName: Optional[str] = None
    naturalWidth: Optional[float] = None
    naturalHeight: Optional[float] = None

    # Table fields
    html: Optional[str] = None
    rows: Optional[int] = None
    cols: Optional[int] = None
    tableData: Optional[Dict[str, Any]] = None

    # Vector fields
    svg: Optional[str] = None

    # Formula fields
    latex: Optional[str] = None
    mathml: Optional[str] = None
    renderedHtml: Optional[str] = None

class PageData(BaseModel):
    pageNumber: int
    width: float
    height: float
    isScanned: bool = False
    elements: List[DocumentElement] = Field(default_factory=list)
    originalImageSrc: Optional[str] = None  # Base64 data URL
    backgroundColor: Optional[str] = "#ffffff"

class DocumentData(BaseModel):
    title: str = "Reconstructed Document"
    pageCount: int = 1
    pages: List[PageData] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# ==============================================================================
# 2. UTILITIES & GEOMETRY HELPERS
# ==============================================================================

def normalize_font_family(font_name: str) -> str:
    """Normalizes PDF embedded font names to standard clean CSS font families."""
    if not font_name:
        return "'Segoe UI', Helvetica, Arial, sans-serif"
    clean = re.sub(r'^[A-Z]{6}\+', '', font_name)
    clean = re.sub(r'-\d+$', '', clean)
    cl = clean.lower()

    if 'type3' in cl:
        return "'Segoe UI', Helvetica, Arial, sans-serif"
    elif any(k in cl for k in ['times', 'roman', 'minion']):
        return "'Times New Roman', Times, 'Nimbus Roman', serif"
    elif 'georgia' in cl:
        return "Georgia, 'Times New Roman', serif"
    elif any(k in cl for k in ['courier', 'mono', 'consolas']):
        return "'Courier New', Courier, Consolas, monospace"
    elif 'arial' in cl:
        return "Arial, 'Helvetica Neue', Helvetica, sans-serif"
    elif 'calibri' in cl:
        return "Calibri, Candara, 'Segoe UI', sans-serif"
    elif 'helvetica' in cl:
        return "'Helvetica Neue', Helvetica, Arial, sans-serif"
    elif 'cambria' in cl:
        return "Cambria, Georgia, serif"
    elif 'verdana' in cl:
        return "Verdana, Geneva, sans-serif"
    elif 'trebuchet' in cl:
        return "'Trebuchet MS', Helvetica, sans-serif"
    elif 'garamond' in cl:
        return "Garamond, 'Times New Roman', serif"
    elif 'tahoma' in cl:
        return "Tahoma, Verdana, sans-serif"
    elif any(k in cl for k in ['noto', 'cjk', 'sans']):
        return "'Segoe UI', 'Helvetica Neue', Arial, sans-serif"
    elif 'serif' in cl:
        return "'Times New Roman', serif"
    return f"'{clean}', 'Segoe UI', Arial, sans-serif"

def color_to_hex(color_val) -> str:
    """Converts PyMuPDF color int (sRGB) or tuple to hex string #RRGGBB."""
    if color_val is None:
        return "#000000"
    if isinstance(color_val, int):
        r = (color_val >> 16) & 0xFF
        g = (color_val >> 8) & 0xFF
        b = color_val & 0xFF
        return f"#{r:02x}{g:02x}{b:02x}"
    if isinstance(color_val, (list, tuple)):
        if len(color_val) == 1:
            val = int(round(float(color_val[0] or 0) * 255))
            return f"#{val:02x}{val:02x}{val:02x}"
        elif len(color_val) >= 3:
            r = int(round(float(color_val[0] or 0) * 255))
            g = int(round(float(color_val[1] or 0) * 255))
            b = int(round(float(color_val[2] or 0) * 255))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#000000"

def _rgb_to_hex(color_tuple) -> Optional[str]:
    if not color_tuple:
        return None
    try:
        if len(color_tuple) == 1:
            val = int(round(float(color_tuple[0] or 0) * 255))
            return f"#{val:02x}{val:02x}{val:02x}"
        elif len(color_tuple) >= 3:
            r = int(round(float(color_tuple[0] or 0) * 255))
            g = int(round(float(color_tuple[1] or 0) * 255))
            b = int(round(float(color_tuple[2] or 0) * 255))
            return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return None
    return None

def compute_bbox_iou(b1, b2) -> float:
    ix0, iy0 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix1, iy1 = min(b1[2], b2[2]), min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area1 = max((b1[2] - b1[0]) * (b1[3] - b1[1]), 0.001)
    area2 = max((b2[2] - b2[0]) * (b2[3] - b2[1]), 0.001)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

def is_bbox_inside_any(bbox, container_bboxes, tolerance=3.0) -> bool:
    bx0, by0, bx1, by1 = bbox
    b_area = max((bx1 - bx0) * (by1 - by0), 0.001)
    for cx0, cy0, cx1, cy1 in container_bboxes:
        ix0, iy0 = max(bx0, cx0 - tolerance), max(by0, cy0 - tolerance)
        ix1, iy1 = min(bx1, cx1 + tolerance), min(by1, cy1 + tolerance)
        if ix1 > ix0 and iy1 > iy0:
            if ((ix1 - ix0) * (iy1 - iy0) / b_area) > 0.65:
                return True
    return False

def is_blank_or_uniform_image(img_bytes: bytes) -> bool:
    try:
        im = Image.open(io.BytesIO(img_bytes))
        arr = np.array(im)
        if arr.size == 0 or float(arr.std()) < 0.5:
            return True
        if arr.ndim == 3 and arr.shape[2] == 4 and np.all(arr[:, :, 3] == 0):
            return True
    except Exception:
        pass
    return False

def get_image_color_score(img_bytes: bytes) -> float:
    try:
        im = Image.open(io.BytesIO(img_bytes))
        arr = np.array(im)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            return float(arr.std()) + float(arr[:, :, :3].std(axis=2).mean()) * 3.0
        return float(arr.std())
    except Exception:
        return 0.0

# Font stacks for recognized typography styles.
FONT_STACK_GEOMETRIC_SANS = "'Plus Jakarta Sans', 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"
# Georgia first, not Playfair Display. Playfair had to be fetched from Google while
# the page rendered, and it is not a face we can measure against locally, so a serif
# page was fitted with sans metrics and then drawn in a serif that might not have
# arrived. Georgia ships with Windows and macOS, Liberation Serif stands in for it on
# Linux, and both can be measured, so what is fitted is what is drawn.
FONT_STACK_EDITORIAL_SERIF = "Georgia, 'Times New Roman', 'Liberation Serif', serif"

# The typeface travels with the page.
#
# The head used to carry a Google Fonts <link>, which made a reconstruction depend on
# a network fetch finishing before anyone looked at it. Two consequences, both
# measured: an exported page opened offline rendered in whatever the system had --
# the whole point of the reconstruction is that it looks like the original, and it
# did not -- and the benchmark itself moved by two SSIM points between runs of
# identical code, because the screenshot sometimes won the race against the font.
# 4.91% of the page's pixels differ between a render that got the font and one that
# did not, which is the size of the entire error budget.
#
# It also asked for six families to use two. Plus Jakarta Sans is the one that
# matters: it heads the sans stack and it is the face every text run is measured
# against, so embedding it is what makes the rendered size equal the fitted size.
_EMBED_FACES = (
    ("Plus Jakarta Sans", "PlusJakartaSans.ttf"),
)
_font_css_cache: Optional[str] = None


def embedded_font_css() -> str:
    """@font-face rules with the shipped fonts inlined, or "" if none are present."""
    global _font_css_cache
    if _font_css_cache is not None:
        return _font_css_cache
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
    rules = []
    for family, filename in _EMBED_FACES:
        path = os.path.join(here, filename)
        try:
            with open(path, 'rb') as fh:
                b64 = base64.b64encode(fh.read()).decode('ascii')
        except OSError:
            logger.warning("Font %s not found; pages will fall back to a system face.",
                           filename)
            continue
        # A variable font covers every weight from one file, so one rule is enough.
        rules.append(
            "@font-face{font-family:'%s';src:url(data:font/ttf;base64,%s) format('truetype');"
            "font-weight:100 900;font-style:normal;font-display:block}" % (family, b64))
    _font_css_cache = "\n".join(rules)
    return _font_css_cache


# Words the recogniser ran together are left as it read them.
#
# PaddleOCR returns tightly set type as one token -- 'yourbuyersread.',
# 'Cargocontainer'. This was a table of the exact phrases in one benchmark screenshot,
# which fixed that screenshot and nothing else, and claimed a general capability the
# engine does not have.
#
# Splitting them from the pixels was tried and does not work: the gap between two
# words is several times the gap between two letters and easy to find, but turning a
# gap position into a position in the string is not, because the letters merge. That
# headline is 15 characters and 11 marks, so there is no mapping from one to the
# other, and the stand-in font's advance widths drift by more than a character over
# the length of a word. It split 'yourbuyersread.' into 'yourbuyersr ead.'.
#
# A break in the wrong place is worse than a missing one: it reads as a typo rather
# than as a limit of the recogniser. So the text is left as read, and this is listed
# as a known limitation rather than papered over for one image.

def detect_font_family(crops: List[np.ndarray]) -> str:
    """Serif or sans, from the weight distribution of the strokes.

    Serifs put ink at the top and bottom of a stem that a sans does not, so a stem
    whose ends are markedly heavier than its middle is a serif. Monospace is not
    detected: it needs even advance widths, which is a different measurement, and
    claiming it here while never returning it was worse than not claiming it.
    """
    if not crops:
        return FONT_STACK_GEOMETRIC_SANS
    serif_votes = 0
    total_samples = 0
    for crop in crops[:15]:
        if crop.shape[0] < 14 or crop.shape[1] < 14:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        h, w = binary.shape
        if h < 16:
            continue
        col_sums = np.sum(binary > 0, axis=0)
        stems = np.where(col_sums >= h * 0.55)[0]
        if len(stems) > 0:
            total_samples += 1
            top_w = np.sum(binary[:int(h*0.2), :] > 0)
            mid_w = np.sum(binary[int(h*0.4):int(h*0.6), :] > 0)
            if mid_w > 0 and (top_w / mid_w) > 1.4:
                serif_votes += 1

    if total_samples > 0 and (serif_votes / total_samples) >= 0.35:
        return FONT_STACK_EDITORIAL_SERIF
    return FONT_STACK_GEOMETRIC_SANS

# Keyed by (serif, bold). The sans entries lead with the file the page embeds, so
# the measurement and the render are the same outlines rather than two approximations
# of each other.
_MEASURE_FACES = {
    (False, False): ['PlusJakartaSans.ttf', 'Inter.ttf', 'segoeui.ttf', 'Roboto-Regular.ttf',
                     'Helvetica.ttc', 'arial.ttf', 'DejaVuSans.ttf',
                     'LiberationSans-Regular.ttf'],
    (False, True): ['PlusJakartaSans.ttf', 'Inter.ttf', 'segoeuib.ttf', 'Roboto-Bold.ttf',
                    'Helvetica-Bold.ttf', 'arialbd.ttf', 'DejaVuSans-Bold.ttf',
                    'LiberationSans-Bold.ttf'],
    (True, False): ['georgia.ttf', 'Georgia.ttf', 'times.ttf', 'Times New Roman.ttf',
                    'LiberationSerif-Regular.ttf', 'DejaVuSerif.ttf'],
    (True, True): ['georgiab.ttf', 'Georgia Bold.ttf', 'timesbd.ttf',
                   'LiberationSerif-Bold.ttf', 'DejaVuSerif-Bold.ttf'],
}
_FONT_DIRS = [
    os.path.join(os.path.dirname(__file__), 'fonts'),
    os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'),
    '/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/truetype/liberation',
    '/usr/share/fonts', '/Library/Fonts', '/System/Library/Fonts',
]
_MEASURE_REF_SIZE = 100  # metrics scale linearly, so measure once and scale
_font_cache: Dict[Tuple[bool, bool], Any] = {}

def _get_measure_font(bold: bool, serif: bool = False):
    """Loads the face this run will be rendered in, at a fixed reference size."""
    key = (serif, bold)
    if key in _font_cache:
        return _font_cache[key]
    from PIL import ImageFont
    font = None
    for face in _MEASURE_FACES[key]:
        for d in _FONT_DIRS:
            p = os.path.join(d, face)
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, _MEASURE_REF_SIZE)
                    if bold:
                        try:
                            font.set_variation_by_name('Bold')
                        except Exception:
                            pass
                    break
                except Exception:
                    continue
        if font is not None:
            break
    if font is None and serif:
        # No serif to measure with; the sans metrics are closer than nothing.
        return _get_measure_font(bold, serif=False)
    if font is None:
        logger.warning("No measurement font found; falling back to box-height heuristic.")
    _font_cache[key] = font
    return font

# The weights a variable face can be asked for, and what PIL calls each one. Type is
# not set in two weights: a page has a light label, a medium nav, a semibold card
# title and an extrabold headline, and calling all of them "bold" or "normal" loses
# most of what makes it look like itself.
WEIGHT_NAMES = ((300, 'Light'), (400, 'Regular'), (500, 'Medium'),
                (600, 'SemiBold'), (700, 'Bold'), (800, 'ExtraBold'))
_weight_font_cache: Dict[Tuple[bool, int], Any] = {}


def _weight_face(serif: bool, weight: int):
    """The measurement face set to one weight, or None if it cannot be."""
    key = (serif, weight)
    if key in _weight_font_cache:
        return _weight_font_cache[key]
    from PIL import ImageFont
    base = _get_measure_font(weight >= 600, serif)
    font = None
    path = getattr(base, 'path', None)
    if path:
        name = dict(WEIGHT_NAMES).get(weight, 'Regular')
        try:
            font = ImageFont.truetype(path, _MEASURE_REF_SIZE)
            font.set_variation_by_name(name)
        except Exception:
            # A static face cannot be asked for a weight; it is the weight it is.
            font = base if weight >= 600 else _get_measure_font(False, serif)
    _weight_font_cache[key] = font
    return font


def is_heavy(weight: Any) -> bool:
    """Whether a weight -- numeric, 'bold' or 'normal' -- is one of the heavy ones."""
    try:
        return int(str(weight)) >= 600
    except (TypeError, ValueError):
        return str(weight).lower() in ('bold', 'bolder')


def _ink_density(mask: np.ndarray) -> float:
    """Share of a mark's own bounding box that is ink. Scale-free, so it can be
    compared between a crop off the page and a render of the same string."""
    if mask is None or mask.size == 0:
        return 0.0
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return 0.0
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return float(len(ys)) / float(h * w)


# Below this ink height the source's own blur and compression are too large a share of
# the strokes for their density to mean anything, and the lettering measures heavy. At
# 14px, a screenshot's grey regular body copy measured 600 and the four nav links came
# out as three different weights; display type is where weight is both visible and
# measurable, and below it the old stroke-ratio guess is used instead.
WEIGHT_MIN_INK_HEIGHT = 24


def estimate_font_weight(text: str, ink: np.ndarray, serif: bool = False,
                         fallback: int = 400) -> int:
    """Which weight the lettering is set at, from how much of its box is ink.

    It used to be `bold if stroke_ratio > 0.28 or box_h > 32`, which is two answers to
    a question with six, and the second half of it called every headline bold whatever
    its strokes were doing -- a light 40px title came out at 700.

    The same string is rendered in the measurement face at each weight and the one
    whose ink density matches the crop's is chosen. Density rather than stroke width
    because it needs no skeleton and no assumption about which strokes are stems, and
    because it is scale-free: the render and the crop need not be the same size.
    """
    clean = (text or '').strip()
    if not clean or ink is None or ink.size == 0:
        return fallback
    if ink.shape[0] < WEIGHT_MIN_INK_HEIGHT:
        return fallback
    measured = _ink_density(ink)
    if measured <= 0.0:
        return fallback

    from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
    # Render the reference at the size the crop is, not at the reference size. A 10px
    # glyph is mostly the blur at the edge of its own strokes, so its ink density is
    # far higher than the same letter drawn at 100px -- comparing across sizes made
    # every small label come out extrabold.
    target_h = float(ink.shape[0])
    best, best_err = fallback, 1e9
    for weight, _name in WEIGHT_NAMES:
        font = _weight_face(serif, weight)
        if font is None:
            continue
        try:
            bb = font.getbbox(clean)
            ref_h = float(bb[3] - bb[1])
            if ref_h > 0 and target_h >= 4:
                size = max(6, min(200, int(round(_MEASURE_REF_SIZE * target_h / ref_h))))
                path = getattr(font, 'path', None)
                if path:
                    sized = _ImageFont.truetype(path, size)
                    try:
                        sized.set_variation_by_name(dict(WEIGHT_NAMES)[weight])
                    except Exception:
                        pass
                    font = sized
                    bb = font.getbbox(clean)
            im = _Image.new('L', (max(1, bb[2] - bb[0] + 8), max(1, bb[3] - bb[1] + 8)), 0)
            _ImageDraw.Draw(im).text((4 - bb[0], 4 - bb[1]), clean, font=font, fill=255)
        except Exception:
            continue
        ref = _ink_density((np.asarray(im) > 110).astype(np.uint8))
        if ref <= 0.0:
            continue
        err = abs(ref - measured)
        if err < best_err:
            best_err, best = err, weight
    return best


def cluster_font_sizes(runs: List[Dict[str, Any]], tol: float = 0.07) -> None:
    """Snaps independently fitted sizes onto the few sizes a real design system uses.

    Each run is fitted against its own ink box, so two lines of the same paragraph come
    out at 19.4px and 20.1px. That jitter defeats every downstream rule that groups by
    size -- paragraphs never merge and each stray value claims its own heading level.
    Collapsing near-equal sizes restores the structure and yields cleaner CSS.
    """
    if not runs:
        return
    ordered = sorted(runs, key=lambda r: r['fontSize'])
    clusters: List[List[Dict[str, Any]]] = [[ordered[0]]]
    for r in ordered[1:]:
        # Measured from the smallest member of the cluster, not the last one added. By
        # the last one, a page whose sizes step 9.2, 9.9, 10.3 ... 17.4 -- each within 7%
        # of the one before -- chains into a single cluster, and its median was imposed
        # on all of it: nav, labels and body copy all set at 13.3px, with the spacing
        # re-solved to spread the too-small text across boxes measured for 17px. What
        # was meant to merge 19.4 with 20.1 was flattening the whole type scale.
        if r['fontSize'] <= clusters[-1][0]['fontSize'] * (1.0 + tol):
            clusters[-1].append(r)
        else:
            clusters.append([r])

    for group in clusters:
        centre = float(np.median([r['fontSize'] for r in group]))
        # Display lines set at one size share one weight. Two lines of a single headline
        # measured 800 and 700, because the second is half accent colour and the lighter
        # ink reads as a lighter stroke. When they disagree the heavier reading wins: a
        # low-contrast run under-measures, and nothing over-measures at this size.
        display = [r for r in group if (r['bbox'][3] - r['bbox'][1]) >= WEIGHT_MIN_INK_HEIGHT]
        if len(display) >= 2:
            votes = Counter(str(r['fontWeight']) for r in display)
            top = max(votes.values())
            weight = max((w for w, n in votes.items() if n == top),
                         key=lambda w: int(w) if str(w).isdigit() else 400)
            for r in display:
                r['fontWeight'] = weight
                r['style'].fontWeight = weight
                for sp in (r.get('spans') or []):
                    if sp.style:
                        sp.style.fontWeight = weight
        for r in group:
            if abs(r['fontSize'] - centre) < 0.01:
                continue
            # Re-solve spacing and line-height against the snapped size so the run still
            # lands on the ink box it was measured from.
            b = r['bbox']
            _, ls, lh, ws, sx = fit_text_to_box(
                r['text'], b[2] - b[0], b[3] - b[1], is_heavy(r['fontWeight']),
                force_size=centre,
                serif=(r['style'].fontFamily == FONT_STACK_EDITORIAL_SERIF))
            r['fontSize'] = centre
            r['style'].fontSize = centre
            r['style'].letterSpacing = ls
            r['style'].lineHeight = lh
            r['style'].wordSpacing = ws
            r['style'].scaleX = sx
            if r.get('spans'):
                for sp in r['spans']:
                    if sp.style:
                        sp.style.fontSize = centre

def cluster_text_colors(runs: List[Dict[str, Any]], tol: float = 26.0) -> None:
    """Snaps per-run sampled colours onto shared values.

    Colour is measured from the glyph pixels of each run, so the same ink comes back as
    #4a4a4a on one line and #4b494b on the next. Exact-match grouping then fails, and
    the CSS carries dozens of near-identical colours instead of a palette.
    """
    centres: List[Tuple[np.ndarray, List[Any]]] = []
    targets: List[Tuple[str, Any]] = []
    for r in sorted(runs, key=lambda r: -(r['bbox'][2] - r['bbox'][0])):
        targets.append((r['color'], r))
        if r.get('spans'):
            for sp in r['spans']:
                if sp.style and sp.style.color:
                    targets.append((sp.style.color, sp.style))

    for c_hex, obj in targets:
        c = (c_hex or '').lstrip('#')
        try:
            rgb = np.array([int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)], dtype=np.float64)
        except Exception:
            continue
        for centre, members in centres:
            if float(np.linalg.norm(rgb - centre)) <= tol:
                members.append(obj)
                break
        else:
            centres.append((rgb, [obj]))

    for centre, members in centres:
        hexed = f"#{int(centre[0]):02x}{int(centre[1]):02x}{int(centre[2]):02x}"
        for obj in members:
            if isinstance(obj, dict):
                obj['color'] = hexed
                if 'style' in obj and obj['style']:
                    obj['style'].color = hexed
            elif hasattr(obj, 'color'):
                obj.color = hexed

def fit_text_to_box(text: str, box_w: float, box_h: float, bold: bool,
                    force_size: Optional[float] = None,
                    serif: bool = False) -> Tuple[float, float, float, float, float]:
    """Fits a text run to an OCR ink box, returning (fontSize, letterSpacing, lineHeight).

    The OCR box bounds *ink*, not the em square, so its height depends on which glyphs
    the run happens to contain -- 'Product' (cap to baseline) and 'Pricing' (cap to
    descender) are different heights at the same font size. Measuring the actual string
    removes that dependence:

      * fontSize    scales the reference ink height onto the box height, so glyphs come
                    out the size they look in the image.
      * letterSpacing absorbs the residual width error, so the run still spans the box
                    even when the real face is wider or narrower than our stand-in.
      * lineHeight  is solved from the CSS inline box model so the ink top lands exactly
                    on the box top, instead of floating on an arbitrary 1.2 multiplier.
    """
    clean = (text or '').strip()
    font = _get_measure_font(bold, serif) if clean else None
    if font is None:
        return max(9.0, round(box_h * 0.82, 1)), 0.0, 1.2, 0.0, 1.0

    try:
        ink = font.getbbox(clean)
        ascent, descent = font.getmetrics()
    except Exception:
        return max(9.0, round(box_h * 0.82, 1)), 0.0, 1.2, 0.0, 1.0

    ink_h = float(ink[3] - ink[1])
    ink_w = float(ink[2] - ink[0])
    if ink_h <= 0:
        return max(9.0, round(box_h * 0.82, 1)), 0.0, 1.2, 0.0, 1.0

    if force_size is not None and force_size > 0:
        font_size = float(force_size)
        scale = font_size / float(_MEASURE_REF_SIZE)
    else:
        scale = float(box_h) / ink_h
        font_size = max(6.0, _MEASURE_REF_SIZE * scale)

    # Spread (or pull in) the residual width across the gaps between glyphs. CSS adds
    # letter-spacing after every character, but the ink of the run ends before the last
    # one, so the gaps that matter number len-1.
    # Display type is often set with wide gaps between words rather than wide tracking.
    # Pushing that slack into letter-spacing pulls the words together into one string --
    # a headline reading "THE ART" came back as "THEART" -- so when a run has spaces the
    # residual goes into word-spacing first, where the original put it, and only what is
    # left over is spread between letters.
    # Both limits are what type is actually set at. Tracking runs from about -0.05em on
    # a tight display face to 0.2em on a spaced-out eyebrow; an extra word gap beyond
    # half an em is not a thing anyone designs. They were 0.06em and 2.2em, which is
    # backwards -- far too tight to express the tracking on "LINKEDIN CREATOR CAMPAIGNS
    # FOR BRANDS", and loose enough to put 35px between the words of a 57px headline
    # and 22px between the words of a 10px label. Anything these cannot absorb goes to
    # the glyph width below, where being wrong is least visible.
    WORD_SPACE_LIMIT = 0.6
    TRACKING_LIMIT = 0.20

    residual = float(box_w) - ink_w * scale
    words = clean.count(' ')
    word_spacing = 0.0
    if words and residual > font_size * 0.06:
        word_spacing = round(min(residual / words, font_size * WORD_SPACE_LIMIT), 2)
        residual -= word_spacing * words

    gaps = max(len(clean) - 1, 1)
    letter_spacing = residual / gaps
    limit = font_size * TRACKING_LIMIT
    letter_spacing = max(-limit, min(limit, letter_spacing))

    # CSS centres the (ascent + descent) content box inside the line box, so the ink top
    # sits at  top + (L - ascent - descent)/2 + ink_offset.  Setting that equal to the
    # element top and solving for L gives the line-height that pins ink to the box.
    line_px = (ascent + descent - 2.0 * ink[1]) * scale
    line_height = max(0.1, line_px / font_size)

    # Whatever spacing could not absorb, take out of the glyph width. A display face
    # is often far narrower than the stand-in measured against it -- a headline fitted
    # by cap height came out 110px wider than its own box and ran into the next word --
    # and the spacing clamp exists precisely so it cannot paper over that. Condensing
    # keeps the run inside the box it was measured from, which is the geometry that
    # matters; the glyph proportions are already approximate without the real font.
    achieved = ink_w * scale + word_spacing * words + letter_spacing * gaps
    scale_x = (float(box_w) / achieved) if achieved > 1.0 else 1.0
    scale_x = float(min(max(scale_x, 0.45), 1.8))
    if abs(scale_x - 1.0) < 0.02:
        scale_x = 1.0

    return (round(font_size, 2), round(letter_spacing, 2), round(line_height, 3),
            word_spacing, round(scale_x, 4))

REFERENCE_WIDTH = 1400.0    # the capture width these pixel thresholds were calibrated on

def page_scale(w: int, h: int) -> float:
    """How much larger this capture is than the one the pixel thresholds assume.

    Every threshold expressed in pixels -- a tile size, a minimum component area, how
    far a shadow reaches, how wide an antialiased rim is -- describes a physical feature
    of a rendered page, and all of them grow with the capture. Left absolute they only
    hold near one resolution: the same page captured at 2x would have its components
    rejected as noise by an area floor four times too small, and its texture measured in
    tiles covering a quarter of the ground they were meant to. Scaling them keeps the
    engine reading the page rather than the screenshot's dimensions.
    """
    return float(np.clip(max(w, h) / REFERENCE_WIDTH, 0.4, 4.0))

def scaled_kernel(scale: float, base: int = 3) -> np.ndarray:
    size = max(3, int(round(base * scale)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

TEXT_COLOUR_TOL = 96.0      # how far an antialiased glyph pixel may sit from its own ink

# How far towards the strongest ink pixel a pixel has to be to count as the core of a
# stroke rather than its antialiased edge.
INK_CORE_SHARE = 0.7


def dominant_ink_colour(crop: np.ndarray, candidate: np.ndarray) -> Optional[np.ndarray]:
    """The most common colour among candidate glyph pixels, as BGR.

    A median is pulled off the mark when the candidate set is a mixture -- headline type
    set over artwork picks up whatever is behind it -- and lands on a colour that is in
    neither the glyphs nor the background. The modal quantised colour survives that,
    because the glyphs are the one thing in the set that shares a single colour.
    """
    mask = candidate > 0
    if int(mask.sum()) < 8:
        return None
    # Only the core of the strokes. The edge of a letter is a blend of the ink and what
    # it sits on, and a thin word is mostly edge: "in", "once" and "set the" came out
    # two shades lighter than the words beside them in the same paragraph, and each
    # was given its own grey. The pixels furthest from the background are the ink.
    border = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0)
    dist = np.linalg.norm(crop.astype(np.float32)
                          - np.median(border, axis=0).astype(np.float32), axis=2)
    core = mask & (dist >= float(dist[mask].max()) * INK_CORE_SHARE)
    px = crop[core] if int(core.sum()) >= 12 else crop[mask]
    if px.size < 24:
        return None
    q = (px.astype(np.int32) // 16 * 16)
    packed = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    best = vals[int(np.argmax(counts))]
    bucket = packed == best
    if int(bucket.sum()) < 12:
        return None
    return np.median(px[bucket].reshape(-1, 3), axis=0)

# A band has to hold this much of the run's width, and its colour differ this far from
# its neighbour, before the lettering counts as having changed colour.
BAND_MIN_SHARE = 0.12
BAND_COLOUR_GAP = 70.0
BAND_RUN = 5                    # consecutive columns of the new colour before believing it
# ...and the change has to be a change of colour, not of shade. Text darkening as it
# crosses a photograph, or antialiasing on a thin stroke, moves the grey about without
# ever picking up a colour; half a headline set in an accent does.
BAND_NEUTRAL = 20               # max-min across the channels: below this it is a grey
BAND_COLOURED = 45              # ...and above this it is definitely not
BAND_HUE_SHIFT = 25.0           # degrees, when both sides carry a colour


def _chroma(c: np.ndarray) -> float:
    return float(int(c.max()) - int(c.min()))


def _is_a_colour_change(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two ink colours differ in kind, rather than only in lightness.

    Neutral against neutral is a shade of the same thing however far apart the two are
    -- #1d1d1d and #4f4f4f are both simply grey. Neutral against a colour is the case
    this exists for: black lettering that turns into an accent. Two colours have to
    differ in hue, which is what separates a headline in two colours from one
    travelling over a photograph that changes underneath it.
    """
    ca, cb = _chroma(a), _chroma(b)
    if ca < BAND_NEUTRAL and cb < BAND_NEUTRAL:
        return False
    if min(ca, cb) < BAND_NEUTRAL:
        return max(ca, cb) >= BAND_COLOURED
    pair = np.array([[a, b]], dtype=np.uint8)
    hsv = cv2.cvtColor(pair, cv2.COLOR_BGR2HSV)[0].astype(float)
    dh = abs(hsv[0][0] - hsv[1][0]) * 2.0        # OpenCV packs hue into 0-179
    return min(dh, 360.0 - dh) >= BAND_HUE_SHIFT


def ink_colour_bands(crop: np.ndarray, candidate: np.ndarray, diff: np.ndarray
                     ) -> Optional[List[Tuple[float, str]]]:
    """Where along a run the lettering changes colour, or None if it does not.

    Read column by column: the ink in each column has a colour, and a run set in one
    colour gives the same answer all the way across. Where the answer changes and
    stays changed for a while, the lettering changed colour.

    The position is returned as a share of the width rather than a character index,
    because the pixels know the first and not the second: the letters merge, so there
    is no mapping from marks to characters -- this headline is 15 characters and 11
    marks. The renderer can place a colour at a position without being told which
    letter it starts on.

    Only the firm middle of each stroke is sampled. The edge of a letter is a blend
    into whatever it sits on, and reading colour there invents a band at the end of
    every word.
    """
    h, w = candidate.shape[:2]
    if w < 24 or h < 6:
        return None
    # Firmness is judged within each column, not across the run. A single threshold is
    # set by whichever colour contrasts most with the ground -- black on white here --
    # and throws away the other one entirely, which is the case this exists for.
    cols: List[Optional[np.ndarray]] = []
    for x in range(w):
        m = candidate[:, x] > 0
        if int(m.sum()) < 3:
            cols.append(None)
            continue
        d = diff[:, x][m]
        core = m.copy()
        core[m] = d >= float(d.max()) * 0.75
        if int(core.sum()) < 2:
            core = m
        cols.append(np.median(crop[core, x], axis=0))

    bands: List[Tuple[int, int, np.ndarray]] = []
    cur: List[np.ndarray] = []
    cur_start = 0
    pending: List[Tuple[int, np.ndarray]] = []
    for x, c in enumerate(cols):
        if c is None:
            continue
        if not cur:
            cur, cur_start = [c], x
            continue
        median = np.median(np.asarray(cur), axis=0)
        if float(np.linalg.norm(c - median)) <= BAND_COLOUR_GAP:
            cur.append(c)
            pending = []
            continue
        # Different. Believe it only once it has held for a few columns, so a single
        # odd column inside a letter does not start a band.
        pending.append((x, c))
        if len(pending) >= BAND_RUN:
            bands.append((cur_start, pending[0][0], np.median(np.asarray(cur), axis=0)))
            cur = [c for _, c in pending]
            cur_start = pending[0][0]
            pending = []
    if cur:
        bands.append((cur_start, w, np.median(np.asarray(cur), axis=0)))

    wide = [b for b in bands if (b[1] - b[0]) >= max(8.0, w * BAND_MIN_SHARE)]
    if len(wide) < 2:
        return None
    # Neighbours that ended up the same colour are one band after all.
    merged: List[Tuple[int, int, np.ndarray]] = []
    for b in wide:
        if merged and float(np.linalg.norm(b[2] - merged[-1][2])) <= BAND_COLOUR_GAP:
            merged[-1] = (merged[-1][0], b[1], (merged[-1][2] + b[2]) / 2.0)
        else:
            merged.append(b)
    if len(merged) < 2:
        return None

    # Every neighbouring pair has to be a real change of colour, or this is one run of
    # text crossing something that changed underneath it.
    for i in range(len(merged) - 1):
        if not _is_a_colour_change(merged[i][2].astype(np.uint8),
                                   merged[i + 1][2].astype(np.uint8)):
            return None

    out: List[Tuple[float, str]] = []
    for i, b in enumerate(merged):
        start = 0.0 if i == 0 else max(0.0, min(1.0, b[0] / float(w)))
        out.append((round(start, 4), _bgr_to_hex(b[2].astype(int))))
    return out


def colour_selective_ink(crop: np.ndarray, candidate: np.ndarray,
                         colour: Optional[np.ndarray]) -> np.ndarray:
    """Narrows candidate ink to the pixels that are actually the text's own colour.

    Measuring ink as 'anything unlike the local background' is right on a plain ground
    and wrong over artwork: a headline crossing a gilt picture frame takes the frame
    into its ink box, which inflates the fitted size and pushes the run into its
    neighbour. Selecting by the glyph colour instead leaves the artwork behind.
    """
    if colour is None:
        return candidate
    dist = np.linalg.norm(crop.astype(np.float32) - colour.astype(np.float32), axis=2)
    selective = (dist <= TEXT_COLOUR_TOL).astype(np.uint8)
    # Only trust it when it still explains most of the glyph; a wrong colour estimate
    # would otherwise erase the run entirely.
    if np.count_nonzero(selective) < max(0.25 * np.count_nonzero(candidate), 12):
        return candidate
    return selective

def keep_core_ink(ink: np.ndarray) -> np.ndarray:
    """Drops ink blobs that never reach this line's own body band.

    Tightly stacked headlines overlap, so a detection box routinely catches the
    descenders of the line above. Measuring those as part of this line inflates its ink
    height and oversizes the font. A real descender hangs off a letter whose body sits
    in the middle of the box; a stray one from another line does not.
    """
    h = ink.shape[0]
    if h < 8:
        return ink
    lo, hi = int(h * 0.30), int(np.ceil(h * 0.70))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    if n <= 1:
        return ink
    keep = np.zeros_like(ink)
    for i in range(1, n):
        top = stats[i, cv2.CC_STAT_TOP]
        bottom = top + stats[i, cv2.CC_STAT_HEIGHT]
        if top < hi and bottom > lo:
            keep[lab == i] = 1
    return keep if keep.any() else ink

def erase_text_from_crop(crop: np.ndarray, text_mask: np.ndarray,
                         textured: bool = False, scale: float = 1.0) -> np.ndarray:
    """Paints out masked text pixels so a raster crop can sit underneath live text.

    Which method is right depends on what the text sits on. Over a flat fill, taking
    the median of the ring immediately around each glyph restores a button's solid
    colour exactly, where inpainting would smear a gradient across it. Over a
    photograph that same fill lands as a visible flat patch, because the surroundings
    are not one colour -- which is what left ghost blocks behind the headline copy on a
    photo-backed page. There, inpainting is the correct tool: it propagates the
    surrounding texture into the hole instead of averaging it away.

    The mask is also grown before erasing. A glyph's antialiased rim sits below the
    threshold that detected its core, and leaving that rim behind outlines every letter
    that was supposedly removed.
    """
    if crop.size == 0 or not np.any(text_mask):
        return crop

    out = crop.copy()
    rim = max(1, int(round(2 * scale)))
    mask = cv2.dilate((text_mask > 0).astype(np.uint8), _K3, iterations=rim)

    if textured:
        return cv2.inpaint(out, mask * 255, max(3, int(round(4 * scale))), cv2.INPAINT_NS)

    num, labels = cv2.connectedComponents(mask)

    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    unresolved = np.zeros(mask.shape, np.uint8)

    for label in range(1, num):
        region = (labels == label).astype(np.uint8)
        ring = cv2.subtract(cv2.dilate(region, ring_kernel), region)
        ring[mask > 0] = 0  # never sample from other text
        ring_px = out[ring > 0]
        if ring_px.size == 0:
            unresolved[region > 0] = 255
            continue
        out[region > 0] = np.median(ring_px, axis=0).astype(out.dtype)

    if np.any(unresolved):
        out = cv2.inpaint(out, unresolved, 3, cv2.INPAINT_TELEA)
    return out

# ==============================================================================
# 3. FORMULA HANDLER
# ==============================================================================

MATH_SYMBOLS = [
    '∑', '∫', '∏', '√', 'α', 'β', 'γ', 'δ', 'ε', 'θ', 'λ', 'μ', 'π', 'σ', 'ω',
    'Δ', '∂', '∞', '±', '∓', '×', '÷', '≈', '≠', '≤', '≥', '∈', '∉', '⊂', '⊆',
    '→', '⇒', '↔', '⇔', '∀', '∃', '∇', '∫', '∬', '∭', '∮'
]

class FormulaHandler:
    @staticmethod
    def is_likely_formula(text: str) -> bool:
        if not text or len(text.strip()) == 0:
            return False
        clean = text.strip()
        if sum(1 for sym in MATH_SYMBOLS if sym in clean) >= 1 and any(op in clean for op in ['=', '+', '-', '/', '^']):
            return True
        if re.search(r'\b[A-Za-z]\s*=\s*[-+]?[0-9A-Za-z]', clean) and any(op in clean for op in ['^', '√', '²', '³', '∑', '∫', '±', '/', '\\']):
            return True
        if any(cmd in clean for cmd in ['\\frac', '\\sqrt', '\\sum', '\\int', '\\partial']):
            return True
        return False

    @staticmethod
    def convert_to_mathml(text: str) -> str:
        safe_text = html.escape(text.strip())
        safe_text = safe_text.replace('²', '<msup><mi></mi><mn>2</mn></msup>')
        safe_text = safe_text.replace('³', '<msup><mi></mi><mn>3</mn></msup>')
        return f'<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" class="rendered-math"><mrow><mtext>{safe_text}</mtext></mrow></math>'

    @staticmethod
    def create_formula_element(bbox, text: str, latex: Optional[str] = None) -> Dict[str, Any]:
        return {
            'type': 'formula',
            'bbox': bbox,
            'text': text,
            'latex': latex or text,
            'mathml': FormulaHandler.convert_to_mathml(text),
            'renderedHtml': f'<div class="formula-block">{FormulaHandler.convert_to_mathml(text)}</div>'
        }

# ==============================================================================
# 4. PDF ANALYZER
# ==============================================================================

class PDFAnalyzer:
    @staticmethod
    def analyze_document(file_path: str) -> Dict[str, Any]:
        doc = pymupdf.open(file_path)
        if doc.is_encrypted:
            raise ValueError("The PDF document is password-protected or encrypted. Please provide an unlocked PDF.")

        pages_info = []
        total_text_len = 0

        for page_num in range(len(doc)):
            page = doc[page_num]
            rect = page.rect
            width, height = float(rect.width), float(rect.height)
            text = page.get_text()
            text_length = len(text.strip())
            total_text_len += text_length

            images = page.get_images()
            drawings = page.get_drawings()

            is_scanned = False
            if text_length < 25 and len(images) > 0:
                is_scanned = True
            elif len(images) > 0:
                page_area = width * height
                for img_info in images:
                    for r in page.get_image_rects(img_info[0]):
                        if page_area > 0 and (r.width * r.height / page_area) > 0.80 and text_length < 100:
                            is_scanned = True
                            break
                    if is_scanned:
                        break

            pages_info.append({
                "page": page_num + 1,
                "width": width,
                "height": height,
                "isScanned": is_scanned,
                "textLength": text_length,
                "imageCount": len(images),
                "drawingCount": len(drawings)
            })

        scanned_count = sum(1 for p in pages_info if p["isScanned"])
        doc_type = "scanned" if scanned_count == len(doc) else ("hybrid" if scanned_count > 0 else "digital")

        return {
            "pageCount": len(doc),
            "documentType": doc_type,
            "totalTextLength": total_text_len,
            "pages": pages_info
        }

# ==============================================================================
# 5. VECTOR EXTRACTOR
# ==============================================================================

class VectorExtractor:
    @staticmethod
    def extract_vectors(page, table_bboxes: List[List[float]] = None, text_bboxes: List[List[float]] = None) -> List[Dict[str, Any]]:
        table_bboxes = table_bboxes or []
        text_bboxes = text_bboxes or []
        try:
            drawings = page.get_drawings()
        except Exception:
            return []
        if not drawings:
            return []

        page_w, page_h = float(page.rect.width), float(page.rect.height)
        page_area = page_w * page_h

        def is_inside_table(r):
            for tb in table_bboxes:
                if (float(r.x0) >= tb[0] - 2 and float(r.y0) >= tb[1] - 2 and
                    float(r.x1) <= tb[2] + 2 and float(r.y1) <= tb[3] + 2):
                    return True
            return False

        def is_text_outline(d):
            if not text_bboxes:
                return False
            r = d.get('rect')
            if not r or len(d.get('items', [])) < 20:
                return False
            rx0, ry0, rx1, ry1 = float(r.x0), float(r.y0), float(r.x1), float(r.y1)
            r_area = max((rx1 - rx0) * (ry1 - ry0), 0.001)
            for tb in text_bboxes:
                ix0, iy0 = max(rx0, tb[0] - 3.0), max(ry0, tb[1] - 3.0)
                ix1, iy1 = min(rx1, tb[2] + 3.0), min(ry1, tb[3] + 3.0)
                if ix1 > ix0 and iy1 > iy0 and ((ix1 - ix0) * (iy1 - iy0) / r_area) > 0.65:
                    return True
            return False

        valid_drawings = []
        for d in drawings:
            r = d.get('rect')
            if not r:
                continue
            if page_area > 0 and (r.width * r.height) / page_area >= 0.92:
                continue  # Skip full-page background boxes
            if is_inside_table(r) or is_text_outline(d):
                continue
            valid_drawings.append(d)

        if not valid_drawings:
            return []

        # Cluster adjacent drawings within 10px
        clusters: List[List[Dict[str, Any]]] = []
        for d in valid_drawings:
            r = d['rect']
            if r.width < 0.5 and r.height < 0.5:
                continue
            merged = False
            for cluster in clusters:
                cx0 = min(item['rect'].x0 for item in cluster)
                cy0 = min(item['rect'].y0 for item in cluster)
                cx1 = max(item['rect'].x1 for item in cluster)
                cy1 = max(item['rect'].y1 for item in cluster)
                if not (r.x1 < cx0 - 10 or r.x0 > cx1 + 10 or r.y1 < cy0 - 10 or r.y0 > cy1 + 10):
                    cluster.append(d)
                    merged = True
                    break
            if not merged:
                clusters.append([d])

        results = []
        for idx, cluster in enumerate(clusters):
            min_x = min(float(d['rect'].x0) for d in cluster)
            min_y = min(float(d['rect'].y0) for d in cluster)
            max_x = max(float(d['rect'].x1) for d in cluster)
            max_y = max(float(d['rect'].y1) for d in cluster)
            w = max(float(max_x - min_x), 1.0)
            h = max(float(max_y - min_y), 1.0)

            svg_elements = []
            for d in cluster:
                stroke_color = _rgb_to_hex(d.get('color')) or 'none'
                fill_color = _rgb_to_hex(d.get('fill')) or 'none'
                stroke_width = float(d.get('width') or 1.0)
                stroke_opacity = float(d.get('stroke_opacity') or 1.0)
                fill_opacity = float(d.get('opacity') or 1.0)
                fill_rule = 'evenodd' if d.get('even_odd', False) else 'nonzero'
                close_path = d.get('closePath', False)

                path_cmds = []
                current_pt = None
                for item in d.get('items', []):
                    cmd = item[0]
                    if cmd == 'l':
                        p1, p2 = item[1], item[2]
                        x1, y1 = float(p1.x) - min_x, float(p1.y) - min_y
                        x2, y2 = float(p2.x) - min_x, float(p2.y) - min_y
                        if current_pt is None or abs(current_pt[0] - x1) > 0.05 or abs(current_pt[1] - y1) > 0.05:
                            path_cmds.append(f'M {x1:.2f} {y1:.2f}')
                        path_cmds.append(f'L {x2:.2f} {y2:.2f}')
                        current_pt = (x2, y2)
                    elif cmd == 're':
                        r = item[1]
                        rx, ry = float(r.x0) - min_x, float(r.y0) - min_y
                        rw, rh = max(float(r.width), 0.5), max(float(r.height), 0.5)
                        path_cmds.append(f'M {rx:.2f} {ry:.2f} h {rw:.2f} v {rh:.2f} h {-rw:.2f} Z')
                        current_pt = None
                    elif cmd == 'c':
                        p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                        x1, y1 = float(p1.x) - min_x, float(p1.y) - min_y
                        x2, y2 = float(p2.x) - min_x, float(p2.y) - min_y
                        x3, y3 = float(p3.x) - min_x, float(p3.y) - min_y
                        x4, y4 = float(p4.x) - min_x, float(p4.y) - min_y
                        if current_pt is None or abs(current_pt[0] - x1) > 0.05 or abs(current_pt[1] - y1) > 0.05:
                            path_cmds.append(f'M {x1:.2f} {y1:.2f}')
                        path_cmds.append(f'C {x2:.2f} {y2:.2f}, {x3:.2f} {y3:.2f}, {x4:.2f} {y4:.2f}')
                        current_pt = (x4, y4)

                if path_cmds:
                    if close_path:
                        path_cmds.append('Z')
                    eff_stroke = stroke_color if (stroke_color != 'none' or fill_color != 'none') else '#000000'
                    svg_elements.append(
                        f'<path d="{" ".join(path_cmds)}" fill="{fill_color}" stroke="{eff_stroke}" '
                        f'stroke-width="{stroke_width:.2f}" fill-opacity="{fill_opacity:.2f}" '
                        f'stroke-opacity="{stroke_opacity:.2f}" fill-rule="{fill_rule}" />'
                    )

            if svg_elements:
                results.append({
                    'id': f'vector-{idx+1}',
                    'type': 'vector',
                    'bbox': [round(min_x, 2), round(min_y, 2), round(min_x + w, 2), round(min_y + h, 2)],
                    'svg': (
                        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.2f} {h:.2f}" '
                        f'width="100%" height="100%" preserveAspectRatio="none" style="display:block; overflow:visible;">\n'
                        f'  {"\n  ".join(svg_elements)}\n</svg>'
                    )
                })
        return results

# ==============================================================================
# 6. TABLE EXTRACTOR
# ==============================================================================

class TableExtractor:
    @staticmethod
    def is_valid_digital_table(tab, data: List[List[Any]], page_w: float, page_h: float, all_tabs: List[Any] = None) -> bool:
        if not data:
            return False
        tb = tab.bbox
        t_w, t_h = max(tb[2] - tb[0], 1.0), max(tb[3] - tb[1], 1.0)
        area_ratio = (t_w * t_h) / max(page_w * page_h, 1.0)

        # 1. Reject 1-row tables: they are layout containers, column headers, or cards
        if tab.row_count <= 1:
            return False

        # 2. Reject outer layout containers enclosing an inner table
        if all_tabs:
            for other in all_tabs:
                if other is tab or other.row_count <= 1:
                    continue
                ob = other.bbox
                if (ob[0] >= tb[0] - 2 and ob[1] >= tb[1] - 2 and
                    ob[2] <= tb[2] + 2 and ob[3] <= tb[3] + 2):
                    if (t_w * t_h) > (ob[2] - ob[0]) * (ob[3] - ob[1]) * 1.3:
                        if tab.row_count <= 3:
                            return False

        # 3. Minimum non-empty cells
        non_empty = sum(1 for row in data for cell in row if cell and str(cell).strip() != '')
        if non_empty < 2:
            return False

        # 4. Multi-column article check (2 rows with large text blocks)
        if tab.row_count == 2 and tab.col_count <= 2:
            cell_texts = [str(c).strip() for row in data for c in row if c and str(c).strip()]
            long_cells = sum(1 for t in cell_texts if len(t) > 180 or t.count('\n') >= 4)
            if long_cells >= 2:
                return False

        # 5. Full page false positive
        if tb[0] <= 10.0 and tb[1] <= 10.0 and tb[2] >= page_w - 10.0 and tb[3] >= page_h - 10.0:
            if tab.row_count < 4 or tab.col_count < 2:
                return False

        # 6. Area ratio >= 0.50 with sparse cells
        if area_ratio >= 0.50:
            if tab.row_count < 3 or tab.col_count < 2:
                return False
            total_cells = tab.row_count * tab.col_count
            if total_cells > 0 and (non_empty / total_cells) < 0.35:
                return False

        return True

    @staticmethod
    def clean_table_data(data: List[List[Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """Cleans table data extracted by find_tables():
        - Eliminates phantom ghost columns with 0 non-empty cells
        - Distinguishes primary tabular columns from boundary columns
        - Identifies merged full-width rows (e.g. callout notes) with colspan
        """
        if not data:
            return [], 0
        num_cols = max(len(r) for r in data)
        col_counts = [
            sum(1 for r in data if c < len(r) and r[c] is not None and str(r[c]).strip() != '')
            for c in range(num_cols)
        ]
        non_zero_cols = [c for c in range(num_cols) if col_counts[c] > 0]
        if not non_zero_cols:
            return [], 0

        max_count = max(col_counts)
        if max_count >= 3:
            primary_cols = [c for c in non_zero_cols if col_counts[c] >= 2]
        else:
            primary_cols = non_zero_cols

        if not primary_cols:
            primary_cols = non_zero_cols

        cleaned_rows = []
        for r in data:
            row_vals = [r[c] if c < len(r) else None for c in primary_cols]
            # Check if this row only had text in a non-primary column (e.g. full-width callout note)
            if all(v is None or str(v).strip() == '' for v in row_vals):
                other_vals = [r[c] for c in range(len(r)) if r[c] is not None and str(r[c]).strip() != '']
                if other_vals:
                    cleaned_rows.append({'is_merged': True, 'text': str(other_vals[0]).strip()})
            else:
                non_empty_in_row = [v for v in row_vals if v is not None and str(v).strip() != '']
                if len(non_empty_in_row) == 1 and ('\n' in str(non_empty_in_row[0]) or len(str(non_empty_in_row[0])) > 40):
                    cleaned_rows.append({'is_merged': True, 'text': str(non_empty_in_row[0]).strip()})
                else:
                    cleaned_rows.append({'is_merged': False, 'cells': row_vals})

        return cleaned_rows, len(primary_cols)

    @staticmethod
    def extract_digital_tables(page) -> List[Dict[str, Any]]:
        results = []
        page_w, page_h = float(page.rect.width), float(page.rect.height)
        try:
            tabs = page.find_tables()
            if not tabs or not hasattr(tabs, 'tables'):
                return results

            all_tabs = list(tabs.tables)
            drawings = page.get_drawings()

            # First filter valid tables
            valid_tabs = [tab for tab in all_tabs if TableExtractor.is_valid_digital_table(tab, tab.extract(), page_w, page_h, all_tabs)]

            for idx, tab in enumerate(valid_tabs):
                raw_data = tab.extract()

                # Step 1: Detect and trim callout note rows (e.g. EXPECTED APPLICATION BEHAVIOUR)
                valid_row_indices = []
                for r_idx in range(tab.row_count):
                    row_vals = raw_data[r_idx] if r_idx < len(raw_data) else []
                    row_text = ' '.join(str(c) for c in row_vals if c and str(c).strip() != '').strip()
                    if 'EXPECTED APPLICATION BEHAVIOUR' in row_text or row_text.startswith('NOTE:') or row_text.startswith('SOURCE:'):
                        continue
                    valid_row_indices.append(r_idx)

                if len(valid_row_indices) < 2:
                    continue

                # Step 2: Determine active columns across valid rows (eliminate ghost columns)
                active_cols = []
                for c_idx in range(tab.col_count):
                    if any(
                        raw_data[r_idx][c_idx] is not None and str(raw_data[r_idx][c_idx]).strip() != ''
                        for r_idx in valid_row_indices
                        if c_idx < len(raw_data[r_idx])
                    ):
                        active_cols.append(c_idx)

                if not active_cols:
                    continue

                # Step 3: Compute exact table bbox from active cells
                cell_boxes = []
                for r_idx in valid_row_indices:
                    r_cells = tab.rows[r_idx].cells
                    for c_idx in active_cols:
                        if c_idx < len(r_cells) and r_cells[c_idx] is not None:
                            cell_boxes.append(r_cells[c_idx])

                if not cell_boxes:
                    continue

                t_x0 = min(b[0] for b in cell_boxes)
                t_y0 = min(b[1] for b in cell_boxes)
                t_x1 = max(b[2] for b in cell_boxes)
                t_y1 = max(b[3] for b in cell_boxes)
                table_w = max(t_x1 - t_x0, 1.0)

                # Step 4: Compute active column widths & percentages
                col_widths = []
                for c_idx in active_cols:
                    col_x0_list = [
                        tab.rows[r_idx].cells[c_idx][0]
                        for r_idx in valid_row_indices
                        if c_idx < len(tab.rows[r_idx].cells) and tab.rows[r_idx].cells[c_idx] is not None
                    ]
                    col_x1_list = [
                        tab.rows[r_idx].cells[c_idx][2]
                        for r_idx in valid_row_indices
                        if c_idx < len(tab.rows[r_idx].cells) and tab.rows[r_idx].cells[c_idx] is not None
                    ]
                    cw = (max(col_x1_list) - min(col_x0_list)) if col_x0_list and col_x1_list else (table_w / len(active_cols))
                    col_widths.append(cw)

                total_cw = sum(col_widths) or table_w
                col_pcts = [(cw / total_cw) * 100.0 for cw in col_widths]

                # Step 5: Compute row heights from active cells
                row_heights = []
                for r_idx in valid_row_indices:
                    cell_heights = [
                        tab.rows[r_idx].cells[c][3] - tab.rows[r_idx].cells[c][1]
                        for c in active_cols
                        if c < len(tab.rows[r_idx].cells) and tab.rows[r_idx].cells[c] is not None
                    ]
                    if cell_heights:
                        rh = min(cell_heights)
                    else:
                        rh = tab.rows[r_idx].bbox[3] - tab.rows[r_idx].bbox[1]
                    row_heights.append(max(rh, 12.0))

                # Step 6: Helper for cell background fills from drawings
                def get_cell_fill(c_rect):
                    for d in drawings:
                        if d.get('fill') and d['rect'].intersects(c_rect):
                            inter = d['rect'] & c_rect
                            if (inter.width * inter.height) >= (c_rect.width * c_rect.height) * 0.40:
                                return color_to_hex(d['fill'])
                    return None

                # Step 7: Helper to check if any other valid table is nested inside this cell
                def cell_contains_other_table(c_rect):
                    for other in valid_tabs:
                        if other is tab:
                            continue
                        ob = pymupdf.Rect(other.bbox)
                        if c_rect.contains(ob) or (c_rect.intersects(ob) and (c_rect & ob).get_area() >= ob.get_area() * 0.85):
                            return True
                    return False

                # Step 8: Extract cell content with high fidelity
                def extract_cell_content(c_rect, raw_val):
                    if cell_contains_other_table(c_rect):
                        return '&nbsp;', 7.5, 'left', False

                    td = page.get_text('dict', clip=c_rect, flags=pymupdf.TEXT_PRESERVE_WHITESPACE | pymupdf.TEXT_PRESERVE_LIGATURES)
                    lines_data = []
                    all_sizes = []
                    for b in td.get('blocks', []):
                        if b.get('type') == 0:
                            for l in b.get('lines', []):
                                spans = []
                                for s in l.get('spans', []):
                                    stext = s.get('text', '')
                                    if not stext:
                                        continue
                                    sz = float(round(s.get('size', 8.0), 1))
                                    all_sizes.append(sz)
                                    sfont = s.get('font', '')
                                    sflags = s.get('flags', 0)
                                    is_bold = bool(sflags & 16) or any(k in sfont.lower() for k in ['bold', 'black', 'heavy'])
                                    is_italic = bool(sflags & 2) or any(k in sfont.lower() for k in ['italic', 'oblique'])
                                    scolor = color_to_hex(s.get('color'))
                                    spans.append({
                                        'text': stext,
                                        'size': sz,
                                        'bold': is_bold,
                                        'italic': is_italic,
                                        'color': scolor
                                    })
                                if spans:
                                    lines_data.append((l.get('bbox', [0, 0, 0, 0]), spans))

                    if not lines_data:
                        if raw_val is not None and str(raw_val).strip() != '':
                            return html.escape(str(raw_val).strip()), 7.5, 'left', False
                        return '&nbsp;', 7.5, 'left', False

                    dom_size = (sum(all_sizes) / len(all_sizes)) if all_sizes else 7.5

                    # Alignment
                    l_bbox = lines_data[0][0]
                    mid_diff = abs((l_bbox[0] + l_bbox[2]) / 2.0 - (c_rect.x0 + c_rect.x1) / 2.0)
                    r_diff = abs(l_bbox[2] - c_rect.x1)
                    l_diff = abs(l_bbox[0] - c_rect.x0)
                    if mid_diff < 8.0:
                        align = 'center'
                    elif r_diff < 12.0 and l_diff > 15.0:
                        align = 'right'
                    else:
                        align = 'left'

                    line_htmls = []
                    has_heading = False
                    first_line_all_bold = all(s['bold'] for s in lines_data[0][1])
                    if len(lines_data) > 1 and first_line_all_bold:
                        has_heading = True

                    for l_idx, (l_bbox, spans) in enumerate(lines_data):
                        line_str = ''
                        for s in spans:
                            esc = html.escape(s['text'])
                            if has_heading and l_idx == 0:
                                line_str += esc
                            else:
                                inner = esc
                                if s['bold']:
                                    inner = f'<b>{inner}</b>'
                                if s['italic']:
                                    inner = f'<i>{inner}</i>'
                                if s['color'] and s['color'] not in ['#000000', '#111827', '#000']:
                                    inner = f'<span style="color: {s["color"]};">{inner}</span>'
                                line_str += inner
                        line_htmls.append(line_str)

                    if has_heading:
                        heading_html = f'<div style="font-weight: 700; margin-bottom: 2px;">{line_htmls[0]}</div>'
                        body_html = f'<div>{"<br>".join(line_htmls[1:])}</div>'
                        return heading_html + body_html, dom_size, align, True
                    else:
                        is_all_bold = all(s['bold'] for _, spans in lines_data for s in spans)
                        full_text = '<br>'.join(line_htmls)
                        return full_text, dom_size, align, is_all_bold

                # Step 9: Build HTML
                html_parts = [
                    '<table class="reconstructed-table" style="width: 100%; height: 100%; '
                    'table-layout: fixed; border-collapse: collapse; box-sizing: border-box; '
                    'font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Arial, sans-serif; '
                    'color: #000000;">'
                ]
                html_parts.append('  <colgroup>')
                for pct in col_pcts:
                    html_parts.append(f'    <col style="width: {pct:.2f}%;">')
                html_parts.append('  </colgroup>')
                html_parts.append('  <tbody>')

                for row_pos, r_idx in enumerate(valid_row_indices):
                    r_height = row_heights[row_pos]
                    r_cells = tab.rows[r_idx].cells
                    r_vals = raw_data[r_idx]
                    html_parts.append(f'    <tr style="height: {r_height:.2f}px;">')

                    for col_pos, c_idx in enumerate(active_cols):
                        cell_box = r_cells[c_idx] if c_idx < len(r_cells) else None
                        raw_val = r_vals[c_idx] if c_idx < len(r_vals) else None

                        if cell_box:
                            c_rect = pymupdf.Rect(cell_box)
                            cell_bg = get_cell_fill(c_rect)
                            content_html, font_sz, align, is_bold = extract_cell_content(c_rect, raw_val)
                        else:
                            cell_bg = None
                            content_html, font_sz, align, is_bold = html.escape(str(raw_val or '')), 7.5, 'left', False

                        bg_style = f'background-color: {cell_bg}; ' if cell_bg else 'background-color: #ffffff; '
                        bold_style = 'font-weight: 700; ' if is_bold else 'font-weight: normal; '
                        valign = 'middle' if ('<br>' not in content_html and '<div' not in content_html) else 'top'

                        html_parts.append(
                            f'      <td style="border: 1px solid #333333; padding: 2px 6px; '
                            f'text-align: {align}; vertical-align: {valign}; font-size: {font_sz:.1f}pt; '
                            f'line-height: 1.25; {bold_style}{bg_style}color: #000000; '
                            f'box-sizing: border-box; overflow: hidden; word-break: break-word;">{content_html}</td>'
                        )
                    html_parts.append('    </tr>')
                html_parts.append('  </tbody></table>')

                # Determine zIndex: if nested inside another table, zIndex=9, else 8
                is_nested = any(other is not tab and pymupdf.Rect(other.bbox).contains(pymupdf.Rect(t_x0, t_y0, t_x1, t_y1)) for other in valid_tabs)
                t_z_index = 9 if is_nested else 8

                results.append({
                    'id': f'table-{idx+1}',
                    'type': 'table',
                    'bbox': [float(round(t_x0, 2)), float(round(t_y0, 2)), float(round(t_x1, 2)), float(round(t_y1, 2))],
                    'rows': len(valid_row_indices),
                    'cols': len(active_cols),
                    'html': '\n'.join(html_parts),
                    'data': raw_data,
                    'zIndex': t_z_index
                })
        except Exception as e:
            logger.warning(f"Table extraction error: {e}")
        return results

# ==============================================================================
# 7. DIGITAL EXTRACTOR
# ==============================================================================

class DigitalExtractor:
    @staticmethod
    def extract_page(doc: pymupdf.Document, page_num: int, asset_dir: Optional[str] = None) -> PageData:
        page = doc[page_num]
        rect = page.rect
        page_width, page_height = float(rect.width), float(rect.height)
        page_area = max(page_width * page_height, 1.0)

        # High-res render for original image & fidelity check
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        orig_img_base64 = f"data:image/png;base64,{base64.b64encode(img_bytes).decode('utf-8')}"

        elements: List[DocumentElement] = []

        # 1. Extract tables
        raw_tables = TableExtractor.extract_digital_tables(page)
        table_bboxes = [t['bbox'] for t in raw_tables]
        for t in raw_tables:
            elements.append(DocumentElement(
                id=t['id'],
                type='table',
                bbox=t['bbox'],
                rows=t['rows'],
                cols=t['cols'],
                html=t['html'],
                tableData={'rows': t.get('data', [])},
                zIndex=t.get('zIndex', 8)
            ))

        # 2. Extract line-level text with exact 2D positioning and typography
        flags = pymupdf.TEXT_PRESERVE_IMAGES | pymupdf.TEXT_DEHYPHENATE | pymupdf.TEXT_PRESERVE_LIGATURES | pymupdf.TEXT_PRESERVE_WHITESPACE
        blocks = page.get_text("dict", flags=flags).get("blocks", [])

        text_counter = 1
        formula_counter = 1
        text_block_bboxes: List[List[float]] = []

        for block in blocks:
            if block.get("type") == 0:
                block_bbox = [float(round(v, 2)) for v in block.get("bbox", [0, 0, 0, 0])]
                if is_bbox_inside_any(block_bbox, table_bboxes):
                    continue

                text_block_bboxes.append(block_bbox)
                for line in block.get("lines", []):
                    line_bbox = [float(round(v, 2)) for v in line.get("bbox", [0, 0, 0, 0])]
                    if is_bbox_inside_any(line_bbox, table_bboxes):
                        continue

                    line_spans = line.get("spans", [])
                    line_text = "".join([s.get("text", "") for s in line_spans]).strip()
                    if not line_text:
                        continue

                    spans_data: List[TextSpan] = []
                    font_sizes, font_families, colors, weights, styles = [], [], [], [], []

                    for span in line_spans:
                        stext = span.get("text", "")
                        if not stext:
                            continue
                        s_font = span.get("font", "")
                        raw_size = span.get("size")
                        s_size = float(round(raw_size, 2)) if raw_size is not None else 12.0
                        s_flags = span.get("flags", 0)
                        s_color = color_to_hex(span.get("color"))
                        s_bbox = [float(round(v, 2)) for v in span.get("bbox", [0, 0, 0, 0])]

                        is_bold = bool(s_flags & 16) or any(k in s_font.lower() for k in ['bold', 'black', 'heavy']) or ('type3' in s_font.lower() and s_size >= 18.0)
                        is_italic = bool(s_flags & 2) or any(k in s_font.lower() for k in ['italic', 'oblique'])

                        weight_str = 'bold' if is_bold else 'normal'
                        style_str = 'italic' if is_italic else 'normal'
                        clean_font = normalize_font_family(s_font)

                        font_sizes.append(s_size)
                        font_families.append(clean_font)
                        colors.append(s_color)
                        weights.append(weight_str)
                        styles.append(style_str)

                        spans_data.append(TextSpan(
                            text=stext,
                            bbox=s_bbox,
                            style=TextStyle(
                                fontFamily=clean_font,
                                fontSize=s_size,
                                fontWeight=weight_str,
                                fontStyle=style_str,
                                color=s_color,
                                lineHeight=1.0
                            )
                        ))

                    if not spans_data:
                        continue

                    dom_font = max(set(font_families), key=font_families.count) if font_families else "'Segoe UI', Arial, sans-serif"
                    dom_size = max(font_sizes) if font_sizes else 12.0
                    dom_color = max(set(colors), key=colors.count) if colors else "#000000"
                    dom_weight = 'bold' if 'bold' in weights else 'normal'
                    dom_italic = 'italic' if 'italic' in styles else 'normal'

                    line_mid = (line_bbox[0] + line_bbox[2]) / 2.0
                    align = 'left'
                    if abs(line_mid - (page_width / 2.0)) < 25.0 and line_bbox[0] > 40.0 and (line_bbox[2] - line_bbox[0]) < page_width * 0.7:
                        align = 'center'
                    elif line_bbox[2] > page_width - 50.0 and line_bbox[0] > page_width * 0.5:
                        align = 'right'

                    if FormulaHandler.is_likely_formula(line_text):
                        f_elem = FormulaHandler.create_formula_element(line_bbox, line_text)
                        elements.append(DocumentElement(
                            id=f"formula-{formula_counter}",
                            type='formula',
                            bbox=f_elem['bbox'],
                            text=f_elem['text'],
                            latex=f_elem['latex'],
                            mathml=f_elem['mathml'],
                            renderedHtml=f_elem['renderedHtml'],
                            zIndex=15
                        ))
                        formula_counter += 1
                    else:
                        elements.append(DocumentElement(
                            id=f"text-{text_counter}",
                            type='text',
                            bbox=line_bbox,
                            text=line_text,
                            spans=spans_data,
                            style=TextStyle(
                                fontFamily=dom_font,
                                fontSize=dom_size,
                                fontWeight=dom_weight,
                                fontStyle=dom_italic,
                                color=dom_color,
                                textAlign=align,
                                lineHeight=1.0
                            ),
                            zIndex=10
                        ))
                        text_counter += 1

        # 3. Extract vectors
        for v in VectorExtractor.extract_vectors(page, table_bboxes, text_block_bboxes):
            elements.append(DocumentElement(
                id=v['id'],
                type='vector',
                bbox=v['bbox'],
                svg=v['svg'],
                zIndex=5
            ))

        # 4. Extract images with transparency & deduplication
        candidate_images = []
        for block in blocks:
            if block.get("type") == 1:
                bbox = [float(round(v, 2)) for v in block.get("bbox", [0, 0, 0, 0])]
                raw_bytes = block.get("image")
                mask_bytes = block.get("mask")
                ext = block.get("ext", "png")
                if raw_bytes:
                    final_bytes = raw_bytes
                    final_ext = ext
                    if mask_bytes:
                        try:
                            pix_img = pymupdf.Pixmap(raw_bytes)
                            pix_mask = pymupdf.Pixmap(mask_bytes)
                            final_bytes = pymupdf.Pixmap(pix_img, pix_mask).tobytes("png")
                            final_ext = "png"
                        except Exception:
                            pass
                    candidate_images.append({
                        "bbox": bbox,
                        "bytes": final_bytes,
                        "ext": final_ext,
                        "naturalWidth": float(block.get("width", bbox[2] - bbox[0])),
                        "naturalHeight": float(block.get("height", bbox[3] - bbox[1]))
                    })

        for img_info in page.get_images():
            xref = img_info[0]
            for r in page.get_image_rects(xref):
                bbox = [float(round(r.x0, 2)), float(round(r.y0, 2)), float(round(r.x1, 2)), float(round(r.y1, 2))]
                if any(compute_bbox_iou(bbox, c["bbox"]) > 0.85 for c in candidate_images):
                    continue
                try:
                    base_img = doc.extract_image(xref)
                    if base_img:
                        raw_bytes = base_img["image"]
                        ext = base_img.get("ext", "png")
                        smask_xref = base_img.get("smask")
                        final_bytes = raw_bytes
                        final_ext = ext
                        if smask_xref:
                            try:
                                pix_img = pymupdf.Pixmap(doc, xref)
                                pix_mask = pymupdf.Pixmap(doc, smask_xref)
                                final_bytes = pymupdf.Pixmap(pix_img, pix_mask).tobytes("png")
                                final_ext = "png"
                            except Exception:
                                pass
                        candidate_images.append({
                            "bbox": bbox,
                            "bytes": final_bytes,
                            "ext": final_ext,
                            "naturalWidth": float(base_img.get("width", r.width)),
                            "naturalHeight": float(base_img.get("height", r.height))
                        })
                except Exception:
                    pass

        # Deduplicate & filter images
        non_blank = [img for img in candidate_images if not is_blank_or_uniform_image(img["bytes"])]
        deduped = []
        for img in non_blank:
            merged = False
            for group in deduped:
                if compute_bbox_iou(img["bbox"], group[0]["bbox"]) > 0.80:
                    group.append(img)
                    merged = True
                    break
            if not merged:
                deduped.append([img])

        img_counter = 1
        for group in deduped:
            img = max(group, key=lambda x: get_image_color_score(x["bytes"])) if len(group) > 1 else group[0]
            bbox, ext, b_bytes = img["bbox"], img["ext"], img["bytes"]
            b64 = base64.b64encode(b_bytes).decode('utf-8')
            src = f"data:image/{ext};base64,{b64}"
            asset_name = f"page_{page_num + 1}_img_{img_counter}.{ext}"

            if asset_dir:
                os.makedirs(asset_dir, exist_ok=True)
                with open(os.path.join(asset_dir, asset_name), "wb") as f:
                    f.write(b_bytes)

            img_area = max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)
            z_idx = 1 if (img_area / page_area) >= 0.40 else 10

            elements.append(DocumentElement(
                id=f"image-{img_counter}",
                type='image',
                bbox=bbox,
                src=src,
                assetName=asset_name,
                naturalWidth=img["naturalWidth"],
                naturalHeight=img["naturalHeight"],
                zIndex=z_idx
            ))
            img_counter += 1

        return PageData(
            pageNumber=page_num + 1,
            width=page_width,
            height=page_height,
            isScanned=False,
            elements=elements,
            originalImageSrc=orig_img_base64
        )

# ==============================================================================
# 7a. CONTENT-TYPE SEGMENTATION  (what kind of thing is this region?)
# ==============================================================================

PHOTO_TILE = 24
PHOTO_MIN_COLOURS = 20      # distinct quantised colours in a tile before it reads as texture
PHOTO_MIN_STD = 9.0         # luminance spread within a tile
PHOTO_MIN_REGION = 0.010    # fraction of the page a photographic region must cover
GRADIENT_MIN_DRIFT = 1.2    # a flat fill does not drift between neighbouring tiles
GRADIENT_MAX_DRIFT = 16.0   # beyond this it is a boundary, not a ramp

def photographic_mask(img: np.ndarray, text_ink: Optional[np.ndarray] = None,
                      scale: float = 1.0) -> np.ndarray:
    """Marks the areas of an image that are photographic or heavily textured.

    Everything downstream assumes a page is built from flat fills, and on a flat design
    that holds. On a photograph it fails catastrophically and silently: colour
    segmentation returns hundreds of organic blobs, hole-filling squares each one off
    into a convincing rounded rectangle, and because every slab is painted with its own
    region's median colour the result matches the source closely enough per pixel that
    the residual pass finds nothing left to rasterise. A forest comes back as 274 flat
    slabs and no image at all.

    The missing question is not whether a particular candidate looks like a rectangle --
    it is whether this part of the page is interface in the first place. Texture answers
    it: interface is flat and its edges are axis-aligned, photographs are neither.

    Glyphs are excluded before measuring, since text is high-variance everywhere and
    would otherwise drag whole paragraphs into the photographic class.
    """
    h, w = img.shape[:2]
    tile = max(8, int(round(PHOTO_TILE * scale)))
    quant = (cv2.medianBlur(img, 3).astype(np.int32) // 8 * 8)
    packed = (quant[:, :, 0] << 16) | (quant[:, :, 1] << 8) | quant[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    th, tw = (h + tile - 1) // tile, (w + tile - 1) // tile
    flags = np.zeros((th, tw), np.uint8)
    means = np.zeros((th, tw, 3), np.float32)
    texty = np.zeros((th, tw), bool)
    for ty in range(th):
        y0, y1 = ty * tile, min(h, (ty + 1) * tile)
        for tx in range(tw):
            x0, x1 = tx * tile, min(w, (tx + 1) * tile)
            means[ty, tx] = img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)
            if text_ink is not None and float(np.count_nonzero(text_ink[y0:y1, x0:x1])) \
                    > 0.22 * (y1 - y0) * (x1 - x0):
                texty[ty, tx] = True
                continue                        # a tile of type, not of texture
            if len(np.unique(packed[y0:y1, x0:x1])) >= PHOTO_MIN_COLOURS \
                    and float(gray[y0:y1, x0:x1].std()) >= PHOTO_MIN_STD:
                flags[ty, tx] = 255

    # Smooth photographic areas -- fog, sky, a soft background blur -- carry almost
    # no local texture and would pass as flat fill. What separates them from a real
    # fill is that a fill is piecewise constant: neighbouring tiles either match it
    # exactly or sit across a hard boundary. A gradient drifts, by a little,
    # everywhere.
    if th >= 3 and tw >= 3:
        drift = np.zeros((th, tw), np.float32)
        for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            shifted = np.roll(np.roll(means, dy, axis=0), dx, axis=1)
            drift = np.maximum(drift, np.abs(means - shifted).max(axis=2))
        ramp = ((drift >= GRADIENT_MIN_DRIFT) & (drift <= GRADIENT_MAX_DRIFT)
                & (~texty))
        flags[ramp] = 255

    raw = cv2.resize(flags, (w, h), interpolation=cv2.INTER_NEAREST)
    # Join neighbouring textured tiles; a photograph is continuous, stray tiles are not.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (tile * 2 + 1, tile * 2 + 1))
    joined = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, k)

    # Closing reaches across whatever lies between two textured tiles, so a dense panel
    # of interface -- a browser mockup, a card of controls -- gets absorbed along with
    # the artwork beside it, and every component inside is then vetoed as photographic.
    # Flat is flat whatever it borders: a fill has zero local variance, while even a
    # smooth photograph keeps a little everywhere. Carving those pixels back out keeps
    # the mask on the artwork without needing the texture thresholds retuned per page.
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(g, -1, (5, 5), normalize=True)
    sq = cv2.boxFilter(g * g, -1, (5, 5), normalize=True)
    flat = ((np.clip(sq - mean * mean, 0.0, None) < FLAT_VARIANCE).astype(np.uint8)) * 255
    # Erode first so the flat side of a photograph's own edge is not carved away with it.
    flat = cv2.erode(flat, _K3, iterations=max(1, int(round(2 * scale))))
    joined = cv2.bitwise_and(joined, cv2.bitwise_not(flat))
    joined = cv2.morphologyEx(joined, cv2.MORPH_OPEN, scaled_kernel(scale, 5))
    # Two masks with different jobs. `raw` vetoes component detection and must stay
    # tight, because a flat control sitting on a photograph is still a control and the
    # joined mask would swallow it. `joined` bounds the area kept as pixels.
    return raw, joined

FLAT_VARIANCE = 0.05        # a CSS fill is bit-identical; a photograph never quite is

def photographic_regions(mask: np.ndarray, page_area: float,
                         min_frac: float = PHOTO_MIN_REGION) -> List[Tuple[int, int, int, int]]:
    """Connected photographic areas large enough to be worth keeping as pixels.

    The mask has already decided what is photographic; this only discards the specks.
    A region smaller than a favicon is noise in the mask rather than a picture, and
    keeping it as pixels costs an asset for something nobody would call an image.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    out = []
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < page_area * min_frac or cw < 24 or ch < 24:
            continue
        out.append((int(x), int(y), int(cw), int(ch)))
    return out

# ==============================================================================
# 7b. SURFACE ANALYSIS  (flat fills -> real CSS boxes)
# ==============================================================================

MIN_SURFACE_AREA = 140
_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Stacking is derived from containment depth, never assigned globally. A flat z for
# artwork floats every raster above every reconstructed component, which is what put
# decorative pixels on top of real buttons and swallowed their clicks.
Z_SURFACE = 2      # a surface sits at its own depth
# Containment is expressed by DOM nesting, so z only orders siblings by layer.
# Scaling it by depth as well double-counted the hierarchy and let any nested
# surface paint over the text of a shallower one -- a page lost every headline
# to six empty panels sitting one level deeper than they were.
Z_ART = 3          # artwork rides just above the surface that contains it
Z_CONTROL = 4      # a promoted control outranks decoration that merely overlaps it
Z_BACKDROP = 1     # photographic ground sits just above whatever surface holds it
Z_TABLE = 5
Z_TEXT = 6         # text is the top layer within its band


# A page is reconstructed by painting candidate surfaces and then covering the parts
# CSS got wrong with the original pixels. Nothing in that process ever asks whether a
# surface still shows once everything above it has been drawn -- and measured across
# the reference pages, most of them do not: 252 of one page's 261 rectangles sat
# entirely under a later element. They cost markup, bytes and meaning while changing
# no pixel, so they are removed here rather than shipped.
CULL_VISIBLE_FRACTION = 0.005   # survives as less than this share of its own footprint
CULL_VISIBLE_PIXELS = 32        # ...and as fewer than this many pixels: a rim, not a box

# Only anonymous decoration is ever removed. A scored component is a claim about what
# the page means, and a buried one is evidence that something above it is wrong -- most
# often a raster crop that should have been trimmed around it. Deleting the component
# would destroy the finding and leave the raster in place: culling the card because a
# screenshot of the card sits on top of it loses on both counts.
CULLABLE_ROLES = frozenset({'surface', 'shape', 'artwork', 'backdrop', 'decoration'})


def _painted_mask(elem: 'DocumentElement') -> Optional[np.ndarray]:
    """What this element's CSS box actually covers, as a crop-local mask.

    Rounded corners and ellipses matter: a pill-shaped button does not hide the
    corners of the card behind it, and treating its box as solid would cull them.
    """
    x0, y0, x1, y1 = [int(round(float(v))) for v in elem.bbox]
    w, h = max(1, x1 - x0), max(1, y1 - y0)
    box = elem.box
    radius = 0
    if box is not None and box.borderRadius:
        r = str(box.borderRadius).strip()
        if r.endswith('%'):
            try:
                radius = int(min(w, h) * float(r[:-1]) / 100.0)
            except ValueError:
                radius = 0
        else:
            try:
                radius = int(round(float(r.rstrip('px'))))
            except ValueError:
                radius = 0
    radius = max(0, min(radius, min(w, h) // 2))
    m = np.zeros((h, w), np.uint8)
    if radius <= 0:
        m[:] = 1
        return m
    cv2.rectangle(m, (radius, 0), (w - radius - 1, h - 1), 1, -1)
    cv2.rectangle(m, (0, radius), (w - 1, h - radius - 1), 1, -1)
    for cx, cy in ((radius, radius), (w - radius - 1, radius),
                   (radius, h - radius - 1), (w - radius - 1, h - radius - 1)):
        cv2.circle(m, (cx, cy), radius, 1, -1)
    return m


def cull_occluded(elements: List['DocumentElement'], w: int, h: int,
                  masks: Dict[str, np.ndarray], scale: float = 1.0) -> List['DocumentElement']:
    """Removes elements that no longer show once the page is fully painted.

    Walks the paint order from the top down, accumulating the pixels already claimed.
    An element whose own footprint is entirely spoken for by what sits above it cannot
    affect the render, so it goes. Live text and tables are never removed and never
    counted as coverage -- glyphs are thin, and treating a text box as solid would
    hide whatever it was written on.

    `masks` supplies the true painted shape for elements whose footprint is not their
    box: a transparent artwork crop covers only its own ink.
    """
    if not elements:
        return elements
    covered = np.zeros((h, w), np.uint8)
    min_px = max(1.0, CULL_VISIBLE_PIXELS * scale * scale)
    dropped: set = set()

    for elem in reversed(elements):          # topmost first
        if elem.type in ('text', 'table') or elem.html:
            continue                         # content, not decoration: always kept
        cullable = (elem.role or 'surface') in CULLABLE_ROLES and elem.text is None
        x0, y0 = int(round(float(elem.bbox[0]))), int(round(float(elem.bbox[1])))
        m = masks.get(elem.id)
        if m is None:
            m = _painted_mask(elem)
        if m is None or m.size == 0:
            continue
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x0 + m.shape[1]), min(h, y0 + m.shape[0])
        if cx1 <= cx0 or cy1 <= cy0:
            if cullable:
                dropped.add(elem.id)         # entirely off-page
            continue
        local = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        own = int(np.count_nonzero(local))
        if own == 0:
            if cullable:
                dropped.add(elem.id)
            continue
        seen = covered[cy0:cy1, cx0:cx1]
        visible = int(np.count_nonzero(local & (seen == 0)))
        if cullable and visible < min_px and visible < own * CULL_VISIBLE_FRACTION:
            dropped.add(elem.id)
            continue
        # Partially visible elements still hide what is beneath the part that shows.
        if float(elem.opacity if elem.opacity is not None else 1.0) > 0.99:
            np.maximum(seen, local, out=seen)

    if not dropped:
        return elements

    # Culling a container would orphan its contents, so adopt them upward instead.
    parent_of = {e.id: getattr(e, 'parentId', None) for e in elements}
    kept = []
    for e in elements:
        if e.id in dropped:
            continue
        pid = parent_of.get(e.id)
        while pid is not None and pid in dropped:
            pid = parent_of.get(pid)
        e.parentId = pid
        kept.append(e)
    return kept


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Returns `mask` with interior holes closed (border-connected background removed)."""
    h, w = mask.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    inv = (mask == 0).astype(np.uint8) * 255
    cv2.floodFill(inv, ff, (0, 0), 0)
    return ((mask > 0) | (inv > 0)).astype(np.uint8)

class Surface:
    """A flat-filled region recovered from the image, with the CSS needed to redraw it.

    `shape` is one of 'rect' (paintable as a div with border-radius), 'ellipse'
    (paintable as an SVG ellipse) or 'complex' (must stay pixels).
    """
    __slots__ = ('bbox', 'vis_bbox', 'mask', 'color', 'shape', 'radius', 'border_color',
                 'border_width', 'area', 'fill_ratio', 'children', 'parent', 'texts', 'shadow')

    def __init__(self, bbox, mask, color, shape, radius, area, fill_ratio,
                 border_color=None, border_width=0.0, vis_bbox=None):
        self.bbox = bbox                    # (x0, y0, x1, y1); may extend past the page
        # Where the mask actually lives. For a shape clipped by the page edge the
        # rendered box is the full shape, but only this part of it was observed.
        self.vis_bbox = vis_bbox or bbox
        self.mask = mask                    # component mask, crop-local
        self.color = color
        self.shape = shape
        self.radius = radius
        self.area = area
        self.fill_ratio = fill_ratio
        self.border_color = border_color
        self.border_width = border_width
        self.children: List['Surface'] = []
        self.parent: Optional['Surface'] = None
        self.texts: List[Dict[str, Any]] = []
        self.shadow: Optional[Tuple[float, float, float, float]] = None  # dx, dy, blur, alpha

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

def _bgr_to_hex(bgr) -> str:
    return f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"

def _edge_straightness(filled: np.ndarray, radius: float) -> float:
    """How straight the sides are, ignoring the corner arcs the radius accounts for.

    Fill ratio alone cannot separate a rounded rectangle from a pill, a clipped
    illustration or a blob of similar bulk. Straight sides can.
    """
    ch, cw = filled.shape
    scores = []
    pad = int(max(radius, 1.0))
    if ch - 2 * pad >= 3:
        band = filled[pad:ch - pad, :]
        left = np.argmax(band, axis=1).astype(np.float32)
        right = (cw - np.argmax(band[:, ::-1], axis=1)).astype(np.float32)
        rows_with_ink = band.any(axis=1)
        if rows_with_ink.sum() >= 3:
            scores.append(float(np.std(left[rows_with_ink])))
            scores.append(float(np.std(right[rows_with_ink])))
    if cw - 2 * pad >= 3:
        band = filled[:, pad:cw - pad]
        top = np.argmax(band, axis=0).astype(np.float32)
        bottom = (ch - np.argmax(band[::-1, :], axis=0)).astype(np.float32)
        cols_with_ink = band.any(axis=0)
        if cols_with_ink.sum() >= 3:
            scores.append(float(np.std(top[cols_with_ink])))
            scores.append(float(np.std(bottom[cols_with_ink])))
    if not scores:
        return 0.0
    # 0 px deviation -> 1.0; 3 px or worse -> 0.0
    return float(max(0.0, 1.0 - (sum(scores) / len(scores)) / 3.0))

def _boundary_is_real(img, x, y, cw, ch, filled) -> float:
    """Fraction of a candidate's outline that sits on an actual edge in the source.

    Hole-filling turns any blob into a plausible rectangle, so a band of a gradient
    comes back with straight sides and symmetric corners just like a card does. The
    difference is that a card has a visible boundary all the way round, while the
    band's rectangle is invented -- most of its outline cuts through smooth pixels.
    """
    if min(cw, ch) < 6:
        return 1.0
    # The edge of a container is the step between it and whatever surrounds it, which
    # lies just outside its own bounding box. Measuring the gradient on the box alone
    # put the outline exactly on the border of the window, where Sobel replicates and
    # sees nothing -- so a full-page panel with a hard edge all the way round scored
    # zero and was demoted to pixels. The patch is padded so the step is inside it.
    ih, iw = img.shape[:2]
    pad = 3
    ex0, ey0 = max(0, x - pad), max(0, y - pad)
    ex1, ey1 = min(iw, x + cw + pad), min(ih, y + ch + pad)
    patch = img[ey0:ey1, ex0:ex1]
    if patch.size == 0:
        return 1.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    strong = cv2.dilate((mag > 22.0).astype(np.uint8), _K3, iterations=1)

    placed = np.zeros(patch.shape[:2], np.uint8)
    placed[y - ey0:y - ey0 + ch, x - ex0:x - ex0 + cw] = filled
    band = cv2.subtract(cv2.dilate(placed, _K3, iterations=1), cv2.erode(placed, _K3, iterations=1))
    total = int(band.sum())
    if total < 12:
        return 1.0
    return float((band & strong).sum()) / total

def _rect_evidence(filled, cw, ch, f_area, box_area, region_px) -> Tuple[float, float]:
    """Scores 'is a rounded rectangle' from several independent signals.

    Returns (score, radius). No single signal is trusted on its own -- a circle has a
    plausible fill ratio, an irregular illustration can have symmetric corners, and a
    clipped graphic can have straight edges on two sides.
    """
    radius = _corner_radius(cw, ch, f_area, box_area)
    score = 0.0

    # 1. Does the area actually match a rounded rect of the solved radius?
    predicted = box_area - (4.0 - np.pi) * radius * radius
    if box_area > 0 and abs(predicted - f_area) / box_area < 0.04:
        score += 0.30

    # 2. Straight sides outside the corner arcs.
    score += 0.25 * _edge_straightness(filled, radius)

    # 3. All four corners give up the same area.
    if _corners_symmetric(filled):
        score += 0.20

    # 4. Polygonal approximation should resolve to a small number of vertices.
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(c, True)
        if peri > 0:
            verts = len(cv2.approxPolyDP(c, 0.02 * peri, True))
            if 4 <= verts <= 10:
                score += 0.15
            elif verts > 20:
                score -= 0.10

    # 5. A CSS fill is one colour by definition.
    if region_px.size > 0 and float(region_px.reshape(-1, 3).std(axis=0).max()) < 16.0:
        score += 0.10

    return score, radius


# The residual pass exists for artwork CSS cannot describe. It also catches every fill
# the segmenter failed to find, and ships it as a photograph of a flat colour: 29% of
# the rasters across the reference pages are a single colour, 20% of the logistics page
# among them. That is worse than wasteful. A residual crop is opaque only where the
# prediction was wrong, so a missed white card becomes a white PNG with a hole at every
# glyph -- and the surface painted the wrong colour underneath shows through each hole
# as a grey shadow around the text. Recovering the fill removes the raster and the
# shadows together.
RESIDUAL_FLAT_SPREAD = 3.0      # mean channel deviation a fill has to stay under
RESIDUAL_FLAT_COVERAGE = 0.90   # over this much of its own box, once holes are closed
RESIDUAL_FLAT_STRAY = 0.02      # ...and explaining all but this share of the rest


def flat_residual_fill(crop: np.ndarray, mask: np.ndarray,
                       cw: int, ch: int) -> Optional[Tuple[str, float]]:
    """(colour, radius) when a residual region is really one flat box, else None.

    Deliberately strict on both counts. Uniform colour alone would promote a slice of
    sky; rectangularity alone would promote a mascot on a plain background. Something
    has to be both before its pixels are thrown away for a hex value.
    """
    px = crop[mask > 0]
    if px.size < 1200:
        return None
    px = px.reshape(-1, 3).astype(np.float32)
    med = np.median(px, axis=0)
    if float(np.abs(px - med).mean()) > RESIDUAL_FLAT_SPREAD:
        return None
    filled = _fill_holes((mask > 0).astype(np.uint8))
    f_area = int(filled.sum())
    box_area = float(cw * ch)
    if box_area <= 0 or f_area / box_area < RESIDUAL_FLAT_COVERAGE:
        return None

    # The fill is painted over the whole box, but the mask only says the prediction was
    # wrong *inside* it. Everywhere else the prediction was already right, and painting
    # over that is how promoting one card cost 1.6 points of the logistics page: the
    # box also spanned icons and a strip of the page behind it. So the colour has to
    # explain the pixels the mask never claimed, or the region is a card with things on
    # it rather than a flat card, and its pixels are the honest answer.
    outside = crop[(mask == 0)]
    if outside.size:
        stray = np.abs(outside.reshape(-1, 3).astype(np.float32) - med).mean(axis=1)
        if float((stray > 24.0).mean()) > RESIDUAL_FLAT_STRAY:
            return None

    # Rectangularity decides the corner radius here, not whether to promote at all. A
    # residual mask is shaped by where the prediction was wrong, which has no reason to
    # follow the real outline, so the radius solver reads the missing area as enormous
    # corners -- 139px of them on a 710x396 card. One colour over nine tenths of a box
    # is a fill whatever its rim looks like; the corners are the uncertain part, so an
    # implausible radius is dropped rather than allowed to reshape the box.
    rect_score, radius = _rect_evidence(filled, cw, ch, f_area, box_area, px)
    if rect_score < 0.45 or radius > 0.25 * min(cw, ch):
        radius = 0.0
    return _bgr_to_hex(med), radius


def _classify_region(img, x, y, cw, ch, comp, area, text_ink=None) -> Optional[Surface]:
    """Decides whether a connected flat region can be redrawn as CSS, or must stay pixels."""
    filled = _fill_holes(comp)
    f_area = int(filled.sum())
    box_area = float(cw * ch)
    if box_area <= 0 or f_area <= 0:
        return None

    fill_ratio = f_area / box_area
    solid_ratio = area / float(f_area)      # well under 1 means this is an outline, not a fill
    region_px = img[y:y+ch, x:x+cw][comp > 0]
    if region_px.size == 0:
        return None
    color = _bgr_to_hex(np.median(region_px.reshape(-1, 3), axis=0))
    bbox = (float(x), float(y), float(x + cw), float(y + ch))

    # A thin ring around a hole is a border, not a fill: record it as one so the
    # promoter can hand it to whatever surface sits inside. Kept deliberately narrow --
    # gradients and artwork also segment into annular bands, and calling one of those a
    # border paints a huge stroked box across the page.
    if solid_ratio < 0.55 and fill_ratio > 0.80:
        ring_thickness = (f_area - area) / max(2.0 * (cw + ch), 1.0)
        rect_score, radius = _rect_evidence(filled, cw, ch, f_area, box_area, region_px)

        # Background surrounding a word is a ring by geometry but not a border. Without
        # this, every bold label became a stroked box sitting on top of the real element.
        hole_is_text = False
        if text_ink is not None:
            hole = (filled > 0) & (comp == 0)
            if hole.any():
                sub_ink = text_ink[y:y+ch, x:x+cw]
                hole_is_text = float((hole & (sub_ink > 0)).sum()) / float(hole.sum()) > 0.30

        if (0.5 <= ring_thickness <= 6.0 and min(cw, ch) >= 12
                and rect_score >= 0.55 and not hole_is_text):
            return Surface(bbox, filled, None, 'rect', radius, area, fill_ratio,
                           border_color=color, border_width=round(max(ring_thickness, 1.0), 1))
        if hole_is_text:
            # Text on a fill and the page showing around text look identical by area:
            # both are a ring whose hole is glyphs. What separates them is that a fill
            # has an edge of its own all the way round, while the page background bleeds
            # into the rest of the page and its bounding box is arbitrary. Without this
            # an orange banner with a headline on it was discarded outright.
            if min(cw, ch) >= 20 and _boundary_is_real(img, x, y, cw, ch, filled) >= 0.5:
                return Surface(bbox, filled, color, 'rect', radius, area, fill_ratio)
            return None

        # Not a ring after all. A container is perforated by everything drawn on top of
        # it, so its own fill covers only part of its box and looks annular by area
        # alone -- calling that complex is what reduced entire cards to photographs.
        # But a band of a gradient hole-fills into a plausible rectangle too, and
        # painting those as solid blocks wrecks the illustration. What separates them is
        # the perimeter: a container's fill wraps the whole way round its own edge.
        band = cv2.subtract(filled, cv2.erode(filled, _K3, iterations=2))
        band_px = int(band.sum())
        if band_px > 0 and float((band & comp).sum()) / band_px < 0.75:
            return Surface(bbox, filled, color, 'complex', 0.0, area, fill_ratio)

        # And check what is actually inside. A container's holes are the content drawn
        # on it -- text, icons, images -- so they vary wildly. A gradient band's holes
        # are its neighbouring shades, which barely vary at all. Painting one of those
        # as a solid box replaces a gradient with a flat slab.
        hole = (filled > 0) & (comp == 0)
        if hole.sum() >= 40:
            hole_px = img[y:y+ch, x:x+cw][hole].reshape(-1, 3).astype(np.float32)
            if float(hole_px.std(axis=0).max()) < 26.0:
                return Surface(bbox, filled, color, 'complex', 0.0, area, fill_ratio)

    # A shape running off the page edge is only partly observed, so its fill ratio is
    # inflated and the plain ellipse test below rejects it -- which is what turned the
    # carousel buttons into octagons. Fit an ellipse to what is visible and, if it
    # agrees, emit the whole circle and let the page clip it the way the source did.
    ih, iw = img.shape[:2]
    EDGE = 4  # a clipped shape may still leave a hairline of ground at the margin
    if (x <= EDGE or y <= EDGE or (x + cw) >= iw - EDGE or (y + ch) >= ih - EDGE) and min(cw, ch) >= 8:
        cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            if len(c) >= 5:
                try:
                    (ecx, ecy), (ew, eh), ang = cv2.fitEllipse(c)
                except cv2.error:
                    ew = eh = 0.0
                if ew > 4 and eh > 4 and 0.75 <= (ew / eh) <= 1.34:
                    probe = np.zeros((ch, cw), np.uint8)
                    cv2.ellipse(probe, (int(round(ecx)), int(round(ecy))),
                                (max(int(round(ew / 2)), 1), max(int(round(eh / 2)), 1)),
                                ang, 0, 360, 1, -1)
                    inter = float(np.count_nonzero(probe & filled))
                    union = float(np.count_nonzero(probe | filled))
                    if union > 0 and inter / union >= 0.88:
                        gx0, gy0 = x + ecx - ew / 2.0, y + ecy - eh / 2.0
                        return Surface((float(gx0), float(gy0), float(gx0 + ew), float(gy0 + eh)),
                                       filled, color, 'ellipse', 0.0, area, fill_ratio,
                                       vis_bbox=bbox)

    # Ellipse first: a circle scores plausibly on fill ratio, so test it explicitly
    # against a fitted ellipse before the rectangle path can claim it.
    if min(cw, ch) >= 6 and fill_ratio <= 0.88:
        probe = np.zeros((ch, cw), np.uint8)
        cv2.ellipse(probe, (cw // 2, ch // 2), (max(cw // 2, 1), max(ch // 2, 1)), 0, 0, 360, 1, -1)
        inter = float(np.count_nonzero(probe & filled))
        union = float(np.count_nonzero(probe | filled))
        if union > 0 and inter / union >= 0.90:
            return Surface(bbox, filled, color, 'ellipse', 0.0, area, fill_ratio)

    rect_score, radius = _rect_evidence(filled, cw, ch, f_area, box_area, region_px)
    if rect_score >= 0.62 and _boundary_is_real(img, x, y, cw, ch, filled) >= 0.35:
        return Surface(bbox, filled, color, 'rect', radius, area, fill_ratio)

    return Surface(bbox, filled, color, 'complex', 0.0, area, fill_ratio)

def _corner_radius(cw, ch, f_area, box_area) -> float:
    """Solves the corner radius from the area a rounded rectangle gives up: (4 - pi)r^2."""
    missing = max(box_area - f_area, 0.0)
    r = float(np.sqrt(missing / (4.0 - np.pi))) if missing > 1.0 else 0.0
    return float(min(round(r, 1), min(cw, ch) / 2.0))

def _corners_symmetric(filled: np.ndarray, tol: float = 0.34) -> bool:
    """Guards the radius solve: real rounded rects give up the same area at all 4 corners."""
    h, w = filled.shape
    k = max(2, int(min(h, w) * 0.25))
    if k * 2 >= min(h, w):
        return True
    quads = [filled[:k, :k], filled[:k, -k:], filled[-k:, :k], filled[-k:, -k:]]
    missing = [1.0 - float(q.sum()) / float(k * k) for q in quads]
    return (max(missing) - min(missing)) <= tol

def detect_surfaces(img: np.ndarray, page_bg: str, text_ink: Optional[np.ndarray] = None,
                    max_colors: int = 90, photo_mask: Optional[np.ndarray] = None,
                    photo_region: Optional[np.ndarray] = None,
                    scale: float = 1.0) -> List[Surface]:
    """Segments the image into flat-filled regions by quantised colour.

    UI is built from flat fills, so connected regions of one colour recover the real
    component boxes -- including nested ones (a panel inside a card inside a page),
    which a single global contour pass collapses into one blob. Anything that is not
    flat (gradients, artwork, photos) falls out as 'complex' and stays pixels.
    """
    h, w = img.shape[:2]
    min_area = MIN_SURFACE_AREA * scale * scale
    min_side = max(5, int(round(5 * scale)))
    smooth = cv2.medianBlur(img, 3)
    quant = (smooth.astype(np.int32) // 8 * 8).astype(np.uint32)
    packed = (quant[:, :, 0] << 16) | (quant[:, :, 1] << 8) | quant[:, :, 2]

    vals, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts)
    page_area = float(h * w)

    surfaces: List[Surface] = []
    for idx in order[:max_colors]:
        if counts[idx] < min_area:
            break
        mask = (packed == vals[idx]).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < min_area or cw < min_side or ch < min_side:
                continue
            comp = (lab[y:y+ch, x:x+cw] == i).astype(np.uint8)

            # Glyphs are flat fills too. A bold headline letter is a uniform connected
            # region large enough to pass every shape test, so without this check the
            # page fills with black boxes where the text should be.
            if text_ink is not None:
                overlap = float(np.count_nonzero(comp & text_ink[y:y+ch, x:x+cw]))
                if overlap / float(area) > 0.45:
                    continue

            # A shade of a photograph is not a component, however rectangular its
            # filled hull happens to look. The test is on the region's interior: tiles
            # straddling the edge of a flat control pick up the texture around it, and
            # judging on those vetoes real controls that merely sit on a photograph.
            core = cv2.erode(comp, _K3, iterations=2)
            probe = core if np.any(core) else comp
            probe_area = float(max(np.count_nonzero(probe), 1))

            if photo_mask is not None:
                inside = float(np.count_nonzero(probe & (photo_mask[y:y+ch, x:x+cw] > 0)))
                if inside / probe_area > 0.6:
                    continue

            # Textures alone do not catch a photograph of a man-made object: a shipping
            # container is flat-sided, straight-edged and smoothly lit, so its panels
            # pass every test a real component passes and come back as green slabs.
            # Within a region already judged photographic the bar goes up -- only
            # control-sized shapes that stand out sharply from their surroundings
            # survive, which is what a button on a photo does and a facet of the subject
            # does not.
            if photo_region is not None:
                ih, iw = h, w
                within = float(np.count_nonzero(probe & (photo_region[y:y+ch, x:x+cw] > 0)))
                # A component painted in a single colour is a fill, not a facet of a
                # photograph, however textured its surroundings are. Judging only by the
                # region mask vetoed flat controls that happened to sit inside one.
                own = img[y:y+ch, x:x+cw][probe > 0]
                uniform = (own.size > 0
                           and float(own.reshape(-1, 3).std(axis=0).max()) < 9.0)
                if within / probe_area > 0.6 and not uniform:
                    if (cw * ch) > page_area * 0.03:
                        continue
                    # Sample the ring from the surrounding image, not from inside the
                    # crop: a component fills its own bounding box, so dilating within
                    # those bounds is clipped away and leaves no ring at all -- which
                    # silently vetoed every high-contrast control on a photograph.
                    pad = max(5, int(round(5 * scale)))
                    ex0, ey0 = max(0, x - pad), max(0, y - pad)
                    ex1, ey1 = min(iw, x + cw + pad), min(ih, y + ch + pad)
                    padded = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
                    padded[y - ey0:y - ey0 + ch, x - ex0:x - ex0 + cw] = comp
                    ring = cv2.subtract(cv2.dilate(padded, _K3, iterations=3), padded)
                    outer = img[ey0:ey1, ex0:ex1][ring > 0]
                    inner = img[y:y+ch, x:x+cw][probe > 0]
                    if outer.size < 12 or inner.size < 12:
                        continue
                    contrast = float(np.linalg.norm(
                        np.median(outer.reshape(-1, 3), axis=0)
                        - np.median(inner.reshape(-1, 3), axis=0)))
                    if contrast < 55.0:
                        continue

            surf = _classify_region(img, x, y, cw, ch, comp, area, text_ink)
            if surf is None:
                continue
            # The page ground is not a component.
            if surf.color == page_bg and (cw * ch) / page_area > 0.45:
                continue
            surfaces.append(surf)
    return surfaces

# ==============================================================================
# 7c. CONTAINMENT TREE & SEMANTIC PROMOTION
# ==============================================================================

def _contains(outer, inner, pad: float = 1.5) -> bool:
    return (inner[0] >= outer[0] - pad and inner[1] >= outer[1] - pad and
            inner[2] <= outer[2] + pad and inner[3] <= outer[3] + pad)

def _hex_to_bgr(hex_color: str) -> np.ndarray:
    c = (hex_color or "#000000").lstrip("#")
    return np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], dtype=np.float32)

def _lum_bgr(bgr) -> float:
    return float(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2])

SHADOW_REACH = 30

def detect_box_shadow(img: np.ndarray, surf: 'Surface', parent_bg: str,
                      scale: float = 1.0):
    """Recovers a CSS drop shadow from the darkening in the ring around a surface.

    A shadow is a soft, monotonically decaying darkening of the background just outside
    a box -- not reproducible by a flat fill, so without this it lands in the residual
    pass as a raster the width of the whole control. Reading it back as box-shadow keeps
    it as CSS and stops that overlay existing at all.

    Returns (dx, dy, blur, alpha) or None. Conservative by design: content that merely
    happens to sit near the box (text, an icon, another panel) must not be mistaken for
    one, so the darkening has to decay with distance the way a blur does.
    """
    ih, iw = img.shape[:2]
    reach = max(10, int(round(SHADOW_REACH * scale)))
    x0, y0, x1, y1 = [int(round(v)) for v in surf.vis_bbox]
    if (x1 - x0) < 12 * scale or (y1 - y0) < 12 * scale:
        return None

    ex0, ey0 = max(0, x0 - reach), max(0, y0 - reach)
    ex1, ey1 = min(iw, x1 + reach), min(ih, y1 + reach)
    if (ex1 - ex0) < 16 or (ey1 - ey0) < 16:
        return None

    box = np.zeros((ey1 - ey0, ex1 - ex0), np.uint8)
    m = surf.mask
    mh, mw = min(m.shape[0], y1 - y0), min(m.shape[1], x1 - x0)
    if mh <= 0 or mw <= 0:
        return None
    box[y0 - ey0:y0 - ey0 + mh, x0 - ex0:x0 - ex0 + mw] = (m[:mh, :mw] > 0).astype(np.uint8)
    if not box.any():
        return None

    patch = img[ey0:ey1, ex0:ex1].astype(np.float32)
    bg = _hex_to_bgr(parent_bg)
    lum = 0.114 * patch[:, :, 0] + 0.587 * patch[:, :, 1] + 0.299 * patch[:, :, 2]
    darker = np.clip(_lum_bgr(bg) - lum, 0.0, None)

    # A shadow tints the ground without changing its hue; anything with a different
    # colour out here is other content, not shade.
    ratio = patch / np.maximum(bg.reshape(1, 1, 3), 1.0)
    neutral = (ratio.max(axis=2) - ratio.min(axis=2)) < 0.18
    outside = (box == 0)
    valid = outside & neutral

    dist = cv2.distanceTransform((box == 0).astype(np.uint8), cv2.DIST_L2, 3)
    profile = []
    for r in range(1, reach):
        band = valid & (dist >= r) & (dist < r + 1)
        profile.append(float(darker[band].mean()) if band.sum() >= 12 else 0.0)
    if len(profile) < 8 or profile[0] < 3.0:
        return None

    peak = max(profile[:4])
    if peak < 3.0 or peak > 90.0:
        return None
    # Must actually fade out; a neighbouring dark panel would not.
    tail = sum(profile[-4:]) / 4.0
    if tail > peak * 0.35:
        return None
    blur = float(next((r for r, v in enumerate(profile, 1) if v <= peak * 0.15), reach))
    alpha = float(min(peak / 255.0 * 2.6, 0.40))
    if alpha < 0.04:
        return None                     # too faint to be worth a rule

    # Offset from the imbalance between opposite sides, not from the centroid of all
    # darkening: a centroid is dragged around by whatever content happens to sit nearby,
    # which produced offsets pinned to the clamp in both directions.
    reach = max(int(blur) + 2, 4)
    near = valid & (dist <= reach)
    ys, xs = np.mgrid[0:box.shape[0], 0:box.shape[1]]
    by, bx = np.nonzero(box)
    top_e, bot_e = by.min(), by.max()
    lef_e, rig_e = bx.min(), bx.max()

    def side(sel):
        sel = sel & near
        return float(darker[sel].mean()) if sel.sum() >= 12 else 0.0

    top, bottom = side(ys < top_e), side(ys > bot_e)
    left, right = side(xs < lef_e), side(xs > rig_e)

    # Drop shadows fall downward. Darkening that is stronger above the box is something
    # else sitting behind it.
    if top > bottom * 1.6 and top > 4.0:
        return None

    v_asym = (bottom - top) / max(bottom + top, 1e-3)
    h_asym = (right - left) / max(right + left, 1e-3)
    limit = blur * 0.8
    dy = float(np.clip(round(v_asym * blur, 1), -limit, limit))
    dx = float(np.clip(round(h_asym * blur, 1), -limit, limit))
    # Horizontal drift beyond the blur radius is not a shadow shape any UI produces.
    if abs(dx) > blur * 0.6:
        dx = float(np.sign(dx) * round(blur * 0.6, 1))
    return dx, dy, round(blur, 1), round(alpha, 3)


# A surface is hole-filled out to its bounding box on the assumption that whatever
# perforates it is drawn on top of it. For a card that is true, and refusing to fill
# was what once reduced whole cards to photographs. For a strip of page background that
# happens to wrap around a card it is false: the component is 16% of its own box, and
# painting the other 84% put a sage-green slab across 60% of the logistics page.
# Everything standing on that slab then disagreed with the prediction, came back as one
# 710x396 residual raster, and the wrong colour showed through the glyph-shaped holes in
# that raster as a grey shadow behind every heading. It read as broken drop shadows; the
# cause was a fill claiming ground it never covered.
OVERCLAIM_OWN_SHARE = 0.55   # a fill covering less of its own filled area is suspect
OVERCLAIM_MISMATCH = 0.45    # and is dropped if it cannot explain this much of the rest


def drop_overclaimed_fills(surfaces: List[Surface], img: np.ndarray,
                           text_ink: Optional[np.ndarray] = None,
                           photo_region: Optional[np.ndarray] = None) -> int:
    """Stops a surface painting ground it neither covers nor hands to a child.

    Only fills that claim far more than they occupy are examined -- `area` is the
    component as it was found, before hole-filling squared it off -- and the question
    asked is the one that separates a container from a background: of what this colour
    claims, setting aside what a child will cover and what text will be drawn over,
    does the page actually look like this colour? A card says yes; its interior is its
    own fill, perforated by its contents. A background strip says no, because what it
    claims is the card standing on it.
    """
    h, w = img.shape[:2]
    dropped = 0
    for s in surfaces:
        if s.color is None or s.mask is None or s.mask.size == 0:
            continue
        mh, mw = s.mask.shape
        if mw < 16 or mh < 16:
            continue
        filled = s.mask > 0
        f_area = int(filled.sum())
        if f_area <= 0 or s.area / float(f_area) >= OVERCLAIM_OWN_SHARE:
            continue                       # occupies most of what it claims: a real fill

        x0, y0 = int(round(s.vis_bbox[0])), int(round(s.vis_bbox[1]))
        x1, y1 = min(w, x0 + mw), min(h, y0 + mh)
        if x1 <= x0 or y1 <= y0:
            continue
        test = filled[:y1 - y0, :x1 - x0].copy()
        for kid in s.children:             # a child will paint over its own footprint
            kx0, ky0, kx1, ky1 = [int(round(v)) for v in kid.bbox]
            ax0, ay0 = max(x0, kx0) - x0, max(y0, ky0) - y0
            ax1, ay1 = min(x1, kx1) - x0, min(y1, ky1) - y0
            if ax1 > ax0 and ay1 > ay0:
                test[ay0:ay1, ax0:ax1] = False
        if text_ink is not None:           # and live text will be drawn over its own
            test &= (text_ink[y0:y1, x0:x1] == 0)
        if photo_region is not None:       # and a backdrop will cover the artwork on it
            # Without this the test blames a fill for pixels that were never its job.
            # The SaaS page is a white page under a soft gradient hero: white is the
            # right answer for the page and wrong for every pixel the gradient covers,
            # so silencing it cost 1.3 points to fix a fault it did not have.
            test &= (photo_region[y0:y1, x0:x1] == 0)
        if int(test.sum()) < 800:
            continue

        c = s.color.lstrip('#')
        want = np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], np.float32)
        got = img[y0:y1, x0:x1][test].reshape(-1, 3).astype(np.float32)
        # A textured but correctly-coloured region has many pixels off the fill and a
        # median on it. Judging by the per-pixel count alone silenced an orange panel
        # whose real median was the same orange to within seven levels.
        if float(np.abs(np.median(got, axis=0) - want).mean()) <= 20.0:
            continue
        dev = np.abs(got - want).mean(axis=1)
        if float((dev > 28.0).mean()) > OVERCLAIM_MISMATCH:
            s.color = None                 # keep the container, stop it painting
            dropped += 1
    return dropped

def build_containment(surfaces: List[Surface]) -> List[Surface]:
    """Nests surfaces by area so each one's parent is the smallest box that holds it."""
    ordered = sorted(surfaces, key=lambda s: -(s.width * s.height))
    for i, inner in enumerate(ordered):
        for outer in ordered[:i][::-1]:      # nearest-in-size enclosing box wins
            if outer is not inner and _contains(outer.bbox, inner.bbox):
                inner.parent = outer
                outer.children.append(inner)
                break
    return [s for s in ordered if s.parent is None]

def assign_texts(surfaces: List[Surface], text_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attaches each text run to the innermost surface that contains it."""
    unassigned = []
    by_size = sorted(surfaces, key=lambda s: s.width * s.height)
    for run in text_runs:
        host = None
        for s in by_size:
            if s.shape != 'complex' and _contains(s.bbox, run['bbox'], pad=2.0):
                host = s
                break
        if host is not None:
            host.texts.append(run)
            run['host'] = host
        else:
            unassigned.append(run)
    return unassigned

def _luminance(hex_color: str) -> float:
    try:
        c = hex_color.lstrip('#')
        r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
        return 0.299 * r + 0.587 * g + 0.114 * b
    except Exception:
        return 0.0

def _contrast(a: str, b: str) -> float:
    """Perceptual-ish distance between two hex colours, 0..1."""
    return min(abs(_luminance(a) - _luminance(b)) / 160.0, 1.0)

def _interior_ink(ctx, surf: 'Surface', exclude_boxes: List[List[float]]) -> float:
    """Fraction of a surface's interior holding marks that are not its own text."""
    img, text_ink = ctx['img'], ctx['text_ink']
    x0, y0, x1, y1 = [int(v) for v in surf.bbox]
    x0, y0 = max(0, x0 + 2), max(0, y0 + 2)
    x1, y1 = min(img.shape[1], x1 - 2), min(img.shape[0], y1 - 2)
    if x1 <= x0 or y1 <= y0 or surf.color is None:
        return 0.0
    c = surf.color.lstrip('#')
    fill = np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], dtype=np.int32)
    patch = img[y0:y1, x0:x1].astype(np.int32)
    mark = (np.abs(patch - fill).max(axis=2) > 26)
    mark &= (text_ink[y0:y1, x0:x1] == 0)
    for b in exclude_boxes:
        bx0 = int(max(b[0] - x0, 0)); by0 = int(max(b[1] - y0, 0))
        bx1 = int(min(b[2] - x0, x1 - x0)); by1 = int(min(b[3] - y0, y1 - y0))
        if bx1 > bx0 and by1 > by0:
            mark[by0:by1, bx0:bx1] = False
    return float(mark.sum()) / float(mark.size or 1)

def _leading_icon(ctx, surf: 'Surface', run: Optional[Dict[str, Any]]) -> bool:
    """True when non-text marks sit in the left inset, ahead of any label."""
    img, text_ink = ctx['img'], ctx['text_ink']
    x0, y0, x1, y1 = [int(v) for v in surf.bbox]
    limit = int(run['bbox'][0]) if run else x0 + int(surf.width * 0.2)
    lx0, lx1 = max(0, x0 + 2), min(limit, x1)
    ly0, ly1 = max(0, y0 + 2), min(img.shape[0], y1 - 2)
    if lx1 - lx0 < 6 or ly1 <= ly0 or surf.color is None:
        return False
    c = surf.color.lstrip('#')
    fill = np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], dtype=np.int32)
    patch = img[ly0:ly1, lx0:lx1].astype(np.int32)
    mark = (np.abs(patch - fill).max(axis=2) > 26) & (text_ink[ly0:ly1, lx0:lx1] == 0)
    return float(mark.sum()) / float(mark.size or 1) > 0.04

def score_button(surf: 'Surface', ctx) -> float:
    """Weighted evidence that a surface is a clickable control."""
    if len(surf.texts) != 1 or surf.color is None:
        return 0.0
    run = surf.texts[0]
    label = (run.get('text') or '').strip()
    if not label:
        return 0.0

    w, h = surf.width, surf.height
    aspect = w / max(h, 1.0)
    score = 0.0

    if surf.radius >= 3.0:
        score += 0.25 if surf.radius < h * 0.45 else 0.25
    parent_fill = surf.parent.color if (surf.parent and surf.parent.color) else ctx['page_bg']
    if _contrast(surf.color, parent_fill) > 0.05 or surf.border_width > 0:
        score += 0.15

    tx0, tx1 = run['bbox'][0], run['bbox'][2]
    centred = abs(((tx0 + tx1) / 2.0) - ((surf.bbox[0] + surf.bbox[2]) / 2.0)) <= max(10.0, w * 0.10)
    if centred:
        score += 0.20
    elif aspect > 3.0:
        # A wide box whose label hugs the left edge is a field, not a button. Centring
        # is the defining trait of a control's label, so its absence has to disqualify
        # rather than merely cost points -- generic traits alone were carrying a search
        # field past the threshold.
        return min(score, 0.45)
    if len(label) <= 30:
        score += 0.10
    if 60 <= w <= 420 and 24 <= h <= 72 and 1.4 <= aspect <= 12.0:
        score += 0.10
    if _contrast(run.get('color', '#000000'), surf.color) > 0.30:
        score += 0.10

    # A control is mostly padding. A panel that happens to hold one centred line is not.
    text_area = (run['bbox'][2] - run['bbox'][0]) * (run['bbox'][3] - run['bbox'][1])
    if text_area / max(w * h, 1.0) < 0.55:
        score += 0.10
    return score

def score_input(surf: 'Surface', ctx) -> float:
    """Search/text fields demand strong evidence -- cards and filters look like them."""
    if surf.color is None or len(surf.texts) > 1:
        return 0.0
    w, h = surf.width, surf.height
    if not (24 <= h <= 60 and w >= 180 and w / max(h, 1.0) >= 4.0):
        return 0.0

    run = surf.texts[0] if surf.texts else None
    # A text field is a light box. Without this as a gate, a saturated CTA scored as an
    # input because its white label passed the "placeholder grey is bright" test.
    if _luminance(surf.color) <= 200 and surf.border_width <= 0:
        return 0.0

    score = 0.30                                         # pale fill + gated aspect
    if surf.radius >= 3.0:
        score += 0.10
    if run is not None and _luminance(run.get('color', '#000000')) > 120:
        score += 0.20                                    # placeholder grey, not body copy
    if run is not None:
        inset = run['bbox'][0] - surf.bbox[0]
        if 4.0 <= inset <= w * 0.25 and (surf.bbox[2] - run['bbox'][2]) > w * 0.25:
            score += 0.20                                # text starts at a left inset and
                                                         # leaves the field mostly empty
    if _leading_icon(ctx, surf, run):
        score += 0.25
    if _interior_ink(ctx, surf, [run['bbox']] if run else []) < 0.05:
        score += 0.15
    return score

def score_icon_button(surf: 'Surface', ctx) -> float:
    """Evidence that a textless shape is an icon-only control.

    A carousel arrow or a close button carries no label at all, so the label-based
    button score cannot see it. What it does have is a control-sized shape, a fill that
    stands off its background, and a small glyph floating in the middle of its own
    padding -- which is what separates it from a logo or an avatar, where the artwork
    fills the shape edge to edge.
    """
    if surf.texts or surf.color is None:
        return 0.0
    w, h = surf.width, surf.height
    if not (22 <= min(w, h) <= 76 and 0.78 <= w / max(h, 1.0) <= 1.28):
        return 0.0
    rounded = surf.shape == 'ellipse' or surf.radius >= min(w, h) * 0.25
    if not rounded:
        return 0.0

    ink = _interior_ink(ctx, surf, [])
    if not (0.04 <= ink <= 0.42):
        return 0.0                     # empty decoration, or artwork filling the shape

    # The glyph must float in the middle of its own padding. An illustration fragment
    # has marks running to its edges, and that is what separates a control from a piece
    # of artwork that merely happens to be round and the right size.
    img = ctx['img']
    x0, y0, x1, y1 = [int(v) for v in surf.bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    c = surf.color.lstrip('#')
    fill = np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], dtype=np.int32)
    patch = img[y0:y1, x0:x1].astype(np.int32)
    ph, pw = patch.shape[:2]

    def probe(scale):
        """The shape itself, shrunk -- never its bounding box, whose corners sit
        outside a circle and would always read as marks."""
        m = np.zeros((ph, pw), np.uint8)
        if surf.shape == 'ellipse':
            cv2.ellipse(m, (pw // 2, ph // 2),
                        (max(int(pw * scale), 1), max(int(ph * scale), 1)), 0, 0, 360, 1, -1)
        else:
            iy, ix = int(ph * (0.5 - scale)), int(pw * (0.5 - scale))
            m[iy:ph - iy, ix:pw - ix] = 1
        return m.astype(bool)

    interior, core = probe(0.42), probe(0.30)
    marks = (np.abs(patch - fill).max(axis=2) > 26) & interior
    if marks.sum() < 12:
        return 0.0
    if (marks & ~core).sum() > marks.sum() * 0.45:
        return 0.0                     # marks run to the edge: artwork, not a glyph
    ys, xs = np.nonzero(marks)
    if (abs(ys.mean() - ph / 2.0) > ph * 0.22) or (abs(xs.mean() - pw / 2.0) > pw * 0.22):
        return 0.0                     # off-centre: not a glyph sitting in its padding

    parent_fill = surf.parent.color if (surf.parent and surf.parent.color) else ctx['page_bg']
    score = 0.35                       # shape, size and glyph-with-padding all gated above
    if _contrast(surf.color, parent_fill) > 0.08:
        score += 0.25
    if surf.shape == 'ellipse':
        score += 0.15
    if 0.08 <= ink <= 0.30:
        score += 0.15                  # a glyph, comfortably inset
    if not surf.children:
        score += 0.10
    return score

def score_card(surf: 'Surface', ctx) -> float:
    """A card is a bounded region that genuinely groups aligned content."""
    if surf.color is None:
        return 0.0
    kids = [c for c in surf.children if c.shape != 'complex']
    members = len(kids) + len(surf.texts)
    if members < 2:
        return 0.0

    score = 0.0
    parent_fill = surf.parent.color if (surf.parent and surf.parent.color) else ctx['page_bg']
    if _contrast(surf.color, parent_fill) > 0.03 or surf.border_width > 0:
        score += 0.25                                    # a real visual boundary
    score += 0.25 if members >= 3 else 0.15
    if surf.width >= 120 and surf.height >= 90:
        score += 0.15
    if surf.radius >= 3.0 or surf.border_width > 0:
        score += 0.15

    edges = [c.bbox[0] for c in kids] + [t['bbox'][0] for t in surf.texts]
    if len(edges) >= 2 and float(np.std(edges)) <= max(12.0, surf.width * 0.12):
        score += 0.20                                    # children share an alignment
    return score

def score_switch(surf: 'Surface') -> float:
    """Evidence that a surface is a toggle switch (e.g. dark/light theme switch)."""
    w, h = surf.width, surf.height
    if not (14 <= h <= 56 and 24 <= w <= 140 and 1.35 <= (w / max(h, 1.0)) <= 3.2):
        return 0.0

    score = 0.0
    if surf.radius >= h * 0.35 or surf.shape == 'ellipse':
        score += 0.40

    has_knob = False
    for child in surf.children:
        cw, ch = child.width, child.height
        if child.shape == 'ellipse' or child.radius >= ch * 0.35:
            if 0.5 <= cw / max(ch, 1.0) <= 1.5 and ch <= h * 0.95:
                has_knob = True
                break
    if has_knob:
        score += 0.35

    if not surf.texts:
        score += 0.15
    else:
        label = ' '.join(t.get('text', '') for t in surf.texts).lower()
        if any(w in label for w in ['dark', 'light', 'theme', 'mode', 'on', 'off']):
            score += 0.35

    return min(score, 1.0)

def classify_surface_role(surf: 'Surface', ctx) -> Tuple[str, str, float]:
    """Chooses the HTML tag for a surface from scored evidence.

    Deliberately conservative, and deliberately separate from how the surface is
    painted: geometry and colour are reconstructed identically whatever this returns.
    Below the promotion threshold a surface still renders pixel-for-pixel -- it just
    renders as a <div> instead of claiming to be something it might not be.
    """
    sw = score_switch(surf)
    b = score_button(surf, ctx)
    i = score_input(surf, ctx)
    c = score_card(surf, ctx)
    k = score_icon_button(surf, ctx)
    best = max(sw, b, i, c, k)

    if best < 0.50:
        return ('shape' if surf.shape == 'ellipse' else 'surface'), 'div', best

    if sw == best and sw >= 0.65:
        return 'switch', 'button', sw

    if k == best and k >= 0.75:
        return 'icon-button', 'button', k

    if b == best and b >= 0.75:
        run = surf.texts[0]
        label = (run.get('text') or '').strip()
        if surf.height <= 30 and surf.width <= 170 and len(label) <= 18:
            return 'badge', 'span', b
        return 'button', 'button', b
    if i == best and i >= 0.80:
        return 'input', 'input', i
    if c == best and c >= 0.70:
        return 'card', 'div', c

    # Enough evidence to be a container, not enough to name it. Visually identical.
    return ('input-like' if i == best else 'panel'), 'div', best

def _same_run_style(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Whether two runs look like members of one set.

    Only size is compared. Weight and colour both vary legitimately inside a real set --
    a nav marks its active item bold, an upvote count is tinted when it is active -- and
    requiring either to match split genuine groups apart. The rhythm of the gaps, which
    the caller checks, is far stronger evidence of a set than uniform styling is.
    """
    big = max(a['fontSize'], b['fontSize'], 1.0)
    return abs(a['fontSize'] - b['fontSize']) / big <= 0.20

def _consistent(gaps: List[float], tol: float = 0.45) -> bool:
    """True when a sequence of gaps reads as a deliberate rhythm rather than chance."""
    if not gaps:
        return False
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return False
    spread = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
    return (spread / mean) <= tol

def _runs_of(items, key_gap, min_len=3):
    """Splits an ordered list into maximal runs with a consistent gap and shared style."""
    out, run = [], [items[0]] if items else []
    for prev, cur in zip(items, items[1:]):
        gaps = [key_gap(a, b) for a, b in zip(run, run[1:])] + [key_gap(prev, cur)]
        if _same_run_style(prev, cur) and key_gap(prev, cur) > 0 and _consistent(gaps):
            run.append(cur)
        else:
            if len(run) >= min_len:
                out.append(run)
            run = [cur]
    if len(run) >= min_len:
        out.append(run)
    return out

def find_leading_mark(img, text_ink, run, reach: float = 52.0):
    """Bounding box of non-text ink immediately left of a run, if any.

    A list item or an action control is an icon and a label read as one thing. Scoring
    the label alone both loses the icon from the element's box and throws away the
    strongest clue that the pair is interactive at all.
    """
    ih, iw = img.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in run['bbox']]
    lx1 = max(0, x0 - 4)
    lx0 = max(0, int(x0 - reach))
    ly0, ly1 = max(0, y0 - 6), min(ih, y1 + 6)
    if lx1 - lx0 < 6 or ly1 <= ly0:
        return None
    patch = img[ly0:ly1, lx0:lx1]
    bg = np.median(patch.reshape(-1, 3), axis=0)
    mark = (np.abs(patch.astype(np.int32) - bg.astype(np.int32)).max(axis=2) > 30)
    mark &= (text_ink[ly0:ly1, lx0:lx1] == 0)
    if mark.sum() < 18:
        return None
    ys, xs = np.nonzero(mark)
    return [float(lx0 + xs.min()), float(ly0 + ys.min()),
            float(lx0 + xs.max() + 1), float(ly0 + ys.max() + 1)]

def _in_wrapped_block(r: Dict[str, Any], pool: List[Dict[str, Any]]) -> bool:
    """True when a run is one line of a multi-line label rather than a standalone item.

    Three two-line captions side by side produce two rows that look exactly like a row
    of controls: same count, same rhythm, each with a leading icon. The difference is
    that each caption continues onto the next line, and a control does not.
    """
    for o in pool:
        if o is r or not _same_run_style(r, o):
            continue
        if abs(o['bbox'][0] - r['bbox'][0]) > 8.0:
            continue
        below = o['bbox'][1] - r['bbox'][3]
        above = r['bbox'][1] - o['bbox'][3]
        limit = max(r['fontSize'] * 0.9, 6.0)
        if -3.0 <= below <= limit or -3.0 <= above <= limit:
            return True
    return False

def detect_sibling_groups(runs: List[Dict[str, Any]], img, text_ink,
                          page_h: float) -> List[Dict[str, Any]]:
    """Finds repeated elements and classifies them as a set rather than one at a time.

    Four evenly spaced labels in a row is overwhelming evidence of a nav; a single one
    of them in isolation is evidence of nothing. Scoring each separately also produced
    different tags for members of the same obvious group, which is the tell that the
    unit of classification was wrong. Deciding once per group and applying the result to
    every member fixes both the accuracy and the inconsistency.
    """
    free = [r for r in runs
            if not r.get('consumed')
            and r.get('host_role') not in ('button', 'badge', 'input')
            and (r.get('text') or '').strip()]
    if len(free) < 3:
        return []

    groups: List[Dict[str, Any]] = []
    used: set = set()

    # --- rows: shared vertical centre, consistent horizontal gaps -------------------
    rows: List[List[Dict[str, Any]]] = []
    for r in sorted(free, key=lambda r: (r['bbox'][1] + r['bbox'][3]) / 2.0):
        cy = (r['bbox'][1] + r['bbox'][3]) / 2.0
        h = r['bbox'][3] - r['bbox'][1]
        for row in rows:
            rcy = (row[0]['bbox'][1] + row[0]['bbox'][3]) / 2.0
            if abs(cy - rcy) <= max(5.0, h * 0.6):
                row.append(r)
                break
        else:
            rows.append([r])

    for row in rows:
        if len(row) < 3:
            continue
        row.sort(key=lambda r: r['bbox'][0])
        for run_items in _runs_of(row, lambda a, b: b['bbox'][0] - a['bbox'][2]):
            if any(id(r) in used for r in run_items):
                continue
            groups.append({'axis': 'row', 'members': run_items})
            used.update(id(r) for r in run_items)

    # --- columns: shared left edge, consistent vertical pitch -----------------------
    cols: List[List[Dict[str, Any]]] = []
    for r in sorted(free, key=lambda r: r['bbox'][0]):
        if id(r) in used:
            continue
        for col in cols:
            if abs(r['bbox'][0] - col[0]['bbox'][0]) <= 10.0:
                col.append(r)
                break
        else:
            cols.append([r])

    for col in cols:
        if len(col) < 3:
            continue
        col.sort(key=lambda r: r['bbox'][1])
        for run_items in _runs_of(col, lambda a, b: b['bbox'][1] - a['bbox'][1]):
            if any(id(r) in used for r in run_items):
                continue
            groups.append({'axis': 'col', 'members': run_items})
            used.update(id(r) for r in run_items)

    # --- decide what each group is -------------------------------------------------
    for g in groups:
        members = g['members']
        marks = [find_leading_mark(img, text_ink, r) for r in members]
        with_marks = sum(1 for m in marks if m)
        g['marks'] = marks
        top = min(r['bbox'][1] for r in members)
        labels_short = all(len((r.get('text') or '')) <= 26 for r in members)

        if g['axis'] == 'row' and top <= page_h * 0.12 and labels_short and with_marks == 0:
            g['role'], g['tag'] = 'navlink', 'a'
        elif g['axis'] == 'col' and labels_short and with_marks >= len(members) - 1:
            g['role'], g['tag'] = 'listitem', 'a'
        elif (g['axis'] == 'row' and labels_short
              and with_marks >= max(2, len(members) - 1)
              and sum(1 for r in members
                      if r.get('host_role') in ('card', 'panel')) >= len(members) - 1
              and sum(1 for r in members if _in_wrapped_block(r, free)) <= len(members) // 2):
            # Inside a card, a row of icon-and-label pairs is a toolbar. The same shape
            # on the page background is a set of feature callouts, so what separates
            # them is the container, not the pairs themselves.
            g['role'], g['tag'] = 'action', 'button'
        else:
            # Repetition alone is not evidence of interactivity. Keep the members
            # consistent with each other and leave the tag neutral.
            g['role'], g['tag'] = 'group-item', 'span'
    return groups

def assign_heading_levels(runs: List[Dict[str, Any]]) -> None:
    """Ranks distinct body-independent font sizes into h1/h2/h3.

    Headings are not 'big text' in absolute terms -- they are big relative to this
    page's body copy, so the body size is derived first (the most common size,
    weighted by how much text is set in it) and only clearly larger runs promote.
    """
    free = [r for r in runs if not r.get('host_role') in ('button', 'badge', 'input')]
    if not free:
        return
    weight: Dict[float, int] = {}
    for r in free:
        size = round(float(r['fontSize']), 1)
        weight[size] = weight.get(size, 0) + len((r.get('text') or ''))
    body = max(weight.items(), key=lambda kv: kv[1])[0]

    # Only the genuinely distinct display sizes become headings. A page has a handful
    # of heading ranks, not one per measured size, and a document that is all <h6> is
    # no more structured than one with none.
    big = sorted({round(float(r['fontSize']), 1) for r in free if r['fontSize'] >= body * 1.35},
                 reverse=True)[:3]
    level_of = {size: f'h{i + 1}' for i, size in enumerate(big)}
    for r in free:
        size = round(float(r['fontSize']), 1)
        if size in level_of:
            r['tag'] = level_of[size]
            r['role'] = 'heading'
        elif abs(size - body) < 0.6:
            r['role'] = 'body'

def group_paragraphs(runs: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Merges consecutive same-style lines on a shared left edge into paragraph groups.

    Requires a consistent line pitch as well as matching style, so a stack of unrelated
    labels that happen to share an edge is not welded into one <p>.
    """
    cands = [r for r in runs
             if r.get('role') == 'body' and not r.get('grouped')
             and not r.get('host_role') in ('button', 'badge', 'input')]
    cands.sort(key=lambda r: (round(r['bbox'][0] / 4.0), r['bbox'][1]))

    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    def flush():
        if len(current) >= 2:
            groups.append(list(current))
        current.clear()

    for r in cands:
        if not current:
            current.append(r)
            continue
        prev = current[-1]
        same_style = (abs(prev['fontSize'] - r['fontSize']) < 0.8 and
                      prev['color'] == r['color'] and
                      prev['fontWeight'] == r['fontWeight'] and
                      prev.get('host') is r.get('host'))
        # Ink left edges differ by a few px purely from side bearings (an R vs a t).
        aligned = abs(prev['bbox'][0] - r['bbox'][0]) <= 6.0
        gap = r['bbox'][1] - prev['bbox'][3]
        line_pitch = r['bbox'][1] - prev['bbox'][1]
        plausible = -2.0 <= gap <= prev['fontSize'] * 1.1 and line_pitch > 0
        if len(current) >= 2:
            first_pitch = current[1]['bbox'][1] - current[0]['bbox'][1]
            plausible = plausible and abs(line_pitch - first_pitch) <= max(3.0, first_pitch * 0.25)
        if same_style and aligned and plausible:
            current.append(r)
        else:
            flush()
            current.append(r)
    flush()
    return groups

def _columns_are_wrapped_labels(rows: List[List[Dict[str, Any]]]) -> bool:
    """True when a candidate lattice is really a row of multi-line labels side by side.

    Three stacked two-line captions line up exactly like a 2x3 table. The tell is that
    each column is one continuous label -- same style, line-spacing pitch -- rather than
    independent cell values.
    """
    if len(rows) > 2:
        return False
    n_cols = len(rows[0])
    wrapped = 0
    for c in range(n_cols):
        col = [row[c] for row in rows]
        a, b = col[0], col[-1]
        same_style = (abs(a['fontSize'] - b['fontSize']) < 0.8 and a['color'] == b['color'])
        tight = 0 < (b['bbox'][1] - a['bbox'][3]) <= a['fontSize'] * 0.9
        if same_style and tight:
            wrapped += 1
    return wrapped >= max(1, n_cols - 1)

def detect_text_lattice(runs: List[Dict[str, Any]], min_rows: int = 3,
                        min_cols: int = 2) -> List[List[List[Dict[str, Any]]]]:
    """Finds 2-D grids of text runs that should become real <table> elements.

    Demands both a repeated row structure and columns that line up across those rows;
    a single column of labels or one wide row is not a table.
    """
    free = [r for r in runs if r.get('host_role') not in ('button', 'badge', 'input')
            and not r.get('consumed') and not r.get('grouped')]
    if len(free) < min_rows * min_cols:
        return []

    rows: List[List[Dict[str, Any]]] = []
    for r in sorted(free, key=lambda r: r['bbox'][1]):
        placed = False
        for row in rows:
            ref = row[0]
            overlap = min(ref['bbox'][3], r['bbox'][3]) - max(ref['bbox'][1], r['bbox'][1])
            if overlap > 0.5 * min(ref['bbox'][3] - ref['bbox'][1], r['bbox'][3] - r['bbox'][1]):
                row.append(r)
                placed = True
                break
        if not placed:
            rows.append([r])

    grid_rows = [sorted(row, key=lambda r: r['bbox'][0]) for row in rows if len(row) >= min_cols]
    if len(grid_rows) < min_rows:
        return []

    tables = []
    run_group = [grid_rows[0]]
    for row in grid_rows[1:]:
        prev = run_group[-1]
        if len(row) == len(prev) and all(
            abs(a['bbox'][0] - b['bbox'][0]) <= 12.0 for a, b in zip(row, prev)
        ):
            run_group.append(row)
        else:
            if len(run_group) >= min_rows and not _columns_are_wrapped_labels(run_group):
                tables.append(list(run_group))
            run_group = [row]
    if len(run_group) >= min_rows and not _columns_are_wrapped_labels(run_group):
        tables.append(list(run_group))
    return tables

# ==============================================================================
# 8. IMAGE RECONSTRUCTOR (LAYERED VISUAL ENGINE)
# ==============================================================================

class ImageReconstructor:
    _cached_ocr = None
    _ocr_disabled = False

    @classmethod
    def _get_ocr(cls):
        if cls._ocr_disabled:
            return None
        if cls._cached_ocr is None:
            try:
                from paddleocr import PaddleOCR
                try:
                    # Screenshots and rendered pages are already axis-aligned and flat.
                    # PaddleOCR's document preprocessing (orientation classification and
                    # UVDoc dewarping) geometrically resamples the page, which shifts every
                    # returned box away from the pixels we position elements against.
                    cls._cached_ocr = PaddleOCR(
                        lang='en',
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                except TypeError:
                    # Older PaddleOCR builds do not expose these switches.
                    cls._cached_ocr = PaddleOCR(lang='en')
            except Exception as e:
                logger.warning(f"Could not initialize PaddleOCR: {e}")
                cls._ocr_disabled = True
                return None
        return cls._cached_ocr

    @staticmethod
    def reconstruct_image(file_path: str, asset_dir: Optional[str] = None) -> PageData:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Unable to read image at {file_path}")
        return ImageReconstructor.reconstruct_image_from_cv2(img, asset_dir=asset_dir)

    @staticmethod
    def _extract_text_runs(img: np.ndarray) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """OCRs the image into text runs sized against their true ink bounds."""
        h, w = img.shape[:2]
        runs: List[Dict[str, Any]] = []
        text_mask = np.zeros((h, w), np.uint8)

        ocr = ImageReconstructor._get_ocr()
        if not ocr:
            return runs, text_mask

        try:
            ocr_out = ocr.ocr(img)
        except Exception as e:
            logger.warning(f"OCR failed during image reconstruction: {e}")
            return runs, text_mask
        if not ocr_out:
            return runs, text_mask

        res = ocr_out[0]
        if isinstance(res, dict):
            texts = res.get('rec_texts', [])
            boxes = res.get('rec_polys', [])
            scores = res.get('rec_scores', [])
        else:
            texts = [l[1][0] for l in res if len(l) >= 2]
            boxes = [l[0] for l in res if len(l) >= 2]
            scores = [l[1][1] for l in res if len(l) >= 2 and len(l[1]) >= 2]

        # Which family this page is set in, decided before the first run is fitted.
        # It used to be worked out afterwards and written over the styles, so every
        # size on a serif page had been fitted against sans metrics and then rendered
        # in a serif -- the two faces do not have the same proportions, and the whole
        # point of measuring is that they should be the same face.
        sample = []
        for text, box, score in zip(texts, boxes, scores):
            if float(score) < 0.35 or not str(text).strip():
                continue
            ys = [int(p[1]) for p in box]
            xs = [int(p[0]) for p in box]
            by0, by1 = max(0, min(ys)), min(h, max(ys))
            bx0, bx1 = max(0, min(xs)), min(w, max(xs))
            if by1 - by0 >= 16 and bx1 > bx0:
                sample.append((by1 - by0, img[by0:by1, bx0:bx1]))
        sample.sort(key=lambda t: -t[0])
        page_family = detect_font_family([c for _, c in sample[:15]])
        page_is_serif = page_family == FONT_STACK_EDITORIAL_SERIF

        for text, box, score in zip(texts, boxes, scores):
            label = str(text).strip()
            if float(score) < 0.35 or not label:
                continue
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x0, y0 = max(0, min(xs)), max(0, min(ys))
            x1, y1 = min(w, max(xs)), min(h, max(ys))
            if x1 <= x0 or y1 <= y0:
                continue

            # Real page copy is strictly horizontal and square. Lettering on angled
            # objects or 3D perspective mockups (tilted laptop/phone screens, books)
            # has skewed horizontal edges or non-vertical side edges.
            quad = np.asarray(box, dtype=np.float32)
            if quad.shape == (4, 2):
                e_top = quad[1] - quad[0]
                skew_top = abs(np.degrees(np.arctan2(float(e_top[1]), float(e_top[0]))))
                e_bot = quad[2] - quad[3]
                skew_bot = abs(np.degrees(np.arctan2(float(e_bot[1]), float(e_bot[0]))))
                e_left = quad[3] - quad[0]
                tilt_left = abs(np.degrees(np.arctan2(float(e_left[0]), float(e_left[1]))))
                e_right = quad[2] - quad[1]
                tilt_right = abs(np.degrees(np.arctan2(float(e_right[0]), float(e_right[1]))))

                top_dev = min(skew_top, 180.0 - skew_top)
                bot_dev = min(skew_bot, 180.0 - skew_bot)
                if (top_dev > 2.5 and bot_dev > 2.0) or (top_dev > 4.0 or bot_dev > 4.0):
                    continue
                if tilt_left > 3.5 or tilt_right > 3.5:
                    continue

            # A single character with no word around it is almost always an icon the
            # recogniser forced into the alphabet (a magnifier read as 'Q', a bell as
            # 'D'). Leave those pixels to the artwork pass instead of inventing text.
            if len(label) == 1 and not label.isalnum():
                continue
            if len(label) == 1 and (x1 - x0) >= (y1 - y0) * 0.9:
                continue

            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            x0_box, y0_box = x0, y0      # crop origin, before the ink box narrows it

            box_h = y1 - y0
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, bin_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            stroke_ratio = np.count_nonzero(bin_crop) / float(crop.shape[0] * crop.shape[1])
            font_weight = "bold" if (stroke_ratio > 0.28 or box_h > 32) else "normal"

            border_px = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0)
            local_bg = np.median(border_px, axis=0)
            diff = np.linalg.norm(crop.astype(float) - local_bg.astype(float), axis=2)
            max_d = float(np.max(diff))

            text_color = "#111111"
            ink_box = None
            ink_mask = None
            candidate = None
            if max_d > 22.0:
                # Detection boxes carry padding, so they are not ink bounds. Fitting a
                # font to them oversizes every run. Recover the true ink extent from the
                # pixels that differ from the local background -- which also works for
                # light-on-dark, where an Otsu "dark pixels are ink" test is backwards.
                candidate = (diff >= max(20.0, max_d * 0.45)).astype(np.uint8)
                glyph_bgr = dominant_ink_colour(crop, candidate)
                if glyph_bgr is not None:
                    text_color = _bgr_to_hex(glyph_bgr.astype(int))
                else:
                    stroke_px = crop[diff >= max(18.0, float(np.percentile(diff, 75)))]
                    if len(stroke_px) > 0:
                        text_color = _bgr_to_hex(np.median(stroke_px, axis=0).astype(int))

                ink = colour_selective_ink(crop, candidate, glyph_bgr)
                # Mask every candidate pixel so no text ghost is left on the background,
                # whatever colour it was.
                ink_mask = np.maximum(ink, candidate)
                # Both bounds are taken from all the ink on the line, not just the ink
                # matching its dominant colour. A two-tone line -- "your buyers read.",
                # black then accent -- measured only its majority colour, and since
                # "your" has no ascender and no capital the run was read as 56px tall
                # where the line is 72, so one headline came out as two sizes, 75px and
                # 57px. keep_core_ink still drops what does not belong: the descender of
                # the line above hangs into this box but never reaches its body band.
                # Bounds from every ink pixel, not only those matching the run's
                # dominant colour. A line that changes colour part way through --
                # "your buyers read.", black then accent -- was measured from whichever
                # half won the vote, and since "your" carries no ascender and no capital
                # the run came out 56px tall against a real 72. One headline, two sizes.
                # keep_core_ink still drops what does not belong: the descender of the
                # line above hangs into this box but never reaches its body band.
                core_ink = keep_core_ink(ink_mask)
                rows = np.any(core_ink, axis=1) if core_ink.any() else np.any(candidate, axis=1)
                # Width still comes from every candidate pixel. keep_core_ink exists to
                # drop what hangs in from the line above, which is a question about
                # height; applying it sideways also discarded marks that are simply
                # narrow, and pulled the box in off the ends of the run.
                cols = np.any(candidate, axis=0) if candidate.any() else np.any(ink, axis=0)

                if rows.any() and cols.any():
                    r0 = int(np.argmax(rows))
                    r1 = len(rows) - int(np.argmax(rows[::-1]))
                    c0 = int(np.argmax(cols))
                    c1 = len(cols) - int(np.argmax(cols[::-1]))
                    # Protect against over-narrowing: OCR detection box found the line span.
                    # Never shrink width by more than 20% on lines with multiple characters.
                    orig_w = x1 - x0
                    if len(label) > 3 and (c1 - c0) < orig_w * 0.75:
                        c0, c1 = 0, orig_w
                    if r1 > r0 and c1 > c0:
                        ink_box = (x0 + c0, y0 + r0, x0 + c1, y0 + r1)
            else:
                lum = 0.299 * local_bg[2] + 0.587 * local_bg[1] + 0.114 * local_bg[0]
                text_color = "#ffffff" if lum < 128 else "#111111"

            # Record the glyph pixels themselves, not the box. A filled box would hide
            # whatever a control's background is doing between the letters.
            if ink_mask is not None:
                stroke = ink_mask.astype(np.uint8) * 255
                sub = text_mask[y0_box:y0_box + crop.shape[0], x0_box:x0_box + crop.shape[1]]
                np.maximum(sub, stroke, out=sub)

            if ink_box:
                x0, y0, x1, y1 = ink_box

            # What weight the lettering is actually set at, from the ink itself. The
            # provisional guess above is a stroke-ratio threshold with a clause that
            # calls anything taller than 32px bold, which is wrong for every light
            # headline and has only two answers for a question with six.
            weight_mask = None
            if ink_box is not None and ink_mask is not None:
                bx0, by0, bx1, by1 = ink_box
                weight_mask = ink_mask[by0 - y0_box:by1 - y0_box,
                                       bx0 - x0_box:bx1 - x0_box]
            # Too small to measure falls back to the old stroke-ratio guess, which is
            # crude but is not systematically wrong the way a blurred measurement is.
            font_weight = str(estimate_font_weight(
                label, weight_mask if weight_mask is not None and weight_mask.size
                else (bin_crop > 0).astype(np.uint8), page_is_serif,
                fallback=700 if stroke_ratio > 0.28 else 400))

            font_size, letter_spacing, line_height, word_spacing, scale_x = fit_text_to_box(
                label, x1 - x0, y1 - y0, is_heavy(font_weight), serif=page_is_serif
            )

            # Where the lettering changes colour along the run. Read from the same crop
            # the box was measured from, so the positions line up with what is drawn.
            colour_stops = None
            if candidate is not None and ink_box:
                bx0, by0, bx1, by1 = ink_box
                sub_c = candidate[by0 - y0_box:by1 - y0_box, bx0 - x0_box:bx1 - x0_box]
                sub_d = diff[by0 - y0_box:by1 - y0_box, bx0 - x0_box:bx1 - x0_box]
                sub_img = crop[by0 - y0_box:by1 - y0_box, bx0 - x0_box:bx1 - x0_box]
                if sub_c.size and sub_c.shape[1] > 8:
                    colour_stops = ink_colour_bands(sub_img, sub_c, sub_d)

            spans = None
            words = label.split()
            if len(words) >= 2 and candidate is not None and np.count_nonzero(candidate) >= 20:
                col_ink = np.sum(candidate, axis=0)
                in_ink = False
                start_c = 0
                clusters = []
                for ci, c_val in enumerate(col_ink):
                    if c_val > 0 and not in_ink:
                        in_ink = True
                        start_c = ci
                    elif c_val == 0 and in_ink:
                        in_ink = False
                        clusters.append((start_c, ci))
                if in_ink:
                    clusters.append((start_c, len(col_ink)))

                if len(clusters) >= len(words):
                    gaps = []
                    for gi in range(len(clusters) - 1):
                        g_start = clusters[gi][1]
                        g_end = clusters[gi + 1][0]
                        gaps.append((g_end - g_start, g_start, g_end))

                    gaps_sorted = sorted(gaps, key=lambda g: g[0], reverse=True)
                    num_spaces = len(words) - 1
                    split_gaps = sorted(gaps_sorted[:num_spaces], key=lambda g: g[1])

                    word_spans_x = []
                    w_start = clusters[0][0]
                    for g in split_gaps:
                        w_end = g[1]
                        word_spans_x.append((w_start, w_end))
                        w_start = g[2]
                    word_spans_x.append((w_start, clusters[-1][1]))

                    word_colors = []
                    for (ws_x, we_x) in word_spans_x:
                        sub_crop = crop[:, ws_x:we_x]
                        sub_cand = candidate[:, ws_x:we_x]
                        wbgr = dominant_ink_colour(sub_crop, sub_cand)
                        if wbgr is None:
                            diff_sub = diff[:, ws_x:we_x]
                            sub_stroke = sub_crop[diff_sub >= max(18.0, float(np.percentile(diff_sub, 75)))]
                            if len(sub_stroke) > 0:
                                wbgr = np.median(sub_stroke, axis=0)
                        word_colors.append(wbgr)

                    has_distinct_colors = False
                    valid_colors = [c for c in word_colors if c is not None]
                    if len(valid_colors) == len(words):
                        for i in range(len(valid_colors)):
                            for j in range(i + 1, len(valid_colors)):
                                # A change of colour, not of shade: two greys are one
                                # ink sampled through different amounts of edge, and
                                # splitting on them gave "Sign" and "in" two colours.
                                if _is_a_colour_change(
                                        np.clip(valid_colors[i], 0, 255).astype(np.uint8),
                                        np.clip(valid_colors[j], 0, 255).astype(np.uint8)):
                                    has_distinct_colors = True
                                    break
                            if has_distinct_colors:
                                break

                    if has_distinct_colors:
                        groups = []
                        curr_group_words = [words[0]]
                        curr_group_color = word_colors[0]
                        curr_group_start = word_spans_x[0][0]
                        curr_group_end = word_spans_x[0][1]

                        for wi in range(1, len(words)):
                            w_c = word_colors[wi]
                            if w_c is not None and curr_group_color is not None and np.linalg.norm(w_c - curr_group_color) <= 30.0:
                                curr_group_words.append(words[wi])
                                curr_group_end = word_spans_x[wi][1]
                            else:
                                groups.append((curr_group_words, curr_group_color, curr_group_start, curr_group_end))
                                curr_group_words = [words[wi]]
                                curr_group_color = w_c
                                curr_group_start = word_spans_x[wi][0]
                                curr_group_end = word_spans_x[wi][1]
                        groups.append((curr_group_words, curr_group_color, curr_group_start, curr_group_end))

                        spans = []
                        for gi, (g_words, g_col, g_x0, g_x1) in enumerate(groups):
                            g_text = " ".join(g_words)
                            if gi < len(groups) - 1:
                                g_text += " "
                            g_hex = _bgr_to_hex(g_col.astype(int)) if g_col is not None else text_color
                            spans.append(TextSpan(
                                text=g_text,
                                bbox=[float(x0_box + g_x0), float(y0), float(x0_box + g_x1), float(y1)],
                                style=TextStyle(
                                    fontFamily=page_family,
                                    fontSize=float(font_size),
                                    fontWeight=font_weight,
                                    color=g_hex,
                                )
                            ))
                        if spans:
                            text_color = spans[0].style.color

            run_dict = {
                'text': label,
                'bbox': [float(x0), float(y0), float(x1), float(y1)],
                'fontSize': float(font_size),
                'fontWeight': font_weight,
                'color': text_color,
                'style': TextStyle(
                    fontFamily=page_family,
                    fontSize=float(font_size),
                    fontWeight=font_weight,
                    color=text_color,
                    lineHeight=line_height,
                    letterSpacing=letter_spacing,
                    wordSpacing=word_spacing,
                    scaleX=scale_x,
                    colorStops=colour_stops,
                ),
            }
            if spans:
                run_dict['spans'] = spans
            runs.append(run_dict)
        return runs, text_mask

    @staticmethod
    def _dedupe_surfaces(surfaces: List['Surface']) -> List['Surface']:
        """Collapses near-identical surfaces produced by colour quantisation.

        One painted fill can straddle a quantisation boundary and come back as two
        components a shade apart occupying the same box. Both then compete: the text
        attaches to whichever is innermost, leaving its twin an unpromoted div sitting
        over the real control. Keeping the larger of each pair removes the competition.
        """
        kept: List['Surface'] = []
        for s in sorted(surfaces, key=lambda s: -(s.width * s.height)):
            dup = False
            for k in kept:
                if compute_bbox_iou(s.bbox, k.bbox) < 0.75:
                    continue
                if s.color is None or k.color is None:
                    dup = True
                    break
                a, b = s.color.lstrip('#'), k.color.lstrip('#')
                try:
                    d = sum((int(a[i:i+2], 16) - int(b[i:i+2], 16)) ** 2 for i in (0, 2, 4)) ** 0.5
                except Exception:
                    d = 999.0
                if d <= 24.0:
                    # Keep whichever carries the stronger corner evidence.
                    if s.radius > k.radius:
                        k.radius = s.radius
                    dup = True
                    break
            if not dup:
                kept.append(s)
        return kept

    @staticmethod
    def _merge_borders(surfaces: List['Surface']) -> List['Surface']:
        """Folds an outline ring into the fill it encloses, yielding one bordered box."""
        rings = [s for s in surfaces if s.color is None and s.border_color]
        fills = [s for s in surfaces if s.color is not None]
        consumed = set()
        for ring in rings:
            best = None
            for f in fills:
                if not _contains(ring.bbox, f.bbox, pad=ring.border_width + 2.0):
                    continue
                if f.width * f.height < ring.width * ring.height * 0.55:
                    continue
                if best is None or f.width * f.height > best.width * best.height:
                    best = f
            if best is not None:
                best.border_color = ring.border_color
                best.border_width = ring.border_width
                best.bbox = ring.bbox
                if ring.radius > best.radius:
                    best.radius = ring.radius
                consumed.add(id(ring))
        return [s for s in surfaces if id(s) not in consumed and
                (s.color is not None or s.border_color is not None)]

    @staticmethod
    def _extract_residual_art(img, w, h, mean_bg, text_mask, paintable,
                              asset_dir, surface_ids=None,
                              explained=None, scale=1.0,
                              masks=None) -> List[DocumentElement]:
        """Rasterises whatever no surface or text run explained.

        This is the deliberate fallback for complex artwork -- gradients, mascots,
        photographs, multi-colour icons. Trying to express those as CSS would be a
        guess; keeping the original pixels is the honest answer.
        """
        # Paint what the CSS boxes will actually produce, then keep only the pixels that
        # prediction gets wrong. Subtracting surface *areas* instead would erase every
        # piece of artwork sitting on top of a card, and would silently hide a surface
        # that was painted the wrong colour.
        predicted = np.full_like(img, mean_bg, dtype=np.uint8)
        edges = np.zeros((h, w), np.uint8)
        for s in sorted(paintable, key=lambda s: -(s.width * s.height)):
            if s.color is None:
                continue
            # Paint against where the mask was observed, not the (possibly larger)
            # rendered box, or a clipped shape's mask gets stretched across it.
            x0, y0, x1, y1 = [int(v) for v in s.vis_bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            m = s.mask
            if m.shape != (y1 - y0, x1 - x0):
                m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            if s.shadow:
                # Approximate the CSS shadow into the prediction, so the darkening it
                # accounts for is no longer treated as unexplained artwork.
                sdx, sdy, sblur, salpha = s.shadow
                pad = int(min(sblur * 2 + abs(sdx) + abs(sdy) + 4, 60))
                sx0, sy0 = max(0, x0 - pad), max(0, y0 - pad)
                sx1, sy1 = min(w, x1 + pad), min(h, y1 + pad)
                layer = np.zeros((sy1 - sy0, sx1 - sx0), np.float32)
                ty0, tx0 = int(y0 - sy0 + sdy), int(x0 - sx0 + sdx)
                mh2 = min(m.shape[0], layer.shape[0] - max(ty0, 0))
                mw2 = min(m.shape[1], layer.shape[1] - max(tx0, 0))
                if mh2 > 0 and mw2 > 0 and ty0 >= 0 and tx0 >= 0:
                    layer[ty0:ty0 + mh2, tx0:tx0 + mw2] = (m[:mh2, :mw2] > 0).astype(np.float32)
                    k = max(int(sblur) | 1, 3)
                    layer = cv2.GaussianBlur(layer, (k, k), sblur / 2.0 + 0.5)
                    a = (layer * salpha)[:, :, None]
                    reg = predicted[sy0:sy1, sx0:sx1].astype(np.float32)
                    predicted[sy0:sy1, sx0:sx1] = np.clip(reg * (1.0 - a), 0, 255).astype(np.uint8)

            c = s.color.lstrip('#')
            bgr = np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], dtype=np.uint8)
            region = predicted[y0:y1, x0:x1]
            region[m > 0] = bgr
            outline = cv2.subtract(cv2.dilate(m, _K3), cv2.erode(m, _K3))
            np.maximum(edges[y0:y1, x0:x1], outline * 255, out=edges[y0:y1, x0:x1])

        # A painted box has hard edges; the source has antialiased ones. That one-pixel
        # rim always mismatches, and because it traces the whole outline its contour's
        # bounding box covers the entire control -- which is what put a raster crop on
        # top of every button. Exclude each surface's own boundary from the comparison.
        rim = cv2.dilate(edges, _K3, iterations=1) if edges is not None else None

        delta = cv2.absdiff(img, predicted)
        _, content = cv2.threshold(cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY), 16, 255, cv2.THRESH_BINARY)
        # Text is drawn as live text, so its ink is already accounted for.
        ink = cv2.dilate(text_mask, _K3, iterations=1)
        if rim is not None:
            ink = cv2.bitwise_or(ink, rim)
        if explained is not None:
            # Already kept verbatim as a backdrop; re-cropping it would stack a second
            # copy of the same pixels on top of the first.
            ink = cv2.bitwise_or(ink, explained)
        residual = cv2.bitwise_and(content, cv2.bitwise_not(ink))
        residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, _K3)

        grouped = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, scaled_kernel(scale, 7))
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Closing welds a control's arrow glyph, its drop shadow and its antialiased rim
        # into one contour whose box spans the whole control while being almost entirely
        # transparent. Emitted as a rectangle that lands on top of the real button and
        # eats its clicks. A sparse group is therefore split back into its parts.
        boxes: List[Tuple[int, int, int, int]] = []
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 3 or ch < 3:
                continue
            ink = int(np.count_nonzero(residual[y:y+ch, x:x+cw]))
            if ink and (ink / float(cw * ch)) < 0.35 and (cw * ch) > 2500 * scale * scale:
                n, lab, stats, _ = cv2.connectedComponentsWithStats(
                    residual[y:y+ch, x:x+cw], 8)
                for i in range(1, n):
                    sx, sy = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
                    sw, sh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                    if stats[i, cv2.CC_STAT_AREA] >= 12 * scale * scale:
                        boxes.append((x + sx, y + sy, sw, sh))
            else:
                boxes.append((x, y, cw, ch))

        # Innermost surface containing a piece of artwork becomes its parent, so the
        # artwork renders inside that component instead of covering it.
        # Containment, not paint: a surface that was stopped from painting is still the
        # box its contents live in. Filtering these searches on colour sent 58 of the
        # SaaS page's elements back to the page root the moment one over-claiming fill
        # was silenced.
        nest = sorted(paintable, key=lambda s: (s.width * s.height))

        out: List[DocumentElement] = []
        idx = 1
        for (x, y, cw, ch) in boxes:
            if cw < 3 or ch < 3:
                continue
            ink = int(np.count_nonzero(residual[y:y+ch, x:x+cw]))
            # Small is fine -- icons are small. Sparse noise is not.
            if ink < 40 * scale * scale or (cw * ch) < 40 * scale * scale:
                continue
            if cw > w * 0.985 and ch > h * 0.985:
                continue

            crop = img[y:y+ch, x:x+cw]
            local = residual[y:y+ch, x:x+cw]
            if crop.size == 0:
                continue

            # Text was subtracted from the residual so it would not be rasterised twice.
            # But the mask subtracted is the *dilated* ink, which is wider than the
            # glyph it protects, so what is left is a letter-shaped hole a size too big.
            # Live text fills the letter and not the gap around it, and the page shows
            # through the difference as a grey halo behind every heading -- read, quite
            # reasonably, as broken drop shadows. Close those gaps and paint the type
            # out of the pixels instead, which is what backdrops have always done.
            gap = (_fill_holes((local > 0).astype(np.uint8)) > 0) & (local == 0)
            local_ink = text_mask[y:y+ch, x:x+cw]
            gap &= cv2.dilate(local_ink, _K3,
                              iterations=max(2, int(round(3 * scale)))) > 0
            if np.any(gap):
                crop = erase_text_from_crop(crop, local_ink, textured=False, scale=scale)
                local = local.copy()
                local[gap] = 255

            art_bbox = [float(x), float(y), float(x + cw), float(y + ch)]
            parent_id = None
            for s in nest:
                if _contains(s.bbox, art_bbox, pad=2.0):
                    parent_id = (surface_ids or {}).get(id(s))
                    break

            flat = flat_residual_fill(crop, local, cw, ch)
            if flat is not None:
                colour, radius = flat
                out.append(DocumentElement(
                    id=f"fill-{idx}", type='rect', bbox=art_bbox,
                    tag='div', role='surface',
                    box=BoxStyle(backgroundColor=colour,
                                 borderRadius=(f"{radius:.1f}px" if radius >= 1.0
                                               else None)),
                    confidence=0.6, parentId=parent_id,
                    zIndex=Z_SURFACE + Z_ART,
                ))
                idx += 1
                continue

            bgra = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = np.where(local > 0, 255, 0).astype(np.uint8)

            ok, enc = cv2.imencode('.png', bgra)
            if not ok:
                continue
            src = f"data:image/png;base64,{base64.b64encode(enc.tobytes()).decode('utf-8')}"
            asset_name = f"art_{idx}.png"
            if asset_dir:
                os.makedirs(asset_dir, exist_ok=True)
                cv2.imwrite(os.path.join(asset_dir, asset_name), bgra)

            if masks is not None:
                # An artwork crop is mostly transparent; only its ink hides anything.
                masks[f"art-{idx}"] = (local > 0).astype(np.uint8)
            out.append(DocumentElement(
                id=f"art-{idx}", type='image',
                bbox=art_bbox,
                src=src, assetName=asset_name,
                naturalWidth=float(cw), naturalHeight=float(ch),
                tag='img', role='artwork',
                parentId=parent_id,
                zIndex=Z_SURFACE + Z_ART,
            ))
            idx += 1
        return out

    @staticmethod
    def reconstruct_image_from_cv2(
        img: np.ndarray,
        asset_dir: Optional[str] = None,
        page_num: int = 1,
    ) -> PageData:
        h, w = img.shape[:2]
        # Pixel thresholds below describe features of a rendered page, not of this file,
        # so they are expressed against a reference capture width and scaled to whatever
        # this one happens to be.
        scale = page_scale(w, h)

        # 1. Original pixels, kept for fidelity comparison and as the ultimate fallback
        _, enc_img = cv2.imencode('.png', img)
        orig_img_b64 = f"data:image/png;base64,{base64.b64encode(enc_img.tobytes()).decode('utf-8')}"

        # 2. Page ground colour (median of the border ring)
        borders = np.concatenate([img[0:8, :], img[-8:, :], img[:, 0:8], img[:, -8:]], axis=None).reshape(-1, 3)
        mean_bg = np.median(borders, axis=0).astype(np.uint8)
        page_bg_hex = _bgr_to_hex(mean_bg)

        # 3. Text runs, with sizes fitted to real ink bounds
        text_runs, text_mask = ImageReconstructor._extract_text_runs(img)

        # The family was decided inside _extract_text_runs, before the sizes were
        # fitted against it, and is already on every style.

        # 4. What kind of content is where, before asking what components are in it
        dilated_ink = cv2.dilate((text_mask > 0).astype(np.uint8), _K3,
                                 iterations=max(1, int(round(2 * scale))))
        photo_raw, photo_joined = photographic_mask(img, dilated_ink, scale=scale)
        photo_boxes = photographic_regions(photo_joined, float(w * h))

        # 5. Flat fills -> candidate CSS surfaces
        surfaces = detect_surfaces(img, page_bg_hex, text_ink=dilated_ink,
                                   photo_mask=photo_raw, photo_region=photo_joined,
                                   scale=scale)
        paintable = [s for s in surfaces if s.shape in ('rect', 'ellipse')]

        # A bordered control arrives as two regions: a ring and the fill inside it.
        # Fold the ring into the fill so it becomes one element with a real border.
        paintable = ImageReconstructor._merge_borders(paintable)
        paintable = ImageReconstructor._dedupe_surfaces(paintable)

        build_containment(paintable)
        drop_overclaimed_fills(paintable, img, dilated_ink, photo_joined)
        assign_texts(paintable, text_runs)

        for s in paintable:
            ground = s.parent.color if (s.parent and s.parent.color) else page_bg_hex
            s.shadow = detect_box_shadow(img, s, ground, scale=scale)

        # 5. Decide what each surface is, then let those roles inform the text roles
        ctx = {'img': img, 'text_ink': (text_mask > 0).astype(np.uint8), 'page_bg': page_bg_hex}
        roles: Dict[int, Tuple[str, str, float]] = {}
        for s in paintable:
            roles[id(s)] = classify_surface_role(s, ctx)
            for run in s.texts:
                run['host_role'] = roles[id(s)][0]

        cluster_font_sizes(text_runs)
        cluster_text_colors(text_runs)
        assign_heading_levels(text_runs)

        # Repeated elements are classified as a set. A member's own box grows to take in
        # its leading icon, because an icon and its label are one control, not two.
        text_ink_mask = (text_mask > 0).astype(np.uint8)
        sibling_groups = detect_sibling_groups(text_runs, img, text_ink_mask, h)
        for g in sibling_groups:
            if g['role'] == 'group-item':
                # Repetition alone only buys consistency among members that have no
                # semantic tag yet. Stacked headlines share a left edge and a rhythm
                # like any column group, and overwriting them here demoted every
                # heading on the page back to a span.
                for r in g['members']:
                    if not r.get('tag'):
                        r['tag'] = 'span'
                continue
            for r, mark in zip(g['members'], g['marks']):
                r['tag'], r['role'] = g['tag'], g['role']
                r['grouped'] = True
                if mark:
                    b = r['bbox']
                    new_x0 = min(b[0], mark[0])
                    r['style'].paddingLeft = max(0.0, b[0] - new_x0)
                    r['bbox'] = [new_x0, min(b[1], mark[1]),
                                 max(b[2], mark[2]), max(b[3], mark[3])]

        elements: List[DocumentElement] = []
        paint_masks: Dict[str, np.ndarray] = {}
        counters: Dict[str, int] = {}

        def next_id(kind: str) -> str:
            counters[kind] = counters.get(kind, 0) + 1
            return f"{kind}-{counters[kind]}"

        # 6. Emit surfaces as real boxes, absorbing a button's label into the button
        surface_ids: Dict[int, str] = {}
        for s in sorted(paintable, key=lambda s: -(s.width * s.height)):
            role, tag, conf = roles[id(s)]

            eid = next_id(role if role in ('button', 'badge', 'card', 'input', 'switch') else 'surface')
            surface_ids[id(s)] = eid

            box = BoxStyle(
                backgroundColor=s.color,
                borderColor=s.border_color,
                borderWidth=s.border_width,
                borderRadius=(f"{s.radius:.1f}px" if s.radius >= 1.0 else None),
            )
            if s.shape == 'ellipse':
                box.borderRadius = "50%"
            if s.shadow:
                sdx, sdy, sblur, salpha = s.shadow
                box.boxShadow = (f"{sdx:.1f}px {sdy:.1f}px {sblur:.1f}px "
                                 f"rgba(0, 0, 0, {salpha:.3f})")

            elem = DocumentElement(
                id=eid,
                type='rect',
                bbox=[float(v) for v in s.bbox],
                tag=tag,
                role=role,
                box=box,
                confidence=conf,
                parentId=surface_ids.get(id(s.parent)) if s.parent is not None else None,
                zIndex=(Z_SURFACE
                        + (Z_CONTROL if role in ('button', 'badge', 'input', 'switch') else 0)),
            )

            # An <input> is void, so its label has to become the placeholder attribute.
            # Every other control keeps its label as a nested child at the exact
            # coordinates it was measured at: the label really is inside the <button>,
            # but naming the surface never moves a pixel.
            if role == 'input' and len(s.texts) == 1:
                run = s.texts[0]
                elem.text = run['text']
                elem.style = run['style']
                run['consumed'] = True

            elements.append(elem)

        # 7. Tables, then paragraphs, then whatever text is left over
        for table_rows in detect_text_lattice(text_runs):
            cells = [r for row in table_rows for r in row]
            tx0 = min(r['bbox'][0] for r in cells)
            ty0 = min(r['bbox'][1] for r in cells)
            tx1 = max(r['bbox'][2] for r in cells)
            ty1 = max(r['bbox'][3] for r in cells)
            n_cols = len(table_rows[0])
            widths = []
            for c in range(n_cols):
                col = [row[c] for row in table_rows]
                widths.append(max(r['bbox'][2] for r in col) - min(r['bbox'][0] for r in col))
            total = sum(widths) or 1.0

            parts = ['<table style="width:100%;height:100%;border-collapse:collapse;table-layout:fixed;">',
                     '<colgroup>']
            parts += [f'<col style="width:{(cw / total) * 100.0:.2f}%;">' for cw in widths]
            parts.append('</colgroup><tbody>')
            for row in table_rows:
                parts.append('<tr>')
                for cell in row:
                    st = cell['style']
                    parts.append(
                        f'<td style="padding:2px 4px;vertical-align:middle;'
                        f'font-size:{st.fontSize:.2f}px;font-weight:{st.fontWeight};'
                        f'color:{st.color};">{html.escape(cell["text"])}</td>'
                    )
                    cell['consumed'] = True
                parts.append('</tr>')
            parts.append('</tbody></table>')

            elements.append(DocumentElement(
                id=next_id('table'), type='table',
                bbox=[float(tx0), float(ty0), float(tx1), float(ty1)],
                tag='table', role='table', rows=len(table_rows), cols=n_cols,
                html=''.join(parts), confidence=0.7, zIndex=Z_SURFACE + Z_TABLE,
            ))

        for group in group_paragraphs(text_runs):
            if any(r.get('consumed') for r in group):
                continue
            px0 = min(r['bbox'][0] for r in group)
            py0 = min(r['bbox'][1] for r in group)
            px1 = max(r['bbox'][2] for r in group)
            py1 = max(r['bbox'][3] for r in group)
            lead = group[0]
            host = lead.get('host')

            # The paragraph is a real <p>, but its lines keep the positions they were
            # measured at. Joining the text and letting the browser re-wrap it breaks
            # at different points than the original -- the substitute face is not the
            # same width -- so the block gained a line and overprinted whatever sat
            # below it. Each line stays a placed child instead.
            para_id = next_id('para')
            elements.append(DocumentElement(
                id=para_id, type='text',
                bbox=[float(px0), float(py0), float(px1), float(py1)],
                tag='p', role='paragraph', confidence=0.75,
                parentId=surface_ids.get(id(host)) if host is not None else None,
                zIndex=Z_SURFACE + Z_TEXT,
            ))
            for r in group:
                r['consumed'] = True
                elements.append(DocumentElement(
                    id=next_id('text'), type='text',
                    bbox=[float(v) for v in r['bbox']],
                    text=r['text'], tag='span', role='paragraph-line',
                    style=r['style'], parentId=para_id,
                    spans=r.get('spans'),
                    zIndex=Z_SURFACE + Z_TEXT,
                ))

        for run in text_runs:
            if run.get('consumed'):
                continue
            host = run.get('host')
            elements.append(DocumentElement(
                id=next_id('text'), type='text',
                bbox=[float(v) for v in run['bbox']],
                text=run['text'],
                tag=run.get('tag') or 'span',
                role=run.get('role') or 'text',
                style=run['style'],
                spans=run.get('spans'),
                parentId=surface_ids.get(id(host)) if host is not None else None,
                zIndex=Z_SURFACE + Z_TEXT,
            ))

        # Photographic and gradient areas are kept whole, underneath the components
        # drawn on them. Text is erased from the raster because it is re-emitted live.
        #
        # "Underneath" was only ever true of controls. A promoted control is lifted by
        # Z_CONTROL and outranks a backdrop, but a card or a panel sits at the plain
        # surface depth and loses to every photograph laid over it -- 20 of the 43
        # scored components across the reference pages were detected and then painted
        # out of existence, two whole cards on the Reddit page among them. Raising the
        # containers instead would replace the photograph with their flat fill, which
        # trades a real defect for a worse one.
        #
        # So the photograph is cut around them: each swallowed container is handed the
        # slice of it that sits on that container, as a child, and that slice is erased
        # from the backdrop. The same pixels land in the same places, but they arrive
        # inside a real card instead of on top of one nothing ever renders.
        photo_ids = []
        containers = [e for e in elements
                      if e.type == 'rect' and (e.role or 'surface') not in CULLABLE_ROLES
                      and e.role not in ('button', 'badge', 'input')]
        b_idx = 0

        def _immediate(region, pool):
            """Containers inside `region` with no other candidate in between."""
            inside = [c for c in pool if _contains(region, c.bbox, pad=2.0)
                      and not _contains(c.bbox, region, pad=2.0)]
            return [c for c in inside
                    if not any(d is not c and _contains(d.bbox, c.bbox, pad=2.0)
                               and not _contains(c.bbox, d.bbox, pad=2.0)
                               for d in inside)]

        # Each entry is a rectangle of photograph still looking for an owner:
        # (x, y, w, h, parent element id, alpha mask or None when fully opaque).
        queue = [(px, py, pw, ph, None, None) for (px, py, pw, ph) in photo_boxes]
        for i, (px, py, pw, ph, _, _) in enumerate(list(queue)):
            for host in sorted(paintable, key=lambda s: s.width * s.height):
                if _contains(host.bbox, [float(px), float(py),
                                         float(px + pw), float(py + ph)], pad=3.0):
                    queue[i] = (px, py, pw, ph, surface_ids.get(id(host)), None)
                    break

        while queue:
            rx, ry, rw, rh, r_parent, r_alpha = queue.pop(0)
            if rw < 2 or rh < 2 or rx + rw > w or ry + rh > h:
                continue
            crop = img[ry:ry+rh, rx:rx+rw]
            if crop.size == 0:
                continue
            region = [float(rx), float(ry), float(rx + rw), float(ry + rh)]
            if r_alpha is not None:
                alpha = r_alpha.copy()
            else:
                # Crop the photograph to its own silhouette, not to its bounding box.
                # A diagonal truck, a rounded badge, an object photographed at an angle
                # -- each arrives as a rectangle up to twice its own size, and every
                # heading and paragraph inside that rectangle is painted over. On the
                # logistics page the two backdrops claimed half the page while only
                # 54% of what they claimed was photograph, burying 43,528 pixels of
                # type. The silhouette was already measured; it was being thrown away
                # in favour of a bounding box.
                shape = (photo_joined[ry:ry+rh, rx:rx+rw] > 0).astype(np.uint8)
                # Glyph tiles and smooth patches read as non-photographic, which leaves
                # holes in the middle of an object. Anything fully enclosed by the
                # photograph belongs to it: text sits *on* the picture and needs the
                # picture behind it.
                alpha = _fill_holes(shape) * 255

            for c in _immediate(region, containers):
                cm = _painted_mask(c)
                if cm is None:
                    continue
                cx0 = int(round(float(c.bbox[0])))
                cy0 = int(round(float(c.bbox[1])))
                ax0, ay0 = max(rx, cx0), max(ry, cy0)
                ax1 = min(rx + rw, cx0 + cm.shape[1])
                ay1 = min(ry + rh, cy0 + cm.shape[0])
                if ax1 - ax0 < 4 or ay1 - ay0 < 4:
                    continue
                piece = cm[ay0-cy0:ay1-cy0, ax0-cx0:ax1-cx0] > 0
                window = alpha[ay0-ry:ay1-ry, ax0-rx:ax1-rx]
                child_alpha = np.where(piece, window, 0).astype(np.uint8)
                if not np.any(child_alpha):
                    continue              # an ancestor already claimed these pixels
                window[piece] = 0
                queue.append((ax0, ay0, ax1-ax0, ay1-ay0, c.id, child_alpha))

            if not np.any(alpha):
                continue                  # handed entirely to the containers inside it
            ink = text_mask[ry:ry+rh, rx:rx+rw]
            if np.any(ink):
                crop = erase_text_from_crop(crop, ink, textured=True, scale=scale)
            if bool(np.all(alpha)):
                out_png = crop            # nothing was cut out; stay three-channel
            else:
                out_png = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
                out_png[:, :, 3] = alpha
                # Colour under a hole is never drawn, but PNG still stores it, and a
                # carved backdrop would keep a full copy of every card cut out of it.
                # Flatten the dead area so the encoder can run it away to nothing.
                out_png[:, :, :3][alpha == 0] = 0
            ok, enc = cv2.imencode('.png', out_png)
            if not ok:
                continue
            b_idx += 1
            asset_name = f"backdrop_{b_idx}.png"
            if asset_dir:
                os.makedirs(asset_dir, exist_ok=True)
                cv2.imwrite(os.path.join(asset_dir, asset_name), out_png)
            eid = f"backdrop-{b_idx}"
            photo_ids.append(eid)
            paint_masks[eid] = (alpha > 0).astype(np.uint8)
            elements.append(DocumentElement(
                id=eid, type='image',
                bbox=[float(rx), float(ry), float(rx + rw), float(ry + rh)],
                src=f"data:image/png;base64,{base64.b64encode(enc.tobytes()).decode('utf-8')}",
                assetName=asset_name,
                naturalWidth=float(rw), naturalHeight=float(rh),
                tag='img', role='backdrop', parentId=r_parent,
                zIndex=Z_SURFACE + Z_BACKDROP,
            ))

        # 8. Residual artwork: everything no surface or text explained stays as pixels
        elements.extend(ImageReconstructor._extract_residual_art(
            img, w, h, mean_bg, text_mask, paintable, asset_dir, surface_ids,
            explained=photo_joined if photo_boxes else None, scale=scale,
            masks=paint_masks
        ))

        elements.sort(key=lambda e: e.zIndex or 1)
        elements = cull_occluded(elements, w, h, paint_masks, scale=scale)

        return PageData(
            pageNumber=page_num,
            width=float(w),
            height=float(h),
            isScanned=True,
            elements=elements,
            originalImageSrc=orig_img_b64,
            backgroundColor=page_bg_hex
        )

# ==============================================================================
# 8b. SCANNED EXTRACTOR
# ==============================================================================

class ScannedExtractor:
    @staticmethod
    def extract_page(doc: pymupdf.Document, page_num: int, asset_dir: Optional[str] = None) -> PageData:
        page = doc[page_num]

        # 1. Check if the page is an image wrapper (e.g. single large image covering the page)
        imgs = page.get_images()
        if len(imgs) == 1:
            try:
                xref = imgs[0][0]
                base_img = doc.extract_image(xref)
                img_bytes = base_img["image"]
                cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if cv_img is not None and cv_img.shape[0] > 100 and cv_img.shape[1] > 100:
                    return ImageReconstructor.reconstruct_image_from_cv2(
                        cv_img, asset_dir=asset_dir, page_num=page_num + 1
                    )
            except Exception as e:
                logger.warning(f"Could not extract direct image from scanned PDF page: {e}")

        # 2. Render page at 150 DPI for high-res reconstruction
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        return ImageReconstructor.reconstruct_image_from_cv2(
            cv_img, asset_dir=asset_dir, page_num=page_num + 1
        )

# ==============================================================================
# 9. HTML RENDERER & EXPORTER
# ==============================================================================

BASE_PAGE_CSS = """/* PDF Reconstructed Document Styling */
* { box-sizing: border-box; }
body.reconstructed-doc {
    margin: 0; padding: 24px; background-color: #525659;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    display: flex; flex-direction: column; align-items: center; gap: 24px;
}
.pdf-page {
    position: relative; background-color: #ffffff; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    overflow: hidden; transform-origin: top center; user-select: text; -webkit-user-select: text; margin: 0 auto;
    color: #000000 !important;
}
.pdf-element { box-sizing: border-box; color: #000000; }
.pdf-text { cursor: text; outline: none; white-space: nowrap; overflow: visible; user-select: text; }
.pdf-text[contenteditable="true"]:focus { outline: 1px dashed #3b82f6; background-color: rgba(59, 130, 246, 0.05); }
.pdf-image-container { user-select: none; }
.pdf-image-container[data-role="artwork"] { pointer-events: none; }
.pdf-image-container img { display: block; width: 100%; height: 100%; object-fit: fill; pointer-events: none; }
.pdf-table-container { overflow: hidden; user-select: text; color: #000000 !important; }
.reconstructed-table {
    width: 100%; height: 100%; border-collapse: collapse;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    color: #000000; background-color: transparent;
    table-layout: fixed;
}
.reconstructed-table th, .reconstructed-table td {
    color: inherit; border: 1px solid #333333;
    padding: 2px 6px; box-sizing: border-box;
}
.reconstructed-table th {
    background-color: #f1f5f9; font-weight: 700;
}
.pdf-table-container table { width: 100%; height: 100%; border-collapse: collapse; table-layout: fixed; }
.pdf-table-container th, .pdf-table-container td { outline: none; }
.pdf-table-container th[contenteditable="true"]:focus, .pdf-table-container td[contenteditable="true"]:focus {
    outline: 2px solid #3b82f6; background-color: rgba(59, 130, 246, 0.08);
}
.pdf-vector-container { pointer-events: none; }
.pdf-formula-container { display: flex; align-items: center; justify-content: center; }
.pdf-custom-container { width: 100%; height: 100%; overflow: hidden; }
/* Reconstructed surfaces are real elements (button, h1, p, div), so the UA defaults
   those tags carry -- margins, padding, borders, system button chrome -- have to be
   cleared or they fight the measured geometry. */
.pdf-rect { margin: 0; padding: 0; border: 0 none; background: none; appearance: none;
    -webkit-appearance: none; font: inherit; text-decoration: none; }
button.pdf-rect, .pdf-rect button { cursor: pointer; }
.pdf-text { margin: 0; padding: 0; }
h1.pdf-text, h2.pdf-text, h3.pdf-text, h4.pdf-text, h5.pdf-text, h6.pdf-text,
p.pdf-text { margin: 0; padding: 0; font-weight: inherit; font-size: inherit; }
button.pdf-text, a.pdf-text, span.pdf-text {
    background: none; border: 0 none; appearance: none; -webkit-appearance: none;
    font: inherit; color: inherit; text-decoration: none; text-align: inherit;
}
button.pdf-text { cursor: pointer; }
@media print {
    body.reconstructed-doc { padding: 0; background: transparent; gap: 0; }
    .pdf-page { box-shadow: none; page-break-after: always; break-after: page; margin: 0; }
}"""

INTERACTIVE_DEMO_SNIPPET = """
<div id="ramen-snackbar" role="status" aria-live="polite"></div>
<div id="ramen-hl" aria-hidden="true"><div id="ramen-hl-box"></div><div id="ramen-hl-tip"></div></div>
<button id="ramen-inspect-toggle" type="button" title="Toggle element inspector (press i)">Inspect</button>
<style>
#ramen-snackbar {
    position: fixed; left: 50%; bottom: 28px; transform: translate(-50%, 16px);
    background: #101828; color: #fff; padding: 12px 18px; border-radius: 10px;
    font: 500 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    box-shadow: 0 10px 30px rgba(0,0,0,.28); opacity: 0; pointer-events: none;
    transition: opacity .18s ease, transform .18s ease; z-index: 2147483647; max-width: 78vw;
}
#ramen-snackbar.show { opacity: 1; transform: translate(-50%, 0); }
#ramen-snackbar b { color: #7cc4ff; }
#ramen-inspect-toggle {
    position: fixed; right: 18px; bottom: 18px; z-index: 2147483647;
    background: #101828; color: #cbd5e1; border: 1px solid #334155; border-radius: 8px;
    padding: 8px 14px; cursor: pointer;
    font: 600 12px/1 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
#ramen-inspect-toggle.on { background: #2563eb; color: #fff; border-color: #2563eb; }
#ramen-hl { position: absolute; inset: 0; pointer-events: none; z-index: 2147483646; display: none; }
#ramen-hl.on { display: block; }
#ramen-hl-box {
    position: absolute; border: 1px solid #2563eb; background: rgba(37,99,235,.16);
    border-radius: 2px; transition: all .05s linear;
}
#ramen-hl-tip {
    position: absolute; background: #101828; color: #e5e7eb; padding: 5px 9px;
    border-radius: 6px; white-space: nowrap; box-shadow: 0 6px 18px rgba(0,0,0,.35);
    font: 500 11.5px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
#ramen-hl-tip .t { color: #7cc4ff; } #ramen-hl-tip .r { color: #fbbf24; }
#ramen-hl-tip .d { color: #94a3b8; }
</style>
<script>
(function () {
    var bar = document.getElementById('ramen-snackbar'), timer = null;
    function toast(msg) {
        bar.innerHTML = msg;
        bar.classList.add('show');
        clearTimeout(timer);
        timer = setTimeout(function () { bar.classList.remove('show'); }, 2600);
    }

    // ---- Proof a reconstructed control is a control -------------------------------
    // Reports the tag the browser actually dispatched a real click to. If these were
    // pictures of buttons, nothing would fire.
    document.addEventListener('click', function (ev) {
        // Resolve the control, not the label inside it: a button's text is its own
        // element with its own data-role, so the nearest [data-role] is the span.
        var el = ev.target.closest('button, input, select, textarea, a')
              || ev.target.closest('[data-role]');
        if (!el) { return; }
        var role = el.getAttribute('data-role') || el.tagName.toLowerCase();
        if (role === 'artwork' || role === 'surface') { return; }
        var label = (el.innerText || el.placeholder || '').trim().slice(0, 40);
        toast('clicked &lt;' + el.tagName.toLowerCase() + '&gt; role=<b>' + role + '</b>'
              + (label ? ' \u2014 \u201c' + label + '\u201d' : ''));
    });

    // ---- Element inspector --------------------------------------------------------
    var hl = document.getElementById('ramen-hl');
    var box = document.getElementById('ramen-hl-box');
    var tip = document.getElementById('ramen-hl-tip');
    var btn = document.getElementById('ramen-inspect-toggle');
    var on = false, current = null;

    function setOn(v) {
        on = v;
        hl.classList.toggle('on', on);
        btn.classList.toggle('on', on);
        if (!on) { current = null; }
    }
    btn.addEventListener('click', function (e) { e.stopPropagation(); setOn(!on); });
    document.addEventListener('keydown', function (e) {
        if (e.key === 'i' && !/^(INPUT|TEXTAREA)$/.test(document.activeElement.tagName)) { setOn(!on); }
        if (e.key === 'Escape') { setOn(false); }
    });

    function describe(el) {
        var cs = getComputedStyle(el);
        var r = el.getBoundingClientRect();
        var role = el.getAttribute('data-role') || '';
        var bits = [];
        bits.push('<span class="t">&lt;' + el.tagName.toLowerCase() + '&gt;</span>');
        if (role) { bits.push('<span class="r">' + role + '</span>'); }
        bits.push('<span class="d">' + Math.round(r.width) + '\u00d7' + Math.round(r.height) + '</span>');
        var bg = cs.backgroundColor;
        if (bg && bg !== 'rgba(0, 0, 0, 0)') { bits.push(bg.replace(/\\s+/g, '')); }
        if (parseFloat(cs.borderTopLeftRadius) > 0) { bits.push('r' + cs.borderTopLeftRadius); }
        if (parseFloat(cs.borderTopWidth) > 0) { bits.push('bd ' + cs.borderTopWidth); }
        if (cs.boxShadow && cs.boxShadow !== 'none') { bits.push('shadow'); }
        if (el.classList.contains('pdf-text')) {
            bits.push(Math.round(parseFloat(cs.fontSize)) + 'px ' + cs.fontWeight);
        }
        return bits.join(' \u00b7 ');
    }

    document.addEventListener('mousemove', function (ev) {
        if (!on) { return; }
        var el = document.elementFromPoint(ev.clientX, ev.clientY);
        el = el && el.closest('.pdf-element');
        if (!el) { box.style.display = 'none'; tip.style.display = 'none'; current = null; return; }
        if (el !== current) { current = el; tip.innerHTML = describe(el); }
        var r = el.getBoundingClientRect();
        var sx = window.scrollX, sy = window.scrollY;
        box.style.display = 'block';
        box.style.left = (r.left + sx) + 'px';
        box.style.top = (r.top + sy) + 'px';
        box.style.width = r.width + 'px';
        box.style.height = r.height + 'px';
        tip.style.display = 'block';
        var above = r.top > 26;
        tip.style.left = (r.left + sx) + 'px';
        tip.style.top = (above ? (r.top + sy - 24) : (r.bottom + sy + 6)) + 'px';
    }, true);

    window.addEventListener('load', function () {
        var n = document.querySelectorAll('button').length;
        var h = document.querySelectorAll('h1,h2,h3,h4,h5,h6').length;
        toast('Reconstructed page ready \u2014 <b>' + n + '</b> real &lt;button&gt; and <b>'
              + h + '</b> heading elements. Click one, or press <b>i</b> to inspect.');
    });
})();
</script>
"""


class HTMLRenderer:
    VOID_TAGS = {'input', 'img', 'br', 'hr'}

    @staticmethod
    def _box_css(box: Optional[BoxStyle]) -> str:
        if box is None:
            return ""
        parts = []
        if box.backgroundColor:
            parts.append(f"background-color: {box.backgroundColor};")
        if box.backgroundImage:
            parts.append(f"background-image: {box.backgroundImage};")
        if box.borderColor and box.borderWidth and box.borderWidth > 0:
            parts.append(f"border: {box.borderWidth:.1f}px {box.borderStyle} {box.borderColor};")
        if box.borderRadius:
            parts.append(f"border-radius: {box.borderRadius};")
        if box.boxShadow:
            parts.append(f"box-shadow: {box.boxShadow};")
        return " ".join(parts)

    @staticmethod
    def _text_css(style: Optional[TextStyle], default_align: str = 'left') -> str:
        s = style
        ff = s.fontFamily if (s and s.fontFamily) else "'Segoe UI', Arial, sans-serif"
        fs = float(s.fontSize if (s and s.fontSize is not None) else 12.0)
        fw = s.fontWeight if (s and s.fontWeight) else "normal"
        fst = s.fontStyle if (s and s.fontStyle) else "normal"
        col = s.color if (s and s.color) else "#000000"
        ta = (s.textAlign if (s and s.textAlign) else default_align)
        lh = float(s.lineHeight if (s and s.lineHeight is not None) else 1.0)
        raw_ls = float(s.letterSpacing) if (s and s.letterSpacing) else 0.0
        ls = f"letter-spacing: {raw_ls:.2f}px; " if abs(raw_ls) > 0.01 else ""
        raw_ws = float(s.wordSpacing) if (s and s.wordSpacing) else 0.0
        ws = f"word-spacing: {raw_ws:.2f}px; " if abs(raw_ws) > 0.01 else ""
        bg = f"background-color: {s.backgroundColor};" if (s and s.backgroundColor) else ""
        pl = float(s.paddingLeft) if (s and s.paddingLeft) else 0.0

        # A run set in more than one colour is painted with a gradient of hard stops,
        # clipped to the glyphs. The alternative is to split the text where the colour
        # changes, which needs to know which character it changes on -- and the pixels
        # do not say: the letters merge, so there is no mapping from marks to
        # characters. The position is known exactly, so it is used as a position.
        # Skipped when the run has its own background, which background-clip would
        # otherwise cut to the letters as well.
        stops = (s.colorStops if (s and s.colorStops) else None)
        paint = f"color: {col}; "
        if stops and len(stops) >= 2 and not bg:
            parts = []
            for i, (at, hexed) in enumerate(stops):
                start = max(0.0, min(1.0, float(at))) * 100.0
                end = (max(0.0, min(1.0, float(stops[i + 1][0]))) * 100.0
                       if i + 1 < len(stops) else 100.0)
                parts.append(f"{hexed} {start:.2f}%, {hexed} {end:.2f}%")
            paint = ("color: transparent; -webkit-text-fill-color: transparent; "
                     f"background-image: linear-gradient(90deg, {', '.join(parts)}); "
                     "-webkit-background-clip: text; background-clip: text; ")

        return (f"font-family: {ff}; font-size: {fs:.2f}px; font-weight: {fw}; "
                f"font-style: {fst}; {paint}text-align: {ta}; "
                f"line-height: {lh:.3f}; {ls}{ws}margin: 0; padding: 0 0 0 {pl:.2f}px; {bg}")

    @staticmethod
    def render_element(elem: DocumentElement, editable: bool = True,
                       origin: Tuple[float, float] = (0.0, 0.0),
                       children: Optional[Dict[str, List[DocumentElement]]] = None) -> str:
        bbox = elem.bbox or [0, 0, 10, 10]
        x0 = float(bbox[0] if len(bbox) > 0 and bbox[0] is not None else 0.0)
        y0 = float(bbox[1] if len(bbox) > 1 and bbox[1] is not None else 0.0)
        x1 = float(bbox[2] if len(bbox) > 2 and bbox[2] is not None else x0 + 10.0)
        y1 = float(bbox[3] if len(bbox) > 3 and bbox[3] is not None else y0 + 10.0)
        width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        left, top = x0 - origin[0], y0 - origin[1]

        z_index = elem.zIndex or 1
        rot_deg = float(elem.rotation or 0.0)
        rot = f"transform: rotate({rot_deg}deg);" if rot_deg != 0.0 else ""
        elem_op = float(elem.opacity if elem.opacity is not None else 1.0)
        op_str = f"opacity: {elem_op:.2f}; " if elem_op < 0.999 else ""

        kids = (children or {}).get(elem.id, [])
        kids_html = "".join(
            HTMLRenderer.render_element(k, editable=editable, origin=(x0, y0), children=children)
            for k in kids
        )

        # A container must not open its own stacking context. With one, a descendant's
        # z-index is confined to it, so a headline at z=8 inside a panel at z=2 lost to
        # any sibling of that panel at z=3 -- an entire page of text disappeared behind
        # a backdrop that way. Leaving containers on `z-index: auto` lets every element
        # compete in one order, which is the order these values were chosen for.
        z_rule = f"z-index: {z_index}; " if not kids else ""
        common_style = (f"position: absolute; left: {left:.2f}px; top: {top:.2f}px; "
                        f"width: {width:.2f}px; height: {height:.2f}px; "
                        f"{z_rule}{op_str}{rot}")
        data_attrs = (f'id="{elem.id}" data-id="{elem.id}" data-type="{elem.type}"'
                      f'{f" data-role={chr(34)}{elem.role}{chr(34)}" if elem.role else ""}')

        # --- Real CSS boxes: buttons, cards, inputs, panels, shapes ---
        if elem.type == 'rect':
            tag = (elem.tag or 'div').lower()
            box_css = HTMLRenderer._box_css(elem.box)
            if elem.text is not None:
                # A control owns its label, so centre it inside the control itself.
                label_css = HTMLRenderer._text_css(elem.style, default_align='center')
                inner = (f"{common_style} {box_css} {label_css} display: flex; "
                         f"align-items: center; justify-content: center; "
                         f"margin: 0; padding: 0; cursor: pointer; white-space: nowrap;")
                if tag in HTMLRenderer.VOID_TAGS:
                    ph = html.escape(elem.text or "")
                    return (f'<input class="pdf-element pdf-rect" {data_attrs} '
                            f'placeholder="{ph}" style="{inner} justify-content: flex-start; '
                            f'padding-left: 10px; border-style: {elem.box.borderStyle if elem.box else "solid"};"/>')
                edit_attr = ' contenteditable="true" spellcheck="false"' if editable else ''
                return (f'<{tag} class="pdf-element pdf-rect" {data_attrs}{edit_attr} '
                        f'style="{inner}">{html.escape(elem.text)}</{tag}>')
            if tag in HTMLRenderer.VOID_TAGS:
                return f'<{tag} class="pdf-element pdf-rect" {data_attrs} style="{common_style} {box_css}"/>'
            return (f'<{tag} class="pdf-element pdf-rect" {data_attrs} '
                    f'style="{common_style} {box_css} margin: 0; padding: 0;">{kids_html}</{tag}>')

        if elem.type == 'text':
            tag = (elem.tag or 'div').lower()
            if tag in HTMLRenderer.VOID_TAGS:
                tag = 'div'
            # Paragraphs are allowed to wrap; single runs are positioned by their ink box
            # and must not reflow, or they would drift off the coordinates they were fitted to.
            wrap = "white-space: normal; overflow-wrap: break-word;" if tag == 'p' else "white-space: nowrap;"
            sx = float(elem.style.scaleX) if (elem.style and elem.style.scaleX) else 1.0
            squeeze = (f"transform: scaleX({sx:.4f}); transform-origin: left top; "
                       if abs(sx - 1.0) > 0.001 else "")
            text_style = (f"{common_style} {HTMLRenderer._text_css(elem.style)} "
                          f"{squeeze}{wrap} overflow: visible;")

            if elem.spans and len(elem.spans) > 1:
                base = elem.style
                ff = base.fontFamily if (base and base.fontFamily) else ""
                fs = float(base.fontSize if (base and base.fontSize is not None) else 12.0)
                fw = base.fontWeight if (base and base.fontWeight) else "normal"
                fst = base.fontStyle if (base and base.fontStyle) else "normal"
                col = base.color if (base and base.color) else "#000000"
                parts = []
                for sp in elem.spans:
                    s = sp.style
                    s_ff = f"font-family:{s.fontFamily};" if (s and s.fontFamily and s.fontFamily != ff) else ""
                    raw_s_fs = float(s.fontSize if (s and s.fontSize is not None) else fs)
                    s_fs = f"font-size:{raw_s_fs:.2f}px;" if abs(raw_s_fs - fs) > 0.5 else ""
                    s_fw = f"font-weight:{s.fontWeight};" if (s and s.fontWeight and s.fontWeight != fw) else ""
                    s_fst = f"font-style:{s.fontStyle};" if (s and s.fontStyle and s.fontStyle != fst) else ""
                    s_col = f"color:{s.color};" if (s and s.color and s.color != col) else ""
                    sp_inline = f"{s_ff}{s_fs}{s_fw}{s_fst}{s_col}"
                    style_attr = f' style="{sp_inline}"' if sp_inline else ""
                    parts.append(f'<span{style_attr}>{html.escape(sp.text or "")}</span>')
                content_html = "".join(parts)
            else:
                content_html = html.escape(elem.text or '')

            edit_attr = ' contenteditable="true" spellcheck="false"' if editable else ''
            return (f'<{tag} class="pdf-element pdf-text" {data_attrs}{edit_attr} '
                    f'style="{text_style}">{content_html}{kids_html}</{tag}>')

        if elem.type == 'image':
            src = elem.src or ''
            # Recovered artwork is decoration. Its container is a full rectangle even
            # when the pixels inside are mostly transparent, so leaving it clickable
            # lets a stray rim or drop shadow swallow the clicks of the control beneath.
            inert = ' pointer-events: none;' if elem.role == 'artwork' else ''
            return (f'<div class="pdf-element pdf-image-container" {data_attrs} '
                    f'style="{common_style}{inert}"><img src="{src}" alt="{elem.role or "Embedded Asset"}" '
                    f'loading="lazy" /></div>')

        if elem.type == 'table':
            content = elem.html or '<table><tr><td>Table</td></tr></table>'
            if editable:
                content = content.replace('<td', '<td contenteditable="true" spellcheck="false"').replace(
                    '<th', '<th contenteditable="true" spellcheck="false"')
            return (f'<div class="pdf-element pdf-table-container" {data_attrs} '
                    f'style="{common_style}">{content}</div>')

        if elem.type == 'vector':
            return (f'<div class="pdf-element pdf-vector-container" {data_attrs} '
                    f'style="{common_style}">{elem.svg or "<svg></svg>"}</div>')

        if elem.type == 'formula':
            formula_html = elem.renderedHtml or f'<div>{html.escape(elem.text or "")}</div>'
            return (f'<div class="pdf-element pdf-formula-container" {data_attrs} '
                    f'style="{common_style}">{formula_html}</div>')

        if elem.type == 'custom' or elem.html:
            return (f'<div class="pdf-element pdf-custom-container" {data_attrs} '
                    f'style="{common_style}">{elem.html or ""}</div>')
        return ''

    @staticmethod
    def render_page(page: PageData, editable: bool = True) -> str:
        w = float(page.width if page.width is not None else 595.0)
        h = float(page.height if page.height is not None else 842.0)
        bg = f" background-color: {page.backgroundColor};" if getattr(page, 'backgroundColor', None) else ""

        # Nest by parentId so a card really contains its contents in the markup.
        ids = {e.id for e in page.elements}
        children: Dict[str, List[DocumentElement]] = {}
        roots: List[DocumentElement] = []
        for e in page.elements:
            pid = getattr(e, 'parentId', None)
            if pid and pid in ids and pid != e.id:
                children.setdefault(pid, []).append(e)
            else:
                roots.append(e)

        rendered = [HTMLRenderer.render_element(e, editable=editable, children=children)
                    for e in roots]
        return (
            f'<div class="pdf-page" id="pdf-page-{page.pageNumber}" data-page="{page.pageNumber}" '
            f'data-scanned="{"true" if page.isScanned else "false"}" style="width: {w:.2f}px; height: {h:.2f}px;{bg}">\n'
            f'{"\n".join(rendered)}\n</div>'
        )

    @staticmethod
    def render_document(doc: DocumentData, editable: bool = False, title: Optional[str] = None,
                        interactive: Optional[bool] = None) -> str:
        doc_title = title or doc.title or "Reconstructed PDF Document"
        pages_html = "\n\n".join([HTMLRenderer.render_page(p, editable=editable) for p in doc.pages])
        # Default to on when the page actually contains promoted controls, so a
        # reconstruction that claims to have buttons ships the means to check that
        # claim. Pass interactive=False for a clean document.
        if interactive is None:
            interactive = any(e.role in ('button', 'badge', 'input')
                              for p in doc.pages for e in p.elements)
        demo = INTERACTIVE_DEMO_SNIPPET if interactive else ""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(doc_title)}</title>
    <style>
{embedded_font_css()}
{BASE_PAGE_CSS}
    </style>
</head>
<body class="reconstructed-doc">
{pages_html}
{demo}
</body>
</html>"""

class Exporter:
    @staticmethod
    def export_standalone_html(doc_data: DocumentData, interactive: Optional[bool] = None) -> str:
        return HTMLRenderer.render_document(doc_data, editable=False, interactive=interactive)

    @staticmethod
    def export_json(doc_data: DocumentData) -> str:
        return json.dumps(doc_data.model_dump(), indent=2, ensure_ascii=False)

    @staticmethod
    def export_zip_bundle(doc_data: DocumentData) -> bytes:
        zip_buffer = io.BytesIO()
        assets_map = {}

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            img_index = 1
            for p_idx, page in enumerate(doc_data.pages):
                for elem in page.elements:
                    if elem.type == 'image' and elem.src and elem.src.startswith('data:image'):
                        match = re.match(r'data:image/([a-zA-Z0-9\+\-\.]+);base64,(.*)', elem.src)
                        if match:
                            ext = match.group(1).replace('jpeg', 'jpg')
                            try:
                                img_bytes = base64.b64decode(match.group(2))
                                asset_filename = elem.assetName or f"page-{p_idx+1}-image-{img_index:02d}.{ext}"
                                asset_path = f"assets/{asset_filename}"
                                zip_file.writestr(f"document/{asset_path}", img_bytes)
                                assets_map[elem.id] = asset_path
                                img_index += 1
                            except Exception:
                                pass

            zip_file.writestr("document/document.json", json.dumps(doc_data.model_dump(), indent=2, ensure_ascii=False))
            zip_file.writestr("document/styles.css", BASE_PAGE_CSS.strip())

            exported_pages_html = []
            for page in doc_data.pages:
                # Swap embedded data URLs for the extracted asset files, then let
                # render_page rebuild the containment tree -- rendering elements flat
                # here would emit every nested child twice.
                relinked = page.model_copy(update={'elements': [
                    e.model_copy(update={'src': assets_map[e.id]})
                    if (e.type == 'image' and e.id in assets_map) else e
                    for e in page.elements
                ]})
                exported_pages_html.append(HTMLRenderer.render_page(relinked, editable=False))

            index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{doc_data.title or "Reconstructed PDF"}</title>
    <link rel="stylesheet" href="styles.css">
</head>
<body class="reconstructed-doc">
{"\n\n".join(exported_pages_html)}
</body>
</html>"""
            zip_file.writestr("document/index.html", index_html)

        zip_buffer.seek(0)
        return zip_buffer.getvalue()

# ==============================================================================
# 10. FIDELITY CHECKER
# ==============================================================================

class FidelityChecker:
    @staticmethod
    def compare_images(orig_img_bytes: bytes, recon_img_bytes: bytes) -> Dict[str, Any]:
        orig_cv = cv2.imdecode(np.frombuffer(orig_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        recon_cv = cv2.imdecode(np.frombuffer(recon_img_bytes, np.uint8), cv2.IMREAD_COLOR)
        if orig_cv is None or recon_cv is None:
            return {"error": "Failed to decode one or both comparison images"}

        orig_h, orig_w = orig_cv.shape[:2]
        recon_h, recon_w = recon_cv.shape[:2]
        if (orig_h, orig_w) != (recon_h, recon_w):
            interp = cv2.INTER_AREA if (recon_w >= orig_w and recon_h >= orig_h) else cv2.INTER_LINEAR
            recon_cv = cv2.resize(recon_cv, (orig_w, orig_h), interpolation=interp)

        orig_gray = cv2.cvtColor(orig_cv, cv2.COLOR_BGR2GRAY)
        recon_gray = cv2.cvtColor(recon_cv, cv2.COLOR_BGR2GRAY)

        score, _ = ssim(orig_gray, recon_gray, full=True)
        err = mse(orig_gray, recon_gray)

        diff_sub = cv2.absdiff(orig_gray, recon_gray)
        _, thresh = cv2.threshold(diff_sub, 30, 255, cv2.THRESH_BINARY)
        heatmap = cv2.applyColorMap(diff_sub, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(orig_cv, 0.6, heatmap, 0.4, 0)

        _, enc_diff = cv2.imencode(".png", blended)
        diff_b64 = f"data:image/png;base64,{base64.b64encode(enc_diff.tobytes()).decode('utf-8')}"

        diff_pixels = np.count_nonzero(thresh)
        total_pixels = orig_w * orig_h
        return {
            "ssim": round(float(score), 4),
            "similarityPercent": round(score * 100.0, 2),
            "mse": round(float(err), 2),
            "diffPixelPercent": round((diff_pixels / total_pixels) * 100.0, 2),
            "diffHeatmapSrc": diff_b64,
            "dimensions": {"width": orig_w, "height": orig_h}
        }

# ==============================================================================
# 11. MODEL MANAGER
# ==============================================================================

class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(base_dir, "models"))
        os.makedirs(self.model_dir, exist_ok=True)

    def check_pdf_engine(self) -> Dict[str, Any]:
        import importlib.util
        installed = importlib.util.find_spec('pymupdf') is not None
        return {
            "name": "PyMuPDF Engine",
            "installed": installed,
            "version": "1.28.2" if installed else None,
            "status": "Ready" if installed else "Missing",
            "description": "Native digital PDF parser, geometry extractor, and high-DPI rasterizer"
        }

    def check_ocr_model(self) -> Dict[str, Any]:
        import importlib.util
        installed = importlib.util.find_spec('paddleocr') is not None
        return {
            "name": "PaddleOCR (PP-OCRv6)",
            "installed": installed,
            "version": "3.x" if installed else None,
            "status": "Ready" if installed else "Not installed",
            "description": "Deterministic text recognition for scanned document regions"
        }

    def check_layout_model(self) -> Dict[str, Any]:
        import importlib.util
        installed = importlib.util.find_spec('paddleocr') is not None
        return {
            "name": "PP-DocLayout_plus-L",
            "architecture": "RT-DETR-L",
            "installed": installed,
            "version": "v3" if installed else None,
            "status": "Ready" if installed else "Not installed",
            "description": "Document layout detector identifying titles, tables, images, text, and formulas"
        }

    def check_table_models(self) -> Dict[str, Any]:
        import importlib.util
        installed = importlib.util.find_spec('paddleocr') is not None
        return {
            "name": "PP-StructureV3 Table Engine",
            "installed": installed,
            "version": "v3" if installed else None,
            "status": "Ready" if installed else "Not installed",
            "description": "Cell, row, column, and merged cell recognition for complex tables"
        }

    def get_all_status(self) -> Dict[str, Any]:
        pdf_stat = self.check_pdf_engine()
        ocr_stat = self.check_ocr_model()
        layout_stat = self.check_layout_model()
        table_stat = self.check_table_models()
        return {
            "modelDirectory": self.model_dir,
            "offlineReady": pdf_stat["installed"] and ocr_stat["installed"],
            "components": [pdf_stat, layout_stat, ocr_stat, table_stat]
        }

    def download_models(self) -> Dict[str, Any]:
        results = {}
        try:
            from paddleocr import PaddleOCR
            PaddleOCR(use_angle_cls=True, lang='en')
            results["ocr"] = "Successfully initialized"
        except Exception as e:
            results["ocr"] = f"Error or pending installation: {e}"
        return {"status": "completed", "details": results}
