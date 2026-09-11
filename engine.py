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
from typing import List, Optional, Dict, Any, Union, Literal, Tuple
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
    backgroundColor: Optional[str] = None
    opacity: float = 1.0

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

# Font stack the renderer applies to OCR'd text. The measurement faces below are
# ordered to match how a browser resolves it, so measured metrics track what is drawn.
OCR_FONT_STACK = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

_MEASURE_FACES = {
    False: ['segoeui.ttf', 'Roboto-Regular.ttf', 'Helvetica.ttc', 'arial.ttf',
            'DejaVuSans.ttf', 'LiberationSans-Regular.ttf'],
    True: ['segoeuib.ttf', 'Roboto-Bold.ttf', 'Helvetica-Bold.ttf', 'arialbd.ttf',
           'DejaVuSans-Bold.ttf', 'LiberationSans-Bold.ttf'],
}
_FONT_DIRS = [
    os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts'),
    '/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/truetype/liberation',
    '/usr/share/fonts', '/Library/Fonts', '/System/Library/Fonts',
]
_MEASURE_REF_SIZE = 100  # metrics scale linearly, so measure once and scale
_font_cache: Dict[bool, Any] = {}

def _get_measure_font(bold: bool):
    """Loads a TTF approximating the rendered font stack, at a fixed reference size."""
    if bold in _font_cache:
        return _font_cache[bold]
    from PIL import ImageFont
    font = None
    for face in _MEASURE_FACES[bold]:
        for d in _FONT_DIRS:
            p = os.path.join(d, face)
            if os.path.exists(p):
                try:
                    font = ImageFont.truetype(p, _MEASURE_REF_SIZE)
                    break
                except Exception:
                    continue
        if font is not None:
            break
    if font is None:
        logger.warning("No measurement font found; falling back to box-height heuristic.")
    _font_cache[bold] = font
    return font

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
        if r['fontSize'] <= clusters[-1][-1]['fontSize'] * (1.0 + tol):
            clusters[-1].append(r)
        else:
            clusters.append([r])

    for group in clusters:
        centre = float(np.median([r['fontSize'] for r in group]))
        for r in group:
            if abs(r['fontSize'] - centre) < 0.01:
                continue
            # Re-solve spacing and line-height against the snapped size so the run still
            # lands on the ink box it was measured from.
            b = r['bbox']
            _, ls, lh = fit_text_to_box(r['text'], b[2] - b[0], b[3] - b[1],
                                        r['fontWeight'] == 'bold', force_size=centre)
            r['fontSize'] = centre
            r['style'].fontSize = centre
            r['style'].letterSpacing = ls
            r['style'].lineHeight = lh

def cluster_text_colors(runs: List[Dict[str, Any]], tol: float = 26.0) -> None:
    """Snaps per-run sampled colours onto shared values.

    Colour is measured from the glyph pixels of each run, so the same ink comes back as
    #4a4a4a on one line and #4b494b on the next. Exact-match grouping then fails, and
    the CSS carries dozens of near-identical colours instead of a palette.
    """
    centres: List[Tuple[np.ndarray, List[Dict[str, Any]]]] = []
    for r in sorted(runs, key=lambda r: -(r['bbox'][2] - r['bbox'][0])):
        c = r['color'].lstrip('#')
        try:
            rgb = np.array([int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)], dtype=np.float64)
        except Exception:
            continue
        for centre, members in centres:
            if float(np.linalg.norm(rgb - centre)) <= tol:
                members.append(r)
                break
        else:
            centres.append((rgb, [r]))

    for centre, members in centres:
        hexed = f"#{int(centre[0]):02x}{int(centre[1]):02x}{int(centre[2]):02x}"
        for r in members:
            r['color'] = hexed
            r['style'].color = hexed

def fit_text_to_box(text: str, box_w: float, box_h: float, bold: bool,
                    force_size: Optional[float] = None) -> Tuple[float, float, float]:
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
    font = _get_measure_font(bold) if clean else None
    if font is None:
        return max(9.0, round(box_h * 0.82, 1)), 0.0, 1.2

    try:
        ink = font.getbbox(clean)
        ascent, descent = font.getmetrics()
    except Exception:
        return max(9.0, round(box_h * 0.82, 1)), 0.0, 1.2

    ink_h = float(ink[3] - ink[1])
    ink_w = float(ink[2] - ink[0])
    if ink_h <= 0:
        return max(9.0, round(box_h * 0.82, 1)), 0.0, 1.2

    if force_size is not None and force_size > 0:
        font_size = float(force_size)
        scale = font_size / float(_MEASURE_REF_SIZE)
    else:
        scale = float(box_h) / ink_h
        font_size = max(6.0, _MEASURE_REF_SIZE * scale)

    # Spread (or pull in) the residual width across the gaps between glyphs. CSS adds
    # letter-spacing after every character, but the ink of the run ends before the last
    # one, so the gaps that matter number len-1.
    # Both boxes are ink bounds now, so the residual is genuine face-width mismatch and
    # should be small. Clamp hard: a wrong measurement font must not be allowed to shred
    # the run into spaced-out characters.
    gaps = max(len(clean) - 1, 1)
    letter_spacing = (float(box_w) - ink_w * scale) / gaps
    limit = font_size * 0.06
    letter_spacing = max(-limit, min(limit, letter_spacing))

    # CSS centres the (ascent + descent) content box inside the line box, so the ink top
    # sits at  top + (L - ascent - descent)/2 + ink_offset.  Setting that equal to the
    # element top and solving for L gives the line-height that pins ink to the box.
    line_px = (ascent + descent - 2.0 * ink[1]) * scale
    line_height = max(0.1, line_px / font_size)

    return round(font_size, 2), round(letter_spacing, 2), round(line_height, 3)

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

def ink_top_offset(text: str, font_size: float, line_px: float, bold: bool) -> float:
    """Distance from a text element's top edge down to the first line's ink.

    A paragraph has to use the measured line pitch for its line-height, which overrides
    the per-run value that pinned a single line's ink to its box. Solving the same CSS
    inline box model for the offset instead lets the element's top be shifted to
    compensate, so the first line still lands where it was measured.
    """
    clean = (text or '').strip()
    font = _get_measure_font(bold) if clean else None
    if font is None or font_size <= 0:
        return 0.0
    try:
        ink = font.getbbox(clean)
        ascent, descent = font.getmetrics()
    except Exception:
        return 0.0
    scale = font_size / float(_MEASURE_REF_SIZE)
    content = (ascent + descent) * scale
    return (line_px - content) / 2.0 + ink[1] * scale

def erase_text_from_crop(crop: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
    """Paints out masked text pixels so a raster crop can sit underneath live text.

    UI screenshots are dominated by flat fills, so each masked region is filled with
    the median colour of the ring of pixels immediately surrounding it — that restores
    a button's solid fill far more cleanly than inpainting, which smears gradients
    across the glyphs. Falls back to Telea inpainting when a region has no clean ring
    (e.g. text running to the crop edge).
    """
    if crop.size == 0 or not np.any(text_mask):
        return crop

    out = crop.copy()
    mask = (text_mask > 0).astype(np.uint8)
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
                table_h = max(t_y1 - t_y0, 1.0)

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
                    f'<table class="reconstructed-table" style="width: 100%; height: 100%; '
                    f'table-layout: fixed; border-collapse: collapse; box-sizing: border-box; '
                    f'font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, Arial, sans-serif; '
                    f'color: #000000;">'
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

    @staticmethod
    def structure_v3_to_html(pred_html: str) -> str:
        clean = pred_html.replace('<html>', '').replace('</html>', '').replace('<body>', '').replace('</body>', '').strip()
        if not clean.startswith('<table'):
            clean = f'<table class="reconstructed-table">{clean}</table>'
        style_inject = 'style="width: 100%; height: 100%; border-collapse: collapse; font-family: inherit; font-size: 11px; line-height: 1.3;"'
        if '<table' in clean and 'style=' not in clean:
            clean = clean.replace('<table', f'<table {style_inject}', 1)
        return clean

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
# 7b. SURFACE ANALYSIS  (flat fills -> real CSS boxes)
# ==============================================================================

MIN_SURFACE_AREA = 140
_K3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

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
    __slots__ = ('bbox', 'mask', 'color', 'shape', 'radius', 'border_color',
                 'border_width', 'area', 'fill_ratio', 'children', 'parent', 'texts')

    def __init__(self, bbox, mask, color, shape, radius, area, fill_ratio,
                 border_color=None, border_width=0.0):
        self.bbox = bbox                    # (x0, y0, x1, y1) in page pixels
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

def _classify_region(img, x, y, cw, ch, comp, area) -> Optional[Surface]:
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
        if 0.5 <= ring_thickness <= 6.0 and min(cw, ch) >= 12 and rect_score >= 0.55:
            return Surface(bbox, filled, None, 'rect', radius, area, fill_ratio,
                           border_color=color, border_width=round(max(ring_thickness, 1.0), 1))
        return Surface(bbox, filled, color, 'complex', 0.0, area, fill_ratio)

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
    if rect_score >= 0.62:
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
                    max_colors: int = 90) -> List[Surface]:
    """Segments the image into flat-filled regions by quantised colour.

    UI is built from flat fills, so connected regions of one colour recover the real
    component boxes -- including nested ones (a panel inside a card inside a page),
    which a single global contour pass collapses into one blob. Anything that is not
    flat (gradients, artwork, photos) falls out as 'complex' and stays pixels.
    """
    h, w = img.shape[:2]
    smooth = cv2.medianBlur(img, 3)
    quant = (smooth.astype(np.int32) // 8 * 8).astype(np.uint32)
    packed = (quant[:, :, 0] << 16) | (quant[:, :, 1] << 8) | quant[:, :, 2]

    vals, counts = np.unique(packed, return_counts=True)
    order = np.argsort(-counts)
    page_area = float(h * w)

    surfaces: List[Surface] = []
    for idx in order[:max_colors]:
        if counts[idx] < MIN_SURFACE_AREA:
            break
        mask = (packed == vals[idx]).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _K3)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, cw, ch, area = stats[i]
            if area < MIN_SURFACE_AREA or cw < 5 or ch < 5:
                continue
            comp = (lab[y:y+ch, x:x+cw] == i).astype(np.uint8)

            # Glyphs are flat fills too. A bold headline letter is a uniform connected
            # region large enough to pass every shape test, so without this check the
            # page fills with black boxes where the text should be.
            if text_ink is not None:
                overlap = float(np.count_nonzero(comp & text_ink[y:y+ch, x:x+cw]))
                if overlap / float(area) > 0.45:
                    continue

            surf = _classify_region(img, x, y, cw, ch, comp, area)
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
    if abs(((tx0 + tx1) / 2.0) - ((surf.bbox[0] + surf.bbox[2]) / 2.0)) <= max(10.0, w * 0.10):
        score += 0.20
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
    score = 0.0
    if _luminance(surf.color) > 200 or surf.border_width > 0:
        score += 0.15
    score += 0.15                                        # aspect already gated above
    if surf.radius >= 3.0:
        score += 0.10
    if run is not None and _luminance(run.get('color', '#000000')) > 120:
        score += 0.20                                    # placeholder grey, not body copy
    if _leading_icon(ctx, surf, run):
        score += 0.25
    if _interior_ink(ctx, surf, [run['bbox']] if run else []) < 0.05:
        score += 0.15
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

def classify_surface_role(surf: 'Surface', ctx) -> Tuple[str, str, float]:
    """Chooses the HTML tag for a surface from scored evidence.

    Deliberately conservative, and deliberately separate from how the surface is
    painted: geometry and colour are reconstructed identically whatever this returns.
    Below the promotion threshold a surface still renders pixel-for-pixel -- it just
    renders as a <div> instead of claiming to be something it might not be.
    """
    b = score_button(surf, ctx)
    i = score_input(surf, ctx)
    c = score_card(surf, ctx)
    best = max(b, i, c)

    if best < 0.50:
        return ('shape' if surf.shape == 'ellipse' else 'surface'), 'div', best

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
             if r.get('role') == 'body' and not r.get('host_role') in ('button', 'badge', 'input')]
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
            and not r.get('consumed')]
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
        return ImageReconstructor.reconstruct_image_from_cv2(img, asset_dir=asset_dir, source_file=file_path)

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
            if max_d > 22.0:
                thresh = max(18.0, float(np.percentile(diff, 75)))
                stroke_px = crop[diff >= thresh]
                if len(stroke_px) > 0:
                    text_color = _bgr_to_hex(np.median(stroke_px, axis=0).astype(int))

                # Detection boxes carry padding, so they are not ink bounds. Fitting a
                # font to them oversizes every run. Recover the true ink extent from the
                # pixels that differ from the local background -- which also works for
                # light-on-dark, where an Otsu "dark pixels are ink" test is backwards.
                ink = (diff >= max(20.0, max_d * 0.45)).astype(np.uint8)
                ink = keep_core_ink(ink)
                rows, cols = np.any(ink, axis=1), np.any(ink, axis=0)
                if rows.any() and cols.any():
                    r0 = int(np.argmax(rows))
                    r1 = len(rows) - int(np.argmax(rows[::-1]))
                    c0 = int(np.argmax(cols))
                    c1 = len(cols) - int(np.argmax(cols[::-1]))
                    if r1 > r0 and c1 > c0:
                        ink_box = (x0 + c0, y0 + r0, x0 + c1, y0 + r1)
            else:
                lum = 0.299 * local_bg[2] + 0.587 * local_bg[1] + 0.114 * local_bg[0]
                text_color = "#ffffff" if lum < 128 else "#111111"

            # Record the glyph pixels themselves, not the box. A filled box would hide
            # whatever a control's background is doing between the letters.
            if max_d > 22.0:
                stroke = (diff >= max(20.0, max_d * 0.45)).astype(np.uint8) * 255
                sub = text_mask[y0_box:y0_box + crop.shape[0], x0_box:x0_box + crop.shape[1]]
                np.maximum(sub, stroke, out=sub)

            if ink_box:
                x0, y0, x1, y1 = ink_box

            font_size, letter_spacing, line_height = fit_text_to_box(
                label, x1 - x0, y1 - y0, font_weight == "bold"
            )
            runs.append({
                'text': label,
                'bbox': [float(x0), float(y0), float(x1), float(y1)],
                'fontSize': float(font_size),
                'fontWeight': font_weight,
                'color': text_color,
                'style': TextStyle(
                    fontFamily=OCR_FONT_STACK,
                    fontSize=float(font_size),
                    fontWeight=font_weight,
                    color=text_color,
                    lineHeight=line_height,
                    letterSpacing=letter_spacing,
                ),
            })
        return runs, text_mask

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
                              asset_dir) -> List[DocumentElement]:
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
        for s in sorted(paintable, key=lambda s: -(s.width * s.height)):
            if s.color is None:
                continue
            x0, y0, x1, y1 = [int(v) for v in s.bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            m = s.mask
            if m.shape != (y1 - y0, x1 - x0):
                m = cv2.resize(m, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            c = s.color.lstrip('#')
            bgr = np.array([int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16)], dtype=np.uint8)
            region = predicted[y0:y1, x0:x1]
            region[m > 0] = bgr

        delta = cv2.absdiff(img, predicted)
        _, content = cv2.threshold(cv2.cvtColor(delta, cv2.COLOR_BGR2GRAY), 16, 255, cv2.THRESH_BINARY)
        # Text is drawn as live text, so its ink is already accounted for.
        ink = cv2.dilate(text_mask, _K3, iterations=1)
        residual = cv2.bitwise_and(content, cv2.bitwise_not(ink))
        residual = cv2.morphologyEx(residual, cv2.MORPH_OPEN, _K3)

        grouped = cv2.morphologyEx(
            residual, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        out: List[DocumentElement] = []
        idx = 1
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if cw < 3 or ch < 3:
                continue
            ink = int(np.count_nonzero(residual[y:y+ch, x:x+cw]))
            # Small is fine -- icons are small. Sparse noise is not.
            if ink < 40 or (cw * ch) < 40:
                continue
            if cw > w * 0.985 and ch > h * 0.985:
                continue

            crop = img[y:y+ch, x:x+cw]
            local = residual[y:y+ch, x:x+cw]
            if crop.size == 0:
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

            out.append(DocumentElement(
                id=f"art-{idx}", type='image',
                bbox=[float(x), float(y), float(x + cw), float(y + ch)],
                src=src, assetName=asset_name,
                naturalWidth=float(cw), naturalHeight=float(ch),
                tag='img', role='artwork', zIndex=9,
            ))
            idx += 1
        return out

    @staticmethod
    def reconstruct_image_from_cv2(
        img: np.ndarray,
        asset_dir: Optional[str] = None,
        page_num: int = 1,
        source_file: Optional[str] = None
    ) -> PageData:
        h, w = img.shape[:2]

        # 1. Original pixels, kept for fidelity comparison and as the ultimate fallback
        _, enc_img = cv2.imencode('.png', img)
        orig_img_b64 = f"data:image/png;base64,{base64.b64encode(enc_img.tobytes()).decode('utf-8')}"

        # 2. Page ground colour (median of the border ring)
        borders = np.concatenate([img[0:8, :], img[-8:, :], img[:, 0:8], img[:, -8:]], axis=None).reshape(-1, 3)
        mean_bg = np.median(borders, axis=0).astype(np.uint8)
        page_bg_hex = _bgr_to_hex(mean_bg)

        # 3. Text runs, with sizes fitted to real ink bounds
        text_runs, text_mask = ImageReconstructor._extract_text_runs(img)

        # 4. Flat fills -> candidate CSS surfaces
        surfaces = detect_surfaces(img, page_bg_hex, text_ink=(text_mask > 0).astype(np.uint8))
        paintable = [s for s in surfaces if s.shape in ('rect', 'ellipse')]

        # A bordered control arrives as two regions: a ring and the fill inside it.
        # Fold the ring into the fill so it becomes one element with a real border.
        paintable = ImageReconstructor._merge_borders(paintable)

        build_containment(paintable)
        assign_texts(paintable, text_runs)

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

        elements: List[DocumentElement] = []
        counters: Dict[str, int] = {}

        def next_id(kind: str) -> str:
            counters[kind] = counters.get(kind, 0) + 1
            return f"{kind}-{counters[kind]}"

        # 6. Emit surfaces as real boxes, absorbing a button's label into the button
        surface_ids: Dict[int, str] = {}
        for s in sorted(paintable, key=lambda s: -(s.width * s.height)):
            role, tag, conf = roles[id(s)]
            depth = 0
            p = s.parent
            while p is not None:
                depth += 1
                p = p.parent

            eid = next_id(role if role in ('button', 'badge', 'card', 'input') else 'surface')
            surface_ids[id(s)] = eid

            box = BoxStyle(
                backgroundColor=s.color,
                borderColor=s.border_color,
                borderWidth=s.border_width,
                borderRadius=(f"{s.radius:.1f}px" if s.radius >= 1.0 else None),
            )
            if s.shape == 'ellipse':
                box.borderRadius = "50%"

            elem = DocumentElement(
                id=eid,
                type='rect',
                bbox=[float(v) for v in s.bbox],
                tag=tag,
                role=role,
                box=box,
                confidence=conf,
                parentId=surface_ids.get(id(s.parent)) if s.parent is not None else None,
                zIndex=2 + depth * 2,
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
                html=''.join(parts), confidence=0.7, zIndex=11,
            ))

        for group in group_paragraphs(text_runs):
            if any(r.get('consumed') for r in group):
                continue
            px0 = min(r['bbox'][0] for r in group)
            py0 = min(r['bbox'][1] for r in group)
            px1 = max(r['bbox'][2] for r in group)
            py1 = max(r['bbox'][3] for r in group)
            lead = group[0]
            pitch = max(group[1]['bbox'][1] - group[0]['bbox'][1], 1.0)
            fs = max(lead['style'].fontSize, 1.0)
            style = lead['style'].model_copy(update={'lineHeight': round(pitch / fs, 3)})
            # Shift the block up by however far the first line's ink now sits below the
            # element top, so switching to the measured pitch does not move line one.
            py0 -= ink_top_offset(lead['text'], fs, pitch, lead['fontWeight'] == 'bold')
            for r in group:
                r['consumed'] = True
            elements.append(DocumentElement(
                id=next_id('para'), type='text',
                bbox=[float(px0), float(py0), float(px1), float(py1)],
                text=' '.join(r['text'] for r in group),
                tag='p', role='paragraph', style=style, confidence=0.75,
                parentId=surface_ids.get(id(lead.get('host'))) if lead.get('host') is not None else None,
                zIndex=12,
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
                parentId=surface_ids.get(id(host)) if host is not None else None,
                zIndex=12,
            ))

        # 8. Residual artwork: everything no surface or text explained stays as pixels
        elements.extend(ImageReconstructor._extract_residual_art(
            img, w, h, mean_bg, text_mask, paintable, asset_dir
        ))

        elements.sort(key=lambda e: e.zIndex or 1)

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
                        cv_img, asset_dir=asset_dir, page_num=page_num + 1, source_file=getattr(doc, 'name', None)
                    )
            except Exception as e:
                logger.warning(f"Could not extract direct image from scanned PDF page: {e}")

        # 2. Render page at 150 DPI for high-res reconstruction
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        cv_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        return ImageReconstructor.reconstruct_image_from_cv2(
            cv_img, asset_dir=asset_dir, page_num=page_num + 1, source_file=getattr(doc, 'name', None)
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
@media print {
    body.reconstructed-doc { padding: 0; background: transparent; gap: 0; }
    .pdf-page { box-shadow: none; page-break-after: always; break-after: page; margin: 0; }
}"""

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
        bg = f"background-color: {s.backgroundColor};" if (s and s.backgroundColor) else ""
        return (f"font-family: {ff}; font-size: {fs:.2f}px; font-weight: {fw}; "
                f"font-style: {fst}; color: {col}; text-align: {ta}; "
                f"line-height: {lh:.3f}; {ls}{bg}")

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

        common_style = (f"position: absolute; left: {left:.2f}px; top: {top:.2f}px; "
                        f"width: {width:.2f}px; height: {height:.2f}px; "
                        f"z-index: {z_index}; {op_str}{rot}")

        kids = (children or {}).get(elem.id, [])
        kids_html = "".join(
            HTMLRenderer.render_element(k, editable=editable, origin=(x0, y0), children=children)
            for k in kids
        )
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
            text_style = (f"{common_style} {HTMLRenderer._text_css(elem.style)} "
                          f"margin: 0; padding: 0; {wrap} overflow: visible;")

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
            return (f'<div class="pdf-element pdf-image-container" {data_attrs} '
                    f'style="{common_style}"><img src="{src}" alt="{elem.role or "Embedded Asset"}" '
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
    def render_document(doc: DocumentData, editable: bool = False, title: Optional[str] = None) -> str:
        doc_title = title or doc.title or "Reconstructed PDF Document"
        pages_html = "\n\n".join([HTMLRenderer.render_page(p, editable=editable) for p in doc.pages])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(doc_title)}</title>
    <style>
{BASE_PAGE_CSS}
    </style>
</head>
<body class="reconstructed-doc">
{pages_html}
</body>
</html>"""

class Exporter:
    @staticmethod
    def export_standalone_html(doc_data: DocumentData) -> str:
        return HTMLRenderer.render_document(doc_data, editable=False)

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
