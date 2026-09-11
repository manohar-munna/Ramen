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

class DocumentElement(BaseModel):
    id: str = Field(default_factory=lambda: f'elem-{uuid.uuid4().hex[:8]}')
    type: Literal['text', 'image', 'table', 'vector', 'formula', 'custom']
    bbox: List[float]  # [x1, y1, x2, y2] in points/pixels
    zIndex: int = 1
    rotation: float = 0.0
    opacity: float = 1.0

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

def fit_text_to_box(text: str, box_w: float, box_h: float, bold: bool) -> Tuple[float, float, float]:
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
    def reconstruct_image_from_cv2(
        img: np.ndarray,
        asset_dir: Optional[str] = None,
        page_num: int = 1,
        source_file: Optional[str] = None
    ) -> PageData:
        h, w = img.shape[:2]

        # 1. Base64 of original image
        _, enc_img = cv2.imencode('.png', img)
        orig_img_b64 = f"data:image/png;base64,{base64.b64encode(enc_img.tobytes()).decode('utf-8')}"

        # 2. Detect page background color (median of border pixels)
        borders = np.concatenate([img[0:8, :], img[-8:, :], img[:, 0:8], img[:, -8:]], axis=None).reshape(-1, 3)
        mean_bg = np.median(borders, axis=0).astype(np.uint8)
        page_bg_hex = f"#{mean_bg[2]:02x}{mean_bg[1]:02x}{mean_bg[0]:02x}"

        elements: List[DocumentElement] = []
        elem_counter = 1

        # 3. PaddleOCR for precision full-image text detection and typography
        ocr = ImageReconstructor._get_ocr()
        text_mask = np.zeros((h, w), np.uint8)

        if ocr:
            try:
                ocr_out = ocr.ocr(img)
                if ocr_out and len(ocr_out) > 0:
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
                        if float(score) < 0.35 or not str(text).strip():
                            continue
                        xs = [int(pt[0]) for pt in box]
                        ys = [int(pt[1]) for pt in box]
                        x0, y0, x1, y1 = max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))
                        if x1 <= x0 or y1 <= y0:
                            continue

                        # Mark text mask with padding
                        cv2.rectangle(text_mask, (max(0, x0 - 4), max(0, y0 - 4)), (min(w, x1 + 4), min(h, y1 + 4)), 255, -1)

                        box_h = y1 - y0

                        # Typography & background-contrast stroke color sampling
                        crop = img[y0:y1, x0:x1]
                        text_color = "#111111"
                        font_weight = "normal"
                        ink_box = None
                        if crop.size > 0:
                            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            _, bin_crop = cv2.threshold(gray_crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                            stroke_ratio = np.count_nonzero(bin_crop) / float(crop.shape[0] * crop.shape[1])
                            if stroke_ratio > 0.28 or box_h > 32:
                                font_weight = "bold"

                            border_px = np.concatenate([crop[0, :], crop[-1, :], crop[:, 0], crop[:, -1]], axis=0)
                            local_bg = np.median(border_px, axis=0)
                            diff = np.linalg.norm(crop.astype(float) - local_bg.astype(float), axis=2)
                            max_d = np.max(diff)
                            if max_d > 22.0:
                                thresh = max(18.0, np.percentile(diff, 75))
                                stroke_px = crop[diff >= thresh]
                                if len(stroke_px) > 0:
                                    c_bgr = np.median(stroke_px, axis=0).astype(int)
                                    text_color = f"#{c_bgr[2]:02x}{c_bgr[1]:02x}{c_bgr[0]:02x}"

                                # Detection boxes carry padding, so they are not ink bounds.
                                # Fitting a font to them oversizes every run. Recover the
                                # true ink extent from the pixels that differ from the local
                                # background -- that works for light-on-dark too, which an
                                # Otsu "dark pixels are ink" test gets backwards.
                                ink = diff >= max(20.0, float(max_d) * 0.45)
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

                        if ink_box:
                            x0, y0, x1, y1 = ink_box
                        font_size, letter_spacing, line_height = fit_text_to_box(
                            str(text), x1 - x0, y1 - y0, font_weight == "bold"
                        )

                        elements.append(DocumentElement(
                            id=f"text-{elem_counter}",
                            type='text',
                            bbox=[float(x0), float(y0), float(x1), float(y1)],
                            text=str(text).strip(),
                            style=TextStyle(
                                fontFamily=OCR_FONT_STACK,
                                fontSize=float(font_size),
                                fontWeight=font_weight,
                                color=text_color,
                                lineHeight=line_height,
                                letterSpacing=letter_spacing
                            ),
                            zIndex=10
                        ))
                        elem_counter += 1
            except Exception as e:
                logger.warning(f"OCR failed during image reconstruction: {e}")

        # 4. Extract Graphic Regions & Shapes (after text is masked out)
        non_text = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(text_mask))
        diff = cv2.absdiff(non_text, np.full_like(non_text, mean_bg, dtype=np.uint8))
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, graphic_mask = cv2.threshold(diff_gray, 24, 255, cv2.THRESH_BINARY)
        graphic_mask = cv2.bitwise_and(graphic_mask, graphic_mask, mask=cv2.bitwise_not(text_mask))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        closed = cv2.morphologyEx(graphic_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        img_counter = 1
        for c in contours:
            area = cv2.contourArea(c)
            if area < 400:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            # Ignore border slivers (e.g. scrollbars or window borders)
            if cw < 60 and ch > h * 0.7:
                continue
            # Ignore full canvas wrappers
            if cw > w * 0.96 and ch > h * 0.96:
                continue

            # Check for divider line
            if ch <= 4 and cw > 40:
                line_color = f"#{img[y, x+cw//2][2]:02x}{img[y, x+cw//2][1]:02x}{img[y, x+cw//2][0]:02x}"
                svg_line = f'<svg viewBox="0 0 {cw} {ch}" width="100%" height="100%"><line x1="0" y1="{ch/2}" x2="{cw}" y2="{ch/2}" stroke="{line_color}" stroke-width="{ch}"/></svg>'
                elements.append(DocumentElement(
                    id=f"vector-{elem_counter}",
                    type='vector',
                    bbox=[float(x), float(y), float(x + cw), float(y + ch)],
                    svg=svg_line,
                    zIndex=3
                ))
                elem_counter += 1
                continue

            # Check for circle
            peri = cv2.arcLength(c, True)
            circularity = 4 * np.pi * (area / (peri * peri)) if peri > 0 else 0
            aspect = cw / float(max(ch, 1))
            if circularity > 0.82 and 0.8 < aspect < 1.25 and cw < 120:
                # A round outline is not the same as a solid disc. Logos, avatars and
                # icon buttons are circular too, and replacing them with a flat fill
                # destroys the artwork inside. Only vectorise when the interior really
                # is one colour; otherwise fall through and keep the pixels.
                disc = np.zeros((ch, cw), np.uint8)
                cv2.ellipse(disc, (cw // 2, ch // 2),
                            (max(cw // 2 - 2, 1), max(ch // 2 - 2, 1)), 0, 0, 360, 255, -1)
                interior = img[y:y+ch, x:x+cw][disc > 0]
                if interior.size > 0 and float(interior.reshape(-1, 3).std(axis=0).max()) < 12.0:
                    c_bgr = np.median(interior.reshape(-1, 3), axis=0).astype(int)
                    circle_color = f"#{c_bgr[2]:02x}{c_bgr[1]:02x}{c_bgr[0]:02x}"
                    svg_circle = f'<svg viewBox="0 0 {cw} {ch}" width="100%" height="100%"><ellipse cx="{cw/2}" cy="{ch/2}" rx="{cw/2}" ry="{ch/2}" fill="{circle_color}"/></svg>'
                    elements.append(DocumentElement(
                        id=f"vector-{elem_counter}",
                        type='vector',
                        bbox=[float(x), float(y), float(x + cw), float(y + ch)],
                        svg=svg_circle,
                        zIndex=3
                    ))
                    elem_counter += 1
                    continue

            # Graphic asset extraction
            cropped = img[y:y+ch, x:x+cw]
            if cropped.size == 0:
                continue

            # Every glyph is re-emitted as a live text element, so it must be erased from
            # the raster crop underneath. Skipping this double-renders all text.
            is_bg_shape = (cw * ch > w * h * 0.25)
            c_mask = text_mask[y:y+ch, x:x+cw]
            if np.any(c_mask):
                cropped = erase_text_from_crop(cropped, c_mask)

            bgra = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
            if not is_bg_shape:
                bg_diff = np.max(np.abs(cropped.astype(np.int32) - mean_bg.astype(np.int32)), axis=2)
                bgra[bg_diff < 16, 3] = 0

            _, enc_asset = cv2.imencode('.png', bgra)
            src = f"data:image/png;base64,{base64.b64encode(enc_asset.tobytes()).decode('utf-8')}"
            asset_name = f"image_asset_{img_counter}.png"

            if asset_dir:
                os.makedirs(asset_dir, exist_ok=True)
                cv2.imwrite(os.path.join(asset_dir, asset_name), bgra)

            elements.append(DocumentElement(
                id=f"image-{img_counter}",
                type='image',
                bbox=[float(x), float(y), float(x + cw), float(y + ch)],
                src=src,
                assetName=asset_name,
                naturalWidth=float(cw),
                naturalHeight=float(ch),
                zIndex=1 if is_bg_shape else 5
            ))
            img_counter += 1

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
@media print {
    body.reconstructed-doc { padding: 0; background: transparent; gap: 0; }
    .pdf-page { box-shadow: none; page-break-after: always; break-after: page; margin: 0; }
}"""

class HTMLRenderer:
    @staticmethod
    def render_element(elem: DocumentElement, editable: bool = True) -> str:
        bbox = elem.bbox or [0, 0, 10, 10]
        x0 = float(bbox[0] if len(bbox) > 0 and bbox[0] is not None else 0.0)
        y0 = float(bbox[1] if len(bbox) > 1 and bbox[1] is not None else 0.0)
        x1 = float(bbox[2] if len(bbox) > 2 and bbox[2] is not None else x0 + 10.0)
        y1 = float(bbox[3] if len(bbox) > 3 and bbox[3] is not None else y0 + 10.0)
        width, height = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        z_index = elem.zIndex or 1
        rot_deg = float(elem.rotation or 0.0)
        rot = f"transform: rotate({rot_deg}deg);" if rot_deg != 0.0 else ""
        elem_op = float(elem.opacity if elem.opacity is not None else 1.0)
        op_str = f"opacity: {elem_op:.2f}; " if elem_op < 0.999 else ""

        common_style = f"position: absolute; left: {x0:.2f}px; top: {y0:.2f}px; width: {width:.2f}px; height: {height:.2f}px; z-index: {z_index}; {op_str}{rot}"

        if elem.type == 'text':
            style = elem.style
            ff = style.fontFamily if (style and style.fontFamily) else "'Segoe UI', Arial, sans-serif"
            fs = float(style.fontSize if (style and style.fontSize is not None) else 12.0)
            fw = style.fontWeight if (style and style.fontWeight) else "normal"
            fst = style.fontStyle if (style and style.fontStyle) else "normal"
            col = style.color if (style and style.color) else "#000000"
            ta = style.textAlign if (style and style.textAlign) else "left"
            lh = float(style.lineHeight if (style and style.lineHeight is not None) else 1.0)
            bg = f"background-color: {style.backgroundColor};" if (style and style.backgroundColor) else ""
            raw_ls = float(style.letterSpacing) if (style and style.letterSpacing) else 0.0
            ls = f"letter-spacing: {raw_ls:.2f}px; " if abs(raw_ls) > 0.01 else ""

            text_style = f"{common_style} font-family: {ff}; font-size: {fs:.2f}px; font-weight: {fw}; font-style: {fst}; color: {col}; text-align: {ta}; line-height: {lh:.3f}; {ls}white-space: nowrap; overflow: visible; {bg}"

            if elem.spans and len(elem.spans) > 1:
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
            return f'<div class="pdf-element pdf-text" id="{elem.id}" data-id="{elem.id}" data-type="text"{edit_attr} style="{text_style}">{content_html}</div>'

        elif elem.type == 'image':
            src = elem.src or ''
            return f'<div class="pdf-element pdf-image-container" id="{elem.id}" data-id="{elem.id}" data-type="image" style="{common_style}"><img src="{src}" alt="Embedded Asset" loading="lazy" /></div>'

        elif elem.type == 'table':
            content = elem.html or '<table><tr><td>Table</td></tr></table>'
            if editable:
                content = content.replace('<td', '<td contenteditable="true" spellcheck="false"').replace('<th', '<th contenteditable="true" spellcheck="false"')
            return f'<div class="pdf-element pdf-table-container" id="{elem.id}" data-id="{elem.id}" data-type="table" style="{common_style}">{content}</div>'

        elif elem.type == 'vector':
            return f'<div class="pdf-element pdf-vector-container" id="{elem.id}" data-id="{elem.id}" data-type="vector" style="{common_style}">{elem.svg or "<svg></svg>"}</div>'

        elif elem.type == 'formula':
            formula_html = elem.renderedHtml or f'<div>{html.escape(elem.text or "")}</div>'
            return f'<div class="pdf-element pdf-formula-container" id="{elem.id}" data-id="{elem.id}" data-type="formula" style="{common_style}">{formula_html}</div>'

        elif elem.type == 'custom' or elem.html:
            content = elem.html or ''
            return f'<div class="pdf-element pdf-custom-container" id="{elem.id}" data-id="{elem.id}" data-type="custom" style="{common_style}">{content}</div>'
        return ''

    @staticmethod
    def render_page(page: PageData, editable: bool = True) -> str:
        w = float(page.width if page.width is not None else 595.0)
        h = float(page.height if page.height is not None else 842.0)
        bg = f" background-color: {page.backgroundColor};" if getattr(page, 'backgroundColor', None) else ""
        rendered_elements = [HTMLRenderer.render_element(elem, editable=editable) for elem in page.elements]
        return (
            f'<div class="pdf-page" id="pdf-page-{page.pageNumber}" data-page="{page.pageNumber}" '
            f'data-scanned="{"true" if page.isScanned else "false"}" style="width: {w:.2f}px; height: {h:.2f}px;{bg}">\n'
            f'{"\n".join(rendered_elements)}\n</div>'
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
                rendered_elems = []
                for elem in page.elements:
                    if elem.type == 'image' and elem.id in assets_map:
                        cloned_elem = elem.model_copy(update={'src': assets_map[elem.id]})
                        rendered_elems.append(HTMLRenderer.render_element(cloned_elem, editable=False))
                    else:
                        rendered_elems.append(HTMLRenderer.render_element(elem, editable=False))

                w = float(page.width if page.width is not None else 595.0)
                h = float(page.height if page.height is not None else 842.0)
                exported_pages_html.append(
                    f'<div class="pdf-page" id="pdf-page-{page.pageNumber}" style="width: {w:.2f}px; height: {h:.2f}px;">\n'
                    + "\n".join(rendered_elems) + '\n</div>'
                )

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
