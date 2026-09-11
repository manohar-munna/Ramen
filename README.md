# RAMEN: PDF-to-High-Fidelity-Editable-HTML Reconstruction System

A desktop-first application that reconstructs PDF documents into **pixel-perfect, editable HTML pages** preserving original page composition, spacing, typography, tables, images, formulas, alignment, dimensions, and relative positioning.

**STRICT DETERMINISTIC ENGINE — NO LLM / NO VLM / NO GENERATIVE AI**

---

## Key Features

1. **2D Coordinate-Based Canvas**:
   - Treats each PDF page as a fixed-size 2D coordinate canvas.
   - Elements are placed using `position: absolute` with explicit bounding boxes `[x1, y1, x2, y2]`.
   - Eliminates layout float drift, wrapping issues, and structural deformation.

2. **Bifurcated Pipeline (Digital vs. Scanned)**:
   - **Digital PDFs**: Extracts native PDF text, font family, font size, weight, color, alignment, embedded images, vector shapes (as SVG), and tables directly via PyMuPDF.
   - **Scanned PDFs**: High-resolution rasterization, PP-DocLayout_plus-L / CV layout detection, PaddleOCR text recognition, and PP-StructureV3 table recognition.

3. **Editable Tables**:
   - Converts tabular regions into clean, editable `<table>` elements with cell borders, row/col spans, background colors, and typography intact.

4. **Vector Graphics & Formulas**:
   - Vector paths, rules, and lines extracted natively as scalable SVG figures.
   - Mathematical equations identified and formatted with LaTeX / native browser MathML representation.

5. **Visual Fidelity Comparison Mode**:
   - Side-by-side comparison, interactive split wipe slider, and pixel discrepancy heatmap.
   - Computes structural similarity index (SSIM) and pixel difference percentage against the original PDF render.

6. **Interactive Desktop-First Editor UI**:
   - Clean, focused professional document tool (no bloated dashboards or fake analytics).
   - Page thumbnail sidebar.
   - Main canvas with zoom, pan, element drag-and-drop, resize handles, inline text editing, and table cell editing.
   - Contextual properties panel for typography, alignment, position, dimensions, and images.

7. **Flexible Independent Export**:
   - **HTML + Assets ZIP**: Standalone bundle with `index.html`, `styles.css`, `assets/`, and `document.json`. Works completely independently without server.
   - **Standalone HTML**: Single self-contained HTML page.
   - **Intermediate JSON**: Single source of truth containing 2D coordinates, geometry, and styling.

8. **Local Model Management & Offline Operation**:
   - Dedicated `ModelManager` and verification script (`scripts/download_models.py`).
   - Runs fully offline once models are downloaded.

---

## Simplified Directory Structure

```text
Ramen/
├── app.py           # FastAPI backend server & REST API endpoints
├── engine.py        # Complete deterministic reconstruction engine (schema, extractors, renderer, fidelity, exporter)
├── index.html       # Unified single-page desktop editor UI (HTML, CSS, JS integrated)
├── sample.pdf       # Sample test PDF
├── test_app.py      # Comprehensive test suite for engine & API endpoints
├── requirements.txt # Python dependencies
├── .gitignore       # Git ignore configuration
└── README.md        # Documentation
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python app.py
```
Or with Uvicorn directly:
```bash
uvicorn app:app --port 8000 --reload
```

Open your browser at **http://127.0.0.1:8000**.

### 3. Run Tests

```bash
pytest test_app.py
```
