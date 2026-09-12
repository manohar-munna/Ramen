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

## Directory Structure

```text
Ramen/
├── app.py                  # FastAPI server & REST API endpoints
├── engine.py               # Reconstruction engine (schema, extractors, renderer, exporter)
├── index.html              # Single-page desktop editor UI
├── test_app.py             # Test suite for engine & API
├── requirements.txt
├── benchmarks/             # Everything the engine is measured against
│   ├── images/             # Reference screenshots for the image -> HTML path
│   ├── pdfs/               # Reference PDFs for the digital path
│   └── ground_truth/       # Hand-read component lists, one JSON per image
├── tools/
│   ├── eval_image.py       # Fidelity (SSIM / pixel diff) for the image path
│   ├── eval_pdf.py         # Fidelity for the digital PDF path
│   └── audit_components.py # Component-level accuracy against ground truth
└── scratch/                # Working area, gitignored: renders, composites, debug
```

## Measuring

```bash
python tools/eval_image.py                 # every image in benchmarks/images
python tools/eval_image.py path/to/one.png # or just one
python tools/eval_pdf.py                   # the 20-page PDF benchmark
python tools/audit_components.py           # component accuracy where ground truth exists
```

**Adding a reference**: drop a screenshot into `benchmarks/images/`. Fidelity tooling
picks it up with no further setup. To score component accuracy on it as well, add
`benchmarks/ground_truth/<same-name>.json` listing what the page actually contains —
see `reddit-ads-hero.json` for the shape.

Fidelity tells you whether the page *looks* right; the component audit tells you
whether a button is a button. They disagree often, and the audit is the one worth
optimising — SSIM read 89% on a page where 18 of 25 components carried the wrong tag.

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
