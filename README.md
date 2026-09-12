# Ramen — screenshot & PDF to editable HTML

Reconstructs a page image or a PDF into HTML that both **looks like the original** and
**is made of real elements** — a button that is a `<button>`, a link that is an `<a>`, a
heading that is an `<h1>` — with an editor for correcting what the engine got wrong.

**Deterministic. No LLM, no VLM, no generative model.** Everything is measured from the
pixels: colour segmentation, shape evidence, OCR, and typographic fitting.

---

## What it does

### Image → HTML

A screenshot goes through content-type segmentation first, because the question *"what
components are here"* only makes sense once you know *"is this part of the page even
interface"*. Photographic and gradient regions are kept as pixels; flat regions go to
the component pipeline.

- **Surfaces from flat fills.** Connected regions of one colour, classified by several
  independent signals — area against the solved corner radius, edge straightness outside
  the corner arcs, corner symmetry, polygon vertex count, colour uniformity. Recovers
  nested boxes a single contour pass collapses into one blob.
- **Real CSS, not rasters.** Fill, corner radius, border and drop shadow are extracted
  and emitted as CSS. A control is a `<button>` with a `background` and a
  `border-radius`, never a picture of one.
- **Containment tree.** Elements nest, so a card really contains its contents and a
  label really sits inside its button.
- **Scored promotion.** Buttons, links, inputs, cards and badges are promoted on
  weighted evidence, never on a single rule. Below threshold a surface still renders
  pixel-for-pixel — it just renders as a `<div>` rather than claiming to be something it
  might not be. Naming an element never moves a pixel.
- **Groups vote.** Repeated elements — a nav, a sidebar, a list — are classified as a
  set, because four evenly spaced labels in a row are evidence a single one is not.
- **Typography is fitted, not guessed.** Size, letter-spacing, word-spacing, line-height
  and horizontal scale are solved so each run lands on the ink box it was measured from.
- **Text stays text.** Glyphs are erased from the raster underneath so nothing is
  double-rendered, and lettering that belongs to a photograph's subject is left in the
  image.

### PDF → HTML

Digital pages are read natively through PyMuPDF: text with per-span typography, tables
with recovered column widths and cell fills, vector paths as SVG, and embedded images
with their soft masks. Scanned pages are rasterised and sent through the image path
above.

### Editor

Zoom and pan, drag and resize, inline text and table-cell editing, a properties panel,
and a devtools-style inspector (press **I**) that reports what each element actually is.
Export as a standalone HTML file, a ZIP bundle with assets, or the intermediate JSON.

---

## Directory Structure

```text
Ramen/
├── app.py                       # FastAPI server & REST API endpoints
├── engine.py                    # Reconstruction engine
├── index.html                   # Single-page desktop editor UI
├── test_app.py                  # Test suite for engine & API
├── requirements.txt
├── benchmarks/                  # Everything the engine is measured against
│   ├── images/                  # Reference screenshots for the image path
│   ├── pdfs/                    # Reference PDFs for the digital path
│   └── ground_truth/            # Hand-read component lists, one JSON per image
├── tools/
│   ├── eval_image.py            # Fidelity (SSIM / pixel diff) for the image path
│   ├── eval_pdf.py              # Fidelity for the digital PDF path
│   ├── audit_components.py      # Component accuracy against ground truth
│   └── test_scale_invariance.py # Same page at several capture sizes
└── scratch/                     # Working area, gitignored: renders, composites, debug
```

---

## Quick Start

```bash
pip install -r requirements.txt
python app.py           # or: uvicorn app:app --port 8000 --reload
```

Open **http://127.0.0.1:8000** and load a PNG/JPG screenshot or a PDF.

```bash
pytest test_app.py
```

---

## Measuring

```bash
python tools/eval_image.py                  # every image in benchmarks/images
python tools/eval_image.py path/to/one.png  # or just one
python tools/eval_pdf.py                    # the 20-page PDF benchmark
python tools/audit_components.py            # component accuracy where ground truth exists
python tools/test_scale_invariance.py       # the same page at 0.6x, 1x, 1.6x
```

**Adding a reference**: drop a screenshot into `benchmarks/images/`. The fidelity tools
pick it up with no further setup. To score component accuracy on it too, add
`benchmarks/ground_truth/<same-name>.json` listing what the page actually contains — see
`reddit-ads-hero.json` for the shape.

### Two metrics, and which one to trust

Fidelity says whether the page *looks* right. The component audit says whether a button
is a button. **They disagree, and the audit is the one worth optimising.** SSIM read 89%
on a page where 18 of 25 components carried the wrong tag, and later read *higher* on a
page that had lost every heading than on the same page with its headings restored. Treat
fidelity as a guardrail and read the rendered output yourself.

### Where it stands

| | |
|---|---|
| Image path, 6 reference pages | **89.98%** mean SSIM (87.3 – 93.7) |
| Component accuracy | **24/25** on the page with ground truth |
| Digital PDF, 20-page benchmark | **90.81%** mean SSIM |
| Spread across a 2.7× resolution range | ≤ 5.4 points |

---

## Known limitations

- **Typeface is approximated.** The engine fits size, spacing and width, but cannot
  identify the original face; a display serif comes back as a condensed sans. This is
  the largest remaining visual difference on most pages.
- **Ground truth exists for one page.** Component accuracy is therefore a single-page
  measurement, and generalisation is unproven. Adding references is the highest-value
  contribution.
- **Translucency is rasterised.** Glassmorphic panels are kept as pixels rather than
  `backdrop-filter` + `rgba()`.
- **Layout is absolute.** Output is pixel-accurate but does not reflow; there is no
  flex/grid inference yet.
- **Scanned PDFs get no table recognition.** They are routed through the image path,
  which has no table model.
- **`html2canvas` is loaded from a CDN** for the in-app fidelity comparison, so that one
  feature needs network access.
