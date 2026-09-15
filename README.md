# Ramen — screenshot & PDF to editable HTML

Reconstructs a page image or a PDF into HTML that both **looks like the original** and
**is made of real elements** — a button that is a `<button>`, a link that is an `<a>`, a
heading that is an `<h1>` — with an editor for correcting what the engine got wrong.

**Reconstruction is deterministic. No LLM, no VLM, no generative model.** Everything is
measured from the pixels: colour segmentation, shape evidence, OCR, and typographic
fitting. The only learned component in the reconstruction path is the OCR, and it only
reads text.

There is also an **optional** pass that hands the finished HTML to Gemini to be rebuilt
with real layout and transitions. It is off unless you supply a key, it runs after
reconstruction rather than inside it, and it cannot change what the engine measured — see
[Enhancement](#enhancement-optional).

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
- **Photographs are cut to their own outline.** A picture is not a rectangle. The
  silhouette is used as the alpha, so an object shot at an angle stops painting over the
  type beside it. Holes enclosed by the picture are filled back in, because text sits
  *on* a photograph and needs the photograph behind it.
- **A fill that cannot explain its own ground stops painting.** A card's interior is its
  own fill perforated by its contents; a strip of page background wrapping a card is not.
  The second keeps its box and its place in the hierarchy but paints nothing.
- **Nothing invisible is shipped.** The paint order is walked from the top down and any
  anonymous decoration whose footprint is entirely covered is dropped — on the reference
  pages that is roughly 40% of all elements, at zero pixel cost.

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
├── enhancer.py                  # Optional Gemini rewrite pass (not part of the engine)
├── test_app.py                  # Test suite for engine & API
├── requirements.txt
├── .env.example                 # Copy to .env for the enhancement key
├── benchmarks/                  # Everything the engine is measured against
│   ├── images/                  # 11 reference screenshots; SOURCES.md says where from
│   ├── pdfs/                    # Reference PDFs for the digital path
│   └── ground_truth/            # Hand-read component lists, one JSON per image
├── tools/
│   ├── eval_image.py            # Fidelity (SSIM / pixel diff) for the image path
│   ├── eval_pdf.py              # Fidelity for the digital PDF path
│   ├── audit_components.py      # Component accuracy against ground truth
│   ├── test_scale_invariance.py # Same page at several capture sizes
│   ├── audit_enhanced.py        # What a rewrite did to the content it was given
│   └── enhance_all.py           # Enhance every reference and score it; resumable
└── mlkit/                       # Trained UI detector — a side experiment, not wired in
```

Gitignored working directories: `scratch/` (renders and debug output), `cache/` and
`outputs/` (per-job server state), `datasets/` and `models/` and `weights/` (training
data and checkpoints, all reproducible).

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
| Image path, 6 reference pages | **89.91%** mean SSIM (86.9 – 93.4) |
| Component accuracy | **24/25** on the page with ground truth |
| Digital PDF, 20-page benchmark | **90.81%** mean SSIM |
| Test suite | **10/10** |
| Spread across a 2.7x resolution range | **6.51** points, worst page |

Every page scores higher at 1.6x than at 0.6x, and the gap has widened rather than
closed (5.4 points before the last round of changes, 6.51 after). Resolution
normalisation makes the thresholds scale-aware but does not make small captures
reconstruct as well as large ones, and nothing currently measures which threshold is
responsible.

---

## Enhancement (optional)

The engine optimises for faithfulness and is measured on it, which is why its output is a
pile of absolutely-positioned divs: nothing in the pipeline is rewarded for the page being
well *built*. Rewriting it as flowing, semantic, animated HTML has no ground truth to
measure against, so it is a separate pass and a language model does it.

```bash
cp .env.example .env          # then put a Gemini key in it
python enhancer.py benchmarks/images/axion-logistics.png
python enhancer.py page.html --dry-run    # prompt size, no network call
```

Or press **Enhance** in the editor. The result opens in a new tab; the reconstruction is
untouched.

**Images are held back.** About 95% of a reconstructed page by weight is inline base64
PNG — the logistics page is 1.52M characters, roughly 380k tokens, none of which a
language model can use, since it cannot see pixels. Each data URI is swapped for a short
marker (identical assets sharing one), which takes that page to 67k characters and ~17k
tokens; the markers are restored afterwards. So the images never depend on the model
reproducing a megabyte of base64 exactly, and a marker it drops is counted and reported
rather than silently costing you an image.

**What it will and will not do.** The prompt forbids changing any visible text, dropping
any image marker, or moving far from the original palette and type scale; it asks for
flex/grid layout, semantic tags, hover and focus states, transitions, responsiveness down
to 480px, and `prefers-reduced-motion` support. None of that is enforced — it is a model
following instructions, and the output is not measured by any of the tools above. Treat
the enhanced page as a draft to read, not as a reconstruction.

| | |
|---|---|
| `GEMINI_API_KEY` | required, from [AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | optional, defaults to `gemini-2.5-pro` |

`.env` is gitignored. No new dependency — the call goes through `urllib`.

---

## mlkit — the trained detector

A YOLO-nano detector (~2.6M parameters) that finds interactive elements in a screenshot.
Not a VLM and not a language model: a convolutional detector returning boxes and classes.
See `mlkit/README.md` for the pipeline.

**It is not wired into `engine.py`.** Reconstruction is entirely deterministic and does
not call it. The branch exists because the question "can a small trained model decide what
a component is more reliably than hand-written scoring" is worth an answer, and the answer
is specific: yes for things you can click (mAP50 0.47 on buttons, 0.41 on links against
held-out real screenshots), and nothing at all for headings, cards or images, because the
only real-screenshot dataset available annotates only clickable elements. Generated
training pages do not close that gap — the two domains are trivially separable, so a model
trained on both learns to tell them apart and applies a different prior to each.

---

## Known limitations

- **Pill-shaped controls are not promoted.** Every strongly-rounded surface on the six
  reference pages has zero text runs attached to it, including a 1203×68 pill nav bar, so
  the button scorer returns 0 before it starts. The scoring is not rejecting them; the
  label never reaches them.
- **Icons are absorbed into the text layer.** OCR reads a glyph-like icon as characters —
  one renders as the literal string `83` — so the icon is neither drawn nor available as
  an element.
- **Ground truth exists for one page.** Component accuracy is therefore a single-page
  measurement, and generalisation is unproven. Adding references is the highest-value
  contribution.
- **Output is mostly base64.** Roughly 90% of each exported file is inline PNG data, and
  a page runs 0.6–1.8 MB. Fine to view, not yet a clean hand-editable document.
- **Typeface is approximated.** The engine fits size, spacing and width, but cannot
  identify the original face; a display serif comes back as a condensed sans. This is
  the largest remaining visual difference on most pages.
- **Layout is absolute.** Output is pixel-accurate but does not reflow; there is no
  flex/grid inference yet.
- **Translucency is rasterised.** Glassmorphic panels are kept as pixels rather than
  `backdrop-filter` + `rgba()`.
- **Gradients are rasterised.** A fitter was written and reverted: wired onto the
  rectangle path it fired once across six pages, because gradient regions classify as
  `complex` and never reach it.
- **Scanned PDFs get no table recognition.** They are routed through the image path,
  which has no table model.
- **The server is a local tool, not a service.** CORS is open to `*`, there is no
  authentication, no upload size limit and no cleanup policy for `outputs/`.
- **`html2canvas` is loaded from a CDN** for the in-app fidelity comparison, so that one
  feature needs network access.
