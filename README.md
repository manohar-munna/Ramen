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
python tools/audit_enhanced.py A.html B.html   # what a rewrite did to the content
python tools/enhance_all.py                 # enhance every reference and score it
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
| Image path, 11 reference pages | **92.33%** mean SSIM (86.9 – 97.2) |
| Component accuracy | **24/25** on the one page with ground truth |
| Digital PDF, 20-page benchmark | **90.81%** mean SSIM |
| Test suite | **10/10** |
| Spread across a 2.7x resolution range | **6.51** points, worst page |

**That mean went up without the engine getting better.** The six pages collected while
building it still average 89.91%, unchanged; the five real-site captures added later
average 95.24%, and pull the combined figure to 92.33%. They score higher because they
are mostly flat interface at 1920x941 — larger captures and less photography — which is
exactly the kind of thing this pipeline finds easy. Read the two groups separately or the
number will flatter the next change you make.

| | |
|---|---|
| The original six | **89.91%** — hero sections, heavy photography |
| Five real sites | **95.24%** — bandcamp, netflix, stanford, whatsapp, yelp |

Every page scores higher at 1.6x than at 0.6x, and the gap has widened rather than
closed (5.4 points before the last round of changes, 6.51 after). Resolution
normalisation makes the thresholds scale-aware but does not make small captures
reconstruct as well as large ones, and nothing currently measures which threshold is
responsible. That figure is from the original six; it has not been re-measured across
all eleven.

---

## Enhancement (optional)

The engine optimises for faithfulness and is measured on it, which is why its output is a
pile of absolutely-positioned divs: nothing in the pipeline is rewarded for the page being
well *built*. Rewriting it as flowing, semantic, animated HTML has no ground truth to
measure against, so it is a separate pass and a language model does it.

```bash
cp .env.example .env                        # then put a Gemini key in it
python enhancer.py benchmarks/images/axion-logistics.png
python enhancer.py page.json --dry-run      # prompt size, no network call
python enhancer.py --list-models            # what this key can actually call
python tools/enhance_all.py                 # every reference, scored; resumable
```

Or press **Enhance** in the editor. The result opens in a new tab; the reconstruction is
untouched.

### Images are held back

About 95% of a reconstructed page by weight is inline base64 PNG — the logistics page is
1.52M characters, roughly 380k tokens, none of which a language model can use, since it
cannot see pixels. Each data URI is swapped for a short marker (identical assets sharing
one), which takes that page to ~17k tokens; the markers are restored afterwards, so the
images never depend on the model reproducing a megabyte of base64 exactly.

Each marker is sent with its pixel size, dominant colours and whether it is a photograph
or a small graphic. That costs about 1.4k characters and is what took the reference
photographs from scattered fragments to correctly proportioned images — a model asked to
lay out pictures it cannot see needs to be told something about them.

### Fragments are composed first

The residual pass cuts out exactly the pixels CSS could not explain, so a photographic
page arrives as dozens of crops that only add up to a picture because each is pinned to a
measured coordinate. In normal flow they scatter. Overlapping fragments are therefore
painted onto one canvas before the model sees them — woodnest goes from 107 to 73, yelp
19 to 8, stanford 18 to 3.

**This is verified rather than assumed.** Compositing is what the browser does anyway, so
flattening ought to be invisible; it is not invisible by construction. The renderer leaves
a container on `z-index: auto` exactly when it has children, so merging or re-parenting
changes who has children and restacks things a long way from the edit. Checked across the
eleven references, five pages rendered differently — by up to 237 levels on a channel.
Two analytic guards were written for that and both failed in the same run, in opposite
directions: one refused a page that was provably identical while passing another that was
114 levels out. So both versions are now rendered and compared, and a page that would
change is handed over unflattened. Six of the eleven flatten, five fall back, and all
eleven render identically to their reconstruction.

### When the model drops an image

It is asked for that image back by anchor rather than by rewrite. Requesting a corrected
copy of the whole document meant regenerating seventy thousand characters to add four
`<img>` tags, and the same four came back missing every round. Instead the model returns
one line per lost image — the marker and a short run of text copied from its own output —
and the insertion is done locally: the anchor must appear exactly once or it is refused
and counted. Whichever attempt lost least is kept.

### What it will and will not do

The prompt forbids changing any visible text, dropping any image marker, or moving far
from the original palette and type scale; it asks for flex/grid layout, semantic tags,
hover and focus states, transitions, responsiveness down to 480px, and
`prefers-reduced-motion` support. **None of that is enforced** — it is a model following
instructions. Run `tools/audit_enhanced.py` on the result: it diffs visible words and
embedded images against the reconstruction and says what actually moved.

Output varies between runs of the same page. Axion came back once with no text lost and
once 29 words short, same prompt, different model. Treat the enhanced page as a draft,
and treat the audit as the thing that tells you which draft you got.

### Measured across all eleven references

| | |
|---|---|
| Every image preserved | **11 of 11 pages** |
| No visible text lost | **10 of 11** (axion lost 29 word occurrences) |
| Images dropped in total | **0**, so the repair pass never had to fire |
| `position: absolute` | 78 → 7, 128 → 4, 118 → 6, typical |
| Flex/grid containers | 9 to 33 per page, from 2 |

| | |
|---|---|
| `GEMINI_API_KEY` | required, from [AI Studio](https://aistudio.google.com/apikey) |
| `GEMINI_MODEL` | optional; defaults to a flash model that answers |

`.env` is gitignored. No new dependency — the call goes through `urllib`.

**On models and quota.** Availability is genuinely unreliable and worth knowing before
you debug the wrong thing. `gemini-2.5-pro` is still listed by the models endpoint while
returning 404 to any key created after it was retired; pro models 429 on the free tier
entirely; and a flash model returned 503 to a full page four times running while
answering a one-line prompt instantly. The free tier allows **20 requests per day per
model**, so enhancing eleven pages meant rotating across four of them. `--list-models`
asks the API rather than trusting documentation, retries honour the server's own
`retryDelay`, and an exhausted daily allowance fails immediately instead of backing off
against something no backoff can fix.

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

- **Pill-shaped controls are not promoted.** Every strongly-rounded surface across the
  reference pages has zero text runs attached to it, including a 1203×68 pill nav bar, so
  the button scorer returns 0 before it starts. The scoring is not rejecting them; the
  label never reaches them. Diagnosed, not fixed, and probably the best ratio of value to
  effort left in the project.
- **Icons are absorbed into the text layer.** OCR reads a glyph-like icon as characters —
  one renders as the literal string `83` — so the icon is neither drawn nor available as
  an element.
- **Ground truth exists for one page.** Component accuracy is therefore a single-page
  measurement, and generalisation is unproven. There are now eleven reference images but
  only one has a hand-read component list, so "24/25" is a claim about one page. Adding
  ground truth is the highest-value contribution to this repo.
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

### Not limitations, though they look like them

- **Words that come back cut off.** The logistics page reconstructs `pertise`, `ogistics`
  and `olutions`. Every one of those starts at x≈890, which is the left edge of a white
  card the design lays over the panel behind it: the `Ex` and the `L` are genuinely not
  in the pixels, and a reader of the screenshot sees the same thing. Completing them
  would mean inferring glyphs from context, which is invention rather than measurement,
  so the engine leaves them. The enhancement prompt forbids "fixing" them for the same
  reason, and `tools/audit_enhanced.py` reports it as *added* text when a model does it
  anyway.
