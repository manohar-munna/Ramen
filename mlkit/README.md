# mlkit — the UI component detector

A YOLO-nano detector (~2.6M parameters, ~15ms per page on GPU) that finds interactive
elements in a screenshot. Not a VLM and not a language model: a convolutional detector
returning boxes, classes and confidences.

## Where data goes

Everything lives under `datasets/` and `models/`, both gitignored.

```text
datasets/
├── ui-elements-hf/       # downloaded real screenshots (mlkit/fetch_dataset.py)
│   ├── train/{images,labels}
│   ├── val/{images,labels}
│   └── test/{images,labels}
├── generated-pages/      # optional, produced by mlkit/generate_pages.py
│   ├── images/{train,val}
│   └── labels/{train,val}
└── prepared/             # what the trainer reads (mlkit/prepare_data.py writes it)

models/
└── ui-detector.pt        # trained weights
```

**To add your own data**, drop it in `datasets/<your-set>/` in YOLO format — either
`train/images` + `train/labels` or `images/train` + `labels/train`, both are accepted —
then point `prepare_data.py` at it. If its class ids differ, add a mapping to
`taxonomy.py` rather than renumbering the files.

## Pipeline

```bash
python mlkit/fetch_dataset.py                 # ~314MB, 691 annotated real screenshots
python mlkit/prepare_data.py                  # clean, remap, merge -> datasets/prepared
python mlkit/train.py --epochs 45 --imgsz 960 # ~15 min on a laptop GPU
python mlkit/predict.py <image> --conf 0.25   # writes an annotated PNG to scratch/
```

`generate_pages.py` renders randomised landing pages and takes labels straight from the
DOM. It is **off by default** — see below.

## What was learned

**The real-screenshot dataset is what made this work.** An earlier model trained only on
generated pages scored 0.995 mAP on its own validation split and, on a real page, missed
every button, every nav link and the entire headline. The failure was domain gap, not
capacity.

**Generated pages do not transfer, and mixing them in creates a shortcut.** Trained
together, the structural classes again scored 0.995 on generated validation and produced
*zero* headings, cards or images on a real page at any confidence down to 0.08. The two
domains are trivially separable, so the model learned to tell them apart and apply a
different prior to each. Real data only is the better model: cleaner boxes, higher
confidence, no spurious structure.

**The downloaded dataset only annotates things you can click.** It has 15.5k `link` and
5.1k `button` boxes and *zero* `image`, `heading`, `card` or `nav`. Its labels come from
the DOM, so they carry the DOM's shape rather than the page's — an anchor wrapping a
button yields two boxes on the same pixels, and handlers on empty elements yield boxes on
nothing. `prepare_data.py` de-duplicates and drops those.

## What the detector is and is not for

| | primary | why |
|---|---|---|
| Is this a button, link or input? | **detector** | measured mAP50 0.47 / 0.41 / 0.29 on held-out real pages, against hand-written rules that were guessing |
| Where exactly is it, what colour, what radius? | **engine** | the engine measures geometry from pixels; a detector's boxes are approximate by construction |
| Is this a heading, card, image or nav? | **engine** | the detector has no real supervision for these at all |

So the detector decides *what* interactive things are, and the engine decides *where
everything is* and what the structural elements are. Confidences from both are combined
in `mlkit/detect.py`.
