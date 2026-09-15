# Where the reference images come from

## `site-*.png`

Five real-site captures taken from
[YashJain/UI-Elements-Detection-Dataset](https://huggingface.co/datasets/YashJain/UI-Elements-Detection-Dataset)
(Apache-2.0), all 1920x941. They are here as test fixtures: each one breaks something
different, which is the only reason it was chosen.

| file | captured site | what it exercises |
|---|---|---|
| `site-bandcamp.png` | bandcamp.com | dense tile grid — containment and sibling grouping |
| `site-netflix.png` | netflix.com | dark theme, photographic cards — photo/flat segmentation |
| `site-stanford.png` | stanford.edu | text-heavy institutional layout — typography and headings |
| `site-whatsapp.png` | web.whatsapp.com | pill-shaped controls — the button promotion gap |
| `site-yelp.png` | yelp.com | search field and listings — input and control promotion |

The pages themselves belong to their respective owners. Nothing here is redistributed as
content; they are screenshots used to measure a reconstruction pipeline.

## The rest

`arda-guler-editorial`, `art-hall-exhibition`, `axion-logistics`, `reddit-ads-hero`,
`reelers-saas-hero` and `woodnest-hero` are hero and landing sections collected while
building the engine.

## Adding your own

Drop a PNG in this directory and the fidelity tools pick it up with no further setup. To
score component accuracy on it as well, add `benchmarks/ground_truth/<same-name>.json` —
see `reddit-ads-hero.json` for the shape.
