"""The class list the detector predicts, and how each source dataset maps onto it.

Two sources feed this model and neither covers the whole job on its own.

The HuggingFace set (YashJain/UI-Elements-Detection-Dataset) is real screenshots of real
sites, annotated from the DOM, and it is the only source of genuine interactive elements
-- buttons and links as they actually appear on Amazon, Coursera, Figma. What it does
not annotate is structure: headlines, paragraphs, photographs and cards are simply
absent from its labels, because it was built to find things you can click.

The generated pages supply exactly that missing half, with exact boxes, and cost nothing
to produce. What they cannot supply is the messiness of the real web.

So the taxonomy is the union, each source contributing the classes it actually
annotates, and neither is asked to supervise a class it never labels.
"""

# What the engine needs to tell apart, in the order the model predicts them.
CLASSES = [
    "button",    # a control you press
    "link",      # navigational text or a linked tile
    "input",     # a field you type into
    "heading",   # display or section type
    "text",      # body copy
    "card",      # a bounded container grouping other things
    "image",     # photographic or illustrative content
    "icon",      # a small glyph, usually paired with a label
    "nav",       # the top bar as a whole
    "badge",     # a small pill of status text
]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}

# --- HuggingFace set -------------------------------------------------------------
# Its own ids, in its dataset.yaml order.
HF_NAMES = ["link", "button", "input", "select", "textarea", "label", "checkbox", "radio",
            "dropdown", "slider", "toggle", "menu_item", "clickable", "icon", "image", "text"]

# Only the classes it annotates densely enough to learn from. `clickable` is dropped as
# a catch-all that overlaps everything else, and the form controls it barely contains
# (checkbox, radio, slider, toggle) are folded into `input` rather than given their own
# thin class. `label` is its form-label class, which is body text here.
HF_TO_OURS = {
    "link": "link",
    "button": "button",
    "input": "input",
    "select": "input",
    "textarea": "input",
    "checkbox": "input",
    "radio": "input",
    "dropdown": "input",
    "slider": "input",
    "toggle": "input",
    "menu_item": "link",
    "label": "text",
    "text": "text",
    "icon": "icon",
    "image": "image",
    # "clickable" deliberately unmapped: it labels anything with a handler, so it
    # collides with button, link and card at once and teaches the model nothing.
}

# --- generated pages --------------------------------------------------------------
# These already speak the taxonomy, but only the structural half is taken from them.
# Letting them also supervise buttons and links would drown the real examples in
# synthetic ones and teach the generator's idiom back to the model.
GEN_CLASSES = ["heading", "text", "card", "image", "icon", "nav", "badge"]


def hf_id_to_ours(hf_id: int):
    """Our class id for a HuggingFace class id, or None if that class is dropped."""
    if not (0 <= hf_id < len(HF_NAMES)):
        return None
    ours = HF_TO_OURS.get(HF_NAMES[hf_id])
    return CLASS_ID[ours] if ours else None
