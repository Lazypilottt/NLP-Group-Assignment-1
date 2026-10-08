from __future__ import annotations

import os
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "dependency_tree_screenshots"

SENTENCES = [
    "The cat sat on the mat.",
    "She eats a green salad.",
    "I saw the man with a telescope.",
]

PARSED_EXAMPLES = [
    {
        "sentence": "The cat sat on the mat .",
        "tokens": ["The", "cat", "sat", "on", "the", "mat", "."],
        "tags": ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "PUNCT"],
        "arcs": [
            (0, 1, "det"),
            (1, 2, "nsubj"),
            (2, 2, "root"),
            (3, 5, "case"),
            (4, 5, "det"),
            (5, 2, "obl"),
            (6, 2, "punct"),
        ],
    },
    {
        "sentence": "She eats a green salad .",
        "tokens": ["She", "eats", "a", "green", "salad", "."],
        "tags": ["PRON", "VERB", "DET", "ADJ", "NOUN", "PUNCT"],
        "arcs": [
            (0, 1, "nsubj"),
            (1, 1, "root"),
            (2, 4, "det"),
            (3, 4, "amod"),
            (4, 1, "obj"),
            (5, 1, "punct"),
        ],
    },
    {
        "sentence": "I saw the man with a telescope .",
        "tokens": ["I", "saw", "the", "man", "with", "a", "telescope", "."],
        "tags": ["PRON", "VERB", "DET", "NOUN", "ADP", "DET", "NOUN", "PUNCT"],
        "arcs": [
            (0, 1, "nsubj"),
            (1, 1, "root"),
            (2, 3, "det"),
            (3, 1, "obj"),
            (4, 6, "case"),
            (5, 6, "det"),
            (6, 3, "nmod"),
            (7, 1, "punct"),
        ],
    },
]


def make_standalone_svg(example: dict, output_path: Path) -> None:
    """Generate a clean, standalone SVG dependency tree without external dependencies."""
    tokens = example["tokens"]
    tags = example["tags"]
    arcs = example["arcs"]

    n = len(tokens)
    word_spacing = 110
    start_x = 70
    base_y = 180
    width = max(600, start_x * 2 + (n - 1) * word_spacing)
    height = 240

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color: #ffffff; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
        '<defs>',
        '  <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        '    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#4A5568" />',
        '  </marker>',
        '</defs>',
    ]

    word_x = [start_x + i * word_spacing for i in range(n)]

    # Draw words and POS tags
    for i, (w, t) in enumerate(zip(tokens, tags)):
        x = word_x[i]
        svg_parts.append(f'<text x="{x}" y="{base_y}" text-anchor="middle" font-size="16" font-weight="bold" fill="#1A202C">{w}</text>')
        svg_parts.append(f'<text x="{x}" y="{base_y + 20}" text-anchor="middle" font-size="12" font-weight="bold" fill="#718096">{t}</text>')

    # Draw dependency arcs
    for dep_idx, head_idx, label in arcs:
        if dep_idx == head_idx or label == "root":
            continue
        x1 = word_x[head_idx]
        x2 = word_x[dep_idx]
        dist = abs(head_idx - dep_idx)
        arc_height = min(90, 30 + dist * 18)
        y_peak = base_y - 25 - arc_height

        cx = (x1 + x2) / 2
        path_d = f"M {x1} {base_y - 20} Q {cx} {y_peak} {x2} {base_y - 20}"

        svg_parts.append(f'<path d="{path_d}" fill="none" stroke="#4A5568" stroke-width="1.6" marker-end="url(#arrow)" />')
        label_y = y_peak + (arc_height * 0.35)
        svg_parts.append(f'<rect x="{cx - 24}" y="{label_y - 10}" width="48" height="14" fill="#ffffff" rx="3"/>')
        svg_parts.append(f'<text x="{cx}" y="{label_y + 1}" text-anchor="middle" font-size="11" font-weight="600" fill="#2B6CB0">{label}</text>')

    svg_parts.append('</svg>')
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")


def render_all():
    OUTPUT_DIR.mkdir(exist_ok=True)

    use_spacy = False
    try:
        import spacy
        from spacy import displacy
        nlp = spacy.load("en_core_web_sm")
        use_spacy = True
    except Exception:
        use_spacy = False

    for idx, ex in enumerate(PARSED_EXAMPLES, start=1):
        out = OUTPUT_DIR / f"example_{idx}_dependency_tree.svg"
        if use_spacy:
            doc = nlp(ex["sentence"])
            svg = displacy.render(doc, style="dep", options={"compact": False, "distance": 120, "bg": "#ffffff"}, jupyter=False)
            out.write_text(svg, encoding="utf-8")
        else:
            make_standalone_svg(ex, out)
        print(f"Saved: {out}")

    print(f"\nAll dependency tree screenshots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    render_all()
