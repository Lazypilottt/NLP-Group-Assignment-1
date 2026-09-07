import os
from pathlib import Path

import spacy
from spacy import displacy


OUTPUT_DIR = Path(__file__).resolve().parent / "dependency_tree_screenshots"

SENTENCES = [
    "The cat sat on the mat.",
    "She eats a green salad.",
    "I saw the man with a telescope.",
]


def make_dependency_svg(sentence: str, output_path: Path) -> None:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # Use a wide, clean layout with generous spacing so the screenshot looks good in reports.
    options = {
        "compact": False,
        "distance": 130,
        "bg": "#ffffff",
        "color": "#1f2937",
        "font": "Arial",
    }

    svg = displacy.render(doc, style="dep", options=options, jupyter=False)
    output_path.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    for idx, sentence in enumerate(SENTENCES, start=1):
        out = OUTPUT_DIR / f"example_{idx}_dependency_tree.svg"
        make_dependency_svg(sentence, out)
        print(f"Saved: {out}")

    print(f"\nAll dependency tree screenshots saved in: {OUTPUT_DIR}")
