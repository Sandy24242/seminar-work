"""Writes colab/Run_Dental_Caries_AI_Website.ipynb from tools/colab_cells.py.   Run: python tools/make_colab_notebook.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colab_cells import CELLS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INTRO = """# Dental Caries AI Assessment System — run the website in Google Colab

**Before you start**
1. `Runtime → Change runtime type → T4 GPU` (CPU also works, but model loading and analysis are slower).
2. `Dental_Caries_AI_Website.zip` is in `MyDrive/dentalcaries/` (or `MyDrive/`).
3. The trained models are in `MyDrive/dentalcaries/pipeline_stage1_to_4_corrected/` (written by the training notebook).

Run the cells **from top to bottom**. Use **Option A or Option B** in Step 7 to open the website.
After a disconnect: run Steps 1–7 again (Steps 3–5 are quick on a re-run).

*Academic proof of concept — not a clinical diagnosis.*"""

NOTES = {
    "proxy": "If the Option A link shows a blank page or keeps reconnecting, use Option B below.",
    "tunnel": ("Option B gives a public `https://….trycloudflare.com` link. Anyone with the link can use the app while this "
               "session runs — do not share it, and stop it (Step 9) when you are finished."),
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(keepends=True)}


def main():
    cells = [md(INTRO)]
    for n, (sid, title, src) in enumerate(CELLS, 1):
        cells.append(md(f"## {title}" + (f"\n{NOTES[sid]}" if sid in NOTES else "")))
        cells.append(code(src))
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = os.path.join(ROOT, "colab", "Run_Dental_Caries_AI_Website.ipynb")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(nb, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
