# Simplex Take-Home — Matthew Fong

## Writeup

The main submission is in [`writeup/WRITEUP.pdf`](writeup/WRITEUP.pdf).  
The source markdown is [`writeup/WRITEUP.md`](writeup/WRITEUP.md).  
All figures referenced in the writeup are in [`writeup/figures/`](writeup/figures/).

## Repository Structure

```
src/
  train.py          — trains the transformer and saves model weights
  analyze.py        — loads the trained model and produces all figures

tests/
  test_task1_dataset.py   — verifies non-ergodic dataset construction
  test_task1_training.py  — verifies training loop on a short debug run

models/
  model.pt          — saved model weights (5000 steps, final loss 0.9438)
  model.json        — loss history

writeup/
  WRITEUP.md        — full writeup (Tasks 1–4)
  WRITEUP.pdf       — rendered PDF
  figures/          — all analysis figures
  pandoc_style.css  — stylesheet for PDF generation
```

## Reproducing the Results

The code depends on [`fwh_core`](https://github.com/Astera-org/factored-reps) and
[TransformerLens](https://github.com/TransformerLensOrg/TransformerLens). Install both:

```bash
pip install "git+https://github.com/Astera-org/factored-reps.git"
pip install transformer-lens scikit-learn matplotlib
```

All scripts are run from the project root.

**Retrain the model** (optional — `models/model.pt` is already included):

```bash
python src/train.py
# saves → models/model.pt, models/model.json
```

**Reproduce all figures:**

```bash
python src/analyze.py
# loads models/model.pt, saves → writeup/figures/
```

**Regenerate the PDF:**

```bash
cd writeup
pandoc WRITEUP.md -o WRITEUP.html --standalone --embed-resources --css pandoc_style.css
# open WRITEUP.html in a browser and print to PDF
```
