# EffiAuL

**Influence-Guided Approximate Unlearning for Privacy-Preserving Medical Image Analysis**

Research implementation accompanying the manuscript by Mohammad Zavid Parvez, Rafiqul Islam, Lihong Zheng, and Md Zahidul Islam (manuscript dated 6 November 2026).

## Scope of this release

This repository packages the **four-class DenseNet121 experimental workflow** from the supplied research notebook. The manuscript also describes a **primary binary benchmark** with five-seed results; the supplied notebook does **not** contain a separate verified binary-benchmark implementation. Accordingly, this release must not be cited as reproducing the manuscript's binary results or every architecture mentioned in the paper.

The four-class workflow includes baseline transfer learning, gradient-norm influence estimation, adaptive strategy selection, retained-data refinement with teacher–student KL control and forget-set loss ascent, a validation F1 utility gate with full-retraining fallback, and comparator implementations for full retraining, a no-deletion fine-tuning control, SCRUB-style updates, SISA, and IF-Diag. It reports classification, forgetting-related, membership-inference, distributional, runtime, and statistical measures.

**Interpretation:** This is empirical *approximate* unlearning. Jensen–Shannon divergence and membership-inference statistics are diagnostics, **not** certificates of exact deletion, formal privacy guarantees, or evidence of regulatory compliance. The SCRUB, SISA, and IF-Diag implementations are experimental adaptations; do not assume they are drop-in official implementations.

## Installation

Python 3.10 or 3.11 is recommended. TensorFlow installation depends on the operating system and accelerator; see its official installation instructions. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Data

The public chest X-ray images are **not** redistributed here. Obtain and use the source dataset subject to its original terms and applicable research approvals. Supply a directory containing **four** class subdirectories:

```text
Dataset/CovidxRay4Class/
  COVID-19/
  Normal/
  Pneumonia Bacteria/
  Pneumonia Viral/
```

The original notebook also supports `Train/<class>/...` and `Validate/<class>/...` under the same root. It consolidates these before generating a new stratified 70/15/15 split. Do not place an additional `Test` folder in that legacy layout expecting it to be included. Class names are derived alphabetically from the folder names; verify the saved class mapping before comparing with the manuscript.

The manuscript reports 4,852 four-class images (1,248 Normal, 1,200 COVID-19, 1,263 bacterial pneumonia, and 1,141 viral pneumonia). A different image collection, class mapping, preprocessing implementation, or TensorFlow version can change results. **Do not publish or upload patient-identifiable data or generated CSV files containing local image paths.**

## Run

From the repository root, with the dataset prepared:

```bash
python src/effiaul_four_class.py
```

To use custom paths or a single-seed smoke test:

```bash
EFFIAUL_DATA_DIR=/path/to/CovidxRay4Class \
EFFIAUL_RESULTS_DIR=./results/smoke \
EFFIAUL_SEEDS=11 \
python src/effiaul_four_class.py
```

On Windows PowerShell:

```powershell
$env:EFFIAUL_DATA_DIR = 'D:\datasets\CovidxRay4Class'
$env:EFFIAUL_RESULTS_DIR = '.\results\smoke'
$env:EFFIAUL_SEEDS = '11'
python src/effiaul_four_class.py
```

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `EFFIAUL_DATA_DIR` | `./Dataset/CovidxRay4Class` | Image directory |
| `EFFIAUL_RESULTS_DIR` | `./results/four_class` | Results directory |
| `EFFIAUL_SEEDS` | `11,22,33,44,55` | Training seeds |
| `EFFIAUL_N_FORGET` | `20` | Requested training samples |
| `EFFIAUL_FORGET_SELECTION` | `random` | `random`, `low`, or `high` |
| `EFFIAUL_BOOTSTRAP` | `1000` | Bootstrap iterations |

A smoke test checks execution only; it does not reproduce five-seed uncertainty estimates. The complete experiment trains several models per seed and can require substantial GPU time and memory.

## Reproducibility and output

The program saves split CSVs, seed-specific model weights and figures, forgetting and prediction diagnostics, method-level results, aggregated summaries, and significance analyses under the results directory. It uses a fixed stratified split seed (42), default training seeds (11, 22, 33, 44, 55), 224×224 inputs, batch size 32, and a validation utility tolerance of 0.02. The held-out test set is intended for final evaluation only.

Run basic source checks without training:

```bash
python -m unittest discover -s tests -v
```

No original dataset, trained model checkpoints, or recorded experimental outputs were provided with the upload. **The packaged code has been syntax-checked but has not been executed end-to-end on the research dataset in this environment.** Numerical results in the manuscript therefore have not been independently verified here.

## Manuscript and attribution

Parvez, M. Z., Islam, R., Zheng, L., & Islam, M. Z. (2026). *EffiAuL: Influence-Guided Approximate Unlearning for Privacy-Preserving Medical Image Analysis*. Manuscript supplied with this code release. Update this citation after publication with the final journal details and DOI.

For methodological details, limitations, and the distinction between predictive utility and forgetting-specific evidence, consult the manuscript. The implementation is provided for research and reproducibility; it is not a clinical decision-support product.

## License

MIT License (see `LICENSE`). Dataset and third-party model weights remain subject to their respective licenses and conditions.
