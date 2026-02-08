# Music Transformer for Symbolic Music Modeling

This project implements and compares two Transformer variants for symbolic music modeling:

- **Baseline Transformer** with absolute sinusoidal positional encoding  
- **Music Transformer** with relative positional attention  

The task is **autoregressive next-token prediction** over event-based symbolic music token streams.

---

## Datasets

- **JSB Chorales**  
  SATB symbolic music dataset

- **Piano-e**  
  MIDI-only subset of the MAESTRO dataset

---

## Project Pipeline (Fixed Order)

The pipeline must be executed in the following order:

1. Ensure the `outputs/` directory structure exists (**no deletion**).
2. Train baseline Transformer on JSB.
3. Train relative Transformer on JSB.
4. Train baseline Transformer on Piano-e.
5. Train relative Transformer on Piano-e.
6. Aggregate metrics and generate plots.
7. Run a JSB-only temperature sweep:
   - `0.8`
   - `1.0`
   - `1.2`

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
````

> **Note:** CUDA-enabled GPU is required.

---

## Key Hyperparameters

| Parameter     | Value |
| ------------- | ----- |
| d_model       | 128   |
| num_layers    | 4     |
| num_heads     | 4     |
| d_ff          | 256   |
| dropout       | 0.1   |
| batch_size    | 32    |
| learning_rate | 3e-4  |
| grad_clip     | 1.0   |
| block_size    | 512   |

---

## Training Epochs

* **JSB Chorales:** 30 epochs
* **Piano-e:** 10 epochs

---

## Output Directory Structure

```text
outputs/
├── jsb/
│   ├── baseline/
│   └── relative/
├── piano_e/
│   ├── baseline/
│   └── relative/
├── comparison/
│   ├── metrics/
│   └── plots/
```

---

## Per-Dataset / Per-Model Outputs

Each dataset–model combination produces:

* `*_val_nll.png` — Validation negative log-likelihood curve
* `*_metrics.json` — Train / validation / test NLL and PPL
* `*_sample.png` — Pianoroll sample generation
* `*_similarity.png` — Self-similarity heatmap
* `*_stability.png` — Train vs validation NLL comparison

---

## Cross-Model Comparison Outputs

Located in `outputs/comparison/`.

### Metrics

* `metrics/summary.json`
* `metrics/summary.csv`
* `metrics/per_split.csv`
* `metrics/per_split.md`

### Plots

* `plots/comparison_nll.png`
* `plots/comparison_ppl.png`

---

## Temperature Sweep (JSB Only)

Generated samples and metrics for each temperature.

### Samples

* `baseline_temp_0.8.png`
* `baseline_temp_1.0.png`
* `baseline_temp_1.2.png`
* `relative_temp_0.8.png`
* `relative_temp_1.0.png`
* `relative_temp_1.2.png`

### Metrics

* `baseline_temperature_metrics.csv`
* `relative_temperature_metrics.csv`

---

## Troubleshooting

### NumPy 2.x Compatibility Warning

If you encounter NumPy 2.x compatibility issues with PyTorch, run:

```bash
python -m pip install "numpy<2"
```

