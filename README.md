# Mental Fatigue Detection from Wearable Biosignals: Generalisation to Unseen Workers

Code accompanying the paper *"Mental Fatigue Detection from Wearable Biosignals:
Generalisation to Unseen Workers under Subject-Independent Evaluation"*.

The package reproduces every analysis in the paper on the public
[MEFAR dataset](https://doi.org/10.17632/z3g26tphnv.5) (Derdiyok, Akbulut & Catal, 2024):

* **P1** – the sample-level holdout used in prior work on the preprocessed release;
* **P2** – participant-grouped cross-validation on a 1-Hz table rebuilt from the raw files;
* the partitioning, session-proxy and windowed-feature controls;
* the seven-configuration ablation, gating/masking analysis and computational cost;
* participant-bootstrap confidence intervals and session-level permutation tests.

## 1. Data

Download MEFAR from Mendeley Data (`https://doi.org/10.17632/z3g26tphnv.5`). Two inputs are used:

| Input | Location inside the download | Used by |
|---|---|---|
| Preprocessed release `MEFAR_DOWN.csv` | `MEFAR_preprocessed/MEFAR_preprocessed/` | P1 (script 01), script 09 |
| Raw folder `MEFAR/` | `MEFAR_raw_data/MEFAR/` | P2 (scripts 02, 09, 11) |

The raw folder must contain `general_info.xlsx` and `subject_<k>/1.morning`,
`subject_<k>/2.evening`, each with `BVP.csv`, `EDA.csv`, `TEMP.csv`, `ACC.csv`,
`HR.csv` (Empatica E4) and `EEG.csv` (NeuroSky MindWave).

## 2. Installation

Python 3.10–3.12.

```bash
# conda (recommended on Windows)
conda create -n mefar python=3.11 -y
conda activate mefar
pip install -r requirements.txt

# or venv (macOS / Linux)
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Running everything

```bash
bash run_all.sh  /path/to/MEFAR  /path/to/MEFAR_DOWN.csv  results          # macOS / Linux
run_all.bat "C:\path\to\MEFAR" "C:\path\to\MEFAR_DOWN.csv" results         # Windows
```

All scripts write to `--out-dir` (default `results/`); P2 outputs go to `results/p2/`
and the windowed baseline to `results/windowed/`.

| Script | Paper output | Runtime (CPU, indicative) |
|---|---|---|
| `01_protocol1_sample_level.py --release ...` | Table 7, Figure 3 | ~20 min |
| `02_reconstruct_subjectwise.py --raw-root ...` | Reconstructed P2 table (Section 4.6) | ~3 min |
| `03_protocol2_conventional.py` | Table 8 (conventional models and Random Forest) | ~20 min |
| `04_protocol2_deep.py` | Table 8 (deep models), Table 14 | ~15 min |
| `05_session_proxy.py` | Table 11 | ~20 min |
| `06_ablation_statistics.py` | Table 12 and statistical tests | ~2–3 h |
| `07_gating_and_masking.py` | Table 13 | ~10 min |
| `08_partitioning_control.py` | Table 10 | ~3 min |
| `09_descriptives.py --release ... --raw-root ...` | Tables 3–4, EEG check (Section 4.4), parameter counts | ~2 min |
| `10_uncertainty.py` | Table 9 | ~10 min |
| `11_windowed_features.py --raw-root ...` | Table 15 | ~10 min |

Scripts 03 and 05 cache results per model and fold (`results/p2/cache/`), so they
can be interrupted and resumed, or run piecewise with `--models` and `--folds`.
`06_ablation_statistics.py --stats-only` recomputes the statistics from saved
fold results without retraining; `10_uncertainty.py --models ...` processes a
subset of models and merges the results.

Quick smoke test with reduced settings (not for reporting):
```bash
MEFAR_EPOCHS=2 MEFAR_ABLATION_SEEDS=42 bash run_all.sh /path/to/MEFAR /path/to/MEFAR_DOWN.csv results_smoke
```

## 4. Protocols in brief

* **P1** – stratified random 80:20 split of the cleaned release (seed 42).
  Participants appear in both partitions; included only because prior work used it.
* **P2** – five-fold participant-grouped cross-validation on the reconstructed table.
  Conventional models: `StratifiedGroupKFold(shuffle=True, random_state=42)`;
  deep models: `GroupKFold` with a participant-grouped inner validation split
  (20% of training participants, `random_state = seed + fold`). Participant overlap
  is checked for every fold and raises an error if found.
* **Reconstruction** – E4 files start with two header rows (start timestamp,
  sampling rate); these are excluded from the signal values. Signals are linearly
  interpolated onto a 1-Hz grid from each file's start and truncated to the shortest
  stream of the session.
* **Labels** – a session is fatigued if its Chalder Fatigue Scale score (Likert,
  items 0–3) is ≥ 12, the threshold defined by the dataset authors; every second of
  the session inherits that label.
* **LSTM** – under P1, windows are formed over rows already shuffled by the random
  split (reproducing prior practice; they are *not* temporal sequences). Under P2,
  windows are chronological and never cross a session boundary.
* **Uncertainty** – pooled out-of-fold ROC-AUC (scores rank-normalised within each
  fold), participant bootstrap 95% CI, and a label permutation test over the 46 sessions.

## 5. Reproducibility notes

* **Randomness.** All scripts fix Python/NumPy/TensorFlow seeds. Neural-network
  results can differ in the second or third decimal across hardware and TensorFlow
  versions (GPU/oneDNN kernels are not bit-deterministic); tree and linear models
  reproduce exactly.
* **Parameter counts.** MF-AFEN has 23,745 parameters in scripts 01/04 and 24,001
  in the ablation and masking implementation, which adds one batch-normalisation
  layer after the 64-unit neurophysiological layer.
* **EEG channel.** `09_descriptives.py --raw-root ...` reports the structure of the
  EEG band powers in the distributed files (Section 4.4 of the paper).
* **`--legacy-e4-parsing`** (script 02) reproduces an earlier development version
  that did not skip the E4 header rows. It is kept for transparency only and is not
  used for any result in the paper.

## 6. Repository layout

```
src/        shared code: configuration, data loading, raw-file reading, models, metrics
scripts/    one script per analysis (01–11)
run_all.sh  / run_all.bat   full pipeline
```

## 7. Citation

If you use this code, please cite the paper (reference to be added on publication)
and the MEFAR dataset:

> Derdiyok, S., Akbulut, F. P., & Catal, C. (2024). Neurophysiological and biosignal
> data for investigating occupational mental fatigue: MEFAR dataset. *Data in Brief*,
> 52, 109896. https://doi.org/10.1016/j.dib.2023.109896

## 8. Licence

Code: MIT (see `LICENSE`). The MEFAR dataset is distributed by its authors under its
own licence.
