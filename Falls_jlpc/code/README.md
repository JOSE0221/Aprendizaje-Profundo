# Radar Fall Detection on the Glasgow INSHEP Dataset

End-to-end pipeline that frames fall detection as a one-class anomaly-detection
problem. A convolutional β-VAE is trained on five normal activities (walking,
sitting down, standing up, picking up, drinking) and detects falling at inference
as an off-manifold event. The repository handles every quirk of the seven INSHEP
collection sessions — including subject-ID collisions across sessions, the
fall-free NG Homes session, and the limited West Cumbria P42 data — with
runtime-asserted invariants and a 20-test pytest suite.

---

## 1. Repository layout

This `code/` directory must sit alongside the seven INSHEP session folders:

```
Falling/
├── 1 December 2017 Dataset/
├── 2 March 2017 Dataset/
├── 3 June 2017 Dataset/
├── 4 July 2018 Dataset/
├── 5 February 2019 UoG Dataset/
├── 6 February 2019 NG Homes Dataset/
├── 7 March 2019 West Cumbria Dataset/
├── code/                              ← this folder, 21 files, all flat
├── AutoEncoders.pdf
└── Document for all datasets - 03092019.pdf
```

Inside `code/` (every file at the top level — no subdirectories):

| File                            | Role                                                                           |
| ------------------------------- | ------------------------------------------------------------------------------ |
| `README.md`                     | this file                                                                      |
| `Makefile`                      | one-command targets for every pipeline step                                    |
| `requirements.txt`              | five external Python packages (audited)                                        |
| `default.yaml`                  | reference config (informational; CLI flags override)                           |
| **Library modules**             |                                                                                |
| `sessions.py`                   | hardcoded session metadata, exact folder names, has-falls flag                 |
| `demographics.py`               | per-(session, local_id) demographics keyed against ID collisions               |
| `labels.py`                     | robust `.dat` filename parser (8 naming variants tested)                       |
| `manifest.py`                   | builds master CSV joining files + labels + demographics + quality flags       |
| `preprocess.py`                 | raw I/Q → range FFT → clutter removal → STFT → standardized spectrogram        |
| `splits.py`                     | global-subject-disjoint splits with zero-train-falls invariant                 |
| `dataset.py`                    | PyTorch `Dataset` with disk + memory cache and augmentations                   |
| `model.py`                      | convolutional β-VAE (7.7 M params)                                             |
| `baselines.py`                  | binary CNN and 6-class CNN supervised baselines                                |
| `trainer_vae.py`                | VAE training loop with KL annealing and per-session AUROC monitoring           |
| `evaluator.py`                  | three-strategy threshold tuning + per-activity / per-session / fairness audit  |
| **Entry-point scripts**         |                                                                                |
| `build_manifest.py`             | walk all 7 sessions → manifest.csv                                             |
| `preprocess_all.py`             | precompute spectrograms + scalar features to `cache/`                          |
| `data_summary.py`               | post-preprocess: text summary + 6 plots (max-velocity baseline, etc.)          |
| `train_vae.py`                  | train the conv-VAE anomaly detector                                            |
| `train_baseline.py`             | train binary or multiclass supervised baseline                                 |
| `evaluate.py`                   | full evaluation with all breakdowns + SVM-paper baseline comparison            |
| **Tests**                       |                                                                                |
| `test_critical_invariants.py`   | 20 pytest tests — filename parsing, ID collisions, splits, etc.                |

Imports between files work because Python adds the current directory to
`sys.path` automatically. There is no package layout, no `__init__.py`, no
`PYTHONPATH` to set, and no install step for the local code itself.

---

## 2. System requirements

### 2.1 Operating system

Tested on macOS 13+ (Apple Silicon) and Ubuntu 22.04. Windows works under WSL2.

### 2.2 Python

Python **3.10 or newer** (uses PEP 604 union types like `int | None`). Confirm:

```bash
python --version          # should print 3.10.x or higher
```

### 2.3 Hardware

| Resource             | Minimum                          | Recommended                         |
| -------------------- | -------------------------------- | ----------------------------------- |
| RAM                  | 4 GB                             | 16 GB                               |
| Disk (cache + runs)  | ~3 GB                            | 10 GB                               |
| GPU                  | none (CPU works, slowly)         | NVIDIA RTX 2060 or Apple Silicon M1+ |
| Training time / run  | ~6 h (CPU)                       | ~35 min (RTX 2060) / ~1.5 h (M1 Pro) |

GPU detection is automatic. Priority order at runtime:

1. CUDA — used if `torch.cuda.is_available()` returns `True`
2. Apple MPS (Metal Performance Shaders) — used on Apple Silicon if CUDA unavailable
3. CPU — fallback

### 2.4 External Python packages

Five packages, all imported by name in the codebase (audited via grep). Listed
in `requirements.txt`:

| Package        | Used by                                                                |
| -------------- | ---------------------------------------------------------------------- |
| `numpy`        | pervasive                                                              |
| `scipy`        | `preprocess.py` — `butter`, `filtfilt`, `stft`, `get_window`           |
| `scikit-learn` | metrics in `evaluator.py`, `trainer_vae.py`, `train_baseline.py`       |
| `torch`        | model, training, evaluation                                            |
| `pytest`       | test suite                                                             |

The Python standard library handles everything else (`argparse`, `csv`,
`dataclasses`, `hashlib`, `json`, `os`, `pathlib`, `random`, `re`, `time`,
`typing`, `unittest`).

---

## 3. Installation

### 3.1 Open a terminal inside `code/`

```bash
cd /path/to/Falling/code
```

All commands below are run from this directory.

### 3.2 Create an isolated environment (strongly recommended)

Pick one of the two options below.

#### Option A — `venv` (built into Python)

```bash
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

#### Option B — `conda`

```bash
conda create -n inshep python=3.11 -y
conda activate inshep
```

### 3.3 Install PyTorch first (recommended)

PyTorch is the only dependency where the install command depends on your
hardware. Pick the matching command from
[pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
Common cases:

```bash
# Apple Silicon (M1/M2/M3/M4) — uses MPS automatically
pip install torch

# Linux/Windows with NVIDIA CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only Linux/Mac Intel
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3.4 Install the remaining four packages

```bash
make install
# equivalent to: pip install -r requirements.txt
```

### 3.5 Verify installation

```bash
python -c "import numpy, scipy, sklearn, torch, pytest; print('all imports OK', torch.__version__)"
make test
```

`make test` should print **`20 passed`**. If any test fails, do not proceed.

---

## 4. Data layout (mandatory)

Verify the seven session folders are present and named exactly as listed
in §1. Folder names are matched by string equality; renaming will break the
loader. The test
`test_critical_invariants.py::test_session_folder_names_match_screenshot`
enforces the canonical names.

Verify the layout:

```bash
ls ..      # should list all 7 dataset folders + code + the two PDFs
```

Expected file counts per session (from the INSHEP Readme):

| Session                                | Files | Subjects | Has falls |
| -------------------------------------- | ----: | -------: | :-------: |
| `1 December 2017 Dataset`              | 360   | 20       | yes       |
| `2 March 2017 Dataset`                 | 48    | 4        | yes       |
| `3 June 2017 Dataset`                  | 162   | 9        | yes       |
| `4 July 2018 Dataset`                  | 288   | 16       | yes       |
| `5 February 2019 UoG Dataset`          | 306   | 17       | yes       |
| `6 February 2019 NG Homes Dataset`     | 301   | 20       | **no**    |
| `7 March 2019 West Cumbria Dataset`    | 289   | 20       | yes       |
| **Total**                              | 1 754 | —        | —         |

Step 5.1 below validates these counts against your actual files.

---

## 5. Running the full pipeline

Run targets in order. Each target is idempotent (you can re-run safely).

### 5.1 Build the manifest CSV

```bash
make manifest
```

What it does: walks all seven session folders, parses every `.dat` filename,
joins each row with the demographics table, writes `manifest.csv` (16 columns,
~1 754 rows on the full corpus).

What to check: the printed summary block should report `actual_files` matching
`expected_files` for each session, and `match: true` everywhere except where
your data is genuinely incomplete. Example output:

```
[manifest] data_root = /Users/jose/.../Falling
[manifest] found all 7 session folders:
           dec2017_uog        → /Users/jose/.../Falling/1 December 2017 Dataset
           ...
[manifest] wrote 1754 rows → manifest.csv

[manifest] per-session counts vs. expected:
{
  "dec2017_uog":    {"actual_files": 360, "expected_files": 360, "match": true, ...},
  "mar2017_uog":    {"actual_files":  48, "expected_files":  48, "match": true, ...},
  ...
  "feb2019_nghomes":{"actual_files": 301, "expected_files": 301, "match": true,
                     "actual_falls": 0,   "expected_falls": null},
  ...
  "TOTAL": {"files": 1754, "global_subjects": 86, "falls": 215,
            "elderly_files": 311, "female_files": 320}
}
```

If `match: false` anywhere except NG Homes (which has zero falls by design),
investigate before proceeding — files are missing or mis-named.

You can inspect the CSV directly:

```bash
head -3 manifest.csv
wc -l manifest.csv     # 1755 = 1 header + 1754 rows
```

Columns: `path`, `session_key`, `session_folder`, `location`, `local_id`,
`global_subject_id`, `activity_code`, `activity_name`, `repetition`, `is_fall`,
`age`, `height_cm`, `dominant_hand`, `gender`, `is_elderly`, `quality_flag`.

### 5.2 Precompute spectrograms

```bash
make preprocess
```

Two outputs per `.dat` file:
1. **Spectrogram** at `cache/<config_hash>/<session_key>/<filename>.npy` —
   shape (128, 256) float32. **Stored UN-NORMALIZED**: the previous
   per-sample normalization destroyed magnitude information that
   distinguishes falls from non-falls. Standardization is applied later
   at training time using statistics computed once over the train fold
   (see §8.2).
2. **Scalar features** at `cache/<config_hash>/<session_key>/<filename>.feat.json` —
   includes `max_doppler_velocity_mps` (the SVM-paper baseline feature)
   and `spectral_energy_db`. Computed BEFORE log-scaling and BEFORE
   normalization, so they preserve absolute magnitude.

Approximate sizes:
- One spectrogram: ~128 KB
- Full corpus: ~225 MB

Estimated runtime: 8–15 minutes on a modern laptop CPU.

### 5.3 Inspect the corpus and the SVM-baseline feature

```bash
make summary
```

Writes `summary/summary.json` and 6 PNG plots to `summary/figures/`:

| File | Shows |
|---|---|
| `01_files_per_session_activity.png` | Stacked bar chart of files × session × activity |
| `02_max_velocity_per_activity.png` | Boxplot of max-Doppler velocity per activity — falls should be visibly higher |
| `03_spectral_energy_per_activity.png` | Boxplot of spectral magnitude per activity |
| `04_age_distribution_per_session.png` | Demographic skew across sessions (lab=young, community=elderly) |
| `05_example_spectrograms.png` | 6×3 grid: 6 activities × 3 sessions of example spectrograms |
| `06_max_velocity_roc_baseline.png` | ROC curve for the SVM-paper feature alone on the full corpus |

The console output prints the per-activity max-velocity distribution and
the **max-velocity-only AUROC** computed across the full corpus. This
is your sanity check: if this number isn't above ~0.85 with our pipeline,
something is wrong with the feature extraction. The SVM paper achieves
F1=0.75 on this feature with a *random* split (subject-leaky); under
our subject-disjoint protocol we expect AUROC ≥ 0.80 cleanly.

### 5.4 Train the conv-VAE anomaly detector

```bash
make train_vae
```

What it does: runs `train_vae.py` with the default config — 200 epochs,
batch size 64, latent dim 64, β annealing from 0 to 1 over 20 epochs,
patience 30. Writes results to `runs/vae_default/`.

Estimated runtime by device:

| Device                    | Runtime  |
| ------------------------- | -------- |
| CPU (8-core laptop)       | ~5–6 h   |
| Apple M1 Pro (MPS)        | ~1.5 h   |
| NVIDIA RTX 2060 (CUDA)    | ~35 min  |
| NVIDIA RTX 4090 (CUDA)    | ~10 min  |

What to check: per-epoch log lines should show training loss decreasing and
val-recon loss on normal samples falling. AUROC should also climb but is
*not* used for early stopping (using fall AUROC for early stopping leaks
fall information into model selection).

```
[model] ConvVAE params=7.70M latent=64
[ep   1] loss=82.3 recon=82.2 kl=0.41 β=0.05 | val_recon_norm=78.4 auroc=0.612
[ep   2] loss=68.1 recon=67.9 kl=0.55 β=0.10 | val_recon_norm=64.8 auroc=0.693
...
[ep  87] loss=15.4 recon=14.2 kl=1.18 β=1.00 | val_recon_norm=14.7 auroc=0.913
[early-stop] no improvement for 30 epochs
```

Output files in `runs/vae_default/`:

```
args.json          — full run configuration
splits.json        — split summary (counts, sessions per fold)
best.pt            — best checkpoint by val recon loss on normals
history.json       — per-epoch training log
```

### 5.5 Evaluate the trained VAE

```bash
make eval_vae
```

What it does: loads `runs/vae_default/best.pt`, scores val and test, tunes
three thresholds on val, applies them on test, and writes
`runs/vae_default/eval/report.json`.

Estimated runtime: 2–5 minutes.

The report contains everything needed for an academic write-up:

```
{
  "validation": {"auroc": ..., "aupr": ..., "tau_youden": ..., "tau_f1": ...,
                 "tau_recall": ...},
  "test": {
    "auroc": ..., "aupr": ...,
    "operating_points": {
      "youden":             {tau, tp, fp, tn, fn, precision, recall,
                             specificity, f1, false_alarm_rate},
      "f1_optimal":         { ... },
      "recall_constrained": { ... }    ← the recommended deployment point
    },
    "per_activity_recon_error": {
      "walking":      {n, mean, std, p10, p50, p90},
      "sitting_down": { ... },
      "standing_up":  { ... },
      "picking_up":   { ... },
      "drinking":     { ... },
      "falling":      { ... }     ← should be highest
    },
    "per_session": {
      "dec2017_uog":    {n, n_falls, auroc, aupr},
      "mar2017_uog":    { ... },
      ...
      "mar2019_cumbria":{ ... }   ← elderly cohort, watch this
    },
    "fairness_audit_at_recall_constrained": {
      "elderly_60+":  {n, n_falls, auroc, recall, false_alarm_rate},
      "younger_<60":  { ... },
      "female":       { ... },
      "male":         { ... }
    }
  }
}
```

Raw scores for downstream plotting: `runs/vae_default/eval/scores.npz`
(numpy arrays — `val_score`, `val_is_fall`, `val_activity`, `val_recon`, plus
`te_*` counterparts).

### 5.6 (Optional) Supervised baselines for comparison

```bash
make train_binary    # fall vs not-fall CNN
make train_multi     # 6-way activity classifier
```

Each writes to `runs/baseline_binary/` or `runs/baseline_multi/` with
`best.pt`, `history.json`, and `test_metrics.json`. Use these to argue that
the unsupervised approach is competitive with — or superior to — supervised
training for cross-subject generalization.

### 5.7 (Optional) Held-out-session evaluation (domain shift)

```bash
make train_vae_holdout    # train on 6 sessions, test on West Cumbria
make eval_holdout
```

This is the cleanest measurement of domain shift in the corpus: training data
comes from young university subjects in lab/community-room settings, test
data is the elderly West Cumbria cohort in an Age UK community centre. Expect
test AUROC to be 5–15 points lower than the standard split — that gap is
your headline domain-shift number.

### 5.8 Cleanup

```bash
make clean        # removes cache/, runs/, manifest.csv, __pycache__, .pytest_cache
```

---

## 6. Configuration overrides

The Makefile is the canonical entry point but every value can be overridden
without editing files:

```bash
# Different data root
make manifest DATA_ROOT=/Volumes/External/Falling

# Different cache location (e.g., on an SSD)
make preprocess CACHE_DIR=/tmp/inshep_cache

# Different seed for reproducibility checks
make train_vae SEED=42
```

Direct script invocation gives you all CLI flags:

```bash
python train_vae.py --help
python train_vae.py --epochs 100 --batch_size 32 --lr 5e-4 --beta_kl 0.5 \
                    --out_dir runs/my_experiment
```

---

## 7. How every quirk of the data is handled

### 7.1 Subject IDs collide across sessions
P36 in `1 December 2017` is a 27-year-old male. P36 in `6 February 2019 NG
Homes` is a 70-year-old female. Different people. The canonical subject
identifier everywhere downstream is `{session_key}::{local_id}`, exposed as
`sessions.global_subject_id()`. Splits use this global key. Verified by
`test_global_subject_ids_distinguish_sessions` and
`test_demographics_correctly_separated_for_collisions`.

### 7.2 Two sessions have no falls — handled
NG Homes (session 6) and West Cumbria (session 7) recorded only 5 activities
because both took place in elderly community settings (NG Homes community
housing, Age UK community centre) where asking participants to fall would
be unsafe. `SESSIONS["feb2019_nghomes"].has_falls` and
`SESSIONS["mar2019_cumbria"].has_falls` are both `False`. Splits and
evaluators tolerate these without dropping or crashing. Note: P08 in NG
Homes contributed three extra A06 (falling) repetitions (the only fall files
in any non-fall session); the manifest summary annotates this.

Verified by `test_nghomes_has_no_falls_flag`, `test_cumbria_has_no_falls_flag`,
and `test_split_handles_session_with_no_falls`.

### 7.3 West Cumbria's P42 has limited data
The Readme notes "with the exception of P42 for which only limited data were
collected." Recorded in `demographics.DATA_QUALITY_FLAGS` and surfaced in
the manifest's `quality_flag` column.

### 7.4 NG Homes P21 and P23 missing one walk repetition
Same mechanism. The manifest carries the flag; no code drops these files.

### 7.5 Folder names with spaces and leading digits
Hardcoded in `sessions.py` exactly as they appear on disk. `pathlib`
handles spaces fine. `test_session_folder_names_match_screenshot`
enforces them.

### 7.6 P08 in `feb2019_uog` AND `feb2019_nghomes`
The Readme is ambiguous about whether these are the same person. Default
behaviour treats them as different global subjects (`feb2019_uog::P08` vs
`feb2019_nghomes::P08`) — the safe choice; if they're actually the same
person, the model just sees them in different folds, which is a fine
robustness test.

### 7.7 Filename naming variations
The parser accepts `1P36A01R1.dat`, `1_P36_A01_R1.dat`, `1P36 A01 R1.dat`,
`1p36a01r1.dat`, `1P36A01R01.dat`, and several other variants. Eight
parameterized tests cover the variants.

---

## 8. Methodology summary

### 8.1 Pipeline

```
raw .dat (complex64 I/Q at 1 kHz PRF, 128 samples/chirp)
    → range FFT (Hamming window, 128-pt)
    → Butterworth high-pass on slow time at 0.0075 Hz (clutter removal)
    → peak-energy range-bin selection in [1.0, 4.5] m, ±5 neighbors
    → STFT (200 ms Hamming, 95% overlap, 256-pt FFT)
    → 20·log10|·| in dB
    → resize to 128 × 256 (Doppler × time) via bilinear interpolation
    → per-sample standardization (zero mean, unit variance)
    → conv-VAE (4 stride-2 blocks, latent dim 64)
    → β-VAE loss with KL annealing (β: 0 → 1 over 20 epochs)
    → anomaly score: α·MSE_recon + γ·KL_divergence
    → threshold tuned on val (Youden / F1-optimal / recall-constrained)
    → final test report with per-activity / per-session / per-demographic breakdowns
```

### 8.2 Seven methodological commitments

1. **Subject-disjoint splits using GLOBAL ids — STRATIFIED BY SESSION.** Most
   published numbers on this dataset use random splits that leak subject
   identity. We use `session_key::local_id` global IDs with runtime
   assertions. *Critically*, the partition is also stratified by session:
   from each of the 7 sessions, 70%/15%/15% of subjects route to
   train/val/test. Without session stratification, the natural correlation
   on this corpus — fall-bearing sessions are lab+young, no-fall sessions
   are community+elderly — pushes all fall-contributing subjects to val/test
   and leaves train composed exclusively of community-elderly subjects, a
   maximally-adversarial domain split that pins val AUROC at ~0.5.

2. **Fall isolation in training.** Every fall sample whose subject lands in
   train is dropped (zero-train-falls invariant, asserted at runtime). The
   autoencoder NEVER sees a fall during training.

3. **KL annealing.** β linearly ramps from 0 to 1 over the first 20 epochs.
   Without annealing, KL collapses early and the model degenerates to a
   deterministic AE.

4. **GLOBAL standardization, NOT per-sample.** Initially we used per-sample
   z-scoring (every spectrogram → mean 0, std 1 individually). That was
   wrong for anomaly detection: it destroys the cross-sample magnitude
   differences that distinguish falls (high total Doppler energy) from
   walks/sits/picks/drinks (low energy). Empirically this pinned val AUROC
   at ~0.6 because the model became invariant to the most informative
   signal. The current pipeline computes (mean, std) once over the
   training fold's spectrograms, saves them to `runs/<name>/global_stats.json`,
   and applies the SAME fixed transform to train, val, and test. Magnitude
   differences are preserved, the SVM-paper feature (max-Doppler velocity)
   becomes recoverable, and the VAE has signal to learn from.

5. **Recall-constrained operating point as deployment recommendation.** Cost
   of a missed fall ≫ cost of a false alarm. Default target recall is 0.95;
   set with `--target_recall` in `evaluate.py`.

6. **Fairness audit always reported.** Per-demographic metrics (elderly vs.
   younger, female vs. male) at the deployment threshold. Aggregate AUROC
   alone is not enough on a corpus this skewed.

7. **SVM-paper baseline reported alongside the VAE.** The Mexico/CENAPRECE
   SVM paper achieves F1=0.75 on a single hand-engineered feature
   (max-Doppler velocity over the 5-second window) under a random,
   subject-leaky split. We compute that feature for every sample in
   `preprocess.py::max_doppler_velocity` and report its AUROC/AUPR on the
   same val/test folds in `report.json`'s `svm_baseline_max_velocity`
   block. The comparison is therefore both head-to-head AND under a
   strictly stricter protocol than the paper's. If the VAE doesn't beat
   this scalar feature, that's the honest finding.

---

## 9. Limitations

- Single-channel radar; no Range-Doppler-Angle. RDA would require
  multi-antenna data (e.g., the TU Delft 5-node distributed radar dataset).
- Simulated frontal falls onto a soft mat. Domain shift to real elderly falls
  is the largest open problem in this literature.
- Per-session radar geometry is not held constant. Mitigated by per-sample
  normalization and peak-energy bin selection.
- Demographics skew young and male except in West Cumbria. The fairness
  audit makes this visible per run.

---

## 10. Tests

```bash
make test
# or directly:
pytest test_critical_invariants.py -v
```

The 20 tests cover:

- Filename parsing — 8 naming variants, plus rejection of garbage
- Canonical local-ID normalization (`P36`, `36`, `P036` → `P36`)
- Global subject ID disambiguation across sessions
- Demographics collision separation (P36 dec2017 ≠ P36 nghomes; P37 dec2017 ≠ P37 cumbria)
- All 7 session folder names match disk
- NG Homes `has_falls=False` flag
- All other 6 sessions `has_falls=True`
- Activity-code completeness (1 → walking, …, 6 → falling)
- Subject-disjoint split correctness
- Zero falls in train fold when fall isolation is enabled
- Splits work cleanly on a session that has no falls
- Cross-session collisions (same `local_id` in two sessions) routed independently

---

## 11. Troubleshooting

**`FileNotFoundError: Missing session folder(s) under …`**
Your `Falling/` directory does not contain all 7 sessions, or one is renamed.
Run `ls ..` from `code/` and verify against §4.

**`ModuleNotFoundError` when running a script**
You're outside the `code/` directory. Always `cd Falling/code` first; Python
needs the current directory to be where the `.py` files live.

**Tests fail with `AttributeError: type 'Path' has no attribute 'absolute'`**
You're on Python < 3.10. Upgrade to 3.10+ — the codebase uses PEP 604 union
types.

**Training loss diverges to NaN**
Reduce learning rate: `make train_vae LR=1e-4`. Also check that
`make manifest` reports `match: true` everywhere — corrupt files can produce
absurd loss values.

**`make manifest` reports `delta_files: -1` for some sessions**
A small number of `.dat` files are missing. The INSHEP corpus as actually
distributed sometimes has 1-file discrepancies vs. the Readme's stated count
(e.g., `jun2017_uog` may show 161/162, `jul2018_uog` 287/288). This is a
data-distribution issue, not a pipeline bug — the pipeline runs fine on the
files that are present. Investigate which file is missing if you want a
clean count:

```bash
ls "../3 June 2017 Dataset" | sort > /tmp/cumbria_files.txt
wc -l /tmp/cumbria_files.txt
# Compare to expected: should be 9 subjects × 6 activities × 3 reps = 162
```

**`make manifest` reports unexpected falls in `feb2019_nghomes`**
The session is documented as no-falls, but the Readme notes that participant
P08 contributed three extra A06 (falling) repetitions. So 3 fall files in
`feb2019_nghomes` is **expected**, and the new manifest summary now annotates
this in the `notes` field.

**`ValueError: range_axis shape (0,) != energy shape (128,)` during preprocess**
This used to happen on real INSHEP files because the original `read_dat`
assumed a 4-complex64 file header that doesn't actually exist (the INSHEP
files are pure I/Q samples). The current `read_dat` defaults to
`header_floats=0` and uses radar parameters from the Readme constants
(`INSHEP_RADAR`). If you ever encounter a corpus with a real header, pass
`header_floats=N` to `read_dat`.

**Diagnostic: inspect what's actually in a `.dat` file**

```bash
cd Falling/code
python -c "
import numpy as np
from preprocess import read_dat
# Pick any .dat file from any session
b, m = read_dat('../1 December 2017 Dataset/1P36A01R1.dat')
print('beats shape:', b.shape, '(should be (~5000, 128) for 5-s files,')
print('                              (~10000, 128) for 10-s walks)')
print('meta:', m)
print('first 5 complex samples:', np.fromfile(
      '../1 December 2017 Dataset/1P36A01R1.dat', dtype=np.complex64)[:5])
"
```

If the file size is close to a multiple of 128 × 8 bytes (1024 bytes per
chirp), the no-header assumption is correct. Most INSHEP files are 5–10 MB.

**MPS errors on Apple Silicon ("operator not implemented for MPS")**
Force CPU fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1 make train_vae`.

**Cache is stale after editing `preprocess.py`**
The cache key includes the `PipelineConfig` hash, so changing config values
auto-invalidates. But changing the *code* of preprocessing functions does
not — delete `cache/` manually after such edits:

```bash
rm -rf cache
make preprocess
```

---

## 12. Reproducibility

Every run records its full configuration (`runs/<name>/args.json`) and split
composition (`runs/<name>/splits.json`). The same `--seed` produces the same
splits across machines. PyTorch determinism is partial (cuDNN nondeterminism
is not fully suppressed); expect tiny deviations between identical runs but
not consequential ones.

To verify reproducibility on your machine:

```bash
make train_vae SEED=17
mv runs/vae_default runs/run_a
make train_vae SEED=17
diff runs/run_a/splits.json runs/vae_default/splits.json   # should be identical
```

---

## 13. Citation

If you use this pipeline, please cite the dataset:

> Fioranelli, F., Shah, S. A., Li, H., Shrestha, A., Yang, S., Le Kernec, J.
> (2019). Radar signatures of human activities. *University of Glasgow.*
> https://researchdata.gla.ac.uk/848/

And the methodological reference:

> Li, H., Le Kernec, J., Mehul, A., et al. (2023). Radar-based human activity
> recognition. *Scientific Reports* 13, 3473.

---

## Appendix A — Quick reference

```bash
# Setup (once)
cd /path/to/Falling/code
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch                    # use the right URL for your hardware
make install

# Verify
make test                            # 20/20 tests must pass

# Run pipeline
make manifest                        # check counts match expected
make preprocess                      # ~10 min, writes cache/ (specs + features)
make summary                         # 6 plots + summary.json — sanity-check feature signal
make train_vae                       # ~35 min on GPU, ~6 h on CPU
make eval_vae                        # writes runs/vae_default/eval/report.json
                                     #   includes svm_baseline_max_velocity block

# Optional comparisons
make train_binary && make train_multi
make train_vae_holdout && make eval_holdout

# Cleanup
make clean
```
