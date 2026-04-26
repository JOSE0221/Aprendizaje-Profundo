# Radar Human-Activity Recognition with a CNN

TensorFlow/Keras pipeline that classifies one of six human activities from
a recording of a 5.8 GHz FMCW radar:

| Id | Activity                   |
|----|----------------------------|
| 1  | Walking back and forth     |
| 2  | Sitting down on a chair    |
| 3  | Standing up from a chair   |
| 4  | Picking up an object       |
| 5  | Drinking water             |
| 6  | Falling                    |

Each `.dat` file in `radar_data/` is a text file whose first four records
store the radar parameters (carrier frequency, chirp duration,
samples-per-chirp, bandwidth) and whose remaining records are the complex
baseband samples of the radar echo.

## Project layout

```
radar_cnn_clasif/
├── radar_data/           # symlink to the folder with all .dat files
├── radar_utils.py        # file parsing + signal processing (FFT, Butterworth, STFT)
├── preprocess.py         # batch-convert .dat files into a cached .npz dataset
├── model.py              # CNN architecture
├── train.py              # training script (saves models, plots, metrics)
├── predict.py            # classify a single .dat file with a trained model
├── requirements.txt
└── README.md
```

## Signal-processing pipeline

For every 5-second segment the pipeline performs:

1. **Parse the file** — read the four header lines and the complex
   samples (format `R+Ii` or `R-Ii`). Activity 1 files are 10 s long
   and are split into two non-overlapping 5 s segments.
2. **Reshape to pulses** — `(n_chirps, 128)` matrix where each row is
   one chirp (slow-time × fast-time).
3. **Range-time map** — apply a Hamming window along fast-time and take
   the FFT, keeping the positive half (64 range bins).
4. **Static-clutter removal** — apply a 4-th order high-pass
   Butterworth filter with a 0.0075 Hz cut-off along slow-time on every
   range bin (real and imaginary parts filtered independently via
   zero-phase `filtfilt`).
5. **Target range-bin selection** — pick the range bins that carry ~70 %
   of the total energy after clutter removal.
6. **Micro-Doppler spectrogram** — compute a Short-Time Fourier Transform
   per selected range bin (0.2 s Hamming window, 95 % overlap, two-sided,
   `fftshift`-ed to centre 0 Hz). Powers are summed across bins and the
   result is converted to dB.
7. **Resize & normalise** — every spectrogram is resized (bilinear) to a
   fixed shape (default 128 × 256) and rescaled to `[0, 1]`.

## CNN architecture

Four VGG-style convolution blocks (32 → 64 → 128 → 128 filters, each
with two 3×3 Conv + BatchNorm + ReLU and a 2×2 max-pool), then global
average pooling, a 128-unit dense layer with dropout, and a 6-way
softmax. Optimiser: Adam with `ReduceLROnPlateau` and early stopping on
validation accuracy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow ≥ 2.13 is required. Use `tensorflow-macos` on Apple-silicon
if needed.

## 1. Preprocess (run once)

```bash
python preprocess.py --workers 4
```

Output: `processed/dataset.npz` containing `X`, `y`, `persons`,
`filenames`, `class_names`. This step takes a few minutes depending on
the number of files and cores.

Useful flags:

* `--shape 128x256` — change the cached spectrogram size.
* `--no-normalize` — keep raw dB values.
* `--limit 50` — quick sanity-check run.

## 2. Train

```bash
python train.py --epochs 40 --batch-size 32
```

The train/test split is done **by person** so the test set contains
subjects that the network has never seen. Outputs are stored under
`models/`:

* `best_model.keras` — highest validation accuracy checkpoint
* `radar_cnn.keras` — final model after training
* `training_curves.png`, `confusion_matrix.png`, `metrics.json`

## 3. Predict on one file

```bash
python predict.py radar_data/5P01A05R1.dat
python predict.py radar_data/1P01A01R01.dat --show-spectrogram
```

The script reports the predicted activity, per-class probabilities and
optionally saves the spectrogram as a PNG.

## Notes

* The spec lists ~1754 files; the provided dataset contains ~1636. The
  pipeline tolerates alternate filename styles (`R1` vs. `R01`) and
  common suffixes (`" - Copy"`, `" (2)"`).
* All constants (carrier frequency, PRF, chirp duration, bandwidth,
  Butterworth parameters, STFT parameters) live at the top of
  `radar_utils.py` and are easy to tweak.
