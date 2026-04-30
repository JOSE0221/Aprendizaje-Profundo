"""
dataset.py
==========
PyTorch Dataset over a manifest of RadarSamples. Two cache modes:

  * memory: keep all spectrograms in RAM. Good for ~1700 samples × 128 ×
            256 × 4 B ≈ 220 MB. Fits easily.
  * disk:   precompute spectrograms to .npy files in a cache directory
            and memory-map at load time. Use when running multiple
            training runs with the same preprocessing config.

Augmentation:
  * time-shift (axis 1)   — start the activity slightly earlier/later
  * Doppler-shift (axis 0) — small radial velocity offset
  * additive Gaussian noise — coarse thermal-noise model
Augmentations apply only to training data and only to the model input;
clean copies are kept for reconstruction targets and for the supervised
baselines.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from labels import RadarSample
from preprocess import file_to_spectrogram, PipelineConfig


# ----------------------------------------------------------------------
# Disk cache
# ----------------------------------------------------------------------
def _config_hash(cfg: PipelineConfig) -> str:
    payload = json.dumps(cfg.__dict__, sort_keys=True).encode()
    return hashlib.sha1(payload).hexdigest()[:10]


def cache_path(cache_root: Path | str, sample: RadarSample, cfg_hash: str) -> Path:
    """Spectrogram cache path: cache_root/<cfg_hash>/<session>/<file>.npy"""
    safe = sample.path.stem.replace(" ", "_")
    return Path(cache_root) / cfg_hash / sample.session.key / f"{safe}.npy"


def features_path(cache_root: Path | str, sample: RadarSample, cfg_hash: str) -> Path:
    """Per-sample scalar features: same naming as the spectrogram .npy
    but with .feat.json extension."""
    safe = sample.path.stem.replace(" ", "_")
    return Path(cache_root) / cfg_hash / sample.session.key / f"{safe}.feat.json"


def precompute(samples: list[RadarSample],
               cfg: PipelineConfig,
               cache_root: Path,
               verbose: bool = True) -> None:
    """Compute and save .npy spectrograms AND .feat.json scalar features
    (max-Doppler velocity, spectral energy) for every sample. The scalar
    features are computed BEFORE normalization so they're invariant to
    the normalize_mode used for the spectrogram. Idempotent."""
    h = _config_hash(cfg)
    skipped = computed = 0
    for s in samples:
        spec_out = cache_path(cache_root, s, h)
        feat_out = features_path(cache_root, s, h)
        if spec_out.exists() and feat_out.exists():
            skipped += 1
            continue
        spec, feats = file_to_spectrogram(s.path, cfg=cfg, return_features=True)
        spec_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(spec_out, spec)
        with open(feat_out, "w") as f:
            json.dump(feats, f)
        computed += 1
        if verbose and (computed % 50 == 0):
            print(f"  precomputed {computed}, skipped {skipped}")
    if verbose:
        print(f"  done. computed={computed}  skipped={skipped}")


# ----------------------------------------------------------------------
# Global standardization stats — computed once over the train fold,
# then applied identically to train/val/test. Preserves cross-sample
# magnitude differences (which per-sample standardization destroyed).
# ----------------------------------------------------------------------
def compute_global_stats(samples: list[RadarSample],
                         cfg: PipelineConfig,
                         cache_root: Path) -> tuple[float, float]:
    """Compute (mean, std) over a representative subset of train samples
    using Welford's online algorithm to avoid loading everything in RAM."""
    h = _config_hash(cfg)
    n = 0
    mean = 0.0
    M2 = 0.0
    for s in samples:
        cp = cache_path(cache_root, s, h)
        if not cp.exists():
            continue
        spec = np.load(cp).astype(np.float64)
        # Streaming update over flat pixel values
        for v in spec.ravel():
            n += 1
            delta = v - mean
            mean += delta / n
            M2 += delta * (v - mean)
    if n < 2:
        return 0.0, 1.0
    var = M2 / (n - 1)
    return float(mean), float(max(np.sqrt(var), 1e-8))


def load_features(cache_root: Path, sample: RadarSample,
                  cfg_hash: str) -> dict:
    """Load the cached scalar features for one sample."""
    fp = features_path(cache_root, sample, cfg_hash)
    if not fp.exists():
        return {}
    with open(fp) as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------
class RadarSpectrogramDataset(Dataset):
    def __init__(self,
                 samples: list[RadarSample],
                 cfg: PipelineConfig | None = None,
                 cache_root: Path | None = None,
                 augment: bool = False,
                 noise_std: float = 0.05,
                 max_time_shift: int = 32,
                 max_doppler_shift: int = 5,
                 global_mean: float | None = None,
                 global_std:  float | None = None):
        """If `global_mean` and `global_std` are provided, every spectrogram
        is normalized as (x - mean) / std using those FIXED statistics.
        This is the recommended path for anomaly detection because it
        preserves cross-sample magnitude differences (falls have higher
        Doppler energy than walks/sits/etc — a key discriminator that
        per-sample standardization destroys).

        If global_mean/std are None and cfg.normalize_mode == "per_sample",
        the spectrogram has already been per-sample-normalized at cache
        time and no further scaling is applied here.
        """
        self.samples = samples
        self.cfg = cfg or PipelineConfig()
        self.cache_root = Path(cache_root) if cache_root else None
        self.cfg_hash = _config_hash(self.cfg)
        self.augment = augment
        self.noise_std = noise_std
        self.max_time_shift = max_time_shift
        self.max_doppler_shift = max_doppler_shift
        self.global_mean = global_mean
        self.global_std  = global_std

        self._mem_cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _spectrogram(self, idx: int) -> np.ndarray:
        if idx in self._mem_cache:
            return self._mem_cache[idx]

        s = self.samples[idx]
        if self.cache_root is not None:
            cp = cache_path(self.cache_root, s, self.cfg_hash)
            if cp.exists():
                spec = np.load(cp)
            else:
                spec = file_to_spectrogram(s.path, cfg=self.cfg)
                cp.parent.mkdir(parents=True, exist_ok=True)
                np.save(cp, spec)
        else:
            spec = file_to_spectrogram(s.path, cfg=self.cfg)

        # Apply fixed global standardization if requested. Cached spectra
        # are stored UN-normalized when cfg.normalize_mode != "per_sample",
        # so this is the ONLY normalization step in that path.
        if self.global_mean is not None and self.global_std is not None:
            spec = ((spec - self.global_mean) / self.global_std).astype(np.float32)

        self._mem_cache[idx] = spec
        return spec

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.max_doppler_shift > 0:
            d = np.random.randint(-self.max_doppler_shift,
                                  self.max_doppler_shift + 1)
            if d != 0:
                x = np.roll(x, shift=d, axis=0)
        if self.max_time_shift > 0:
            t = np.random.randint(-self.max_time_shift,
                                  self.max_time_shift + 1)
            if t != 0:
                x = np.roll(x, shift=t, axis=1)
        if self.noise_std > 0:
            x = x + np.random.normal(0, self.noise_std,
                                     size=x.shape).astype(np.float32)
        return x

    def __getitem__(self, idx: int) -> dict:
        spec = self._spectrogram(idx)
        x = spec.copy()
        if self.augment:
            x = self._augment(x)

        s = self.samples[idx]
        # Load cached scalar features (max-velocity, spectral energy)
        feats = {}
        if self.cache_root is not None:
            feats = load_features(self.cache_root, s, self.cfg_hash)

        return {
            "x":          torch.from_numpy(x).unsqueeze(0).float(),
            "x_clean":    torch.from_numpy(spec).unsqueeze(0).float(),
            "activity":   torch.tensor(s.activity, dtype=torch.long),
            "is_fall":    torch.tensor(int(s.is_fall), dtype=torch.long),
            "subject":    s.subject_id,
            "session":    s.session.key,
            "path":       str(s.path),
            "max_velocity_mps": float(feats.get("max_doppler_velocity_mps", 0.0)),
            "spectral_energy_db": float(feats.get("spectral_energy_db", 0.0)),
        }


def make_loaders(splits, cfg: PipelineConfig,
                 cache_root: Path | None = None,
                 batch_size: int = 64, num_workers: int = 4,
                 global_stats: tuple[float, float] | None = None):
    """Returns (train_loader, val_loader, test_loader, stats_used).

    If `global_stats` is None and `cfg.normalize_mode == "global"`,
    statistics are computed once over the train fold's cached spectrograms
    and applied identically to all three folds. The function returns the
    (mean, std) pair so it can be saved alongside the model checkpoint
    (the same stats must be applied at evaluation time).

    If cfg.normalize_mode == "per_sample", spectrograms in the cache are
    already normalized and no further scaling is applied.
    If cfg.normalize_mode == "none",      no normalization at all.
    """
    if cfg.normalize_mode == "global":
        if global_stats is None and cache_root is not None:
            print("[loaders] computing global standardization stats over train fold...")
            global_stats = compute_global_stats(splits.train, cfg, cache_root)
            print(f"[loaders]   train mean={global_stats[0]:+.4f}  std={global_stats[1]:.4f}")
    elif cfg.normalize_mode == "per_sample" or cfg.normalize_mode == "none":
        global_stats = None
    else:
        raise ValueError(f"Unknown normalize_mode: {cfg.normalize_mode}")

    mu, sd = (global_stats if global_stats else (None, None))
    train_ds = RadarSpectrogramDataset(splits.train, cfg, cache_root,
                                       augment=True, global_mean=mu, global_std=sd)
    val_ds   = RadarSpectrogramDataset(splits.val,   cfg, cache_root,
                                       augment=False, global_mean=mu, global_std=sd)
    test_ds  = RadarSpectrogramDataset(splits.test,  cfg, cache_root,
                                       augment=False, global_mean=mu, global_std=sd)

    pin = bool(torch.cuda.is_available())

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin),
        global_stats,
    )
