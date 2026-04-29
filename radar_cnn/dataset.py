"""PyTorch Dataset: load .dat -> log spectrogram; optional disk cache per split."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from radar_cnn.io_dat import read_glasgow_dat
from radar_cnn.labels import activity_to_binary_label, activity_to_label, parse_filename
from radar_cnn.spectrogram import SpectrogramConfig, compute_log_spectrogram


def _cache_key(path: Path, spec_cfg: SpectrogramConfig) -> str:
    d = {
        "path": str(path.resolve()),
        "range_bin_mode": spec_cfg.range_bin_mode,
        "fixed_range_bin": spec_cfg.fixed_range_bin,
        "range_band": spec_cfg.range_band,
        "fixed_num_chirps": spec_cfg.fixed_num_chirps,
        "stft_nperseg": spec_cfg.stft_nperseg,
        "stft_noverlap": spec_cfg.stft_noverlap,
        "spec_hw": (spec_cfg.spec_height, spec_cfg.spec_width),
    }
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha256(s.encode()).hexdigest()[:24]


def _doppler_centroid_and_accel(
    spec_logmag: np.ndarray,
    seq_len: int,
    frame_reduce: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Log-magnitude spectrogram -> per-bin Doppler centroid (linear mag weighted) and Δcentroid.
    Centroid index proxies mean radial velocity bin; its difference highlights impacts/falls.
    """
    if spec_logmag.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spec_logmag.shape}")
    h, w = spec_logmag.shape
    lin = np.exp(spec_logmag.astype(np.float64))
    edges = np.linspace(0, w, num=seq_len + 1, dtype=int)
    v = np.zeros(seq_len, dtype=np.float32)
    rows = np.arange(h, dtype=np.float64)
    for t in range(seq_len):
        s = edges[t]
        e = edges[t + 1]
        if e <= s:
            e = min(w, s + 1)
        chunk = lin[:, s:e]
        if frame_reduce == "max":
            col = np.max(chunk, axis=1)
        else:
            col = np.mean(chunk, axis=1)
        wsum = float(col.sum()) + 1e-8
        v[t] = np.float32(np.dot(rows, col) / wsum)
    a = np.zeros(seq_len, dtype=np.float32)
    a[1:] = v[1:] - v[:-1]
    return v, a


def spectrogram_to_sequence_features(
    spec: np.ndarray,
    seq_len: int,
    frame_reduce: str = "mean",
    kinematic: bool = False,
) -> np.ndarray:
    """
    Convert (H, W) spectrogram to sequence (T, F) by slicing W into T chunks.
    F is H (or H+2 if kinematic: append Doppler centroid + Δcentroid per timestep).
    """
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if spec.ndim != 2:
        raise ValueError(f"Expected 2D spectrogram, got shape {spec.shape}")

    h, w = spec.shape
    edges = np.linspace(0, w, num=seq_len + 1, dtype=int)
    out = np.zeros((seq_len, h), dtype=np.float32)
    for t in range(seq_len):
        s = edges[t]
        e = edges[t + 1]
        if e <= s:
            e = min(w, s + 1)
        chunk = spec[:, s:e]
        if frame_reduce == "max":
            out[t] = np.max(chunk, axis=1)
        else:
            out[t] = np.mean(chunk, axis=1)
    if not kinematic:
        return out
    v, a = _doppler_centroid_and_accel(spec, seq_len, frame_reduce)
    return np.concatenate([out, v[:, np.newaxis], a[:, np.newaxis]], axis=1)


class RadarSpectrogramDataset(Dataset):
    """
    Returns (tensor 1xHxW, label int64, subject_id int for eval grouping).
    Normalization (mean/std) applied externally if train_stats provided.
    """

    def __init__(
        self,
        paths: list[Path | str],
        spec_cfg: SpectrogramConfig,
        train_mean: float | None = None,
        train_std: float | None = None,
        cache_dir: Path | str | None = None,
        split_name: str = "train",
        label_override: np.ndarray | None = None,
        binary_labels: bool = False,
    ) -> None:
        self.paths = [Path(p) for p in paths]
        self.spec_cfg = spec_cfg
        self.train_mean = train_mean
        self.train_std = train_std
        self.binary_labels = bool(binary_labels)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name
        if label_override is not None:
            lo = np.asarray(label_override, dtype=np.int64).reshape(-1)
            if len(lo) != len(self.paths):
                raise ValueError("label_override length must match paths")
            self.label_override = lo
        else:
            self.label_override = None
        if self.cache_dir:
            self.cache_dir = self.cache_dir / split_name
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_spec(self, path: Path) -> np.ndarray:
        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            if cpath.exists():
                return np.load(cpath)

        data, hdr = read_glasgow_dat(str(path))
        tc = hdr["tsweep_s"]
        spec = compute_log_spectrogram(data, tc, self.spec_cfg)

        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            np.save(cpath, spec)

        return spec

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        spec = self._load_spec(path)
        x = torch.from_numpy(spec).float().unsqueeze(0)
        if self.train_mean is not None and self.train_std is not None:
            x = (x - self.train_mean) / (self.train_std + 1e-8)

        if self.label_override is not None:
            y = int(self.label_override[idx])
        else:
            _, activity, _ = parse_filename(path)
            if self.binary_labels:
                y = activity_to_binary_label(activity)
            else:
                y = activity_to_label(activity)
        sid, _, _ = parse_filename(path)

        return x, torch.tensor(y, dtype=torch.long), sid


class RadarSequenceDataset(Dataset):
    """
    Returns (sequence tensor TxF, label int64, subject_id int).
    Sequence is derived from the same cached spectrogram path used by CNN runs.
    If kinematic=True, F = spec_height + 2 (Doppler centroid + delta); use vector mean/std.
    """

    def __init__(
        self,
        paths: list[Path | str],
        spec_cfg: SpectrogramConfig,
        seq_len: int,
        frame_reduce: str = "mean",
        binary_labels: bool = True,
        kinematic: bool = False,
        train_mean: float | np.ndarray | None = None,
        train_std: float | np.ndarray | None = None,
        cache_dir: Path | str | None = None,
        split_name: str = "train",
        label_override: np.ndarray | None = None,
    ) -> None:
        if frame_reduce not in {"mean", "max"}:
            raise ValueError("frame_reduce must be 'mean' or 'max'")
        self.paths = [Path(p) for p in paths]
        self.spec_cfg = spec_cfg
        self.seq_len = int(seq_len)
        self.frame_reduce = frame_reduce
        self.binary_labels = bool(binary_labels)
        self.kinematic = bool(kinematic)
        self.train_mean = train_mean
        self.train_std = train_std
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.split_name = split_name
        if label_override is not None:
            lo = np.asarray(label_override, dtype=np.int64).reshape(-1)
            if len(lo) != len(self.paths):
                raise ValueError("label_override length must match paths")
            self.label_override = lo
        else:
            self.label_override = None
        if self.cache_dir:
            self.cache_dir = self.cache_dir / split_name
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.paths)

    def _load_spec(self, path: Path) -> np.ndarray:
        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            if cpath.exists():
                return np.load(cpath)

        data, hdr = read_glasgow_dat(str(path))
        tc = hdr["tsweep_s"]
        spec = compute_log_spectrogram(data, tc, self.spec_cfg)

        if self.cache_dir:
            key = _cache_key(path, self.spec_cfg)
            cpath = self.cache_dir / f"{key}.npy"
            np.save(cpath, spec)
        return spec

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        spec = self._load_spec(path)
        if self.kinematic:
            x = spectrogram_to_sequence_features(
                spec, self.seq_len, self.frame_reduce, kinematic=True
            )
            if self.train_mean is not None and self.train_std is not None:
                tm = np.asarray(self.train_mean, dtype=np.float32)
                ts = np.asarray(self.train_std, dtype=np.float32)
                x = (x - tm) / (ts + 1e-8)
        else:
            if self.train_mean is not None and self.train_std is not None:
                spec = (spec - float(self.train_mean)) / (float(self.train_std) + 1e-8)
            x = spectrogram_to_sequence_features(
                spec, self.seq_len, self.frame_reduce, kinematic=False
            )
        x_t = torch.from_numpy(x).float()

        if self.label_override is not None:
            y = int(self.label_override[idx])
        else:
            _, activity, _ = parse_filename(path)
            if self.binary_labels:
                y = activity_to_binary_label(activity)
            else:
                y = activity_to_label(activity)
        sid, _, _ = parse_filename(path)
        return x_t, torch.tensor(y, dtype=torch.long), sid


def compute_train_statistics(
    paths: list[Path],
    spec_cfg: SpectrogramConfig,
    cache_dir: Path | None,
    max_files: int | None = None,
    verbose: bool = False,
) -> tuple[float, float]:
    """
    Mean and std of log-spectrogram pixels over training files (single pass).

    This pass reads each .dat (slow: large ASCII), builds spectrogram, aggregates
    global mean/var. Use cache_dir so subsequent epochs reuse .npy caches.
    """
    acc_sum = 0.0
    acc_sq = 0.0
    n_pix = 0
    use_paths = paths[: max_files] if max_files else paths
    n_total = len(use_paths)
    if verbose:
        print(
            f"[train_stats] Computing mean/std over {n_total} files "
            f"(first full pass; use --cache_dir to speed reruns).",
            file=sys.stderr,
            flush=True,
        )
    ds_temp = RadarSpectrogramDataset(
        use_paths,
        spec_cfg,
        train_mean=None,
        train_std=None,
        cache_dir=cache_dir,
        split_name="train_stats",
    )
    pbar = tqdm(
        range(len(ds_temp)),
        desc="train_stats (global mean/std)",
        unit="file",
        disable=not sys.stdout.isatty(),
    )
    for i in pbar:
        path = ds_temp.paths[i]
        if verbose:
            pbar.set_postfix(file=path.name[:48], refresh=False)
        spec, _, _ = ds_temp[i]
        x = spec.squeeze().numpy()
        acc_sum += float(x.sum())
        acc_sq += float(np.square(x).sum())
        n_pix += x.size
        if verbose and (i % max(1, n_total // 20) == 0 or i == n_total - 1):
            print(
                f"[train_stats] {i + 1}/{n_total}  {path}  "
                f"running_mean≈{acc_sum / max(n_pix, 1):.4f}",
                file=sys.stderr,
                flush=True,
            )
    mean = acc_sum / max(n_pix, 1)
    var = acc_sq / max(n_pix, 1) - mean * mean
    std = float(np.sqrt(max(var, 1e-12)))
    return mean, std


def compute_train_statistics_sequence(
    paths: list[Path],
    spec_cfg: SpectrogramConfig,
    seq_len: int,
    frame_reduce: str,
    cache_dir: Path | None,
    max_files: int | None = None,
    verbose: bool = False,
    kinematic: bool = False,
) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]:
    """
    Mean/std over sequence features (T,F) built from training spectrograms.
    If kinematic: per-feature mean/std (F = spec_height + 2). Else: scalar global mean/std.
    """
    use_paths = paths[: max_files] if max_files else paths
    n_total = len(use_paths)
    if verbose:
        print(
            f"[train_stats_seq] Computing mean/std over {n_total} files (kinematic={kinematic}).",
            file=sys.stderr,
            flush=True,
        )

    ds_temp = RadarSequenceDataset(
        use_paths,
        spec_cfg,
        seq_len=seq_len,
        frame_reduce=frame_reduce,
        binary_labels=True,
        kinematic=kinematic,
        train_mean=None,
        train_std=None,
        cache_dir=cache_dir,
        split_name="train_stats_seq",
    )

    if kinematic:
        pbar = tqdm(
            range(len(ds_temp)),
            desc="train_stats_seq (per-dim mean/std)",
            unit="file",
            disable=not sys.stdout.isatty(),
        )
        f_dim: int | None = None
        acc_sum: np.ndarray | None = None
        acc_sq: np.ndarray | None = None
        n_rows = 0
        for i in pbar:
            x, _, _ = ds_temp[i]
            a = x.numpy()
            if f_dim is None:
                f_dim = a.shape[1]
                acc_sum = np.zeros(f_dim, dtype=np.float64)
                acc_sq = np.zeros(f_dim, dtype=np.float64)
            acc_sum += a.sum(axis=0)
            acc_sq += np.square(a).sum(axis=0)
            n_rows += a.shape[0]
            if verbose and (i % max(1, n_total // 20) == 0 or i == n_total - 1):
                print(
                    f"[train_stats_seq] {i + 1}/{n_total} rows={n_rows}",
                    file=sys.stderr,
                    flush=True,
                )
        assert acc_sum is not None and acc_sq is not None and f_dim is not None
        mean = acc_sum / max(n_rows, 1)
        var = acc_sq / max(n_rows, 1) - mean * mean
        std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
        return mean.astype(np.float32), std

    pbar = tqdm(
        range(len(ds_temp)),
        desc="train_stats_seq (global mean/std)",
        unit="file",
        disable=not sys.stdout.isatty(),
    )
    acc_sum_s = 0.0
    acc_sq_s = 0.0
    n_pix = 0
    for i in pbar:
        x, _, _ = ds_temp[i]
        a = x.numpy()
        acc_sum_s += float(a.sum())
        acc_sq_s += float(np.square(a).sum())
        n_pix += a.size
        if verbose and (i % max(1, n_total // 20) == 0 or i == n_total - 1):
            print(
                f"[train_stats_seq] {i + 1}/{n_total} running_mean≈{acc_sum_s / max(n_pix, 1):.4f}",
                file=sys.stderr,
                flush=True,
            )
    mean_s = acc_sum_s / max(n_pix, 1)
    var_s = acc_sq_s / max(n_pix, 1) - mean_s * mean_s
    std_s = float(np.sqrt(max(var_s, 1e-12)))
    return mean_s, std_s
