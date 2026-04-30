"""
preprocess.py
=============
End-to-end signal processing for the INSHEP FMCW radar:

    raw I/Q  →  range-time map  →  clutter-suppressed  →
    target range bins  →  STFT  →  micro-Doppler spectrogram

Output: a (H, W) float32 standardized log-magnitude spectrogram. Same
pipeline is reused by the autoencoder anomaly detector and by the
supervised baselines, so all comparisons are on identical inputs.

Defaults follow Li et al. (2023, Sci. Reports 13:3473).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.signal import butter, filtfilt, stft, get_window


# Radar parameters from the INSHEP Readme. The .dat files are pure
# complex64 I/Q samples with NO file header — these constants come from
# the data sheet, not from any embedded metadata.
INSHEP_RADAR = {
    "fc_hz":             5.8e9,    # carrier frequency
    "bw_hz":             400.0e6,  # sweep bandwidth
    "chirp_s":           1.0e-3,   # chirp duration → 1 kHz PRF
    "samples_per_chirp": 128,
}


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def read_dat(path: str | Path,
             samples_per_chirp: int = 128,
             fc_hz: float = INSHEP_RADAR["fc_hz"],
             bw_hz: float = INSHEP_RADAR["bw_hz"],
             chirp_s: float = INSHEP_RADAR["chirp_s"],
             header_floats: int = 0) -> tuple[np.ndarray, dict]:
    """Read an Ancortek INSHEP .dat file as raw complex64 I/Q samples.

    The INSHEP files have NO embedded header — they are pure complex64
    samples. Radar parameters come from the Readme's data sheet (see
    INSHEP_RADAR). Override via kwargs only if you have a session that
    used different settings.

    The `samples_per_chirp` argument is authoritative for both the
    reshape AND the meta dict (so range_axis() always gets a non-empty
    array, regardless of file content).

    `header_floats` is provided as an escape hatch if a file format ever
    DOES have a leading header — pass the number of leading complex64
    samples to skip. Default 0 (no header).
    """
    raw = np.fromfile(path, dtype=np.complex64)
    if raw.size <= header_floats + samples_per_chirp:
        raise ValueError(f"{path}: file too short "
                         f"({raw.size} complex samples; need at least "
                         f"{header_floats + samples_per_chirp})")

    body = raw[header_floats:] if header_floats > 0 else raw
    n_chirps = body.size // samples_per_chirp
    if n_chirps == 0:
        raise ValueError(f"{path}: no full chirps after header strip "
                         f"(samples_per_chirp={samples_per_chirp})")
    body = body[: n_chirps * samples_per_chirp]
    beats = body.reshape(n_chirps, samples_per_chirp)

    return beats, {
        "fc_hz":             fc_hz,
        "chirp_s":           chirp_s,
        "samples_per_chirp": samples_per_chirp,
        "bw_hz":             bw_hz,
        "n_chirps":          n_chirps,
    }


# ----------------------------------------------------------------------
# Range FFT
# ----------------------------------------------------------------------
def range_fft(beats: np.ndarray, window: str = "hamming") -> np.ndarray:
    n_chirps, n_samples = beats.shape
    win = get_window(window, n_samples)
    return np.fft.fft(beats * win[None, :], axis=1)


def range_axis(meta: dict) -> np.ndarray:
    c = 299_792_458.0
    n = meta["samples_per_chirp"]
    bw = meta["bw_hz"]
    return np.arange(n) * c / (2 * bw)


# ----------------------------------------------------------------------
# Clutter suppression (high-pass over slow time)
# ----------------------------------------------------------------------
def remove_static_clutter(rt: np.ndarray, prf_hz: float,
                          cutoff_hz: float = 1.0,
                          order: int = 4) -> np.ndarray:
    """High-pass over slow time to suppress static clutter (walls,
    furniture, the radar mount).

    cutoff_hz=1.0 corresponds to a velocity threshold of:
        v = c * f_d / (2 * fc) = 3e8 * 1.0 / (2 * 5.8e9) = 0.026 m/s
    so motion slower than ~2.6 cm/s (very slow drift, breathing while
    motionless) is rejected. Human activities easily exceed this.

    The previous default of 0.0075 Hz corresponded to v = 0.0002 m/s —
    only literal DC was rejected. Static furniture leaked through and
    dominated every spectrogram, pinning max-Doppler velocity at 0.
    """
    nyq = 0.5 * prf_hz
    wn = max(min(cutoff_hz / nyq, 0.999), 1e-6)
    b, a = butter(order, wn, btype="highpass")
    re = filtfilt(b, a, rt.real, axis=0)
    im = filtfilt(b, a, rt.imag, axis=0)
    return re + 1j * im


# ----------------------------------------------------------------------
# Range-bin selection
# ----------------------------------------------------------------------
def select_target_bins(rt: np.ndarray, meta: dict,
                       n_neighbors: int = 5,
                       min_range_m: float = 1.0,
                       max_range_m: float = 4.5) -> tuple[np.ndarray, int]:
    """Find the dominant range bin in the [min, max] window and sum the
    complex returns over a ±n_neighbors band. Robust to per-session
    geometry variability."""
    rng = range_axis(meta)
    energy = np.sum(np.abs(rt) ** 2, axis=0)

    # Defensive: range axis MUST match the second axis of the rt array.
    # If someone passes a meta dict with wrong samples_per_chirp this
    # would silently broadcast or crash; instead, fail with a clear msg.
    if rng.shape != energy.shape:
        raise ValueError(
            f"range_axis shape {rng.shape} != energy shape {energy.shape}. "
            f"meta['samples_per_chirp']={meta.get('samples_per_chirp')} "
            f"but rt has {rt.shape[1]} bins. This usually means read_dat "
            f"returned a meta dict whose samples_per_chirp does not match "
            f"the actual reshape. Pass samples_per_chirp explicitly to "
            f"read_dat or use the INSHEP_RADAR defaults.")

    valid = (rng >= min_range_m) & (rng <= max_range_m)
    if not np.any(valid):
        valid[:] = True

    energy_masked = np.where(valid, energy, -np.inf)
    peak = int(np.argmax(energy_masked))

    lo = max(0, peak - n_neighbors)
    hi = min(rt.shape[1], peak + n_neighbors + 1)
    return rt[:, lo:hi].sum(axis=1), peak


# ----------------------------------------------------------------------
# Micro-Doppler STFT
# ----------------------------------------------------------------------
def micro_doppler(target: np.ndarray, prf_hz: float,
                  window_s: float = 0.20, overlap: float = 0.95,
                  nfft: int = 256):
    # Clamp window to the actual input length (defensive for short files)
    n_input = len(target)
    nperseg = max(8, int(round(window_s * prf_hz)))
    nperseg = min(nperseg, max(8, n_input))
    noverlap = min(int(round(overlap * nperseg)), nperseg - 1)
    nfft = max(nfft, nperseg)
    f, t, Sxx = stft(target, fs=prf_hz, window="hamming",
                     nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                     return_onesided=False, boundary=None, padded=False)
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    return f, t, Sxx


# ----------------------------------------------------------------------
# Resize + log + standardize
# ----------------------------------------------------------------------
def to_canonical(spec_complex: np.ndarray,
                 out_h: int = 128, out_w: int = 256,
                 log: bool = True, normalize: bool = True) -> np.ndarray:
    mag = np.abs(spec_complex).astype(np.float32)
    if log:
        mag = 20.0 * np.log10(mag + 1e-12)

    h_in, w_in = mag.shape
    # Resize along height (Doppler axis): for each column, interp from h_in → out_h
    if h_in != out_h:
        idx_h = np.linspace(0, h_in - 1, out_h)
        new = np.empty((out_h, w_in), dtype=np.float32)
        for j in range(w_in):
            new[:, j] = np.interp(idx_h, np.arange(h_in), mag[:, j])
        mag = new
    # Resize along width (time axis): for each row, interp from w_in → out_w
    if w_in != out_w:
        idx_w = np.linspace(0, w_in - 1, out_w)
        new = np.empty((mag.shape[0], out_w), dtype=np.float32)
        for i in range(mag.shape[0]):
            new[i, :] = np.interp(idx_w, np.arange(w_in), mag[i, :])
        mag = new

    if normalize:
        mu, sd = mag.mean(), mag.std() + 1e-8
        mag = (mag - mu) / sd
    return mag.astype(np.float32)


# ----------------------------------------------------------------------
# Max-velocity / peak-Doppler feature (the SVM-paper baseline feature)
# ----------------------------------------------------------------------
def max_doppler_velocity(spec_complex: np.ndarray,
                         prf_hz: float = 1000.0,
                         fc_hz: float = 5.8e9,
                         nfft: int = 256,
                         percentile: float = 90.0,
                         dc_mask_mps: float = 0.3) -> float:
    """Peak radial velocity reached during the recording (m/s).

    Implements the SVM-paper feature. For each time slice, find the
    Doppler bin with maximum magnitude (excluding bins inside the
    ±dc_mask_mps clutter band), convert to velocity, then take the
    percentile-th highest value across time.

    Falls produce brief high-velocity transients, walking produces
    sustained moderate velocity, sit/stand/pick/drink barely move.

    Computed BEFORE log-scaling and BEFORE normalization, on the
    complex STFT output, so the magnitude information is preserved.

    The dc_mask_mps parameter rejects residual clutter that the
    upstream high-pass filter could not fully suppress: any Doppler
    bin corresponding to |v| < dc_mask_mps is excluded from the
    argmax. With dc_mask_mps=0.3 m/s, slow rocking, breathing, and
    static reflections cannot dominate the feature.
    """
    c = 299_792_458.0
    f_doppler = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / prf_hz))
    velocity = np.abs(c * f_doppler / (2.0 * fc_hz))   # shape (nfft,)

    mag = np.abs(spec_complex).astype(np.float32)
    n_freq, n_time = mag.shape

    if n_freq != len(velocity):
        f = np.fft.fftshift(np.fft.fftfreq(n_freq, d=1.0 / prf_hz))
        velocity = np.abs(c * f / (2.0 * fc_hz))

    # Mask out the clutter band so its residual energy doesn't dominate.
    # We multiply mag by 0 wherever |v| < dc_mask_mps; argmax then can
    # only pick bins outside that band.
    dc_mask = (velocity >= dc_mask_mps).astype(mag.dtype)
    mag_masked = mag * dc_mask[:, None]

    # If everything is masked away, fall back to 0 (no signal above
    # the clutter band — likely a stationary recording).
    if mag_masked.max() <= 0:
        return 0.0

    peak_v_per_time = velocity[np.argmax(mag_masked, axis=0)]
    return float(np.percentile(peak_v_per_time, percentile))


# ----------------------------------------------------------------------
# Walking-recording handling: split 10 s walks into two 5 s halves so
# that all activities have uniform 5 s duration (Li et al. 2023).
# ----------------------------------------------------------------------
def maybe_split_walking(spec: np.ndarray, activity_code: int,
                        out_w: int = 256) -> list[np.ndarray]:
    """Walking is 10 s; everything else is 5 s. After STFT and resize,
    a walk has 2× the time bins of any other activity, but we resize
    everything to out_w. Recover the two halves by NOT resizing first
    and splitting the time axis. This function is a hook — by default
    we resize uniformly and accept the slight smoothing for walks."""
    return [spec]


# ----------------------------------------------------------------------
# Convenience
# ----------------------------------------------------------------------
@dataclass
class PipelineConfig:
    samples_per_chirp: int = 128
    clutter_cutoff_hz: float = 1.0      # was 0.0075 — 130× too low
    clutter_order: int = 4
    bin_neighbors: int = 5
    bin_min_m: float = 1.0
    bin_max_m: float = 4.5
    stft_window_s: float = 0.20
    stft_overlap: float = 0.95
    stft_nfft: int = 256
    out_h: int = 128
    out_w: int = 256
    log_scale: bool = True
    # Normalization mode — see notes in to_canonical for the rationale.
    #   "none"        — no normalization (raw log-magnitude in dB)
    #   "global"      — apply a fixed (mu, sigma) computed once over the
    #                   training set; preserves magnitude differences
    #                   across samples (recommended for anomaly detection)
    #   "per_sample"  — old default; zeroes magnitude info, harmful for
    #                   anomaly detection because falls have higher total
    #                   Doppler energy than other activities
    normalize_mode: str = "global"


def file_to_spectrogram(path: str | Path,
                        cfg: PipelineConfig | None = None,
                        return_features: bool = False
                        ) -> np.ndarray | tuple[np.ndarray, dict]:
    """Compute the canonical (out_h, out_w) spectrogram. If
    return_features=True, also returns a dict of scalar features
    (e.g. max-Doppler velocity) computed BEFORE normalization, so that
    magnitude-dependent features are preserved regardless of the
    normalize_mode chosen for the spectrogram itself.
    """
    cfg = cfg or PipelineConfig()
    beats, meta = read_dat(path, samples_per_chirp=cfg.samples_per_chirp)
    prf = 1.0 / meta["chirp_s"]
    rt = range_fft(beats)
    rt = remove_static_clutter(rt, prf_hz=prf,
                               cutoff_hz=cfg.clutter_cutoff_hz,
                               order=cfg.clutter_order)
    target, _ = select_target_bins(rt, meta,
                                   n_neighbors=cfg.bin_neighbors,
                                   min_range_m=cfg.bin_min_m,
                                   max_range_m=cfg.bin_max_m)
    _, _, Sxx = micro_doppler(target, prf_hz=prf,
                              window_s=cfg.stft_window_s,
                              overlap=cfg.stft_overlap,
                              nfft=cfg.stft_nfft)

    # Magnitude-dependent features — must be computed BEFORE normalization
    features = {}
    if return_features:
        features["max_doppler_velocity_mps"] = max_doppler_velocity(
            Sxx, prf_hz=prf, fc_hz=meta["fc_hz"], nfft=cfg.stft_nfft)
        features["spectral_energy_db"] = float(
            20.0 * np.log10(np.abs(Sxx).mean() + 1e-12))

    # Per-sample normalization is left available for backward compatibility
    # but not the default — see normalize_mode in PipelineConfig.
    spec = to_canonical(Sxx, out_h=cfg.out_h, out_w=cfg.out_w,
                        log=cfg.log_scale,
                        normalize=(cfg.normalize_mode == "per_sample"))

    return (spec, features) if return_features else spec
