"""
radar_utils.py
--------------
Utilities for loading FMCW radar `.dat` files and transforming them into
micro-Doppler spectrograms suitable for CNN classification.

Signal-processing pipeline (as specified in the project statement):

    raw file  ->  parse metadata + complex samples
              ->  reshape to (n_chirps, samples_per_chirp)
              ->  Hamming-windowed FFT along fast-time (range-time map)
              ->  4th-order Butterworth high-pass filter along slow-time
                  (removes static clutter; cut-off 0.0075 Hz)
              ->  Short-Time Fourier Transform on range bins containing the
                  target (0.2 s Hamming window, 95 % overlap)
              ->  sum |STFT|^2 across those range bins  =>  micro-Doppler
                  spectrogram (log-scaled magnitude)

Activity 1 ("Walking back and forth") files are 10 s long and are split
into two consecutive 5-second segments, producing two training samples
each.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, get_window, stft


ACTIVITY_NAMES = {
    1: "Walking back and forth",
    2: "Sitting down on a chair",
    3: "Standing up from a chair",
    4: "Picking up an object",
    5: "Drinking water",
    6: "Falling",
}
NUM_CLASSES = len(ACTIVITY_NAMES)

SAMPLES_PER_CHIRP = 128
CHIRP_DURATION_S = 1e-3
PRF_HZ = 1.0 / CHIRP_DURATION_S
CARRIER_HZ = 5.8e9
BANDWIDTH_HZ = 400e6
SEGMENT_SECONDS = 5.0
CHIRPS_PER_SEGMENT = int(round(SEGMENT_SECONDS * PRF_HZ))

STFT_WINDOW_SECONDS = 0.2
STFT_OVERLAP_FRACTION = 0.95
BUTTER_ORDER = 4
BUTTER_CUTOFF_HZ = 0.0075


_FILENAME_RE = re.compile(
    r"^(?P<act>\d)P(?P<person>\d{2})A(?P<actid>\d{2})R(?P<rep>\d{1,2})(?:[ _\-].*)?\.dat$",
    re.IGNORECASE,
)

_COMPLEX_RE = re.compile(r"(-?\d+(?:\.\d+)?)([-+]\d+(?:\.\d+)?)i")


def parse_filename(path: str | Path) -> Tuple[int, int, int, int]:
    """Return (activity_digit, person_id, activity_id, repetition) from the filename.

    Accepts both ``1P01A01R1.dat`` and ``1P01A01R01.dat`` styles, and tolerates
    simple suffixes such as ``" (2)"`` or ``" - Copy"``.
    """
    name = Path(path).name
    m = _FILENAME_RE.match(name)
    if m is None:
        raise ValueError(f"Unrecognized radar filename: {name!r}")
    act = int(m.group("act"))
    person = int(m.group("person"))
    actid = int(m.group("actid"))
    rep = int(m.group("rep"))
    if act != actid:
        # Spec guarantees the first digit equals the two-digit activity id.
        # Warn but do not fail; we trust the first digit.
        pass
    return act, person, actid, rep


def load_radar_file(path: str | Path) -> Tuple[dict, np.ndarray]:
    """Read a radar `.dat` file and return (metadata_dict, complex_samples).

    The metadata dict contains the four header records:
        carrier_hz, chirp_duration_ms, samples_per_chirp, bandwidth_hz.

    Complex samples are returned as a 1-D ``np.complex64`` array, ordered
    chronologically as they appear in the file.
    """
    text = Path(path).read_text()
    # Split into the four header lines and the body in one pass.
    header_plus_body = text.split("\n", 4)
    if len(header_plus_body) < 5:
        raise ValueError(f"File {path} does not contain a body.")

    carrier = float(header_plus_body[0].strip())
    chirp_dur_ms = float(header_plus_body[1].strip())
    n_per_chirp = int(float(header_plus_body[2].strip()))
    bandwidth = float(header_plus_body[3].strip())
    body = header_plus_body[4]

    pairs = _COMPLEX_RE.findall(body)
    if not pairs:
        raise ValueError(f"No complex samples parsed from {path}.")

    real = np.fromiter((float(r) for r, _ in pairs), dtype=np.float32,
                       count=len(pairs))
    imag = np.fromiter((float(i) for _, i in pairs), dtype=np.float32,
                       count=len(pairs))
    samples = real + 1j * imag
    samples = samples.astype(np.complex64)

    meta = {
        "carrier_hz": carrier,
        "chirp_duration_ms": chirp_dur_ms,
        "samples_per_chirp": n_per_chirp,
        "bandwidth_hz": bandwidth,
    }
    return meta, samples


def reshape_to_pulses(samples: np.ndarray,
                      samples_per_chirp: int = SAMPLES_PER_CHIRP) -> np.ndarray:
    """Reshape a flat sample array to ``(n_chirps, samples_per_chirp)``."""
    n = (samples.size // samples_per_chirp) * samples_per_chirp
    return samples[:n].reshape(-1, samples_per_chirp)


def compute_range_time_map(pulses: np.ndarray) -> np.ndarray:
    """Hamming-windowed FFT along fast-time for each pulse (chirp).

    Parameters
    ----------
    pulses : complex ``(n_chirps, n_samples)`` array.

    Returns
    -------
    complex array with shape ``(n_chirps, n_range_bins)`` where
    ``n_range_bins = n_samples // 2`` (positive half of the spectrum).
    """
    n_samples = pulses.shape[1]
    hamming = np.hamming(n_samples).astype(np.float32)
    windowed = pulses * hamming[np.newaxis, :]
    rtm = np.fft.fft(windowed, axis=1)
    return rtm[:, : n_samples // 2]


def remove_static_clutter(rtm: np.ndarray,
                          prf_hz: float = PRF_HZ,
                          cutoff_hz: float = BUTTER_CUTOFF_HZ,
                          order: int = BUTTER_ORDER) -> np.ndarray:
    """High-pass Butterworth filter along slow-time for each range bin.

    The filter is applied independently to the real and imaginary components
    so that the phase information is preserved.
    """
    nyq = prf_hz / 2.0
    wn = cutoff_hz / nyq
    # scipy requires 0 < Wn < 1
    wn = min(max(wn, 1e-8), 0.99)
    b, a = butter(order, wn, btype="highpass")

    # filtfilt doubles the effective order but cancels phase distortion.
    # Use a padlen that is safe even for tiny cut-off frequencies.
    padlen = 3 * (max(len(a), len(b)) - 1)
    real = filtfilt(b, a, rtm.real, axis=0, padlen=padlen)
    imag = filtfilt(b, a, rtm.imag, axis=0, padlen=padlen)
    return (real + 1j * imag).astype(np.complex64)


def select_target_range_bins(rtm: np.ndarray,
                             energy_fraction: float = 0.70,
                             min_bins: int = 3,
                             max_bins: int = 40,
                             skip_dc: int = 1) -> np.ndarray:
    """Pick the range bins that most likely contain the target.

    A simple yet effective heuristic: rank bins by total slow-time energy
    (after clutter removal) and keep the top ones that together hold
    ``energy_fraction`` of the overall energy.

    ``skip_dc`` discards the first few range bins (which sit near DC and
    often carry leakage regardless of the clutter filter).
    """
    energy = np.sum(np.abs(rtm) ** 2, axis=0)
    energy = energy.copy()
    energy[:skip_dc] = 0.0

    order = np.argsort(energy)[::-1]
    cumulative = np.cumsum(energy[order])
    total = cumulative[-1] + 1e-12
    k = int(np.searchsorted(cumulative / total, energy_fraction)) + 1
    k = max(min_bins, min(k, max_bins, order.size))
    return np.sort(order[:k])


def compute_micro_doppler(rtm: np.ndarray,
                          prf_hz: float = PRF_HZ,
                          window_seconds: float = STFT_WINDOW_SECONDS,
                          overlap_fraction: float = STFT_OVERLAP_FRACTION,
                          selected_bins: np.ndarray | None = None,
                          log_scale: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the micro-Doppler spectrogram from a range-time map.

    Returns ``(frequencies_hz, times_s, spectrogram)`` where the
    spectrogram has shape ``(n_freq_bins, n_time_frames)``. Frequencies
    are ``fftshift``-ed to be centred on 0 Hz, i.e. they run from
    ``-prf/2`` to ``+prf/2``.
    """
    nperseg = int(round(window_seconds * prf_hz))
    noverlap = int(round(overlap_fraction * nperseg))
    if selected_bins is None:
        selected_bins = select_target_range_bins(rtm)

    window = get_window("hamming", nperseg)

    accumulated = None
    f_out, t_out = None, None
    for b in selected_bins:
        f, t, Z = stft(
            rtm[:, b],
            fs=prf_hz,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False,
            boundary=None,
            padded=False,
        )
        Z = np.fft.fftshift(Z, axes=0)
        f = np.fft.fftshift(f)
        power = np.abs(Z) ** 2
        if accumulated is None:
            accumulated = power
            f_out, t_out = f, t
        else:
            accumulated = accumulated + power

    spec = accumulated
    if log_scale:
        spec = 10.0 * np.log10(spec + 1e-12)
    return f_out, t_out, spec.astype(np.float32)


def normalize_spectrogram(spec: np.ndarray) -> np.ndarray:
    """Per-sample 0-1 normalisation. Safe for constant spectrograms."""
    lo = float(np.min(spec))
    hi = float(np.max(spec))
    if hi - lo < 1e-9:
        return np.zeros_like(spec, dtype=np.float32)
    return ((spec - lo) / (hi - lo)).astype(np.float32)


def segment_pulses(pulses: np.ndarray,
                   activity_digit: int,
                   chirps_per_segment: int = CHIRPS_PER_SEGMENT
                   ) -> List[np.ndarray]:
    """Split a pulse matrix into 5-second segments.

    Activity 1 recordings (10 s) produce two segments; every other activity
    yields a single 5-second segment, trimmed/padded if necessary.
    """
    n_chirps = pulses.shape[0]
    if activity_digit == 1:
        if n_chirps >= 2 * chirps_per_segment:
            return [pulses[:chirps_per_segment],
                    pulses[chirps_per_segment:2 * chirps_per_segment]]
        if n_chirps >= chirps_per_segment:
            return [pulses[:chirps_per_segment]]
        return []
    if n_chirps >= chirps_per_segment:
        return [pulses[:chirps_per_segment]]
    return []


def file_to_spectrograms(path: str | Path) -> List[Tuple[np.ndarray, int]]:
    """Convert one radar `.dat` file into a list of ``(spectrogram, label)``.

    Labels are 0-indexed activity ids (0..5).
    """
    activity, _, _, _ = parse_filename(path)
    meta, samples = load_radar_file(path)
    pulses = reshape_to_pulses(samples, int(meta["samples_per_chirp"]))

    outputs: List[Tuple[np.ndarray, int]] = []
    for segment in segment_pulses(pulses, activity):
        rtm = compute_range_time_map(segment)
        rtm = remove_static_clutter(rtm)
        bins = select_target_range_bins(rtm)
        _, _, spec = compute_micro_doppler(rtm, selected_bins=bins)
        outputs.append((spec, activity - 1))
    return outputs


def iter_radar_files(root: str | Path) -> Iterable[Path]:
    """Yield every parseable `.dat` file under ``root`` (non-recursively)."""
    root = Path(root)
    for p in sorted(root.glob("*.dat")):
        try:
            parse_filename(p)
        except ValueError:
            continue
        yield p
