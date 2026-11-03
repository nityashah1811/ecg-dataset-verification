"""Deterministic ECG preprocessing that reproduces the official ECGFounder pipeline.

Official reference (``dataset.py`` + ``util.filter_bandpass`` in PKUDigitalHealth/ECGFounder):

1. signal arranged as (leads, samples) in the order I, II, III, aVR, aVL, aVF, V1-V6, at 500 Hz;
2. NaN -> 0;
3. ``filter_bandpass``: 50 Hz IIR notch (Q=30) -> 4th-order Butterworth band-pass 0.67-40 Hz
   (zero-phase ``filtfilt``) -> subtraction of a 0.4 s median-filter baseline;
4. z-score normalisation using the mean and std of the *whole* (leads x samples) array
   (not per lead), with epsilon 1e-8;
5. input length 5000 samples (10 s at 500 Hz).

The README states that the weights only reproduce published results when these steps are
followed strictly. Note that the official ``ptbxl_eval.py`` omits step 3; we follow
``dataset.py`` (configurable via ``filter: none``).

Additions needed for heterogeneous datasets (all deterministic, all logged per record):
resampling to 500 Hz (polyphase by default; ``linear`` reproduces the official
``resample_unequal``), lead-name harmonisation, and windowing of records longer than 10 s.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample_poly

from .ecgfounder import MODEL_LEAD_ORDER

TARGET_FS = 500
WINDOW_SAMPLES = 5000

_LEAD_ALIASES = {
    "I": "I", "DI": "I", "LEAD I": "I", "L1": "I",
    "II": "II", "DII": "II", "LEAD II": "II", "L2": "II",
    "III": "III", "DIII": "III", "LEAD III": "III", "L3": "III",
    "AVR": "aVR", "AVL": "aVL", "AVF": "aVF",
    **{f"V{i}": f"V{i}" for i in range(1, 7)},
}


class IncompatibleRecord(Exception):
    """Raised when a recording cannot be fed to ECGFounder without fabricating data."""


def canonical_lead(name: str) -> str | None:
    key = name.strip().upper().replace("-", " ").replace("_", " ")
    return _LEAD_ALIASES.get(key)


@dataclass
class PreprocessConfig:
    target_fs: int = TARGET_FS
    window_samples: int = WINDOW_SAMPLES
    filter: str = "official"            # official | none
    resample_method: str = "polyphase"  # polyphase | linear (official resample_unequal)
    min_duration_s: float = 9.5         # shorter records are rejected (default) ...
    pad_short: bool = True              # ... records in [min_duration_s, 10 s) are edge-padded
    long_record_policy: str = "windows"  # first | windows
    max_windows: int = 6
    max_nan_fraction: float = 0.05
    max_flat_leads: int = 2             # more flat leads than this -> incompatible
    flat_std_mv: float = 1e-4


@dataclass
class Prepared:
    windows: np.ndarray                 # (n_windows, n_leads, 5000) float32
    warnings: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)


def reorder_leads(signal: np.ndarray, lead_names: list[str], wanted: list[str]) -> np.ndarray:
    canon = [canonical_lead(n) for n in lead_names]
    idx = []
    for lead in wanted:
        if lead not in canon:
            raise IncompatibleRecord(f"missing lead {lead} (have {lead_names})")
        idx.append(canon.index(lead))
    return signal[idx]


def resample(signal: np.ndarray, fs_in: float, fs_out: int, method: str) -> np.ndarray:
    if fs_in == fs_out:
        return signal
    if method == "linear":  # official resample_unequal semantics
        n_out = int(round(signal.shape[1] * fs_out / fs_in))
        t = signal.shape[1] / fs_in
        x_old = np.linspace(0, t, num=signal.shape[1], endpoint=True)
        x_new = np.linspace(0, t, num=n_out, endpoint=True)
        return interp1d(x_old, signal, kind="linear", axis=1)(x_new)
    if method == "polyphase":
        frac = Fraction(fs_out / fs_in).limit_denominator(1000)
        return resample_poly(signal, frac.numerator, frac.denominator, axis=1)
    raise ValueError(f"unknown resample method {method}")


def official_filter(signal: np.ndarray, fs: int = TARGET_FS) -> np.ndarray:
    """Exact re-implementation of ``util.filter_bandpass`` from the official repository."""
    b, a = iirnotch(50, 30, fs)
    out = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        out[c] = filtfilt(b, a, signal[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype="bandpass", fs=fs)
    for c in range(signal.shape[0]):
        out[c] = filtfilt(b, a, out[c])
    kernel = int(0.4 * fs) + 1
    if kernel % 2 == 0:
        kernel += 1
    baseline = np.zeros_like(out)
    for c in range(out.shape[0]):
        baseline[c] = medfilt(out[c], kernel_size=kernel)
    return out - baseline


def zscore(signal: np.ndarray) -> np.ndarray:
    """Official normalisation: global mean/std over all leads and samples."""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def preprocess(signal: np.ndarray, fs: float, lead_names: list[str], n_leads: int,
               cfg: PreprocessConfig) -> Prepared:
    """Turn a raw (leads, samples) recording into model-ready 10 s windows."""
    warnings: list[str] = []
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 2:
        raise IncompatibleRecord(f"expected 2-D signal, got shape {signal.shape}")
    if not fs or fs <= 0:
        raise IncompatibleRecord(f"invalid sampling frequency {fs}")
    wanted = MODEL_LEAD_ORDER if n_leads == 12 else ["I"]
    signal = reorder_leads(signal, lead_names, wanted)

    nan_frac = float(np.isnan(signal).mean())
    if nan_frac > cfg.max_nan_fraction:
        raise IncompatibleRecord(f"NaN fraction {nan_frac:.3f} > {cfg.max_nan_fraction}")
    if nan_frac > 0:
        warnings.append(f"nan_fraction={nan_frac:.4f}")
        signal = np.nan_to_num(signal, nan=0.0)

    duration = signal.shape[1] / fs
    if duration < cfg.min_duration_s:
        raise IncompatibleRecord(f"duration {duration:.2f}s < {cfg.min_duration_s}s")

    flat = int((signal.std(axis=1) < cfg.flat_std_mv).sum())
    if flat > cfg.max_flat_leads:
        raise IncompatibleRecord(f"{flat} flat leads (> {cfg.max_flat_leads})")
    if flat:
        warnings.append(f"flat_leads={flat}")

    signal = resample(signal, fs, cfg.target_fs, cfg.resample_method)
    if fs != cfg.target_fs:
        warnings.append(f"resampled_from_{fs:g}Hz")
    if cfg.filter == "official":
        signal = official_filter(signal, cfg.target_fs)
    elif cfg.filter != "none":
        raise ValueError(f"unknown filter {cfg.filter}")

    w = cfg.window_samples
    n = signal.shape[1]
    if n < w:
        if not cfg.pad_short:
            raise IncompatibleRecord(f"{n} samples < {w} and pad_short is false")
        signal = np.pad(signal, ((0, 0), (0, w - n)), mode="edge")
        warnings.append(f"padded_{w - n}_samples")
        starts = [0]
    elif cfg.long_record_policy == "first" or n < 2 * w:
        starts = [0]
        if n > w + cfg.target_fs // 10:
            warnings.append(f"truncated_to_first_10s_of_{n / cfg.target_fs:.1f}s")
    elif cfg.long_record_policy == "windows":
        starts = list(range(0, n - w + 1, w))[: cfg.max_windows]
    else:
        raise ValueError(f"unknown long_record_policy {cfg.long_record_policy}")

    windows = np.stack([zscore(signal[:, s:s + w]) for s in starts]).astype(np.float32)
    return Prepared(windows=windows, warnings=warnings,
                    info={"duration_s": round(duration, 3), "n_windows": len(starts)})
