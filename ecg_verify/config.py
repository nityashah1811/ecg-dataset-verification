"""Run configuration: defaults <- YAML file <- command-line overrides."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "dataset": {
        "format": "auto",            # auto | ptbxl | wfdb_dx | wfdb_manifest | code15_hdf5
        "mapping": None,             # mapping name in mappings/ or path; None = adapter default
        "absence_semantics": None,   # unknown | negative; None = adapter default
        "options": {},               # adapter-specific (e.g. ptbxl: min_label_likelihood)
        "limit": None,               # process only the first N records (smoke tests)
    },
    "model": {
        "checkpoint": "checkpoints/12_lead_ECGFounder.pth",
        "n_leads": 12,
        "device": "auto",            # auto | cpu | cuda | cuda:N
        "batch_size": 64,
        "num_workers": 2,
        "deterministic": True,
        "seed": 0,
    },
    "preprocessing": {
        "filter": "official",
        "resample_method": "polyphase",
        "min_duration_s": 9.5,
        "pad_short": True,
        "long_record_policy": "windows",
        "window_aggregation": "mean",  # mean | max (over 10 s windows of long records)
        "max_windows": 6,
        "max_nan_fraction": 0.05,
        "max_flat_leads": 2,
    },
    "verification": {
        # Mapping qualities treated as comparable. "approximate" mappings are always scored
        # and reported, but only produce decisions if listed here.
        "comparable_qualities": ["exact"],
        "calibration": {"method": "platt", "n_folds": 5, "min_positives": 20,
                        "min_negatives": 20},
        # Calibrated labels. Existing labels are judged by the Bayes factor
        # BF = p(ECG | label) / p(ECG | no label) (posterior odds / prior odds), so that rarity
        # alone never counts as evidence against a label. Kass & Raftery (1995) scale:
        # BF >= 3 "positive" support, BF <= 1/20 "strong" evidence against. An existing label is
        # also consistent when the posterior P(annotators assign label | ECG) >= support_posterior
        # (needed for very common labels, whose BF is bounded by their high prior).
        # Absent labels are judged by the posterior.
        "calibrated_thresholds": {"support_bayes_factor": 3.0, "support_posterior": 0.5,
                                  "contradict_bayes_factor": 0.05, "missing": 0.95},
        # Conservative thresholds on the raw sigmoid score, used where calibration is unavailable.
        "uncalibrated_thresholds": {"support": 0.5, "contradict": 0.02, "missing": 0.98},
        # Per-label overrides, e.g. {"AFIB": {"missing": 0.9, "contradict_bayes_factor": 0.02}}.
        "threshold_overrides": {},
        "severity_bins": {"high": 0.99, "medium": 0.97},
        "native_report_threshold": 0.5,
    },
    "suggestions": {
        "propose_additions": True,
        "propose_removals": True,
        "qualities": ["exact"],
    },
    "output": {
        "plots": True,
        "top_labels_in_plots": 24,
    },
}

# Keys whose change invalidates cached ECGFounder outputs.
INFERENCE_KEYS = (("model", "checkpoint"), ("model", "n_leads"), ("preprocessing",),
                  ("dataset", "format"), ("dataset", "options"))


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        deep_update(cfg, yaml.safe_load(Path(path).read_text()) or {})
    deep_update(cfg, overrides or {})
    return cfg


def fingerprint(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def inference_fingerprint(cfg: dict, checkpoint_sha256: str, dataset_root: str) -> str:
    parts = {"/".join(k): _get(cfg, k) for k in INFERENCE_KEYS}
    parts["checkpoint_sha256"] = checkpoint_sha256
    parts["dataset_root"] = str(Path(dataset_root).resolve())
    return fingerprint(parts)


def _get(d: dict, keys: tuple[str, ...]):
    for k in keys:
        d = d.get(k) if isinstance(d, dict) else None
    return d
