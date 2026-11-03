"""Shared loading for the PTB-XL follow-up analyses.

Rebuilds, from the cached ECGFounder outputs of a pipeline run, exactly the quantities used by
`ecg_verify.verification.verify` (label scores, presence, folds, cross-fitted posteriors, Bayes
factors), so that sensitivity analyses and baselines are evaluated on identical inputs.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from ecg_verify.calibration import CalibrationConfig, crossfit_calibrate  # noqa: E402
from ecg_verify.config import load_config  # noqa: E402
from ecg_verify.inference import load_cache  # noqa: E402
from ecg_verify.loaders import get_adapter  # noqa: E402
from ecg_verify.mapping import load_mapping  # noqa: E402
from ecg_verify.verification import assign_folds, bayes_factors  # noqa: E402
from ptbxl_report_concordance import KEYWORDS  # noqa: E402,F401


@dataclass
class Run:
    refs: list
    record_ids: np.ndarray
    probs: np.ndarray           # (N, 150) raw ECGFounder sigmoid outputs
    mapping: object
    labels: list[str]           # comparable (exact) labels
    scores: np.ndarray          # (N, L) any-of raw label scores
    presence: np.ndarray        # (N, L) bool
    folds: np.ndarray
    posterior: np.ndarray       # (N, L) cross-fitted calibrated posterior (raw if uncalibrated)
    bf: np.ndarray              # (N, L) Bayes factor (nan if uncalibrated)
    calibrated: np.ndarray      # (L,) bool
    covered: np.ndarray         # (N, L) bool: an equivalent label (covered_by) is present
    conflict: np.ndarray        # (N, L) bool: adding the label would conflict with a present one
    reports: pd.Series          # record_id -> lower-cased free-text report
    meta: pd.DataFrame          # ptbxl_database rows indexed by record_id


def load_run(dataset: str, results: str, cfg_path: str | None = None) -> Run:
    cfg = load_config(cfg_path or str(Path(results) / "config_resolved.yaml"))
    adapter = get_adapter(dataset, cfg["dataset"]["format"], **(cfg["dataset"].get("options") or {}))
    mapping = load_mapping(cfg["dataset"]["mapping"] or adapter.default_mapping)
    refs = adapter.records()
    cache_dirs = sorted((Path(results) / "cache").glob("inference_*"))
    if len(cache_dirs) != 1:
        raise RuntimeError(f"expected one inference cache in {results}/cache, found {cache_dirs}")
    probs_by_id, _ = load_cache(cache_dirs[0])
    refs = [r for r in refs if r.record_id in probs_by_id]
    probs = np.stack([probs_by_id[r.record_id] for r in refs]).astype(np.float64)

    q = tuple(cfg["verification"]["comparable_qualities"])
    labels = [m.label for m in mapping.comparable(q)]
    scores = mapping.label_scores(probs, labels)
    presence = np.array([[lab in r.labels for lab in labels] for r in refs], dtype=bool)
    ccfg = CalibrationConfig(**cfg["verification"]["calibration"])
    folds = assign_folds(refs, ccfg.n_folds)
    cal = crossfit_calibrate(scores, presence, folds, labels, ccfg)
    posterior = np.where(cal.calibrated[None, :], cal.probs, scores)
    bf = bayes_factors(np.nan_to_num(cal.probs, nan=0.5), presence, folds)
    bf[:, ~cal.calibrated] = np.nan

    col = {lab: j for j, lab in enumerate(labels)}
    covered = np.zeros_like(presence)
    conflict = np.zeros_like(presence)
    for j, lab in enumerate(labels):
        m = mapping.labels[lab]
        for c in m.covered_by:
            if c in col:
                covered[:, j] |= presence[:, col[c]]
            else:
                covered[:, j] |= np.array([c in r.labels for r in refs])
        if m.exclusive_with:
            conflict[:, j] = [bool(m.conflicting(set(r.labels))) for r in refs]

    db = pd.read_csv(Path(dataset) / "ptbxl_database.csv")
    db["record_id"] = db.ecg_id.astype(str)
    db = db.set_index("record_id")
    ids = np.array([r.record_id for r in refs])
    return Run(refs, ids, probs, mapping, labels, scores, presence, folds, posterior, bf,
               cal.calibrated, covered, conflict, db.loc[ids, "report"].fillna("").str.lower(),
               db.loc[ids])


def flags_from_rule(run: Run, contradict_bf=0.05, missing_p=0.95, raw_contradict=0.02,
                    raw_missing=0.98):
    """(incorrect, missing) boolean (N, L) matrices under the pipeline's decision rule."""
    cal = run.calibrated[None, :]
    incorrect = run.presence & np.where(cal, run.bf <= contradict_bf, run.scores <= raw_contradict)
    strong = np.where(cal, run.posterior >= missing_p, run.scores >= raw_missing)
    missing = ~run.presence & strong & ~run.covered & ~run.conflict
    return incorrect, missing


def report_mentions(run: Run) -> dict[str, np.ndarray]:
    return {lab: run.reports.str.contains(pat, regex=True).to_numpy()
            for lab, pat in KEYWORDS.items() if lab in run.labels}


def report_support(run: Run, incorrect: np.ndarray, missing: np.ndarray) -> dict:
    """Pooled report-text support for flags on keyword-checkable labels, vs unflagged controls."""
    men = report_mentions(run)
    out = {"missing": [0, 0, 0, 0], "incorrect": [0, 0, 0, 0]}  # supported, flagged, ctrl_sup, ctrl_n
    out["tables"] = {"missing": [], "incorrect": []}            # per-label tables (for MH OR)
    for lab, mention in men.items():
        j = run.labels.index(lab)
        for kind, flag, pool, supports in (("missing", missing[:, j], ~run.presence[:, j], mention),
                                           ("incorrect", incorrect[:, j], run.presence[:, j], ~mention)):
            f, c = pool & flag, pool & ~flag
            t = (int((supports & f).sum()), int(f.sum()), int((supports & c).sum()), int(c.sum()))
            out["tables"][kind].append(t)
            for q in range(4):
                out[kind][q] += t[q]
    return out
