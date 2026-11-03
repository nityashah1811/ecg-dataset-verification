"""Cross-fitted, per-label probability calibration.

ECGFounder's sigmoid outputs are not calibrated, and they certainly are not calibrated to the
annotation conventions of an external dataset (domain shift, different vocabularies, different
prevalence). For each comparable dataset label we therefore map the any-of label score ``s`` to
``P(dataset annotators assigned this label | ECG)`` with Platt scaling on ``logit(s)`` (or isotonic
regression), fitted with K-fold *cross-fitting*: the calibrator applied to a record never saw
that record (folds are patient-grouped where patient ids exist).

Caveats, surfaced in the reports:
* the calibration target is the dataset's own labels, which contain the very errors being
  searched for. Label noise flattens the calibration curve, which makes flags *more*
  conservative, not less;
* under positive-only annotation an absent label is treated as negative for fitting, which
  under-estimates P(label) for under-annotated findings;
* labels with too few positives/negatives in any training split stay uncalibrated and are judged
  with the conservative raw-probability thresholds instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-6


@dataclass
class CalibrationConfig:
    method: str = "platt"        # platt | isotonic | none
    n_folds: int = 5
    min_positives: int = 20      # per training split
    min_negatives: int = 20


@dataclass
class CalibrationResult:
    probs: np.ndarray            # (N, L) calibrated probabilities; NaN where uncalibrated
    calibrated: np.ndarray       # (L,) bool
    diagnostics: list[dict] = field(default_factory=list)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def _fit_predict(method: str, s_train, y_train, s_test) -> np.ndarray:
    if method == "platt":
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
        lr.fit(_logit(s_train)[:, None], y_train)
        return lr.predict_proba(_logit(s_test)[:, None])[:, 1]
    if method == "isotonic":
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(s_train, y_train)
        # Isotonic steps can hit exactly 0/1; keep a small floor so discrepancy ranks stay finite.
        return np.clip(iso.predict(s_test), 1e-4, 1 - 1e-4)
    raise ValueError(f"unknown calibration method {method}")


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_curve(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    mean_p, frac_pos, count = [], [], []
    for b in range(n_bins):
        m = bins == b
        mean_p.append(float(p[m].mean()) if m.any() else None)
        frac_pos.append(float(y[m].mean()) if m.any() else None)
        count.append(int(m.sum()))
    return {"mean_predicted": mean_p, "fraction_labelled": frac_pos, "count": count}


def crossfit_calibrate(scores: np.ndarray, y: np.ndarray, folds: np.ndarray,
                       labels: list[str], cfg: CalibrationConfig) -> CalibrationResult:
    n, n_labels = scores.shape
    out = np.full((n, n_labels), np.nan)
    calibrated = np.zeros(n_labels, dtype=bool)
    diagnostics = []
    fold_ids = np.unique(folds)
    for j, label in enumerate(labels):
        s, t = scores[:, j], y[:, j].astype(int)
        diag = {"label": label, "n_labelled": int(t.sum()), "n_records": int(n),
                "calibrated": False, "reason": None}
        if t.min() != t.max():
            diag["auroc_raw_vs_label"] = float(roc_auc_score(t, s))
            diag["auprc_raw_vs_label"] = float(average_precision_score(t, s))
        if cfg.method == "none":
            diag["reason"] = "calibration disabled"
        elif len(fold_ids) < 2:
            diag["reason"] = "fewer than 2 folds"
        else:
            ok = True
            for k in fold_ids:
                tr = folds != k
                if t[tr].sum() < cfg.min_positives or (1 - t[tr]).sum() < cfg.min_negatives:
                    ok = False
                    diag["reason"] = (f"fewer than {cfg.min_positives} positives / "
                                      f"{cfg.min_negatives} negatives in a training split")
                    break
            if ok:
                for k in fold_ids:
                    te = folds == k
                    out[te, j] = _fit_predict(cfg.method, s[~te], t[~te], s[te])
                calibrated[j] = True
                p = out[:, j]
                diag.update(calibrated=True, method=cfg.method,
                            ece_crossfit=expected_calibration_error(p, t),
                            ece_raw=expected_calibration_error(s, t),
                            brier_crossfit=float(np.mean((p - t) ** 2)),
                            brier_raw=float(np.mean((s - t) ** 2)),
                            reliability_crossfit=reliability_curve(p, t))
        diagnostics.append(diag)
    return CalibrationResult(probs=out, calibrated=calibrated, diagnostics=diagnostics)
