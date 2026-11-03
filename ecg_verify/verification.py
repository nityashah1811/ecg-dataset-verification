"""Model-versus-label comparison, discrepancy scoring and label suggestions.

Terminology (kept deliberately strict):
* "agreement"/"disagreement" always means agreement between ECGFounder and the dataset label.
  It is NOT label accuracy: ECGFounder is an independent model-based reference, not ground truth.
* "potentially incorrect" / "potentially missing" are model-supported *candidates* for review,
  never verified errors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .calibration import CalibrationConfig, CalibrationResult, crossfit_calibrate
from .ecgfounder import CLASS_NAMES
from .loaders import RecordRef, stable_fold
from .mapping import LabelMapping

CONSISTENT = "consistent"
POT_INCORRECT = "potentially_incorrect"
POT_MISSING = "potentially_missing"
UNCERTAIN = "uncertain"
NOT_COMPARABLE = "not_comparable"
NOT_ASSESSED = "absent_not_assessed"      # absent label, absence not a confirmed negative
CONSISTENT_ABSENT = "consistent_absent"   # absent label, absence is a negative, model agrees

FLAG_STATUSES = (POT_INCORRECT, POT_MISSING)


@dataclass
class VerificationResult:
    records: pd.DataFrame
    assessments: pd.DataFrame
    label_stats: pd.DataFrame
    calibration: CalibrationResult
    scored_labels: list[str]
    label_scores: np.ndarray          # (N, L) raw any-of scores
    label_probs: np.ndarray           # (N, L) decision probabilities (calibrated where possible)
    presence: np.ndarray              # (N, L) bool
    folds: np.ndarray


def assign_folds(refs: list[RecordRef], n_folds: int) -> np.ndarray:
    folds = np.empty(len(refs), dtype=int)
    for i, r in enumerate(refs):
        if r.fold_hint is not None:
            folds[i] = (int(r.fold_hint) - 1) % n_folds
        else:
            folds[i] = stable_fold(r.patient_id or r.record_id, n_folds)
    return folds


def _thresholds(vcfg: dict, label: str, calibrated: bool) -> dict:
    """Thresholds on the *evidence* scale (present labels) and posterior scale (missing).

    Calibrated labels are configured as Bayes factors and converted to prior-neutral
    probabilities BF / (1 + BF); uncalibrated labels use raw-score thresholds directly.
    """
    base = dict(vcfg["calibrated_thresholds"] if calibrated else vcfg["uncalibrated_thresholds"])
    base.update(vcfg.get("threshold_overrides", {}).get(label, {}))
    if calibrated:
        sup, con = base.pop("support_bayes_factor"), base.pop("contradict_bayes_factor")
        if not 0 < con < 1 < sup:
            raise ValueError(f"thresholds for {label}: need 0 < contradict_bayes_factor < 1 < "
                             "support_bayes_factor")
        base.update(support=sup / (1 + sup), contradict=con / (1 + con),
                    support_bayes_factor=sup, contradict_bayes_factor=con)
    if not 0 < base["contradict"] < base["support"] < 1 or not 0 < base["missing"] < 1:
        raise ValueError(f"thresholds for {label} must satisfy 0 < contradict < support < 1 "
                         "and 0 < missing < 1")
    return base


def _odds(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return p / (1 - p)


def bayes_factors(posterior: np.ndarray, presence: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Likelihood ratio p(ECG | label) / p(ECG | no label) implied by a cross-fitted posterior.

    The calibrator applied to fold k was fitted on the other folds, so its implicit prior is the
    label prevalence outside fold k. Dividing the posterior odds by that prior odds removes the
    base rate: a rare label then no longer looks "contradicted" merely because it is rare.
    """
    bf = np.full(posterior.shape, np.nan)
    for k in np.unique(folds):
        te = folds == k
        if te.all():
            continue  # single fold: nothing was calibrated, BF undefined
        prior = presence[~te].mean(axis=0)
        bf[te] = _odds(posterior[te]) / _odds(prior)[None, :]
    return bf


def _severity(d: float, bins: dict) -> str:
    if d >= bins["high"]:
        return "high"
    if d >= bins["medium"]:
        return "medium"
    return "low"


def verify(refs: list[RecordRef], probs: np.ndarray, mapping: LabelMapping, cfg: dict,
           absence_semantics: str, record_info: dict | None = None) -> VerificationResult:
    vcfg, scfg = cfg["verification"], cfg["suggestions"]
    comparable_q = tuple(vcfg["comparable_qualities"])
    suggest_q = tuple(scfg["qualities"])
    if absence_semantics not in ("unknown", "negative"):
        raise ValueError("absence_semantics must be 'unknown' or 'negative'")
    record_info = record_info or {}

    scored = [m.label for m in mapping.labels.values() if m.supported]
    col = {lab: j for j, lab in enumerate(scored)}
    scores = mapping.label_scores(probs, scored)
    presence = np.array([[lab in r.labels for lab in scored] for r in refs], dtype=bool)
    presence = presence.reshape(len(refs), len(scored))

    ccfg = CalibrationConfig(**vcfg["calibration"])
    folds = assign_folds(refs, ccfg.n_folds)
    cal = crossfit_calibrate(scores, presence, folds, scored, ccfg)
    # posterior: P(dataset annotators assign label | ECG) -> used for "potentially missing"
    decision = np.where(cal.calibrated[None, :], cal.probs, scores)
    # evidence: prior-neutral probability BF/(1+BF) -> used for judging existing labels
    bf = bayes_factors(np.nan_to_num(cal.probs, nan=0.5), presence, folds)
    evidence = np.where(cal.calibrated[None, :], bf / (1 + bf), scores)
    thr = {lab: _thresholds(vcfg, lab, bool(cal.calibrated[j])) for lab, j in col.items()}

    native_thr = vcfg["native_report_threshold"]
    assessments, records = [], []
    for i, ref in enumerate(refs):
        present = set(ref.labels)
        items = []

        def add(label, status, reason="", j=None, suggestion=""):
            m = mapping.get(label)
            item = {
                "record_id": ref.record_id, "label": label,
                "label_present": label in present,
                "original_annotation_value": ref.labels.get(label),
                "mapping_quality": (m.quality if m and m.supported else
                                    ("unsupported" if m else "unmapped")),
                "model_classes": "; ".join(m.model_classes) if m else "",
                "raw_score": None, "calibrated_probability": None, "bayes_factor": None,
                "calibrated": None,
                "status": status, "discrepancy": 0.0, "suggestion": suggestion, "reason": reason,
            }
            if j is not None:
                item["raw_score"] = float(scores[i, j])
                item["calibrated"] = bool(cal.calibrated[j])
                if cal.calibrated[j]:
                    item["calibrated_probability"] = float(cal.probs[i, j])
                    item["bayes_factor"] = float(bf[i, j])
                if status == POT_INCORRECT:
                    item["discrepancy"] = 1.0 - float(evidence[i, j])
                elif status == POT_MISSING:
                    item["discrepancy"] = float(decision[i, j])
            items.append(item)

        # 1) every original label
        for label in sorted(present):
            m = mapping.get(label)
            if m is None:
                add(label, NOT_COMPARABLE, "label not in mapping (unmapped)")
                continue
            if not m.supported:
                add(label, NOT_COMPARABLE, f"unsupported: {m.unsupported_reason}")
                continue
            j = col[label]
            if m.quality not in comparable_q:
                add(label, NOT_COMPARABLE, f"{m.quality} mapping not enabled for decisions", j)
                continue
            t, e = thr[label], evidence[i, j]
            what = f"BF={bf[i, j]:.3g}" if cal.calibrated[j] else f"raw score={e:.3f}"
            if e >= t["support"] or decision[i, j] >= t.get("support_posterior", 1.1):
                add(label, CONSISTENT, "", j)
            elif e <= t["contradict"]:
                sugg = "remove" if (scfg["propose_removals"] and m.quality in suggest_q) else ""
                add(label, POT_INCORRECT, f"{what}: strong evidence against the label", j, sugg)
            else:
                add(label, UNCERTAIN, f"{what}: inconclusive", j)

        # 2) comparable labels the record does NOT carry
        for label, j in col.items():
            if label in present:
                continue
            m = mapping.labels[label]
            if m.quality not in comparable_q:
                continue
            t, p = thr[label], decision[i, j]
            if p >= t["missing"]:
                covering = [c for c in m.covered_by if c in present]
                conflicts = m.conflicting(present)
                if covering:
                    continue  # finding already expressed by an equivalent label
                if not m.suggest:
                    add(label, UNCERTAIN, "strong prediction but additions disabled for label", j)
                elif conflicts:
                    add(label, UNCERTAIN, "strong prediction conflicts with present label(s): "
                        + ", ".join(conflicts), j)
                else:
                    sugg = "add" if (scfg["propose_additions"] and m.quality in suggest_q) else ""
                    add(label, POT_MISSING, f"P={p:.3f} >= missing threshold {t['missing']}", j,
                        sugg)
            elif absence_semantics == "negative":
                if evidence[i, j] < t["support"]:
                    pass  # counted in label stats as consistent_absent; not itemised
                else:
                    add(label, UNCERTAIN, f"absent label with supporting evidence but "
                        f"P={p:.3f} < missing threshold", j)

        assessments.extend(items)
        records.append(_record_row(ref, items, probs[i], native_thr, vcfg["severity_bins"],
                                   record_info.get(ref.record_id, {}), folds[i]))

    adf = pd.DataFrame(assessments)
    rdf = pd.DataFrame(records)
    stats = _label_stats(mapping, scored, presence, evidence, cal, thr, comparable_q,
                         absence_semantics, adf)
    return VerificationResult(rdf, adf, stats, cal, scored, scores, decision, presence, folds)


def _record_row(ref, items, native_probs, native_thr, bins, info, fold) -> dict:
    by = {s: [it for it in items if it["status"] == s] for s in
          (CONSISTENT, POT_INCORRECT, POT_MISSING, UNCERTAIN, NOT_COMPARABLE)}
    flagged = by[POT_INCORRECT] + by[POT_MISSING]
    d = max((it["discrepancy"] for it in flagged), default=0.0)
    if flagged:
        agreement, final = "disagreement", "potential_label_issue"
    elif by[UNCERTAIN]:
        agreement, final = "uncertain", "uncertain"
    elif by[CONSISTENT]:
        agreement, final = "agreement", "consistent"
    else:
        agreement, final = "not_comparable", "not_comparable"
    removals = sorted(it["label"] for it in by[POT_INCORRECT] if it["suggestion"] == "remove")
    additions = sorted(it["label"] for it in by[POT_MISSING] if it["suggestion"] == "add")
    original = sorted(ref.labels)
    proposed = sorted((set(original) - set(removals)) | set(additions))
    scores = {it["label"]: round(it["calibrated_probability"] if it["calibrated"]
                                 else it["raw_score"], 4)
              for it in items if it["raw_score"] is not None}
    predicted = sorted(it["label"] for it in items
                       if it["status"] in (CONSISTENT, POT_MISSING))
    native = {CLASS_NAMES[k]: round(float(native_probs[k]), 4)
              for k in np.argsort(-native_probs)[:15] if native_probs[k] >= native_thr}
    return {
        "record_id": ref.record_id,
        "patient_id": ref.patient_id,
        "calibration_fold": int(fold),
        "original_labels": ";".join(original),
        "original_label_values": json.dumps(ref.labels, sort_keys=True),
        "comparable_labels": ";".join(sorted(it["label"] for it in items if it["label_present"]
                                             and it["status"] != NOT_COMPARABLE)),
        "not_comparable_labels": ";".join(sorted(it["label"] for it in by[NOT_COMPARABLE])),
        "predicted_labels": ";".join(predicted),
        "prediction_scores": json.dumps(scores, sort_keys=True),
        "ecgfounder_native_predictions": json.dumps(native),
        "agreement_status": agreement,
        "n_consistent": len(by[CONSISTENT]),
        "n_potentially_incorrect": len(by[POT_INCORRECT]),
        "n_potentially_missing": len(by[POT_MISSING]),
        "n_uncertain": len(by[UNCERTAIN]),
        "n_not_comparable": len(by[NOT_COMPARABLE]),
        "potentially_incorrect_labels": ";".join(sorted(it["label"] for it in by[POT_INCORRECT])),
        "potentially_missing_labels": ";".join(sorted(it["label"] for it in by[POT_MISSING])),
        "uncertain_labels": ";".join(sorted(it["label"] for it in by[UNCERTAIN])),
        "discrepancy_score": round(d, 6),
        "discrepancy_severity": _severity(d, bins) if flagged else "none",
        "suggested_additions": ";".join(additions),
        "suggested_removals": ";".join(removals),
        "proposed_labels": ";".join(proposed),
        "automated_classification": final,
        "signal_warnings": ";".join(info.get("warnings", [])),
        "n_windows": info.get("n_windows"),
        **{f"meta_{k}": v for k, v in ref.metadata.items()
           if not isinstance(v, (dict, list)) and k != "report"},
    }


def _label_stats(mapping, scored, presence, evidence, cal, thr, comparable_q, absence, adf):
    rows = []
    for j, label in enumerate(scored):
        m = mapping.labels[label]
        pres = presence[:, j]
        p = evidence[:, j]
        t = thr[label]
        comparable = m.quality in comparable_q
        sub = adf[adf["label"] == label] if len(adf) else adf
        counts = sub["status"].value_counts().to_dict() if len(sub) else {}
        n_pres, n_abs = int(pres.sum()), int((~pres).sum())
        row = {
            "label": label, "description": m.description, "mapping_quality": m.quality,
            "comparable": comparable, "calibrated": bool(cal.calibrated[j]),
            "n_present": n_pres, "n_absent": n_abs,
            "threshold_support": t["support"], "threshold_contradict": t["contradict"],
            "threshold_support_bayes_factor": t.get("support_bayes_factor"),
            "threshold_contradict_bayes_factor": t.get("contradict_bayes_factor"),
            "threshold_missing": t["missing"],
            "n_consistent": int(counts.get(CONSISTENT, 0)),
            "n_potentially_incorrect": int(counts.get(POT_INCORRECT, 0)),
            "n_uncertain_present": int(((sub["status"] == UNCERTAIN) & sub["label_present"]).sum())
            if len(sub) else 0,
            "n_potentially_missing": int(counts.get(POT_MISSING, 0)),
            "n_uncertain_absent": int(((sub["status"] == UNCERTAIN) & ~sub["label_present"]).sum())
            if len(sub) else 0,
            "n_consistent_absent": (int(((~pres) & (p < t["support"])).sum())
                                    if comparable and absence == "negative" else None),
            "n_not_comparable_present": int(counts.get(NOT_COMPARABLE, 0)),
        }
        row["potentially_incorrect_rate"] = (row["n_potentially_incorrect"] / n_pres
                                             if comparable and n_pres else None)
        row["consistent_rate"] = row["n_consistent"] / n_pres if comparable and n_pres else None
        row["potentially_missing_rate"] = (row["n_potentially_missing"] / n_abs
                                           if comparable and n_abs else None)
        row["model_positive_rate"] = float((p >= t["support"]).mean())
        row["label_prevalence"] = float(pres.mean())
        diag = cal.diagnostics[j]
        for k in ("auroc_raw_vs_label", "auprc_raw_vs_label", "ece_raw", "ece_crossfit",
                  "brier_raw", "brier_crossfit"):
            row[k] = diag.get(k)
        row["calibration_note"] = diag.get("reason")
        rows.append(row)
    return pd.DataFrame(rows)
