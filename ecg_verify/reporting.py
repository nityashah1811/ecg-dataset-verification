"""CSV / JSON reports and plots."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .ecgfounder import CLASS_NAMES
from .verification import (CONSISTENT, NOT_COMPARABLE, POT_INCORRECT, POT_MISSING, UNCERTAIN,
                           VerificationResult)

log = logging.getLogger(__name__)

INTERPRETATION = (
    "All rates in this file measure AGREEMENT BETWEEN ECGFounder AND THE DATASET LABELS. They are "
    "not label accuracy and not a label-error rate: ECGFounder is an independent model-based "
    "reference with its own errors, trained on a different population (HEEDB) with a different "
    "label vocabulary (MUSE statements). 'potentially_incorrect' and 'potentially_missing' are "
    "model-supported candidates for expert review, not verified errors. Estimating the true "
    "label-error rate requires an independent expert re-annotation of a (random) sample."
)

RECORD_COLUMNS = [
    "record_id", "original_labels", "predicted_labels", "prediction_scores", "agreement_status",
    "discrepancy_score", "discrepancy_severity", "suggested_additions", "suggested_removals",
]


def _quantiles(x: np.ndarray) -> dict | None:
    if len(x) == 0:
        return None
    q = np.quantile(x, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return dict(zip(["min", "p05", "p25", "median", "p75", "p95", "max"],
                    [round(float(v), 4) for v in q]))


def _hist(x: np.ndarray, bins: int = 20) -> list[int]:
    return np.histogram(x, bins=bins, range=(0, 1))[0].astype(int).tolist()


def write_reports(out: Path, res: VerificationResult, probs: np.ndarray, record_ids: list[str],
                  failed: dict[str, str], n_total: int, run_meta: dict, cfg: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rdf, adf, stats = res.records, res.assessments, res.label_stats

    # record_results.csv: required columns first, then everything else
    cols = RECORD_COLUMNS + [c for c in rdf.columns if c not in RECORD_COLUMNS]
    rdf = rdf[cols]
    if failed:
        frows = pd.DataFrame([{"record_id": k, "agreement_status": "failed",
                               "automated_classification": "failed", "failure_reason": v}
                              for k, v in failed.items()])
        rdf_all = pd.concat([rdf, frows], ignore_index=True)
    else:
        rdf_all = rdf
    rdf_all.to_csv(out / "record_results.csv", index=False)

    flagged_items = adf[adf["status"].isin([POT_INCORRECT, POT_MISSING])] if len(adf) else adf
    disc = rdf[rdf["automated_classification"] == "potential_label_issue"].copy()
    if len(disc):
        detail = (flagged_items.assign(item=lambda d: d["status"].str.replace("potentially_", "")
                                       + ":" + d["label"] + "(" + d["discrepancy"].round(4)
                                       .astype(str) + ")")
                  .groupby("record_id")["item"].apply(lambda s: "; ".join(sorted(s))))
        disc["flag_details"] = disc["record_id"].map(detail)
        disc = disc.sort_values(["discrepancy_score", "n_potentially_incorrect",
                                 "n_potentially_missing", "record_id"],
                                ascending=[False, False, False, True])
    disc.to_csv(out / "label_discrepancies.csv", index=False)

    adf.to_csv(out / "label_assessments.csv", index=False)
    stats.to_csv(out / "label_statistics.csv", index=False)
    rdf[["record_id", "original_labels", "proposed_labels", "suggested_additions",
         "suggested_removals", "discrepancy_score", "discrepancy_severity",
         "automated_classification"]].to_csv(out / "proposed_labels.csv", index=False)
    pd.DataFrame(probs, columns=CLASS_NAMES).round(5).assign(record_id=record_ids)[
        ["record_id"] + CLASS_NAMES].to_csv(out / "ecgfounder_probabilities.csv.gz", index=False)
    pd.DataFrame([{"record_id": k, "reason": v} for k, v in failed.items()],
                 columns=["record_id", "reason"]).to_csv(out / "failed_records.csv", index=False)

    summary = build_summary(res, failed, n_total, run_meta, cfg)
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    if cfg["output"].get("plots", True):
        try:
            make_plots(out / "plots", res, cfg["output"].get("top_labels_in_plots", 24))
        except Exception as e:  # plotting must never lose the computed results
            log.exception("plotting failed: %s", e)
    return summary


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def build_summary(res: VerificationResult, failed: dict, n_total: int, run_meta: dict,
                  cfg: dict) -> dict:
    rdf, adf, stats = res.records, res.assessments, res.label_stats
    n_proc = len(rdf)
    status_counts = rdf["agreement_status"].value_counts().to_dict()
    assessed = n_proc - status_counts.get("not_comparable", 0)
    item_counts = adf["status"].value_counts().to_dict() if len(adf) else {}
    present_items = adf[adf["label_present"]] if len(adf) else adf

    unsupported = (present_items[present_items["status"] == NOT_COMPARABLE]
                   .groupby(["label", "mapping_quality"]).size().reset_index(name="count")
                   .sort_values("count", ascending=False))
    per_label = {}
    for row in stats.to_dict("records"):
        j = res.scored_labels.index(row["label"])
        p, pres = res.label_probs[:, j], res.presence[:, j]
        per_label[row["label"]] = {
            **{k: v for k, v in row.items() if k != "label"},
            "decision_probability_present": _quantiles(p[pres]),
            "decision_probability_absent": _quantiles(p[~pres]),
            "histogram_present_20bins": _hist(p[pres]),
            "histogram_absent_20bins": _hist(p[~pres]),
        }
    calibrated_labels = [d["label"] for d in res.calibration.diagnostics if d["calibrated"]]
    comparable = stats[stats["comparable"]]
    flagged_scores = rdf.loc[rdf["discrepancy_score"] > 0, "discrepancy_score"].to_numpy()

    def rate(n, d):
        return round(n / d, 6) if d else None

    by_strata = {}
    for col in [c for c in rdf.columns if c.startswith("meta_")]:
        vals = rdf[col]
        if vals.nunique() > 30 or vals.nunique() < 2:
            continue
        g = rdf.groupby(col)["agreement_status"].value_counts().unstack(fill_value=0)
        by_strata[col[5:]] = {
            str(k): {"n": int(r.sum()),
                     "disagreement_rate_among_assessed": rate(
                         int(r.get("disagreement", 0)),
                         int(r.sum() - r.get("not_comparable", 0)))}
            for k, r in g.iterrows()}

    return {
        "interpretation": INTERPRETATION,
        "run": run_meta,
        "counts": {
            "total_recordings_in_dataset": n_total,
            "total_recordings_processed": n_proc,
            "failed_or_incompatible_recordings": len(failed),
            "recordings_with_at_least_one_comparable_assessment": assessed,
            "recordings_not_comparable": status_counts.get("not_comparable", 0),
        },
        "record_level": {
            "agreement_rate": rate(status_counts.get("agreement", 0), assessed),
            "disagreement_rate": rate(status_counts.get("disagreement", 0), assessed),
            "uncertain_rate": rate(status_counts.get("uncertain", 0), assessed),
            "denominator": "recordings with at least one comparable label assessment",
            "status_counts": status_counts,
            "severity_counts": rdf["discrepancy_severity"].value_counts().to_dict(),
        },
        "label_level": {
            "assessment_counts": item_counts,
            "present_label_counts_by_status": (present_items["status"].value_counts().to_dict()
                                               if len(present_items) else {}),
            "present_label_agreement_rate": rate(
                item_counts.get(CONSISTENT, 0),
                int(len(present_items)) - int((present_items["status"] == NOT_COMPARABLE).sum())
                if len(present_items) else 0),
            "uncertain_labels": int(item_counts.get(UNCERTAIN, 0)),
            "unsupported_or_unmapped_label_instances": int(
                (present_items["status"] == NOT_COMPARABLE).sum()) if len(present_items) else 0,
            "unsupported_or_unmapped_by_label": unsupported.to_dict("records"),
        },
        "suggestions": {
            "suggested_additions": sum(_explode_counts(rdf["suggested_additions"]).values()),
            "suggested_removals": sum(_explode_counts(rdf["suggested_removals"]).values()),
            "records_with_any_suggestion": int(((rdf["suggested_additions"] != "")
                                                | (rdf["suggested_removals"] != "")).sum()),
            "additions_by_label": _explode_counts(rdf["suggested_additions"]),
            "removals_by_label": _explode_counts(rdf["suggested_removals"]),
        },
        "per_diagnosis": per_label,
        "confidence": {
            "calibration_method": cfg["verification"]["calibration"]["method"],
            "calibrated_labels": calibrated_labels,
            "uncalibrated_comparable_labels": comparable.loc[~comparable["calibrated"], "label"]
            .tolist(),
            "note": ("Calibrated probabilities estimate P(dataset annotators assigned the label | "
                     "ECG) via cross-fitted calibration against the dataset's own labels; labels "
                     "listed as uncalibrated use raw ECGFounder sigmoid scores, which are NOT "
                     "calibrated probabilities, with conservative thresholds."),
            "discrepancy_score_of_flagged_records": _quantiles(flagged_scores),
        },
        "stratified": by_strata,
        "calibration_diagnostics": res.calibration.diagnostics,
    }


def _explode_counts(s: pd.Series) -> dict:
    v = s.str.split(";").explode()
    v = v[v.fillna("") != ""]
    return {k: int(c) for k, c in v.value_counts().items()}


# --------------------------------------------------------------------------- plots
# Palette: validated reference categorical slots (light mode) + neutral gray.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_plots(out: Path, res: VerificationResult, top_n: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    stats = res.label_stats[res.label_stats["comparable"] & (res.label_stats["n_present"] > 0)]
    stats = stats.sort_values("n_present", ascending=False).head(top_n).iloc[::-1]
    if len(stats):
        # 1) per-label outcome of PRESENT labels + missing rate among ABSENT records
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(stats) + 1.6), sharey=True,
                                 gridspec_kw={"width_ratios": [3, 2]})
        y = np.arange(len(stats))
        n = stats["n_present"].to_numpy()
        parts = [("consistent", stats["n_consistent"] / n, BLUE),
                 ("uncertain", stats["n_uncertain_present"] / n, GRAY),
                 ("potentially incorrect", stats["n_potentially_incorrect"] / n, ORANGE)]
        left = np.zeros(len(stats))
        for name, frac, color in parts:
            axes[0].barh(y, frac, left=left, color=color, height=0.7, label=name,
                         edgecolor="white", linewidth=1)
            left += frac.to_numpy()
        axes[0].set_yticks(y, [f"{lab} (n={k})" for lab, k in zip(stats["label"], n)])
        axes[0].set_xlim(0, 1)
        axes[0].set_xlabel("fraction of records carrying the label", color=INK2, fontsize=8)
        axes[0].set_title("ECGFounder vs existing labels", color=INK, fontsize=10, loc="left")
        axes[0].legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.04),
                       ncol=3)
        miss = stats["potentially_missing_rate"].fillna(0).to_numpy()
        axes[1].barh(y, miss, color=AQUA, height=0.7)
        axes[1].set_xlim(0, max(0.05, float(miss.max()) * 1.1))
        axes[1].set_xlabel("fraction of records WITHOUT the label", color=INK2, fontsize=8)
        axes[1].set_title("Potentially missing", color=INK, fontsize=10, loc="left")
        for a in axes:
            _style(a)
        fig.text(0.01, 0.005, "Model-vs-label agreement, not label accuracy.", fontsize=7,
                 color=INK2)
        fig.tight_layout()
        fig.savefig(out / "per_label_agreement.png", dpi=150)
        plt.close(fig)

        # 2) confidence distributions, present vs absent, per label (small multiples)
        labs = stats["label"].tolist()[::-1][:min(top_n, 16)]
        ncol = 4
        nrow = int(np.ceil(len(labs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow), squeeze=False)
        bins = np.linspace(0, 1, 26)
        for k, lab in enumerate(labs):
            ax = axes[k // ncol][k % ncol]
            j = res.scored_labels.index(lab)
            p, pres = res.label_probs[:, j], res.presence[:, j]
            ax.hist(p[~pres], bins=bins, color=GRAY, alpha=0.9, density=True, label="label absent")
            ax.hist(p[pres], bins=bins, color=BLUE, alpha=0.75, density=True, label="label present")
            ax.set_yscale("log")
            cal = bool(res.calibration.calibrated[j])
            ax.set_title(f"{lab} ({'calibrated' if cal else 'raw score'})", fontsize=8, color=INK,
                         loc="left")
            _style(ax)
            ax.grid(False)
        for k in range(len(labs), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        axes[0][0].legend(frameon=False, fontsize=7)
        fig.suptitle("Decision probability by original label status (log density)", fontsize=10,
                     color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "confidence_distributions.png", dpi=150)
        plt.close(fig)

    # 3) discrepancy scores of flagged records
    d = res.records.loc[res.records["discrepancy_score"] > 0, "discrepancy_score"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(d, bins=np.linspace(0.9, 1, 21) if len(d) and d.min() >= 0.9 else 20, color=ORANGE,
            edgecolor="white")
    ax.set_xlabel("record discrepancy score (flagged records only)", color=INK2, fontsize=8)
    ax.set_ylabel("records", color=INK2, fontsize=8)
    ax.set_title(f"Discrepancy scores (n={len(d)} flagged records)", fontsize=10, color=INK,
                 loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "discrepancy_scores.png", dpi=150)
    plt.close(fig)

    # 4) reliability diagrams of the cross-fitted calibration
    diags = [d for d in res.calibration.diagnostics if d.get("calibrated")]
    diags = sorted(diags, key=lambda d: -d["n_labelled"])[:12]
    if diags:
        ncol = 4
        nrow = int(np.ceil(len(diags) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.8 * nrow), squeeze=False)
        for k, dg in enumerate(diags):
            ax = axes[k // ncol][k % ncol]
            rc = dg["reliability_crossfit"]
            xs = [x for x in rc["mean_predicted"] if x is not None]
            ys = [y for x, y in zip(rc["mean_predicted"], rc["fraction_labelled"]) if x is not None]
            ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1, linestyle="--")
            ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=4)
            ax.set_title(f"{dg['label']}  ECE {dg['ece_crossfit']:.3f}", fontsize=8, color=INK,
                         loc="left")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            _style(ax)
        for k in range(len(diags), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.suptitle("Cross-fitted calibration: predicted vs fraction labelled", fontsize=10,
                     color=INK, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "calibration_reliability.png", dpi=150)
        plt.close(fig)
