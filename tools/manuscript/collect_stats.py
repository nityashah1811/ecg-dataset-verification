#!/usr/bin/env python
"""Collect every number the manuscript quotes from reports/ptbxl_ecgfounder/ into one JSON.

The manuscript builder (build_manuscript.js) reads only this file and the figures, so every
figure in the paper traces back to a pipeline output.

    python tools/manuscript/collect_stats.py --reports reports --out docs/manuscript/build
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, GRAY, INK, INK2, GRID = "#2a78d6", "#a3a29c", "#0b0b0b", "#52514e", "#e6e5e0"
EXAMPLES = ["13424", "11583", "287", "635", "16195", "7783"]


def pct(n, d, k=1):
    return round(100.0 * n / d, k) if d else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default="docs/manuscript/build")
    ap.add_argument("--ptbxl", help="PTB-XL root (for report excerpts of example records)")
    a = ap.parse_args(argv)
    rep, out = Path(a.reports), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = rep / "ptbxl_ecgfounder"

    s = json.loads((res / "dataset_summary.json").read_text())
    manifest = json.loads((res / "run_manifest.json").read_text())
    pre = json.loads((rep / "ptbxl_preflight" / "data_check.json").read_text())
    inv = pd.read_csv(rep / "ptbxl_preflight" / "label_inventory.csv")
    stats = pd.read_csv(res / "label_statistics.csv")
    conc = pd.read_csv(res / "report_concordance.csv")
    rr = pd.read_csv(res / "record_results.csv.gz", dtype={"record_id": str}, keep_default_na=False)

    ex = stats[stats["comparable"]].copy().sort_values("n_present", ascending=False)
    cal = ex[ex["calibrated"]]
    rl, ll = s["record_level"], s["label_level"]
    present = ll["present_label_counts_by_status"]
    n_present_comp = sum(v for k, v in present.items() if k != "not_comparable")
    lab_status = inv.groupby("mapping")["count"].sum().to_dict()
    n_lab = int(inv["count"].sum())

    strat = s["stratified"]
    dev = {k: v for k, v in strat["device"].items() if v["n"] > 500}
    dev_rates = [v["disagreement_rate_among_assessed"] for v in dev.values()]

    per_label = []
    for r in ex.itertuples():
        per_label.append({
            "label": r.label, "description": r.description, "n": int(r.n_present),
            "consistent_pct": pct(r.n_consistent, r.n_present),
            "uncertain_pct": pct(r.n_uncertain_present, r.n_present),
            "incorrect": int(r.n_potentially_incorrect),
            "missing": int(r.n_potentially_missing),
            "auroc": round(float(r.auroc_raw_vs_label), 3),
            "ece_raw": None if np.isnan(r.ece_raw) else round(float(r.ece_raw), 3),
            "ece_cal": None if np.isnan(r.ece_crossfit) else round(float(r.ece_crossfit), 3),
            "calibrated": bool(r.calibrated),
        })

    data = {
        "n_records": pre["records"], "n_patients": int(rr["patient_id"].nunique()), "n_failed": s["counts"]["failed_or_incompatible_recordings"],
        "n_validated": int(rr["meta_validated_by_human"].astype(str).eq("True").sum()),
        "n_label_instances": n_lab,
        "lab_exact": int(lab_status.get("exact", 0)), "lab_approx": int(lab_status.get("approximate", 0)),
        "lab_unsupported": int(lab_status.get("unsupported", 0)),
        "lab_exact_pct": pct(lab_status.get("exact", 0), n_lab), "lab_approx_pct": pct(lab_status.get("approximate", 0), n_lab),
        "lab_unsupported_pct": pct(lab_status.get("unsupported", 0), n_lab),
        "n_assessable": s["counts"]["recordings_with_at_least_one_comparable_assessment"],
        "n_not_comparable_records": s["counts"]["recordings_not_comparable"],
        "rec_counts": rl["status_counts"], "severity": rl["severity_counts"],
        "agreement_pct": round(100 * rl["agreement_rate"], 1), "uncertain_pct": round(100 * rl["uncertain_rate"], 1),
        "disagreement_pct": round(100 * rl["disagreement_rate"], 2),
        "present": present, "n_present_comparable": n_present_comp,
        "present_consistent_pct": pct(present.get("consistent", 0), n_present_comp),
        "present_uncertain_pct": pct(present.get("uncertain", 0), n_present_comp),
        "present_incorrect_pct": pct(present.get("potentially_incorrect", 0), n_present_comp, 2),
        "n_missing": s["suggestions"]["suggested_additions"], "n_incorrect": s["suggestions"]["suggested_removals"],
        "records_with_missing": int((rr["n_potentially_missing"].replace("", 0).astype(float) > 0).sum()),
        "records_with_incorrect": int((rr["n_potentially_incorrect"].replace("", 0).astype(float) > 0).sum()),
        "additions_by_label": s["suggestions"]["additions_by_label"],
        "removals_by_label": s["suggestions"]["removals_by_label"],
        "n_exact_labels": len(ex), "n_calibrated": int(ex["calibrated"].sum()),
        "uncalibrated": ex.loc[~ex["calibrated"], "label"].tolist(),
        "median_auroc": round(float(ex["auroc_raw_vs_label"].median()), 3),
        "n_auroc_095": int((ex["auroc_raw_vs_label"] >= 0.95).sum()),
        "ece_raw_median": round(float(cal["ece_raw"].median()), 3), "ece_raw_max": round(float(cal["ece_raw"].max()), 3),
        "ece_cal_median": round(float(cal["ece_crossfit"].median()), 4), "ece_cal_max": round(float(cal["ece_crossfit"].max()), 3),
        "per_label": per_label,
        "strat_validated": strat["validated_by_human"], "strat_autogen": strat["initial_autogenerated_report"],
        "device_rate_min": round(100 * min(dev_rates), 1), "device_rate_max": round(100 * max(dev_rates), 1),
        "n_devices_500": len(dev),
        "concordance": conc.drop(columns=["examples"]).to_dict("records"),
        "checkpoint_sha256": manifest["checkpoint_sha256"], "torch": manifest["torch"],
        "python": manifest["python"], "mapping": manifest["label_mapping"],
    }
    # Counterfactual: the earlier rule "existing label potentially incorrect if posterior <= 0.05
    # (raw score <= 0.02 if uncalibrated)" evaluated on the same cached outputs.
    ass = pd.read_csv(res / "label_assessments.csv.gz", dtype={"record_id": str},
                      keep_default_na=False)
    pres = ass[ass["label_present"].astype(str).eq("True")
               & ass["status"].isin(["consistent", "uncertain", "potentially_incorrect"])].copy()
    calp = pd.to_numeric(pres["calibrated_probability"], errors="coerce")
    raw = pd.to_numeric(pres["raw_score"], errors="coerce")
    old = np.where(calp.notna(), calp <= 0.05, raw <= 0.02)
    data["posterior_rule_incorrect"] = int(old.sum())
    data["posterior_rule_ami"] = int((old & pres["label"].eq("AMI").to_numpy()).sum())
    data["n_ami"] = int((pres["label"] == "AMI").sum())

    disc = pd.read_csv(res / "label_discrepancies.csv", dtype={"record_id": str},
                       keep_default_na=False).set_index("record_id")
    reports = {}
    if a.ptbxl:
        db = pd.read_csv(Path(a.ptbxl) / "ptbxl_database.csv")
        reports = dict(zip(db.ecg_id.astype(str), db.report.fillna("")))
    data["examples"] = [{"record_id": r, "labels": disc.loc[r, "original_labels"],
                         "flags": disc.loc[r, "flag_details"],
                         "report": reports.get(r, "")}
                        for r in EXAMPLES if r in disc.index]
    # Follow-up analyses (tools/analysis/*.py)
    an = res / "analyses"
    if (an / "baselines.csv").exists():
        bl = pd.read_csv(an / "baselines.csv")
        data["baselines"] = json.loads(bl.to_json(orient="records"))
        data["overlap"] = json.loads((an / "baselines_overlap.json").read_text())
        sens = pd.read_csv(an / "sensitivity_flags.csv")
        data["sensitivity"] = json.loads(sens.to_json(orient="records"))
        data["sensitivity_support"] = json.loads(pd.read_csv(an / "sensitivity_support.csv").to_json(orient="records"))
    adj = Path(a.reports).parent / "docs" / "review" / "adjudication" / "protocol.md"
    if adj.exists():
        import re as _re
        t = adj.read_text()
        m = _re.search(r"(\d+) cases and (\d+) case-finding pairs", t)
        data["adjudication"] = {"cases": int(m.group(1)), "pairs": int(m.group(2))}
    (out / "manuscript_data.json").write_text(json.dumps(data, indent=1))

    # Figure 3: report-text concordance (flagged vs unflagged control), missing flags only
    c = conc[(conc["flag"] == "potentially_missing") & (conc["n_flagged"] >= 10)].copy()
    c = c.sort_values("flagged_rate")
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    y = np.arange(len(c))
    ax.barh(y + 0.19, 100 * c["flagged_rate"], height=0.36, color=BLUE, label="flagged as potentially missing")
    ax.barh(y - 0.19, 100 * c["control_rate"], height=0.36, color=GRAY, label="unflagged records without the label")
    for yi, (fr, n, k) in enumerate(zip(c["flagged_rate"], c["n_flagged"], c["flagged_report_supports_flag"])):
        ax.text(100 * fr + 1, yi + 0.19, f"{k}/{n}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, [f"{lab}" for lab in c["label"]])
    ax.set_xlabel("records whose free-text report names the finding (%)", fontsize=8, color=INK2)
    ax.set_xlim(0, 100)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_concordance.png", dpi=300)
    plt.close(fig)
    print(f"wrote {out / 'manuscript_data.json'} and fig_concordance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
