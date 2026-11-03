#!/usr/bin/env python
"""Analyse completed blinded adjudication sheets (pre-specified in docs/review/adjudication/protocol.md).

    python tools/analysis/analyze_adjudication.py --key docs/review/adjudication_KEY_do_not_send/UNBLINDING_KEY_do_not_send.csv \
        --reader1 ratings_reader1.xlsx --reader2 ratings_reader2.xlsx [--third ratings_reader3.xlsx] \
        --out reports/ptbxl_ecgfounder/adjudication

Reference standard per (case, finding): agreement of readers 1 and 2. A Present-vs-Absent
disagreement is resolved by reader 3 if given, else left 'Unresolved'; Uncertain from either
reader -> 'Uncertain'. Precision is reported conservatively (Uncertain/Unresolved count as not
confirming the flag) and with those excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import cohen_kappa_score


def read_sheet(path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Ratings", header=1)
    df = df[df["Case"].astype(str).str.match(r"^C\d{3}$")]
    return df.rename(columns={"Case": "case_id", "finding_code": "label", "Rating": "rating",
                              "Signal quality (case)": "quality"})[["case_id", "label", "rating", "quality"]]


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(p, 3), round(c - h, 3), round(c + h, 3))


def reference(r1, r2, r3=None):
    out = []
    for a, b, c in zip(r1, r2, r3 if r3 is not None else [None] * len(r1)):
        if pd.isna(a) or pd.isna(b):
            out.append("Missing")
        elif a == b:
            out.append(a)
        elif "Uncertain" in (a, b):
            out.append("Uncertain")
        elif c in ("Present", "Absent"):
            out.append(c)
        else:
            out.append("Unresolved")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--reader1", required=True)
    ap.add_argument("--reader2", required=True)
    ap.add_argument("--third")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    key = pd.read_csv(a.key, dtype={"record_id": str})
    df = key.merge(read_sheet(a.reader1).rename(columns={"rating": "r1", "quality": "q1"}),
                   on=["case_id", "label"], how="left")
    df = df.merge(read_sheet(a.reader2).rename(columns={"rating": "r2", "quality": "q2"}),
                  on=["case_id", "label"], how="left")
    r3 = None
    if a.third:
        df = df.merge(read_sheet(a.third).rename(columns={"rating": "r3"})[["case_id", "label", "r3"]],
                      on=["case_id", "label"], how="left")
        r3 = df["r3"]
    df["ref"] = reference(df["r1"], df["r2"], r3)
    df.to_csv(out / "adjudication_merged.csv", index=False)

    both = df.dropna(subset=["r1", "r2"])
    summary = {
        "pairs_rated": int(len(both)), "pairs_total": int(len(df)),
        "kappa_all": round(float(cohen_kappa_score(both.r1, both.r2)), 3) if len(both) else None,
        "reference_counts": df["ref"].value_counts().to_dict(),
        "decoy_present_rate": wilson(int(((df.decoy) & (df.ref == "Present")).sum()),
                                     int((df.decoy & df.ref.isin(["Present", "Absent"])).sum())),
    }

    # Flag precision per method x flag kind
    rows = []
    long = df[df.flagged].assign(method=lambda d: d.flag_methods.str.split(";")).explode("method")
    long = long.assign(kind=long.flag_kind.str.split(";")).explode("kind")
    for (m, k), g in long.groupby(["method", "kind"]):
        target = "Absent" if k == "incorrect" else "Present"
        hit = int((g.ref == target).sum())
        decided = int(g.ref.isin(["Present", "Absent"]).sum())
        rows.append({"method": m, "flag": k, "n": len(g),
                     "ppv_conservative": wilson(hit, len(g)), "ppv_decided_only": wilson(hit, decided)})
    pd.DataFrame(rows).to_csv(out / "flag_precision.csv", index=False)

    # Stratum-weighted label error rate among coded findings and missed findings
    rng = np.random.default_rng(0)
    cases = df.groupby("case_id").agg(stratum=("stratum", "first"), size=("stratum_size", "first"))
    w = cases["size"] / cases.groupby("stratum")["size"].transform("count")
    df["w"] = df.case_id.map(w)

    def err_rates(d):
        coded = d[d.coded_in_ptbxl & d.ref.isin(["Present", "Absent"])]
        unc = d[~d.coded_in_ptbxl & ~d.decoy & d.ref.isin(["Present", "Absent"])]
        wrong = np.average(coded.ref == "Absent", weights=coded.w) if len(coded) else np.nan
        missed = np.average(unc.ref == "Present", weights=unc.w) if len(unc) else np.nan
        return wrong, missed

    est = err_rates(df)
    boots = []
    case_ids = df.case_id.unique()
    for _ in range(a.n_boot):
        pick = []
        for s, g in cases.groupby("stratum"):
            pick += list(rng.choice(g.index, len(g), replace=True))
        bd = pd.concat([df[df.case_id == c] for c in pick])
        boots.append(err_rates(bd))
    boots = np.array(boots)
    summary["coded_label_error_rate_weighted"] = {
        "estimate": round(float(est[0]), 4),
        "ci95": [round(float(np.nanpercentile(boots[:, 0], q)), 4) for q in (2.5, 97.5)]}
    summary["flagged_uncoded_finding_present_rate_weighted"] = {
        "estimate": round(float(est[1]), 4),
        "ci95": [round(float(np.nanpercentile(boots[:, 1], q)), 4) for q in (2.5, 97.5)]}
    summary["note"] = ("Error rates are weighted by inverse sampling fraction within strata and refer to "
                       "the audited (exact-mapped) findings only.")
    (out / "adjudication_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
