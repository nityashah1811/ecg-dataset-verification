#!/usr/bin/env python
"""Threshold sensitivity analysis on cached ECGFounder outputs (no re-inference).

For a grid of decision thresholds it reports flag counts and the pooled report-text support of
the flags on keyword-checkable labels, relative to unflagged controls. Writes
<out>/sensitivity_flags.csv and <out>/sensitivity_support.csv.

    python tools/analysis/sensitivity.py --dataset /data/ptb-xl --results results/ptbxl \
        --out reports/ptbxl_ecgfounder/analyses
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from baselines import mh_ci
from common import flags_from_rule, load_run, report_support

CONTRADICT_BF = [0.2, 0.1, 0.05, 0.02, 0.01, 0.001]
MISSING_P = [0.8, 0.9, 0.95, 0.98, 0.99]
SUPPORT_BF = [2.0, 3.0, 10.0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    run = load_run(a.dataset, a.results)

    rows = []
    for c in CONTRADICT_BF:
        for m in MISSING_P:
            inc, mis = flags_from_rule(run, contradict_bf=c, missing_p=m)
            sup = report_support(run, inc, mis)
            row = {"contradict_bf": c, "missing_posterior": m,
                   "n_incorrect": int(inc.sum()), "n_missing": int(mis.sum()),
                   "n_flagged_records": int((inc.any(1) | mis.any(1)).sum()),
                   "default": c == 0.05 and m == 0.95}
            for kind in ("missing", "incorrect"):
                s, f, cs, cn = sup[kind]
                row[f"{kind}_keyword_flags"] = f
                row[f"{kind}_support_rate"] = round(s / f, 3) if f else None
                row[f"{kind}_control_rate"] = round(cs / cn, 4) if cn else None
                ci = mh_ci(sup["tables"][kind])  # label-adjusted (Mantel-Haenszel) odds ratio
                row[f"{kind}_mh_or"], row[f"{kind}_mh_ci"] = (ci[0], ci[1]) if ci else (None, None)
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sensitivity_flags.csv", index=False)

    # Share of existing labels judged consistent for different support thresholds
    cal = run.calibrated[None, :]
    srows = []
    for sbf in SUPPORT_BF:
        for sp in (0.5, None):
            ok = np.where(cal, (run.bf >= sbf) | ((run.posterior >= sp) if sp else False),
                          run.scores >= 0.5)
            n = run.presence.sum()
            srows.append({"support_bf": sbf, "support_posterior": sp if sp else "not used",
                          "consistent_pct": round(100 * (ok & run.presence).sum() / n, 1),
                          "default": sbf == 3.0 and sp == 0.5})
    pd.DataFrame(srows).to_csv(out / "sensitivity_support.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[["contradict_bf", "missing_posterior", "n_incorrect", "n_missing", "n_flagged_records",
                  "missing_mh_or", "missing_mh_ci", "incorrect_keyword_flags", "incorrect_mh_or",
                  "incorrect_mh_ci"]].to_string(index=False))
        print(pd.DataFrame(srows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
