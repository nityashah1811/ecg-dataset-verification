#!/usr/bin/env python
"""Compare the pipeline's flags with alternative label-audit methods on PTB-XL.

Methods (all restricted to the same exact-mapped labels; "missing" flags of every method use the
same equivalence/conflict filter as the pipeline, so they are comparable):

  ours        ECGFounder, cross-fitted calibration, Bayes-factor rule (pipeline default)
  raw         ECGFounder raw sigmoid scores, no calibration (incorrect <= 0.02, missing >= 0.98)
  posterior   calibrated posterior rule (incorrect P <= 0.05, missing P >= 0.95)
  cl_probe    in-domain confident learning: cross-fitted logistic regression on ECGFounder
              embeddings trained on the PTB-XL labels themselves, issues found with cleanlab
  gesl        disagreement with GE Marquette 12SL machine statements (PTB-XL+)
  text        report/code mismatch found by keywords (8 labels only)

There is no ground truth. Each method is scored by two plausibility signals that are independent
of ECGFounder: (a) the free-text report (not used for 'text'), (b) the 12SL statements (not used
for 'gesl'; 12SL is not independent of ECGFounder's training labels, which were 12SL-assisted).
Ranked methods are also compared at a matched flag budget (the number of flags of 'ours').

    python tools/analysis/baselines.py --dataset /data/ptb-xl --ptbxl-plus /data/ptb-xl-plus \
        --results results/ptbxl --embeddings results/ptbxl_embeddings.npz \
        --out reports/ptbxl_ecgfounder/analyses
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import flags_from_rule, load_run, report_mentions

# PTB-XL label -> SNOMED concept in PTB-XL+ (chosen by exact co-occurrence with the SCP code;
# SR uses the broader "ECG: sinus rhythm" concept to match the pipeline's any-of definition).
SNOMED = {
    "SR": 4145513, "SBRAD": 4138456, "STACH": 4007310, "SARRH": 4141820, "AFIB": 313217,
    "AFLT": 314665, "SVTAC": 4275423, "PACE": 4304202, "BIGU": 4008580, "1AVB": 314379,
    "3AVB": 320744, "CRBBB": 4088337, "IRBBB": 4088338, "CLBBB": 4267892, "ILBBB": 4088336,
    "LAFB": 4295336, "LPFB": 4268046, "IVCD": 44784220, "WPW": 4253363, "PVC": 4145998,
    "PAC": 4109365, "LVH": 4102533, "VCLVH": 4102533, "RVH": 4231591, "LAO/LAE": 43022066,
    "RAO/RAE": 43021828, "LVOLT": 4088499, "IMI": 4247796, "ASMI": 4064610, "AMI": 4180609,
    "ALMI": 4184348, "LMI": 4064350, "PMI": 4253065, "LNGQT": 4008859, "NORM": 4065279,
}


def gesl_matrix(run, ptbxl_plus: str) -> tuple[np.ndarray, np.ndarray]:
    st = pd.read_csv(Path(ptbxl_plus) / "labels" / "12sl_statements.csv")
    st["record_id"] = st.ecg_id.astype(str)
    st = st.set_index("record_id").reindex(run.record_ids)
    have = st["statements_ext_snomed"].fillna("[]").apply(lambda s: {a for a, _ in ast.literal_eval(s)})
    G = np.zeros_like(run.presence)
    avail = np.array([lab in SNOMED for lab in run.labels])
    for j, lab in enumerate(run.labels):
        if lab in SNOMED:
            G[:, j] = have.apply(lambda s, c=SNOMED[lab]: c in s).to_numpy()
    return G, avail


def probe_oof(run, emb_path: str) -> np.ndarray:
    z = np.load(emb_path)
    idx = {r: i for i, r in enumerate(z["record_ids"])}
    X = z["features"][[idx[r] for r in run.record_ids]]
    P = np.full(run.presence.shape, np.nan)
    for k in np.unique(run.folds):
        tr, te = run.folds != k, run.folds == k
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for j in range(len(run.labels)):
            y = run.presence[tr, j]
            if y.sum() < 5:
                continue
            lr = LogisticRegression(C=0.1, max_iter=2000)
            lr.fit(Xtr, y)
            P[te, j] = lr.predict_proba(Xte)[:, 1]
    return P


def cleanlab_flags(run, P):
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    inc = np.zeros_like(run.presence)
    mis = np.zeros_like(run.presence)
    quality = np.ones(run.presence.shape)
    for j in range(len(run.labels)):
        p = P[:, j]
        if np.isnan(p).any():
            continue
        y = run.presence[:, j].astype(int)
        pp = np.stack([1 - p, p], 1)
        issues = find_label_issues(labels=y, pred_probs=pp)
        inc[:, j] = issues & (y == 1)
        mis[:, j] = issues & (y == 0)
        quality[:, j] = get_label_quality_scores(y, pp)
    mis &= ~run.covered & ~run.conflict
    return inc, mis, 1 - quality


def support(flags, pool, signal):
    """(supported, flagged, control_supported, control_n) for flags within pool."""
    f, c = pool & flags, pool & ~flags
    return int((signal & f).sum()), int(f.sum()), int((signal & c).sum()), int(c.sum())


def mh_ci(tables):
    """Mantel-Haenszel OR with Robins-Breslow-Greenland 95% CI and CMH test p (statsmodels)."""
    from statsmodels.stats.contingency_tables import StratifiedTable
    tabs = [np.array([[s, f - s], [cs, cn - cs]], float) for s, f, cs, cn in tables if f and cn]
    tabs = [t for t in tabs if (t[0].sum() > 0 and t[1].sum() > 0 and t[:, 0].sum() > 0 and t[:, 1].sum() > 0)]
    if not tabs:
        return None
    st = StratifiedTable([t + 0.5 if (t == 0).any() else t for t in tabs])
    lo, hi = st.oddsratio_pooled_confint()
    return round(st.oddsratio_pooled, 1), f"{lo:.1f}-{hi:.1f}", float(f"{st.test_null_odds().pvalue:.2g}")


def evaluate(name, run, inc, mis, mentions, G, avail, use_text=True, use_gesl=True):
    row = {"method": name, "n_incorrect": int(inc.sum()), "n_missing": int(mis.sum()),
           "n_flagged_records": int((inc.any(1) | mis.any(1)).sum())}
    for kind, F in (("missing", mis), ("incorrect", inc)):
        if use_text:
            agg = np.zeros(4, int)
            tabs = []
            for lab, men in mentions.items():
                j = run.labels.index(lab)
                pool = ~run.presence[:, j] if kind == "missing" else run.presence[:, j]
                sig = men if kind == "missing" else ~men
                t = support(F[:, j], pool, sig)
                tabs.append(t)
                agg += t
            # Mantel-Haenszel OR (0.5 added to strata with an empty cell), RBG 95% CI, CMH p
            ci = mh_ci(tabs)
            (row[f"{kind}_text_mh_or"], row[f"{kind}_text_mh_ci"],
             row[f"{kind}_text_cmh_p"]) = ci if ci else (None, None, None)
            row[f"{kind}_text_support"] = f"{agg[0]}/{agg[1]}"
            row[f"{kind}_text_rate"] = round(agg[0] / agg[1], 3) if agg[1] else None
            row[f"{kind}_text_control"] = round(agg[2] / agg[3], 4) if agg[3] else None
            row[f"{kind}_text_p"] = (fisher_exact([[agg[0], agg[1] - agg[0]], [agg[2], agg[3] - agg[2]]])[1]
                                     if agg[1] else None)
        if use_gesl:
            agg = np.zeros(4, int)
            tabs = []
            for j in np.where(avail)[0]:
                pool = ~run.presence[:, j] if kind == "missing" else run.presence[:, j]
                sig = G[:, j] if kind == "missing" else ~G[:, j]
                t = support(F[:, j], pool, sig)
                tabs.append(t)
                agg += t
            # Mantel-Haenszel OR (0.5 added to strata with an empty cell), RBG 95% CI, CMH p
            ci = mh_ci(tabs)
            (row[f"{kind}_12sl_mh_or"], row[f"{kind}_12sl_mh_ci"],
             row[f"{kind}_12sl_cmh_p"]) = ci if ci else (None, None, None)
            row[f"{kind}_12sl_support"] = f"{agg[0]}/{agg[1]}"
            row[f"{kind}_12sl_rate"] = round(agg[0] / agg[1], 3) if agg[1] else None
            row[f"{kind}_12sl_control"] = round(agg[2] / agg[3], 4) if agg[3] else None
    return row


def top_k(score, pool, k):
    s = np.where(pool, score, -np.inf).ravel()
    keep = np.zeros(s.shape, bool)
    order = np.argsort(-s, kind="stable")[:k]
    keep[order[np.isfinite(s[order])]] = True
    return keep.reshape(score.shape)


def jaccard(a, b):
    u = (a | b).sum()
    return round(float((a & b).sum() / u), 3) if u else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ptbxl-plus", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    run = load_run(a.dataset, a.results)
    mentions = report_mentions(run)
    G, avail = gesl_matrix(run, a.ptbxl_plus)
    eligible_missing = ~run.presence & ~run.covered & ~run.conflict

    methods = {}
    methods["ours"] = flags_from_rule(run)
    cal = run.calibrated[None, :]
    raw_inc = run.presence & (run.scores <= 0.02)
    raw_mis = eligible_missing & (run.scores >= 0.98)
    methods["raw"] = (raw_inc, raw_mis)
    post_inc = run.presence & np.where(cal, run.posterior <= 0.05, run.scores <= 0.02)
    methods["posterior"] = (post_inc, methods["ours"][1])
    P = probe_oof(run, a.embeddings)
    cl_inc, cl_mis, cl_score = cleanlab_flags(run, P)
    methods["cl_probe"] = (cl_inc, cl_mis)
    methods["gesl"] = (run.presence & ~G & avail[None, :], eligible_missing & G & avail[None, :])
    T = np.zeros_like(run.presence)
    Tm = np.zeros_like(run.presence)
    for lab, men in mentions.items():
        j = run.labels.index(lab)
        T[:, j] = run.presence[:, j] & ~men
        Tm[:, j] = eligible_missing[:, j] & men
    methods["text"] = (T, Tm)

    rows = [evaluate(n, run, i, m, mentions, G, avail, use_text=n != "text", use_gesl=n != "gesl")
            for n, (i, m) in methods.items()]
    res = pd.DataFrame(rows)

    # Matched-budget comparison of the ranked methods (budget = flags of 'ours' per kind)
    oi, om = methods["ours"]
    ev = np.where(cal, 1 / (1 + run.bf), 1 - run.scores)     # strength against an existing label
    scores = {
        "ours": (ev, run.posterior),
        "raw": (1 - run.scores, run.scores),
        "posterior": (1 - run.posterior, run.posterior),
        "cl_probe": (cl_score, cl_score),
    }
    matched = []
    for n, (s_inc, s_mis) in scores.items():
        ki = top_k(s_inc, run.presence & np.isfinite(s_inc), int(oi.sum()))
        km = top_k(s_mis, eligible_missing & np.isfinite(s_mis), int(om.sum()))
        r = evaluate(f"{n}@budget", run, ki, km, mentions, G, avail)
        r["jaccard_incorrect_vs_ours"] = jaccard(ki, oi)
        r["jaccard_missing_vs_ours"] = jaccard(km, om)
        matched.append(r)
    res = pd.concat([res, pd.DataFrame(matched)], ignore_index=True)
    res.to_csv(out / "baselines.csv", index=False)

    # Overlap of 'ours' with text mining and 12SL, and how much of 'ours' text could not see
    kw = np.zeros(len(run.labels), bool)
    kw[[run.labels.index(l) for l in mentions]] = True
    overlap = {
        "ours_flags_total": int(oi.sum() + om.sum()),
        "ours_flags_on_keyword_labels": int(oi[:, kw].sum() + om[:, kw].sum()),
        "ours_flags_on_labels_without_keyword_rule": int(oi[:, ~kw].sum() + om[:, ~kw].sum()),
        "ours_missing_also_text": int((om & Tm).sum()), "text_missing_total": int(Tm.sum()),
        "ours_missing_also_12sl": int((om & G).sum()),
        "ours_incorrect_also_12sl_absent": int((oi & ~G & avail[None, :]).sum()),
        "jaccard_missing": {n: jaccard(m, om) for n, (_, m) in methods.items()},
        "jaccard_incorrect": {n: jaccard(i, oi) for n, (i, _) in methods.items()},
        "probe_auroc_median": float(np.nanmedian([
            __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(run.presence[:, j], P[:, j])
            for j in range(len(run.labels)) if not np.isnan(P[:, j]).any() and run.presence[:, j].any()])),
    }
    (out / "baselines_overlap.json").write_text(json.dumps(overlap, indent=1))
    # Per-record method membership (used to build the adjudication sample)
    recs = []
    for n, (i, m) in methods.items():
        for kind, F in (("incorrect", i), ("missing", m)):
            r, c = np.where(F)
            recs += [{"record_id": run.record_ids[x], "label": run.labels[y], "flag": kind, "method": n}
                     for x, y in zip(r, c)]
    pd.DataFrame(recs).to_csv(out / "flags_by_method.csv.gz", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 40):
        cols = ["method", "missing_text_mh_or", "missing_text_mh_ci", "missing_12sl_mh_or", "missing_12sl_mh_ci",
                "incorrect_text_support", "incorrect_text_mh_or", "incorrect_text_mh_ci", "incorrect_text_cmh_p",
                "incorrect_12sl_mh_or", "incorrect_12sl_mh_ci"]
        print(res[[c for c in cols if c in res]].to_string(index=False))
    print(json.dumps(overlap, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
