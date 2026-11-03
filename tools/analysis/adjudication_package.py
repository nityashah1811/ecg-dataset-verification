#!/usr/bin/env python
"""Build a blinded cardiologist adjudication package for the PTB-XL label audit.

Design (pre-specified, seeded):
  stratum A  records with a 'potentially incorrect' flag from our method      (up to N_A)
  stratum B  records with a 'potentially missing' flag from our method        (N_B, by severity)
  stratum C  records flagged only by other methods (not ours)                (N_C)
  stratum D  random records with no flag from any method (controls)           (N_D)

For every sampled record the reader rates a shuffled list of findings: all comparable labels
coded in PTB-XL, every label flagged by any method, and two random decoy labels. Readers see
only the ECG and the finding names: never the coded labels, the report, the model output or the
stratum. Two cardiologists rate independently; disagreements are resolved by consensus or a
third reader.

Outputs (in --out):
  ecg_cases.pdf                one page per case (blinded case id, 12-lead ECG + rhythm strip)
  rating_sheet.xlsx            one row per (case, finding) with Present/Absent/Uncertain dropdown
  ../<out>_KEY_do_not_send/UNBLINDING_KEY_do_not_send.csv   case -> record, stratum, flags (separate folder)
  protocol.md                  reader instructions and analysis plan

    python tools/analysis/adjudication_package.py --dataset /data/ptb-xl \
        --results results/ptbxl --flags reports/ptbxl_ecgfounder/analyses/flags_by_method.csv.gz \
        --out docs/review/adjudication
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import wfdb  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from common import load_run  # noqa: E402

SEED = 20261031
N_A, N_B, N_C, N_D = 60, 100, 60, 80
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PLAIN = {  # reader-facing names (no PTB-XL codes)
    "SR": "Sinus rhythm (any rate)", "SBRAD": "Sinus bradycardia", "STACH": "Sinus tachycardia",
    "SARRH": "Sinus arrhythmia", "AFIB": "Atrial fibrillation", "AFLT": "Atrial flutter",
    "SVTAC": "Supraventricular tachycardia", "PACE": "Paced rhythm / pacemaker activity",
    "BIGU": "Bigeminy", "1AVB": "First-degree AV block", "3AVB": "Third-degree (complete) AV block",
    "CRBBB": "Complete right bundle branch block", "IRBBB": "Incomplete right bundle branch block",
    "CLBBB": "Complete left bundle branch block", "ILBBB": "Incomplete left bundle branch block",
    "LAFB": "Left anterior fascicular block", "LPFB": "Left posterior fascicular block",
    "IVCD": "Non-specific intraventricular conduction delay", "WPW": "Wolff-Parkinson-White pattern",
    "PVC": "Premature ventricular complex(es)", "PAC": "Premature atrial complex(es)",
    "LVH": "Left ventricular hypertrophy", "VCLVH": "Voltage criteria for LVH",
    "RVH": "Right ventricular hypertrophy", "LAO/LAE": "Left atrial enlargement / overload",
    "RAO/RAE": "Right atrial enlargement / overload", "LVOLT": "Low QRS voltage",
    "IMI": "Inferior myocardial infarction (old or acute)", "ASMI": "Anteroseptal myocardial infarction",
    "AMI": "Anterior myocardial infarction", "ALMI": "Anterolateral myocardial infarction",
    "LMI": "Lateral myocardial infarction", "IPMI": "Inferoposterior myocardial infarction",
    "PMI": "Posterior myocardial infarction", "LNGQT": "Prolonged QT interval",
    "NORM": "Normal ECG",
}


def sample(run, flags: pd.DataFrame, rng, results: str) -> pd.DataFrame:
    rr = pd.read_csv(Path(results) / "record_results.csv", dtype={"record_id": str},
                     keep_default_na=False).set_index("record_id")
    ours = flags[flags.method == "ours"]
    inc_ids = sorted(set(ours.loc[ours.flag == "incorrect", "record_id"]))
    mis_ids = sorted(set(ours.loc[ours.flag == "missing", "record_id"]) - set(inc_ids))
    base_ids = sorted(set(flags.loc[flags.method != "ours", "record_id"]) - set(ours.record_id))
    flagged_any = set(flags.record_id)
    ctrl_pool = sorted(set(run.record_ids) - flagged_any)
    # the four strata must partition the dataset for the weighted estimates to be valid
    assert len(inc_ids) + len(mis_ids) + len(base_ids) + len(ctrl_pool) == len(run.record_ids)

    A = list(rng.choice(inc_ids, min(N_A, len(inc_ids)), replace=False))
    # stratum B: stratified by severity (proportional, at least 10 per severity level)
    sev = rr.loc[mis_ids, "discrepancy_severity"]
    B = []
    for level in ("high", "medium", "low"):
        pool = sorted(sev.index[sev == level])
        k = max(10, round(N_B * len(pool) / len(sev)))
        B += list(rng.choice(pool, min(k, len(pool)), replace=False))
    C = list(rng.choice(base_ids, min(N_C, len(base_ids)), replace=False))
    D = list(rng.choice(ctrl_pool, N_D, replace=False))
    rows = [{"record_id": r, "stratum": s, "stratum_size": n}
            for s, ids, n in (("A_ours_incorrect", A, len(inc_ids)), ("B_ours_missing", B, len(mis_ids)),
                              ("C_baseline_only", C, len(base_ids)), ("D_unflagged", D, len(ctrl_pool)))
            for r in ids]
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["case_id"] = [f"C{i + 1:03d}" for i in range(len(df))]
    return df


def findings_for(run, rid, flags_rec, rng):
    i = int(np.where(run.record_ids == rid)[0][0])
    coded = {run.labels[j] for j in np.where(run.presence[i])[0]}
    flagged = set(flags_rec.get(rid, []))
    pool = sorted(set(run.labels) - coded - flagged)
    decoys = set(rng.choice(pool, 2, replace=False))
    items = sorted(coded | flagged | decoys)
    rng.shuffle(items)
    return [(lab, lab in coded, lab in flagged, lab in decoys) for lab in items]


def plot_case(pdf, dataset, src, case_id):
    rec = wfdb.rdrecord(str(Path(dataset) / src))
    x = np.nan_to_num(rec.p_signal)
    names = [n.upper() for n in rec.sig_name]
    fs = rec.fs
    order = [[0, 3, 6, 9], [1, 4, 7, 10], [2, 5, 8, 11]]
    fig = plt.figure(figsize=(11.69, 8.27))
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.86])
    seg = int(2.5 * fs)
    rows = 4
    for r in range(3):
        for c in range(4):
            k = order[r][c]
            lead = LEADS[k]
            idx = names.index(lead.upper())
            y = x[c * seg:(c + 1) * seg, idx]
            y = y - np.median(y)
            t = np.arange(len(y)) / fs + c * 2.5
            off = (rows - 1 - r) * 3.0
            ax.plot(t, y + off, color="black", linewidth=0.6)
            ax.text(c * 2.5 + 0.05, off + 1.1, lead, fontsize=9, weight="bold")
    ii = x[:, names.index("II")]
    ax.plot(np.arange(len(ii)) / fs, ii - np.median(ii), color="black", linewidth=0.6)
    ax.text(0.05, 1.1, "II (rhythm)", fontsize=9, weight="bold")
    # ECG paper: small boxes 0.04 s x 0.1 mV, large boxes 0.2 s x 0.5 mV
    for k, t in enumerate(np.arange(0, 10.001, 0.04)):
        ax.axvline(t, color="#f4b6b6", linewidth=0.45 if k % 5 == 0 else 0.12, zorder=0)
    for k, v in enumerate(np.arange(-1.5, 11.501, 0.1)):
        ax.axhline(v, color="#f4b6b6", linewidth=0.45 if k % 5 == 0 else 0.12, zorder=0)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.5, 11.5)
    ax.set_aspect(0.4)  # 25 mm/s and 10 mm/mV: 0.2 s box = 0.5 mV box
    ax.axis("off")
    fig.text(0.03, 0.95, f"Case {case_id}", fontsize=16, weight="bold")
    fig.text(0.97, 0.95, "25 mm/s · 10 mm/mV · 500 Hz · baseline-centred, unfiltered",
             fontsize=8, ha="right", color="#555555")
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    run = load_run(a.dataset, a.results)
    flags = pd.read_csv(a.flags, dtype={"record_id": str})
    cases = sample(run, flags, rng, a.results)
    flags_rec = flags.groupby("record_id")["label"].apply(lambda s: sorted(set(s))).to_dict()
    by_method = flags.groupby(["record_id", "label"])["method"].apply(lambda s: ";".join(sorted(set(s)))).to_dict()
    by_kind = flags.groupby(["record_id", "label"])["flag"].apply(lambda s: ";".join(sorted(set(s)))).to_dict()

    key, sheet = [], []
    for c in cases.itertuples():
        for lab, coded, flagged, decoy in findings_for(run, c.record_id, flags_rec, rng):
            key.append({"case_id": c.case_id, "record_id": c.record_id, "stratum": c.stratum,
                        "stratum_size": c.stratum_size, "label": lab, "coded_in_ptbxl": coded,
                        "flagged": flagged, "flag_kind": by_kind.get((c.record_id, lab), ""),
                        "flag_methods": by_method.get((c.record_id, lab), ""), "decoy": decoy})
            sheet.append((c.case_id, PLAIN.get(lab, lab), lab))
    keydir = out.parent / (out.name + "_KEY_do_not_send")  # kept outside the folder sent to readers
    keydir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(key).to_csv(keydir / "UNBLINDING_KEY_do_not_send.csv", index=False)

    # Rating sheet
    wb = Workbook()
    ins = wb.active
    ins.title = "Instructions"
    text = [
        "Blinded ECG review: PTB-XL label audit", "",
        "For each case, open the same case number in ecg_cases.pdf.",
        "Rate each listed finding as Present, Absent or Uncertain from the ECG alone.",
        "The list mixes findings that may or may not be present: do not infer anything from it.",
        "Do not consult any other source (reports, labels, colleagues) during the first reading.",
        "Use 'Signal quality' if the tracing does not allow a judgement for that case.",
        "Fill in only the yellow columns. Row 2 of 'Ratings' is an example.",
        "", f"Cases: {cases.case_id.nunique()}   Findings to rate: {len(sheet)}",
        "Reader initials go in cell B1 of 'Ratings'.",
    ]
    for i, t in enumerate(text, 1):
        ins.cell(i, 1, t).font = Font(name="Arial", bold=i == 1, size=14 if i == 1 else 10)
    ins.column_dimensions["A"].width = 100
    ws = wb.create_sheet("Ratings")
    ws["A1"], ws["B1"] = "Reader initials:", None
    ws["B1"].fill = PatternFill("solid", fgColor="FFF2CC")
    hdr = ["Case", "Finding", "Rating", "Signal quality (case)", "Comment", "finding_code"]
    for j, h in enumerate(hdr, 1):
        cell = ws.cell(2, j, h)
        cell.font = Font(name="Arial", bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F3864")
    ex = ["EXAMPLE", "Atrial fibrillation", "Absent", "Adequate", "regular P waves", "AFIB"]
    for j, v in enumerate(ex, 1):
        cell = ws.cell(3, j, v)
        cell.font = Font(name="Arial", italic=True, color="375623")
        cell.fill = PatternFill("solid", fgColor="E2EFDA")
    for i, (case, plain, code) in enumerate(sorted(sheet, key=lambda r: r[0]), 4):
        ws.cell(i, 1, case)
        ws.cell(i, 2, plain)
        ws.cell(i, 6, code).font = Font(name="Arial", size=8, color="999999")
        for j in (3, 4, 5):
            ws.cell(i, j).fill = PatternFill("solid", fgColor="FFF2CC")
        for j in (1, 2):
            ws.cell(i, j).font = Font(name="Arial", size=10)
    last = ws.max_row
    dv1 = DataValidation(type="list", formula1='"Present,Absent,Uncertain"', allow_blank=True)
    dv2 = DataValidation(type="list", formula1='"Adequate,Poor but readable,Unreadable"', allow_blank=True)
    ws.add_data_validation(dv1)
    ws.add_data_validation(dv2)
    dv1.add(f"C4:C{last}")
    dv2.add(f"D4:D{last}")
    for col, w in zip("ABCDEF", (9, 46, 13, 22, 40, 12)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A4"
    ws.cell(1, 4, "Progress:").font = Font(name="Arial", bold=True)
    ws.cell(1, 5, f'=COUNTA(C4:C{last})&" of "&COUNTA(B4:B{last})&" rated"')
    wb.save(out / "rating_sheet.xlsx")

    with PdfPages(out / "ecg_cases.pdf") as pdf:
        src = {r.record_id: r.source for r in run.refs}
        for c in sorted(cases.itertuples(), key=lambda r: r.case_id):
            plot_case(pdf, a.dataset, src[c.record_id], c.case_id)

    (out / "protocol.md").write_text(PROTOCOL.format(
        n=len(cases), nf=len(sheet), **{f"n{k}": int((cases.stratum.str[0] == k).sum()) for k in "ABCD"},
        sizes=cases.groupby("stratum")["stratum_size"].first().to_dict(), seed=SEED))
    print(cases.stratum.value_counts().to_dict(), "findings", len(sheet))
    return 0


PROTOCOL = """# Blinded adjudication protocol: PTB-XL label audit

## Purpose
Estimate, against independent cardiologist reading, (1) the precision of "potentially
incorrect" and "potentially missing" flags from ECGFounder and from the comparison methods, and
(2) the label-error rate of PTB-XL for the audited diagnoses, with confidence intervals.

## Sample (seed {seed})
{n} cases and {nf} case-finding pairs, drawn from four strata. The stratum sizes in the
population are {sizes}.
- A: {nA} records with a potentially-incorrect flag from our method.
- B: {nB} records with a potentially-missing flag from our method, stratified by severity.
- C: {nC} records flagged only by other methods (confident learning, GE 12SL, raw scores,
  posterior rule, report-text mining), not by ours.
- D: {nD} random records flagged by no method.

Each case lists all PTB-XL-coded comparable findings, every finding flagged by any method, and
two random decoy findings, in random order.

## Reading
- Two cardiologists read independently. They see the ECG (ecg_cases.pdf) and the finding names
  only. They are blinded to the PTB-XL codes, the reports, the model outputs, the strata and each
  other.
- Each finding is rated Present, Absent or Uncertain. Signal quality is recorded per case.
- Disagreements (Present vs Absent) are resolved by consensus or by a third reader. That
  reference is final.
- Send only this folder to readers. The unblinding key is in the separate `adjudication_KEY_do_not_send/` folder.

## Analysis (pre-specified; tools/analysis/analyze_adjudication.py)
- Inter-reader agreement: Cohen's kappa per finding group.
- Flag precision (PPV) per method and flag type. A missing flag is correct if the reference says
  Present; an incorrect flag is correct if it says Absent. Uncertain counts as not correct
  (conservative) and as excluded (sensitivity analysis). Wilson 95% CIs.
- PTB-XL label-error rate for coded and uncoded findings: stratum-weighted (inverse sampling
  fraction) estimate with bootstrap 95% CI.
- Decoy findings give the false-positive rate of the readers' "Present".
"""

if __name__ == "__main__":
    raise SystemExit(main())
