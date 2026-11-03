#!/usr/bin/env python
"""Build an Excel sheet for cardiologist review of mappings/ptbxl_scp.yaml.

    python tools/analysis/mapping_review_sheet.py --dataset /data/ptb-xl \
        --official-label-csv /path/to/ECGFounder/csv/ptbxl_label.csv \
        --out docs/review/ptbxl_mapping_review.xlsx
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ecg_verify.ecgfounder import CLASS_NAMES  # noqa: E402
from ecg_verify.mapping import load_mapping, resolve_mapping_path  # noqa: E402

FONT = "Arial"
HEAD = PatternFill("solid", fgColor="1F3864")
INPUT = PatternFill("solid", fgColor="FFF2CC")
EXAMPLE = PatternFill("solid", fgColor="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")


def yaml_comments(path: Path) -> dict[str, str]:
    """Inline '# ...' comments on quality lines, keyed by label (human rationale)."""
    out, label = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"^  ([A-Za-z0-9_/()\"]+):", line)
        if m:
            label = m.group(1).strip('"')
        if label and "quality:" in line and "#" in line:
            out[label] = line.split("#", 1)[1].strip()
    return out


def official_mapping(csv_path: Path, codes: pd.Series) -> dict[str, str]:
    df = pd.read_csv(csv_path).dropna(subset=["label"])
    Y = np.array([json.loads(s) for s in df["label"]])
    sc = df["scp_codes"].apply(lambda s: set(ast.literal_eval(s)))
    out = {}
    for code in codes:
        has = sc.apply(lambda s, c=code: c in s).to_numpy()
        if not has.any():
            continue
        frac = Y[has].mean(0)
        base = Y[~has].mean(0)
        # classes assigned to >=90% of records with the code and to <50% of the others
        cls = [CLASS_NAMES[i] for i in np.where((frac >= 0.9) & (base < 0.5))[0]]
        out[code] = "; ".join(cls) if cls else "(none)"
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--official-label-csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    mapping = load_mapping("ptbxl_scp")
    comments = yaml_comments(resolve_mapping_path("ptbxl_scp"))
    scp = pd.read_csv(Path(a.dataset) / "scp_statements.csv", index_col=0)
    inv = pd.read_csv(ROOT / "reports" / "ptbxl_preflight" / "label_inventory.csv").set_index("label")
    official = (official_mapping(Path(a.official_label_csv), pd.Series(list(mapping.labels)))
                if a.official_label_csv else {})

    rows = []
    for lab, m in mapping.labels.items():
        kind = ("rhythm" if lab in scp.index and scp.loc[lab, "rhythm"] == 1 else
                "diagnostic" if lab in scp.index and scp.loc[lab, "diagnostic"] == 1 else
                "form" if lab in scp.index else "heart axis")
        desc = scp.loc[lab, "description"] if lab in scp.index else m.description
        status = m.quality if m.supported else "unsupported"
        rationale = m.unsupported_reason or comments.get(lab, "")
        extra = []
        if m.covered_by:
            extra.append("not flagged missing if present: " + ", ".join(m.covered_by))
        if m.exclusive_with:
            extra.append("additions suppressed if any other abnormal label present")
        rows.append({
            "code": lab, "type": kind, "description": desc,
            "records": int(inv.loc[lab, "count"]) if lab in inv.index else 0,
            "status": status, "classes": "; ".join(m.model_classes) or "—",
            "rules": "; ".join(extra), "rationale": rationale,
            "official": official.get(lab, "n/a"),
        })
    rows.sort(key=lambda r: (["exact", "approximate", "unsupported"].index(r["status"]), -r["records"]))

    wb = Workbook()
    ws = wb.active
    ws.title = "Mapping review"
    cols = [("PTB-XL code", 11), ("Type", 11), ("Description (PTB-XL)", 34), ("Records", 9),
            ("Our decision", 13), ("ECGFounder class(es) — any of", 52), ("Extra rules", 30),
            ("Our rationale", 36), ("Official ECGFounder label file maps to", 40),
            ("REVIEW: verdict", 15), ("REVIEW: proposed change", 34), ("REVIEW: comment", 34)]
    ws.append([c for c, _ in cols])
    for i, (_, w) in enumerate(cols, 1):
        ws.column_dimensions[ws.cell(1, i).column_letter].width = w
        c = ws.cell(1, i)
        c.font = Font(name=FONT, bold=True, color="FFFFFF")
        c.fill = HEAD
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[1].height = 32

    # Example row (how a reviewer fills it in)
    ex = ["EXAMPLE", "diagnostic", "(example only — delete or ignore)", None, "exact",
          "LEFT ANTERIOR FASCICULAR BLOCK", "", "", "", "Disagree",
          "Add RBBB AND LEFT ANTERIOR FASCICULAR BLOCK", "Bifascicular block implies LAFB"]
    ws.append(ex)
    for i in range(1, len(cols) + 1):
        ws.cell(2, i).fill = EXAMPLE
        ws.cell(2, i).font = Font(name=FONT, italic=True, color="375623")

    for r in rows:
        ws.append([r["code"], r["type"], r["description"], r["records"], r["status"], r["classes"],
                   r["rules"], r["rationale"], r["official"], None, None, None])
    last = ws.max_row
    for row in ws.iter_rows(min_row=3, max_row=last):
        for c in row:
            c.font = Font(name=FONT, size=10)
            c.alignment = Alignment(wrap_text=True, vertical="top")
            c.border = Border(bottom=THIN)
        for c in row[9:12]:
            c.fill = INPUT
    dv = DataValidation(type="list", formula1='"Agree,Disagree,Unsure"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"J3:J{last}")
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:L{last}"

    # Summary sheet (formulas)
    s = wb.create_sheet("Summary")
    s.column_dimensions["A"].width = 44
    s.column_dimensions["B"].width = 14
    s["A1"] = "PTB-XL → ECGFounder mapping review"
    s["A1"].font = Font(name=FONT, bold=True, size=14)
    s["A3"] = "How to review"
    s["A3"].font = Font(name=FONT, bold=True)
    notes = [
        "Fill in only the yellow columns J–L of 'Mapping review' (row 2 is an example).",
        "Verdict = Agree / Disagree / Unsure with OUR decision (column E) and classes (column F).",
        "exact = same clinical concept; used to flag labels. approximate = related but criteria differ;",
        "    scored but never used for flags. unsupported = no ECGFounder class can verify it.",
        "ECGFounder classes are GE MUSE / 12SL-style statements (training data: Harvard-Emory ECG DB).",
        "Column I shows how the official ECGFounder PTB-XL file maps the code, for comparison.",
    ]
    for i, n in enumerate(notes, 4):
        s.cell(i, 1, n).font = Font(name=FONT, size=10)
    s["A11"] = "Progress"
    s["A11"].font = Font(name=FONT, bold=True)
    rng = f"'Mapping review'!$J$3:$J${last}"
    stat = f"'Mapping review'!$E$3:$E${last}"
    lines = [("Codes to review", f"=COUNTA('Mapping review'!$A$3:$A${last})"),
             ("Reviewed", f'=COUNTIF({rng},"Agree")+COUNTIF({rng},"Disagree")+COUNTIF({rng},"Unsure")'),
             ("Agree", f'=COUNTIF({rng},"Agree")'), ("Disagree", f'=COUNTIF({rng},"Disagree")'),
             ("Unsure", f'=COUNTIF({rng},"Unsure")'),
             ("Disagreements on exact mappings (affect flags)",
              f'=COUNTIFS({rng},"Disagree",{stat},"exact")'),
             ("Share reviewed", "=IF(B12=0,0,B13/B12)")]
    for i, (k, f) in enumerate(lines, 12):
        s.cell(i, 1, k).font = Font(name=FONT, size=10)
        s.cell(i, 2, f).font = Font(name=FONT, size=10)
    s["B18"].number_format = "0.0%"
    s["A20"] = ("Source: mappings/ptbxl_scp.yaml (this repository); record counts: "
                "reports/ptbxl_preflight/label_inventory.csv (PTB-XL v1.0.3).")
    s["A20"].font = Font(name=FONT, size=9, italic=True)
    wb.move_sheet("Summary", offset=-1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)
    print("wrote", a.out, "rows", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
