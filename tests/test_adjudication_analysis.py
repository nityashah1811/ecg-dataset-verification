import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "analysis"))
import analyze_adjudication as aa  # noqa: E402


def test_reference_standard_rules():
    r1 = ["Present", "Present", "Absent", "Uncertain", "Present", None]
    r2 = ["Present", "Absent", "Absent", "Present", "Absent", "Present"]
    r3 = [None, "Absent", None, None, None, None]
    assert aa.reference(r1, r2, r3) == ["Present", "Absent", "Absent", "Uncertain", "Unresolved",
                                        "Missing"]


def test_wilson_interval():
    p, lo, hi = aa.wilson(79, 100)
    assert p == 0.79 and 0.69 < lo < 0.71 and 0.85 < hi < 0.87  # textbook: 0.700-0.858
    assert aa.wilson(0, 0) == (None, None, None)


def _sheet(path, rows):
    wb = Workbook()
    wb.active.title = "Instructions"
    ws = wb.create_sheet("Ratings")
    ws.append(["Reader initials:", "XX"])
    ws.append(["Case", "Finding", "Rating", "Signal quality (case)", "Comment", "finding_code"])
    ws.append(["EXAMPLE", "Atrial fibrillation", "Absent", "Adequate", "", "AFIB"])
    for case, code, rating in rows:
        ws.append([case, code, rating, "Adequate", "", code])
    wb.save(path)


def test_end_to_end_on_synthetic_ratings(tmp_path):
    """Code-path test only: synthetic ratings, not study data."""
    key = pd.DataFrame([
        # case, record, stratum, size, label, coded, flagged, kind, methods, decoy
        ("C001", "1", "A_ours_incorrect", 10, "AFIB", True, True, "incorrect", "ours", False),
        ("C001", "1", "A_ours_incorrect", 10, "SR", False, True, "missing", "ours;gesl", False),
        ("C001", "1", "A_ours_incorrect", 10, "WPW", False, False, "", "", True),
        ("C002", "2", "D_unflagged", 1000, "SR", True, False, "", "", False),
        ("C002", "2", "D_unflagged", 1000, "LVH", False, False, "", "", True),
    ], columns=["case_id", "record_id", "stratum", "stratum_size", "label", "coded_in_ptbxl",
                "flagged", "flag_kind", "flag_methods", "decoy"])
    key.to_csv(tmp_path / "key.csv", index=False)
    rows = [("C001", "AFIB", "Absent"), ("C001", "SR", "Present"), ("C001", "WPW", "Absent"),
            ("C002", "SR", "Present"), ("C002", "LVH", "Absent")]
    _sheet(tmp_path / "r1.xlsx", rows)
    _sheet(tmp_path / "r2.xlsx", rows)
    assert aa.main(["--key", str(tmp_path / "key.csv"), "--reader1", str(tmp_path / "r1.xlsx"),
                    "--reader2", str(tmp_path / "r2.xlsx"), "--out", str(tmp_path / "out"),
                    "--n-boot", "20"]) == 0
    prec = pd.read_csv(tmp_path / "out" / "flag_precision.csv")
    ours = prec[(prec.method == "ours")].set_index("flag")
    assert ours.loc["incorrect", "n"] == 1 and ours.loc["missing", "n"] == 1
    assert (tmp_path / "out" / "adjudication_summary.json").exists()
