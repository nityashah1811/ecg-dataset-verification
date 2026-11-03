#!/usr/bin/env python
"""Model-independent plausibility check of PTB-XL flags against the free-text reports.

PTB-XL ships the original free-text report (German, English or Swedish) next to the SCP codes.
The codes were derived from these reports, so the text is NOT an independent ground truth; but
it is independent of ECGFounder. For findings that can be detected reliably by keywords we ask:

* "potentially missing" flags: how often does the report mention the finding although the code
  is absent? Compared with the same rate among *unflagged* records lacking the code.
* "potentially incorrect" flags: how often does the report fail to mention the finding although
  the code is present? Compared with unflagged records carrying the code.

A flag rate far above its control rate means the flags concentrate on report/code
inconsistencies, i.e. on records where the labelling process demonstrably went wrong.

    python tools/ptbxl_report_concordance.py --dataset /data/ptb-xl --results results/ptbxl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

# Lower-cased regexes. Umlauts appear as upper-case letters in PTB-XL ("ventrikulÄre"), so the
# report is lower-cased first and patterns use "." for umlauts.
KEYWORDS = {
    "LAFB": r"linksanterior|left anterior (fascicular|hemi)|anterior(er)? hemiblock|v.nster fr.mre",
    "CLBBB": r"linksschenkelblock|left bundle branch block|v.nster grenblock|lsb\b",
    "CRBBB": r"(?<!partieller )(?<!inkompletter )rechtsschenkelblock|(?<!incomplete )right bundle branch block|(?<!inkomplett )h.ger grenblock",
    "AFIB": r"vorhofflimmern|atrial fibrillation|f.rmaksflimmer",
    "PVC": r"ventrikul.re? extrasystol|premature ventricular|ventricular premature|ventrikul.ra extraslag|\bvpb|\bves\b",
    "PACE": r"schrittmacher|pacemaker|pacad|paced",
    "WPW": r"wpw|wolff|pr.exzitation|pre-?excitation",
    "ASMI": r"anteroseptal|septal(er)? infarkt|septal infarct",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--results", required=True)
    a = ap.parse_args(argv)

    db = pd.read_csv(Path(a.dataset) / "ptbxl_database.csv")
    db["record_id"] = db.ecg_id.astype(str)
    text = db.set_index("record_id")["report"].fillna("").str.lower()
    ass = pd.read_csv(Path(a.results) / "label_assessments.csv", dtype={"record_id": str},
                      keep_default_na=False)
    rr = pd.read_csv(Path(a.results) / "record_results.csv", dtype={"record_id": str},
                     keep_default_na=False)
    labels = rr.set_index("record_id")["original_labels"].str.split(";").apply(set)

    rows = []
    for lab, pat in KEYWORDS.items():
        mentions = text.str.contains(pat, regex=True)
        has = labels.apply(lambda s, l=lab: l in s)
        for flag, cond_label, expect in (("potentially_missing", False, True),
                                         ("potentially_incorrect", True, False)):
            flagged = set(ass.loc[(ass.label == lab) & (ass.status == flag), "record_id"])
            pool = has[has == cond_label].index
            f = [r for r in pool if r in flagged]
            c = [r for r in pool if r not in flagged]
            if not f:
                continue
            # "supports flag" = report mentions finding (missing) / does not mention it (incorrect)
            fs = int((mentions[f] == expect).sum())
            cs = int((mentions[c] == expect).sum())
            odds, p = fisher_exact([[fs, len(f) - fs], [cs, len(c) - cs]])
            rows.append({
                "label": lab, "flag": flag, "n_flagged": len(f),
                "flagged_report_supports_flag": fs,
                "flagged_rate": round(fs / len(f), 3),
                "n_control": len(c), "control_rate": round(cs / len(c), 4) if c else None,
                "fisher_p": float(f"{p:.3g}"),
                "examples": ", ".join(f[:5]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(Path(a.results) / "report_concordance.csv", index=False)
    print(out.drop(columns="examples").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
