#!/usr/bin/env python
"""Automated ECG dataset label verification with ECGFounder.

    python verify_ecg_labels.py --dataset /path/to/ptb-xl --output results/ptbxl \
        --checkpoint checkpoints/12_lead_ECGFounder.pth

Runs unattended: loads every recording, runs ECGFounder, compares predictions with the
original labels and writes record_results.csv, label_discrepancies.csv, dataset_summary.json,
proposed_labels.csv and plots to --output. Re-running the same command resumes from the cached
model outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ecg_verify.config import load_config
from ecg_verify.pipeline import check_data, run, setup_logging


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset root directory (read-only)")
    p.add_argument("--output", required=True, help="results directory")
    p.add_argument("--config", help="YAML config (see configs/default.yaml)")
    p.add_argument("--checkpoint", help="ECGFounder checkpoint (.pth)")
    p.add_argument("--format", help="dataset format (auto, ptbxl, wfdb_dx, wfdb_manifest, code15_hdf5)")
    p.add_argument("--mapping", help="label mapping name (mappings/*.yaml) or path")
    p.add_argument("--absence-semantics", choices=["unknown", "negative"],
                   help="whether an absent label is a confirmed negative")
    p.add_argument("--device", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--limit", type=int, help="process only the first N recordings")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--check-data", action="store_true",
                   help="only load + preprocess every recording (no model needed) and report")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    ov: dict = {"dataset": {}, "model": {}, "output": {}}
    for key, sect, name in [("checkpoint", "model", "checkpoint"), ("device", "model", "device"),
                            ("batch_size", "model", "batch_size"),
                            ("num_workers", "model", "num_workers"),
                            ("format", "dataset", "format"), ("mapping", "dataset", "mapping"),
                            ("absence_semantics", "dataset", "absence_semantics"),
                            ("limit", "dataset", "limit")]:
        v = getattr(a, key)
        if v is not None:
            ov[sect][name] = v
    if a.no_plots:
        ov["output"]["plots"] = False
    cfg = load_config(a.config, ov)
    setup_logging(Path(a.output), a.verbose)
    try:
        if a.check_data:
            print(json.dumps(check_data(a.dataset, a.output, cfg), indent=2))
            return 0
        summary = run(a.dataset, a.output, cfg)
    except Exception:
        logging.getLogger("ecg_verify").exception("run failed")
        return 1
    rl = summary["record_level"]
    print(json.dumps({"processed": summary["counts"]["total_recordings_processed"],
                      "failed": summary["counts"]["failed_or_incompatible_recordings"],
                      "agreement_rate": rl["agreement_rate"],
                      "disagreement_rate": rl["disagreement_rate"],
                      "uncertain_rate": rl["uncertain_rate"],
                      "output": a.output}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
