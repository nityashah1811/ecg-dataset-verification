"""Generic WFDB adapters.

``WFDBDxAdapter`` handles the PhysioNet/CinC Challenge 2020/2021 layout (CPSC, Georgia, Chapman-
Shaoxing, Ningbo, PTB, St Petersburg ...): ``*.hea`` headers with ``# Dx: <SNOMED-CT codes>``
comment lines next to ``.mat`` or ``.dat`` signal files.

``WFDBManifestAdapter`` handles any WFDB collection whose labels live in a CSV manifest with
columns ``record`` (path relative to the dataset root, without extension), ``labels``
(separator-delimited) and optionally ``patient_id`` / ``fold``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

from .base import DatasetAdapter, RawSignal, RecordRef


def _read_wfdb(path: str) -> RawSignal:
    try:
        rec = wfdb.rdrecord(path)
        return RawSignal(rec.p_signal.T, float(rec.fs), list(rec.sig_name))
    except Exception:
        # Challenge .mat files: fall back to scipy + header gains.
        mat = path + ".mat"
        if not os.path.isfile(mat):
            raise
        from scipy.io import loadmat
        hdr = wfdb.rdheader(path)
        val = np.asarray(loadmat(mat)["val"], dtype=np.float64)
        gain = np.asarray(hdr.adc_gain, dtype=np.float64)[:, None]
        base = np.asarray(hdr.baseline, dtype=np.float64)[:, None]
        return RawSignal((val - base) / gain, float(hdr.fs), list(hdr.sig_name))


class WFDBDxAdapter(DatasetAdapter):
    name = "wfdb_dx"
    default_mapping = "snomed_physionet2021"
    default_absence_semantics = "unknown"
    stratify_by = ("source_dir",)

    @classmethod
    def detect(cls, root: str) -> bool:
        for i, hea in enumerate(Path(root).rglob("*.hea")):
            with open(hea, errors="replace") as f:
                if any(line.startswith("#") and "Dx" in line for line in f):
                    return True
            if i > 20:
                break
        return False

    def records(self) -> list[RecordRef]:
        refs = []
        for hea in sorted(Path(self.root).rglob("*.hea")):
            labels, meta = {}, {"source_dir": str(hea.parent.relative_to(self.root))}
            with open(hea, errors="replace") as f:
                for line in f:
                    if not line.startswith("#"):
                        continue
                    key, _, value = line.lstrip("# ").partition(":")
                    key, value = key.strip(), value.strip()
                    if key == "Dx":
                        labels = {c.strip(): 1.0 for c in value.split(",") if c.strip()}
                    elif key in ("Age", "Sex"):
                        meta[key.lower()] = value
            rel = str(hea.relative_to(self.root).with_suffix(""))
            refs.append(RecordRef(record_id=rel, labels=labels, source=rel, metadata=meta))
        return refs

    def load(self, ref: RecordRef) -> RawSignal:
        return _read_wfdb(os.path.join(self.root, ref.source))


class WFDBManifestAdapter(DatasetAdapter):
    name = "wfdb_manifest"
    default_mapping = None

    @classmethod
    def detect(cls, root: str) -> bool:
        return os.path.isfile(os.path.join(root, "labels.csv"))

    def records(self) -> list[RecordRef]:
        path = self.options.get("labels_csv") or os.path.join(self.root, "labels.csv")
        sep = self.options.get("label_separator", ";")
        df = pd.read_csv(path, dtype=str).fillna("")
        missing = {"record", "labels"} - set(df.columns)
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)}")
        refs = []
        for row in df.to_dict("records"):
            labels = {s.strip(): 1.0 for s in row["labels"].split(sep) if s.strip()}
            extra = {k: v for k, v in row.items() if k not in ("record", "labels")}
            fold = extra.get("fold")
            refs.append(RecordRef(
                record_id=row["record"], labels=labels, source=row["record"],
                patient_id=extra.get("patient_id") or None,
                fold_hint=int(fold) if fold not in (None, "") else None, metadata=extra))
        return refs

    def load(self, ref: RecordRef) -> RawSignal:
        return _read_wfdb(os.path.join(self.root, ref.source))
