"""CODE-15% style HDF5 adapter (Ribeiro et al., Zenodo 4916206).

Layout: ``exams.csv`` (one row per exam: ``exam_id``, ``patient_id``, ``trace_file`` and boolean
columns ``1dAVb, RBBB, LBBB, SB, ST, AF``) and ``exams_part*.hdf5`` files each holding
``tracings`` (N, 4096, 12) at 400 Hz in lead order DI, DII, DIII, aVR, aVL, aVF, V1-V6 and
``exam_id`` (N,). Recordings shorter than 4096 samples are zero-padded symmetrically; the padding
is stripped here so that it is not mistaken for signal. The six columns are exhaustive binary
annotations, so absence is a negative for those labels (absence semantics ``negative``).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import DatasetAdapter, RawSignal, RecordRef

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LABEL_COLUMNS = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]


class CODE15Adapter(DatasetAdapter):
    name = "code15_hdf5"
    default_mapping = "code15"
    default_absence_semantics = "negative"
    overlap_note = (
        "CODE (Telehealth Network of Minas Gerais, Brazil) is not part of HEEDB pretraining. Labels "
        "were derived automatically (Glasgow/Minnesota + text reports), so they are weak labels."
    )

    def __init__(self, root: str, **options):
        super().__init__(root, **options)
        self._handles: dict[str, object] = {}
        self._index: dict[str, dict[int, int]] = {}

    @classmethod
    def detect(cls, root: str) -> bool:
        return os.path.isfile(os.path.join(root, "exams.csv")) and any(
            f.endswith(".hdf5") for f in os.listdir(root))

    def records(self) -> list[RecordRef]:
        df = pd.read_csv(os.path.join(self.root, "exams.csv"))
        present = {os.path.basename(f) for f in os.listdir(self.root) if f.endswith(".hdf5")}
        df = df[df["trace_file"].isin(present)]
        refs = []
        for row in df.to_dict("records"):
            labels = {c: 1.0 for c in LABEL_COLUMNS if bool(row.get(c))}
            refs.append(RecordRef(
                record_id=str(row["exam_id"]), labels=labels,
                source=f"{row['trace_file']}::{row['exam_id']}",
                patient_id=str(row.get("patient_id")),
                metadata={"trace_file": row["trace_file"]}))
        return refs

    def _file(self, name: str):
        import h5py
        if name not in self._handles:  # opened lazily, once per worker process
            h = h5py.File(os.path.join(self.root, name), "r")
            self._handles[name] = h
            self._index[name] = {int(e): i for i, e in enumerate(h["exam_id"][:])}
        return self._handles[name], self._index[name]

    def load(self, ref: RecordRef) -> RawSignal:
        fname, exam = ref.source.split("::")
        h, index = self._file(fname)
        x = np.asarray(h["tracings"][index[int(exam)]], dtype=np.float64).T  # (12, 4096)
        nonzero = np.flatnonzero(np.abs(x).sum(axis=0) > 0)
        if len(nonzero):
            x = x[:, nonzero[0]: nonzero[-1] + 1]
        return RawSignal(x, 400.0, LEADS)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"], state["_index"] = {}, {}
        return state
