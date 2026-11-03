"""Dataset adapter interface.

An adapter enumerates recordings (with their *original* labels and metadata) and loads the raw
signal of one recording. Adapters only ever read from the dataset directory.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np


@dataclass
class RecordRef:
    record_id: str
    labels: dict[str, float]          # original dataset label -> annotation certainty (1.0 if n/a)
    source: str                       # path or key used by the adapter to load the signal
    patient_id: str | None = None
    fold_hint: int | None = None      # dataset-provided split, used for calibration cross-fitting
    metadata: dict = field(default_factory=dict)


@dataclass
class RawSignal:
    data: np.ndarray                  # (n_leads, n_samples), physical units
    fs: float
    lead_names: list[str]


class DatasetAdapter:
    name: str = "base"
    default_mapping: str | None = None
    # How to interpret a label that is absent from a record (see README, "absence semantics").
    default_absence_semantics: str = "unknown"
    # What is known about overlap with ECGFounder's training / evaluation data.
    overlap_note: str = (
        "Unknown. ECGFounder was pretrained on the Harvard-Emory ECG Database (MGH + Emory). "
        "Verify that this dataset shares no recordings or patients with HEEDB."
    )
    stratify_by: tuple[str, ...] = ()

    def __init__(self, root: str, **options):
        self.root = root
        self.options = options

    @classmethod
    def detect(cls, root: str) -> bool:
        return False

    def records(self) -> list[RecordRef]:
        raise NotImplementedError

    def load(self, ref: RecordRef) -> RawSignal:
        raise NotImplementedError


def stable_fold(key: str, n_folds: int) -> int:
    """Deterministic fold assignment from a string key (patient id or record id)."""
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % n_folds
