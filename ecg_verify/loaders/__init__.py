from __future__ import annotations

from .base import DatasetAdapter, RawSignal, RecordRef, stable_fold
from .hdf5_code import CODE15Adapter
from .ptbxl import PTBXLAdapter
from .wfdb_dx import WFDBDxAdapter, WFDBManifestAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    a.name: a for a in (PTBXLAdapter, CODE15Adapter, WFDBManifestAdapter, WFDBDxAdapter)
}


def get_adapter(root: str, fmt: str = "auto", **options) -> DatasetAdapter:
    if fmt != "auto":
        if fmt not in ADAPTERS:
            raise ValueError(f"unknown dataset format {fmt!r}; choose from {sorted(ADAPTERS)}")
        return ADAPTERS[fmt](root, **options)
    for cls in ADAPTERS.values():  # most specific first
        if cls.detect(root):
            return cls(root, **options)
    raise ValueError(
        f"could not detect the dataset format of {root}; pass --format "
        f"({', '.join(sorted(ADAPTERS))})")


__all__ = ["ADAPTERS", "DatasetAdapter", "RawSignal", "RecordRef", "get_adapter", "stable_fold"]
