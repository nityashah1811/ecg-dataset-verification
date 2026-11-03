"""Dataset-label <-> ECGFounder-class mapping."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .ecgfounder import CLASS_INDEX, SERIAL_COMPARISON_CLASSES

MAPPINGS_DIR = Path(__file__).resolve().parent.parent / "mappings"
QUALITIES = ("exact", "approximate")


@dataclass(frozen=True)
class LabelMap:
    label: str
    description: str = ""
    model_classes: tuple[str, ...] = ()
    quality: str | None = None          # exact | approximate | None (unsupported)
    unsupported_reason: str | None = None
    covered_by: tuple[str, ...] = ()
    exclusive_with: tuple[str, ...] = ()
    compatible_with: tuple[str, ...] = ()
    suggest: bool = True

    @property
    def supported(self) -> bool:
        return self.unsupported_reason is None

    @property
    def class_indices(self) -> list[int]:
        return [CLASS_INDEX[c] for c in self.model_classes]

    def conflicting(self, present: set[str]) -> list[str]:
        """Present labels that make adding this label clinically incoherent."""
        out = []
        for other in sorted(present):
            if other == self.label or _match(other, self.compatible_with):
                continue
            if "*" in self.exclusive_with or _match(other, self.exclusive_with):
                out.append(other)
        return out


def _match(label: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(label, p) for p in patterns if p != "*")


@dataclass
class LabelMapping:
    name: str
    vocabulary: str
    labels: dict[str, LabelMap] = field(default_factory=dict)

    def get(self, label: str) -> LabelMap | None:
        return self.labels.get(label)

    def comparable(self, qualities: tuple[str, ...]) -> list[LabelMap]:
        return [m for m in self.labels.values() if m.supported and m.quality in qualities]

    def label_scores(self, probs: np.ndarray, labels: list[str]) -> np.ndarray:
        """Any-of score per dataset label: max ECGFounder probability over its model classes."""
        out = np.empty((probs.shape[0], len(labels)), dtype=np.float64)
        for j, lab in enumerate(labels):
            out[:, j] = probs[:, self.labels[lab].class_indices].max(axis=1)
        return out


def resolve_mapping_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        return p
    cand = MAPPINGS_DIR / f"{name_or_path}.yaml"
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"label mapping {name_or_path!r} not found (looked in {MAPPINGS_DIR})")


def load_mapping(name_or_path: str) -> LabelMapping:
    path = resolve_mapping_path(name_or_path)
    raw = yaml.safe_load(path.read_text())
    mapping = LabelMapping(name=raw.get("name", path.stem), vocabulary=raw.get("vocabulary", ""))
    errors = []
    for label, spec in (raw.get("labels") or {}).items():
        label = str(label)
        spec = spec or {}
        if "unsupported" in spec:
            mapping.labels[label] = LabelMap(label=label, description=spec.get("description", ""),
                                             unsupported_reason=str(spec["unsupported"]))
            continue
        classes = tuple(spec.get("model_classes") or ())
        quality = spec.get("quality")
        if not classes:
            errors.append(f"{label}: no model_classes and not marked unsupported")
        if quality not in QUALITIES:
            errors.append(f"{label}: quality must be one of {QUALITIES}, got {quality!r}")
        for c in classes:
            if c not in CLASS_INDEX:
                errors.append(f"{label}: unknown ECGFounder class {c!r}")
            elif c in SERIAL_COMPARISON_CLASSES:
                errors.append(f"{label}: {c!r} is a serial-comparison statement and cannot be "
                              "verified from a single ECG")
        mapping.labels[label] = LabelMap(
            label=label, description=spec.get("description", ""), model_classes=classes,
            quality=quality, covered_by=tuple(str(x) for x in spec.get("covered_by", ())),
            exclusive_with=tuple(str(x) for x in spec.get("exclusive_with", ())),
            compatible_with=tuple(str(x) for x in spec.get("compatible_with", ())),
            suggest=bool(spec.get("suggest", True)))
    for m in mapping.labels.values():
        for other in m.covered_by:
            if other not in mapping.labels:
                errors.append(f"{m.label}: covered_by references unknown label {other!r}")
    if errors:
        raise ValueError(f"invalid label mapping {path}:\n  " + "\n  ".join(errors))
    return mapping
