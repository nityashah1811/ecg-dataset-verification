"""Official ECGFounder architecture, label space and checkpoint loading.

Facts taken from the official repository (PKUDigitalHealth/ECGFounder, commit 04edac7):

* ``tasks.txt`` lists the 150 output classes, in output-index order. They are GE MUSE-style
  interpretation statements from the Harvard-Emory ECG Database (HEEDB), the pretraining data.
* ``ptbxl_eval.py`` builds ``Net1D(in_channels=12, base_filters=64, ratio=1,
  filter_list=[64,160,160,400,400,1024,1024], m_blocks_list=[2,2,2,3,3,4,4], kernel_size=16,
  stride=2, groups_width=16, use_bn=False, use_do=False, n_classes=150)``, loads
  ``checkpoint['state_dict']`` and applies ``sigmoid`` to the logits (independent multilabel
  probabilities; not mutually exclusive and not calibrated).
* The 1-lead checkpoint uses the same architecture with ``in_channels=1`` (lead I).
"""

from __future__ import annotations

import hashlib
import logging
from importlib import resources
from pathlib import Path

import torch

from .net1d import Net1D

log = logging.getLogger(__name__)

N_CLASSES = 150
MODEL_LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

ARCH_KWARGS = dict(
    base_filters=64,
    ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16,
    stride=2,
    groups_width=16,
    verbose=False,
    use_bn=False,
    use_do=False,
)


def load_class_names() -> list[str]:
    text = resources.files(__package__).joinpath("tasks.txt").read_text()
    names = [line.strip() for line in text.splitlines() if line.strip()]
    if len(names) != N_CLASSES:
        raise RuntimeError(f"tasks.txt has {len(names)} classes, expected {N_CLASSES}")
    return names


CLASS_NAMES = load_class_names()
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

# Statements that describe a change relative to a previous ECG of the same patient. A single
# recording cannot support or contradict them, so they are never used for label comparison.
_SERIAL_MARKERS = (
    "NOW EVIDENT", "NO LONGER", "HAS SHORTENED", "HAS LENGTHENED", "HAS REPLACED",
    "HAVE REPLACED", "LESS EVIDENT", "MORE EVIDENT", "HAS INCREASED", "HAS DECREASED", "NOW DEPRESSED",
    "NOW PRESENT", "LESS DEPRESSED", "MORE DEPRESSED", "LESS ELEVATED", "MORE ELEVATED",
    "NON-SPECIFIC CHANGE IN ST SEGMENT IN",
)
SERIAL_COMPARISON_CLASSES = frozenset(
    c for c in CLASS_NAMES if any(m in c.upper() for m in _SERIAL_MARKERS)
)

# Sentence fragments / modifiers that only make sense attached to another statement. They may be
# used in a mapping (e.g. "IN A PATTERN OF BIGEMINY") but are reported as such.
MODIFIER_CLASSES = frozenset({
    "WITH RAPID VENTRICULAR RESPONSE", "WITH SLOW VENTRICULAR RESPONSE", "OR DIGITALIS EFFECT",
    "ANTEROLATERAL LEADS", "R IN AVL", "ACUTE", "BLOCKED", "MASKED BY FASCICULAR BLOCK",
    "PEDIATRIC ECG ANALYSIS", "WITH QRS WIDENING", "WITH REPOLARIZATION ABNORMALITY",
    "WITH QRS WIDENING AND REPOLARIZATION ABNORMALITY", "IN A PATTERN OF BIGEMINY",
    "WITH A COMPETING JUNCTIONAL PACEMAKER", "WITH RETROGRADE CONDUCTION",
    "CONSIDER RIGHT VENTRICULAR INVOLVEMENT IN ACUTE INFERIOR INFARCT",
})


def build_model(n_leads: int = 12) -> Net1D:
    if n_leads not in (1, 12):
        raise ValueError("ECGFounder checkpoints exist for 12 leads and 1 lead (lead I) only")
    return Net1D(in_channels=n_leads, n_classes=N_CLASSES, **ARCH_KWARGS)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _safe_torch_load(path: Path):
    """``torch.load`` with ``weights_only=True`` (no arbitrary code execution).

    The official checkpoint also stores training metadata (``val_auroc`` as a numpy scalar,
    optimizer/scheduler state), so the numpy scalar/dtype reconstructors are allowlisted. They
    only rebuild plain numbers and cannot execute code.
    """
    import numpy as np
    try:
        from numpy._core.multiarray import scalar
    except ImportError:  # numpy < 2
        from numpy.core.multiarray import scalar
    allowed = [(scalar, "numpy.core.multiarray.scalar"), np.dtype,
               *(type(np.dtype(t)) for t in ("float64", "float32", "int64", "int32", "bool"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def load_pretrained(checkpoint: str | Path, n_leads: int, device: torch.device) -> Net1D:
    """Load an official ECGFounder checkpoint with *strict* key matching.

    The official scripts use ``strict=False``, which silently leaves layers randomly initialised
    if keys do not match. For label auditing that failure mode is unacceptable, so any missing or
    unexpected key is an error here.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"ECGFounder checkpoint not found: {checkpoint}\n"
            "Download the official weights from https://huggingface.co/PKUDigitalHealth/ECGFounder "
            "(12_lead_ECGFounder.pth or 1_lead_ECGFounder.pth) and pass --checkpoint."
        )
    obj = _safe_torch_load(checkpoint)
    state_dict = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model = build_model(n_leads)
    model.load_state_dict(state_dict, strict=True)
    dense = state_dict["dense.weight"]
    if dense.shape[0] != N_CLASSES:
        raise RuntimeError(f"checkpoint head has {dense.shape[0]} outputs, expected {N_CLASSES}")
    first = state_dict["first_conv.conv.weight"]
    if first.shape[1] != n_leads:
        raise RuntimeError(f"checkpoint expects {first.shape[1]} input leads, config says {n_leads}")
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    log.info("Loaded ECGFounder checkpoint %s (%d leads) on %s", checkpoint, n_leads, device)
    return model
