"""End-to-end orchestration: load -> ECGFounder inference -> verification -> reports."""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from . import __version__
from .config import fingerprint, inference_fingerprint
from .ecgfounder import load_pretrained, sha256_file
from .inference import (_PreparedDataset, preprocess_config, run_inference, select_device,
                        set_determinism)
from .loaders import get_adapter
from .mapping import load_mapping
from .reporting import write_reports
from .verification import verify

log = logging.getLogger("ecg_verify")


def setup_logging(out: Path, verbose: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(out / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def _checkpoint_sha(path: Path, out: Path) -> str:
    """sha256 of the checkpoint, memoised on (path, size, mtime) to keep resumes fast."""
    st = path.stat()
    key = f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    memo = out / "cache" / "checkpoint_sha256.json"
    if memo.exists():
        d = json.loads(memo.read_text())
        if d.get("key") == key:
            return d["sha256"]
    sha = sha256_file(path)
    memo.parent.mkdir(parents=True, exist_ok=True)
    memo.write_text(json.dumps({"key": key, "sha256": sha}))
    return sha


def run(dataset: str, output: str, cfg: dict) -> dict:
    t0 = time.time()
    dataset_p, out = Path(dataset).resolve(), Path(output).resolve()
    if out == dataset_p or dataset_p in out.parents:
        raise ValueError("the output directory must not be inside the dataset directory "
                         "(original data are never modified)")
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    dcfg, mcfg = cfg["dataset"], cfg["model"]
    adapter = get_adapter(str(dataset_p), dcfg["format"], **(dcfg.get("options") or {}))
    mapping_name = dcfg.get("mapping") or adapter.default_mapping
    if not mapping_name:
        raise ValueError(f"adapter {adapter.name} has no default label mapping; pass --mapping")
    mapping = load_mapping(mapping_name)
    absence = dcfg.get("absence_semantics") or adapter.default_absence_semantics
    log.info("Dataset %s (format %s), mapping %s, absence semantics %s", dataset_p, adapter.name,
             mapping.name, absence)

    refs = adapter.records()
    n_total = len(refs)
    if dcfg.get("limit"):
        refs = refs[: int(dcfg["limit"])]
    log.info("%d recordings found, %d selected", n_total, len(refs))
    ids = [r.record_id for r in refs]
    if len(set(ids)) != len(ids):
        raise ValueError("record ids are not unique")

    ckpt = Path(mcfg["checkpoint"])
    sha = _checkpoint_sha(ckpt, out) if ckpt.is_file() else None
    if sha is None:
        load_pretrained(ckpt, mcfg["n_leads"], torch.device("cpu"))  # raises a helpful error
    set_determinism(int(mcfg.get("seed", 0)), bool(mcfg.get("deterministic", True)))
    device = select_device(mcfg.get("device", "auto"))
    model = load_pretrained(ckpt, mcfg["n_leads"], device)
    fp = inference_fingerprint(cfg, sha, str(dataset_p))
    cache = out / "cache" / f"inference_{fp}"
    probs_by_id, info, errors = run_inference(model, device, adapter, refs, cache, cfg)

    ok_refs = [r for r in refs if r.record_id in probs_by_id]
    failed = {r.record_id: errors.get(r.record_id, "not processed")
              for r in refs if r.record_id not in probs_by_id}
    if not ok_refs:
        raise RuntimeError("no recording could be processed; see failed records in the log")
    probs = np.stack([probs_by_id[r.record_id] for r in ok_refs]).astype(np.float64)
    log.info("Verifying %d recordings (%d failed)", len(ok_refs), len(failed))
    res = verify(ok_refs, probs, mapping, cfg, absence, info)

    run_meta = {
        "tool": "ecg_verify", "version": __version__,
        "started_utc": datetime.fromtimestamp(t0, timezone.utc).isoformat(),
        "runtime_s": None,
        "dataset_root": str(dataset_p), "dataset_format": adapter.name,
        "dataset_overlap_with_ecgfounder_training": adapter.overlap_note,
        "label_mapping": mapping.name, "absence_semantics": absence,
        "checkpoint": str(ckpt.resolve()), "checkpoint_sha256": sha,
        "device": str(device), "inference_fingerprint": fp, "config_fingerprint": fingerprint(cfg),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
    }
    summary = write_reports(out, res, probs, [r.record_id for r in ok_refs], failed, n_total,
                            run_meta, cfg)
    run_meta["runtime_s"] = round(time.time() - t0, 1)
    summary["run"] = run_meta
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "run_manifest.json").write_text(json.dumps(run_meta, indent=2))
    log.info("Done in %.1fs. Reports in %s", time.time() - t0, out)
    return summary


def check_data(dataset: str, output: str, cfg: dict) -> dict:
    """Load and preprocess every recording without running the model (dataset preflight)."""
    import pandas as pd
    from torch.utils.data import DataLoader

    out = Path(output).resolve()
    dcfg, mcfg = cfg["dataset"], cfg["model"]
    adapter = get_adapter(str(Path(dataset).resolve()), dcfg["format"],
                          **(dcfg.get("options") or {}))
    mapping = load_mapping(dcfg.get("mapping") or adapter.default_mapping)
    refs = adapter.records()
    if dcfg.get("limit"):
        refs = refs[: int(dcfg["limit"])]
    ds = _PreparedDataset(adapter, refs, mcfg["n_leads"], preprocess_config(cfg["preprocessing"]))
    loader = DataLoader(ds, batch_size=64, num_workers=int(mcfg.get("num_workers", 0)),
                        collate_fn=lambda b: b)
    rows = []
    for batch in loader:
        for b in batch:
            rows.append({"record_id": b["record_id"], "ok": "error" not in b,
                         "error": b.get("error", ""),
                         "warnings": ";".join(b.get("warnings", [])),
                         "n_windows": b.get("info", {}).get("n_windows")})
        log.info("checked %d/%d", len(rows), len(refs))
    df = pd.DataFrame(rows)
    df.to_csv(out / "data_check.csv", index=False)
    labels = pd.Series([l for r in refs for l in r.labels]).value_counts()
    status = {lab: ("unmapped" if mapping.get(lab) is None else
                    "unsupported" if not mapping.get(lab).supported else mapping.get(lab).quality)
              for lab in labels.index}
    label_df = pd.DataFrame({"label": labels.index, "count": labels.values,
                             "mapping": [status[l] for l in labels.index]})
    label_df.to_csv(out / "label_inventory.csv", index=False)
    summary = {
        "dataset_format": adapter.name, "label_mapping": mapping.name,
        "records": len(refs), "compatible": int(df.ok.sum()), "incompatible": int((~df.ok).sum()),
        "error_types": df.loc[~df.ok, "error"].str.split(":").str[:2].str.join(":")
        .value_counts().to_dict(),
        "records_with_warnings": int((df.warnings != "").sum()),
        "label_instances_by_mapping_status": label_df.groupby("mapping")["count"].sum().to_dict(),
    }
    (out / "data_check.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
