"""Batched ECGFounder inference with GPU/CPU selection and shard-level checkpointing.

ECGFounder outputs are written to ``<cache>/shard_XXXXX.npz`` as soon as each batch finishes.
A rerun with the same inference fingerprint skips every record already present in a shard, so an
interrupted run resumes where it stopped. Records that fail to load or preprocess are logged to
``<cache>/errors.jsonl`` (and retried on the next run, since failures may be transient I/O).
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .ecgfounder import N_CLASSES
from .loaders import DatasetAdapter, RecordRef
from .preprocessing import IncompatibleRecord, PreprocessConfig, preprocess

log = logging.getLogger(__name__)


def select_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    dev = torch.device(spec)
    if dev.type == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return dev


def set_determinism(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


class _PreparedDataset(Dataset):
    def __init__(self, adapter: DatasetAdapter, refs: list[RecordRef], n_leads: int,
                 pcfg: PreprocessConfig):
        self.adapter, self.refs, self.n_leads, self.pcfg = adapter, refs, n_leads, pcfg

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, i):
        ref = self.refs[i]
        try:
            raw = self.adapter.load(ref)
        except Exception as e:  # corrupted / unreadable file
            return {"record_id": ref.record_id, "error": f"load: {type(e).__name__}: {e}"}
        try:
            prep = preprocess(raw.data, raw.fs, raw.lead_names, self.n_leads, self.pcfg)
        except IncompatibleRecord as e:
            return {"record_id": ref.record_id, "error": f"incompatible: {e}"}
        except Exception as e:
            return {"record_id": ref.record_id, "error": f"preprocess: {type(e).__name__}: {e}"}
        return {"record_id": ref.record_id, "windows": prep.windows,
                "warnings": prep.warnings, "info": prep.info}


def _collate(items):
    return items


def preprocess_config(pre: dict) -> PreprocessConfig:
    keys = PreprocessConfig.__dataclass_fields__.keys()
    return PreprocessConfig(**{k: v for k, v in pre.items() if k in keys})


def load_cache(cache: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    probs, info = {}, {}
    for shard in sorted(cache.glob("shard_*.npz")):
        with np.load(shard, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            for rid, p, m in zip(z["record_ids"], z["probs"], meta):
                probs[str(rid)] = p
                info[str(rid)] = m
    return probs, info


def run_inference(model: torch.nn.Module, device: torch.device, adapter: DatasetAdapter,
                  refs: list[RecordRef], cache: Path, cfg: dict) -> tuple[dict, dict, dict]:
    """Return ({record_id: probs[150]}, {record_id: info}, {record_id: error})."""
    cache.mkdir(parents=True, exist_ok=True)
    done, info = load_cache(cache)
    todo = [r for r in refs if r.record_id not in done]
    log.info("Inference: %d records cached, %d to process", len(done), len(todo))
    errors: dict[str, str] = {}
    if not todo:
        return done, info, errors

    mcfg, pre = cfg["model"], cfg["preprocessing"]
    aggregation = pre.get("window_aggregation", "mean")
    ds = _PreparedDataset(adapter, todo, mcfg["n_leads"], preprocess_config(pre))
    workers = int(mcfg.get("num_workers", 0))
    loader = DataLoader(ds, batch_size=int(mcfg["batch_size"]), shuffle=False,
                        num_workers=workers, collate_fn=_collate,
                        persistent_workers=False, pin_memory=device.type == "cuda")
    shard_idx = len(list(cache.glob("shard_*.npz")))
    err_f = open(cache / "errors.jsonl", "a")
    t0, n_done = time.time(), 0
    try:
        for batch in loader:
            ok = [b for b in batch if "windows" in b]
            for b in batch:
                if "error" in b:
                    errors[b["record_id"]] = b["error"]
                    err_f.write(json.dumps({"record_id": b["record_id"], "error": b["error"]}) + "\n")
            err_f.flush()
            if ok:
                x = torch.from_numpy(np.concatenate([b["windows"] for b in ok]))
                with torch.inference_mode():
                    p = torch.sigmoid(model(x.to(device, non_blocking=True)).float()).cpu().numpy()
                rec_probs, start = [], 0
                for b in ok:
                    k = len(b["windows"])
                    w = p[start:start + k]
                    start += k
                    rec_probs.append(w.max(0) if aggregation == "max" else w.mean(0))
                ids = np.array([b["record_id"] for b in ok])
                meta = [{"warnings": b["warnings"], **b["info"]} for b in ok]
                arr = np.stack(rec_probs).astype(np.float32)
                assert arr.shape[1] == N_CLASSES
                tmp = cache / f".shard_{shard_idx:05d}.tmp.npz"
                np.savez(tmp, record_ids=ids, probs=arr, meta=json.dumps(meta))
                os.replace(tmp, cache / f"shard_{shard_idx:05d}.npz")  # atomic: no partial shards
                shard_idx += 1
                for rid, pr, m in zip(ids, arr, meta):
                    done[str(rid)] = pr
                    info[str(rid)] = m
            n_done += len(batch)
            rate = n_done / max(time.time() - t0, 1e-9)
            log.info("Inference %d/%d records (%.1f rec/s, %d errors)", n_done, len(todo), rate,
                     len(errors))
    finally:
        err_f.close()
    return done, info, errors
