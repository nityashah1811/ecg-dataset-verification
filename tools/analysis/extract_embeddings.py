#!/usr/bin/env python
"""Extract ECGFounder penultimate-layer embeddings (1024-d) for every recording.

Uses the same adapter and preprocessing as the main pipeline. Embeddings feed the in-domain
confident-learning baseline (tools/analysis/baselines.py).

    python tools/analysis/extract_embeddings.py --dataset /data/ptb-xl \
        --checkpoint checkpoints/12_lead_ECGFounder.pth --out results/ptbxl_embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ecg_verify.config import load_config  # noqa: E402
from ecg_verify.ecgfounder import load_pretrained  # noqa: E402
from ecg_verify.inference import _PreparedDataset, preprocess_config, select_device  # noqa: E402
from ecg_verify.loaders import get_adapter  # noqa: E402

log = logging.getLogger("embeddings")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=3)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cfg = load_config()
    adapter = get_adapter(a.dataset)
    refs = adapter.records()
    device = select_device("auto")
    model = load_pretrained(a.checkpoint, 12, device)
    model.return_features = True
    ds = _PreparedDataset(adapter, refs, 12, preprocess_config(cfg["preprocessing"]))
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=a.num_workers,
                        collate_fn=lambda b: b)
    ids, feats = [], []
    with torch.inference_mode():
        for k, batch in enumerate(loader):
            ok = [b for b in batch if "windows" in b]
            x = torch.from_numpy(np.concatenate([b["windows"] for b in ok])).to(device)
            _, f = model(x)
            f = f.float().cpu().numpy()
            start = 0
            for b in ok:
                n = len(b["windows"])
                feats.append(f[start:start + n].mean(0))
                ids.append(b["record_id"])
                start += n
            if k % 20 == 0:
                log.info("%d/%d", len(ids), len(refs))
    np.savez_compressed(a.out, record_ids=np.array(ids), features=np.stack(feats).astype(np.float32))
    log.info("wrote %s (%d x %d)", a.out, len(ids), feats[0].shape[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
