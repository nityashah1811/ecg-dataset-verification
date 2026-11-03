# ECG dataset label verification with ECGFounder

A fully automated research pipeline that runs the official
[ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) foundation model on every recording
in an ECG dataset. It compares the 150-class predictions with the dataset's own diagnostic
labels, flags labels that may be incorrect or missing, scores how strong each discrepancy is,
and writes proposed label corrections, CSV reports, dataset statistics and plots. The original
data are never modified.

> ECGFounder is an independent model-based reference, **not ground truth**. Every rate this tool
> reports measures *agreement between the model and the labels*, not label accuracy. Flags are
> candidates for expert review. See [docs/METHODS.md](docs/METHODS.md).

## Quick start

```bash
pip install -r requirements.txt

# 1. Official weights (Hugging Face: PKUDigitalHealth/ECGFounder, ~370 MB)
mkdir -p checkpoints
wget -O checkpoints/12_lead_ECGFounder.pth \
  https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/main/12_lead_ECGFounder.pth

# 2. Optional preflight: load + preprocess every record, no model needed
python verify_ecg_labels.py --dataset /data/ptb-xl --output results/ptbxl --check-data

# 3. Full unattended run (GPU used automatically, CPU fallback; resumable)
python verify_ecg_labels.py --dataset /data/ptb-xl --output results/ptbxl \
    --checkpoint checkpoints/12_lead_ECGFounder.pth
```

To resume after an interruption, re-run the same command. Model outputs are cached in atomic
shards and keyed by a fingerprint of the checkpoint SHA-256, the preprocessing settings and the
dataset. You can change thresholds, mappings or calibration settings and re-run without new
inference.

Useful options: `--config my.yaml` (see `configs/default.yaml`), `--format`, `--mapping`,
`--absence-semantics {unknown,negative}`, `--device`, `--batch-size`, `--num-workers`,
`--limit N`, `--no-plots`.

## Supported datasets

| `--format` | Layout | Default mapping | Absence of a label means |
|---|---|---|---|
| `ptbxl` | PTB-XL v1.0.3 (`ptbxl_database.csv`, WFDB 500 Hz) | `ptbxl_scp` | unknown |
| `wfdb_dx` | PhysioNet/CinC 2020/2021 WFDB with `# Dx:` SNOMED-CT codes | `snomed_physionet2021` | unknown |
| `wfdb_manifest` | any WFDB folder + `labels.csv` (`record,labels[,patient_id,fold]`) | pass `--mapping` | unknown |
| `code15_hdf5` | CODE-15% (`exams.csv` + `exams_part*.hdf5`, 400 Hz) | `code15` | negative |

The format is auto-detected. To add a dataset, write a label mapping YAML (format documented at
the top of `mappings/ptbxl_scp.yaml`) and, if needed, a small adapter in `ecg_verify/loaders/`.

## Outputs (`--output`)

| File | Content |
|---|---|
| `record_results.csv` | one row per recording: original labels, predicted labels, calibrated scores, agreement status, discrepancy score/severity, suggested additions/removals, proposed labels, final automated classification, signal warnings, dataset metadata. Failed recordings are included with a reason. |
| `label_discrepancies.csv` | only recordings with potentially incorrect/missing labels, sorted by discrepancy score, with per-flag details |
| `dataset_summary.json` | totals, agreement/disagreement/uncertain rates, per-diagnosis rates and confidence distributions, suggestion counts, unsupported/uncertain counts, stratified rates (e.g. PTB-XL `validated_by_human`), calibration diagnostics, run provenance and dataset-overlap note |
| `proposed_labels.csv` | separate proposed-label dataset: original vs proposed labels per recording |
| `label_assessments.csv` | long table, one row per (recording, label) decision, with raw score, calibrated probability, status and reason |
| `label_statistics.csv` | per-label counts, rates, thresholds, AUROC/AUPRC (model vs label) and ECE/Brier |
| `ecgfounder_probabilities.csv.gz` | raw 150-class ECGFounder probabilities per recording |
| `failed_records.csv` | recordings that could not be read or are incompatible, with reason |
| `plots/` | per-label agreement and missing rates, confidence distributions, discrepancy scores, calibration reliability |
| `run_manifest.json`, `config_resolved.yaml`, `run.log` | provenance: versions, device, checkpoint SHA-256, fingerprints, full log |

### Label categories

* **consistent**: ECGFounder supports an existing label.
* **potentially incorrect**: ECGFounder strongly contradicts an existing label.
* **potentially missing**: ECGFounder strongly predicts a supported diagnosis that is absent.
* **uncertain**: confidence is between the decision thresholds, or a strong prediction
  conflicts with another existing label.
* **not comparable**: the label is unsupported by ECGFounder, unmapped, or only approximately
  mapped.

Decision rules, thresholds, per-label overrides, comparable mapping qualities and suggestion
policy are all set in the config file (`configs/default.yaml`).

## Project layout

```
verify_ecg_labels.py        single-command entry point
ecg_verify/
  ecgfounder/               vendored official Net1D (MIT) + tasks.txt; strict checkpoint loading
  preprocessing.py          official filter + global z-score, resampling, lead mapping, windows
  loaders/                  dataset adapters (PTB-XL, WFDB Dx, WFDB manifest, CODE-15 HDF5)
  mapping.py                label-mapping loader and validation
  calibration.py            cross-fitted Platt / isotonic calibration + diagnostics
  verification.py           comparison, discrepancy scoring, suggestions
  inference.py              batched GPU/CPU inference, checkpointed shards, error capture
  reporting.py              CSV / JSON / plots
  pipeline.py               orchestration, provenance, --check-data preflight
mappings/                   curated dataset-label -> ECGFounder-class mappings
configs/default.yaml        all tunable settings
docs/METHODS.md             methods, assumptions, limitations
docs/checkpoint_provenance.md   provenance check of the ECGFounder weights used
docs/manuscript/build/      numbers and Fig. 3 used by the manuscript builder
tools/analysis/             follow-up analyses: baselines, threshold sensitivity, adjudication tooling
tools/manuscript/           builds the manuscript (.docx) directly from the result files
tools/ptbxl_report_concordance.py   model-independent check of flags against PTB-XL report text
reports/                    results of the full PTB-XL run (see below)
tests/                      unit + end-to-end tests (pytest)
```

## Tests

```bash
python -m pytest
```

The tests cover:

* preprocessing parity with a verbatim copy of the official filter;
* lead reordering, resampling, windowing and rejection rules;
* mapping validation, including rejection of serial-comparison classes;
* every label category and the multilabel and any-of logic;
* `covered_by` and exclusivity rules, and both absence semantics;
* threshold overrides;
* cross-fit calibration without leakage;
* the proposed-label identity;
* strict checkpoint loading;
* end-to-end runs on synthetic PTB-XL and PhysioNet-Challenge datasets, including a corrupted
  record, a 1000 Hz long record and a too-short record. These runs also check that the dataset
  is unchanged, that results are resumable, and that output is deterministic.

The end-to-end tests use a **randomly initialised** network in the official checkpoint format,
so they test the plumbing, not the model.

## Results on PTB-XL

The full run on PTB-XL v1.0.3 with the official 12-lead weights is in
[`reports/ptbxl_ecgfounder/`](reports/ptbxl_ecgfounder/), and the follow-up analyses are in
[`reports/ptbxl_ecgfounder/analyses/`](reports/ptbxl_ecgfounder/analyses/). In brief:

* **Processed:** all 21,799 recordings, 0 failures. 64.1% of label instances have an exact
  ECGFounder equivalent.
* **Model–label agreement:** median AUROC 0.942 over 36 labels. It is ≥0.99 for left bundle
  branch block, sinus tachycardia, atrial fibrillation, PVCs and atrial flutter, and lowest for
  infarct-territory statements (0.76–0.84).
* **Flags:** 62.7% of assessable recordings agree with the model, 35.3% are inconclusive, and 425
  (1.96%) are flagged. The flags are 93 potentially incorrect and 346 potentially missing labels.
* **Plausibility, independent of the model:** flags are strongly enriched for records whose own
  free-text report contradicts their codes. The label-adjusted Mantel–Haenszel odds ratios are
  75.9 (95% CI 54–106) for missing and 7.7 (2.6–22.4) for incorrect flags. Enrichment rises
  monotonically with stricter thresholds.
* **Base-rate pitfall:** thresholding the calibrated posterior instead of the Bayes factor would
  flag 3,414 labels, including every anterior-MI label, purely because rare labels have low
  posteriors.
* **Comparison methods:** raw scores, a posterior rule, in-domain confident learning, GE 12SL
  disagreement and keyword mismatch flag 9–55× more labels, and their flags barely overlap with
  ours.

These are **agreement** results, not label accuracy. A blinded cardiologist adjudication is in
progress.

### Reproducing the follow-up analyses

```bash
python tools/ptbxl_report_concordance.py --dataset /path/to/ptb-xl --results results/ptbxl
python tools/analysis/extract_embeddings.py --dataset /path/to/ptb-xl \
    --checkpoint checkpoints/12_lead_ECGFounder.pth --out results/ptbxl_embeddings.npz
python tools/analysis/sensitivity.py --dataset /path/to/ptb-xl --results results/ptbxl \
    --out reports/ptbxl_ecgfounder/analyses
python tools/analysis/baselines.py --dataset /path/to/ptb-xl --ptbxl-plus /path/to/ptb-xl-plus \
    --results results/ptbxl --embeddings results/ptbxl_embeddings.npz \
    --out reports/ptbxl_ecgfounder/analyses
```

PTB-XL+ (for the GE 12SL comparison) is available from PhysioNet.

### Rebuilding the manuscript

```bash
python tools/manuscript/collect_stats.py --ptbxl /path/to/ptb-xl   # numbers + Fig. 3 from reports/
NODE_PATH=$(npm root -g) node tools/manuscript/build_manuscript.js   # -> docs/manuscript/*.docx (needs the docx npm package)
# public preprint (no line numbers, no placeholders); fill in docs/manuscript/preprint_meta.json first
MODE=preprint NODE_PATH=$(npm root -g) node tools/manuscript/build_manuscript.js   # -> docs/manuscript/preprint/
```

## Citation

If you use this code or these results, please cite the preprint (link to be added once posted):

> [Author]. *Auditing diagnostic labels of a public ECG benchmark with a foundation model: a
> calibration-aware, fully automated analysis of PTB-XL using ECGFounder.* TechRxiv, 2026.

## Data, model and licences

* **PTB-XL** (Wagner et al., *Sci Data* 2020) and **PTB-XL+** (Strodthoff et al., *Sci Data*
  2023) are available from PhysioNet under CC BY 4.0. They are not redistributed here; the
  derived per-record results in `reports/` refer to records by `ecg_id`.
* **ECGFounder** (Li et al., *NEJM AI* 2025): the weights are downloaded from Hugging Face, and
  `ecg_verify/ecgfounder/net1d.py` is vendored from the official repository under the MIT licence
  (see `ecg_verify/ecgfounder/LICENSE`).
* The code in this repository is released under the MIT licence (see `LICENSE`).

## Acknowledgements

This project was developed with substantial assistance from an AI coding assistant (Claude,
Anthropic), under the author's direction. The author reviewed the code, the analyses and the
results.

