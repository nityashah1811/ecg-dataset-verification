# Provenance of the ECGFounder checkpoint used

Checked 2026-10-31.

## The file
* Hugging Face `PKUDigitalHealth/ECGFounder`, file `12_lead_ECGFounder.pth`, uploaded in commit
  `47a965ea` on 2025-03-12. The repository history has six commits, and this is the only upload
  of the 12-lead weights. Later commits only touched the README and `config.json`, so no other
  version of these weights has been published.
* SHA-256 `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`. This equals the
  Hugging Face LFS hash of our downloaded copy.
* Stored training metadata: `epoch: 0`, `step: 10000`, `val_auroc: 0.9204`. The file also
  contains the optimizer and scheduler state.
* The model card licence is MIT, matching the GitHub repository.

## What the paper says (arXiv 2410.04133v4)
* Pretraining used a batch size of 1,024, at most 20 epochs, early stopping, and AdamW with a
  learning rate of 1e-3.
* "The model weights with the highest AUC on the validation set are saved as a checkpoint."
* HEEDB split (Table S5): 6,683,587 training, 835,448 validation and 834,926 test ECGs.
* External PTB-XL validation: "average AUROC of 0.924 (95% CI, 0.917 to 0.931)" over the classes
  that are shared with the baselines.
* Labels come from cardiologists and ECG technicians, assisted by the GE Marquette 12SL program.
  The pretraining loss includes "positive label augmentation" because real-world annotations have
  missing diagnoses.

## Assessment
* **Consistent with the paper:** a checkpoint selected for best validation AUC that stores a
  macro validation AUROC of 0.92 fits the stated procedure. The paper reports AUROC above 0.95
  for 80 of the 150 diagnoses, so a 150-class macro average around 0.92 is plausible.
* **Open question:** 6.68 M training ECGs at batch size 1,024 is about 6,527 steps per epoch, so
  "epoch 0, step 10000" does not map onto a full epoch. The step counter may count something else,
  such as per-device steps or validation intervals.
* **Our own check:** on PTB-XL our label-level median AUROC is 0.942 over 36 exact-mapped labels.
  That is in line with the paper's PTB-XL average of 0.924, although the label sets differ. The
  released weights therefore appear to behave as published.
* **Recommendation:** before submission, e-mail the corresponding author (Shenda Hong, Peking
  University) to confirm that the Hugging Face checkpoint is the one evaluated in the paper.
  Cite the exact file hash in the manuscript.
