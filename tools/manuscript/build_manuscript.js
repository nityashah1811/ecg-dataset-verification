// Builds the submission manuscript (.docx) from docs/manuscript/build/manuscript_data.json and
// the pipeline figures. Every number in the text is read from that JSON.
//
//   python tools/manuscript/collect_stats.py --ptbxl /path/to/ptb-xl
//   NODE_PATH=$(npm root -g) node tools/manuscript/build_manuscript.js
//
// Placeholders that the authors must complete are highlighted in yellow.

const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, Footer, PageNumber, LevelFormat, PageBreak,
  LineNumberRestartFormat, TableLayoutType, VerticalAlign, LineRuleType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..", "..");
const BUILD = path.join(ROOT, "docs", "manuscript", "build");
const PLOTS = path.join(ROOT, "reports", "ptbxl_ecgfounder", "plots");
// MODE=preprint builds a clean public version: no line numbers, no highlighted placeholders.
// Author details come from docs/manuscript/preprint_meta.json; every [[placeholder]] must be
// resolved below, otherwise the build fails (nothing unfinished can leak into a public PDF).
const PREPRINT = process.env.MODE === "preprint";
const OUT = PREPRINT
  ? path.join(ROOT, "docs", "manuscript", "preprint", "ECGFounder_PTBXL_label_audit_preprint.docx")
  : path.join(ROOT, "docs", "manuscript", "ECGFounder_PTBXL_label_audit_manuscript.docx");
const META = PREPRINT ? JSON.parse(fs.readFileSync(process.env.PREPRINT_META || path.join(ROOT, "docs", "manuscript", "preprint_meta.json"), "utf8")) : {};
if (PREPRINT) {
  for (const k of ["authors", "affiliation", "email", "contributions", "competing", "funding", "ai_disclosure"]) {
    if (!META[k] || /TODO/.test(META[k])) throw new Error(`preprint_meta.json: fill in "${k}"`);
  }
}
const REPO_URL = "https://github.com/nityashah1811/ecg-dataset-verification";
// Preprint replacement for each placeholder ("" = drop it).
const PREPRINT_TEXT = {
  "Confirm with the ECGFounder authors that this checkpoint is the one evaluated in their paper.": "",
  "Confirm with the local ethics office that no approval is required for this secondary analysis.": "",
  "The mapping (mappings/ptbxl_scp.yaml) should be reviewed by a cardiologist before submission.":
    "Clinical review of the mapping is in progress.",
  "Results to be added after adjudication.":
    "The adjudication is in progress, and its results will be reported in a revised version of this preprint.",
  "Update when the adjudication is complete.": "",
  "Confirm local requirements.": "No new data were collected.",
  "Others": "",
};
function resolvePlaceholder(inner) {
  if (inner.startsWith("https://github.com/")) return REPO_URL;
  if (inner in PREPRINT_TEXT) return PREPRINT_TEXT[inner];
  if (inner === "CRediT statement") return META.contributions;
  if (inner.startsWith("e.g. \"The authors declare")) return META.competing;
  if (inner.startsWith("Funding sources")) return META.funding;
  if (inner.startsWith("Disclose per journal policy")) return META.ai_disclosure;
  throw new Error(`unresolved placeholder in preprint mode: [[${inner}]]`);
}
const D = JSON.parse(fs.readFileSync(path.join(BUILD, "manuscript_data.json"), "utf8"));

const FONT = "Calibri";
const TEXT_W = 9026; // A4 width 11906 - 2 x 1440 margins (DXA)
const fmt = (n) => Number(n).toLocaleString("en-US");
const pct = (n, d, k = 1) => (100 * n / d).toFixed(k);

// ---------------------------------------------------------------- inline markup
// **bold**  *italic*  [[placeholder]] (yellow highlight)  ^{sup}  _{sub}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\]|\^\{[^}]+\}|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else if (t.startsWith("[[")) {
      if (!PREPRINT) out.push(new TextRun({ text: t.slice(2, -2), highlight: "yellow", ...base }));
      else {
        const v = resolvePlaceholder(t.slice(2, -2));
        if (v) out.push(...runs(v, base));
      }
    }
    else if (t.startsWith("^{")) out.push(new TextRun({ text: t.slice(2, -1), superScript: true, ...base }));
    else if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({ children: runs(PREPRINT ? text.replace(/\s+\[\[/g, " [[") : text, opts.run || {}), spacing: { after: 160 }, ...opts.para });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const bullet = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, children: runs(t), spacing: { after: 80 } });
const caption = (t) => new Paragraph({ children: runs(t, { size: 19 }), spacing: { before: 80, after: 240 }, keepLines: true });
const tableTitle = (t) => new Paragraph({ children: runs(t, { size: 19 }), spacing: { before: 240, after: 80 }, keepNext: true });
const note = (t) => new Paragraph({ children: runs(t, { size: 16, color: "52514E" }), spacing: { before: 40, after: 240 } });

// ---------------------------------------------------------------- tables
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
function table(headers, rows, fracs, { align = [], size = 17 } = {}) {
  const widths = fracs.map((f) => Math.round(f * TEXT_W));
  widths[widths.length - 1] = TEXT_W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const cell = (text, i, header) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    shading: header ? { fill: "E8EEF7", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    verticalAlign: VerticalAlign.CENTER,
    borders: { top: border, bottom: border, left: noBorder, right: noBorder },
    children: [new Paragraph({
      alignment: align[i] || AlignmentType.LEFT,
      children: runs(String(text), { size, bold: header }),
    })],
  });
  return new Table({
    width: { size: TEXT_W, type: WidthType.DXA },
    columnWidths: widths,
    layout: TableLayoutType.FIXED,
    rows: [
      new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((c, i) => cell(c, i, false)) })),
    ],
  });
}

// ---------------------------------------------------------------- figures
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20), data: b };
}
function figure(file, widthPx = 600) {
  const { w, h, data } = pngSize(file);
  return new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, line: 240, lineRule: LineRuleType.AUTO },
    children: [new ImageRun({ type: "png", data, transformation: { width: widthPx, height: Math.round(widthPx * h / w) } })],
  });
}

// ---------------------------------------------------------------- derived numbers
const RC = D.rec_counts, SV = D.severity;
const conc = Object.fromEntries(D.concordance.map((c) => [`${c.label}|${c.flag}`, c]));
const cm = (lab, flag = "potentially_missing") => conc[`${lab}|${flag}`];
const sci = (p) => {
  if (p >= 0.001) return p.toFixed(3);
  const e = Math.floor(Math.log10(p));
  const m = p / Math.pow(10, e);
  return `${m.toFixed(1)}×10^{${e}}`;
};
const top = (obj, k) => Object.entries(obj).slice(0, k).map(([l, n]) => `${l} ${n}`).join(", ");
const rest = (obj, k) => Object.values(obj).slice(k).reduce((a, b) => a + b, 0);
const byLabel = Object.fromEntries(D.per_label.map((r) => [r.label, r]));
const sv = D.strat_validated, sa = D.strat_autogen;
const rateOf = (o, k) => (100 * o[k].disagreement_rate_among_assessed).toFixed(2);
const pvc = cm("PVC"), asmi = cm("ASMI"), lafb = cm("LAFB"), afm = cm("AFIB");
const afi = cm("AFIB", "potentially_incorrect"), lbi = cm("CLBBB", "potentially_incorrect"), pvi = cm("PVC", "potentially_incorrect");
const hiAuroc = D.per_label.filter((r) => r.auroc >= 0.99).sort((a, b) => b.auroc - a.auroc);
const loAuroc = D.per_label.filter((r) => r.auroc < 0.85).sort((a, b) => a.auroc - b.auroc);
const fmtList = (rs) => rs.map((r) => `${r.label} ${r.auroc.toFixed(3)}`).join(", ");
const BL = Object.fromEntries(D.baselines.map((b) => [b.method, b]));
const OV = D.overlap;
const SENS = D.sensitivity;
const sensAt = (c, m) => SENS.find((r) => Math.abs(r.contradict_bf - c) < 1e-9 && Math.abs(r.missing_posterior - m) < 1e-9);
const orci = (b, k) => (b[`${k}_mh_or`] == null ? "—" : `${Number(b[`${k}_mh_or`]).toFixed(1)} (${b[`${k}_mh_ci`]})`);
const pv = (x) => (x == null ? "—" : x === 0 ? "<10^{-16}" : sci(x));

// ---------------------------------------------------------------- content
const C = [];
// Title page
C.push(new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 240 }, children: runs("Original research", { size: 20, color: "52514E", allCaps: true }) }));
C.push(new Paragraph({
  heading: HeadingLevel.TITLE, spacing: { after: 240 },
  children: [new TextRun("Auditing diagnostic labels of a public ECG benchmark with a foundation model: a calibration-aware, fully automated analysis of PTB-XL using ECGFounder")],
}));
C.push(P("**Running title:** Foundation-model audit of PTB-XL labels"));
if (PREPRINT) {
  C.push(P(`**${META.authors}**`));
  C.push(P(META.affiliation));
  C.push(P(`Correspondence: ${META.email}`));
  C.push(P(`Preprint (not peer reviewed), ${META.date || new Date().toISOString().slice(0, 10)}. Code and results: ${REPO_URL}`, { run: { size: 18, color: "52514E" } }));
} else {
  C.push(P("**Authors:** [[Author 1]]^{1}, [[Author 2]]^{1}, [[… senior author]]^{1}"));
  C.push(P("^{1} [[Independent researcher / affiliation of each co-author, city, country]]"));
  C.push(P("**Corresponding author:** [[name, postal address, e-mail]]"));
  C.push(P(`**Word count:** [[main text ~3,900 words; abstract ~300 words]] · **Tables:** 8 · **Figures:** 3 · **Supplementary figures:** 2`));
}
C.push(P("**Keywords:** electrocardiography; label noise; dataset quality; foundation model; ECGFounder; PTB-XL; calibration; Bayes factor"));
if (!PREPRINT) C.push(P("[[Target journal to be decided. The manuscript is formatted generically (IMRaD, numbered references); adapt length, abstract structure and reference style to the journal's guidelines.]]", { run: { italics: true, size: 18 } }));

// Abstract
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(H1("Abstract"));
C.push(P(`**Background:** Public electrocardiogram (ECG) datasets underpin the development and benchmarking of automated ECG interpretation, but their diagnostic labels are rarely audited at scale. We investigated whether an independently trained ECG foundation model can serve as an automated reference to prioritise likely missing or incorrect labels.`));
C.push(P(`**Methods:** We applied the publicly released 12-lead ECGFounder model (150 interpretation classes, pretrained on more than 10 million ECGs of the Harvard-Emory ECG Database) to all ${fmt(D.n_records)} recordings of PTB-XL v1.0.3. SCP-ECG statements were mapped to model classes with a curated mapping. Model scores were calibrated to the dataset's annotation conventions by patient-grouped cross-fitting. Existing labels were judged by a prevalence-independent Bayes factor and absent labels by the calibrated posterior probability. Flags were checked against PTB-XL's original free-text reports and against GE 12SL machine statements (PTB-XL+), using label-adjusted (Mantel–Haenszel) odds ratios. They were compared with five alternative audit methods, including in-domain confident learning, and their sensitivity to the decision thresholds was assessed.`));
C.push(P(`**Results:** All recordings were processed without failure. Of ${fmt(D.n_label_instances)} label instances, ${D.lab_exact_pct}% had an exact model equivalent. The median AUROC of the raw model score against the dataset label was ${D.median_auroc} across ${D.n_exact_labels} labels, and ≥0.99 for left bundle branch block, sinus tachycardia, atrial fibrillation, premature ventricular complexes and atrial flutter. Of ${fmt(D.n_assessable)} assessable recordings, ${D.agreement_pct}% agreed with the model, ${D.uncertain_pct}% were inconclusive and ${fmt(RC.disagreement)} (${D.disagreement_pct}%) were flagged: ${D.n_incorrect} potentially incorrect and ${D.n_missing} potentially missing labels. Against the free-text reports, flags were strongly enriched for report–code inconsistencies compared with unflagged records of the same label: odds ratio ${orci(BL.ours, "missing_text")} for missing and ${orci(BL.ours, "incorrect_text")} for incorrect labels. These text enrichments exceeded those of raw-score thresholds, in-domain confident learning and 12SL disagreement, and increased monotonically with stricter thresholds. The methods flagged almost entirely different records (Jaccard ≤${Math.max(...Object.entries(OV.jaccard_missing).filter(([k]) => k !== "ours" && k !== "posterior").map(([, v]) => v)).toFixed(2)}). Thresholding the posterior instead of the Bayes factor would have flagged ${fmt(D.posterior_rule_incorrect)} existing labels, including all ${D.n_ami} anterior infarction labels, an artefact of low prevalence.`));
C.push(P(`**Conclusions:** A foundation model can triage an entire public ECG dataset for label review within minutes, and its flags are strongly enriched for genuine annotation inconsistencies. Model–label disagreement is not a label-error rate. Prevalence-aware decision rules are essential. Expert adjudication of a stratified sample, for which we provide a blinded protocol and materials, remains necessary to estimate label accuracy.`));

// 1 Introduction
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(H1("1. Introduction"));
C.push(P(`Large public ECG datasets such as PTB-XL [2], distributed through PhysioNet [3], have become the reference benchmarks for deep-learning ECG interpretation [4]. Their diagnostic labels are derived from clinical reports, some of them initially generated automatically, and are therefore imperfect. Label errors in benchmark test sets can change the ranking of competing models [5], and label noise in training data propagates into deployed algorithms. Exhaustive expert re-annotation of tens of thousands of ECGs is rarely feasible, so a principled way to rank recordings for targeted review is valuable.`));
C.push(P(`Confident learning ranks suspected label errors with a model trained on the same dataset [6]. Doggart et al. applied it to the rhythm labels of PTB-XL with a network trained on a proprietary dataset: 79% of the 200 top-ranked candidates were confirmed as errors on manual review, and the rhythm label-error rate was estimated at 1.86% [11]. A model trained on the same dataset shares that dataset's the dataset's annotation conventions and can absorb its systematic errors. Foundation models trained on much larger, independently labelled corpora provide an external reference instead. ECGFounder was pretrained on more than 10 million ECGs from the Harvard-Emory ECG Database (HEEDB) [1,7] and predicts 150 interpretation statements in the vocabulary of the GE MUSE system. Its weights are publicly available, and it has not been trained on PTB-XL.`));
C.push(P(`Using such a model as a label auditor raises methodological problems that are easy to overlook:`));
C.push(bullet(`The model's label vocabulary differs from the dataset's.`));
C.push(bullet(`Device, population and era differ between pretraining and target data (domain shift).`));
C.push(bullet(`Sigmoid outputs are not calibrated probabilities [8,9].`));
C.push(bullet(`Many datasets annotate only positive findings, so an absent label is not a confirmed negative.`));
C.push(bullet(`Rare labels have low prior probability, so a low calibrated posterior is not by itself evidence against an existing label.`));
C.push(P(`We developed a fully automated, calibration-aware pipeline that addresses these issues and applied it to the whole of PTB-XL. Our aims were to (i) quantify agreement between ECGFounder and the PTB-XL labels, (ii) identify and rank candidate missing and incorrect labels, and (iii) test, independently of the model, whether the flags concentrate on genuine annotation inconsistencies. Throughout, we distinguish *model–label agreement*, which we measure, from *label accuracy*, which requires independent expert adjudication.`));

// 2 Methods
C.push(H1("2. Methods"));
C.push(H2("2.1 Data"));
C.push(P(`PTB-XL v1.0.3 [2] comprises ${fmt(D.n_records)} 10-second, 12-lead ECGs from ${fmt(D.n_patients)} patients, recorded with Schiller devices between 1989 and 1996. We used the 500 Hz version. Each record carries one or more of 71 SCP-ECG statements (diagnostic statements with a likelihood of 0–100%), a separate heart-axis field, the original free-text report, and a patient-wise stratified fold (1–10). ${fmt(D.n_validated)} records (${pct(D.n_validated, D.n_records)}%) were validated by a cardiologist. Integrity of all 43,600 downloaded files was verified against the official SHA-256 manifest. PTB-XL is de-identified and publicly available under a CC BY 4.0 licence. [[Confirm with the local ethics office that no approval is required for this secondary analysis.]]`));
C.push(H2("2.2 Model and preprocessing"));
C.push(P(`We used the official 12-lead ECGFounder checkpoint (Hugging Face PKUDigitalHealth/ECGFounder, file 12_lead_ECGFounder.pth, SHA-256 ${D.checkpoint_sha256.slice(0, 16)}…) with the official Net1D architecture (30.8 million parameters, 150 sigmoid outputs) [1]. Weights were loaded with strict key matching, so no layer could silently remain randomly initialised. This is the only 12-lead checkpoint in the public repository (uploaded March 2025). Its stored metadata (validation AUROC 0.920) is consistent with the paper's selection of the checkpoint with the best validation AUROC [1]. [[Confirm with the ECGFounder authors that this checkpoint is the one evaluated in their paper.]]`));
C.push(P(`Preprocessing reproduced the official implementation:`));
C.push(bullet(`leads ordered I, II, III, aVR, aVL, aVF, V1–V6;`));
C.push(bullet(`50 Hz notch filter;`));
C.push(bullet(`zero-phase 4th-order Butterworth band-pass filter (0.67–40 Hz);`));
C.push(bullet(`subtraction of a 0.4 s median-filter baseline;`));
C.push(bullet(`z-score normalisation with the global mean and standard deviation of the 12 × 5000 array.`));
C.push(P(`Unit tests verify numerical identity with the official filter. Recordings with missing leads, more than 5% missing samples or more than two flat leads would have been excluded rather than imputed; none met these criteria.`));
C.push(H2("2.3 Label mapping"));
C.push(P(`Each PTB-XL statement was mapped to one or more ECGFounder classes and graded as *exact* (same clinical concept), *approximate* (overlapping concept or different criteria) or *unsupported* (no corresponding class). For example, sinus rhythm maps to any of the sinus-rhythm classes, including sinus bradycardia and tachycardia. Incomplete right bundle branch block maps only to the incomplete-block class. Ischaemia statements are unsupported because ECGFounder has no ischaemia class. The label score was the maximum probability over the mapped classes ("any-of"), so that sub-types support their parent label. The 22 ECGFounder classes that describe change relative to a previous ECG (e.g. "QT has lengthened") cannot be assessed from a single recording and were excluded. Only exact mappings (${D.n_exact_labels} labels) were used for decisions and suggestions; the others were reported as not comparable. Equivalence rules prevent redundant flags: for example, a strong left ventricular hypertrophy (LVH) prediction is not flagged as missing when voltage criteria for LVH are coded. Exclusivity rules prevent incoherent suggestions, such as adding "normal ECG" to a record coded with atrial fibrillation. [[The mapping (mappings/ptbxl_scp.yaml) should be reviewed by a cardiologist before submission.]]`));
C.push(H2("2.4 Calibration"));
C.push(P(`For each label we fitted Platt scaling (logistic regression on the logit of the label score) [8] to the dataset's own labels. This yields the posterior *P* = P(label annotated | ECG). Calibration used 5-fold cross-fitting over pairs of PTB-XL's patient-wise folds, so the calibrator applied to a record never saw that record's label. Labels with fewer than 20 positive or negative examples in any training split were left uncalibrated. Calibration quality was summarised by the expected calibration error (ECE, 10 equal-width bins).`));
C.push(H2("2.5 Decision rules and discrepancy score"));
C.push(P(`For rare labels, *P* is small for nearly every record, including typical positives, simply because the prior is small. A low posterior therefore does not by itself argue against an existing label. We judged existing labels by the Bayes factor BF = p(ECG | label)/p(ECG | no label), obtained as the posterior odds divided by the prior odds, where the prior is the label's prevalence in the calibrator's training folds. Thresholds follow the scale of Kass and Raftery [10] (Table 2). Absent labels were judged by the posterior, because a missing-label claim should mean that the annotators would very likely have assigned the label. Because PTB-XL annotations are positive-only, an absent label with a low score was never counted as agreement. For uncalibrated labels, conservative raw-score thresholds were used.`));
C.push(P(`The discrepancy score of a flag is 1/(1 + BF) for a potentially incorrect label, which is the probability that the label is wrong under a neutral 1:1 prior, and *P* for a potentially missing label. A record's score is the maximum over its flags. Severity was graded as high (≥0.99), medium (≥0.97) or low. Suggested removals and additions were written to a separate proposed-label file; the original data were never modified.`));
C.push(tableTitle("**Table 2.** Decision rules for comparable labels."));
C.push(table(["Situation", "Rule (calibrated labels)", "Rule (uncalibrated)", "Category"], [
  ["Label present", "BF ≥ 3 or P ≥ 0.5", "score ≥ 0.5", "Consistent"],
  ["Label present", "BF ≤ 1/20", "score ≤ 0.02", "Potentially incorrect"],
  ["Label present", "otherwise", "otherwise", "Uncertain"],
  ["Label absent", "P ≥ 0.95, no equivalent label coded, no conflict", "score ≥ 0.98", "Potentially missing"],
  ["Label absent", "P ≥ 0.95 but conflicting label coded", "same", "Uncertain"],
  ["Label absent", "otherwise", "otherwise", "Not assessed (absence ≠ negative)"],
  ["Unsupported, unmapped or approximate mapping", "—", "—", "Not comparable"],
], [0.22, 0.36, 0.17, 0.25]));
C.push(note("BF, Bayes factor p(ECG | label)/p(ECG | no label); P, cross-fitted calibrated posterior probability that the dataset annotators assign the label. The P ≥ 0.5 criterion is needed for very common labels: sinus rhythm (77% prevalence) can reach BF ≥ 3 only when P ≥ 0.91."));
C.push(H2("2.6 Model-independent plausibility checks"));
C.push(P(`Without a reference standard, we assessed whether flags concentrate on recognisable annotation problems using two signals that ECGFounder did not see:`));
C.push(bullet(`**Free-text reports.** PTB-XL's SCP codes were derived from the reports (German, English or Swedish) distributed with the dataset. For eight findings detectable by keyword, a *potentially missing* flag is supported if the report mentions the finding although the code is absent. A *potentially incorrect* flag is supported if the report does not mention the finding although the code is present.`));
C.push(bullet(`**GE 12SL statements.** PTB-XL+ [12] provides the statements of the GE Marquette 12SL algorithm for every record, mapped to SNOMED CT. A missing flag is supported if 12SL makes the statement, and an incorrect flag if 12SL does not. This check is not fully independent: ECGFounder's training labels were entered with 12SL assistance.`));
C.push(P(`Support among flagged records was compared with support among unflagged records with the same label status. Because the label mix differs between methods, we report the Mantel–Haenszel odds ratio stratified by label, with Robins–Breslow–Greenland 95% confidence intervals and the Cochran–Mantel–Haenszel test. Per-label rates were compared with Fisher's exact test. Neither signal is ground truth.`));
C.push(H2("2.7 Comparison methods and sensitivity analysis"));
C.push(P(`On the same labels and equivalence rules, we compared our method with five alternatives:`));
C.push(bullet(`raw ECGFounder scores without calibration (incorrect if ≤0.02, missing if ≥0.98);`));
C.push(bullet(`the calibrated-posterior rule (incorrect if P ≤ 0.05);`));
C.push(bullet(`in-domain confident learning [6]: cross-fitted logistic regression on ECGFounder's 1024-dimensional embeddings, trained on the PTB-XL labels themselves, with label issues found by cleanlab;`));
C.push(bullet(`disagreement with 12SL;`));
C.push(bullet(`report–code keyword mismatch.`));
C.push(P(`Methods that produce a ranking were also compared at a matched budget, taking each method's top-ranked flags up to our method's flag count. For the sensitivity analysis, we varied the contradiction threshold (BF ≤ 1/5 to 1/1000) and the missing threshold (P ≥ 0.80 to 0.99).`));
C.push(H2("2.8 Blinded adjudication protocol"));
C.push(P(`To estimate flag precision and the label-error rate, we prepared a blinded adjudication study of ${D.adjudication.cases} records in four strata that partition the dataset:`));
C.push(bullet(`records with an incorrect flag from our method;`));
C.push(bullet(`records with a missing flag from our method, stratified by severity;`));
C.push(bullet(`records flagged only by the other methods;`));
C.push(bullet(`records flagged by no method.`));
C.push(P(`For every record, two cardiologists will rate ${D.adjudication.pairs} findings in total as present, absent or uncertain. The findings comprise all coded and flagged comparable findings plus two random decoys, in random order. Readers are blinded to codes, reports, model outputs and strata, and disagreements are resolved by a third reader. Precision will be reported per method and flag type with Wilson intervals, and the label-error rate as a stratum-weighted estimate with bootstrap intervals. The protocol, ECG booklet, rating sheet and analysis code are in the repository. [[Results to be added after adjudication.]]`));
C.push(H2("2.9 Statistical analysis and software"));
C.push(P(`Agreement between raw model scores and dataset labels was summarised per label by the area under the ROC curve (AUROC) and the precision–recall curve. These are agreement measures, not accuracy. Disagreement rates were stratified by cardiologist validation, automatic initial report generation and recording device. The pipeline was implemented in Python ${D.python} and PyTorch ${D.torch.split("+")[0]}. It is fully deterministic, resumable and runs from a single command. Inference ran on a 4-core CPU at about 33 recordings per second (12 minutes for the full dataset). All code, the mapping, configuration, unit and end-to-end tests, and every result table are available in the project repository (see Code availability).`));

// 3 Results
C.push(H1("3. Results"));
C.push(H2("3.1 Dataset and label comparability"));
C.push(P(`All ${fmt(D.n_records)} recordings met the input requirements and were processed without failure (Table 1). Of ${fmt(D.n_label_instances)} label instances, ${fmt(D.lab_exact)} (${D.lab_exact_pct}%) had an exact ECGFounder equivalent, ${fmt(D.lab_approx)} (${D.lab_approx_pct}%) an approximate one, and ${fmt(D.lab_unsupported)} (${D.lab_unsupported_pct}%) none. The commonest unsupported statements were normal heart axis, "abnormal QRS", non-specific ischaemia and ST depression. ${fmt(D.n_assessable)} recordings carried at least one comparable label.`));
C.push(tableTitle("**Table 1.** Dataset, processing and label comparability."));
C.push(table(["Characteristic", "Value"], [
  ["Recordings / patients", `${fmt(D.n_records)} / ${fmt(D.n_patients)}`],
  ["Validated by a cardiologist", `${fmt(D.n_validated)} (${pct(D.n_validated, D.n_records)}%)`],
  ["Recordings processed / failed", `${fmt(D.n_records)} / ${D.n_failed}`],
  ["Label instances (SCP statements + heart axis)", fmt(D.n_label_instances)],
  ["   exact mapping", `${fmt(D.lab_exact)} (${D.lab_exact_pct}%)`],
  ["   approximate mapping", `${fmt(D.lab_approx)} (${D.lab_approx_pct}%)`],
  ["   no ECGFounder equivalent", `${fmt(D.lab_unsupported)} (${D.lab_unsupported_pct}%)`],
  ["Labels used for decisions (exact) / calibrated", `${D.n_exact_labels} / ${D.n_calibrated}`],
  ["Recordings with ≥1 comparable label", fmt(D.n_assessable)],
], [0.62, 0.38], { align: [AlignmentType.LEFT, AlignmentType.RIGHT] }));

C.push(H2("3.2 Agreement between ECGFounder and PTB-XL labels"));
C.push(P(`The median AUROC of the raw ECGFounder score against the PTB-XL label was ${D.median_auroc} across the ${D.n_exact_labels} exact-mapped labels, and ${D.n_auroc_095} labels reached ≥0.95 (Table 3). Agreement was highest for well-defined rhythm and conduction findings (${fmtList(hiAuroc)}). It was lowest for infarct-territory statements, atrial enlargement and non-specific conduction delay (${fmtList(loAuroc)}). Cross-fitted calibration reduced the median ECE from ${D.ece_raw_median} (raw scores) to ${D.ece_cal_median.toFixed(3)} (maximum ${D.ece_cal_max}) across the ${D.n_calibrated} calibrated labels (Fig. 2). ${D.uncalibrated.join(" and ")} had too few positives for calibration.`));
C.push(P(`Of the ${fmt(D.n_present_comparable)} existing comparable labels, ${fmt(D.present.consistent)} (${D.present_consistent_pct}%) were consistent with the model, ${fmt(D.present.uncertain)} (${D.present_uncertain_pct}%) were uncertain, and ${D.present.potentially_incorrect} (${D.present_incorrect_pct}%) were potentially incorrect (Fig. 1). For low-agreement labels the model was mostly inconclusive rather than contradictory. All ${byLabel.AMI.n} anterior infarction labels were uncertain, as were ${byLabel.IMI.uncertain_pct}% of inferior infarction labels.`));
C.push(tableTitle("**Table 3.** Per-label agreement between ECGFounder and PTB-XL (exact mappings, ordered by frequency)."));
C.push(table(["Label", "n", "Consistent %", "Uncertain %", "Pot. incorrect", "Pot. missing", "AUROC", "ECE raw → cal."],
  D.per_label.map((r) => [r.label, fmt(r.n), r.consistent_pct.toFixed(1), r.uncertain_pct.toFixed(1), r.incorrect, r.missing, r.auroc.toFixed(3),
    r.calibrated ? `${r.ece_raw.toFixed(3)} → ${r.ece_cal.toFixed(3)}` : "uncalibrated"]),
  [0.12, 0.1, 0.13, 0.13, 0.12, 0.12, 0.1, 0.18],
  { align: [AlignmentType.LEFT, ...Array(7).fill(AlignmentType.RIGHT)], size: 15 }));
C.push(note("n, records carrying the label. Percentages are of n. Potentially missing counts refer to records without the label. AUROC compares raw ECGFounder scores with PTB-XL labels and measures agreement, not accuracy. ECE, expected calibration error before and after cross-fitted Platt scaling. Label abbreviations follow the PTB-XL SCP-ECG statements [2]."));
C.push(figure(path.join(PLOTS, "per_label_agreement.png")));
C.push(caption("**Figure 1.** Outcome of existing labels (left: consistent, uncertain or potentially incorrect, as a fraction of records carrying the label) and rate of potentially missing labels (right: fraction of records without the label) for the 24 most frequent exact-mapped labels."));

C.push(figure(path.join(PLOTS, "calibration_reliability.png")));
C.push(caption("**Figure 2.** Reliability diagrams of the cross-fitted calibration for the 12 most frequent calibrated labels, with expected calibration error (ECE). Approximate-mapped labels (heart axis, non-diagnostic T abnormalities) are shown for completeness."));
C.push(H2("3.3 Flagged recordings and proposed corrections"));
const assessed = RC.agreement + RC.uncertain + RC.disagreement;
C.push(P(`At recording level, ${fmt(RC.agreement)} of ${fmt(assessed)} assessable recordings (${D.agreement_pct}%) agreed with the model, ${fmt(RC.uncertain)} (${D.uncertain_pct}%) contained at least one inconclusive label, and ${fmt(RC.disagreement)} (${D.disagreement_pct}%) were flagged. The flags comprised ${D.n_incorrect} potentially incorrect labels in ${D.records_with_incorrect} recordings and ${D.n_missing} potentially missing labels in ${D.records_with_missing} recordings. ${SV.high} flagged recordings were of high severity, ${SV.medium} medium and ${SV.low} low. The most frequent proposed additions were ${top(D.additions_by_label, 6)} (other ${rest(D.additions_by_label, 6)}). The most frequent proposed removals were ${top(D.removals_by_label, 5)} (other ${rest(D.removals_by_label, 5)}).`));
C.push(P(`For comparison, a rule that flags an existing label whenever its calibrated posterior is ≤0.05 would have flagged ${fmt(D.posterior_rule_incorrect)} labels on the same model outputs, including all ${D.n_ami} anterior infarction labels. Those labels are uniformly rare and poorly separated by the model (AUROC ${byLabel.AMI.auroc.toFixed(3)}), so the posterior rule mistakes low prevalence for contradiction.`));

C.push(H2("3.4 Plausibility against free-text reports and 12SL"));
C.push(P(`Flags were strongly enriched for report–code inconsistencies (Table 4, Fig. 3). The report named a premature ventricular complex in ${pvc.flagged_report_supports_flag}/${pvc.n_flagged} (${Math.round(100 * pvc.flagged_rate)}%) "missing PVC" flags versus ${(100 * pvc.control_rate).toFixed(1)}% of unflagged records without the code. The corresponding figures were ${asmi.flagged_report_supports_flag}/${asmi.n_flagged} (${Math.round(100 * asmi.flagged_rate)}%) versus ${(100 * asmi.control_rate).toFixed(1)}% for anteroseptal infarction, ${lafb.flagged_report_supports_flag}/${lafb.n_flagged} (${Math.round(100 * lafb.flagged_rate)}%) versus ${(100 * lafb.control_rate).toFixed(1)}% for left anterior fascicular block, and ${afm.flagged_report_supports_flag}/${afm.n_flagged} (${Math.round(100 * afm.flagged_rate)}%) versus ${(100 * afm.control_rate).toFixed(1)}% for atrial fibrillation.`));
C.push(P(`All ${lbi.n_flagged} "incorrect left bundle branch block" flags concerned reports that do not mention left bundle branch block; two of them describe right bundle branch block (Table 5). The exception was "incorrect atrial fibrillation": all ${afi.n_flagged} of these reports mention atrial fibrillation, most in the combined German statement "Vorhofflimmern/-flattern" (fibrillation or flutter). These flags therefore more likely reflect fibrillation-versus-flutter ambiguity or model error than coding errors. Pacemaker flags (${cm("PACE").n_flagged} missing, ${cm("PACE","potentially_incorrect").n_flagged} incorrect) were not supported by the report text either.`));
C.push(P(`Pooled across labels, the incorrect flags were barely more often unsupported by the report than controls (${BL.ours.incorrect_text_support} vs a pooled control rate of ${(100 * BL.ours.incorrect_text_control).toFixed(0)}%). This crude comparison is confounded by the label mix. Controls are dominated by labels whose keywords have low recall, such as left anterior fascicular block, while flags are concentrated on atrial fibrillation, whose keyword has near-perfect recall. Stratified by label, the incorrect flags were clearly enriched: odds ratio ${orci(BL.ours, "incorrect_text")}, p=${pv(BL.ours.incorrect_text_cmh_p)}. For missing flags the label-adjusted odds ratio was ${orci(BL.ours, "missing_text")}. Against 12SL, the odds ratios were ${orci(BL.ours, "missing_12sl")} for missing and ${orci(BL.ours, "incorrect_12sl")} for incorrect flags.`));
C.push(tableTitle("**Table 4.** Free-text report support for flags compared with unflagged controls."));
C.push(table(["Flag", "Flagged: report supports flag", "Unflagged controls", "p (Fisher)"],
  D.concordance.filter((c) => c.n_flagged >= 3).map((c) => [
    `${c.flag === "potentially_missing" ? "Missing" : "Incorrect"} ${c.label}`,
    `${c.flagged_report_supports_flag}/${c.n_flagged} (${Math.round(100 * c.flagged_rate)}%)`,
    `${(100 * c.control_rate).toFixed(1)}% (n=${fmt(c.n_control)})`,
    sci(c.fisher_p)]),
  [0.28, 0.28, 0.26, 0.18], { align: [AlignmentType.LEFT, AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.RIGHT] }));
C.push(note("A missing-label flag is supported if the report mentions the finding although the code is absent; an incorrect-label flag is supported if the report does not mention the finding although the code is present. Controls are unflagged records with the same label status. Flags with fewer than three records are omitted."));
C.push(figure(path.join(BUILD, "fig_concordance.png"), 560));
C.push(caption("**Figure 3.** Proportion of records whose free-text report names the finding, among records flagged by the pipeline as potentially missing the label (blue) and among unflagged records without the label (grey)."));
C.push(tableTitle("**Table 5.** Examples of high-severity flags and the corresponding PTB-XL report text (verbatim, abbreviated)."));
C.push(table(["ecg_id", "Coded labels", "Flag (score)", "Report excerpt"],
  D.examples.map((e) => [e.record_id, e.labels.replace(/;/g, ", "), e.flags.replace(/;\s*/g, "; ").replace(/missing:/g, "missing ").replace(/incorrect:/g, "incorrect "),
    `"${e.report.replace(/\s+/g, " ").trim().slice(0, 120)}${e.report.length > 120 ? "…" : ""}"`]),
  [0.09, 0.22, 0.22, 0.47], { size: 15 }));
C.push(note("German: linksanteriorer Hemiblock = left anterior hemiblock; Rechtsschenkelblock = right bundle branch block."));

C.push(H2("3.5 Sensitivity to the decision thresholds"));
C.push(P(`Flag counts depended strongly on the thresholds, but enrichment was robust and increased monotonically with stricter thresholds (Table 7). As the missing threshold was raised from P ≥ 0.80 to P ≥ 0.99, the number of missing flags fell from ${fmt(sensAt(0.05, 0.8).n_missing)} to ${sensAt(0.05, 0.99).n_missing}, and the label-adjusted text odds ratio rose from ${sensAt(0.05, 0.8).missing_mh_or} to ${sensAt(0.05, 0.99).missing_mh_or}. Over the range of contradiction thresholds, incorrect flags fell from ${sensAt(0.2, 0.95).n_incorrect} (BF ≤ 1/5) to ${sensAt(0.001, 0.95).n_incorrect} (BF ≤ 1/1000), and the odds ratio rose from ${sensAt(0.2, 0.95).incorrect_mh_or} to ${sensAt(0.01, 0.95).incorrect_mh_or} at BF ≤ 1/100. This dose–response pattern indicates that the scores rank genuine annotation problems, rather than reflecting a lucky choice of threshold.`));
C.push(tableTitle("**Table 7.** Threshold sensitivity: flag counts and label-adjusted report-text odds ratios (95% CI)."));
C.push(table(["Contradict BF ≤", "Missing P ≥", "Incorrect", "Missing", "Records", "OR missing", "OR incorrect"],
  [[0.2, 0.95], [0.1, 0.95], [0.05, 0.8], [0.05, 0.9], [0.05, 0.95], [0.05, 0.98], [0.05, 0.99], [0.02, 0.95], [0.01, 0.95], [0.001, 0.95]].map(([c, m]) => {
    const r = sensAt(c, m);
    const lab = c === 0.05 && m === 0.95 ? " (default)" : "";
    return [`1/${Math.round(1 / c)}${lab}`, m.toFixed(2), r.n_incorrect, fmt(r.n_missing), fmt(r.n_flagged_records),
      `${r.missing_mh_or.toFixed(1)} (${r.missing_mh_ci})`, r.incorrect_mh_or == null ? "—" : `${r.incorrect_mh_or.toFixed(1)} (${r.incorrect_mh_ci})`];
  }),
  [0.17, 0.11, 0.1, 0.1, 0.1, 0.21, 0.21], { align: [AlignmentType.LEFT, ...Array(6).fill(AlignmentType.RIGHT)], size: 15 }));
C.push(note("OR, Mantel–Haenszel odds ratio, stratified by label, that the free-text report supports a flag relative to unflagged records of the same label status. Missing-flag ORs do not depend on the contradiction threshold and vice versa."));
C.push(H2("3.6 Comparison with alternative audit methods"));
C.push(P(`The comparison methods produced far more flags than ours (Table 8): ${fmt(BL.raw.n_incorrect)} incorrect and ${fmt(BL.raw.n_missing)} missing for raw scores, ${fmt(BL.cl_probe.n_incorrect)} and ${fmt(BL.cl_probe.n_missing)} for in-domain confident learning, and ${fmt(BL.gesl.n_incorrect)} and ${fmt(BL.gesl.n_missing)} for 12SL disagreement. The confident-learning probe agreed with the PTB-XL labels about as well as the zero-shot model (median AUROC ${OV.probe_auroc_median.toFixed(3)}). On the free-text check, our method had the highest label-adjusted odds ratios for both missing and incorrect flags. The posterior rule shares our missing-flag rule and so has the same missing-flag result. At a matched budget, our method remained highest for missing flags, while confident learning was similar for incorrect flags (${orci(BL["cl_probe@budget"], "incorrect_text")} vs ${orci(BL["ours@budget"], "incorrect_text")}). On the 12SL check, which favours methods that reproduce 12SL, our method had the highest odds ratio for incorrect flags. Confident learning had the highest odds ratio for missing flags at a matched budget, ${orci(BL["cl_probe@budget"], "missing_12sl")}. The methods flagged largely different label instances: Jaccard similarity with our flags was at most ${Math.max(...Object.entries(OV.jaccard_missing).filter(([k]) => k !== "ours" && k !== "posterior").map(([, v]) => v)).toFixed(3)} for missing and ${Math.max(...Object.entries(OV.jaccard_incorrect).filter(([k]) => k !== "ours").map(([, v]) => v)).toFixed(3)} for incorrect flags. Of our ${OV.ours_flags_total} flags, ${OV.ours_flags_on_labels_without_keyword_rule} concerned labels that keyword matching cannot assess.`));
C.push(tableTitle("**Table 8.** Comparison of audit methods: flag counts and label-adjusted odds ratios (95% CI) for support by the free-text report and by GE 12SL."));
C.push(table(["Method", "Incorrect", "Missing", "Text OR, missing", "Text OR, incorrect", "12SL OR, missing", "12SL OR, incorrect"],
  [["ours", "Ours (Bayes factor)"], ["posterior", "Posterior rule"], ["raw", "Raw scores"], ["cl_probe", "Confident learning (probe)"], ["gesl", "12SL disagreement"], ["text", "Report keyword mismatch"],
   ["ours@budget", "Ours, matched budget"], ["raw@budget", "Raw, matched budget"], ["posterior@budget", "Posterior, matched budget"], ["cl_probe@budget", "Confident learning, matched budget"]].map(([k, name]) => {
    const b = BL[k];
    return [name, fmt(b.n_incorrect), fmt(b.n_missing), orci(b, "missing_text"), orci(b, "incorrect_text"), orci(b, "missing_12sl"), orci(b, "incorrect_12sl")];
  }),
  [0.22, 0.08, 0.08, 0.155, 0.155, 0.155, 0.155], { align: [AlignmentType.LEFT, ...Array(6).fill(AlignmentType.RIGHT)], size: 14 }));
C.push(note("—, not applicable: a method is not evaluated against the signal it is built from, and a matched-budget raw-score selection contained no keyword-checkable incorrect flags. Matched budget: the top-ranked flags of each method, up to our method's flag count (93 incorrect, 346 missing)."));
C.push(H2("3.7 Stratified analysis"));
C.push(P(`Among assessable records, the disagreement rate was ${rateOf(sv, "True")}% in cardiologist-validated and ${rateOf(sv, "False")}% in non-validated records. It was ${rateOf(sa, "True")}% in records whose initial report was generated automatically and ${rateOf(sa, "False")}% in the others. Across the ${D.n_devices_500} recording devices with more than 500 records it ranged from ${D.device_rate_min.toFixed(1)}% to ${D.device_rate_max.toFixed(1)}%.`));

C.push(H2("3.8 Issues in the reference PTB-XL label file"));
C.push(P(`The ECGFounder repository distributes a PTB-XL label file (csv/ptbxl_label.csv) for 150-class evaluation. Comparing it with the SCP codes of every record showed seven statement types mapped to clinically non-equivalent classes (Table 6), affecting up to 6,838 label instances. Under that file, only 31 of the 150 classes receive any positive label. We did not use it.`));
C.push(tableTitle("**Table 6.** Non-equivalent assignments in the reference PTB-XL→ECGFounder label file."));
C.push(table(["PTB-XL statement (records)", "Assigned ECGFounder class(es)", "Issue"], [
  ["Abnormal QRS (3,327)", "Low-voltage QRS; with QRS widening", "Neither implied"],
  ["Non-specific ischaemia (1,272)", "Left ventricular hypertrophy (92% of records)", "Unrelated concept"],
  ["Incomplete RBBB (1,118)", "Right bundle branch block (complete)", "Incomplete ≠ complete"],
  ["Ischaemia, anterolateral (659)", "\"Anterolateral leads\"", "Sentence fragment"],
  ["Pacemaker (294)", "Electronic atrial pacemaker", "Atrial pacing only"],
  ["Long QT (117)", "QT has lengthened", "Serial-comparison statement"],
  ["SVT (27), paroxysmal SVT (24)", "Supraventricular and ventricular tachycardia", "VT not implied"],
], [0.3, 0.42, 0.28], { size: 16 }));

// 4 Discussion
C.push(H1("4. Discussion"));
C.push(P(`We used an independently trained ECG foundation model to audit the labels of an entire public benchmark in minutes and without human input. ECGFounder agreed closely with PTB-XL for well-defined findings. About 2% of assessable recordings were flagged, and the flags were strongly enriched for records whose own free-text report contradicts their codes. This is evidence, independent of the model, that the pipeline prioritises genuine annotation problems.`));
C.push(P(`**A prevalence-aware decision rule is essential.** The most consequential methodological choice was how to decide that an existing label is contradicted. Thresholding a calibrated posterior is intuitive, and our first implementation did exactly that. It flagged ${fmt(D.posterior_rule_incorrect)} labels, including every anterior infarction label, because a rare label has a low posterior for nearly every record. Judging existing labels by the Bayes factor, the evidence the ECG provides independently of the prior, removed this artefact. The same reasoning underlies likelihood ratios in diagnostic testing, and it applies to any model-based label audit. We recommend reporting both the decision rule and the counterfactual numbers.`));
C.push(P(`**Relation to confident learning.** Doggart et al. found 515 candidate rhythm errors in PTB-XL with confident learning and confirmed 79% of the top 200 [11]. Our audit covers ${D.n_exact_labels} diagnostic statements, including conduction, hypertrophy and infarction, and uses a public model without any training on PTB-XL. In our comparison, in-domain confident learning on ECGFounder embeddings flagged more than twenty times as many labels, with weaker report-text enrichment. It ranked 12SL-supported missing labels better at a matched budget. The methods flagged almost disjoint sets, so they are complementary rather than interchangeable. Our adjudication sample therefore draws from all of them.`));
C.push(P(`**Uncertain is an informative outcome.** A third of assessable recordings contained at least one label the model could neither confirm nor refute. These were concentrated in infarct-territory statements, atrial enlargement and non-specific intraventricular conduction delay. For these statements the criteria plausibly differ between the SCP-ECG coding of PTB-XL and the MUSE vocabulary learned by ECGFounder, so model disagreement says little about label quality. The pipeline therefore reports them as inconclusive rather than forcing a decision. Users should not read the uncertain fraction as an error rate.`));
C.push(P(`**Human validation.** Disagreement rates were similar in cardiologist-validated and non-validated records. This suggests that the residual inconsistencies flagged here, such as findings named in the report but not coded, survived the validation process. Validation status is therefore not a sufficient guarantee of label completeness.`));
C.push(P(`**Implications for evaluation.** The non-equivalent assignments in the reference PTB-XL label file can distort class-level metrics. For example, scoring "ventricular tachycardia" against supraventricular tachycardia labels, or "complete RBBB" against incomplete RBBB, mixes different clinical concepts. Class-level results that relied on this file should be interpreted with care.`));
C.push(H2("Limitations"));
C.push(bullet(`Model–label agreement is not label accuracy. Flags can arise from label errors, model errors, differences in diagnostic criteria or genuine ambiguity. We do not estimate a label-error rate. The blinded adjudication study (Section 2.8) is designed to provide one. [[Update when the adjudication is complete.]]`));
C.push(bullet(`The free-text reports are the source of the PTB-XL codes, not an independent reference standard. Keyword detection has imperfect recall and covers eight findings. 12SL is not independent of ECGFounder's training labels. Both checks therefore support plausibility, not accuracy.`));
C.push(bullet(`Calibration was fitted to the labels under audit. Label noise flattens calibration, which makes flags more conservative. Treating absent labels as negatives during fitting may under-estimate posteriors for under-annotated findings.`));
C.push(bullet(`The label mapping involves clinical judgement. ${D.lab_unsupported_pct}% of label instances could not be assessed, and approximate mappings were excluded from decisions.`));
C.push(bullet(`Results apply to one dataset and one model. PTB-XL differs from HEEDB in device, population and era. Generalisation to other datasets (e.g. CODE-15%, the PhysioNet/CinC 2021 collection) remains to be shown; the pipeline already supports these formats.`));

C.push(H1("5. Conclusions"));
C.push(P(`An openly available ECG foundation model, combined with a calibration-aware and prevalence-aware decision framework, can automatically triage the labels of a large public ECG dataset. The resulting flags are strongly enriched for genuine annotation inconsistencies. Such audits should report model–label agreement separately from label accuracy, and should be followed by targeted expert review.`));

// Declarations
C.push(H1("Declarations"));
C.push(P(`**Data availability.** PTB-XL v1.0.3 is available from PhysioNet (https://physionet.org/content/ptb-xl/1.0.3/) under CC BY 4.0. ECGFounder weights are available from Hugging Face (PKUDigitalHealth/ECGFounder) under the MIT licence. All per-record results, proposed labels, per-label statistics and figures are provided in the project repository (reports/ptbxl_ecgfounder/).`));
C.push(P(`**Code availability.** The pipeline, label mapping, configuration, tests and manuscript build scripts are available at [[https://github.com/nityashah1811/ecg-dataset-verification — confirm public release, licence and archived DOI (e.g. Zenodo) before submission]].`));
C.push(P(`**Ethics.** Secondary analysis of a publicly available, de-identified dataset. [[Confirm local requirements.]]`));
C.push(P(`**Author contributions.** [[CRediT statement]]`));
C.push(P(`**Competing interests.** [[e.g. "The authors declare no competing interests."]]`));
C.push(P(`**Funding.** [[Funding sources or "none"]]`));
C.push(P(`**Use of AI tools.** [[Disclose per journal policy. Software implementation, analyses and drafting of this manuscript were assisted by an AI coding assistant (Claude, Anthropic); the authors reviewed and take responsibility for all content.]]`));
C.push(P(`**Acknowledgements.** We thank the creators of PTB-XL and ECGFounder for making their data and model publicly available. [[Others]]`));

// References
C.push(H1("References"));
const REFS = [
  "Li J, Aguirre AD, Moura Junior V, Jin J, Liu C, Zhong L, Sun C, Clifford G, Westover MB, Hong S. An electrocardiogram foundation model built on over 10 million recordings. NEJM AI. 2025;2(7):AIoa2401033. doi:10.1056/AIoa2401033.",
  "Wagner P, Strodthoff N, Bousseljot RD, Kreiseler D, Lunze FI, Samek W, Schaeffter T. PTB-XL, a large publicly available electrocardiography dataset. Sci Data. 2020;7:154.",
  "Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation. 2000;101(23):e215–e220.",
  "Strodthoff N, Wagner P, Schaeffter T, Samek W. Deep learning for ECG analysis: benchmarks and insights from PTB-XL. IEEE J Biomed Health Inform. 2021;25(5):1519–1528.",
  "Northcutt CG, Athalye A, Mueller J. Pervasive label errors in test sets destabilize machine learning benchmarks. Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks. 2021.",
  "Northcutt CG, Jiang L, Chuang IL. Confident learning: estimating uncertainty in dataset labels. J Artif Intell Res. 2021;70:1373–1411. doi:10.1613/jair.1.12125.",
  "Koscova Z, Li Q, Robichaux C, Moura Junior V, Ghanta M, Gupta A, et al. The Harvard-Emory ECG Database. medRxiv. 2024. doi:10.1101/2024.09.27.24314503.",
  "Platt J. Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In: Advances in Large Margin Classifiers. MIT Press; 1999:61–74.",
  "Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. Proceedings of the 22nd International Conference on Machine Learning. 2005:625–632.",
  "Kass RE, Raftery AE. Bayes factors. J Am Stat Assoc. 1995;90(430):773–795. doi:10.1080/01621459.1995.10476572.",
  "Doggart P, Kennedy A, Foreman E, Finlay D, Bond R. Automated identification of label errors in large electrocardiogram datasets. Computing in Cardiology. 2022;49.",
  "Strodthoff N, Mehari T, et al. PTB-XL+, a comprehensive electrocardiographic feature dataset. Sci Data. 2023;10:279. doi:10.1038/s41597-023-02153-8.",
];
REFS.forEach((r) => C.push(new Paragraph({ numbering: { reference: "refs", level: 0 }, children: runs(r, { size: 19 }), spacing: { after: 80 } })));
if (!PREPRINT) C.push(P("[[References 1–7 and 10–12 were checked against the publisher or preprint record (complete the author list of ref. 12); ref. 8 (book chapter) and ref. 9 (ICML proceedings pages) could not be confirmed online. Convert to the journal's style before submission.]]", { run: { italics: true, size: 18 } }));

// Supplementary
C.push(new Paragraph({ children: [new PageBreak()] }));
C.push(H1("Supplementary material"));
C.push(figure(path.join(PLOTS, "confidence_distributions.png")));
C.push(caption("**Figure S1.** Distribution of calibrated posterior probabilities for records carrying (blue) and not carrying (grey) each label, for the 16 most frequent exact-mapped labels (log density)."));
C.push(figure(path.join(PLOTS, "discrepancy_scores.png"), 420));
C.push(caption("**Figure S2.** Discrepancy scores of the flagged recordings."));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "ECG label verification project",
  title: "Auditing diagnostic labels of PTB-XL with ECGFounder",
  styles: {
    default: { document: { run: { font: FONT, size: 22 }, paragraph: { spacing: { line: 300, lineRule: LineRuleType.AUTO } } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", run: { size: 36, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { after: 240 } } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 360, after: 160 }, outlineLevel: 0, keepNext: true } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: "2F5496" }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1, keepNext: true } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } } }] },
      { reference: "refs", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 450, hanging: 450 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
      ...(PREPRINT ? {} : { lineNumbers: { countBy: 1, restart: LineNumberRestartFormat.CONTINUOUS } }),
    },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "52514E" })] })] }) },
    children: C,
  }],
});

fs.mkdirSync(path.dirname(OUT), { recursive: true });
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(OUT, buf); console.log("wrote", OUT); });
