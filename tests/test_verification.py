import copy

import numpy as np
import pytest

from ecg_verify.config import DEFAULTS
from ecg_verify.ecgfounder import CLASS_INDEX
from ecg_verify.loaders import RecordRef
from ecg_verify.mapping import load_mapping
from ecg_verify.verification import verify

MAPPING = """
name: toy
labels:
  SR: {model_classes: [SINUS RHYTHM, SINUS BRADYCARDIA], quality: exact}
  AF: {model_classes: [ATRIAL FIBRILLATION], quality: exact}
  LBBB: {model_classes: [LEFT BUNDLE BRANCH BLOCK], quality: exact, covered_by: [CLBBB]}
  CLBBB: {model_classes: [LEFT BUNDLE BRANCH BLOCK], quality: exact, covered_by: [LBBB]}
  NORM: {model_classes: [NORMAL ECG], quality: exact, exclusive_with: ["*"], compatible_with: [SR]}
  STT: {model_classes: [NONSPECIFIC ST ABNORMALITY], quality: approximate}
  X: {unsupported: no class}
"""


@pytest.fixture
def mapping(tmp_path):
    p = tmp_path / "toy.yaml"
    p.write_text(MAPPING)
    return load_mapping(str(p))


def cfg(**verification):
    c = copy.deepcopy(DEFAULTS)
    c["verification"]["calibration"]["method"] = "none"  # raw-score thresholds 0.5/0.02/0.98
    c["verification"].update(verification)
    return c


def make(records):
    """records: list of (labels, {model class: prob}); every other class gets 0.001."""
    refs, probs = [], np.full((len(records), 150), 0.001)
    for i, (labels, p) in enumerate(records):
        refs.append(RecordRef(record_id=f"r{i}", labels={l: 1.0 for l in labels}, source=""))
        for c, v in p.items():
            probs[i, CLASS_INDEX[c]] = v
    return refs, probs


def status(res, rid, label):
    a = res.assessments
    row = a[(a.record_id == rid) & (a.label == label)]
    return None if row.empty else row.iloc[0]["status"]


def record(res, rid):
    return res.records.set_index("record_id").loc[rid]


CASES = [
    (["SR"], {"SINUS RHYTHM": 0.9}),                                            # r0 consistent
    (["AF"], {"ATRIAL FIBRILLATION": 0.01, "SINUS RHYTHM": 0.995}),             # r1 wrong label
    (["SR", "STT"], {"SINUS RHYTHM": 0.3}),                                     # r2 uncertain
    (["SR", "FOO", "X"], {"SINUS BRADYCARDIA": 0.97}),                          # r3 unmapped/unsupp.
    (["SR", "CLBBB"], {"SINUS RHYTHM": 0.9, "LEFT BUNDLE BRANCH BLOCK": 0.99}),  # r4 covered
    (["SR", "AF"], {"SINUS RHYTHM": 0.9, "ATRIAL FIBRILLATION": 0.9,
                    "NORMAL ECG": 0.995}),                                       # r5 conflict
    ([], {"ATRIAL FIBRILLATION": 0.99}),                                        # r6 empty labels
    (["FOO"], {}),                                                              # r7 nothing comparable
]


def test_label_categories(mapping):
    refs, probs = make(CASES)
    res = verify(refs, probs, mapping, cfg(), "unknown")
    assert status(res, "r0", "SR") == "consistent"
    assert status(res, "r1", "AF") == "potentially_incorrect"
    assert status(res, "r1", "SR") == "potentially_missing"
    assert status(res, "r2", "SR") == "uncertain"
    assert status(res, "r2", "STT") == "not_comparable"   # approximate mapping, off by default
    assert status(res, "r3", "FOO") == "not_comparable"
    assert status(res, "r3", "X") == "not_comparable"
    assert status(res, "r3", "SR") == "consistent"        # any-of: sinus brady supports SR
    assert status(res, "r4", "CLBBB") == "consistent"
    assert status(res, "r4", "LBBB") is None             # covered by CLBBB -> no missing flag
    assert status(res, "r5", "NORM") == "uncertain"       # would conflict with AF
    assert status(res, "r6", "AF") == "potentially_missing"
    # absent, weakly predicted labels are not asserted to be negatives
    assert status(res, "r0", "AF") is None


def test_record_level_outputs_and_suggestions(mapping):
    refs, probs = make(CASES)
    res = verify(refs, probs, mapping, cfg(), "unknown")
    r1 = record(res, "r1")
    assert r1.agreement_status == "disagreement"
    assert r1.automated_classification == "potential_label_issue"
    assert r1.suggested_removals == "AF" and r1.suggested_additions == "SR"
    assert r1.proposed_labels == "SR" and r1.original_labels == "AF"
    assert r1.discrepancy_score == pytest.approx(0.995)
    assert r1.discrepancy_severity == "high"
    assert record(res, "r0").agreement_status == "agreement"
    assert record(res, "r2").agreement_status == "uncertain"
    assert record(res, "r7").agreement_status == "not_comparable"
    assert record(res, "r0").discrepancy_score == 0
    # original labels are never altered in the record table
    for ref in refs:
        assert record(res, ref.record_id).original_labels == ";".join(sorted(ref.labels))


def test_proposed_labels_identity(mapping):
    refs, probs = make(CASES)
    res = verify(refs, probs, mapping, cfg(), "unknown")
    for _, r in res.records.iterrows():
        orig = set(filter(None, r.original_labels.split(";")))
        add = set(filter(None, r.suggested_additions.split(";")))
        rem = set(filter(None, r.suggested_removals.split(";")))
        assert set(filter(None, r.proposed_labels.split(";"))) == (orig - rem) | add
        assert rem <= orig and not (add & orig)
        for lab in add | rem:  # only labels ECGFounder can verify are ever proposed
            assert mapping.get(lab).supported and mapping.get(lab).quality == "exact"


def test_approximate_mappings_can_be_enabled(mapping):
    refs, probs = make(CASES)
    res = verify(refs, probs, mapping, cfg(comparable_qualities=["exact", "approximate"]),
                 "unknown")
    assert status(res, "r2", "STT") == "potentially_incorrect"
    # ...but suggestions remain limited to exact mappings by default
    assert record(res, "r2").suggested_removals == ""


def test_absence_semantics_negative(mapping):
    refs, probs = make([(["SR"], {"SINUS RHYTHM": 0.9, "ATRIAL FIBRILLATION": 0.7})])
    unknown = verify(refs, probs, mapping, cfg(), "unknown")
    negative = verify(refs, probs, mapping, cfg(), "negative")
    assert status(unknown, "r0", "AF") is None
    assert status(negative, "r0", "AF") == "uncertain"
    af = negative.label_stats.set_index("label").loc["AF"]
    assert af.n_uncertain_absent == 1


def test_threshold_overrides_and_validation(mapping):
    refs, probs = make([(["AF"], {"ATRIAL FIBRILLATION": 0.04})])
    res = verify(refs, probs, mapping, cfg(), "unknown")
    assert status(res, "r0", "AF") == "uncertain"
    res = verify(refs, probs, mapping,
                 cfg(threshold_overrides={"AF": {"contradict": 0.05}}), "unknown")
    assert status(res, "r0", "AF") == "potentially_incorrect"
    with pytest.raises(ValueError, match="thresholds"):
        verify(refs, probs, mapping, cfg(threshold_overrides={"AF": {"contradict": 0.9}}),
               "unknown")


def test_calibrated_path_is_used_when_enough_labels(mapping):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(600):
        af = i % 3 == 0
        p = rng.beta(8, 2) if af else rng.beta(2, 8)
        recs.append((["SR", "AF"] if af else ["SR"], {"ATRIAL FIBRILLATION": p,
                                                       "SINUS RHYTHM": 0.9}))
    refs, probs = make(recs)
    c = copy.deepcopy(DEFAULTS)
    res = verify(refs, probs, mapping, c, "unknown")
    stats = res.label_stats.set_index("label")
    assert bool(stats.loc["AF", "calibrated"])
    assert not bool(stats.loc["SR", "calibrated"])  # no negatives -> stays uncalibrated
    a = res.assessments
    assert a.loc[a.label == "AF", "calibrated_probability"].notna().all()


def test_rare_label_is_not_contradicted_just_for_being_rare(mapping):
    """Regression: an uninformative score for a 2%-prevalence label must not flag every
    labelled record as potentially incorrect (posterior < 0.05 reflects the base rate only)."""
    rng = np.random.default_rng(0)
    recs = []
    for i in range(3000):
        af = i % 50 == 0
        recs.append((["SR", "AF"] if af else ["SR"],
                     {"ATRIAL FIBRILLATION": rng.uniform(0.01, 0.2), "SINUS RHYTHM": 0.9}))
    refs, probs = make(recs)
    res = verify(refs, probs, mapping, copy.deepcopy(DEFAULTS), "unknown")
    a = res.assessments
    af = a[(a.label == "AF") & a.label_present]
    assert len(af) == 60
    assert (af.calibrated_probability < 0.05).all()          # posterior is tiny for everyone...
    assert (af.status == "potentially_incorrect").sum() <= 3  # ...but that is not evidence
    assert (af.bayes_factor.between(0.3, 3)).mean() > 0.9


def test_bayes_factor_flags_a_genuinely_atypical_labelled_record(mapping):
    rng = np.random.default_rng(1)
    recs = []
    for i in range(3000):
        af = i % 10 == 0
        p = rng.beta(8, 2) if af else rng.beta(1, 30)
        recs.append((["SR", "AF"] if af else ["SR"], {"ATRIAL FIBRILLATION": p, "SINUS RHYTHM": 0.9}))
    recs.append((["SR", "AF"], {"ATRIAL FIBRILLATION": 0.0005, "SINUS RHYTHM": 0.9}))  # odd one out
    refs, probs = make(recs)
    res = verify(refs, probs, mapping, copy.deepcopy(DEFAULTS), "unknown")
    assert status(res, f"r{len(recs) - 1}", "AF") == "potentially_incorrect"
    a = res.assessments
    flagged = a[(a.label == "AF") & (a.status == "potentially_incorrect")]
    assert len(flagged) < 10 and (flagged.bayes_factor <= 0.05).all()
