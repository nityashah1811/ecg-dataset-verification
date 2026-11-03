import numpy as np
import pytest

from ecg_verify.ecgfounder import CLASS_INDEX, CLASS_NAMES, SERIAL_COMPARISON_CLASSES
from ecg_verify.mapping import MAPPINGS_DIR, load_mapping


@pytest.mark.parametrize("name", sorted(p.stem for p in MAPPINGS_DIR.glob("*.yaml")))
def test_shipped_mappings_are_valid(name):
    m = load_mapping(name)
    assert m.labels
    for lm in m.labels.values():
        for c in lm.model_classes:
            assert c in CLASS_INDEX
            assert c not in SERIAL_COMPARISON_CLASSES


def test_class_space_matches_official_tasks():
    assert len(CLASS_NAMES) == 150
    assert CLASS_NAMES[5] == "ATRIAL FIBRILLATION"  # index used by the official label csv
    assert "QT HAS LENGTHENED" in SERIAL_COMPARISON_CLASSES
    assert "PROLONGED QT" not in SERIAL_COMPARISON_CLASSES


def test_ptbxl_mapping_avoids_known_wrong_official_mappings():
    m = load_mapping("ptbxl_scp")
    assert m.get("IRBBB").model_classes == ("INCOMPLETE RIGHT BUNDLE BRANCH BLOCK",)
    assert "VENTRICULAR TACHYCARDIA" not in m.get("SVTAC").model_classes
    assert not m.get("ISC_").supported
    assert not m.get("ABQRS").supported
    assert m.get("LNGQT").model_classes == ("PROLONGED QT",)


def write(tmp_path, body):
    p = tmp_path / "m.yaml"
    p.write_text(body)
    return str(p)


def test_invalid_mappings_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown ECGFounder class"):
        load_mapping(write(tmp_path, "labels:\n  X: {model_classes: [NOT A CLASS], quality: exact}\n"))
    with pytest.raises(ValueError, match="serial-comparison"):
        load_mapping(write(tmp_path, "labels:\n  X: {model_classes: [QT HAS LENGTHENED], quality: exact}\n"))
    with pytest.raises(ValueError, match="quality"):
        load_mapping(write(tmp_path, "labels:\n  X: {model_classes: [ATRIAL FLUTTER], quality: good}\n"))
    with pytest.raises(ValueError, match="covered_by"):
        load_mapping(write(tmp_path, "labels:\n  X: {model_classes: [ATRIAL FLUTTER], quality: exact, covered_by: [Y]}\n"))


def test_any_of_label_score_is_max_over_classes():
    m = load_mapping("ptbxl_scp")
    probs = np.zeros((2, 150))
    probs[0, CLASS_INDEX["SINUS BRADYCARDIA"]] = 0.9
    probs[0, CLASS_INDEX["SINUS RHYTHM"]] = 0.2
    probs[1, CLASS_INDEX["SINUS RHYTHM"]] = 0.4
    s = m.label_scores(probs, ["SR", "SBRAD", "AFIB"])
    np.testing.assert_allclose(s, [[0.9, 0.9, 0.0], [0.4, 0.0, 0.0]])


def test_exclusivity_rules():
    norm = load_mapping("ptbxl_scp").get("NORM")
    assert norm.conflicting({"SR", "SBRAD", "AXIS_MID"}) == []
    assert norm.conflicting({"SR", "LVH"}) == ["LVH"]
