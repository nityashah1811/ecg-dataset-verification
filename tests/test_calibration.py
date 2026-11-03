import numpy as np

from ecg_verify.calibration import CalibrationConfig, crossfit_calibrate, expected_calibration_error


def data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    # Badly calibrated, overconfident-high raw score with real signal
    z = rng.normal(np.where(y == 1, 1.5, -0.5), 1.0)
    s = 1 / (1 + np.exp(-(3 * z + 2)))
    folds = rng.integers(0, 5, n)
    return s[:, None], y[:, None], folds


def test_platt_crossfit_improves_calibration():
    s, y, folds = data()
    res = crossfit_calibrate(s, y, folds, ["L"], CalibrationConfig())
    assert res.calibrated[0]
    d = res.diagnostics[0]
    assert d["ece_crossfit"] < 0.05 < d["ece_raw"]
    assert np.all((res.probs >= 0) & (res.probs <= 1))


def test_crossfit_has_no_leakage():
    s, y, folds = data()
    a = crossfit_calibrate(s, y, folds, ["L"], CalibrationConfig()).probs
    y2 = y.copy()
    y2[folds == 0] = 1 - y2[folds == 0]  # corrupt labels of fold 0 only
    b = crossfit_calibrate(s, y2, folds, ["L"], CalibrationConfig()).probs
    np.testing.assert_allclose(a[folds == 0], b[folds == 0])  # fold-0 outputs never saw fold 0
    assert not np.allclose(a[folds != 0], b[folds != 0])


def test_too_few_positives_stays_uncalibrated():
    s, y, folds = data()
    y[:] = 0
    y[:10] = 1
    res = crossfit_calibrate(s, y, folds, ["L"], CalibrationConfig(min_positives=20))
    assert not res.calibrated[0] and np.isnan(res.probs).all()
    assert "positives" in res.diagnostics[0]["reason"]


def test_isotonic_option():
    s, y, folds = data()
    res = crossfit_calibrate(s, y, folds, ["L"], CalibrationConfig(method="isotonic"))
    assert res.calibrated[0] and res.diagnostics[0]["ece_crossfit"] < 0.05


def test_ece_perfect():
    p = np.repeat(np.linspace(0.05, 0.95, 10), 1000)
    rng = np.random.default_rng(1)
    y = (rng.random(p.size) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.02
