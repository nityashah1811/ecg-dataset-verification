import numpy as np
import pytest
from scipy.signal import butter, filtfilt, iirnotch, medfilt

from ecg_verify.ecgfounder import MODEL_LEAD_ORDER
from ecg_verify.preprocessing import (IncompatibleRecord, PreprocessConfig, canonical_lead,
                                      official_filter, preprocess, resample, zscore)


def official_filter_bandpass_verbatim(signal, fs):
    """Verbatim copy of util.filter_bandpass from PKUDigitalHealth/ECGFounder (MIT)."""
    b, a = iirnotch(50, 30, fs)
    filtered_signal = np.zeros_like(signal)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, signal[c])
    b, a = butter(N=4, Wn=[0.67, 40], btype='bandpass', fs=fs)
    for c in range(signal.shape[0]):
        filtered_signal[c] = filtfilt(b, a, filtered_signal[c])
    baseline = np.zeros_like(filtered_signal)
    for c in range(filtered_signal.shape[0]):
        kernel_size = int(0.4 * fs) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        baseline[c] = medfilt(filtered_signal[c], kernel_size=kernel_size)
    return filtered_signal - baseline


def official_zscore(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def synthetic(n_leads=12, fs=500, seconds=10, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * seconds)) / fs
    base = np.sin(2 * np.pi * 1.2 * t)[None] * rng.uniform(0.2, 1.5, (n_leads, 1))
    return base + 0.05 * rng.standard_normal((n_leads, len(t))) + 0.3 * np.sin(2 * np.pi * 0.2 * t)


def test_filter_matches_official_implementation():
    x = synthetic()
    np.testing.assert_allclose(official_filter(x, 500), official_filter_bandpass_verbatim(x, 500))


def test_zscore_is_global_not_per_lead():
    x = synthetic()
    x[0] *= 10
    np.testing.assert_allclose(zscore(x), official_zscore(x))
    assert abs(zscore(x)[0].std() - 1) > 0.1  # per-lead std is not forced to 1


def test_full_pipeline_matches_official_for_500hz_10s():
    x = synthetic()
    prep = preprocess(x, 500, MODEL_LEAD_ORDER, 12, PreprocessConfig())
    expected = official_zscore(official_filter_bandpass_verbatim(x, 500)).astype(np.float32)
    assert prep.windows.shape == (1, 12, 5000)
    np.testing.assert_allclose(prep.windows[0], expected, rtol=1e-5, atol=1e-5)
    assert prep.warnings == []


def test_preprocessing_is_deterministic():
    x = synthetic(fs=360, seconds=12)
    a = preprocess(x, 360, MODEL_LEAD_ORDER, 12, PreprocessConfig()).windows
    b = preprocess(x.copy(), 360, MODEL_LEAD_ORDER, 12, PreprocessConfig()).windows
    assert np.array_equal(a, b)


def test_leads_are_reordered_by_name():
    x = synthetic()
    names = ["I", "II", "III", "AVR", "AVF", "AVL", "V1", "V2", "V3", "V4", "V5", "V6"]  # MIMIC order
    cfg = PreprocessConfig(filter="none")
    got = preprocess(x, 500, names, 12, cfg).windows[0]
    ref = preprocess(x[[0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 10, 11]], 500, MODEL_LEAD_ORDER, 12,
                     cfg).windows[0]
    np.testing.assert_array_equal(got, ref)
    assert canonical_lead("DII") == "II" and canonical_lead("avl") == "aVL"


def test_missing_lead_is_incompatible_not_zero_filled():
    x = synthetic(n_leads=11)
    with pytest.raises(IncompatibleRecord, match="missing lead"):
        preprocess(x, 500, MODEL_LEAD_ORDER[:11], 12, PreprocessConfig())


def test_single_lead_model_uses_lead_I():
    x = synthetic()
    w = preprocess(x, 500, MODEL_LEAD_ORDER, 1, PreprocessConfig()).windows
    assert w.shape == (1, 1, 5000)


def test_resampling_lengths():
    x = synthetic(fs=1000)
    assert resample(x, 1000, 500, "polyphase").shape[1] == 5000
    assert resample(x, 1000, 500, "linear").shape[1] == 5000
    y = synthetic(fs=257, seconds=10)
    assert abs(resample(y, 257, 500, "polyphase").shape[1] - 5000) <= 1


def test_short_long_and_bad_records():
    cfg = PreprocessConfig()
    with pytest.raises(IncompatibleRecord, match="duration"):
        preprocess(synthetic(seconds=6), 500, MODEL_LEAD_ORDER, 12, cfg)
    p = preprocess(synthetic(seconds=9.8), 500, MODEL_LEAD_ORDER, 12, cfg)
    assert p.windows.shape == (1, 12, 5000) and any("padded" in w for w in p.warnings)
    p = preprocess(synthetic(seconds=35), 500, MODEL_LEAD_ORDER, 12, cfg)
    assert p.windows.shape == (3, 12, 5000)
    p = preprocess(synthetic(seconds=35), 500, MODEL_LEAD_ORDER, 12,
                   PreprocessConfig(long_record_policy="first"))
    assert p.windows.shape == (1, 12, 5000)
    x = synthetic()
    x[:, :1000] = np.nan
    with pytest.raises(IncompatibleRecord, match="NaN"):
        preprocess(x, 500, MODEL_LEAD_ORDER, 12, cfg)
    x = synthetic()
    x[:4] = 0.0
    with pytest.raises(IncompatibleRecord, match="flat"):
        preprocess(x, 500, MODEL_LEAD_ORDER, 12, cfg)
    with pytest.raises(IncompatibleRecord, match="sampling"):
        preprocess(synthetic(), 0, MODEL_LEAD_ORDER, 12, cfg)
