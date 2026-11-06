import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bp.contract import ContractError, check_patient, validate_patient, validate_raw_signals  # noqa: E402
from bp.metrics import per_patient_metrics  # noqa: E402
from bp.preprocess import preprocess_signal  # noqa: E402
from bp.split import assert_no_leak, make_split  # noqa: E402


def _write(tmp_path, sid="p000001", ppg=None, ecg=None, labels=None):
    rng = np.random.default_rng(0)
    ppg = rng.normal(size=(30, 3750)) if ppg is None else ppg
    ecg = rng.normal(size=(30, 3750)) if ecg is None else ecg
    labels = np.tile([120.0, 80.0], (30, 1)) if labels is None else labels
    for mod, arr in (("ppg", ppg), ("ecg", ecg), ("labels", labels)):
        (tmp_path / mod).mkdir(exist_ok=True)
        np.save(tmp_path / mod / f"{sid}_{mod}.npy", arr)
    return sid


def test_valid_patient(tmp_path):
    sid = _write(tmp_path)
    assert validate_patient(sid, tmp_path).all()


def test_nan_segment_dropped(tmp_path):
    ppg = np.random.default_rng(1).normal(size=(30, 3750)); ppg[3, 10] = np.nan
    sid = _write(tmp_path, ppg=ppg)
    mask = validate_patient(sid, tmp_path)
    assert mask.sum() == 29 and not mask[3]


def test_flat_segment_dropped(tmp_path):
    ecg = np.random.default_rng(2).normal(size=(30, 3750)); ecg[5] = 1.0
    sid = _write(tmp_path, ecg=ecg)
    assert not validate_patient(sid, tmp_path)[5]


def test_wrong_shape_rejected(tmp_path):
    sid = _write(tmp_path, ppg=np.random.default_rng(3).normal(size=(30, 3000)))
    with pytest.raises(ContractError):
        validate_patient(sid, tmp_path)


def test_label_out_of_range_and_sbp_le_dbp(tmp_path):
    labels = np.tile([120.0, 80.0], (30, 1)); labels[0] = [300, 80]; labels[1] = [70, 90]
    sid = _write(tmp_path, labels=labels)
    mask = check_patient(sid, tmp_path)["valid_mask"]
    assert not mask[0] and not mask[1] and mask[2:].all()


def test_too_few_valid_segments_rejected(tmp_path):
    ppg = np.random.default_rng(4).normal(size=(30, 3750)); ppg[:20] = 0.0
    sid = _write(tmp_path, ppg=ppg)
    with pytest.raises(ContractError):
        validate_patient(sid, tmp_path)


def test_raw_signal_contract():
    ok = np.random.default_rng(5).normal(size=3750)
    validate_raw_signals(ok)
    bad = ok.copy(); bad[0] = np.nan
    with pytest.raises(ContractError):
        validate_raw_signals(bad)
    with pytest.raises(ContractError):
        validate_raw_signals(ok[:100])
    with pytest.raises(ContractError):
        validate_raw_signals(np.ones(3750))


def test_split_disjoint_and_sizes():
    ids = [f"p{i:06d}" for i in range(1524)]
    s = make_split(ids, seed=0)
    assert [len(s[k]) for k in ("train", "val", "test")] == [914, 305, 305]
    assert sum(len(v) for v in s.values()) == len(set().union(*map(set, s.values()))) == 1524
    assert make_split(ids, seed=0) == s  # reproducible
    s["test"][0] = s["train"][0]
    with pytest.raises(AssertionError):
        assert_no_leak(s)


def test_preprocess_zscore_per_segment():
    x = np.random.default_rng(6).normal(size=(4, 3750)) * 50 + 200
    for mod in ("ppg", "ecg"):
        y = preprocess_signal(x, mod)
        assert y.dtype == np.float32
        assert np.allclose(y.mean(1), 0, atol=1e-4) and np.allclose(y.std(1), 1, atol=1e-3)


def test_metrics_per_patient():
    patient = np.repeat([0, 1], 30)
    y = np.full((60, 2), 100.0)
    pred = y + np.where(patient[:, None] == 0, 5.0, -3.0)
    m = per_patient_metrics(pred, y, patient)
    assert m["mae_pas"] == pytest.approx(4.0) and m["me_pas"] == pytest.approx(1.0)
    assert m["de_pas"] == pytest.approx(0.0)       # sesgo constante por paciente -> DE por paciente 0
    assert m["pooled_sd_pas"] == pytest.approx(4.0)  # la DE agrupada sí lo ve
