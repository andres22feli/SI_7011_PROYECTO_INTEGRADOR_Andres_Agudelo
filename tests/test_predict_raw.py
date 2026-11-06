import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bp.contract import ContractError, duplicate_report  # noqa: E402
from bp.models import build_gru, build_mlp  # noqa: E402
from bp.predict import make_bundle, predict_raw  # noqa: E402
from bp.preprocess import preprocess_signal  # noqa: E402


def _bundle(builder="build_mlp", inputs=("ppg", "ecg")):
    torch.manual_seed(0)
    model = {"build_mlp": build_mlp, "build_gru": build_gru}[builder](len(inputs))
    for p in model.parameters():  # que la salida no sea la media exacta
        torch.nn.init.normal_(p, std=0.01) if p.ndim > 1 else None
    return model.eval(), make_bundle(model.state_dict(), builder, list(inputs), {}, [112.0, 58.0], [18.0, 12.0], {}, 0)


@pytest.mark.parametrize("builder", ["build_mlp", "build_gru"])
def test_predict_raw_matches_manual_pipeline(builder):
    model, bundle = _bundle(builder)
    rng = np.random.default_rng(0)
    ppg, ecg = rng.normal(size=(4, 3750)), rng.normal(size=(4, 3750))
    out = predict_raw(ppg, ecg, bundle=bundle, device="cpu")
    X = np.stack([preprocess_signal(ppg, "ppg"), preprocess_signal(ecg, "ecg")], axis=1)
    with torch.no_grad():
        ref = model(torch.from_numpy(X)).numpy() * np.array([18.0, 12.0]) + np.array([112.0, 58.0])
    assert out.shape == (4, 2) and np.allclose(out, ref, atol=1e-4)


def test_predict_raw_rejects_bad_input():
    _, bundle = _bundle()
    ok = np.random.default_rng(1).normal(size=3750)
    bad = ok.copy(); bad[3] = np.nan
    with pytest.raises(ContractError):
        predict_raw(bad, ok, bundle=bundle, device="cpu")
    with pytest.raises(ContractError):
        predict_raw(ok, None, bundle=bundle, device="cpu")  # el modelo necesita ECG
    with pytest.raises(ContractError):
        predict_raw(ok[:100], ok, bundle=bundle, device="cpu")


def test_duplicate_report_drops_replicas(tmp_path):
    rng = np.random.default_rng(2)
    base = {m: rng.normal(size=(3, 3750)) for m in ("ppg", "ecg")}
    for sid, extra in (("a", 0), ("b", 1)):
        for m in ("ppg", "ecg"):
            arr = rng.normal(size=(3, 3750))
            if extra:
                arr[1] = base[m][0]  # un segmento de b replica uno de a
            else:
                arr[0] = base[m][0]
            (tmp_path / m).mkdir(exist_ok=True)
            np.save(tmp_path / m / f"{sid}_{m}.npy", arr)
    rep = duplicate_report(["a", "b"], tmp_path)
    assert rep["drop_segments"] == {"b": [1]} and rep["cross_patient_duplicates"] == {"b": ["a"]}
