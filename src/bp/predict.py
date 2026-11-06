"""Bundle portable y predict_raw: señales crudas -> mmHg."""
from pathlib import Path

import numpy as np
import torch

from .contract import CONTRACT_SPECS, ContractError, validate_raw_signals
from .paths import ARTIFACTS, FILTERS, FS
from .preprocess import preprocess_signal

BUNDLE_PATH = ARTIFACTS / "best_model_bundle.pt"
_CACHE: dict = {}


def make_bundle(state: dict, builder: str, inputs: list[str], model_kw: dict, y_mean, y_std,
                data_card: dict, best_epoch: int, test_summary: dict | None = None, amp: bool = False, extra: dict | None = None) -> dict:
    return {
        "state_dict": state, "model_builder": builder, "model_kw": model_kw, "inputs": list(inputs),
        "fs": FS, "filters": FILTERS,
        "y_mean": torch.tensor(np.asarray(y_mean), dtype=torch.float32),
        "y_std": torch.tensor(np.asarray(y_std), dtype=torch.float32),
        "contract": CONTRACT_SPECS, "data_card": data_card, "best_epoch": best_epoch,
        "test_summary": test_summary or {}, "amp": bool(amp), **(extra or {}),
    }


def save_bundle(bundle: dict, path: Path = BUNDLE_PATH) -> None:
    torch.save(bundle, path)


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    return torch.load(path, weights_only=False, map_location="cpu")


def _model_from_bundle(bundle: dict, device):
    from .train import build_model, load_state  # evita importar mlflow/etc. hasta que haga falta
    model = build_model(bundle["model_builder"], len(bundle["inputs"]), 0, **bundle["model_kw"])
    load_state(model, bundle["state_dict"])
    return model.to(device).eval()


def predict_raw(ppg_signal=None, ecg_signal=None, bundle: dict | str | Path | None = None,
                device=None, batch_size: int = 256) -> np.ndarray:
    """Predice [PAS, PAD] en mmHg a partir de segmentos crudos de 30 s (125 Hz).

    Acepta una señal (3750,) o un lote (n, 3750). Devuelve (n, 2). Valida el contrato, filtra y
    estandariza por segmento con la misma función de entrenamiento, usa model.eval() y no_grad.
    """
    if bundle is None or isinstance(bundle, (str, Path)):
        key = str(bundle or BUNDLE_PATH)
        if key not in _CACHE:
            _CACHE[key] = load_bundle(Path(key))
        bundle = _CACHE[key]
    device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = {"ppg": ppg_signal, "ecg": ecg_signal}
    missing = [m for m in bundle["inputs"] if raw[m] is None]
    if missing:
        raise ContractError(f"el modelo requiere {bundle['inputs']}, falta(n): {missing}")
    ppg, ecg = validate_raw_signals(raw["ppg"] if "ppg" in bundle["inputs"] else None,
                                    raw["ecg"] if "ecg" in bundle["inputs"] else None)
    clean = {"ppg": ppg, "ecg": ecg}
    X = np.stack([preprocess_signal(clean[m], m, bundle["filters"], bundle["fs"]) for m in bundle["inputs"]], axis=1)
    mkey = (id(bundle), str(device))
    if mkey not in _CACHE:
        _CACHE[mkey] = _model_from_bundle(bundle, device)
    model = _CACHE[mkey].eval()
    outs = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=bool(bundle.get("amp")) and device.type == "cuda"):
        for i in range(0, len(X), batch_size):
            outs.append(model(torch.from_numpy(X[i:i + batch_size]).to(device)).float().cpu())
    pn = torch.cat(outs)
    return (pn * bundle["y_std"] + bundle["y_mean"]).numpy()
