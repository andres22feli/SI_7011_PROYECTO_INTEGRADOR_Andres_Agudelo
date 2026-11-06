"""Utilidades compartidas por los notebooks: estadísticas de etiquetas, datos por partición y resultados de validación."""
import json

import numpy as np
import torch

from .dataset import load_embedding_arrays, load_split_arrays, to_tensors
from .paths import ARTIFACTS
from .split import load_split

LABEL_STATS = ARTIFACTS / "label_stats.json"
VAL_DIR = ARTIFACTS / "val_results"


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_label_stats(y_mean, y_std) -> None:
    LABEL_STATS.write_text(json.dumps({"y_mean": [float(v) for v in y_mean], "y_std": [float(v) for v in y_std],
                                       "fit_on": "train"}), encoding="utf-8")


def load_label_stats() -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(LABEL_STATS.read_text(encoding="utf-8"))
    return np.array(d["y_mean"], np.float32), np.array(d["y_std"], np.float32)


def get_data(split_name: str, inputs: list[str], embeddings: bool = False, to_device=True) -> dict:
    """Tensores de una partición (train/val/test) con etiquetas estandarizadas con las estadísticas de train."""
    ids = load_split()[split_name]
    arrs = (load_embedding_arrays if embeddings else load_split_arrays)(ids, inputs)
    y_mean, y_std = load_label_stats()
    return to_tensors(arrs, y_mean, y_std, device() if to_device else torch.device("cpu"))


def save_val_result(res: dict) -> None:
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    (VAL_DIR / f"{res['run_name']}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


def load_val_results() -> dict:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(VAL_DIR.glob("*.json"))}
