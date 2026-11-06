"""Carga del caché preprocesado y tensores por partición. Solo PPG/ECG son entradas (nunca ABP)."""
import json

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .paths import INPUT_MODALITIES, PREPROCESSED

INPUT_SETS = {"ppg": ["ppg"], "ecg": ["ecg"], "ppg+ecg": ["ppg", "ecg"]}


def input_channels(inputs: list[str]) -> list[str]:
    bad = [m for m in inputs if m not in INPUT_MODALITIES]
    if bad:
        raise ValueError(f"modalidad(es) no permitida(s) como entrada: {bad} (ABP produce las etiquetas)")
    return list(inputs)


def label_stats(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Media y DE de PAS/PAD calculadas SOLO con train."""
    return y_train.mean(0).astype(np.float32), y_train.std(0).astype(np.float32)


def split_rows(split_ids: list[str], cache_dir=PREPROCESSED) -> tuple[np.ndarray, dict]:
    """Filas del caché que pertenecen a los pacientes de la partición (orden estable)."""
    subjects = json.loads((cache_dir / "subjects.json").read_text(encoding="utf-8"))
    idx = np.load(cache_dir / "index.npz")
    pos = {s: i for i, s in enumerate(subjects)}
    keep_subj = np.array([pos[s] for s in split_ids], dtype=int)
    rows = np.flatnonzero(np.isin(idx["subject_idx"], keep_subj))
    return rows, idx


def load_split_arrays(split_ids: list[str], inputs: list[str], cache_dir=PREPROCESSED) -> dict:
    """Arreglos numpy (X: n,C,3750 float32; y: n,2 mmHg; patient: n,) para una partición."""
    inputs = input_channels(inputs)
    rows, idx = split_rows(split_ids, cache_dir)
    X = np.stack([np.load(cache_dir / f"{m}.npy", mmap_mode="r")[rows] for m in inputs], axis=1)
    y = np.load(cache_dir / "labels.npy", mmap_mode="r")[rows]
    return {"X": np.ascontiguousarray(X), "y": np.asarray(y), "patient": idx["subject_idx"][rows],
            "seg": idx["seg_idx"][rows]}


def to_tensors(arrs: dict, y_mean: np.ndarray, y_std: np.ndarray, device) -> dict:
    """Pasa a tensores (en device) y estandariza las etiquetas con las estadísticas de train."""
    yn = (arrs["y"] - y_mean) / y_std
    return {"X": torch.from_numpy(arrs["X"]).to(device), "yn": torch.from_numpy(yn.astype(np.float32)).to(device),
            "y": arrs["y"], "patient": arrs["patient"]}


def train_loader(data: dict, batch_size: int, seed: int = 0) -> DataLoader:
    """Train mezclado con generador de semilla fija. Val/Test se recorren en orden fijo (sin shuffle)."""
    ds = TensorDataset(data["X"], data["yn"])
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=True)


def load_embedding_arrays(split_ids: list[str], inputs: list[str], cache_dir=PREPROCESSED) -> dict:
    """Embeddings TimesFM precalculados (n, C*1280) para el probe lineal, mismas filas que load_split_arrays."""
    inputs = input_channels(inputs)
    rows, idx = split_rows(split_ids, cache_dir)
    E = np.concatenate([np.load(cache_dir / f"timesfm_emb_{m}.npy", mmap_mode="r")[rows] for m in inputs], axis=1)
    y = np.load(cache_dir / "labels.npy", mmap_mode="r")[rows]
    return {"X": np.ascontiguousarray(E), "y": np.asarray(y), "patient": idx["subject_idx"][rows],
            "seg": idx["seg_idx"][rows]}
