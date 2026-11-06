"""Acondicionamiento: Butterworth pasa-banda de fase cero + z-score por segmento."""
import json

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .paths import DATA, FILTERS, FS, PREPROCESSED, N_SAMPLES

EPS = 1e-8


def _sos(modality: str, filters: dict = FILTERS, fs: int = FS):
    band = filters[f"{modality}_band"]
    return butter(filters["order"], band, btype="band", fs=fs, output="sos")


def preprocess_signal(x: np.ndarray, modality: str, filters: dict = FILTERS, fs: int = FS) -> np.ndarray:
    """x: (n, 3750) crudo -> (n, 3750) float32 filtrado y estandarizado por segmento.

    Es la misma función en entrenamiento y en predict_raw.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = sosfiltfilt(_sos(modality, filters, fs), x, axis=-1)
    y = (y - y.mean(axis=-1, keepdims=True)) / (y.std(axis=-1, keepdims=True) + EPS)
    return y.astype(np.float32)


def build_cache(subject_ids: list[str], masks: dict[str, np.ndarray], out_dir=PREPROCESSED) -> dict:
    """Preprocesa una sola vez los segmentos válidos y los guarda como float32.

    Archivos: ppg.npy, ecg.npy (N,3750), labels.npy (N,2) en mmHg, index.npz (subject_idx, seg_idx).
    """
    n_total = sum(int(masks[s].sum()) for s in subject_ids)
    ppg = np.empty((n_total, N_SAMPLES), np.float32)
    ecg = np.empty((n_total, N_SAMPLES), np.float32)
    labels = np.empty((n_total, 2), np.float32)
    subj_idx = np.empty(n_total, np.int32)
    seg_idx = np.empty(n_total, np.int16)
    pos = 0
    for k, sid in enumerate(subject_ids):
        m = masks[sid]
        n = int(m.sum())
        sl = slice(pos, pos + n)
        ppg[sl] = preprocess_signal(np.load(DATA / "ppg" / f"{sid}_ppg.npy")[m], "ppg")
        ecg[sl] = preprocess_signal(np.load(DATA / "ecg" / f"{sid}_ecg.npy")[m], "ecg")
        labels[sl] = np.load(DATA / "labels" / f"{sid}_labels.npy")[m]
        subj_idx[sl] = k
        seg_idx[sl] = np.flatnonzero(m)
        pos += n
    np.save(out_dir / "ppg.npy", ppg)
    np.save(out_dir / "ecg.npy", ecg)
    np.save(out_dir / "labels.npy", labels)
    np.savez(out_dir / "index.npz", subject_idx=subj_idx, seg_idx=seg_idx)
    (out_dir / "subjects.json").write_text(json.dumps(subject_ids), encoding="utf-8")
    return {"n_segments": n_total, "n_subjects": len(subject_ids)}
