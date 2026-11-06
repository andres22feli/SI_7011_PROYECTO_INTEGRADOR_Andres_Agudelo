"""Contrato de datos de MIMIC-BP: validación por paciente, duplicados y data_card.json."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .paths import DATA, DATA_CARD, N_SAMPLES, N_SEG

MIN_STD = 1e-6
PAS_RANGE = (50.0, 250.0)
PAD_RANGE = (20.0, 150.0)
# Política: un segmento inválido se descarta; el paciente se conserva si le quedan
# al menos MIN_VALID_SEGMENTS. Un fallo a nivel de paciente (archivo ausente, forma
# incorrecta) lo excluye entero.
MIN_VALID_SEGMENTS = 15
DOI = "10.7910/DVN/DBM1NF"

CONTRACT_SPECS = {
    "signal_shape": [N_SEG, N_SAMPLES],
    "label_shape": [N_SEG, 2],
    "min_std": MIN_STD,
    "pas_range": list(PAS_RANGE),
    "pad_range": list(PAD_RANGE),
    "min_valid_segments": MIN_VALID_SEGMENTS,
}


class ContractError(ValueError):
    """Los datos no cumplen el contrato."""


def read_subject_ids(path: Path = DATA / "subject_ids.txt") -> list[str]:
    """Lee los IDs tolerando CRLF y líneas vacías."""
    ids = [s.strip() for s in Path(path).read_text(encoding="utf-8").splitlines()]
    ids = [s for s in ids if s]
    if len(set(ids)) != len(ids):
        raise ContractError("subject_ids.txt contiene IDs repetidos")
    return ids


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_signals(x: np.ndarray, name: str) -> tuple[np.ndarray, list[str]]:
    """Devuelve (máscara de segmentos buenos, problemas a nivel de arreglo)."""
    x = np.asarray(x)
    n = len(x) if x.ndim >= 1 else 0
    if x.ndim != 2 or x.shape[1] != N_SAMPLES:
        return np.zeros(n, bool), [f"{name}: forma {x.shape}, se esperaba (n,{N_SAMPLES})"]
    if not np.issubdtype(x.dtype, np.floating):
        return np.zeros(n, bool), [f"{name}: dtype {x.dtype} no es float"]
    finite = np.isfinite(x).all(axis=1)
    safe = np.where(np.isfinite(x), x, 0.0)
    return finite & (safe.std(axis=1) > MIN_STD), []


def check_labels(y: np.ndarray) -> tuple[np.ndarray, list[str]]:
    y = np.asarray(y)
    n = len(y) if y.ndim >= 1 else 0
    if y.ndim != 2 or y.shape[1] != 2:
        return np.zeros(n, bool), [f"labels: forma {y.shape}, se esperaba (n,2)"]
    finite = np.isfinite(y).all(axis=1)
    sbp, dbp = y[:, 0], y[:, 1]
    ok = (finite & (sbp >= PAS_RANGE[0]) & (sbp <= PAS_RANGE[1])
          & (dbp >= PAD_RANGE[0]) & (dbp <= PAD_RANGE[1]) & (sbp > dbp))
    return ok, []


def check_patient(subject_id: str, data_dir: Path = DATA) -> dict:
    """Informe no lanzador: {'issues': [...], 'valid_mask': bool(30,), 'n_valid': int}."""
    bad = {"subject_id": subject_id, "valid_mask": np.zeros(N_SEG, bool), "n_valid": 0}
    issues, arrays = [], {}
    for mod in ("ppg", "ecg", "labels"):
        f = Path(data_dir) / mod / f"{subject_id}_{mod}.npy"
        if not f.exists():
            issues.append(f"falta {f.name}")
            continue
        arrays[mod] = np.load(f)
    if issues:
        return {**bad, "issues": issues}
    expected = {"ppg": (N_SEG, N_SAMPLES), "ecg": (N_SEG, N_SAMPLES), "labels": (N_SEG, 2)}
    for mod, shape in expected.items():
        if arrays[mod].shape != shape:
            issues.append(f"{mod}: forma {arrays[mod].shape}, se esperaba {shape}")
    if issues:
        return {**bad, "issues": issues}
    masks = []
    for mod, fn in (("ppg", lambda a: check_signals(a, "ppg")), ("ecg", lambda a: check_signals(a, "ecg")),
                    ("labels", check_labels)):
        m, iss = fn(arrays[mod])
        issues += iss
        masks.append(m)
    if issues:
        return {**bad, "issues": issues}
    mask = masks[0] & masks[1] & masks[2]
    return {"subject_id": subject_id, "issues": [], "valid_mask": mask, "n_valid": int(mask.sum())}


def validate_patient(subject_id: str, data_dir: Path = DATA) -> np.ndarray:
    """Máscara de segmentos válidos; lanza ContractError si el paciente se excluye."""
    r = check_patient(subject_id, data_dir)
    if r["issues"]:
        raise ContractError(f"{subject_id}: " + "; ".join(r["issues"]))
    if r["n_valid"] < MIN_VALID_SEGMENTS:
        raise ContractError(f"{subject_id}: solo {r['n_valid']} segmentos válidos (< {MIN_VALID_SEGMENTS})")
    return r["valid_mask"]


def validate_raw_signals(ppg, ecg=None) -> tuple[np.ndarray, np.ndarray | None]:
    """Contrato estricto para predict_raw. Acepta (3750,) o (n,3750); lanza ante cualquier problema."""
    out = []
    for name, sig in (("ppg", ppg), ("ecg", ecg)):
        if sig is None:
            out.append(None)
            continue
        a = np.asarray(sig)
        if a.ndim == 1:
            a = a[None]
        if not np.issubdtype(a.dtype, np.floating):
            raise ContractError(f"{name}: dtype {a.dtype} no es float")
        if a.ndim != 2 or a.shape[1] != N_SAMPLES:
            raise ContractError(f"{name}: forma {a.shape}, se esperaba (n,{N_SAMPLES}) o ({N_SAMPLES},)")
        if not np.isfinite(a).all():
            raise ContractError(f"{name}: contiene NaN o Inf")
        if (a.std(axis=1) <= MIN_STD).any():
            raise ContractError(f"{name}: segmento plano (std <= {MIN_STD})")
        out.append(a.astype(np.float64))
    if out[0] is not None and out[1] is not None and len(out[0]) != len(out[1]):
        raise ContractError("ppg y ecg tienen distinto número de segmentos")
    return out[0], out[1]


def duplicate_report(subject_ids: list[str], data_dir: Path = DATA, mask_by_id: dict | None = None) -> dict:
    """SHA-1 de cada segmento (ppg+ecg). El primer paciente que lo posee lo conserva; las réplicas
    (en otro paciente o dentro del mismo) quedan en 'drop_segments' para descartarlas antes de partir."""
    owner: dict[str, str] = {}
    drop: dict[str, list[int]] = {}
    cross: dict[str, set] = {}
    for sid in subject_ids:
        ppg = np.load(Path(data_dir) / "ppg" / f"{sid}_ppg.npy")
        ecg = np.load(Path(data_dir) / "ecg" / f"{sid}_ecg.npy")
        mask = None if mask_by_id is None else mask_by_id.get(sid)
        for i in range(len(ppg)):
            if mask is not None and not mask[i]:
                continue
            h = hashlib.sha1(ppg[i].tobytes() + ecg[i].tobytes()).hexdigest()
            if h in owner:
                drop.setdefault(sid, []).append(i)
                if owner[h] != sid:
                    cross.setdefault(sid, set()).add(owner[h])
            else:
                owner[h] = sid
    return {"drop_segments": drop, "n_segments_dropped": sum(len(v) for v in drop.values()),
            "cross_patient_duplicates": {k: sorted(v) for k, v in cross.items()},
            "n_patients_with_cross_duplicates": len(cross)}


def save_data_card(card: dict, path: Path = DATA_CARD) -> None:
    Path(path).write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")


def label_descriptives(labels: np.ndarray) -> dict:
    out = {}
    for j, t in enumerate(("pas", "pad")):
        v = labels[:, j]
        out[t] = {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()), "max": float(v.max()),
                  "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}
    return out


def build_data_card(subject_ids, kept_ids, masks, excluded, dup, labels_valid) -> dict:
    kept = set(kept_ids)
    return {
        "dataset": "MIMIC-BP", "doi": DOI,
        "subject_ids_sha256": sha256_file(DATA / "subject_ids.txt"),
        "n_subjects_listed": len(subject_ids), "n_subjects_kept": len(kept_ids),
        "n_segments_valid": int(sum(int(m.sum()) for s, m in masks.items() if s in kept)),
        "n_segments_dropped": int(sum(int((~m).sum()) for s, m in masks.items() if s in kept)),
        "excluded_subjects": excluded,
        "policy": {"invalid_segment": "descartar el segmento", "min_valid_segments_per_patient": MIN_VALID_SEGMENTS},
        "duplicates": dup,
        "label_descriptives_mmHg": label_descriptives(labels_valid),
        "contract": CONTRACT_SPECS,
        "inputs_allowed": ["ppg", "ecg"], "abp_used_as_input": False,
    }
