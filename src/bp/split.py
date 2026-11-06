"""Partición por paciente 60/20/20 sin fuga."""
import json
from pathlib import Path

import numpy as np

from .paths import SPLIT_JSON


def make_split(subject_ids: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Mezcla los pacientes (no los segmentos) con semilla fija y corta 60/20/20."""
    ids = sorted(subject_ids)
    perm = np.random.default_rng(seed).permutation(len(ids))
    n = len(ids)
    n_train, n_val = round(0.6 * n), round(0.2 * n)
    shuffled = [ids[i] for i in perm]
    split = {"train": shuffled[:n_train], "val": shuffled[n_train:n_train + n_val],
             "test": shuffled[n_train + n_val:]}
    assert_no_leak(split)
    return split


def assert_no_leak(split: dict[str, list[str]]) -> None:
    sets = {k: set(v) for k, v in split.items()}
    for k, v in split.items():
        assert len(v) == len(sets[k]), f"IDs repetidos dentro de {k}"
    assert not sets["train"] & sets["val"], "pacientes compartidos entre train y val"
    assert not sets["train"] & sets["test"], "pacientes compartidos entre train y test"
    assert not sets["val"] & sets["test"], "pacientes compartidos entre val y test"


def save_split(split: dict, path: Path = SPLIT_JSON) -> None:
    Path(path).write_text(json.dumps(split, indent=1), encoding="utf-8")


def load_split(path: Path = SPLIT_JSON) -> dict[str, list[str]]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    assert_no_leak(split)
    return split
