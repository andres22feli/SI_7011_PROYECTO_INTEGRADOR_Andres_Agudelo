"""Rutas del proyecto resueltas desde la raíz, sin depender del cwd del notebook."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CACHE = ROOT / "cache"
PREPROCESSED = CACHE / "preprocessed"
ARTIFACTS = ROOT / "artifacts"
SPLIT_JSON = CACHE / "split.json"
DATA_CARD = ARTIFACTS / "data_card.json"
MLFLOW_URI = "sqlite:///" + (ROOT / "mlflow.db").as_posix()

FS = 125
N_SEG = 30
N_SAMPLES = 3750
# Solo estas modalidades pueden ser entrada. ABP es de donde salen las etiquetas (fuga).
INPUT_MODALITIES = ("ppg", "ecg")
FILTERS = {"order": 4, "ppg_band": [0.5, 8.0], "ecg_band": [0.5, 40.0]}

for _d in (CACHE, PREPROCESSED, ARTIFACTS):
    _d.mkdir(parents=True, exist_ok=True)
