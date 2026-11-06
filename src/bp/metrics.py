"""MAE, ME y DE del error en mmHg, por paciente y agrupados."""
import numpy as np

TARGETS = ("pas", "pad")


def per_patient_metrics(pred: np.ndarray, y: np.ndarray, patient: np.ndarray) -> dict:
    """pred, y: (n,2) en mmHg; patient: (n,) etiqueta de paciente.

    Devuelve el promedio sobre pacientes de MAE, ME y DE (std del error) por salida,
    más las mismas medidas agrupadas sobre todos los segmentos (la DE agrupada es la
    comparable con el criterio AAMI, pues la por paciente descuenta el sesgo de cada uno).
    """
    err = np.asarray(pred, np.float64) - np.asarray(y, np.float64)
    patient = np.asarray(patient)
    ids = np.unique(patient)
    out = {"n_patients": int(len(ids)), "n_segments": int(len(err))}
    mae = np.empty((len(ids), 2)); me = np.empty_like(mae); de = np.empty_like(mae)
    for i, p in enumerate(ids):
        e = err[patient == p]
        mae[i], me[i], de[i] = np.abs(e).mean(0), e.mean(0), e.std(0)
    for j, t in enumerate(TARGETS):
        out[f"mae_{t}"] = float(mae[:, j].mean())
        out[f"me_{t}"] = float(me[:, j].mean())
        out[f"de_{t}"] = float(de[:, j].mean())
        out[f"pooled_mae_{t}"] = float(np.abs(err[:, j]).mean())
        out[f"pooled_me_{t}"] = float(err[:, j].mean())
        out[f"pooled_sd_{t}"] = float(err[:, j].std())
    return out
