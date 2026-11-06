[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andres22feli/SI_7011_PROYECTO_INTEGRADOR_Andres_Agudelo/blob/main/notebooks/proyecto_completo.ipynb)

# Estimación de PAS/PAD desde PPG y ECG (MIMIC-BP) — SI7011

Diseño factorial **3 entradas (PPG, ECG, PPG+ECG) × 3 arquitecturas (MLP, GRU, TimesFM 2.5)** con la misma partición por paciente
(914 / 305 / 305), el mismo preprocesamiento y el mismo flujo de entrenamiento. Plan: [PLAN_PROYECTO_DL.md](PLAN_PROYECTO_DL.md);
anteproyecto: [Anteproyecto_DL_Presion_Arterial_Andres_Agudelo.pdf](Anteproyecto_DL_Presion_Arterial_Andres_Agudelo.pdf).

## Reproducir

```bash
conda env create -f environment.yml          # o: conda create -n dl_bp python=3.11 + pip install (ver environment.yml)
conda activate dl_bp
pytest tests -q
jupyter nbconvert --to notebook --execute --inplace notebooks/proyecto_completo.ipynb
```

**En Google Colab:** pulsa el botón de arriba, activa la GPU (T4) y ejecuta las partes en orden. Los datos y el caché se leen de
`Mi unidad/dl_bp/{data,cache}` en Drive (la Parte 0 clona el repo, monta Drive y enlaza esas carpetas).

Los datos van en `data/{ppg,ecg,abp,labels}/` (DOI 10.7910/DVN/DBM1NF). ABP **no** se usa como entrada: de ahí salen las etiquetas.

## Estructura

| Ruta | Contenido |
|---|---|
| `src/bp/contract.py` | Contrato de datos, duplicados (SHA-1), `data_card` |
| `src/bp/split.py` | Partición por paciente 60/20/20 (seed 0) y verificación de no fuga |
| `src/bp/preprocess.py` | Butterworth pasa-banda de fase cero + z-score por segmento (misma función en `predict_raw`) |
| `src/bp/models.py` | MLP, GRU con patching, TimesFM 2.5 (probe / LoRA) |
| `src/bp/train.py` | AdamW + warmup/coseno por lote + clipping + early stopping + MLflow + chequeos previos |
| `src/bp/metrics.py` | MAE, ME, DE por paciente y agrupados |
| `src/bp/predict.py` | Bundle portable y `predict_raw` |
| `notebooks/proyecto_completo.ipynb` | Notebook único, por partes: datos → chequeos → MLP → GRU → TimesFM → reporte y `predict_raw` |
| `artifacts/` | `data_card.json`, `label_stats.json`, `val_results/`, `test_results.json`, `tabla_3x3.*`, `best_model_bundle.pt`, figuras |
| `mlflow.db` | Corridas (`mlflow ui --backend-store-uri sqlite:///mlflow.db`) |

## Decisiones y desviaciones respecto al plan

- **Duplicados:** el control SHA-1 encontró 30 segmentos repetidos (1 entre dos pacientes —`p083120` y `p072885`—, el resto dentro de un mismo paciente). Se descartan las réplicas antes de partir: 45 690 segmentos válidos en lugar de 45 720 (train 27 396 / val 9 149 / test 9 145). Los 1 524 pacientes se conservan.
- **MLP:** además de la variante base se entrenó la del plan con `Dropout(0.2)` (la base sobreajusta en las primeras épocas); se elige por `val_loss`.
- **TimesFM:** el contexto máximo (16 384) admite las 3 750 muestras; se rellena a 3 776 (118 parches). Por costo (≈ 10–20 min por época de LoRA en una RTX 4060 de 8 GB) el LoRA se entrena **3 épocas** (no 30), con bf16, batch efectivo 128 por acumulación y tasa de los adaptadores **2e-4** (no 2e-5, que es una tasa de fine-tuning completo y apenas movería los adaptadores en ≈ 640 pasos). La cabeza usa 1e-3 y arranca desde la del probe lineal, cuyos embeddings se calculan una sola vez.
- **Métricas:** la DE por paciente (definición del anteproyecto) descuenta el sesgo de cada paciente y por eso es optimista; la comparación con AAMI (|ME| ≤ 5, DE ≤ 8 mmHg) se hace con ME y DE **agrupadas** sobre todos los segmentos. Ambas se reportan.
- **TimesFM como estrategia de dos pasos:** en cada entrada se reporta el paso (probe o LoRA) con menor `val_loss`: LoRA en PPG y PPG+ECG, probe en ECG (el LoRA sobreajusta en ECG; su mejor época fue la 0). La tabla con ambos pasos está en `artifacts/tabla_timesfm_ambos_pasos.md`.
- **Precisión numérica:** TimesFM se entrena con autocast bf16, pero la inferencia (test y `predict_raw`) es fp32: en bf16 el mismo modelo varía hasta 0.19 mmHg según el tamaño de lote y hasta ≈ 1 mmHg frente a fp32.
- **Cuántas veces se miró test (con transparencia):** (1) una evaluación completa con TimesFM en bf16 (archivada en `artifacts/test_results_bf16_primera_corrida.json`); (2) un diagnóstico sobre 240 filas de test para medir la discrepancia de `predict_raw`; (3) una segunda evaluación completa con inferencia fp32, que es la que se reporta. El cambio se hizo por reproducibilidad, no para mejorar métricas; toda selección (variante de MLP, paso de TimesFM, modelo del bundle) se hizo con validación. Después, el notebook 06 solo repite la inferencia del modelo del bundle para verificar `predict_raw`.
- **MLflow:** `mlflow.db` tiene 21 corridas finalizadas en el experimento `mimic_bp` (las 15 finales más 6 de re-ejecuciones de los notebooks 03 y 04); los artefactos (`best.pt`, `data_card.json`) quedaron en `notebooks/mlruns/` porque el experimento se creó con ese directorio de trabajo.
- **Portabilidad del bundle:** el bundle de TimesFM guarda solo la cabeza y los adaptadores LoRA, y registra el id y la revisión exacta del modelo base (`base_model`, `base_revision`) y las versiones de torch/transformers/peft. `predict_raw` necesita el modelo base en la caché local de Hugging Face o acceso a la red.

## Resultados (test, 305 pacientes, una sola evaluación de selección)

MAE / ME / DE por paciente, promediados sobre pacientes (definición del anteproyecto), en mmHg. Tabla completa, con ME y DE agrupadas y el chequeo AAMI: `artifacts/tabla_3x3.md`.

| Entrada | Modelo | Paso | MAE PAS | ME PAS | DE PAS | MAE PAD | ME PAD | DE PAD |
|---|---|---|---|---|---|---|---|---|
| — | Línea base (media) | | 13.82 | -0.15 | 11.86 | 9.28 | 0.47 | 6.58 |
| PPG | MLP | | 13.76 | 0.29 | 12.26 | 8.99 | 0.84 | 6.94 |
| PPG | GRU | | 13.10 | 0.49 | 11.65 | 8.20 | 1.11 | 6.55 |
| PPG | TimesFM 2.5 | LoRA | **12.70** | 0.13 | 11.47 | **7.92** | 0.36 | 6.52 |
| ECG | MLP | | 13.87 | -0.52 | 12.10 | 9.24 | 0.27 | 6.95 |
| ECG | GRU | | 13.74 | -0.52 | 11.85 | 8.99 | -0.37 | 6.64 |
| ECG | TimesFM 2.5 | probe | 13.66 | -0.07 | 11.93 | 8.75 | 0.50 | 6.75 |
| PPG + ECG | MLP | | 13.74 | -0.86 | 12.00 | 8.93 | -0.17 | 6.78 |
| PPG + ECG | GRU | | 13.27 | -1.61 | 11.64 | 8.29 | -0.82 | 6.69 |
| PPG + ECG | TimesFM 2.5 | LoRA | 12.72 | 1.38 | 11.56 | 7.97 | 0.01 | 6.60 |

**Lectura honesta de los resultados**

- Ningún modelo se acerca a un uso clínico: el mejor (TimesFM-LoRA con PPG) mejora a la línea base de la media en solo ≈ 1.1 mmHg (PAS) y ≈ 1.4 mmHg (PAD), y nadie cumple AAMI con las métricas agrupadas (DE agrupada ≈ 16 mmHg en PAS y ≈ 10 en PAD frente al límite de 8).
- El MLP sobre la forma de onda cruda queda prácticamente en la línea base; la GRU mejora algo; TimesFM da lo mejor, con PPG.
- El **PPG aporta casi toda la señal**: ECG solo apenas supera a la línea base, y añadirlo al PPG no mejora (TimesFM PPG vs PPG+ECG difieren 0.02 mmHg en MAE PAS: empate, no ranking).
- LoRA se entrenó 3 épocas por costo (ver arriba), por lo que no es una comparación a igual número de épocas con MLP/GRU (hasta 30, con early stopping).
- El bundle (`artifacts/best_model_bundle.pt`) es TimesFM-LoRA sobre PPG, elegido por `val_loss`; `predict_raw` reproduce las predicciones de test con diferencia máxima de 5e-5 mmHg y rechaza señales con NaN/Inf, forma o dtype incorrectos y segmentos planos.
