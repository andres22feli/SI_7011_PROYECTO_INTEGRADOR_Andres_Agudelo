# Plan Maestro de Ejecución — Proyecto Deep Learning: Estimación de Presión Arterial (MIMIC-BP)

> **Asignatura:** SI7011 — Aprendizaje Profundo (Universidad EAFIT)  
> **Referencia directa:** [Anteproyecto_DL_Presion_Arterial_Andres_Agudelo.pdf](Anteproyecto_DL_Presion_Arterial_Andres_Agudelo.pdf)  
> **Guía Metodológica:** Skill `flujo-trabajo-dl` (principios, reglas "nunca", ciclo invariante y entregable reproducible)

---

## 0. Entorno de Ejecución con Conda

Para aislar dependencias con soporte CUDA en GPU local (e.g. RTX 4060):

```bash
# 1. Crear entorno con Python 3.11/3.12
conda create -n dl_bp python=3.11 -y
conda activate dl_bp

# 2. PyTorch con aceleración CUDA 12.4 / 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Dependencias científicas, de seguimiento y modelado
pip install numpy scipy pandas scikit-learn matplotlib mlflow jupyter ipykernel pytest
pip install transformers peft accelerate
```

---

## 1. Estructura del Repositorio

El diseño desacopla la lógica reutilizable en un módulo `src/` para garantizar reproducibilidad exacta y que los 9 experimentos del diseño factorial (3 entradas × 3 arquitecturas) compartan exactamente las mismas funciones:

```
dl_bp_mimic_proyecto_integrador/
├── data/                                 # Datos crudos (no se modifican)
│   ├── abp/                              # pXXXXXX_abp.npy (30, 3750)
│   ├── ecg/                              # pXXXXXX_ecg.npy (30, 3750)
│   ├── ppg/                              # pXXXXXX_ppg.npy (30, 3750)
│   ├── labels/                           # pXXXXXX_labels.npy (30, 2) [PAS, PAD]
│   └── subject_ids.txt                   # 1524 IDs
├── cache/                                # Archivos intermedios generados
│   ├── split.json                        # Partición por paciente (914 / 305 / 305)
│   └── preprocessed/                     # Señales filtradas float32 en disco
├── src/bp/
│   ├── __init__.py
│   ├── contract.py                       # Validación de contrato y generación de data_card.json
│   ├── split.py                          # Partición por paciente 60/20/20 y verificación de fugas
│   ├── preprocess.py                     # Filtros Butterworth pasa-banda y z-score por instancia
│   ├── dataset.py                        # Dataset de PyTorch para las 3 modalidades de entrada
│   ├── models.py                         # Arquitecturas: MLP, GRU (con patching) y TimesFM (LoRA/Probe)
│   ├── train.py                          # Ciclo canónico de entrenamiento + MLflow + early stopping
│   ├── metrics.py                        # MAE, ME, DE por paciente sobre test
│   └── predict.py                        # Guardado de bundle y función predict_raw
├── notebooks/
│   ├── 01_datos_contrato_particion.ipynb # Contrato, data_card y partición
│   ├── 02_lineas_base_y_chequeos.ipynb   # Chequeos de sanidad, MSE inicial y línea base
│   ├── 03_experimentos_mlp.ipynb         # MLP sobre PPG, ECG y PPG+ECG
│   ├── 04_experimentos_gru.ipynb         # GRU sobre PPG, ECG y PPG+ECG
│   ├── 05_experimentos_timesfm.ipynb     # TimesFM 2.5 (Probe & LoRA) sobre las 3 entradas
│   └── 06_reporte_y_predict_raw.ipynb    # Test UNA sola vez, tabla comparativa 3x3 y validación predict_raw
├── artifacts/                            # Checkpoints (.pt), data_card.json, resultados.json
├── tests/                                # Pruebas unitarias de contrato y no-fuga
├── .gitignore
├── environment.yml
└── README.md
```

---

## 2. Fases de Implementación (Flujo Canónico SI7011)

### Fase 1: Contrato de Datos y Data Card (`src/bp/contract.py`)
1. **Regla:** Ningún dato se procesa sin validar su contrato.
2. **`validate_patient(subject_id)`**:
   - Comprueba existencia de `ppg`, `ecg` y `labels`.
   - Formas estrictas: señales `(30, 3750)` en float; etiquetas `(30, 2)`.
   - Ausencia total de `NaN` o `Inf`.
   - Varianza no nula (ningún segmento plano: $\text{std} > 10^{-6}$).
   - Rango fisiológico de etiquetas: $\text{PAS} \in [50, 250]$, $\text{PAD} \in [20, 150]$, y condición $\text{PAS} > \text{PAD}$.
3. **Chequeo de duplicados antes de partir:** Hashing SHA-1 sobre segmentos de señales para asegurar que un paciente no sea una réplica exacta de otro.
4. **Artefacto `data_card.json`:** Registro del DOI de MIMIC-BP (10.7910/DVN/DBM1NF), hash sha256 de `subject_ids.txt`, conteo de segmentos válidos (45 720) y estadísticas descriptivas de PAS y PAD.

### Fase 2: Partición por Paciente sin Fuga (`src/bp/split.py`)
1. **Regla de oro:** *La partición es por paciente, jamás mezclando segmentos al azar.* Mezclar segmentos del mismo paciente entre train y test generaría fuga severa por correlación intra-sujeto.
2. **Partición 60 / 20 / 20 con semilla fija (`seed=0`):**
   - **Train:** 914 pacientes (27 420 segmentos).
   - **Val:** 305 pacientes (9 150 segmentos).
   - **Test:** 305 pacientes (9 150 segmentos).
3. **Persistencia:** Guardar `cache/split.json`.
4. **Chequeo:** `assert len(set(train_ids).intersection(set(test_ids))) == 0`.
5. **DataLoaders:**
   - Train: `shuffle=True` con generador manual con semilla fija.
   - Val / Test: `shuffle=False` (evaluación determinista por paciente).

### Fase 3: Acondicionamiento y Normalización (`src/bp/preprocess.py`)
1. **Filtrado pasa-banda Butterworth de 4.° orden (fase cero con `scipy.signal.sosfiltfilt`):**
   - **PPG:** $0.5 - 8.0 \text{ Hz}$ (suprime deriva respiratoria y ruido de alta frecuencia).
   - **ECG:** $0.5 - 40.0 \text{ Hz}$ (conserva el complejo QRS y elimina deriva de línea base).
   - No altera el desfase fisiológico pulso-tránsito (PTT).
2. **Normalización de entradas:**
   - Estandarización z-score **por instancia** dentro de cada segmento de 30 s:
     $$x' = \frac{x - \mu_{\text{segmento}}}{\sigma_{\text{segmento}}}$$
   - No aprende parámetros del dataset $\to$ libre de fuga temporal o entre sujetos.
3. **Normalización de etiquetas:**
   - $\mu_y, \sigma_y$ de PAS y PAD calculadas **únicamente con el conjunto de entrenamiento**:
     $$y' = \frac{y - \mu_{y,\text{train}}}{\sigma_{y,\text{train}}}$$
   - Ambas variables quedan en escala comparable para que el optimizador no privilegie la PAS (de mayor magnitud) sobre la PAD.

### Fase 4: Arquitecturas y Líneas Base (`src/bp/models.py`)

#### Línea Base Obligatoria (LB-0)
- Predictor trivial que emite siempre $[\mu_{\text{PAS, train}}, \mu_{\text{PAD, train}}]$. Establece el umbral de MAE mínimo a superar.

#### 1. MLP (Línea Base Neuronal)
- Entrada aplanada: $C \times 3750$ valores ($C \in \{1, 2\}$).
- Bloques: `Linear(bias=False) -> BatchNorm1d -> ReLU -> [Dropout(0.2)]`.
- Cabezas intermedias: $512 \to 256 \to 128 \to 2$.
- Inicialización Kaiming He en pesos ocultos, sesgos en 0. Última capa inicializada en 0 para que la predicción inicial empiece exactamente en la media.

#### 2. GRU (Modelo Recurrente con Patching)
- Para evitar la pérdida de gradientes sobre 3 750 timesteps, se implementa un bloque de *patching*:
  - Ventanas de 25 muestras (0.2 s, frecuencia equivalente a 5 Hz) $\to$ Secuencia de longitud $T = 150$ pasos con dimensión de entrada $25 \times C$.
- `nn.GRU(input_size=25*C, hidden_size=64, num_layers=2, batch_first=True)`.
- Cabeza de regresión sobre el último estado oculto $h_T$: `Linear(64, 2)`.

#### 3. TimesFM 2.5 (Modelo Fundacional Adaptado)
- Uso del checkpoint `google/timesfm-2.5-200m-transformers`.
- Estrategia en dos pasos:
  1. **Probe lineal:** Backbone congelado, extracción del pool de representaciones contextuales, entrenamiento únicamente de una cabeza `Linear(hidden_dim, 2)`.
  2. **Fine-Tuning con LoRA:** Inserción de adaptadores LoRA ($r=8, \alpha=16$) en las capas de atención, entrenando con tasa de aprendizaje diferenciada (backbone $2\times 10^{-5}$, cabeza $1\times 10^{-3}$).
  - Para 2 canales (PPG + ECG), procesar cada canal por el backbone y concatenar sus representaciones antes de la cabeza de regresión.

### Fase 5: Chequeos Previos al Entrenamiento (Asserción Invariante)
Antes de entrenar las 9 combinaciones:
1. **Chequeo de pérdida inicial:** La MSE inicial estandarizada debe ser $\approx 1.0$ ($\pm 0.2$).
2. **Sobreajuste de un lote:** Tomar 64 ejemplos con AdamW y asegurar que la pérdida converja a $< 0.05$ en $< 300$ pasos.
3. **Chequeo de gradientes:** Normas de gradientes finitas y distintas de cero en todas las capas.
4. **Comportamiento `train()` vs `eval()`:** Confirmar que dos llamadas en `eval()` produzcan idéntico resultado numérico.

### Fase 6: Optimizador, Schedule y Presupuesto
- **Optimizador base:** `AdamW(lr=1e-3, weight_decay=1e-2)`.
- **Schedule:** Warmup lineal de 1 época + decaimiento por coseno con `LambdaLR`, avanzando **por lote**.
- **Gradient Clipping:** `clip_grad_norm_(model.parameters(), max_norm=1.0)` activo en GRU y TimesFM.
- **Early Stopping:** Monitoreo sobre `val_loss` con paciencia de 5 épocas, guardando `copy.deepcopy(model.state_dict())`.
- **Mismo presupuesto:** Batch size = 128 o 256; épocas máximas = 30; semilla fijada con `torch.manual_seed(0)` antes de instanciar cada variante.

### Fase 7: Registro con MLflow
- Servidor local SQLite: `sqlite:///mlflow.db`.
- Cada corrida registra:
  - Hiperparámetros: arquitectura, canales de entrada, learning rate, weight decay.
  - Métricas por época: `train_loss`, `val_loss`, `val_mae_pas_mmHg`, `val_mae_pad_mmHg`, `grad_norm`, `learning_rate`.
  - Artefactos: `data_card.json`, checkpoint `best.pt`.

### Fase 8: Evaluación por Paciente y Test una Sola Vez
1. Se selecciona la mejor corrida de cada una de las 9 combinaciones evaluando **únicamente en validación**.
2. **Evaluación en Test (una sola vez al final):**
   - Para cada uno de los 305 pacientes de prueba (30 segmentos por paciente):
     $$\text{Error}_i = \hat{y}_i - y_i$$
     $$\text{MAE} = \frac{1}{30}\sum |\text{Error}_i|, \quad \text{ME} = \frac{1}{30}\sum \text{Error}_i, \quad \text{DE} = \text{std}(\text{Error}_i)$$
   - Promediar métricas sobre los 305 pacientes.
3. **Tabla Final del Diseño Factorial (3 Entradas × 3 Arquitecturas):**

| Entrada | Modelo | MAE PAS (mmHg) | ME PAS (mmHg) | DE PAS (mmHg) | MAE PAD (mmHg) | ME PAD (mmHg) | DE PAD (mmHg) |
|---|---|---|---|---|---|---|---|
| — | Línea Base (Media) | ... | ... | ... | ... | ... | ... |
| PPG | MLP | ... | ... | ... | ... | ... | ... |
| PPG | GRU | ... | ... | ... | ... | ... | ... |
| PPG | TimesFM 2.5 | ... | ... | ... | ... | ... | ... |
| ECG | MLP | ... | ... | ... | ... | ... | ... |
| ECG | GRU | ... | ... | ... | ... | ... | ... |
| ECG | TimesFM 2.5 | ... | ... | ... | ... | ... | ... |
| PPG + ECG | MLP | ... | ... | ... | ... | ... | ... |
| PPG + ECG | GRU | ... | ... | ... | ... | ... | ... |
| PPG + ECG | TimesFM 2.5 | ... | ... | ... | ... | ... | ... |

*(Referencia clínica orientativa AAMI: ME $\le 5\text{ mmHg}$, DE $\le 8\text{ mmHg}$)*.

### Fase 9: Empaquetado y `predict_raw`
1. **Bundle portable (`artifacts/best_model_bundle.pt`):**
   ```python
   bundle = {
       'state_dict': model.state_dict(),
       'model_builder': 'build_gru',
       'inputs': ['ppg', 'ecg'],
       'fs': 125,
       'filters': {'order': 4, 'ppg_band': [0.5, 8.0], 'ecg_band': [0.5, 40.0]},
       'y_mean': torch.tensor(y_mean, dtype=torch.float32),
       'y_std': torch.tensor(y_std, dtype=torch.float32),
       'contract': contract_specs,
       'data_card': data_card_dict,
   }
   ```
2. **`predict_raw(ppg_signal, ecg_signal=None)`**:
   - Valida contrato sobre las señales en bruto.
   - Aplica filtros Butterworth y estandarización por instancia.
   - Reconstruye arquitectura y carga `state_dict`.
   - Pone `model.eval()` bajo `torch.no_grad()`.
   - Re-escala con `y_mean` y `y_std` devolviendo valores finales en mmHg.
   - **Test de verificación:** Comprueba que reproduce $>99.99\%$ de exactitud numérica con las predicciones del conjunto de prueba y que arroja excepciones descriptivas ante señales corruptas o NaN.

---

## 3. Checklist de Ejecución Rápida para Claude Code

1. [ ] Crear y activar el entorno Conda (`conda activate dl_bp`).
2. [ ] Crear `src/bp/` y escribir `contract.py`, `split.py`, `preprocess.py`, `dataset.py`, `models.py`, `train.py`, `metrics.py`, `predict.py`.
3. [ ] Ejecutar `01_datos_contrato_particion.ipynb` y generar `cache/split.json` y `artifacts/data_card.json`.
4. [ ] Ejecutar `02_lineas_base_y_chequeos.ipynb` (verificar MSE inicial $\approx 1.0$ y memorización de 64 muestras).
5. [ ] Ejecutar `03_experimentos_mlp.ipynb` (3 corridas registradas en MLflow).
6. [ ] Ejecutar `04_experimentos_gru.ipynb` (3 corridas registradas en MLflow).
7. [ ] Ejecutar `05_experimentos_timesfm.ipynb` (3 corridas con TimesFM 2.5 registradas en MLflow).
8. [ ] Ejecutar `06_reporte_y_predict_raw.ipynb` (evaluación final en Test una sola vez, tabla factorial 3x3 y exportación de `best_model_bundle.pt`).
