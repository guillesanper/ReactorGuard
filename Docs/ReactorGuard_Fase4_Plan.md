# Fase 4 (ML Core: PINN) - plan por hitos con guia de ejecucion

Estado: **APROBADO** (2026-11-10: D4-1 a D4-16 tal como se proponen, con un unico ajuste de calendario en D4-9:
la retirada de claves de `training:` pasa de M1 a M3). Plan redactado en F4-M0; **F4-M1 hecho** (seccion 15). Las
sondas del preflight se ejecutaron fuera del repo (scratchpad), leyendo `data/raw/simulation` sin modificarlo. Etiquetas:
**[M]** medida durante la planificacion, **[L]** leida del codigo o de un documento, **[W]** documentacion externa
no medida, **[C]** calculada a partir de cifras [M] (aritmetica, no medicion). Prosa sin tildes, como el resto del
repositorio.

## 0. Contexto

Fase 4 debe entrenar la PINN (TDD 5.2), validarla fisicamente, montar el Rule Engine (TDD 5.4), la fusion parcial
PINN + reglas, el chequeo de seguridad, SHAP y el benchmark de latencia. Criterios de Plan_Fases: satisfaccion de
constraints > 99 % en el test de simulacion, recall CRITICAL > 99,9 %, latencia p99 < 50 ms en CPU y todos los runs
en MLflow. Fase 3 esta cerrada en local (dataset de 34 runs x 1500 pasos x 27 sensores con verdad-terreno). Fase 2
sigue con los criterios 1 y 3 NO MEDIDOS hasta el cluster; este plan no toca GCP ni el camino de streaming.

## 1. Linea base [M] - rama `feat/fase4-plan` desde `afa5e85`, arbol limpio

| Puerta | Resultado |
|---|---|
| `git status` / `git log` | limpio; HEAD `afa5e85` (F3-M7 commiteado); `main` = `d04b1f6`; PR de `feat/stage-simulate` pendiente |
| `ruff check .` | All checks passed (venv: `ruff 0.15.22`; el extra `dev` fija `0.17.3`) |
| `mypy ml/ api/ data/` | 2 errores `no-untyped-call` en `data/storage/parquet_codec.py` (entorno: `mypy 2.3.0` frente a 2.4.1 y `pyarrow 24.0.0` sin tipos; M7-4) |
| `pytest tests/unit tests/safety --cov --cov-fail-under=70` | **2867 passed, 1 skipped**, TOTAL **99,55 %** (7576 sentencias, 34 sin cubrir), **817 s** (13 min 37 s); sin eventos Kernel-Power 506/507 en el intervalo |
| `dvc status` | Data and pipelines are up to date |
| `Verify-Phase2.ps1 -SkipCluster` | salida **2** (criterios 2 y 4 CUMPLIDOS, 1 y 3 NO MEDIDOS), 32 s |
| `Verify-Phase3.ps1` | salida **0** (4/4 CUMPLIDOS), 30 s, sin `-FromScratch` (el pipeline no cambio desde F3-M7) |

La suite tardo 817 s frente a los 33 min 15 s del cierre de F3-M7: no se investigo la diferencia (misma maquina,
sin suspensiones en ninguna de las dos).

## 2. Preflight [M] salvo indicacion

### 2.1 Entorno y andamiaje

| Pregunta | Resultado |
|---|---|
| Versiones | `torch 2.13.0+cpu` (4 hilos intra-op, MKL 2026.1, sin CUDA), `mlflow 3.14.0`, `mapie 1.4.1`, `shap 0.52.0`, `scikit-learn 1.9.0`, `numpy 2.4.6`, `pandas 2.3.3`, `pyarrow 24.0.0`, `scipy 1.18.0`, `dvc 3.67.1`, `feast 0.64.0`. CPU i5-1135G7 4C/8T |
| `ml/models/pinn.py` [L] | `ReactorPINN` es un **LSTM** (4 capas, 256) con cabeza de reconstruccion y cabeza de anomalia sigmoide; **no es la red del TDD 5.2** (MLP `n+4 -> 256 Tanh x3 -> n`, prediccion del paso siguiente). `physics_residual` penaliza `dP/dt` del canal 0 (supuesta potencia), con `.detach()`: no hay perdida de energia ni de masa |
| `ml/training/train.py` [L] | sin loop; `input_size` sale de `features.sensor_selection.expected_sensor_count` (**52, TEP**); registra `placeholder_loss` en MLflow |
| `ml/serving/predictor.py` [M] | **no se puede importar**: `from mapie.regression import MapieRegressor` lanza `ImportError` con `mapie 1.4.1` (la API 1.x ofrece `SplitConformalRegressor`, `CrossConformalRegressor`, ...). Nada lo importa |
| Tests del andamiaje [L] | solo `tests/unit/test_pinn_forward.py` (3 tests de forma y rango del LSTM). `ml/models`, `ml/training`, `ml/serving` estan **fuera** de `[tool.coverage.run] source` (`data`, `ml/features`); `mypy` si los cubre |
| MLflow, backend por defecto | `sqlite:///<cwd>/mlflow.db`: `train.py` ejecutado en la raiz crearia `mlflow.db`, que **no esta en `.gitignore`** (solo `mlruns/`, `*.pt`, `*.pth`) |
| MLflow, backend de ficheros | **rechazado**: `MlflowException: The filesystem tracking backend (e.g., './mlruns') is in maintenance mode` salvo `MLFLOW_ALLOW_FILE_STORE=true` |
| MLflow sqlite, coste | `set_experiment` 2,69 s (creacion del esquema); un run con 2 params y 100 pasos x 5 metricas 1,33 s |

### 2.2 Dataset simulado como entrada de la PINN

| Pregunta | Resultado |
|---|---|
| Lectura | 34 parquet en 1,59 s; 1.377.000 filas; **1.244 MB** en memoria en formato largo (pandas, `deep=True`); pivotado a ancho de los 34 runs (`value` y `true_value`) 1,97 s |
| Matriz | 51.000 instantes x 27 sensores; 50.966 pares (t, t+1) dentro de un run |
| Instantes etiquetados (algun sensor con fallo) | 18.839 (36,9 %); **CRITICAL 4.733** instantes (9,28 %), 91.062 lecturas |
| CRITICAL por tipo (instantes) | excursion 1.653, vacio 1.653, bloqueado 1.427 (3 runs cada uno) |
| Modos (tipo x grupo x nivel) | 144/144 presentes; episodios por modo: **108 modos con 1 episodio**, 12 con 2, 6 con 3, 18 con >= 90 (pico); solo 24 modos (16,7 %) tienen >= 3 episodios y 23 >= 3 runs. Episodios por run: proceso 1, deriva/sesgo/bloqueado 6, intermitente 8-9, pico 236-237 |

**Cobertura de la rejilla y CRITICAL por particion con tres repartos** (k_min = 10 lecturas, como D6):

| Reparto | Rejilla train / val / test | Instantes CRITICAL train / val / test |
|---|---|---|
| (a) actual: temporal dentro de cada tipo sobre runs concatenados | 75,0 / 38,2 / 40,3 % | 2.920 / 1.033 / 780 |
| (b) por run, estratificado (ultimo run de cada escenario a test, penultimo a val) | 50,0 / 41,7 / 42,4 % | 1.474 / 1.446 / 1.813 |
| (c) temporal dentro de cada run (70/15/15 de cada run) | 95,8 / 62,5 / 42,4 % | 3.764 / 780 / 189 |

Ningun reparto por run o por episodio puede llevar a val **y** a test los 108 modos de un solo episodio: es un
limite del dataset (F3, M5-5), no del reparto.

### 2.3 Fisica sobre los datos medidos (5 columnas de la nota de arranque, mediana por rol, cp de tabla)

| Magnitud (instantes normales, n = 32.161) | p50 | p99 | max |
|---|---|---|---|
| Residuo de energia \|Q_gen - Q_rem\| / Q_gen, medido | 0,66 % | 2,51 % | 4,13 % |
| Idem sobre `true_value` | 0,0075 % | 0,153 % | 0,71 % |
| Residuo de masa \|m_hl - m_cl\| / m_cl, medido | 0,67 % | 2,54 % | 4,28 % |

La forma estacionaria de la perdida de energia (sin dU/dt) deja un suelo de 0,15 % (p99) sobre el estado verdadero:
basta para una perdida, no para una igualdad exacta.

**Lineas base triviales del predictor a un paso** (datos normalizados z por sensor, instantes normales): persistencia
(`x(t+1) = x(t)`) MSE **0,0833**; ruido de medida (`value - true_value`) MSE **0,0331**. Un MLP util debe quedar
entre ambas cifras.

**Rule Engine existente como detector** (`evaluate_rules` de `data/simulation/physics_constraints.py`, votado,
0,44 s el dataset entero): **0 / 32.161** instantes normales marcados; instantes CRITICAL detectados **1.727 / 4.733
(36,5 %)**: vacio 1.499 / 1.653 (90,7 %), bloqueado 228 / 1.427 (16,0 %), **excursion 0 / 1.653**. Sobre todas las
etiquetadas, 5.202 / 18.839 (27,6 %); caudal 8 / 1.558.

### 2.4 Coste de modelo en CPU (4 hilos, matriz medida, lote 256, Adam)

| Modelo | Parametros | Una epoca (35.676 pares, 70 %) | Inferencia de 1 muestra p50 / p99 |
|---|---|---|---|
| MLP del TDD 5.2 (31 -> 256 Tanh x3 -> 27) | 146.715 | **0,47 / 0,45 / 0,54 s** (3 epocas) | **0,073 / 0,110 ms** |
| LSTM del andamiaje, ventana 20 (1 h) | 1.924.252 | 73,3 s | 2,44 / 3,26 ms |
| LSTM del andamiaje, ventana 60 (3 h) | 1.924.252 | 209,7 s | 4,49 / 5,32 ms |

- Forward del MLP sobre los 50.966 pares: 0,108 s. Lote de 27 muestras: p99 0,30 ms (un maximo aislado de 43 ms).
- **Determinismo**: una epoca del MLP con `torch.manual_seed(42)` da el mismo digest de pesos (`89c684bf...`) en 3
  procesos distintos y con 1 y 4 hilos OpenMP.
- **SHAP** `GradientExplainer` (fondo de 100 muestras, `nsamples` = 200) sobre el MLP: **787 ms p50**, 1.017 ms max
  por muestra. No cabe en el presupuesto de 50 ms.
- [C] 100 epocas del MLP ~ 50 s; una rejilla de 9 combinaciones de lambda ~ 8 min. Con el LSTM de ventana 60, una
  sola configuracion de 100 epocas ~ 5,8 h.

## 3. Lo que dice el TDD [L]

- **5.2 PINN**: MLP `Linear(n_sensors + 4, 256) -> Tanh -> Linear(256,256) -> Tanh -> Linear(256,256) -> Tanh ->
  Linear(256, n_sensors)`, "+4: time features", "predict next state". Residuo prediccion - medicion = senal de
  anomalia. `energy_balance_loss` con `state[:, 0..3] = flux, flow, t_in, t_out`, `Cp = 4182`, `q_gen = flux * 3.2e-11
  * 2.5e18`; `mass_balance_loss` con `state[:, 1]` frente a `state[:, 4]`; `loss = data + lambda_e energy + lambda_m
  mass`, lambdas 1,0.
- **5.4 Rule Engine**: 6 reglas deterministas; si una se viola, severidad CRITICAL sin importar el modelo.
- **5.1 Fusion**: `0,35 pinn + 0,35 bnn + 0,30 reglas` (Fase 5); Plan_Fases pide en Fase 4 la parcial `0,5 pinn + 0,5
  reglas`.
- **2.3**: latencia p99 < 50 ms; tasa de falsos negativos < 0,1 % en CRITICAL; SHAP por prediccion.
- **9.2**: `physics_check.py --min-satisfaction 0.99` ("99 % of predictions must satisfy all physics constraints"),
  `safety_check.py --fault-types stuck,drift,thermal_excursion --min-recall 0.999`.
- **9.3 / 13.3**: stage `train` con `cmd: python ml/training/train_pinn.py`, deps en features y `ml/models/pinn.py`,
  outs `models/pinn/candidate/`; "starts with TEP, ~30min on CPU".
- **App. B**: energia `Q_gen = Q_removed + dU/dt` como "PINN loss term + hard rule check"; masa solo como regla.

## 4. Hallazgos [M]/[L] y choques

| # | Hallazgo |
|---|---|
| H1 | El andamiaje no es la red del TDD (LSTM frente a MLP) y su `physics_residual` no implementa ninguna de las dos perdidas del TDD |
| H2 | `predictor.py` no importa con `mapie 1.4.1` [M]; `pyproject.toml` admite `mapie>=0.8.0` sin techo. Es Fase 5/6, pero cambiar `ReactorPINN` en F4 rompe ademas su uso de `model(x)` como tupla |
| H3 | MLflow 3.14 rechaza el backend de ficheros y por defecto crea `mlflow.db` (sin ignorar) en el directorio de trabajo [M] |
| H4 | El MLP cuesta 0,5 s por epoca y 0,11 ms p99 por muestra: el criterio de latencia y la rejilla de lambdas son baratos. El LSTM es 150-450 veces mas caro por epoca [M] |
| H5 | El entrenamiento del MLP es determinista entre procesos y con 1/4 hilos [M]: un stage `train` reproducible bit a bit es plausible; falta medir los bytes de `torch.save` |
| H6 | El Rule Engine ya existe y esta probado (`data/simulation/physics_constraints.py`, 7 reglas, 100 % de cobertura): T7.3 es un adaptador, no una reimplementacion |
| H7 | Las reglas solo ven el 36,5 % de los instantes CRITICAL y **ninguno de la excursion** [M]: el recall CRITICAL depende de la PINN |
| H8 | 108 de 144 modos tienen un solo episodio [M]: ninguna particion de val/test puede cubrir la rejilla |
| H9 | `featurize` exige los 52 sensores del TEP y `train.py` toma de ahi `input_size` (52); el dataset simulado tiene 27 y no esta cableado (D13 de F3) [L] |
| H10 | `training:` de `params.yaml` mezcla hiperparametros de la PINN (sin consumidor real) con las claves del stage `split` de Fase 2 (`train_ratio`, `splits_dir`, ...) [L] |
| H11 | SHAP `GradientExplainer` cuesta 0,8-1,0 s por muestra en CPU [M] |
| H12 | Pandas en formato largo ocupa 1,24 GB para el dataset; el ancho es 51.000 x 27 float32 (5,5 MB) [M] |

### Choques

| # | Choque |
|---|---|
| C1 | Plan_Fases ("primera ejecucion con dataset TEP") frente a la fisica: el TEP es una planta quimica sin roles de flujo neutronico, caudal de primario ni temperaturas de nucleo; la perdida de energia/masa no tiene sentido sobre el |
| C2 | Plan_Fases y TDD 9.2 nombran como CRITICAL `stuck, drift, thermal_excursion`; la tabla TDD 2.2 (D10, vigente) da CRITICAL a bloqueado, excursion y **vacio**, y HIGH a la deriva |
| C3 | Recall > 99,9 % no es certificable con el test: con 0 fallos en n instantes, la cota superior al 95 % de la tasa de fallos es ~3/n [C]; con 1.813 instantes CRITICAL en test (reparto b) la cota es 0,17 % > 0,1 %. Ademas los instantes de un episodio no son independientes |
| C4 | TDD 5.2: `Cp = 4182` y `3.2e-11 x 2.5e18` no reconcilian el balance (F3, C8/D8): se usa cp de tabla y `Q_gen = P0 x flux / 100` |
| C5 | TDD 5.2 "+4: time features" no define cuales; con datos sinteticos la hora del dia no tiene contenido fisico |
| C6 | "Physics constraint satisfaction > 99 %" no dice sobre que se evalua (prediccion o medicion), sobre que instantes (normales o todos) ni que reglas (la de dT/dt es inobservable a 180 s, F3 C5) |
| C7 | TDD 5.4 son 6 reglas; el modulo de F3 tiene 7 (`mass_balance` de App. B) |
| C8 | Latencia "< 50 ms" sin definir el camino medido; SHAP no cabe en el (H11) |
| C9 | Stage `train` del TDD con deps en `data/processed/features/` (TEP, 52 sensores); la PINN de este plan entrena sobre la simulacion |

## 5. Decisiones (D4-1 a D4-16) - APROBADAS el 2026-11-10

Cada una con la recomendacion primero; el usuario aprobo la recomendada en las 16, incluidas las cuatro marcadas
**(criterio)**, que cambian la lectura de Plan_Fases. Unico ajuste (F4-M1): en D4-9 la retirada de claves de
`training:` se hace en M3 y no en M1, porque el andamiaje `ml/training/train.py` lee seis de ellas (`hidden_size`,
`n_layers`, `dropout`, `learning_rate`, `weight_decay`, `physics_lambda`) y retirarlas antes de reescribirlo lo dejaria
roto dos hitos.

| # | Decision | Propuesta (recomendada) | Alternativas |
|---|---|---|---|
| D4-1 | Arquitectura | **MLP del TDD 5.2** (prediccion del paso siguiente a 180 s); se retira el LSTM. Dropout 0,1 entre capas ocultas (riesgo de Plan_Fases) como parametro | LSTM con ventana (x150-450 coste por epoca, H4); MLP con k retardos de entrada como variante de la rejilla |
| D4-2 | Las "+4" entradas (C5) | Contexto de operacion: `power_level` (mediana de `P_th` / 100), `dP_th/dt` normalizada, `sin` y `cos` de la hora del dia (TDD 5.5 cita `hour_of_day` y `power_level_pct`) | 4 retardos de potencia; sin entradas extra (n -> n) |
| D4-3 | Estado fisico y orden (C9 de F3) | `[flux, flow_in(FT-CL), t_in(TC-CL), t_out(TC-HL), flow_out(FT-HL)]` = mediana de los canales de cada rol de `configs/sim_sensors.yaml`, calculado sobre la **prediccion desnormalizada**; el orden lo fija una constante con test de contrato | Media en vez de mediana; por canal |
| D4-4 | Perdida de energia | `((Q_gen - Q_rem) / P0)^2` con `Q_gen = P0 flux/100`, `Q_rem = n_loops m_in cp (t_out - t_in)`, **cp de `configs/water_table.yaml`** evaluado en (T_avg, P) predichos y tratado como coeficiente sin gradiente (fuera de rejilla la fila no cuenta). Forma estacionaria: suelo 0,15 % p99 en el estado verdadero [M] | cp diferenciable (interpolacion bilineal en torch); forma con dU/dt |
| D4-5 | Perdida de masa | `((m_out - m_in) / m_nom)^2` solo en filas cuasi-estacionarias (mascara de `quasi_steady_dt_avg_k_s` sobre la ENTRADA medida, como la regla) | Sin mascara (penalizaria el insurge del presurizador) |
| D4-6 | Que se entrena | Solo pares (t, t+1) **sin etiqueta en ninguno de los dos instantes** (normales de todos los runs, incluidos los tramos sanos de runs con fallo); objetivo = `value` medido de t+1; z-score por sensor con estadisticos de train guardados en el checkpoint | Objetivo `true_value` (denoising); entrenar tambien con fallos y cabeza de clasificacion |
| D4-7 | Particiones (C9, H8) | **Por run, estratificado** (reparto b): ultimo run de cada escenario a test, penultimo a val, resto a train (normal 2/1/1). Sin fuga entre particiones dentro de un run; CRITICAL equilibrado (1.474 / 1.446 / 1.813) [M]. La cobertura de la rejilla en val/test (~42 %) se informa, no es criterio de F4 | (c) temporal por run (val/test con solo 780/189 CRITICAL); (a) el actual |
| D4-8 | Cableado (H9, C9) | Stage nuevo `prepare_sim` que lee `data/raw/simulation` y escribe la matriz ancha por run, etiquetas por instante y la asignacion de particion; **`featurize`, `split` y el TEP no se tocan**. La primera ejecucion con TEP de Plan_Fases (C1) se marca **NO APLICA** | Cablear la simulacion a `featurize` (exige generalizar los 52 sensores); smoke con TEP y lambda = 0 |
| D4-9 | Parametros (H10) | Seccion nueva **`pinn:`** en `params.yaml` (M1, solo con las claves que algun stage lee); se retiran de `training:` las claves sin consumidor real (`learning_rate`, `weight_decay`, `optimizer`, `batch_size`, `max_epochs`, `early_stopping_patience`, `physics_lambda`, `hidden_size`, `n_layers`, `dropout`) **en M3, junto con la reescritura de `train.py`**; las del stage `split` quedan intactas (no lo invalida) | Ampliar `training:` |
| D4-10 | Conjunto CRITICAL **(criterio)** (C2) | Tabla estatica TDD 2.2 (D10): **bloqueado, excursion y vacio** | La lista de Plan_Fases/TDD 9.2 (bloqueado, deriva, excursion) |
| D4-11 | Unidad y certificacion del recall **(criterio)** (C3) | Recall por **instante** CRITICAL del test con el umbral calibrado en val; se informa ademas la cota de Clopper-Pearson al 95 % y el recall por episodio. CUMPLIDO si la estimacion puntual > 99,9 %; la cota se publica y no decide | Exigir la cota (no alcanzable sin mas datos CRITICAL: decision de Fase 3); evaluar sobre val+test (sesgo por calibrar en val) |
| D4-12 | Satisfaccion fisica **(criterio)** (C6, C7) | Fraccion de instantes de **test (todos)** cuyo estado **predicho** desnormalizado no viola ninguna de las 7 reglas evaluables (dT/dt incluida aunque a 180 s no dispare); se informa aparte el subconjunto normal. Referencia: la medicion normal viola 0/32.161 [M] | Solo instantes normales; solo energia y masa |
| D4-13 | Rule Engine y fusion (T7.3/T7.4) | `ml/models/rules.py` = adaptador sobre `data.simulation.physics_constraints` (7 reglas, votadas por defecto y por canal como opcion medida en M6). Fusion `0,5 s_pinn + 0,5 r`, `s_pinn = min(1, residuo / tau)` con `tau` en un percentil de los normales de val; regla violada -> CRITICAL (override del TDD) | Reimplementar las reglas en ml/ |
| D4-14 | MLflow (H3) | URI explicita en `pinn.mlflow_tracking_uri: sqlite:///mlruns/mlflow.db` (bajo `mlruns/`, ya ignorado) y artefactos en `mlruns/artifacts`; `mlflow.db` se anade a `.gitignore`. El store no es out de DVC. "Todos los runs logeados" = runs en el store == ejecuciones declaradas por el manifiesto de la rejilla | Servidor MLflow; `MLFLOW_ALLOW_FILE_STORE=true` (backend en mantenimiento) |
| D4-15 | Latencia y SHAP **(criterio)** (C8, H11) | p99 de una prediccion de un instante (normalizar + forward + desnormalizar + residuo + reglas), en proceso, CPU 4 hilos, tras calentamiento, N = 10.000. SHAP **fuera** del camino de 50 ms (bajo demanda, como `/explain` en Fase 6); se mide su coste aparte y se ofrece top-3 por prediccion | Incluir SHAP en el camino (inviable: 787 ms [M]); gradiente x entrada como aproximacion barata (no es SHAP) |
| D4-16 | `predictor.py` y MAPIE (H2) | **No se toca en F4** (es Fase 5/6); se documenta como roto. `ml/serving` sigue fuera de cobertura | Adaptarlo minimamente al nuevo `ReactorPINN` y a MAPIE 1.x ya en F4 |

Decisiones abiertas de fases anteriores que **afectan** a F4: **M7-4/M6-7** (fijar `pandas<3` y `pyarrow`): el
stage `prepare_sim` escribira parquet y heredara la misma dependencia de bytes; **M7-1** (criterio de volumen) no
afecta. **M5-6** (votar o no las reglas) se decide en F4-M6 con la medicion delante (D4-13).

## 6. Convenciones transversales (aplican a todos los hitos)

1. **Layout**: `ml/models/` (red, estado fisico, perdidas, reglas, fusion), `ml/training/` (datos, loop, rejilla,
   parametros), `ml/validation/` (`physics_check.py`, `safety_check.py`, benchmark). Cada modulo que se implemente
   entra en `[tool.coverage.run] source` (`ml/models`, `ml/training`, `ml/validation`); `ml/serving` queda fuera (D4-16).
2. **Loaders**: patron de `data/generators/tep_params.py` / `data/simulation/simulation_params.py`
   (`DEFAULT_PARAMS_PATH`, `_SECTION = "pinn"`, dataclass `frozen=True`, `_require` con `KeyError` con la clave
   completa, `ValueError` con cifras).
3. **`main()` fino** como `data/generators/adapt_tep.py`; la logica en libreria testeable sin MLflow real (cliente
   inyectable o URI temporal sqlite en `tmp_path`).
4. **Aleatoriedad**: `torch.Generator` y `numpy.random.Generator` sembrados desde `pinn.seed`; `torch.manual_seed`
   solo en el punto de entrada; `torch.use_deterministic_algorithms(True)`; hilos fijados por parametro
   (`pinn.num_threads`). Prohibido `random` y `np.random.seed`.
5. **Salidas deterministas**: checkpoint con solo tensores y primitivos (`weights_only=True` compatible: pesos,
   `model_kwargs`, medias y desviaciones, orden de columnas, version de contrato); `model_card.json` y metricas sin reloj
   ni rutas absolutas, `sort_keys=True`. El reloj y el host van solo a MLflow.
6. **Unidades**: la perdida fisica opera en unidades de ingenieria (C, kg/s, bar, %); nunca sobre z-scores.
7. **Tests**: un `tests/unit/test_<modulo>.py` por modulo, clases por comportamiento, `tmp_path`, sin red; datasets
   pequenos con `tests/support/sim_frames.py` (se extiende, no se recrea); entrenamientos de test < 10 s. Cobertura
   >= 95 % por modulo nuevo. Tests de gradiente: la perdida fisica es 0 sobre el estado verdadero estacionario y su
   gradiente apunta a reducir el desbalance.
8. **Estandares**: ruff E,W,F,I,N,UP,S,B,A, line-length 100 (tambien tests); mypy strict; docstrings Google; `_LOG` con
   %-lazy; sin emojis; prosa en espanol sin tildes; Python >= 3.11.
9. **No se toca en toda la fase**: `data/schemas/*`, `data/validation/*`, `data/streaming/*`, `ml/features/*`,
   `data/simulation/*` (solo se importa), los stages `download_tep..split` y `simulate`, `Verify-Phase2.ps1`,
   `Verify-Phase3.ps1`. NO actualizar `pyarrow` ni `pandas` en el venv.
10. **Al abrir cada hito**: listar paths a crear/tocar y confirmar linea base. **Al cerrarlo**: puertas completas,
    cifras en tabla, filas T6.x/T7.x de Progreso, registro (seccion 15), mensaje de commit (el usuario commitea).

## 7. Hitos: que tocar y como

Orden: **M1 -> M2 -> M3 -> M4 -> M5 -> M6 -> M7 -> M8**. M7 (SHAP y latencia) puede adelantarse tras M3.

### F4-M1 - Datos de entrenamiento: stage `prepare_sim`

**Crear**: `ml/training/pinn_params.py` (loader de `pinn:`), `ml/training/sim_dataset.py` (lee parquet ->
matriz ancha por run en el orden del catalogo, etiquetas por instante: `labelled`, severidad maxima, tipo por eje,
episodios activos; asignacion de particion D4-7; pares validos D4-6; estadisticos z de train), `ml/training/prepare_sim.py`
(`main()` fino), tests de los tres.
**Tocar**: `params.yaml` (seccion `pinn:`; la retirada de D4-9 pasa a M3), `dvc.yaml` (stage, seccion 8),
`pyproject.toml` (cobertura), `.gitignore` (`mlflow.db`), `Invoke-Pipeline.ps1` (resumen de `data/processed/sim_pinn`).
**Salida**: `data/processed/sim_pinn/{train,val,test}.parquet` (ancho, una fila por instante, con `run_id`, `timestep`
y etiquetas) + `assignment.json` (run -> escenario y particion, orden de sensores) + `metrics/sim_pinn_metrics.json`
(pares por particion, CRITICAL por particion, cobertura informativa).
**Hecho (2026-11-10)**: ver seccion 15. Desviaciones respecto a lo escrito arriba, todas por evidencia del hito:
(1) los episodios de instrumento se SOLAPAN (8.160 instantes con mas de uno activo [M]), asi que en lugar de un
`episode_id` por instante se guardan `episode_ids` (todos, ordenados, unidos por `|`) y `n_episodes`; (2) el reparto se
parametriza con `pinn.val_runs_per_scenario` y `pinn.test_runs_per_scenario` en vez de `pinn.split_strategy` (una sola
estrategia no es un parametro) y `pinn.seed` no se declara (el reparto es determinista, sin aleatoriedad); (3) la
retirada de `training:` pasa a M3 (seccion 5).
**Medir**: que reproduce las cifras de 2.2 (1.474 / 1.446 / 1.813 CRITICAL), tiempo del stage, bytes, y que dos
ejecuciones dan el mismo hash. **Verificar**: `Verify-Phase2 -SkipCluster` = 2 y `Verify-Phase3` = 0 (el stage nuevo
entra en `dvc.lock`: el criterio 4 de Fase 2 compara todas las etapas, H13 de F3).

### F4-M2 - Modelo y perdidas fisicas (T6.1)

**Reescribir** `ml/models/pinn.py`: `ReactorPINN(n_sensors, n_context=4, hidden=256, n_hidden=3, dropout)` del TDD;
`forward(x) -> y`. **Crear** `ml/models/physics_state.py` (roles -> indices, `PINN_STATE_ORDER`, desnormalizar,
estado de 5 componentes por mediana) y `ml/models/physics_loss.py` (`energy_balance_loss`, `mass_balance_loss`,
`PhysicsLossTerms`, cp de `WaterTable` sin gradiente, mascaras de rejilla y cuasi-estacionario).
**Reescribir** `tests/unit/test_pinn_forward.py` y crear `test_physics_state.py`, `test_physics_loss.py`.
**Medir**: perdidas sobre `true_value` y sobre `value` de los runs normales (deben reproducir 2.3), coste por lote.

### F4-M3 - Loop de entrenamiento y MLflow (T6.2, T6.3, T6.4)

**Retirar** de `training:` las diez claves de D4-9 (avisando antes) y sacar `ml/training/train.py` del `omit` de
cobertura de `pyproject.toml`. **Reescribir** `ml/training/train.py`: Adam, `ReduceLROnPlateau`, parada temprana por perdida de val, curriculum
lambda 0 -> objetivo en `pinn.curriculum_epochs` (50 por defecto, riesgo de Plan_Fases), registro por epoca (`data`,
`energy`, `mass`, `total`, `val_*`, `lr`) y params en MLflow (D4-14), checkpoint y `model_card.json`. Stage `train`
(seccion 8). **Medir**: convergencia (curva), epocas hasta parar, tiempo del stage, **bytes de `model.pt` entre dos
ejecuciones y entre procesos** (H5); si `torch.save` no es estable, guardar con un serializador propio de tensores y
documentarlo. MSE de val frente a persistencia (0,0833) y ruido (0,0331).

### F4-M4 - Validacion fisica (T6.5)

**Crear** `ml/validation/physics_check.py` (D4-12): predice el test, desnormaliza, pivota a frame ancho con
`DatetimeIndex` y aplica `evaluate_rules_detailed`; escribe `metrics/pinn_physics_metrics.json` (satisfaccion total,
normal, por regla, no evaluables). Stage `evaluate` (seccion 8). **Medir** el criterio 1 con el modelo de M3.

### F4-M5 - Rejilla de lambdas (T7.1, T7.2)

**Crear** `ml/training/grid_search.py`: rejilla `lambda_energy x lambda_mass` (propuesta {0; 0,1; 1; 10}^2 = 16
combinaciones, [C] ~15 min con 100 epocas maximas), un run de MLflow por combinacion con tag de rejilla, manifiesto
`metrics/pinn_grid.json` (combinacion, run_id de MLflow, val_data, val_energy, val_mass, satisfaccion en val). Seleccion
por regla escrita (minimo de `val_data` con satisfaccion en val >= 99 %); el ganador se copia a `params.yaml` a mano
(cambio visible en el diff). Fuera de DVC (coste y MLflow); el stage `train` usa el lambda elegido.

### F4-M6 - Rule Engine, fusion y seguridad (T7.3, T7.4, T7.5)

**Crear** `ml/models/rules.py` (adaptador D4-13; medir votado frente a por canal y decidir M5-6 con datos),
`ml/models/fusion.py` (parcial 0,5/0,5, normalizacion del residuo, override de reglas; el BNN entra en Fase 5),
`ml/validation/safety_check.py` (umbral y `tau` calibrados en val; recall CRITICAL por instante y por episodio en test,
cota de Clopper-Pearson, tasa de falsos positivos en normales de test, recall por tipo y nivel) ->
`metrics/pinn_safety_metrics.json`. **Medir** el criterio 2. Si el recall sale bajo (H7: la excursion no la ve ninguna
regla), se informa por tipo y se vuelve al usuario antes de tocar nada.

### F4-M7 - SHAP y latencia (T7.6, T7.7)

**Crear** `ml/validation/explain.py` (top-3 sensores por prediccion del residuo con `shap.GradientExplainer`, fondo de
normales de train; medir coste frente a `nsamples`) y `ml/validation/latency_benchmark.py` (D4-15) ->
`metrics/pinn_latency.json` con entorno, hilos, N, p50/p95/p99/max. Fuera de DVC (no determinista).

### F4-M8 - Verificacion y cierre de fase

**Crear** `infra/scripts/Verify-Phase4.ps1` (4 capas, CUMPLIDO/FALLADO/NO MEDIDO, salidas 0/1/2/3, informe
`tests/results/phase4_verification.json`, helpers minimos copiados de `Verify-Phase3.ps1` sin refactorizarlo),
`tests/unit/test_verify_phase4_script.py` (artefactos sinteticos, patron de `test_verify_phase3_script.py`),
`Docs/ReactorGuard_Fase4_PINN.md` (referencia: modelo, perdidas, datos, como regenerar, limites). **Tocar**: Progreso
(T6.x/T7.x), este plan (seccion 15).

## 8. Stages DVC

```
prepare_sim:                                               # HECHO en F4-M1, tal cual en dvc.yaml
  cmd: python ml/training/prepare_sim.py                  # siempre via Invoke-Pipeline.ps1
  deps: [ml/training/prepare_sim.py, ml/training/sim_dataset.py, ml/training/pinn_params.py,
         data/simulation, configs/sim_sensors.yaml, configs/plant_parameters.yaml, data/raw/simulation]
  params: [pinn.data_dir, pinn.metrics_path, pinn.val_runs_per_scenario, pinn.test_runs_per_scenario,
           simulation.output_dir, simulation.sensors_path, simulation.plant_parameters_path]
  outs: [data/processed/sim_pinn]
  metrics: [metrics/sim_pinn_metrics.json: {cache: false}]

train:
  cmd: python ml/training/train.py
  deps: [ml/training/train.py, ml/training/pinn_params.py, ml/models/pinn.py, ml/models/physics_state.py,
         ml/models/physics_loss.py, data/simulation/water_properties.py, configs/water_table.yaml,
         configs/sim_sensors.yaml, configs/plant_parameters.yaml, data/processed/sim_pinn]
  params: [pinn]                                           # seccion entera salvo la URI de MLflow (se excluye)
  outs: [models/pinn/candidate]                            # model.pt + model_card.json
  metrics: [metrics/pinn_train_metrics.json: {cache: false}]

evaluate:
  cmd: python ml/validation/physics_check.py && python ml/validation/safety_check.py
  deps: [ml/validation/physics_check.py, ml/validation/safety_check.py, ml/models, models/pinn/candidate,
         data/processed/sim_pinn, data/simulation/physics_constraints.py]
  metrics: [metrics/pinn_physics_metrics.json, metrics/pinn_safety_metrics.json]   # cache: false
```

- `models/pinn/candidate` no esta ignorado por git salvo `*.pt`: anadir `models/` a `.gitignore` en M3 (out de DVC).
- MLflow escribe en `mlruns/` y no participa en el hash (D4-14). Los params de `train` excluyen la URI.
- Trampas heredadas: `Invoke-Pipeline.ps1` siempre; DVC borra los outs antes de ejecutar; `WinError 3` con rutas largas
  (copias en `%TEMP%\rg-<8hex>`, `--no-run-cache`); md5 de ficheros con CRLF (lock solo en Windows).

## 9. Reutilizacion (no reescribir)

| Que | Donde | Uso |
|---|---|---|
| `evaluate_rules_detailed`, `ConstraintLimits`, `removed_heat_mw`, `quasi_steady_mask` | `data/simulation/physics_constraints.py` | Rule Engine, `physics_check`, mascara de masa |
| `WaterTable.cp`, `load_water_table` | `data/simulation/water_properties.py` | cp de la perdida de energia |
| `load_sensor_catalog`, `SensorCatalog.roles` | `data/simulation/sensor_catalog.py` | orden y roles del estado fisico |
| `wide_frame_from_long`, `labelled_mask`, `LONG_COLUMNS` | `data/simulation/dataset_builder.py` | `sim_dataset.py` |
| `FAULT_SEVERITY`, `ScenarioKind` | `data/simulation/scenario.py` | conjunto CRITICAL (D4-10) |
| patron loader, `main()` fino, Verify | `tep_params.py`, `adapt_tep.py`, `Verify-Phase3.ps1` | M1, M3, M8 |

## 10. Criterios de exito medibles

| # | Criterio | Umbral | Evidencia | Como decide `Verify-Phase4.ps1` |
|---|---|---|---|---|
| 1 | Satisfaccion fisica (D4-12) | > 99 % de instantes de test | `metrics/pinn_physics_metrics.json` | recalcula `satisfied / evaluated`; el `passed` del informe no decide |
| 2 | Recall CRITICAL (D4-10/11) | > 99,9 % por instante en test | `metrics/pinn_safety_metrics.json` | recalcula desde los conteos; publica la cota y el recall por episodio |
| 3 | Latencia p99 (D4-15) | < 50 ms en CPU | `metrics/pinn_latency.json` | exige N >= 10.000 y entorno `local-cpu`; sin fichero = NO MEDIDO |
| 4 | MLflow (D4-14) | todos los runs registrados | `mlruns/mlflow.db` + `metrics/pinn_grid.json` + `train` | cuenta runs del store por run_id del manifiesto; uno ausente = FALLADO |
| repro | `dvc repro train` de extremo a extremo | hashes identicos | `dvc status`, `dvc.lock`, `-FromScratch` | `prepare_sim`, `train`, `evaluate` en copia de `git ls-files` |
| fase2 / fase3 | sin regresion | `Verify-Phase2 -SkipCluster` = 2 (2 y 4 CUMPLIDOS); `Verify-Phase3` = 0 | sus informes | como `Verify-Phase3` con Fase 2 |

Todo lo no medido se marca NO MEDIDO; no se estima.

## 11. Riesgos

| Riesgo | Sev. | Mitigacion |
|---|---|---|
| Recall CRITICAL < 99,9 %: las reglas no ven la excursion (0/1.653) y el bloqueado bajo es casi invisible (H7) | ALTO | medir por tipo en M6 y volver al usuario; la excursion es un evento fisico coherente, el predictor puede no verla si su dinamica es consistente |
| 99,9 % no certificable con el tamano del test (C3) | ALTO | D4-11 lo hace explicito; mas datos CRITICAL es decision de Fase 3 |
| PINN no converge con la perdida fisica | MEDIO | curriculum (Plan_Fases), perdidas relativas adimensionales, lambdas en rejilla |
| Sobreajuste a la simulacion; KAERI nunca usado | MEDIO | limitacion declarada (como en F3); dropout 0,1 |
| `torch.save` no estable en bytes -> `dvc.lock` no reproducible | MEDIO | medir en M3 (H5); serializador propio si hace falta |
| Bytes de `prepare_sim` dependen de `pyarrow`/`pandas` (M7-4) | MEDIO | no actualizar el venv; decision del usuario pendiente |
| Fuga temporal o de episodio entre particiones | MEDIO | reparto por run (D4-7) y test que lo comprueba |
| Memoria del formato largo (1,24 GB) | BAJO | leer columnas necesarias y pivotar por run |
| Suspension de Windows en mediciones largas | MEDIO | contrastar con Kernel-Power 506/507 |

## 12. Verificacion de punta a punta

- Por hito: `ruff check .`, `mypy ml/ api/ data/`, `pytest tests/unit tests/safety --cov --cov-fail-under=70` (+ >= 95 %
  por modulo nuevo), `.\infra\scripts\Invoke-Pipeline.ps1` (stages tocados), `dvc status`,
  `Verify-Phase2.ps1 -SkipCluster` (= 2) y `Verify-Phase3.ps1` (= 0).
- Cierre: `Verify-Phase4.ps1 -FromScratch` (con todo commiteado) y los dos guardias.

## 13. Cosas a tener en cuenta por hito

| Hito | Trampas y puntos de atencion |
|---|---|
| M1 | Avisar antes de editar `params.yaml` (retira claves de `training:`); comprobar con grep que ningun test lee las claves retiradas. El stage nuevo entra en `dvc.lock`: generar en Windows con `Invoke-Pipeline.ps1 -Stage prepare_sim` y rehacer los guardias. Leer solo las columnas necesarias (H12). El orden de sensores sale del catalogo, no del parquet |
| M2 | La perdida en unidades de ingenieria (convencion 6). cp fuera de rejilla -> fila sin perdida, nunca extrapolar. Flujo por canal = total / `n_loops`. Q_gen del flujo, no de `P_th` (M2-8 de F3). Test: perdida ~0 sobre `true_value` estacionario |
| M3 | Avisar antes de retirar las claves de `training:` (D4-9, movido desde M1) y comprobar con grep que solo `train.py` las leia. MLflow solo con URI explicita en `mlruns/`; nunca crear `mlflow.db` en la raiz. Medir bytes de `model.pt` entre dos procesos. `weights_only=True` al cargar. No dejar el reloj en el checkpoint |
| M4 | `evaluate_rules_detailed` exige `DatetimeIndex` y un unico run por frame: evaluar por run. dT/dt no dispara a 180 s (no es un fallo del modelo) |
| M5 | La rejilla no se vuelve a ejecutar en CI; el ganador se fija a mano en `params.yaml` |
| M6 | Calibrar umbral y `tau` SOLO en val; test una vez. Reportar falsos positivos en normales. Override de reglas: una violacion es CRITICAL aunque el score sea bajo |
| M7 | SHAP fuera del camino de latencia; calentar antes de medir; fijar hilos; contrastar con Kernel-Power |
| M8 | Copiar helpers de `Verify-Phase3.ps1`, no refactorizarlo (30 tests lo ejecutan). `-FromScratch` copia `git ls-files`: commitear antes. En PowerShell `R` es alias de `Invoke-History` |

## 14. Cierre de cada hito

Puertas en verde con cifras medidas, resumen de hallazgos, desviaciones con evidencia, deuda abierta, seccion 15
actualizada, filas T6.x/T7.x de Progreso y mensaje de commit (sin linea Co-Authored-By; el usuario commitea).

## 15. Registro de progreso (se rellena al cerrar cada hito)

| Hito | Estado | Fecha | Cifras medidas | Commit |
|---|---|---|---|---|
| F4-M0 | Hecho (plan propuesto, pendiente de aprobacion) | 2026-11-10 | puertas: 2867 passed / 1 skipped / 99,55 % (817 s); V2 = 2, V3 = 0; preflight en la seccion 2 | - |
| F4-M1 | Hecho (stage `prepare_sim`) | 2026-11-10 | 16/9/9 runs; CRITICAL 1.474 / 1.446 / 1.813 instantes y 30.126 / 30.390 / 30.546 lecturas (91.062); pares entrenables 15.105 / 8.071 / 7.645; rejilla 50,0 / 41,7 / 42,4 %; 8.160 instantes con >1 episodio; salidas 9,57 / 5,43 / 5,43 MB; stage 7,2-7,7 s via DVC (6,4 s el script); mismos md5 en 6 ejecuciones; puertas: ruff limpio, mypy con los 2 errores de entorno, 2976 passed / 1 skipped / 99,57 % (812 s, sin 506/507), 100 % en los 3 modulos nuevos, `dvc status` limpio, V2 = 2, V3 = 0. Nota: `dvc repro --force prepare_sim` rehace tambien `simulate` (~4 min, lock identico) | - |
| F4-M2 | Pendiente | | | |
| F4-M3 | Pendiente | | | |
| F4-M4 | Pendiente | | | |
| F4-M5 | Pendiente | | | |
| F4-M6 | Pendiente | | | |
| F4-M7 | Pendiente | | | |
| F4-M8 | Pendiente | | | |
