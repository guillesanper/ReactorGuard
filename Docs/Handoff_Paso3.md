# Prompt de continuacion — ReactorGuard Paso 3

Trabajo en ReactorGuard (c:\Users\guill\Desktop\proyectos\reactorGuard\ReactorGuard),
una plataforma de deteccion de anomalias en reactores nucleares. Windows 11, PowerShell.
La documentacion de fases esta en Docs/ (Plan_Fases, Fase2_Prompts, Progreso, Guia_Tecnica).
El plan completo esta en C:\Users\guill\.claude\plans\compiled-inventing-scroll.md — leelo primero.

Estoy a mitad del Paso 3 (nucleo de la Fase 2). El Paso 0 (entorno), el Paso 1
(contrato de datos) y el Paso 2 (camino TEP, cierra T3.2) estan cerrados y
verificados. Una sesion anterior ya completo el primer bloque del Paso 3; lo que
sigue documenta que se hizo, que se midio y que queda.

---

## 1. Contexto heredado del Paso 2 (sigue vigente)

- `data/generators/tep_loader.py` es la unica fuente de verdad para leer un .dat.
  Existe porque d00.dat viene transpuesto en el repo de Braatz (52x500 frente a
  480x52 de los demas). Normaliza orientacion. No dupliques esa lectura.
- `data/generators/tep_params.py` + `adapt_tep.py` son los que hacen que los
  `params:` de dvc.yaml sean efectivos y no decorativos.
- `TEPAdapter` recibe toda su configuracion por constructor (`from_params`). Su
  bloque `__main__` se retiro; el entrypoint ejecutable es `adapt_tep.py`.
- `tep_explorer.explore_tep` acepta `reports_dir`.
- El `omit` de `[tool.coverage.run]` solo contiene `reactor_simulator.py`.
- `data/generators/reactor_simulator.py` sigue excluido de mypy y su suite
  `tests/safety/test_safety_constraints.py` entera sigue en `pytest.mark.skip`
  ("Fase 3 - reescribir contra el schema canonico"). NO los toques.
- `tests/unit/conftest.py` tiene fixtures reutilizables. Usalas, no reinventes.

Aviso heredado: durante el Paso 1, `pyproject.toml` se corrompio a mitad de
sesion (texto del prompt incrustado en la linea 11, rompiendo el parseo TOML y
con el ruff, mypy y pytest a la vez). Si ves errores de parseo TOML
inexplicables, mira la cabecera del archivo antes de depurar otra cosa.

---

## 2. Hallazgos verificados empiricamente — NO los re-investigues

### A. El fallo del TEP esta presente desde la muestra 0 (verificado)

Se sospechaba que los ficheros de entrenamiento tenian N muestras normales al
principio, lo que inflaria los falsos negativos. **Es falso.** Medido contra la
linea base de d00 (media/desv. de 500 muestras normales), el maximo |z| en las
primeras 20 filas ya es: d01 -> 11,71; d07 -> 18,12; d06 -> 15,25; d14 -> 11,93;
d04 -> 7,44. Los puntos de cambio por CUSUM salen dispersos (29, 67, 240, 300,
455...), sin indice comun.

Explicacion: los ficheros de entrenamiento se simularon 25 h con el fallo
introducido a la hora 1, y esa primera hora se descarto antes de publicar las 480
muestras. El offset de 160 muestras aplica a los ficheros de *test* (d01_te...,
960 muestras), que NO estan descargados.

**Conclusion: no hace falta parametro de instante de inicio de fallo.** Etiquetar
las 480 filas como fallo es correcto.

### B. Solo existe UN sensor pegado en todo el dataset (verificado)

Run-length maximo de valores identicos consecutivos, por fichero:

| fichero      | run maximo | columna              |
|--------------|-----------:|----------------------|
| d00 (normal) |          5 | col_36               |
| d01          |          6 | col_08               |
| d14          |          5 | col_36               |
| d15          |          5 | col_36               |
| **d21**      |    **480** | **col_44 = TEP-XMV-04** |

XMV-04 esta congelado las 480 muestras de d21 (`nunique=1`, `std=0`), y varia con
normalidad en el resto (d00: `nunique=477`, `std=1,19`). Coincide con la
documentacion de d21: "la valvula del Stream 4 se fijo en su posicion de estado
estacionario". Es un positivo limpio.

**Corolario:** el plan original dice "d14/d15 positivos" y **eso es falso** para un
detector de stuck. d14 es *sticking* de valvula (respuesta lenta, no lectura
congelada) y d15 es de los fallos notoriamente indetectables.

**Suelo de diseno:** las tiradas de 5-6 valores repetidos aparecen tambien en d00
(variables de composicion, cuantizacion del analizador). La ventana del detector
stuck debe ser **> 6 muestras**.

---

## 3. Decisiones tomadas y ya implementadas — NO las re-discutas

### Decision 1: quality es ortogonal a fault_type

`QualityFlag` describe la fiabilidad del transmisor; `fault_type` describe la
perturbacion del proceso. El adaptador emitia SUSPECT para d01-d21, lo que
convertia cualquier evaluacion del validador contra fault_type en una tautologia.

**Implementado: el adaptador emite `QualityFlag.GOOD` para los 22 ficheros.** El
dataset no aporta informacion de salud de instrumento, asi que GOOD es la unica
lectura honesta. Verificado sobre el parquet regenerado: `quality != good` -> 0 filas.

**Invariante para el T3.4/T3.6: el SensorValidator NO debe leer `quality` ni
`is_usable` como entrada.**

### Decision 2: spans por sensor, con DOS rangos por entrada

El problema original: `adc_scale_max=3000` global saturaba 32.259 de 550.160
lecturas (5,86%) y a la vez dejaba las composiciones ocupando el 1% del rango ADC.

Al implementarlo aparecio una tension que el plan original no preveia: **los dos
consumidores del fichero quieren anchos distintos.** Medido sobre las 550.160
lecturas:

| margen | saturacion global | falsos positivos en d00 |
|-------:|------------------:|------------------------:|
|    20% |            12,76% |                  0,000% |
|    50% |             9,87% |                  0,000% |
|   100% |             6,99% |                  0,000% |
|   200% |             4,06% |                  0,000% |

El detector `range` quiere el rango estrecho (sensible, cero FP en normal); el
escalado ADC quiere el ancho (recortar el 12,76% destruiria informacion justo en
los datos de fallo). No es un conflicto de fondo: asi funciona la instrumentacion
real, con un **span calibrado** (rango del ADC, generoso) y unos **limites de
alarma** (estrechos, dentro del span).

**Implementado: un unico fichero, dos rangos por sensor.**

```yaml
sensors:
  TEP-XMEAS-07:
    min: 2639.9        # span calibrado -> escalado a raw_counts
    max: 2766.4
    alarm_min: 2685.44 # sobre de operacion normal -> detector range
    alarm_max: 2720.86
    unit: bar
```

Margenes: span 200%, alarma 20%, ambos derivados **solo de d00** (derivarlos del
pool completo seria leakage de los datos de fallo hacia la escala).

### Decision 3: features.sensor_selection

`features.sensor_channels` listaba los 8 canales de un schema plano muerto.
**Implementado** el reemplazo por politica de seleccion (el featurizer del T4.2
aun no lo consume — eso queda pendiente):

```yaml
features:
  sensor_selection:
    include_types: [flow, pressure, thermocouple, normalized, position]
    exclude_sensor_ids: []
    expected_sensor_count: 52
```

### Decision 4: T3.6 con inyeccion sintetica

Con la verdad-terreno correcta por `(sensor_id, timestep)`, la clase positiva de
stuck real son 480 pares de 550.160 (0,087%): un solo sensor de un solo fichero.
Gate fragil.

**Decidido: inyeccion sintetica de fallos sobre d00 con semilla fija** (stuck,
deriva, fuera de rango, saltos) en sensores y ventanas conocidos, que da miles de
positivos y cubre los 5 detectores. **XMV-04/d21 se mantiene como caso real de
validacion cruzada**: si el detector calibrado sobre sinteticos lo encuentra, la
calibracion transfiere.

---

## 4. Trabajo ya completado (Bloque 1) — verificado con numeros medidos

### Ficheros nuevos

- `data/schemas/sensor_spans.py` — dataclass `SensorSpan` (con `to_raw_counts`,
  `contains`, `in_alarm_envelope`) y `load_sensor_spans()`. Vive en `schemas/` y
  no en `generators/` para que `validation/` no dependa de `generators/`. Valida
  que el sobre de alarma este dentro del span y que ambos sean positivos.
- `data/generators/derive_sensor_spans.py` — genera la tabla desde d00
  exclusivamente. NO es un stage de dvc.yaml: se ejecuta a mano, el fichero se
  revisa y se commitea.
- `configs/sensor_spans.yaml` — 52 entradas, generado y commiteado.
- `infra/scripts/Invoke-Pipeline.ps1` — 4 capas (Configuracion / Preflight /
  Ejecucion / Reporte), con `-Stage` y `-Force`. Resuelve el
  `ModuleNotFoundError: tqdm` anteponiendo `.venv\Scripts` al PATH y abortando si
  `python` no resuelve al del venv.

### Ficheros modificados

- `tep_adapter.py` — mapas de identidad de columna ahora publicos
  (`SENSOR_TYPE_MAP`, `LOCATION_MAP`, `UNIT_MAP`, `sensor_id`); `_to_raw_counts`
  escala por span; `quality` siempre GOOD; **nueva columna `timestep`** (indice
  de muestra dentro de su fichero de origen).
- `tep_params.py` — `adc_scale_max` sustituido por `spans_path`.
- `params.yaml` — `spans_path`, `sensor_selection`.
- `dvc.yaml` — `configs/sensor_spans.yaml` como `deps` de adapt_tep (no como
  params: 52 entradas inflarian params.yaml y DVC versiona por hash igual de bien).
- `tests/unit/conftest.py` — fixtures nuevas `sensor_spans` y `spans_file`;
  `write_params` ahora emite `spans_path`.
- `tests/unit/test_tep_adapter.py`, `test_adapt_tep.py`, `test_tep_params.py` —
  actualizados al nuevo contrato.

### Fichero borrado

- `data/generators/prepare_tep.ps1` (borrado con `git rm`, recuperable).
  Orquestaba a mano descarga -> exploracion -> adaptacion generando tres scripts
  Python temporales, y escribia un `adapt_summary.json` dentro de
  `data/processed/tep`, que es un `out` de adapt_tep: eso era lo que ensuciaba el
  estado de DVC. Todo lo que hacia lo cubre hoy dvc.yaml.

### Estado medido tras el Bloque 1

```
pytest tests/unit/                      -> 124 passed
dvc repro (Invoke-Pipeline.ps1)         -> download_tep -> adapt_tep OK
dvc repro (2a vez)                      -> "Data and pipelines are up to date."

Parquet regenerado en data/processed/tep/fault_type=NN/readings.parquet:
  Total lecturas       : 550.160     (sin cambios: 26.000 + 21 x 24.960)
  Saturadas en el ADC  :  22.368 (4,07%)   [antes: 32.259 = 5,86%]
  quality != good      :       0
  timestep rango       : 0..499
```

---

## 5. Lo que queda por hacer

### Inmediato: tests de los modulos nuevos de spans

`data/schemas/sensor_spans.py` y `data/generators/derive_sensor_spans.py` estan
sin tests propios y entran en la cobertura medida. Faltan:
carga valida, seccion `sensors:` ausente, entrada no-mapping, clave ausente, span
no positivo, sobre de alarma fuera del span, `to_raw_counts` con clamp en ambos
extremos, `in_alarm_envelope`, y para el deriver: margenes negativos,
`span_margin < alarm_margin`, canal constante en d00 (`MIN_ABSOLUTE_WIDTH`).

### T4.1 — ml/features/kalman.py

`OnlineKalmanFilter` + `KalmanFilterBank` + `KalmanResult`. Va PRIMERO porque el
`KalmanResidualDetector` del T3.4 debe consumirlo, no duplicarlo.

### T3.4 — data/validation/

`sensor_fault.py` (dataclass `SensorFault`) y `sensor_validator.py` con los 5
detectores como Strategy (stuck, rate_of_change, range, cross_correlation,
kalman) mas el orquestador `SensorValidator`. Tests por detector.

Restricciones ya fijadas:
- El detector `range` consume `configs/sensor_spans.yaml` via
  `load_sensor_spans()`, y usa **`in_alarm_envelope`**, no `contains`.
- El detector `stuck` necesita ventana **> 6** (ver hallazgo B).
- **Ningun detector lee `quality` ni `is_usable`** (ver decision 1).

### T3.6 — tests/unit/test_validator_on_tep.py

Inyector de fallos sinteticos con semilla fija sobre d00 + matriz de confusion
por `(sensor_id, timestep)` + gate de **stuck precision > 0,95**. XMV-04 en d21
como caso real de validacion cruzada. Primer criterio de exito medible del proyecto.

### T4.2 — ml/features/pipeline.py + batch_featurizer.py

Los 7 grupos de features del TDD 5.5, leyendo su config de `params.yaml#features`
(incluido el `sensor_selection` ya anadido: resolver el conjunto real desde los
datos y contrastarlo contra `expected_sensor_count` para que la deriva de schema
falle en voz alta).

- `ml/features/extractor.py` queda superado y no tiene call sites. Consolidar en
  uno: reutilizar su logica de rolling/lags y **retirar el modulo**. Ojo: usa la
  API `fillna(method=)` eliminada en pandas 2.x.
- **DECISION PENDIENTE, hay que consultarme:** el parquet esta en formato LARGO
  (una fila por sensor-timestep). Las ventanas moviles y los lags necesitan
  formato ANCHO (timestep x sensor). Donde va ese pivot es una decision de diseno
  con consecuencias de memoria (550k filas). La columna `timestep` ya existe en
  el parquet y es la clave del pivot junto con `fault_type` y `sensor_id`.

### T4.4 — dvc.yaml completo

Los tres stages actuales + `featurize` + `split`, con
`data/generators/train_val_test_split.py` (estratificado por fault_type,
respetando el orden temporal para evitar leakage).

---

## 6. Objetivo verificable al terminar

```powershell
ruff check .                                                  # 0 findings
mypy ml/ api/ data/ --ignore-missing-imports                  # 0 errores
pytest tests/unit tests/safety --cov --cov-fail-under=70      # verde
pytest tests/unit/test_validator_on_tep.py -v                 # stuck precision >= 0.95
.\infra\scripts\Invoke-Pipeline.ps1     # download -> explore -> adapt -> featurize -> split
.\infra\scripts\Invoke-Pipeline.ps1     # 2a vez: "Data and pipelines are up to date"
dvc metrics show                                              # fault_type por split
```

La cobertura no debe bajar del 70%, y los modulos nuevos entran en el alcance
medido (no los anadas al `omit` para hacer pasar el gate).

Criterios de Fase 2 alcanzables en esta tanda: stuck precision > 95% y dvc repro
reproducible. Throughput Kafka > 50k/s y latencia bajo carga quedan para el Paso 4.

**NO avances al Paso 4 (Kafka, GCP, Feast, Grafana) sin que yo te lo diga.**

---

## 7. Como quiero trabajar

Paso a paso. Antes de borrar o sobrescribir cualquier archivo, muestrame que
contiene y por que. Antes de regenerar el parquet o lanzar cualquier cosa que
escriba en `data/`, dime que va a escribir y donde. Antes de tocar el contrato de
datos (`data/schemas/sensor_reading.py`) o el significado de `quality`/`is_usable`,
preguntame. Al terminar, dime que encontraste y como quedo la linea base, con
numeros medidos, no estimados.

Si algo que te he dicho aqui no coincide con lo que ves en el codigo, dimelo en
vez de adaptarte en silencio.

Principios obligatorios: sin emojis en codigo, comentarios, logs ni docs. Type
hints completos. Docstrings con Args/Returns/Raises. Tests unitarios en cada
modulo nuevo. Scripts de desarrollo local en PowerShell 7+ (.ps1) con arquitectura
de 4 capas; .sh solo dentro de contenedores o GitHub Actions. GCP Project ID real:
sentinel-platform-485714. La infraestructura GCP NO esta desplegada. Todo local-first.

`api/` es andamiaje de fases posteriores (2 de 4 endpoints devuelven 501). NO tocar.
De `ml/`, en este paso SI se tocan `ml/features/` (kalman.py, pipeline.py,
batch_featurizer.py, retirada de extractor.py). `ml/models/pinn.py`,
`ml/training/train.py` y `ml/serving/predictor.py` siguen intactos:
`physics_residual` no implementa la fisica que documenta y se reescribe en Fase 4 (T6.1).
