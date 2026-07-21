# Prompt de continuacion — ReactorGuard Paso 3 (Bloque 3)

Trabajo en ReactorGuard (c:\Users\guill\Desktop\proyectos\reactorGuard\ReactorGuard),
una plataforma de deteccion de anomalias en reactores nucleares. Windows 11, PowerShell.
La documentacion de fases esta en Docs/ (Plan_Fases, Fase2_Prompts, Progreso, Guia_Tecnica).
El plan completo esta en C:\Users\guill\.claude\plans\compiled-inventing-scroll.md — leelo primero.
El handoff anterior (Bloque 1) es Docs/Handoff_Paso3.md — sigue vigente como contexto
heredado; este documento continua a partir de el.

Estoy a mitad del Paso 3 (nucleo de la Fase 2). El Paso 0 (entorno), el Paso 1
(contrato de datos) y el Paso 2 (camino TEP, cierra T3.2) estan cerrados y
verificados. El Bloque 1 del Paso 3 (spans por sensor, adaptador reescalado,
Invoke-Pipeline.ps1) tambien. Esta sesion (Bloque 2 y 3) completo los tests de
spans, T4.1 (Kalman) y T3.4 (validador). Lo que sigue documenta que se hizo, que
se midio y que queda.

---

## 1. Contexto heredado — NO lo re-investigues

Sigue vigente TODO lo del Bloque 1 (Docs/Handoff_Paso3.md secciones 1-3). En
resumen, para no releerlo:

- `data/generators/tep_loader.py` es la unica fuente de verdad para leer un .dat
  (d00 viene transpuesto). No dupliques esa lectura.
- `TEPAdapter` se configura por constructor (`from_params`); el entrypoint es
  `adapt_tep.py`. El adaptador emite `QualityFlag.GOOD` para los 22 ficheros y
  anade la columna `timestep` (0..499 por fichero de origen).
- Spans por sensor con DOS rangos por entrada: `[min, max]` = span calibrado del
  ADC (margen 200%), `[alarm_min, alarm_max]` = sobre de operacion normal (margen
  20%). Ambos derivados SOLO de d00. Vive en `data/schemas/sensor_spans.py`
  (`SensorSpan`, `load_sensor_spans`), se genera con
  `data/generators/derive_sensor_spans.py` (a mano, se commitea), config en
  `configs/sensor_spans.yaml` (52 entradas).
- El `omit` de `[tool.coverage.run]` contiene `reactor_simulator.py` y
  `ml/features/extractor.py` (superado, se retira en T4.2). `source` es
  `["data", "ml/features"]`.
- `reactor_simulator.py` sigue excluido de mypy y su suite
  `tests/safety/test_safety_constraints.py` entera sigue en `pytest.mark.skip`
  ("Fase 3"). NO los toques.
- `tests/unit/conftest.py` tiene fixtures reutilizables (`sensor_spans`,
  `spans_file`, `write_params`, `make_dat_dir`). Usalas.

Aviso heredado (Paso 1): si ves errores de parseo TOML inexplicables que rompen
ruff/mypy/pytest a la vez, mira la cabecera de `pyproject.toml` antes de depurar
otra cosa — se corrompio una vez con texto de prompt incrustado.

---

## 2. Hallazgos verificados empiricamente en esta sesion — NO los re-investigues

### A. El modelo de Kalman de velocidad constante NO ve una deriva lineal sostenida

Una deriva lineal vive en el espacio nulo del modelo: el filtro aprende la
pendiente y el residual decae a cero. Medido: rampa de 0,5 u/paso durante 100
pasos (50 unidades de deriva acumulada) -> maximo 1,21 sigmas de residual, CERO
pasos marcados, mientras la velocidad estimada converge EXACTAMENTE a 0,5.

**Consecuencia de diseno (ya implementada):** el `KalmanResidualDetector` usa DOS
senales del mismo filtro. El `normalized_residual` detecta transitorios
(escalones, picos, arranques, cambios de pendiente) y la `velocity` estimada
detecta la deriva ya establecida (`FaultType.SENSOR_DRIFT`). Esto lo decidiste tu
explicitamente en esta sesion. Ver `ml/features/kalman.py` cabecera y
`tests/unit/test_kalman.py::TestGradualDrift`.

### B. El `process_noise=0.1` por defecto esta calibrado para dt=1 s, no para 180 s

El termino Q[0,0] escala con dt^3. A los 180 s de muestreo del TEP se dispara a
194.400, la sigma de innovacion sube a 602 y un salto de 25 unidades queda en
0,042 sigmas: INVISIBLE. Los tests de kalman.py no lo cazaron porque usan dt=1 s.
El detector, que corre a cadencia TEP, si lo habria sufrido.

Medido a 180 s (salto de 25 unidades tras 30 muestras planas):

| q     | sigma innovacion | residual normalizado | detecta (>3) | FP en 200 planas |
|-------|-----------------:|---------------------:|:------------:|-----------------:|
| 0.1   |           602.30 |                0.042 |      no      |                0 |
| 1e-4  |            19.28 |                1.297 |      no      |                0 |
| 1e-6  |             2.99 |                8.358 |      si      |                0 |

**Implementado:** `DEFAULT_KALMAN_PROCESS_NOISE = 1e-6` SOLO en el detector; el
`OnlineKalmanFilter` conserva su 0.1 (correcto para dt=1 s). Provisional: T3.6 lo
calibra contra los saltos inyectados.

### C. Hallazgos del Bloque 1 que siguen vigentes (resumen)

- El fallo del TEP esta presente desde la muestra 0 en los ficheros de
  entrenamiento (480 filas). No hace falta parametro de instante de inicio.
- Solo XMV-04 (col_44) en d21 esta realmente congelado (480 muestras, std=0). d14
  es sticking de valvula (respuesta lenta, no lectura congelada) y d15 es
  indetectable. El plan original decia "d14/d15 positivos" y ESO ES FALSO para un
  detector de stuck. XMV-04/d21 es el unico positivo real de stuck.
- Suelo empirico: tiradas de 5-6 valores repetidos aparecen en operacion normal
  (cuantizacion del analizador). La ventana de stuck debe ser > 6.

---

## 3. Decisiones tomadas y ya implementadas — NO las re-discutas

Siguen vigentes las 4 decisiones del Bloque 1 (quality ortogonal a fault_type;
spans con dos rangos; features.sensor_selection; T3.6 con inyeccion sintetica).
Anadidas en esta sesion:

### Decision 5: el validador ESCRIBE quality, nunca la lee

Coherente con la Decision 1 (el validador no LEE quality). Como quality describe
la fiabilidad del transmisor, que es justo lo que el validador decide, el
validador es el legitimado para ESCRIBIRLA en el reading enriquecido, no para
leerla. `ValidationResult.enriched_reading` lleva SUSPECT si hay fallos no
criticos, BAD si hay alguno CRITICAL (BAD hace `is_usable=False` y saca la lectura
de la inferencia). El reading original NUNCA se muta: es una copia (`model_copy`),
y en la ruta limpia se devuelve el mismo objeto. Decidido por ti en esta sesion.

### Decision 6: umbrales dependientes de escala en fracciones del ancho del sobre

Un umbral absoluto compartido por 52 canales heterogeneos (caudales en kg/s,
presiones en bar, composiciones en mol%) no significa lo mismo en dos de ellos.
El detector de tasa (`max_envelope_fractions_per_second`) y el de deriva
(`max_drift_envelope_fractions_per_hour`) se expresan normalizados por el ancho
del sobre de alarma. Los defaults son puntos de partida razonados, no calibrados:
T3.6 los convierte en numeros medidos.

### Decision 7: metricas como datos planos, no Prometheus global

`SensorValidator.get_metrics()` devuelve un dict (contadores por fault_type,
lecturas totales, lecturas sin valor, sensores desconocidos, percentiles de
latencia). NO registra en el registro global de Prometheus: eso colisiona entre
tests y ata el modulo a un backend. El consumidor de T3.5, que tendra endpoint de
scrape, es donde estas cifras se convierten en metricas.

---

## 4. Trabajo completado en esta sesion — verificado con numeros medidos

### Ficheros nuevos

- `tests/unit/test_sensor_spans.py` — 44 tests de `data/schemas/sensor_spans.py`
  (geometria, `to_raw_counts` con clamp en ambos extremos y span bipolar, carga
  valida, y todos los rechazos: fichero ausente, seccion ausente, entrada
  no-mapping, cada clave obligatoria, span/sobre no positivo o invertido, sobre
  fuera del span). Cobertura del modulo: 100%.
- `tests/unit/test_derive_sensor_spans.py` — 19 tests del deriver (aritmetica de
  los dos margenes, canal constante con `MIN_ABSOLUTE_WIDTH`, margenes negativos,
  `span_margin < alarm_margin`, d00 ausente, round-trip por load_sensor_spans,
  cabecera con margenes). Cobertura del modulo: 97% (2 stmts del `__main__`).
- `ml/features/kalman.py` — `KalmanResult` (frozen), `OnlineKalmanFilter`,
  `KalmanFilterBank`. Modelo de orden 1 [posicion, velocidad], predict/update/step
  con reloj interno (dt calculado del timestamp anterior). Auto-reinicio a 10
  sigmas. `is_initialized=False` marca "residual no interpretable" (primera lectura
  o reinicio). P se simetriza tras cada correccion (evita sigmas NaN por deriva de
  redondeo). Cobertura: 100%.
- `tests/unit/test_kalman.py` — 57 tests. Los de convergencia miden sobre 40
  semillas, no una (el error final es aleatorio; una tolerancia puntual seria una
  constante magica). Fijan tambien la limitacion de la deriva (hallazgo A).
- `data/validation/sensor_fault.py` — `FaultType` (StrEnum: stuck, noise_spike,
  bias_out_of_range, drift_correlated, kalman_anomaly, sensor_drift), `Severity`,
  dataclass `SensorFault` (frozen, confidence en [0,1] validada, `to_dict`).
  Cobertura: 100%.
- `data/validation/sensor_validator.py` — los 5 detectores como Strategy
  (`StuckValueDetector`, `RateOfChangeDetector`, `RangeValidator`,
  `CrossCorrelationChecker`, `KalmanResidualDetector`) + `ValidationResult` +
  orquestador `SensorValidator`. Cobertura: 99% (1 stmt: guarda defensiva
  `threshold <= 0` de un helper interno).
- `tests/unit/test_sensor_validator.py` — 84 tests, organizados por detector
  (asi se calibran y desactivan en T3.6).

### Ficheros modificados

- `pyproject.toml` — `[tool.coverage.run] source` ampliado a
  `["data", "ml/features"]`; `ml/features/extractor.py` anadido al `omit`.
  Edicion quirurgica, cabecera TOML intacta.

### Invariantes del validador (cada una con test que la nombra)

1. NINGUN detector lee `quality` ni `is_usable`. El validador la ESCRIBE.
2. El detector de rango usa `in_alarm_envelope`, NO `contains` (el span calibrado
   lleva margen 200%; comprobar contra el dejaria al detector casi ciego).
3. La ventana de stuck es > 6 (default 8; el constructor rechaza <7).
4. El Kalman usa residual Y velocidad (hallazgo A).

### Detalles de diseno no obvios (para no romperlos)

- StuckValueDetector detecta por LONGITUD DE TIRADA (adimensional), no por
  varianza (que se mide en unidades^2 y no es comparable entre canales). Reporta
  en CADA lectura mientras dura el stuck, no una vez por episodio: la verdad-terreno
  de T3.6 se puntua por (sensor_id, timestep).
- CrossCorrelationChecker alinea buffers por TIMESTAMP, no por posicion. Devuelve
  `None` (no 0.0) si una serie es constante: un sensor congelado tiene correlacion
  indefinida, y devolver 0 lo marcaria dos veces (stuck + correlacion espuria). Con
  `baseline_correlations` vacio (default) no comprueba nada.
- RangeValidator: severidad CRITICAL si el valor sale del span calibrado (el
  transmisor no puede ni representarlo); HIGH si `excess > 0.25*width`; MEDIUM si
  no. (El umbral era 0.5*width y resultaba INALCANZABLE dentro del span con la
  geometria real: corregido a 0.25 esta sesion.)
- SensorValidator pasa las lecturas sin valor (`value is None`) sin juzgar y las
  cuenta: un filtro de Kalman necesita un numero, imputarlo fabricaria la
  evidencia. Cuenta tambien los sensores sin span (deriva de schema visible).

### Estado medido tras esta sesion

```
pytest tests/unit tests/safety     -> 328 passed, 3 skipped   [124 al inicio -> 328]
ruff check .                        -> All checks passed
mypy ml/ api/ data/                 -> no issues found in 33 source files

Cobertura (gate --cov-fail-under=70):
  TOTAL                             98,81%
  ml/features/kalman.py             100%   (138 stmts)
  data/validation/sensor_fault.py   100%   (40 stmts)
  data/validation/sensor_validator  99%    (258 stmts, 1 guarda defensiva)
  data/schemas/sensor_spans.py      100%
```

NO se toco `params.yaml`, `dvc.yaml` ni nada en `data/`. NO se ejecuto DVC ni se
regenero el parquet.

---

## 5. Lo que queda por hacer

### T3.6 — tests/unit/test_validator_on_tep.py  (SIGUIENTE, es el criterio de exito)

Primer criterio de exito medible del proyecto. Necesita el parquet poblado en
`data/processed/tep` (readings.parquet particionado por fault_type), asi que ES
LA PRIMERA TAREA QUE LEE DE `data/` — avisa antes de leerlo y confirma que existe
(si no, hay que correr `.\infra\scripts\Invoke-Pipeline.ps1`).

- Inyector de fallos sinteticos con SEMILLA FIJA sobre d00 (stuck, deriva, fuera
  de rango, saltos) en sensores y ventanas conocidos -> miles de positivos que
  cubren los 5 detectores. Con la verdad-terreno correcta por (sensor_id,
  timestep), la clase positiva de stuck REAL son 480 pares de 550.160 (0,087%):
  gate fragil, de ahi la inyeccion sintetica.
- Matriz de confusion por (sensor_id, timestep).
- Gate de **stuck precision > 0,95**.
- XMV-04 en d21 como caso REAL de validacion cruzada: si el detector calibrado
  sobre sinteticos lo encuentra, la calibracion transfiere.
- OJO cadencia: el parquet es a 3 min/muestra (180 s). El detector Kalman ya
  tiene el default corregido (hallazgo B), pero cualquier umbral nuevo se calibra
  a esa cadencia, no a 1 s.
- Guarda en `tests/results/validator_metrics_tep.json` (ver Fase2_Prompts T3.6).

### T4.2 — ml/features/pipeline.py + batch_featurizer.py

Los 7 grupos de features del TDD 5.5, leyendo su config de `params.yaml#features`
(incluido `sensor_selection`: resolver el conjunto real desde los datos y
contrastarlo contra `expected_sensor_count` para que la deriva de schema falle en
voz alta).

- `ml/features/extractor.py` queda superado, sin call sites. Consolidar en
  pipeline.py (reutilizar su logica de rolling/lags) y RETIRAR el modulo. Ya esta
  en el `omit` de coverage. Usa la API `fillna(method=)` eliminada en pandas 2.x.
- **DECISION PENDIENTE, hay que consultar al usuario:** el parquet esta en formato
  LARGO (una fila por sensor-timestep). Las ventanas moviles y los lags necesitan
  formato ANCHO (timestep x sensor). Donde va ese pivot es decision de diseno con
  consecuencias de memoria (550k filas). La columna `timestep` ya existe y es la
  clave del pivot junto con `fault_type` y `sensor_id`.

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
De `ml/`, en este paso SI se tocan `ml/features/` (kalman.py ya hecho; pipeline.py,
batch_featurizer.py, retirada de extractor.py pendientes). `ml/models/pinn.py`,
`ml/training/train.py` y `ml/serving/predictor.py` siguen intactos:
`physics_residual` no implementa la fisica que documenta y se reescribe en Fase 4 (T6.1).

---

## 8. Nota de commit

El arbol tiene bastante trabajo sin commitear de esta sesion y de la anterior
(ficheros nuevos de spans, kalman, validation, y modificados). Si el usuario lo
pide, agrupa en commits coherentes por bloque; no commitees sin que te lo pida.
```
