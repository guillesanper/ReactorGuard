# Fase 3 - Capa de simulacion: referencia

Documento de referencia del dataset simulado (F3-M1 a F3-M7). Las ecuaciones, las reglas y todas las
mediciones estan en `Docs/physics_constraints.md`; las decisiones (D1-D13) y los hallazgos por hito, en
`Docs/ReactorGuard_Fase3_Plan.md`. Aqui se explica que hay en el dataset, como se regenera y verifica, que
garantiza y que no. Prosa sin tildes, como el resto del repositorio.

## 1. Que es

Un generador de series temporales de un PWR generico de 3000 MWt (no una planta real) con verdad-terreno
por lectura para los 8 tipos de fallo del TDD 2.2. El TEP de Fase 2 solo trae un positivo real de
instrumento; la simulacion aporta los que faltan.

| Pieza | Donde |
|---|---|
| Modelo fisico (16 estados, LSODA, Jacobiano analitico) | `data/simulation/thermal_model.py` |
| Propiedades del agua (IAPWS-97 tabulada, sin extrapolar) | `data/simulation/water_properties.py` + `configs/water_table.yaml` |
| Planta, limites, niveles de fallo y topes del planificador | `configs/plant_parameters.yaml` (`data/simulation/plant_params.py`) |
| Catalogo de 27 sensores y roles de las reglas | `configs/sim_sensors.yaml` (`data/simulation/sensor_catalog.py`) |
| Spans del validador para el catalogo | `configs/sim_sensor_spans.yaml` (`data/generators/derive_sim_spans.py`) |
| Reglas de TDD 5.4 + masa, y conservacion (D7) | `data/simulation/physics_constraints.py` |
| Fallos de proceso (excursion, caudal, vacio) | `data/simulation/process_faults.py` |
| Fallos de instrumento (deriva, bloqueado, pico, sesgo, intermitente) | `data/simulation/instrument_faults.py` |
| Planificador por presupuesto (D11) y cobertura (D6) | `data/simulation/scenario_planner.py`, `data/simulation/coverage.py` |
| Frame largo, manifiesto y metricas | `data/simulation/dataset_builder.py` |
| Punto de entrada del stage | `data/generators/reactor_simulator.py` (`--workers`, `--params`) |
| Estudios (cadencia, matrices, plan) | `data/generators/simulation_study.py` |
| Verificacion de la fase | `infra/scripts/Verify-Phase3.ps1` |

## 2. Layout (TDD 4.4)

```
data/raw/simulation/                                 (out del stage `simulate`, ignorado por git)
  manifest.json                                      una entrada por run: path, sha256, filas, etiquetas,
                                                     conservacion y violaciones de regla
  fault_type=<escenario>/seed=<semilla>/sim_run_<NNN>.parquet
metrics/simulation_metrics.json                      metricas del stage (cache: false, versionadas en git)
```

`<escenario>` es uno de `normal`, los 3 de proceso (`core_temp_excursion`, `coolant_flow_reduction`,
`void_formation`) o los 5 de instrumento (`drift`, `stuck`, `noise_spike`, `bias`, `intermittent`). Un run
tiene un solo escenario (D3). El directorio es solo layout: la etiqueta de cada lectura esta en sus columnas.

Ni el manifiesto ni las metricas llevan reloj ni rutas absolutas: dos construcciones con la misma semilla
escriben los mismos bytes.

## 3. Contrato de columnas

Un parquet por run, formato largo (una fila por sensor y paso), en este orden (`LONG_COLUMNS`):

| Grupo | Columnas | Notas |
|---|---|---|
| Contrato de `readings_from_frame` (11) | `reading_id`, `timestamp`, `plant_id`, `sensor_id`, `sensor_type`, `sensor_location`, `elevation_m`, `value`, `unit`, `quality`, `raw_counts` | Las mismas que el TEP adaptado: un run se valida offline con `SensorValidator` sin codigo nuevo (H12). `reading_id` = uuid5 con `plant_id`, `run_id`, tag e instante. `quality` es siempre `good`: el instrumento no sabe que miente |
| Identidad | `timestep`, `is_usable`, `run_id`, `scenario` | `timestep` = paso dentro del run |
| Verdad-terreno | `true_value`, `corrupted` | `true_value` = magnitud verdadera del sensor sin ruido ni fallo de instrumento; `corrupted` = el valor medido difiere del que habria sin el fallo |
| Eje de proceso | `process_fault`, `process_severity` | `none` fuera de un fallo de proceso; severidad nula fuera de la ventana |
| Eje de instrumento | `instrument_fault`, `instrument_severity` | `none` fuera de un fallo de instrumento |
| Episodio | `injected_at`, `injected_at_step`, `episode_id`, `fault_level` | Inicio, nivel (`low`/`medium`/`high`) e identificador del episodio ACTIVO en esa lectura; nulos fuera de un episodio |

Reglas que el dataset cumple (verificadas por `tests/safety` y por `Verify-Phase3.ps1`):

- **Toda lectura tiene verdad-terreno**: los dos ejes estan informados en todas las filas (`none` o un fallo).
- **Los ejes son ortogonales**: un run de proceso solo etiqueta el eje de proceso y uno de instrumento solo el
  de instrumento; un run normal no etiqueta ninguno.
- **Severidad estatica** del TDD 2.2 (`FAULT_SEVERITY`, D10): deriva HIGH, bloqueado CRITICAL, pico LOW, sesgo
  HIGH, intermitente MEDIUM, excursion CRITICAL, caudal HIGH, vacio CRITICAL.
- **Un fallo de proceso** etiqueta todos los sensores durante su ventana (de la inyeccion al final de la
  recuperacion) con `corrupted = False`: la planta cambio de verdad. **Uno de instrumento** etiqueta solo los
  sensores afectados (disjuntos por tipo dentro del run) y deja `true_value` intacto.
- **"Etiquetada"** en las metricas significa "con un fallo en algun eje" (`labelled_mask`), no "con
  verdad-terreno".

## 4. El dataset medido (opcion B de D11)

`params.yaml`, seccion `simulation:`: semilla 42, 34 runs de 1500 pasos a 180 s (75 h por run), 27 sensores,
tasa 0,12 y pesos de severidad 0,55 / 0,38 / 0,06 / 0,01.

| Magnitud | Valor |
|---|---|
| Lecturas con verdad-terreno | **1.377.000** (34 x 1500 x 27) |
| Lecturas con fallo | 168.958 (12,27 %); CRITICAL 6,61 % (>= 5 %), HIGH 4,81, MEDIUM 0,72, LOW 0,12 |
| Runs por escenario | normal 4, pico 7, intermitente 5; el resto 3 |
| Cobertura D6 | 8/8 tipos, 144/144 modos; por particion train 75,0 %, val 38,2 %, test 40,3 % |
| Conservacion D7 | 0 fallos; peor residuo por intervalo 3,87e-12 (tolerancia 1e-6) |
| Estado verdadero | solo el vacio viola reglas (energia, potencia/flujo y, con ebullicion franca, ebullicion, rango de caudal y masa); 0 violaciones fuera de la ventana de un fallo |
| Tamano | 75,6 MB |

**Por que 12,27 % y no 40 %**: con los topes de densidad de F3-M5 el 40 % de D11 es inalcanzable (397 runs
necesarios frente a 32 disponibles); la opcion B es el plan factible medido (`physics_constraints.md`, 13.7).

**Criterio de volumen**: Plan_Fases pide "> 1M readings etiquetados". Se interpreta como lecturas con
verdad-terreno (las 1.377.000): con un 40 % de fallos habrian hecho falta 2,5 M lecturas, y el plan fijo
`n_samples` en 50.000 pasos (1,35 M lecturas). Las lecturas con fallo se informan aparte.

## 5. Regenerar y verificar

```powershell
.\infra\scripts\Invoke-Pipeline.ps1 -Stage simulate      # nunca `dvc repro` a pelo (venv en el PATH)
.\infra\scripts\Verify-Phase3.ps1                        # criterios 1-4; salida 0/1/2/3
.\infra\scripts\Verify-Phase3.ps1 -FromScratch           # ademas, rehace simulate en una copia y compara hashes
.\infra\scripts\Verify-Phase2.ps1 -SkipCluster           # guardia contra regresion de Fase 2 (salida 2 esperada)
```

- **Coste del stage**: 10 min 46 s de reloj con `--workers 0` (8 procesos) en el portatil de referencia
  (i5-1135G7 4C/8T, Windows 11), sin suspensiones; 2 min 9 s con 4 procesos en un contenedor Linux. Los
  runs de vacio, los mas caros, se reparten primero (`_cost_key`) y marcan el tiempo total.
- **`-FromScratch`** copia solo lo que `git ls-files` lista: los cambios deben estar commiteados.
- **Determinismo**: un `numpy.random.Generator` por run hijo de `SeedSequence(openmc_seed)`, sin reloj y en
  orden fijo. Identico byte a byte entre 1, 2, 3 y 4 procesos y entre dos construcciones. **Entre
  plataformas no**: con la misma version de `pyarrow`, los runs de 200 pasos coinciden entre Linux y Windows
  y el de 1500 no (deriva de redondeo acumulada; `physics_constraints.md`, 14.4). Por eso `dvc.lock` se
  genera y se verifica en Windows, y los bytes dependen tambien de las versiones de `pyarrow` y `pandas`.

`Verify-Phase3.ps1` comprueba:

| # | Criterio | Evidencia | Como decide |
|---|---|---|---|
| 1 | 8/8 tipos y >= 95 % de la rejilla (D6) | `metrics/simulation_metrics.json` | Recalcula `covered / grid_size`; el `passed` del informe no decide |
| 2 | Conservacion 100 % y matriz esperada (D7) | metricas + JUnit de `tests/safety` | Un test saltado no es un aprobado; un FALLADO de cualquier fuente gana |
| 3 | > 1.000.000 lecturas con verdad-terreno | los parquet, contrastados con `manifest.json` | Cuenta filas y etiquetas en los parquet; sha256, filas o etiquetas distintas del manifiesto son FALLADO |
| 4 | Reproducibilidad con la misma semilla | `dvc status simulate`, `dvc.lock` (+ copia desde cero) | Solo la etapa `simulate`; la copia compara sus dos salidas |

## 6. Estudio de cadencia (D1)

La serie se emite a **180 s** (cadencia del TEP); el modelo se integra entre muestras con LSODA. Medido en
F3-M3 (`physics_constraints.md`, 11.3): en operacion normal no hay ninguna regla violada sobre valores
verdaderos a 1, 10, 60 ni 180 s, y la maxima derivada de temperatura verdadera es 0,22 C/s. El precio de
180 s, medido:

- La regla de dT/dt (10 C/s) es **inobservable**: a 180 s hace falta un salto de 1.800 K; solo es observable
  como detector con `dt <= 16 s`. En el vacio alto la derivada verdadera llega a 283 C/s y la regla no
  dispara con datos muestreados.
- La **oscilacion de la bomba** (300 s) se aliasa a 450 s; toda oscilacion mas rapida que 360 s es
  inobservable. La **firma acustica** del vacio no existe en este modelo.
- La regla de masa solo es evaluable en cuasi-estacionario; a 180 s lo es en el 99,2 % de las muestras.

## 7. Limitaciones

| Limitacion | Consecuencia |
|---|---|
| Realismo no contrastado: KAERI nunca se uso (H10) y la planta es generica | El dataset es coherente con su modelo, no con una planta real. No cerrable sin datos reales |
| Coeficientes neutronicos `LITERATURE_PLACEHOLDER` (OpenMC es F3-M8, opcional, solo en Docker) | El tamano del ciclo del vacio sale de `alpha_void = -0,15`, provisional (M4-5) |
| 180 s oculta firmas (seccion 6) | Excursion y reduccion de caudal no violan ninguna de las 7 reglas: el Rule Engine de Fase 4 no las vera (M4-1) |
| Votacion por mediana en las reglas de relacion | Un unico canal averiado queda enmascarado; votada, la matriz oculta los 3 canales de flujo neutronico (M5-6). El dataset guarda la lectura por sensor: votar o no se decide en T7.3 |
| Particiones 70/15/15 por tipo sobre runs concatenados | Val y test cubren el 38-40 % de la rejilla (M5-5). Si Fase 4 necesita todos los modos en validacion, hay que repartir por episodio o por run |
| Tasa de fallos 12,27 % (opcion B) | Lejos del 60/40 de Plan_Fases; CRITICAL si cumple el minimo |
| Validador de Fase 2 sin recalibrar para la simulacion | El Kalman marca ~48 % de las lecturas normales de caudal (`observation_noise` fija; deuda de Fase 2) |
| Determinismo bit a bit solo dentro de una plataforma (seccion 5) | Un `dvc.lock` generado en Linux no casa con Windows |
| `pandas 3` cambia las marcas de tiempo a microsegundos (M6-7) | `pyproject.toml` admite `pandas>=2.2` sin techo: **decision pendiente** (fijar `pandas<3` o migrar el contrato) |

## 8. Notas de arranque de Fase 4 (T6.1 PINN, T7.1 reentrenamiento, T7.3 reglas)

1. **Orden de columnas de la PINN (C9), propuesta**. TDD 5.2 indexa el estado como
   `[flux, flow, t_in, t_out, flow_out]` sin definir `flow`; con `flow_out` al lado, se lee como caudal de
   entrada. Con este catalogo, cada componente seria la mediana de los canales de su rol
   (`configs/sim_sensors.yaml`, `roles:`), en este orden y con estas unidades:

   | Indice | TDD 5.2 | Rol | Sensores | Unidad |
   |---|---|---|---|---|
   | 0 | `flux` | `flux` | NI-FLUX-01..03 | % del nominal |
   | 1 | `flow` (entrada) | `flow_cold_leg` | FT-CL-1, FT-CL-2 | kg/s por lazo |
   | 2 | `t_in` | `T_in` | TC-CL-1, TC-CL-2 | C |
   | 3 | `t_out` | `T_out` | TC-HL-1, TC-HL-2 | C |
   | 4 | `flow_out` | `flow_hot_leg` | FT-HL-1, FT-HL-2 | kg/s por lazo |

   El caudal de cada canal es el total entre `n_loops` (4). `flow_out - flow` es el insurge del presurizador,
   no una fuga: en estacionario vale 0 y en transitorio `900 kg/s por K/s x dT_avg/dt` (regla de masa,
   `physics_constraints.md`, 3.7). Fijar este orden en `ml/models/pinn.py` antes de entrenar: la perdida
   fisica indexa por posicion.

2. **cp de tabla, no 4182 (D8, C8)**. La perdida de energia debe usar `cp(T, P)` de `configs/water_table.yaml`
   (`WaterTable.cp`): con cp = 4182 J/(kg K) el balance nominal da un 24,7 % de desbalance; con la tabla, el
   1,1 %. La tabla no extrapola (T 200-360 C, P 60-170 bar): fuera de rejilla la fila no se evalua.
   `Q_gen` sale del flujo y `P_th` (calorimetrica, retardo 30 s) se reserva para la regla de potencia/flujo
   (M2-8). La constante `3.2e-11 x 2.5e18` del TDD no reconcilia con 3000 MW.

3. **Etiquetas para entrenar y evaluar**: el objetivo binario es `labelled_mask` (algun eje distinto de
   `none`); la severidad y el tipo salen de las columnas de cada eje. `true_value` permite entrenar la
   reconstruccion contra la magnitud verdadera y medir cuanto ensucia el ruido.

4. **Rule Engine (T7.3)**: la matriz de violaciones esperada sobre el estado verdadero es
   `EXPECTED_TRUE_STATE_VIOLATIONS` (`dataset_builder.py`); la medida sobre valores medidos esta en
   `metrics/simulation_metrics.json` (`compliance.rule_violations_measured`) y, por sensor y nivel, en
   `physics_constraints.md` 13.3-13.5. La regla de dT/dt no aporta nada a 180 s.

5. **El dataset no esta cableado** a `featurize`, `split` ni al streamer (D13): `featurize` exige los 52
   sensores del TEP (H8). Conectarlo es trabajo de Fase 4.
