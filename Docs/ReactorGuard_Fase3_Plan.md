# Fase 3 (capa de simulacion) - plan por hitos con guia de ejecucion

Estado: PLAN APROBADO EN SUS DECISIONES (D1-D13) por el usuario; esta revision anade, por hito, QUE TOCAR y
COMO. No se ha escrito codigo de producto ni se ha tocado el repo (`git status` limpio en `d35f802`). Las
sondas del preflight viven en el scratchpad. Etiquetas: **[M]** medida en esta sesion, **[L]** leida del
codigo/documento, **[W]** documentacion externa no medida.

## 0. Contexto

Fase 3 debe producir un generador de datos sinteticos con verdad-terreno para los modos de fallo que el TEP
no trae (solo un positivo real de instrumento: XMV-04/d21, 0,087 %). Criterios del plan de fases: 8 tipos de
fallo, > 1M lecturas etiquetadas, 100 % de cumplimiento de constraints fisicos, resultados identicos con la
misma semilla. Fase 2 queda cerrada en local con los criterios 1 y 3 NO MEDIDOS a proposito; este plan no
toca GCP.

## 1. Linea base [M] - confirmada, coincide con el cierre de Fase 2

| Puerta | Resultado |
|---|---|
| `git status` | limpio, HEAD `d35f802` |
| `ruff check .` | All checks passed |
| `mypy ml/ api/ data/` | no issues found in 66 source files |
| `pytest tests/unit tests/safety --cov --cov-fail-under=70` | **1836 passed, 3 skipped**, TOTAL **99,41 %** (3747 sentencias, 22 sin cubrir) |
| `dvc status` | Data and pipelines are up to date |

Tiempo de la suite 559,7 s (frente a 458 s): corrio en paralelo con las sondas, no es comparable.

## 2. Preflight [M] salvo indicacion

| Pregunta | Resultado |
|---|---|
| `pip install openmc` (TDD 4.1 lo afirma) | **No existe en PyPI**: `pip index versions openmc` sin resultado; `pypi.org/pypi/openmc/json` 404. La doc dice que el "pip package" es `pip install .` sobre el repo con el binario compilado **[W]** |
| conda-forge | API anaconda.org, listado completo: 998 ficheros, 16 versiones, subdirs **solo `linux-64` y `osx-64`**, **ningun `win-64`**; ultima 0.16.0 |
| Windows nativo / WSL | No soportado nativo **[W]**; WSL2 presente pero solo con la distro `docker-desktop`; `gcc/gfortran/cmake` son los de Strawberry Perl (MinGW), no sirven |
| Imagen Docker `openmc/openmc:latest` | existe, linux/amd64, 9 capas, **1927,6 MB comprimidos**; Docker 29.5.2 operativo |
| Datos nucleares, coste por corrida de OpenMC | **NO medidos** (no se descargo nada); se miden en M8 antes de bajar nada |
| PyDy 0.9.4 | en PyPI, requiere numpy>=1.26.4, scipy>=1.11.4, sympy>=1.12; "multi-body dynamics" (Kane); no instalado, compatibilidad con sympy 1.14 no probada |
| Coste ODE (9 estados, intervalo de 180 s, 2000 muestras) | LSODA **3,27 ms**, Radau 12,9 ms, BDF 14,2 ms, RK45 **7114 ms**; rigidez 2,7e4 (\|lambda\|max 325 1/s). 41.667 pasos (1M lecturas / 24 sensores) ~ **2,3 min**. El modelo de la sonda NO esta calibrado (relaja a n = 0,43): solo vale como medida de coste |
| Agua IAPWS-97 a 15,5 MPa (`iapws 1.5.5`, solo en scratchpad) | cp 5243 / 5664 / 6415 J/(kg K) a 290 / 307,5 / 325 C; T_sat = **344,79 C** (margen 19,8 K a 325 C); T_sat(120 bar) = 324,7 C |
| Balance de energia con cp = 4182 (TDD 5.2), 18.000 kg/s, 30 K, 3000 MW | 2258 MW -> **desbalance 24,7 %**; con cp IAPWS a 579 K: 3031,9 MW -> **1,1 %**. La constante `3.2e-11*2.5e18` da 2,56e21 con flux 3,2e13: no reconcilia con 3,0e9 W |
| Tamano por lectura (parquet largo TEP, d00) | 26.000 filas = 1,20 MB -> ~46 B/lectura -> 1M ~ 46 MB |
| Entorno | i5-1135G7 4C/8T, 15,8 GB RAM, 191,9 GB libres |

Conclusion: OpenMC solo es ejecutable via Docker, es un solver estacionario (no integra transitorios) y por
tanto NO puede ser el motor de la serie temporal; la fisica temporal cabe de sobra en scipy.

## 3. Lo que dice el TDD [L]

**Taxonomia (TDD 2.2): 8 filas.**

| # | Tipo | Eje | Severidad |
|---|---|---|---|
| 1 | Thermocouple drift (~0,1 C/dia) | instrumento | HIGH |
| 2 | Stuck sensor | instrumento | CRITICAL |
| 3 | Noise spike | instrumento | LOW |
| 4 | Bias fault | instrumento | HIGH |
| 5 | Intermittent fault (alternancia valido/invalido) | instrumento | MEDIUM |
| 6 | Core temperature excursion | proceso | CRITICAL |
| 7 | Coolant flow reduction (bomba, oscilacion sutil) | proceso | HIGH |
| 8 | Void formation (ebullicion, firma acustica) | proceso | CRITICAL |

**Constraints.** App. B (leyes): energia `Q_gen = Q_removed + dU/dt`; masa `flow_in = flow_out` (estacionario);
Fourier (es feature, no regla); dT/dt < 10 C/s; T < T_sat(P); `P_thermal ∝ flux`. TDD 5.4 (6 reglas; la columna
Formula es la condicion de VIOLACION): desbalance > 5 %; T_out < T_in; dT/dt > 10 C/s; flow < 0 o > pump_max;
|P_th - k flux| > umbral; P < P_sat(T). **El TDD no tiene catalogo de sensores**: solo 5 tipos, 4 ubicaciones,
5 unidades y tags de ejemplo (`TC-CORE-12`, `TC-CORE-11`, bomba `P-01`).

## 4. Hallazgos [L] salvo indicacion

| # | Hallazgo |
|---|---|
| H1 | `reactor_simulator.py` roto: schema plano pre-`f130d27`, NOMINAL en Kelvin (564/594 K) aunque el contrato solo tiene `celsius`; `params.yaml` comenta 564 K como "outlet" cuando el codigo lo usa como entrada; `dt_seconds: 1.0`; `fault_types` = LOCA/RIA/SGTR/LOFW, no los del TDD |
| H2 | Unico consumidor: `tests/safety/test_safety_constraints.py` (3 tests, skip de modulo). En `omit` de coverage y en `[[tool.mypy.overrides]]`: sacarlo de AMBOS |
| H3 | `TEPFaultInjector`: 4 manipulaciones, acoplado a formato largo TEP, a `FaultType` del validador y a N episodios/segmento; 51 tests; **el criterio 2 de Fase 2 depende de el** |
| H4 | `SensorReading` no tiene `fault_type`/`severity`/`injected_at`. En el TEP `fault_type` (int 0-21) es columna del frame FUERA del contrato y entra en `reading_id` (uuid5) |
| H5 | El contrato cubre lo necesario (6 SensorType, 4 SensorLocation, 5 unidades, elevation en [-10, 100], raw_counts 16 bit). `plant_id` entra en la clave de `reading_id`: un `plant_id` propio evita colisiones |
| H6 | `ParquetDirSource`/`run_name` asumen `fault_type=NN` entero; TDD 4.4 propone `simulation/fault_type=<nombre>/seed=<n>/sim_run_001.parquet` |
| H7 | El validador carga spans de `tep.spans_path`; un sensor sin span no falla: se cuenta en `readings_unknown_sensor` y rango/tasa/deriva devuelven `[]`, stuck usa tolerancia 0. Un catalogo sin spans queda **silenciosamente sin validar** |
| H8 | `featurize` exige `expected_sensor_count: 52` y lee `tep.processed_dir`: meter la simulacion alli no es gratis |
| H9 | Severidad del validador dinamica (stuck por racha; noise spike MEDIUM/HIGH) distinta de la estatica del TDD (stuck CRITICAL, spike LOW) |
| H10 | "Validar contra KAERI en Fase 2" nunca se ejecuto: solo aparece en `Plan_Fases.md` (3 menciones) |
| H11 | `pyproject.toml`: `openmc` comentado; **`scipy` no es dependencia explicita** (llega transitiva); ni `iapws` ni `pydy` figuran |
| H12 | `readings_from_frame` (tep_adapter.py) reconstruye `SensorReading` desde un frame largo con 11 columnas requeridas: **una salida de simulacion con esas columnas se valida offline sin codigo nuevo** |
| H13 | `Invoke-Pipeline.ps1` exige `configs/sensor_spans.yaml` y resume solo `data/processed/tep`; `Verify-Phase2.ps1` compara TODAS las etapas de `dvc.yaml` con `dvc.lock` y `-FromScratch` copia solo `git ls-files`: anadir `simulate` sin lock rompe el criterio 4 de Fase 2, y los ficheros nuevos deben estar trackeados para entrar en la copia |
| H14 | `data/raw/` y `data/processed/` ya estan en `.gitignore` raiz (lineas 112-113): `data/raw/simulation` queda ignorado sin tocar nada [M, `git check-ignore`] |

### Choques

| # | Choque |
|---|---|
| C1 | Plan/Progreso cuentan 7 tipos ("4 base + 3") y dicen "faltan los 4 fisicos"; el TDD tiene **8**: falta *Intermittent*, que es de **instrumento**. Los de proceso son 3 |
| C2 | "Cobertura > 95 % de 8 tipos": 7/8 = 87,5 % -> exige 8/8 salvo redefinir "modo" (D6) |
| C3 | Void formation y core temp excursion VIOLAN por definicion las reglas 5.4: "compliance 100 %" solo es posible sobre el estado verdadero con conservacion (D7) |
| C4 | dT/dt: 10 C/s (App. B y 5.4) frente a 50 C/s (TDD 3.3, ejemplo del detector de tasa) |
| C5 | A 180 s la regla dT/dt < 10 C/s equivale a |dT| < 1800 K entre muestras: inobservable como detector |
| C6 | `fault_injection_rate` en `params.yaml` = "5 % anomalas"; el plan pide 60/40 y >= 5 % CRITICAL |
| C7 | `docs/physics_constraints.md` (plan) frente a `Docs/` (real) |
| C8 | cp = 4182 y la constante de TDD 5.2 no reconcilian el balance |
| C9 | TDD 5.2 indexa el estado de la PINN `[flux, flow, t_in, t_out, flow_out]`: el orden de columnas condicionara Fase 4 (T6.1) |

## 5. Decisiones aprobadas (D1-D13) y su concrecion

| # | Decision aprobada | Concrecion operativa |
|---|---|---|
| D1 | Cadencia **180 s** emitida; el ODE se integra por intervalo con LSODA | `simulation.dt_seconds: 180` es parametro; M3 MIDE a 1/10/60/180 s que se pierde y el validador (FP) antes de congelarlo |
| D2 | Catalogo PWR **propio**, no mapeo a los 52 del TEP | `configs/sim_sensors.yaml` (27 sensores propuestos, M1) + spans propios derivados de operacion normal simulada (M3) |
| D3 | Dos ejes ortogonales; un escenario por run; sin runs compuestos en F3 | columnas `process_fault`/`instrument_fault` y sus severidades; el directorio `fault_type=<nombre>` es solo layout |
| D4 | OpenMC solo para calibrar (Docker, offline); PyDy descartado; motor = scipy/sympy | M8 opcional; el stage `simulate` no depende de Docker; coeficientes con proveniencia `LITERATURE_PLACEHOLDER` hasta M8. **Antes de descargar la imagen y los datos en M8 se mide el tamano y se confirman las cifras contigo** |
| D5 | Modulo nuevo; `TEPFaultInjector` intacto | se reutilizan `SensorSpan` y la semantica de las manipulaciones |
| D6 | Modo = (tipo x grupo de sensor x nivel de magnitud); gate en 8/8 tipos y >= 95 % de la rejilla | `data/simulation/coverage.py`; k_min y n_min se fijan en M5 con datos |
| D7 | Compliance = conservacion (energia con dU/dt, masa) sobre el estado verdadero 100 %, mas matriz de violaciones esperadas de las reglas 5.4 | residuo de conservacion se calcula en generacion con el estado completo y se registra por run |
| D8 | cp(T,P) de IAPWS-97 | **refinamiento**: `iapws` solo como dependencia de desarrollo para GENERAR `configs/water_table.yaml`; el runtime lee la tabla versionada (determinismo y sin dependencia nueva en `simulate`) |
| D9 | 10 C/s como limite sobre la derivada del estado verdadero; 50 C/s del validador no se toca | `ConstraintLimits.max_dt_c_per_s` |
| D10 | Severidad = tabla estatica TDD 2.2 | `FAULT_SEVERITY` unica en `scenario.py` |
| D11 | `fault_injection_rate` = fraccion de lecturas **etiquetadas como afectadas**, defecto 0,40; `critical_min_fraction` 0,05; reparto CRITICAL 0,30 / HIGH 0,30 / MEDIUM 0,25 / LOW 0,15 de ese presupuesto; topes de densidad por tipo | si el tope hace inalcanzable la cuota, el planificador FALLA con mensaje (no fuerza ni redondea) y se vuelve a ti |
| D12 | `openmc_seed` se conserva como semilla maestra | `SeedSequence(openmc_seed).spawn(n_runs)` |
| D13 | F3 produce `data/raw/simulation/` y lo valida offline; NO cablea `featurize/split/streamer` | test offline con `readings_from_frame` + `SensorValidator` (H12) |
| C7 | `physics_constraints.md` en `Docs/` | `Docs/physics_constraints.md` |

## 6. Convenciones transversales (aplican a todos los hitos)

1. **Layout**: paquete nuevo `data/simulation/` con `__init__.py`; configs en `configs/`. `hatch` ya empaqueta
   `data/` y `[tool.coverage.run] source` ya incluye `data`: no hay que tocar la cobertura salvo quitar el `omit` (M3).
2. **Loaders**: copiar la forma de `data/generators/tep_params.py`: `DEFAULT_PARAMS_PATH`, `_SECTION`, dataclass
   `frozen=True`, `_require(section, key)` con `KeyError("params.yaml: missing required key 'simulation.<k>'.")`,
   `load_*` que lanza `FileNotFoundError`/`KeyError`/`ValueError`. Los YAML de `configs/` siguen el mismo esquema.
3. **`main()` fino** como `data/generators/adapt_tep.py`: resolver params, orquestar, reportar; la logica en libreria.
4. **Aleatoriedad**: solo `numpy.random.Generator` hijos de `SeedSequence`. Prohibido `np.random.seed` y `random`
   (el simulador roto los usa con `# noqa: S311`).
5. **Salidas deterministas**: sin reloj, sin rutas absolutas (el `-FromScratch` corre en otra carpeta), orden fijo,
   float64, `json.dumps(sort_keys=True)`.
6. **Tests**: un `tests/unit/test_<modulo>.py` por modulo nuevo, clases por comportamiento (patron de
   `test_tep_params.py`), `tmp_path`, sin red/Docker/iapws en runtime; cobertura >= 95 % por modulo; builders
   compartidos en `tests/support/` (patron de `tep_frames.py`).
7. **Estandares**: ruff E,W,F,I,N,UP,S,B,A con line-length 100 tambien en tests; mypy strict; docstrings Google con
   Args/Returns/Raises; `_LOG` con %-lazy; sin emojis; prosa en espanol sin tildes; Python >= 3.11 sin PEP 695.
8. **Al abrir cada hito**: listar los paths exactos que se van a crear/tocar (regla del usuario) y confirmar
   baseline. **Al cerrarlo**: puertas completas, cifras medidas en una tabla, filas T5.x de
   `Docs/ReactorGuard_Progreso.md` actualizadas, propuesta de mensaje de commit (NUNCA ejecutarlo).
9. **No se toca** en toda la fase: `data/schemas/sensor_reading.py`, `data/validation/fault_injector.py`,
   `data/validation/sensor_validator.py`, `data/streaming/*`, `ml/features/*`, ni el significado de
   `quality`/`is_usable`.

## 7. Hitos: que tocar y como

Orden de dependencia: **M1 -> M2 -> M3 -> {M4, M5} -> M6 -> M7**; **M8** es paralelo y opcional.

### F3-M1 - Modelo fisico, tabla de agua y catalogo de sensores

**Objetivo**: un modelo ODE determinista y calibrado en equilibrio, con propiedades del agua y catalogo.

**Crear**

| Fichero | Contenido |
|---|---|
| `Docs/ReactorGuard_Fase3_Plan.md` | **PRIMER paso del hito**: copia de este plan en una ruta estable del repo (decirlo al usuario antes de crearlo). Todos los prompts de relevo lo referencian; al cerrar cada hito se rellena su seccion 15 "Registro de progreso" |
| `data/simulation/__init__.py` | docstring del paquete |
| `data/simulation/build_water_table.py` | herramienta de desarrollo: importa `iapws` de forma perezosa, barre T 200-360 C (paso 2) x P 60-170 bar (paso 5) y escribe cp, rho, h del liquido comprimido y T_sat(P). `main(argv)` con `--output`. Cabecera con version de iapws |
| `configs/water_table.yaml` | tabla generada y commiteada (~56 x 23 x 3 valores) |
| `data/simulation/water_properties.py` | `WaterTable` (frozen) con `cp(t_c, p_bar)`, `rho`, `h`, `t_sat_c(p_bar)`; `load_water_table(path)`; interpolacion bilineal propia con numpy; **fuera de rejilla lanza `ValueError`, nunca extrapola** |
| `configs/plant_parameters.yaml` | P0 (3000 MWt), caudal nominal (18.000 kg/s), presion (155 bar), T_in0 (291 C), T_f0, T_sec0, capacidades termicas C_f/C_c/C_s, tau de bomba y presurizador, `pump_max`, limites de constraints, `fault_tiers` (vacio hasta M4). Cada valor con comentario de origen |
| `configs/neutronics_coefficients.yaml` | beta_i, lambda_i (6 grupos), Lambda, alpha_f (Doppler), alpha_m, alpha_v, peso de barras; clave `provenance: LITERATURE_PLACEHOLDER` |
| `data/simulation/plant_params.py` | `PlantParams`, `NeutronicsCoefficients`, `load_plant_params`, `load_neutronics` (patron `_require`) |
| `data/simulation/thermal_model.py` | `ReactorModel`: estado de 13 variables (n, 6 precursores, T_f, T_c, T_s, P, void, caudal) mas controlador de barras; `ModelInputs` (carga, insercion de reactividad, demanda de bomba, fuga); `steady_state()`, `advance(y, inputs, dt)`, `observables(y)`, `diagnostics(y, inputs)` |
| `configs/sim_sensors.yaml` | catalogo (ver tabla siguiente) con `source`, `gain`, `offset`, `noise_sigma`, y bloque `roles:` (T_out, T_in, flow, P, flux, P_th) para las constraints |
| `data/simulation/sensor_catalog.py` | `SensorDef`, `SensorCatalog`, `load_sensor_catalog`; valida enums del contrato, tags unicos, elevation en [-10, 100], >= 21 sensores |
| tests | `test_water_properties.py`, `test_build_water_table.py`, `test_plant_params.py`, `test_thermal_model.py`, `test_sensor_catalog.py` |

**Tocar**: `pyproject.toml` (anadir `scipy>=1.11` a `dependencies` bajo "Data / Numerics"; `iapws>=1.5.5` en
`[project.optional-dependencies].dev`; sustituir el comentario de `openmc` por una nota: no esta en PyPI, corre en
Docker, ver M8). **No** tocar `params.yaml` todavia.

**Catalogo propuesto (27 sensores; M1 lo fija y mide)**

| Tags | Tipo / ubicacion / unidad | Fuente (observable) | sigma inicial |
|---|---|---|---|
| TC-CORE-01..04 | thermocouple / core / celsius | T_c + desvio espacial fijo (0,3-1,5 K) | 0,3 K |
| TC-HL-1, TC-HL-2 | thermocouple / primary_loop / celsius | T_c | 0,3 K |
| TC-CL-1, TC-CL-2 | thermocouple / primary_loop / celsius | T_s | 0,3 K |
| TC-SG-1 | thermocouple / secondary_loop / celsius | T_sec | 0,3 K |
| PT-PRZ-01, PT-PRZ-02 | pressure / primary_loop / bar | P | 0,3 bar |
| PT-SG-1 | pressure / secondary_loop / bar | P_sec | 0,2 bar |
| PT-CNT-01 | pressure / containment / bar | 1,0 + acople pequeno a la fuga | 0,01 bar |
| FT-HL-1, FT-CL-1, FT-HL-2, FT-CL-2 | flow / primary_loop / kg_s | caudal/4 (+ termino de insurge en hot leg) | 1 % |
| FT-FW-1 | flow / secondary_loop / kg_s | caudal de agua de alimentacion | 1 % |
| NI-FLUX-01..03 | flux_detector / core / percent | 100 n | 0,3 % |
| ROD-POS-01, ROD-POS-02 | position / core / percent | posicion de banco de control | 0,2 % |
| LT-PRZ-01 | normalized / primary_loop / percent | nivel del presurizador (funcion de T_avg) | 0,3 % |
| LT-SG-1, LT-SG-2 | normalized / secondary_loop / percent | nivel de GV | 0,3 % |
| PWR-TH-01 | normalized / secondary_loop / percent | potencia calorimetrica (calor cedido al secundario, con retardo) | 0,3 % |

**Como (nucleo del modelo)**

1. Cinetica puntual: `dn/dt = ((rho - beta)/Lambda) n + sum(lambda_i C_i)`, `dC_i/dt = (beta_i/Lambda) n - lambda_i C_i`.
2. Reactividad: `rho = rho_rod + alpha_f (T_f - T_f0) + alpha_m (T_avg - T_avg0) + alpha_v void + rho_insercion(t)`.
3. Termica de tres nodos: fuel `C_f dT_f = P0 n - hA (T_f - T_avg)`; nucleo `C_c dT_c = hA (T_f - T_avg) - m cp (T_c - T_s)`;
   GV primario `C_s dT_s = m cp (T_c - T_s) - UA_eff (T_avg - T_sec)`, con `T_avg = (T_c + T_s)/2` y
   `UA_eff = UA * carga(t)`. La suma de los tres nodos es `P0 n - Q_sec`: **la conservacion de energia se cumple
   por construccion** (usar el mismo cp en los dos terminos de transporte).
4. Equilibrio por construccion: fijar T_in0, caudal y P0, iterar 2-3 veces `T_out0 = T_in0 + P0/(m cp(T_avg0))`,
   y DERIVAR `hA = P0/(T_f0 - T_avg0)` y `UA = P0/(T_avg0 - T_sec0)`; asi el punto nominal es equilibrio exacto
   (la sonda de preflight falla justo en esto).
5. Presurizador: `tau_p dP/dt = P_set + k_p (T_avg - T_avg0) - P - fuga(t)`; vacio: `void` relaja con `tau_v` hacia
   `clip((T_c - T_sat(P) + delta)/w, 0, v_max)`.
6. Bomba: `tau_b dm/dt = m_nom * demanda(t) - m`. Controlador PI de barras que mantiene `T_avg = T_ref(carga)`.
7. Propiedades congeladas por intervalo (cp, rho, T_sat evaluados al inicio del paso de 180 s): operador de
   splitting documentado; **se mide** el error de cierre de energia que introduce.
8. Integracion `scipy.integrate.solve_ivp(method="LSODA", rtol=1e-8, atol=1e-10)`; sympy solo para verificar
   simbolicamente la conservacion en un test.

**Aceptacion (cifras a medir y registrar)**

| Medida | Umbral |
|---|---|
| Equilibrio nominal a 24 h simuladas | n = 1 +- 1e-6; max \|dy/dt\| relativo < 1e-9 |
| Cierre de energia sobre rampa de carga | residuo relativo < 1e-6 |
| Coste por paso de 180 s (2000 pasos) | informar ms/paso; objetivo < 10 ms |
| LSODA frente a Radau | trayectorias acordes a 1e-5 relativo |
| Tabla de agua frente a IAPWS directo (200 puntos aleatorios) | informar error maximo; objetivo < 0,5 % |
| Determinismo | misma traza byte a byte entre dos ejecuciones y entre dos procesos |
| Signos de realimentacion | +T_f o +T_avg reduce n; vacio reduce n |

**Cierre**: ruff, mypy, pytest, `dvc status`. Commit propuesto: `F3-M1: modelo termico, tabla de agua y catalogo de sensores`.

### F3-M2 - Constraints fisicos

**Objetivo**: las 6 reglas de 5.4 y las leyes de App. B como funciones puras, verificadas y documentadas.

**Crear**

| Fichero | Contenido |
|---|---|
| `data/simulation/physics_constraints.py` | `ConstraintLimits` (frozen, desde `plant_parameters.yaml`); funciones vectorizadas puras: `energy_balance_residual`, `temperature_ordering_violated`, `max_dt_dt_violated`, `flow_range_violated`, `power_flux_violated`, `boiling_violated` (usa `WaterTable.t_sat_c`); `evaluate_rules(wide_frame, catalog_roles, table, limits) -> DataFrame[bool]`; `RuleReport` (tasa de violacion por regla y escenario); `conservation_residuals(diagnostics)` y `ConservationReport`; `min_observable_rate(dt_s)` que cuantifica C5 |
| `Docs/physics_constraints.md` | ecuaciones implementadas con unidades, cp de tabla, cierre de conservacion, limites y valores, observabilidad por cadencia, inconsistencias del TDD (4182, 3.2e-11 x 2.5e18, 10 frente a 50 C/s, polaridad App. B frente a 5.4) y matriz de violaciones esperadas (se rellena en M4-M6) |
| `tests/unit/test_physics_constraints.py` | casos limite por regla |

**Como**

- Los roles (T_out, T_in, caudal, presion, flujo, P_th) salen del bloque `roles:` de `sim_sensors.yaml`; el
  modulo no cablea tags.
- Operadores estrictos: "igual al umbral" NO es violacion. Un valor `None`/NaN no se evalua y se cuenta aparte.
- Energia sobre mediciones: `Q_rem = m_total * cp_table(T_avg, P) * (T_out - T_in)` con la tabla, no 4182;
  `Q_gen` desde `P_th` o flujo segun el rol; documentar cual.
- Masa: en estado cuasi-estacionario `|m_hl - m_cl| / m < tol`; "cuasi-estacionario" = |dT_avg/dt| bajo un
  umbral de config, medido.
- Conservacion sobre el estado completo (`diagnostics`): `residuo = |dU/dt - (Q_gen - Q_sec)| / P0`.

**Tests clave**: ebullicion con T_out = T_sat -+ eps; flujo 0 y negativo; T_out == T_in; NaN; forma de salida
vectorizada; residuo de conservacion sobre una traza sintetica de M1; `min_observable_rate(180)` aritmetico.

**Cierre**: puertas + commit propuesto `F3-M2: constraints fisicos y documento de ecuaciones`.

### F3-M3 - Simulador en operacion normal, parametros, spans y estudio de cadencia

**Objetivo**: `reactor_simulator.py` reescrito produciendo `SensorReading` canonicos en operacion normal, y
**medir** la decision D1.

**Tocar (existentes)**

| Fichero | Cambio |
|---|---|
| `data/generators/reactor_simulator.py` | REESCRIBIR: borrar NOMINAL/SIGMA y `main` viejo (incluido el `data/raw/simulation.jsonl`); nueva clase `ReactorSimulator.from_params(...)` que orquesta runs y un `main(params_path)` fino |
| `pyproject.toml` | quitar la entrada de `omit` y el bloque `[[tool.mypy.overrides]]` de `reactor_simulator` con sus comentarios |
| `params.yaml` | reescribir la seccion `simulation:` (ver abajo). Con tu aprobacion previa al editar: cambia el significado de `fault_injection_rate` |

**Seccion `simulation:` objetivo** (claves nuevas; las de potencia/temperaturas pasan a `plant_parameters.yaml`;
`fault_types` desaparece): `n_samples: 50000` (PASOS de tiempo, documentado), `fault_injection_rate: 0.40`,
`critical_min_fraction: 0.05`, `openmc_seed: 42`, `dt_seconds: 180`, `run_length_steps: 1500`,
`warmup_fraction: 0.20`, `start_time`, `plant_id: SIM-PWR-01`, `output_dir: data/raw/simulation`,
`metrics_path: metrics/simulation_metrics.json`, `plant_parameters_path`, `neutronics_path`, `sensors_path`,
`spans_path`, `water_table_path`, `severity_weights: {critical: 0.30, high: 0.30, medium: 0.25, low: 0.15}`,
`calibration_date`, `last_maintenance_date`, `drift_coefficient`.

**Crear**

| Fichero | Contenido |
|---|---|
| `data/simulation/simulation_params.py` | `SimulationParams` (frozen) + `load_simulation_params`; valida rangos (tasa en (0,1), pesos suman 1, `critical_min_fraction` <= `fault_injection_rate`, `run_length_steps` > 0) |
| `data/simulation/scenario.py` | `ScenarioKind` (normal + 8), `FAULT_SEVERITY` (tabla TDD 2.2 UNICA), `LoadProfile` (rampas acotadas entre 70 y 100 %, semilla), `RunPlan` |
| `data/simulation/dataset_builder.py` (parte 1) | ensambla el frame largo de un run: observables -> ruido por sensor -> `SensorSpan.to_raw_counts` -> filas; `sim_reading_uuid(plant_id, run_id, tag, timestamp)` con `READING_ID_NAMESPACE` importado de `tep_adapter`; columnas = las 11 de `readings_from_frame` + `timestep`, `is_usable`, `run_id`, `scenario`, `true_value`, `corrupted`, `process_fault`, `process_severity`, `instrument_fault`, `instrument_severity`, `injected_at`, `episode_id` |
| `data/generators/derive_sim_spans.py` | patron de `derive_sensor_spans.py`: importa `DEFAULT_ALARM_MARGIN`, `DEFAULT_SPAN_MARGIN`, `MIN_ABSOLUTE_WIDTH`; observa 5 runs normales con semillas fijas propias (`SPANS_SEED`, **distinta** de `openmc_seed`) y escribe `configs/sim_sensor_spans.yaml` con cabecera propia |
| `configs/sim_sensor_spans.yaml` | generado, revisado a mano y commiteado (`deps` de `simulate`) |
| `tests/support/sim_frames.py` | YA EXISTE desde F3-M2 (`Trace`, `run_load_profile`, `wide_frame` del modelo termico): EXTENDER con el builder de frames de salida de simulacion, no recrear |
| tests | `test_simulation_params.py`, `test_scenario.py`, `test_dataset_builder.py`, `test_reactor_simulator.py`, `test_derive_sim_spans.py` |

**Como**

- Cada run arranca en `start_time + run_index * run_stride` (timestamps unicos; el validador no ve retrocesos).
  `reading_id` incluye `run_id` ademas de `plant_id`.
- `quality = good` siempre (el validador la escribe, nunca se lee); `is_usable = True`.
- Sin fallos: `process_fault = instrument_fault = "none"`, severidades `None`, `injected_at = NaT`, `corrupted = False`.
- Parquet con `DataFrame.to_parquet(index=False)` (mismo metodo que TEP, ya reproducible).
- `reactor_simulator.py` NO importa nada de `data/validation` en produccion; la validacion offline vive en tests.

**Mediciones obligatorias (entregable de M3, en una tabla de `Docs/`)**

1. **Estudio de cadencia**: mismo escenario normal a dt = 1, 10, 60 y 180 s: (a) fraccion de pasos con alguna regla
   violada sobre valores verdaderos (esperado 0), (b) max |dT/dt| de la derivada del estado, (c) alertas del
   `SensorValidator(load_sensor_spans("configs/sim_sensor_spans.yaml"))` por 1000 lecturas, alimentado con
   `readings_from_frame(frame)`, **sin recalibrar** ningun umbral. Esto cierra D1 con datos.
2. **Calibracion de ruido**: sigma por sensor tal que las 6 reglas sobre mediciones normales den 0 violaciones en
   N runs; registrar la distancia minima al umbral.
3. **Determinismo**: dos ejecuciones -> `DataFrame.equals` y bytes de parquet iguales; y entre procesos.
4. **Rendimiento**: pasos/s extremo a extremo.

**Cierre**: puertas (el modulo ya cuenta en cobertura y mypy) + commit `F3-M3: simulador en operacion normal, spans y estudio de cadencia`.

### F3-M4 - Fallos de proceso (3)

**Crear**: `data/simulation/process_faults.py` y `tests/unit/test_process_faults.py`.

**Tocar**: `configs/plant_parameters.yaml` (`fault_tiers:` por tipo y nivel bajo/medio/alto),
`data/simulation/scenario.py` (`ProcessFaultSpec`), `Docs/physics_constraints.md` (matriz medida).

**Como** (los fallos de proceso perturban las ECUACIONES via `ModelInputs`, no la medicion):

| Tipo | Perturbacion | Firma esperada |
|---|---|---|
| core temperature excursion (CRITICAL) | rampa de insercion de reactividad `rho_insercion(t)` con el PI de barras saturado/deshabilitado | n y T_f suben, T_out sube, balance fuera de 5 % durante el transitorio |
| coolant flow reduction (HIGH) | `demanda_bomba` baja a una fraccion con constante de tiempo + termino oscilatorio de amplitud pequena | caudal baja, T_out sube, delta-T crece; la oscilacion mas rapida que ~6 min es inobservable a 180 s (se documenta) |
| void formation (CRITICAL) | `fuga(t)` despresuriza el primario hasta que `T_c > T_sat(P)`; `void` crece y realimenta | P baja, `boiling_violated` se dispara, n baja por realimentacion de vacio |

- Niveles de magnitud: ajustar `fault_tiers` hasta que cada nivel alcance un objetivo medible (p. ej. T_out maximo,
  margen minimo a T_sat, caudal minimo) y registrarlos en la tabla.
- Cada escenario: calentamiento normal (`warmup_fraction`), inicio en `injected_at`, persistencia y recuperacion
  opcional; etiqueta `process_fault` en TODOS los sensores del paso desde el inicio.

**Medir**: por tipo x nivel: pico de T_out, margen minimo a T_sat, max |dT/dt|, reglas que se disparan y retardo
en muestras; y que **no** se disparen reglas ajenas (p. ej. flow reduction en nivel bajo no debe ebullir). La
matriz resultante es la verdad-terreno del Rule Engine de Fase 4 (T7.3).

**Tests**: monotonia de la respuesta, signo de la realimentacion, instante de inicio exacto, severidad == tabla
TDD, conservacion se mantiene durante el transitorio, determinismo.

**Cierre**: commit `F3-M4: fallos de proceso (excursion termica, reduccion de caudal, vacio)`.

### F3-M5 - Fallos de instrumento, planificador, etiquetas y cobertura

**Crear**

| Fichero | Contenido |
|---|---|
| `data/simulation/instrument_faults.py` | `InstrumentFaultKind` (DRIFT, STUCK, NOISE_SPIKE, BIAS, INTERMITTENT), `InstrumentEpisode`, funciones puras `apply_*` sobre la serie medida con el `SensorSpan` |
| `data/simulation/scenario_planner.py` | `plan_runs(params, catalog, rng) -> list[RunPlan]`: presupuesto por tasa y severidad (D11) |
| `data/simulation/coverage.py` | `FaultMode`, `coverage_report(...)` por tipo, modo y particion |
| tests | `test_instrument_faults.py`, `test_scenario_planner.py`, `test_coverage.py` |

**Como (manipulaciones; se aplican DESPUES del ruido, con `true_value` preservado)**

| Tipo | Manipulacion |
|---|---|
| drift | rampa en anchos de sobre por hora; nivel "detectable" en la banda medida 13-17 (misma que `InjectionConfig.ramp_rate_per_hour`) y nivel "sub-detectable" ~0,1 C/dia (TDD) etiquetado igualmente: la verdad es la verdad aunque el validador no la vea (Progreso, hallazgo 2) |
| stuck | congela el ultimo valor valido; longitud > 8 muestras (ventana del detector) en tres niveles de longitud |
| noise spike | una muestra `+-k*sigma_normal` con k en nivel bajo/medio/alto; `raw_counts` saturado por el ADC, valor sin recortar (igual que `TEPFaultInjector`) |
| bias | offset persistente desde un instante entre el 25 y el 50 % del run, en fracciones del ancho de sobre (con niveles dentro y fuera del sobre) |
| intermittent | cadena de Markov de dos estados (`p_enter`, `p_exit`) que alterna valor valido e invalido (carril `span.min`/`span.max`); `quality` sigue `good`; la etiqueta cubre toda la ventana y `corrupted` marca solo las lecturas realmente corruptas |

**Planificador**: (1) `n_runs = ceil(n_samples / run_length_steps)`; (2) objetivo etiquetado =
`fault_injection_rate x lecturas`; cuotas por severidad = `pesos x objetivo`; (3) lecturas etiquetadas esperadas
por run de cada tipo (proceso: todos los sensores x duracion; instrumento: sensores afectados x duracion; spike:
n picos); (4) reparto por maximo resto garantizando >= `n_min_runs` por tipo (8/8) y rotando niveles; (5) topes
de densidad por tipo; (6) `ValueError` explicito si la cuota es inalcanzable. Un run de instrumento puede
llevar varios episodios en sensores DISTINTOS del mismo tipo (como los conjuntos disjuntos de `TEPFaultInjector`).

**Etiquetas por lectura**: `process_fault`, `process_severity`, `instrument_fault`, `instrument_severity`,
`injected_at` (inicio del episodio activo), `injected_at_step`, `episode_id`, `corrupted`, `true_value`.

**Medir**: tasa etiquetada alcanzable por tipo (**riesgo**: los de instrumento etiquetan pocos sensores, el 40 %
saldra sobre todo de proceso; si hay que forzar densidades irreales se vuelve a ti antes de forzar); reparto por
severidad; CRITICAL >= 5 %; cobertura (tipos y rejilla) tambien por particion; y, de forma informativa, deteccion
del validador por tipo inyectado y sus falsos negativos esperados (intermittent, bias dentro del sobre, drift
sub-detectable).

**Cierre**: commit `F3-M5: fallos de instrumento, planificador por presupuesto y cobertura`.

### F3-M6 - Dataset, stage DVC `simulate` y tests de seguridad

**Tocar (existentes)**

| Fichero | Cambio |
|---|---|
| `dvc.yaml` | anadir stage `simulate` (seccion 8) y actualizar la cabecera |
| `infra/scripts/Invoke-Pipeline.ps1` | Capa 2: exigir `configs/sim_sensor_spans.yaml`, `sim_sensors.yaml`, `plant_parameters.yaml`, `neutronics_coefficients.yaml`, `water_table.yaml` solo si `$Stage` esta vacio o es `simulate`; Capa 4: resumen de `data/raw/simulation` por directorio `fault_type=*` |
| `tests/safety/test_safety_constraints.py` | quitar el `skip` de modulo y reescribir contra el contrato canonico |
| `data/simulation/dataset_builder.py` | parte 2: `build_dataset`, `write_dataset`, `write_metrics` |

**Salida**: `data/raw/simulation/fault_type=<escenario>/seed=<n>/sim_run_<NNN>.parquet` (TDD 4.4) +
`data/raw/simulation/manifest.json` (rutas relativas, `sort_keys`, resumen por run: filas, escenario, episodios,
residuo de conservacion maximo, `sha256` del parquet) + `metrics/simulation_metrics.json` (`cache: false`: totales,
por tipo, por severidad, cobertura, compliance, semilla).

**Tests de seguridad** (config pequena, < 30 s): (1) tasa etiquetada dentro de +-2 pp de `fault_injection_rate`;
(2) en runs de fallo solo el eje declarado tiene etiqueta y en los normales ninguna; (3) CRITICAL >= minimo;
(4) residuo de conservacion bajo tolerancia en todos los runs; (5) matriz de violaciones esperada; (6)
determinismo (dos construcciones iguales). Se conservan los 3 tests originales en su semantica.

**Como ejecutar**: `.\infra\scripts\Invoke-Pipeline.ps1 -Stage simulate` (genera/actualiza `dvc.lock`); luego
`dvc status` y **`Verify-Phase2.ps1 -SkipCluster` debe seguir saliendo con 2** y con los criterios 2 y 4
CUMPLIDOS (guardia contra regresion, H13). El `-FromScratch` completo se deja para M7 porque exige que los
ficheros nuevos esten commiteados.

**Volumen esperado** (a medir): 50.000 pasos x 27 sensores = 1,35M lecturas; ~60 MB de parquet por extrapolacion
de 46 B/lectura.

**Cierre**: commit `F3-M6: dataset simulado, stage simulate y suite de seguridad`.

### F3-M7 - Verificacion y cierre de fase

**Crear**

| Fichero | Contenido |
|---|---|
| `infra/scripts/Verify-Phase3.ps1` | 4 capas, estados CUMPLIDO/FALLADO/NO MEDIDO, salidas 0/1/2/3, informe `tests/results/phase3_verification.json`; patron de `Verify-Phase2.ps1` **copiando los helpers minimos** (no se refactoriza el script de Fase 2: 32 tests lo ejecutan) |
| `tests/unit/test_verify_phase3_script.py` | patron de `test_verify_phase2_script.py` sobre artefactos sinteticos |
| `Docs/ReactorGuard_Fase3_Simulacion.md` | referencia: layout, contrato de columnas y etiquetas, invariantes, regeneracion, estudio de cadencia, limitaciones (KAERI, realismo) |

**Tocar**: `Docs/ReactorGuard_Progreso.md` (estado general, tabla T5.1-T5.6 con cifras medidas, linea base,
"siguiente paso"), `Docs/physics_constraints.md` (version final).

**Criterios y su verificacion**

| Criterio | Umbral | Como lo mide `Verify-Phase3.ps1` |
|---|---|---|
| Cobertura de fault types | 8/8 y >= 95 % de la rejilla | `metrics/simulation_metrics.json` |
| Compliance | 100 % conservacion + matriz esperada | `metrics/` + suite de seguridad (JUnit, distingue passed de skipped) |
| Volumen | > 1.000.000 lecturas etiquetadas | `manifest.json` |
| Reproducibilidad | hashes identicos | `dvc status` + lock completo + `-FromScratch` (con todo commiteado) |

**Cierre**: resumen de hallazgos, mensaje de commit propuesto, y **prompt de relevo para Fase 4 (T6.1/T7.1)**
con la plantilla del usuario, incluyendo el orden de columnas (C9) y el cp de tabla (D8) como entradas de la PINN.

### F3-M8 (opcional, paralelo) - Calibracion neutronica con OpenMC en Docker

**Paso 0 (solo lectura, antes de descargar nada)**: medir con una peticion HEAD el tamano de los datos nucleares y
confirmar contigo las cifras (imagen 1,93 GB comprimidos medidos + datos por medir) y el espacio libre (191,9 GB).

**Crear**

| Fichero | Contenido |
|---|---|
| `infra/scripts/Invoke-OpenMCCalibration.ps1` | wrapper PowerShell de 4 capas: verifica Docker, `docker pull openmc/openmc:latest` (fijar digest en el informe), `docker run` con la carpeta de trabajo montada |
| `infra/openmc/calibrate_neutronics.py` | script que corre DENTRO del contenedor (API Python de OpenMC): celda cilindrica UO2 (~3 % enriquecimiento) + Zircaloy + agua con boro; barridos de T_fuel, T_moderador (con densidad de la tabla), fraccion de vacio, absorbente de barras; `batches`/`particles` fijados; semilla propia; salida `k_eff +- sigma` en JSON |
| `data/simulation/neutronics_fit.py` | PURO y testeable sin OpenMC: ajuste por minimos cuadrados de alpha_f, alpha_m, alpha_v y peso de barras con incertidumbre, desde una tabla de k_eff |
| `tests/unit/test_neutronics_fit.py` | recupera coeficientes de una tabla sintetica con ruido conocido |

**Tocar**: `configs/neutronics_coefficients.yaml` (proveniencia pasa a `OPENMC` con version, digest de imagen, semilla,
particulas, batches). El script de contenedor vive en `infra/` (fuera de `source` de cobertura); su logica
numerica esta en `data/simulation/neutronics_fit.py` (cubierta).

**Medir**: segundos por corrida, sigma(k) en pcm, tiempo total del barrido y diferencia entre coeficientes de
literatura y calibrados; **repetir las mediciones de M3** con los coeficientes nuevos y reportar el efecto.
El stage `simulate` no cambia: solo cambia el YAML de coeficientes (que ya es `dep`).

## 8. Stage DVC `simulate`

```
simulate:
  cmd: python data/generators/reactor_simulator.py        # siempre via Invoke-Pipeline.ps1
  deps:
    - data/generators/reactor_simulator.py
    - data/simulation                                      # directorio de codigo
    - data/schemas/sensor_reading.py
    - data/schemas/sensor_spans.py
    - data/generators/tep_adapter.py                       # READING_ID_NAMESPACE y readings_from_frame
    - configs/sim_sensors.yaml
    - configs/sim_sensor_spans.yaml
    - configs/plant_parameters.yaml
    - configs/neutronics_coefficients.yaml
    - configs/water_table.yaml
  params:
    - simulation.n_samples
    - simulation.fault_injection_rate
    - simulation.critical_min_fraction
    - simulation.openmc_seed
    - simulation.dt_seconds
    - simulation.run_length_steps
    - simulation.warmup_fraction
    - simulation.start_time
    - simulation.plant_id
    - simulation.severity_weights
    # (rutas y fechas de metadata tambien, como en adapt_tep)
  outs:
    - data/raw/simulation
  metrics:
    - metrics/simulation_metrics.json:
        cache: false
```

- `data/raw/simulation` es hermano de `data/raw/tep`: no hay anidamiento de outs. `.gitignore` ya lo cubre (H14);
  `metrics/*.json` ya tiene excepcion explicita.
- Reproducibilidad bit a bit (convencion 5): `Generator` por run via `SeedSequence.spawn`, tolerancias LSODA fijas,
  orden de filas fijo, sin reloj ni rutas absolutas en parquet/manifiesto, test de hash entre dos ejecuciones
  y entre dos procesos (riesgo: hilos de BLAS; se fijan variables de entorno si hace falta y se mide).
- Entorno: usar SIEMPRE `Invoke-Pipeline.ps1`; DVC borra los outs antes de ejecutar; `-Stage simulate -Force`
  reejecuta el linaje; `WinError 3` con rutas largas (copias en `%TEMP%\rg-<8hex>`, `--no-run-cache`); la
  suspension de Windows falsea mediciones largas (contrastar con Kernel-Power 506/507).

## 9. Reutilizacion (no reescribir)

| Que | Donde | Uso |
|---|---|---|
| `SensorSpan`, `load_sensor_spans`, `to_raw_counts` | `data/schemas/sensor_spans.py` | escalado a cuentas y deteccion de rango en sim |
| `readings_from_frame`, `READING_ID_NAMESPACE` | `data/generators/tep_adapter.py` | reconstruir `SensorReading` y uuid5 deterministas |
| `DEFAULT_ALARM_MARGIN`, `DEFAULT_SPAN_MARGIN`, `MIN_ABSOLUTE_WIDTH` | `data/generators/derive_sensor_spans.py` | mismos margenes para spans de sim |
| patron loader (`_require`) | `data/generators/tep_params.py` | `simulation_params.py`, `plant_params.py` |
| patron `main()` fino | `data/generators/adapt_tep.py` | `reactor_simulator.py` |
| `SensorValidator`, `FaultType`, `Severity` | `data/validation/*` | solo en tests/mediciones offline (no se modifican) |
| patron de capas y estados | `infra/scripts/Verify-Phase2.ps1`, `Invoke-Pipeline.ps1` | `Verify-Phase3.ps1`, extension de preflight |
| builders de frames | `tests/support/tep_frames.py` | patron de `tests/support/sim_frames.py` |

## 10. Criterios de exito medibles

| Criterio | Umbral | Evidencia |
|---|---|---|
| Cobertura de fault types | 8/8 y >= 95 % de la rejilla (D6) | `simulation_metrics.json` |
| Compliance fisico | 100 % conservacion (D7) y matriz esperada | `metrics/` + tests de seguridad |
| Volumen | > 1.000.000 lecturas etiquetadas | `manifest.json` |
| Reproducibilidad | hashes identicos con la misma semilla | 2 ejecuciones + `-FromScratch` |
| Calidad del normal (adicional) | 0 violaciones de regla; FP del validador reportado | M3 |
| Sin regresion Fase 2 | `Verify-Phase2.ps1 -SkipCluster` = 2 con criterios 2 y 4 CUMPLIDOS | M6 y M7 |

Todo criterio que dependa de la cadencia o de GCP se marca NO MEDIDO si no se midio; no se estima.

## 11. Riesgos abiertos

| Riesgo | Sev. | Mitigacion |
|---|---|---|
| Coste/tamano de OpenMC y datos nucleares sin medir | ALTO | M8 aparte; paso 0 de medicion; el pipeline no depende de el |
| Realismo no contrastado (KAERI nunca se uso, H10) | MEDIO | limitacion declarada; no cerrable sin dataset real |
| 180 s oculta firmas (oscilacion, acustica, dT/dt) | MEDIO | D1 + estudio de M3; perfil rapido opcional despues |
| cp/constantes del TDD inconsistentes (C8) | MEDIO | D8; documentado en `physics_constraints.md` |
| Ruido de medicion viola reglas en normal | MEDIO | sigma calibrada y medida en M3 |
| Determinismo roto por BLAS/hilos/parquet | MEDIO | test de hash entre procesos desde M3 |
| 40 % etiquetado inalcanzable sin densidades irreales (instrumento etiqueta pocos sensores) | ALTO | planificador falla explicito; M5 mide y se vuelve a ti |
| Catalogo sin spans -> validador ciego (H7) | ALTO | `derive_sim_spans.py` en M3 y test que falla si un sensor del catalogo no tiene span |
| `simulate` sin lock rompe criterio 4 de Fase 2 (H13) | MEDIO | M6 genera el lock y ejecuta `Verify-Phase2.ps1 -SkipCluster` |
| Modelo ODE rigido con no suavidades (ebullicion) | MEDIO | LSODA + comparacion con Radau en M1; `void` relajado por `tau_v` |
| Deuda de Fase 2 (DLQ, estado de detectores, 100.934 alertas) | - | no se aborda salvo que la simulacion la toque |

## 12. Verificacion de punta a punta

- Por hito: `ruff check .`, `mypy ml/ api/ data/`, `pytest tests/unit tests/safety --cov --cov-fail-under=70`
  (+ >= 95 % por modulo nuevo), `.\infra\scripts\Invoke-Pipeline.ps1`, `dvc status`.
- M6/M7: `Verify-Phase2.ps1 -SkipCluster` (regresion) y `Verify-Phase3.ps1`.
- Cierre de fase: `Verify-Phase3.ps1 -FromScratch` con hash identico, tras tu commit de los ficheros nuevos.
- NUNCA `git commit/amend/push`: cada hito termina con el mensaje propuesto y lo ejecutas tu.
- Al aprobarse este plan el agente entrega en el chat el **prompt de arranque de F3-M1**; cada hito cierra con
  su prompt de relevo entregado por el agente en su mensaje final, segun la seccion 14 (no se guardan en el plan).

## 13. Cosas a tener en cuenta por hito (insumo de cada prompt de relevo)

| Hito | Trampas y puntos de atencion |
|---|---|
| M1 | `iapws` solo esta en el scratchpad (temporal), NO en el venv: instalarlo como dependencia de desarrollo (pip, ~1 MB, puro Python) y reflejarlo en `pyproject.toml`; `scipy` pasa a explicita. Tabla con T 200-360 C y P 60-170 bar porque el vacio despresuriza. El equilibrio se DERIVA (hA, UA), no se teclea: la sonda del preflight fallo justo ahi (n = 0,43). `water_properties` no extrapola. Sin `np.random.seed`. Mediciones de tiempo largas: contrastar con Kernel-Power 506/507 |
| M2 | Roles de constraints desde `sim_sensors.yaml`, no tags cableados. Polaridad: App. B enuncia la ley (T < T_sat), 5.4 la violacion (P < P_sat(T)). Operadores estrictos. NaN no se evalua. Documento en `Docs/physics_constraints.md` (no `docs/`). Cuantificar C5 (dT/dt inobservable a 180 s) |
| M3 | Antes de editar `params.yaml`: avisar (cambia el significado de `fault_injection_rate` y desaparecen claves). Quitar `omit` Y override de mypy. `tests/safety/test_safety_constraints.py` sigue importando `ReactorSimulator`: el modulo debe seguir exportandolo aunque los tests sigan en skip hasta M6. `SPANS_SEED` distinta de `openmc_seed`. El estudio de cadencia NO recalibra el validador. `readings_from_frame` exige 11 columnas. `reading_id` con `run_id`. Sin `quality` distinta de `good` |
| M4 | Los fallos de proceso van a las ECUACIONES (`ModelInputs`), no a la medicion. La ebullicion es no suave: vigilar el integrador y el determinismo. Fijar niveles por objetivo medible, no a ojo. Severidad solo de `FAULT_SEVERITY` |
| M5 | El 40 % etiquetado probablemente salga sobre todo de proceso: medir antes de forzar. Intermittent = valor alterna valido/invalido con `quality = good` y `corrupted` marca las corruptas. Drift detectable en la banda 13-17 anchos/hora medida en Fase 2; el sub-detectable se etiqueta igual. Sensores disjuntos por tipo dentro de un run. `raw_counts` clampa por ADC, el valor no |
| M6 | `Invoke-Pipeline.ps1` siempre; `-Force` reejecuta el linaje; DVC borra outs antes de ejecutar. Tras anadir `simulate`, `Verify-Phase2.ps1 -SkipCluster` debe seguir en 2 con criterios 2 y 4 CUMPLIDOS. `-FromScratch` copia solo `git ls-files`: los ficheros nuevos deben estar commiteados (los commits son del usuario). `WinError 3`: copias en `%TEMP%\rg-<8hex>`, `--no-run-cache`. `.gitignore` ya cubre `data/raw/simulation` |
| M7 | Copiar los helpers minimos de `Verify-Phase2.ps1`, no refactorizarlo (32 tests lo ejecutan). Los tests de PowerShell ejecutan scripts reales sobre artefactos sinteticos. El prompt de relevo de Fase 4 debe fijar el orden de columnas (C9) y el cp de tabla (D8) |
| M8 | Medir con HEAD el tamano de los datos antes de bajar; confirmar cifras con el usuario; fijar el digest de la imagen; la logica numerica vive en `data/simulation/neutronics_fit.py` (cubierta), el script de contenedor en `infra/openmc/` (fuera de cobertura) |

## 14. Protocolo de relevo (al cerrar CADA hito)

Cada hito termina con un **prompt de relevo completo, autosuficiente y pegable**, en un bloque de codigo, que
sustituye al contexto de la sesion. Debe contener, en este orden y con la misma plantilla del prompt de F3-M0:

1. **Referencias de lectura** en orden: `Docs/ReactorGuard_Fase3_Plan.md` (seccion del hito siguiente y las
   secciones 5, 6, 13, 15), `Docs/ReactorGuard_Progreso.md`, `Docs/physics_constraints.md` si existe,
   y solo lo necesario del TDD/Guia. Ruta de respaldo del plan: `C:\Users\guill\.claude\plans\enchanted-chasing-giraffe.md`.
2. **Regla permanente**: nunca `git commit/amend/push`; proponer el mensaje al cerrar.
3. **Hecho hasta ahora**: lista de hitos cerrados con sus entregables y ficheros creados/tocados (rutas).
4. **Linea base MEDIDA al cierre**: ruff, mypy (n ficheros), pytest (passed/skipped, cobertura total y por modulo
   nuevo), dvc status, y las cifras especificas del hito (tiempos, errores, tasas). Pedir reconfirmarlas.
5. **Decisiones y desviaciones vigentes**: D1-D13 mas las que haya aportado el hito, con motivo y evidencia.
6. **Deuda y preguntas abiertas**, con propietario (usuario/asistente).
7. **Cosas a tener en cuenta**: la fila del hito siguiente de la seccion 13 mas lo aprendido en el hito.
8. **Tarea del hito siguiente**: copiar su seccion 7 (Crear/Tocar/Como/Aceptacion) y ordenar los pasos.
9. **Estandares y forma de trabajar** (secciones 6 y 8 del prompt de F3-M0, sin cambios).
10. **Definicion de hecho y entrega**: puertas verdes con cifras, informe de hallazgos del hito, mensaje de
    commit propuesto, y el prompt de relevo del hito que sigue.

Ademas, al cerrar un hito se actualiza la seccion 15 de `Docs/ReactorGuard_Fase3_Plan.md` y las filas T5.x de
`Docs/ReactorGuard_Progreso.md`. Si algo medido contradice el plan, se dice en el informe y se propone el
cambio; no se adapta en silencio.

## 15. Registro de progreso (se rellena al cerrar cada hito)

| Hito | Estado | Fecha | Cifras medidas | Commit (lo pone el usuario) |
|---|---|---|---|---|
| F3-M0 | Hecho (plan aprobado) | 2026-10-11 | puertas: 1836 passed / 3 skipped / 99,41 %; preflight en la seccion 2 | - (sin cambios en el repo) |
| F3-M1 | Hecho | 2026-10-11 | 2053 passed / 4 skipped / 99,47 %; equilibrio 24 h n = 1,0 y max \|dy/dt\| relativo 5,9e-17; cierre de energia 2,0e-15; 4,2 a 6,7 ms/paso (media), p99 10,3 a 10,9; LSODA frente a Radau 6,5e-8; tabla frente a IAPWS 0,16 % (liquido) y 0,43 % (region de operacion); misma traza entre procesos | `b343fb7` |
| F3-M2 | Hecho | 2026-10-11 | 2184 passed / 4 skipped / 99,50 %; `physics_constraints` 100 %; estado verdadero: 0 violaciones en 5 trazas (equilibrio y rampas de 0,5 a 10 %/min), conservacion instantanea 1,9e-20 y por intervalo 1,8e-15; con ruido inicial 0/4000 (equilibrio) y 2/4000 en energia (rampas); regla 1 a 70 % de carga 0,11 % y a 50 % 1,7 %; C5: 1800 K a 180 s, observable solo a <= 16 s | `36c9d3a` |
| F3-M3 | Hecho (la seccion `simulation:` de `params.yaml` se aplico en F3-M4) | 2026-10-11 | 34 runs normales (51.000 instantes, 1.377.000 lecturas): 0 violaciones de las 7 reglas con sigma de lazo 0,15 K (con 0,30 K: 22); peor residuo de energia 3,92 %, de masa 4,41 %; cadencia 1/10/60/180 s: 0 violaciones sobre valores verdaderos, max abs(dT/dt) 0,22 C/s; regla de masa evaluable 15,0 / 94,9 / 99,7 / 99,2 %; validador a 180 s: 90 alertas/1000, todas Kalman en los 5 canales de caudal; determinismo identico entre procesos; 414 pasos/s, dataset completo 146 s y 75,2 MB | `7051d89` |
| F3-M4 | Hecho (a la espera de tu commit) | 2026-10-13 | 2588 passed / 4 skipped / 99,53 % (6148 sentencias, 29 sin cubrir, las mismas de M3); `process_faults`, `scenario`, `dataset_builder`, `thermal_model` 100 %; 9 niveles (3 tipos x 3) dentro de su banda objetivo; matriz de 27 runs: excursion 0/453 y caudal 0/423 violaciones de las 7 reglas, vacio alto 85 % energia / 81 % ebullicion con retardo de 9 a 10 muestras, 0 violaciones fuera de la ventana; conservacion 2,4e-12; digests de runs normales sin cambio (`28d8799d...`) | - |
| F3-M5 | Pendiente | | | |
| F3-M6 | Pendiente | | | |
| F3-M7 | Pendiente | | | |
| F3-M8 | Opcional | | | |

### Desviaciones y hallazgos de F3-M1 (medidos, no estaban en el plan)

| # | Hallazgo | Consecuencia |
|---|---|---|
| M1-1 | La rejilla T 200-360 C a paso 2 tiene **81** nodos, no "~56" (la seccion 7 del plan tenia un error aritmetico); la tabla es 23 x 81 x 3 (5589 valores) | `configs/water_table.yaml` 79 KB |
| M1-2 | El estado tiene **16 variables**, no 13 + controlador: el PI de barras usa dos estados (integral `z` y reactividad `r` con retardo `tau_rod`) y la potencia calorimetrica filtrada es un tercero (`q`) | `StateIndex` fija el orden; `STATE_NAMES` |
| M1-3 | Con `UA_eff = UA x carga` y `T_sec` fija, la carga 0,8 daba n = 0,7195. Se cambio para que **n de equilibrio = carga** (el salto `T_avg - T_sec` se mantiene y `T_sec` sigue al programa de `T_avg`); se elimino el parametro `t_secondary_slope_k_per_load` | `steady_state(load)` es exacto para cualquier carga en (0; 1,5] |
| M1-4 | `observables(y, inputs)` y `advance(y, inputs, dt, inputs_end)`: las magnitudes dependen de las entradas (carga, fuga) y las entradas pueden variar linealmente dentro del intervalo (rampas) | M3 usa `inputs_end` en los cambios de carga |
| M1-5 | `advance` devuelve `StepResult` con `energy_residual_rel`, acumulando E_gen y E_sec como estados auxiliares del integrador: el residuo de conservacion por run (D7) sale gratis | M2/M3 lo registran |
| M1-6 | El coste con diferencias finitas del Jacobiano era 14,2 ms/paso en transitorio; con **Jacobiano analitico** (verificado contra diferencias centradas, 8,5e-8) y derivada en floats de Python baja a 4,2-6,7 ms | El modo de propiedades vivas no lleva `jac` (solo mide el splitting) |
| M1-7 | Nodos con T >= T_sat(P) guardan el liquido saturado (meseta): una rejilla rectangular no puede guardar liquido comprimido sobre la saturacion y el modelo detecta la ebullicion con `t_sat_c` | Documentado en `water_properties.py` |
| M1-8 | El vacio se acota a [0, void_max] al final de cada paso: el integrador dejaba ruido de +-1e-30 | Sin efecto en la energia |
| M1-9 | `roles:` tiene dos entradas mas que las seis del plan (`flow_hot_leg`, `flow_cold_leg`) porque la regla de masa compara pierna caliente y fria; el rol de presion se llama `P` | M2 las usa |
| M1-10 | `sympy` llega transitivamente con `torch`; el test simbolico usa `importorskip`. No se anadio como dependencia explicita | Decision tuya si quieres declararla en `dev` |
| M1-11 | `iapws` NO esta en el venv: se instalo con `--target --no-deps` en el scratchpad solo para generar la tabla. `pip install -e .[dev]` lo traera; un test de contraste con IAPWS real queda en `skip` hasta entonces | `test_nodes_match_iapws` |
| M1-12 | `fault_tiers: {}` y los limites se cargan en `PlantParams`; `ConstraintLimits` (M2) se derivara de `PlantParams.limits` | M2 puede anadir claves a `limits:` |

### Desviaciones y hallazgos de F3-M2 (medidos, no estaban en el plan)

| # | Hallazgo | Consecuencia |
|---|---|---|
| M2-1 | `limits:` recibe **6 claves nuevas** (autorizado): `n_loops`, `mass_balance_tolerance`, `quasi_steady_dt_avg_k_s`, `power_flux_k`, `power_flux_threshold_percent`, `conservation_tolerance`. Se fijaron con la regla "pico del estado verdadero en rampa normal de 5 %/min + 5 sigma del ruido inicial" | `PlantLimits` y su loader crecen; M3 las recalibra con el ruido y el programa de carga definitivos |
| M2-2 | Son **7 reglas**: las 6 de TDD 5.4 mas `mass_balance` (App. B). `RULE_NAMES` fija el orden de columnas; `TDD_RULE_NAMES` son las 6 primeras | La matriz de violaciones tiene 7 columnas |
| M2-3 | Las reglas de relacion usan la **mediana** de los canales del rol (votacion); las de cota (dT/dt, rango de caudal) van canal a canal | Un unico canal averiado queda enmascarado en las de relacion. M5 decide con la matriz delante si hace falta evaluar por canal |
| M2-4 | Con umbral 2,0 pp la regla potencia/flujo violaba en el **estado verdadero** (retardo calorimetrico de 30 s: 2,26 pp a 5 %/min, 3,70 a 10 %/min) | Umbral 4,5 pp; la rampa de 10 %/min sigue sin cubrirse |
| M2-5 | La regla de energia con ruido: sigma ~1,2 % a 100 % (4,2 sigma frente al 5 % del TDD), 0,02 % de violaciones a 85 %, 0,11 % a 70 %, 1,7 % a 50 %, 14,7 % a 30 % | **Decision de M3**: bajar sigma de termopares, acotar la carga o fijar un suelo de potencia. No se invento un parametro en M2 |
| M2-6 | C5 cuantificado: salto minimo 1800 K a 180 s; la tabla cubre 160 K, asi que la regla solo puede saltar a <= 16 s. En los 3 transitorios mas severos del modelo la derivada verdadera maxima es 3,75 C/s | La regla dT/dt queda implementada pero inactiva a 180 s |
| M2-7 | Un `(T, P)` fuera de la rejilla de la tabla no aborta: la fila queda **no evaluada** (la tabla sigue sin extrapolar) | `RuleStats.n_not_evaluated` separa NaN, fuera de tabla y transitorio (regla de masa) |
| M2-8 | `Q_gen` sale del **flujo**; `P_th` (calorimetrica, retardo 30 s) se reserva para la regla 5, para que 1 y 5 no sean la misma comparacion | Documentado en `Docs/physics_constraints.md` 3.1 |
| M2-9 | `min_observable_rate(dt_s)` conserva el nombre del plan pero devuelve un **salto en K por muestra**, no una tasa | Docstring lo aclara; `max_observable_cadence_s` da la cadencia maxima |
| M2-10 | `tests/support/sim_frames.py` (que el plan situaba en M3) se creo en M2 con `Trace`, `run_load_profile` y `wide_frame` del modelo de M1 | **M3 lo extiende**, no lo recrea |
| M2-11 | En el modelo de M1 la demanda de bomba 0 apaga el reactor por realimentacion de vacio y lleva `T_s` a ~185 C (bajo la rejilla de 200 C) | Material de M4 (rango de validez de ese fallo); no se toco `thermal_model.py` |
| M2-12 | `Diagnostics.conservation_residual_mw` es cero por construccion (1,9e-20): la verificacion con contenido es `StepResult.energy_residual_rel` (1,8e-15) | `ConservationReport` informa los dos y, aparte, la regla 1 sobre el estado verdadero |


### Desviaciones y hallazgos de F3-M3 (medidos, no estaban en el plan)

| # | Hallazgo | Consecuencia |
|---|---|---|
| M3-1 | **Decision heredada de energia, cerrada con datos**: con sigma de lazo 0,30 K la regla 1 viola 22 de 51.000 instantes normales (34 runs), incluso a plena carga (1 de 100.000); acotar la carga no llega a 0. Con **0,15 K en los 4 termopares de lazo (T_out/T_in)**: 0 violaciones, residuo maximo 3,92 % frente al 5 % | `configs/sim_sensors.yaml`; justificacion fisica (RTD de lazo estrecho) es un supuesto de ingenieria. Descartados: acotar carga, suelo de potencia (`Docs/physics_constraints.md` 11.1) |
| M3-2 | Programa de carga: rampas acotadas 70-100 % con maximo 5 %/min (tipico de diseno PWR; dato de diseno, no medido), esperas de 60-360 min. La rampa de 10 %/min queda **fuera** de la operacion normal | 5 claves anadidas a `simulation:` (no estaban en la lista del plan): `load_min_fraction`, `load_max_fraction`, `max_ramp_percent_per_min`, `load_hold_minutes_min/max` |
| M3-3 | **Estudio de cadencia** (6 h, 1/10/60/180 s, con rampas): 0 muestras con regla violada sobre valores verdaderos a todas las cadencias; max abs(dT/dt) del estado 0,22 C/s (46 veces bajo el limite); la regla 3 es observable como detector solo desde dt <= 16 s | D1 queda cerrada a 180 s con datos |
| M3-4 | **Trampa de la regla de masa, medida**: sigma de dT_avg/dt medida 0,105 / 0,0104 / 0,0021 / 0,0012 K/s a 1 / 10 / 60 / 180 s; la regla es evaluable en 15,0 / 94,9 / 99,7 / 99,2 % de las muestras medidas | **Decision: no se filtra ni se escala el umbral**; `quasi_steady_dt_avg_k_s` queda calibrado a 180 s y la tabla cuantifica el coste de ir mas fino |
| M3-5 | **Validador de Fase 2 sobre operacion normal, sin recalibrar**: a 180 s 90 alertas por 1000 lecturas, **todas `kalman_anomaly` en los 5 canales de caudal** (~49 % de sus lecturas); los otros 22 sensores dan 0. A 1-10 s el detector de tasa alerta en casi toda lectura (umbral proporcional a dt). Causa probable: `observation_noise = 1,0` fija en unidades de ingenieria en `KalmanResidualDetector` | **Deuda de Fase 2** (el validador no se toca). Condiciona M5: la deteccion por tipo inyectado de los sensores de caudal estara contaminada por estos FP |
| M3-6 | `limits:` recalibrados con el ruido y el programa definitivos: **ninguno cambia de valor** (la cota inferior de `quasi_steady_dt_avg_k_s` baja de 0,0114 a 0,0084; peor masa 4,41 % frente a 5,5 %; peor potencia/flujo 2,32 pp frente a 4,5) | `configs/plant_parameters.yaml` sin tocar |
| M3-7 | `n_samples` se redondea **hacia arriba a runs completos**: 34 runs x 1500 = 51.000 pasos = 1.377.000 lecturas (no 50.000 / 1,35 M). Medido: **75,2 MB** en 34 ficheros (54,6 B/lectura), 146 s extremo a extremo | `SimulationParams.total_steps`; el plan estimaba ~60 MB |
| M3-8 | `simulation_study.py` vive en `data/generators/`, no en `data/simulation/` (como se anuncio): importa `ReactorSimulator` y asi no se invierte la dependencia. M8 repite las mediciones con `python -m data.generators.simulation_study report` | modulo y `tests/unit/test_simulation_study.py` (no estaba en el plan) |
| M3-9 | `physics_constraints.rule_margins` (nuevo, en un modulo de F3-M2): distancia con signo al umbral de cada regla; invariante probado `margen < 0 <=> violacion`. Permite calibrar por distancia minima y no solo por conteo | 100 % de cobertura mantenida |
| M3-10 | La primera version del estudio de cadencia aplicaba `warmup_fraction` (300 pasos = 15 h) a un escenario de 6 h: **no habia ninguna rampa** (max dT/dt = 0). Se detecto al leer las cifras y se corrigio con un calentamiento en segundos (30 min) y `ramp_fraction` como campo del informe | el informe falla visiblemente si el escenario no ejercita dinamica |
| M3-11 | Cada run empieza en el equilibrio de su carga inicial y los runs son contiguos en el tiempo: en la frontera entre dos runs la carga salta de golpe. **No medido** su efecto sobre el validador | candidato a mirar en M5 si aparecen alertas en las fronteras |
| M3-12 | `tests/unit/test_simulation_params.py::TestRepositoryParams` (3 tests) lee el `params.yaml` del repo: **fallan hasta que se aplique el diff de `simulation:`** (la edicion de `params.yaml` fue denegada por el clasificador de permisos y queda para el usuario) | **Resuelto en F3-M4**: la seccion `simulation:` se aplico con los valores del plan y de la fixture de tests; `start_time` (2026-01-01T00:00:00+00:00), `calibration_date` (2025-06-01) y `last_maintenance_date` (2025-12-01) **se inventaron por indicacion del usuario** (no se pudo recuperar el diff original). Los 56 tests de `test_simulation_params.py` pasan; el digest del run 0 sigue siendo `28d8799d...` |

### Desviaciones y hallazgos de F3-M4 (medidos, no estaban en el plan)

| # | Hallazgo | Consecuencia |
|---|---|---|
| M4-1 | **La excursion y la reduccion de caudal no violan ninguna de las 7 reglas en ningun nivel** (0/453 y 0/423, sobre valores verdaderos y con ruido). La firma "balance fuera de 5 %" no se alcanza con magnitudes validas: el desbalance instantaneo maximo es 4,1 % (rampa de 3 min a 500 pcm) | Solo el vacio es visible a las reglas. La matriz de `Docs/physics_constraints.md` (secciones 9 y 12) sustituye las hipotesis H+/H0/? de proceso por tasas medidas; el Rule Engine de Fase 4 (T7.3) no detecta estos dos fallos |
| M4-2 | **El nivel del vacio es un margen de saturacion, no una fuga en bar**: la misma fuga daba, a carga 0,7, un margen de +4,6 K donde a 1,0 ebullia (-5,6 K). `fault_tiers` guarda `saturation_margin_k` (+1,5 / 0,0 / -8,0 K) y `leak_for_margin` calcula la fuga sobre el equilibrio de la carga vigente | Las bandas del vacio valen a 0,7 y a 1,0. El plan hablaba de `fuga(t)` |
| M4-3 | **El modelo necesito un limitador de presion** (`thermal_model.py`, F3-M1 tocado): un run de vacio alto con recuperacion sacaba P de la tabla (173,9 bar) y la matriz con la cronologia por defecto abortaba. El objetivo de presion se acota a [60, 170] bar | Techo fisico (alivio del presurizador), suelo = limite de validez. Digests normales sin cambio; `pressure_limited_steps` cuenta cuando actua (66 en la excursion alta, 0 en el resto) |
| M4-4 | Rangos de validez medidos: excursion <= 700 pcm (a 800 la presion llega al techo), demanda de bomba >= 0,45 (acantilado entre 0,41 y 0,40 a carga 1,0), margen del vacio >= -16 K (61,2 bar a carga 0,7) | Constantes `MAX_REACTIVITY_PCM`, `MIN_PUMP_DEMAND`, `MIN_SATURATION_MARGIN_K`; `parse_fault_tiers` rechaza valores fuera de rango. Cierra la deuda "rango de validez del disparo de bomba" |
| M4-5 | **El ciclo del vacio sale entero de `alpha_void = -0,15`** (provisional): con el coeficiente a cero n = 1,000 en los tres niveles. Con el valor actual el vacio alto lleva n entre 0,15 y 3,8 entre muestras (relajacion) | El tamano del ciclo es dato de F3-M8; la firma "n baja por realimentacion de vacio" se cumple (n min. 0,39 en el medio) pero con picos de n > 1 |
| M4-6 | La regla 4 (caudal) dispara en el vacio solo en los canales de pierna caliente: el insurge del presurizador llega a +2964 kg/s y FT-HL-1/2 pasan de 4950 kg/s (las piernas frias quedan en 4500) | Efecto del presurizador, no de la bomba; M5 lo tendra en cuenta al decidir si evalua por canal |
| M4-7 | La regla 3 (dT/dt) **no dispara con datos muestreados** (max 0,09 C/s por diferencia finita) aunque la derivada verdadera supera 10 C/s en el 80 % de la ventana del vacio alto (max 283 C/s) | D9 se cumple solo como conservacion; C5 confirmado con un fallo real: a 180 s la regla 3 es inobservable |
| M4-8 | La oscilacion de la bomba (300 s, 1-3 %) exige sub-pasos: `simulate_run` gana `fine_inputs` y `substeps`. A 180 s se aliasa a 450 s (`aliased_period_s`); toda oscilacion mas rapida que 360 s es inobservable | La reduccion de caudal cuesta 4 sub-pasos por muestra (~4 s por run) |
| M4-9 | `RunPlan` gana `process_fault: ProcessFaultSpec | None` y se valida (proceso exige envolvente, el resto la prohibe, la ventana debe caber). La ventana etiquetada va de `injected_step` a `onset + hold + recovery`, ambos incluidos; `corrupted = False` | `assemble_long_frame` solo lanza `NotImplementedError` para instrumento. `injected_at_step` NO se anadio (es de M5) |
| M4-10 | Falso positivo de `mass_balance` por ruido relativo con caudal bajo (4/423 en el nivel alto, 0 sobre el estado verdadero): el ruido absoluto del canal pesa mas sobre una `m` menor | Informativo para M5; no se recalibro nada |
| M4-11 | Con la excursion alta la presion MEDIDA supera 170 bar en 19 muestras: las reglas 1 y 6 quedan "no evaluadas" ahi (0/434) | La tabla no extrapola; es un efecto de medicion sobre el techo |
| M4-12 | Coste: vacio 22 / 50 / 59 s por run (hasta 26.000 evaluaciones por paso en el ciclo), excursion 2-9 s, caudal 4 s; la matriz (27 runs) suma 492 s. `test_process_faults.py` tarda ~2,5 min | M6 debe presupuestar los runs de vacio; relajar rtol a 1e-6 lo abarata x1,8 y cambia n en 1,3e-3 (no hecho) |
| M4-13 | `ReactorSimulator.plans` sigue siendo el dataset normal; los fallos se piden con `process_plan` y se integran con el mismo `simulate`. La pareja (fallo, normal) con la misma semilla e indice es un contrafactico exacto hasta la inyeccion | El planificador de M5 compondra los runs |

---

Nota sobre los prompts: **ningun prompt vive en este plan**. El plan solo define su contenido (seccion 14). El
agente entrega el prompt de relevo, completo y pegable, en su mensaje final al terminar cada hito, y el de
arranque de F3-M1 se entrega en el chat al aprobarse el plan.
