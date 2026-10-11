# Constraints fisicos de ReactorGuard (F3-M2)

Implementacion: `data/simulation/physics_constraints.py`. Tests:
`tests/unit/test_physics_constraints.py`. Modelo contra el que se verifican:
`data/simulation/thermal_model.py` (F3-M1). Limites: bloque `limits:` de
`configs/plant_parameters.yaml`. Roles: bloque `roles:` de `configs/sim_sensors.yaml`.

Todas las cifras de este documento son medidas con el codigo del repositorio en
F3-M2 (2026-10-11) salvo las marcadas como **hipotesis**. El punto de operacion
es el PWR generico de 3000 MWt de `plant_parameters.yaml`: no es una planta real
ni se ha contrastado con datos reales.

## 1. Convenciones comunes

| Convencion | Regla |
|---|---|
| Operadores | Estrictos. "Igual al umbral" NO es violacion (por ejemplo `flow = 0` no viola `flow < 0`). |
| NaN / None / texto | No se evalua: la regla devuelve `False` y la fila no cuenta en `evaluable`. Se informa aparte (`RuleStats.n_not_evaluated`). |
| Tabla de agua | Nunca extrapola. Un `(T, P)` fuera de la rejilla (T 200-360 C, P 60-170 bar) tampoco se evalua; no aborta el run. |
| Roles | Se leen de `sim_sensors.yaml` (`roles:`). El modulo no conoce ningun tag. |
| Redundancia | Las reglas de **relacion** (1, 2, 5, 6, masa) usan el valor del rol = mediana de sus canales validos. Las de **cota** (3 y 4) se evaluan canal a canal. Consecuencia: un unico canal averiado queda enmascarado por la votacion salvo que la regla sea de cota. |
| Tiempo | `evaluate_rules` exige `DatetimeIndex` creciente (un run o escenario por frame). El tiempo entre filas sale del indice; un intervalo no positivo deja la fila sin derivada. |

## 2. Roles y magnitudes

| Rol | Tags (catalogo F3-M1) | Magnitud del modelo (`source`) | Unidad |
|---|---|---|---|
| `T_out` | TC-HL-1, TC-HL-2 | `t_c` (salida del nucleo = pierna caliente) | C |
| `T_in` | TC-CL-1, TC-CL-2 | `t_s` (retorno = pierna fria) | C |
| `P` | PT-PRZ-01, PT-PRZ-02 | `p_primary_bar` | bar |
| `flow` | FT-HL-1, FT-CL-1, FT-HL-2, FT-CL-2 | total / `n_loops` por canal | kg/s |
| `flow_hot_leg` | FT-HL-1, FT-HL-2 | `flow_hot_leg_kg_s` (caudal + insurge) | kg/s |
| `flow_cold_leg` | FT-CL-1, FT-CL-2 | `flow_cold_leg_kg_s` | kg/s |
| `flux` | NI-FLUX-01..03 | `power_fraction` x 100 | % del nominal |
| `P_th` | PWR-TH-01 | `q_cal_fraction` x 100 (retardo 30 s) | % del nominal |

## 3. Reglas (TDD 5.4 y Appendix B)

La columna "Violacion" es la condicion que `evaluate_rules` marca `True`.
Las seis primeras son las de TDD 5.4; la septima es la ley de masa de App. B.

| # | Regla (`RULE_NAMES`) | Violacion | Unidades | Funcion |
|---|---|---|---|---|
| 1 | `energy_balance` | `abs(Q_gen - Q_removed) / Q_gen > 0,05` | MW | `energy_balance_residual`, `energy_balance_violated` |
| 2 | `temperature_ordering` | `T_out < T_in` | C | `temperature_ordering_violated` |
| 3 | `max_dt_dt` | `abs(dT/dt) > 10` (en cualquier canal T_out o T_in) | C/s | `finite_difference_rate`, `max_dt_dt_violated` |
| 4 | `flow_range` | `flow < 0` o `flow > pump_max / n_loops` (por canal) | kg/s | `flow_range_violated` |
| 5 | `power_flux` | `abs(P_th - k flux) > umbral` | % del nominal | `power_flux_violated` |
| 6 | `boiling` | `T_out > T_sat(P)`, equivalente a `P < P_sat(T_out)` | C, bar | `boiling_violated` |
| 7 | `mass_balance` | `abs(m_hl - m_cl) / m > tol` solo si `abs(dT_avg/dt) < umbral` | kg/s | `mass_balance_residual`, `mass_balance_violated`, `quasi_steady_mask` |

### 3.1 Energia (regla 1)

```
Q_gen     = P0 * flux / 100                                   [MW]   (flujo neutronico = potencia de fision)
m_total   = n_loops * mediana(canales de caudal)              [kg/s]
T_avg     = (T_out + T_in) / 2                                [C]
Q_removed = m_total * cp_tabla(T_avg, P) * (T_out - T_in) * 1e-6   [MW]
residuo   = abs(Q_gen - Q_removed) / Q_gen
```

- `Q_gen` sale del flujo, no de `P_th`: `P_th` es la potencia cedida al secundario
  con 30 s de retardo y se reserva para la regla 5. Documentado asi para que las
  dos reglas no sean la misma comparacion.
- `cp` viene de `WaterTable` (no 4182 J/(kg K)): en el punto de operacion
  `cp(307,5 C; 155 bar) = 5664 J/(kg K)`, 35 % por encima de 4182.
- `Q_gen <= 0` deja el residuo sin definir (NaN, no evaluado).
- La regla es relativa: con ruido de termopar constante, el residuo crece al bajar
  la carga (seccion 7).

### 3.2 Orden de temperaturas (regla 2)

`T_out < T_in`. Con `T_out == T_in` (potencia nula) no hay violacion; solo cerca de potencia nula,
con el ruido del termopar, el signo de la diferencia deja de ser fiable.

### 3.3 dT/dt (regla 3)

```
rate[k] = (T[k] - T[k-1]) / dt[k]       [C/s],  dt[k] = t[k] - t[k-1]
violacion: abs(rate[k]) > 10
```

El TDD no pone valor absoluto; se interpreta la **magnitud** (un enfriamiento
brusco tambien es "imposible"). El limite de 10 C/s es el de TDD 5.4 y App. B
(D9); el de 50 C/s del detector de tasa de la validacion (TDD 3.3) no se toca.
Observabilidad por cadencia: seccion 5.

### 3.4 Rango de caudal (regla 4)

`flow < 0` o `flow > pump_max`, con `pump_max = pump_max_fraction x caudal nominal
= 1,10 x 18000 = 19800 kg/s` en total, **4950 kg/s por canal** (cada canal mide
total / 4). `flow = 0` no viola esta regla (operador estricto): una bomba parada
la detecta la regla 1 (`Q_removed = 0` con `Q_gen > 0`) y el validador por rango.

### 3.5 Potencia / flujo (regla 5)

`abs(P_th - k * flux) > umbral`, con `P_th` y `flux` en % del nominal y `k = 1`.
Es una comparacion de **instantes**: `P_th` lleva un retardo de 30 s respecto al
flujo, de modo que en una rampa de carga hay una diferencia verdadera
`tau * dn/dt` (2,26 puntos a 5 %/min, medida).

### 3.6 Ebullicion (regla 6) y polaridad

```
Admisible (App. B):     T < T_sat(P)
Violacion (TDD 5.4):    P < P_sat(T)        <=>   T > T_sat(P)   (P_sat creciente)
```

App. B enuncia la **ley** (`T < T_sat`), 5.4 la **violacion** (`P < P_sat(T)`). Son
la misma regla salvo en el borde: la ley no se cumple en `T == T_sat`; la
violacion estricta de 5.4 tampoco se dispara ahi. Se sigue 5.4 (operador estricto
del plan): `T_out == T_sat` no viola, `T_sat + eps` si (test con eps = 1e-9 K).
Se usa la temperatura de la pierna caliente; la ebullicion local en canales
calientes no se modela. En el modelo, el vacio empieza `void_onset_offset_k = 2 K`
**antes** de `T_sat` (ebullicion subenfriada), por eso la regla 6 puede reaccionar
despues del inicio del vacio.

### 3.7 Masa (App. B)

```
m         = (m_hl + m_cl) / 2
residuo   = abs(m_hl - m_cl) / m
aplica si abs(dT_avg/dt) < quasi_steady_dt_avg_k_s     (dT_avg de las mediciones)
violacion: residuo > mass_balance_tolerance
```

En el modelo, `m_hl - m_cl` es el insurge del presurizador,
`900 kg/s por K/s x dT_avg/dt`, es decir `residuo = 0,05 x |dT_avg/dt|` en K/s
(medido: 0,0037 con 0,073 K/s). Las filas en transitorio (y la primera, sin
derivada) quedan **no evaluadas**, no como "cumple".

## 4. Conservacion sobre el estado completo (D7)

Sobre el estado del modelo (16 variables), no sobre mediciones:

```
dU/dt    = C_f dT_f/dt + C_c dT_c/dt + C_s dT_s/dt            [MW]  (C en MJ/K)
residuo  = abs(dU/dt - (Q_gen - Q_sec)) / P0                   (instantaneo)
residuo  = abs(dU - (E_gen - E_sec)) / E_gen                   (sobre un intervalo)
```

Hay dos cierres y no son lo mismo:

| Cierre | Fuente | Que detecta | Medido |
|---|---|---|---|
| Instantaneo | `conservation_residuals(diagnostics)` | Una regresion algebraica del modelo (es cero por construccion: la suma de las tres ecuaciones de nodo) | max 1,9e-20 (rampas), 0 (equilibrio) |
| Por intervalo | `StepResult.energy_residual_rel` | El error de integracion (LSODA, propiedades congeladas por intervalo) | max 1,8e-15 (todas las trazas) |

`ConservationReport.compliant` es cierto si ningun instante ni intervalo supera
`conservation_tolerance = 1e-6`. Aparte se informa la regla 1 sobre el estado
verdadero (`max_primary_imbalance`, `n_primary_imbalance_violations`): un
transitorio rapido puede violarla **sin** violar la conservacion, porque
`Q_gen - Q_core` es la energia que se acumula en combustible y nucleo. Es lo que
recoge la matriz de la seccion 8: `compliance = conservacion 100 % + violaciones
esperadas`, no "ninguna regla violada".

## 5. Observabilidad por cadencia (C5)

Entre dos muestras separadas `dt`, la regla 3 solo puede saltar si
`|T[k] - T[k-1]| > 10 x dt`.

| Cadencia `dt` | Salto minimo observable (`min_observable_rate`) | Frente al rango de la tabla (160 K) |
|---|---|---|
| 1 s | 10 K | observable |
| 10 s | 100 K | observable |
| 16 s | 160 K | limite: toda la tabla en un salto |
| 60 s | 600 K | inobservable |
| **180 s** | **1800 K** | **inobservable** |

A 180 s la regla 3 **no puede saltar con agua liquida** (1800 K frente a un rango
de 160 K): se conserva por completitud y para cadencias finas, pero a 180 s no
es un detector. Complemento medido sobre los transitorios mas severos del
modelo de F3-M1 (fallos aplicados desde el equilibrio, 20 muestras de 180 s):

| Transitorio | max abs(dT) por muestra (T_f, T_c, T_s) | max abs(dT/dt) instantaneo |
|---|---|---|
| Fuga equivalente a 30 bar | 0,34 K | 0,59 C/s |
| Insercion de +500 pcm | 55,8 K | 0,59 C/s |
| Demanda de bomba a 0 | 675 K | 3,75 C/s |

Ninguno alcanza 10 C/s: en este modelo la derivada verdadera tampoco llega al
limite. (El disparo de bomba lleva al modelo fuera de su validez fisica: el
reactor se apaga por el vacio y la pierna fria baja de 200 C; es material de
F3-M4, no de las constraints.)

## 6. Limites y valores

Provisionales hasta F3-M3. Los de F3-M2 se fijaron con una regla explicita:
**pico del estado verdadero en una rampa normal de 5 %/min + 5 sigma del
estadistico con el ruido inicial del catalogo**, redondeado hacia arriba.

| Clave (`limits:`) | Valor | Origen |
|---|---|---|
| `max_dt_c_per_s` | 10 C/s | TDD 5.4 y App. B (D9), de F3-M1 |
| `energy_imbalance_fraction` | 0,05 | TDD 5.4, de F3-M1 |
| `pump_max_fraction` | 1,10 | F3-M1 (19800 kg/s total, 4950 por canal) |
| `n_loops` | 4 | 4 lazos x 4500 kg/s (preflight); el catalogo usa ganancia 0,25 (test de coherencia) |
| `mass_balance_tolerance` | 0,055 | 0,0020 (pico verdadero a 5 %/min) + 5 x 0,0100 (sigma del desajuste con ruido) = 0,052 |
| `quasi_steady_dt_avg_k_s` | 0,02 K/s | cota inferior 0,0114 (0,0055 pico muestreado a 5 %/min + 5 x 0,00118 de ruido) y superior 1,1 (= tol / 0,05); 0,02 son 3,6 K entre muestras |
| `power_flux_k` | 1,0 | ambos canales normalizados al nominal |
| `power_flux_threshold_percent` | 4,5 pp | 2,26 (retardo calorimetrico a 5 %/min) + 5 x 0,40 (sigma de la diferencia) = 4,26 |
| `conservation_tolerance` | 1e-6 | rtol del integrador 1e-8; medido 2e-15 |

Con el umbral inicial de 2,0 pp la regla 5 se violaba **en el estado verdadero**
en rampas normales (1 de 17 muestras a 5 %/min, 2 de 15 a 10 %/min): ese es el
motivo de 4,5. A 10 %/min (3,70 pp de pico) el umbral sigue sin cubrirla.

## 7. Mediciones de F3-M2

Cadencia 180 s, semilla `SeedSequence(20261011)`, sigma inicial del catalogo.

### 7.1 Estado verdadero (sin ruido): esperado 0

| Escenario | Muestras | Violaciones (7 reglas) | Pico `abs(P_th - flux)` [pp] | Pico desajuste de masa | Pico imbalance regla 1 | Conservacion |
|---|---|---|---|---|---|---|
| Equilibrio 100 % | 40 | 0 | 0 | 0 | 1,4e-15 | cumple |
| Rampa 0,5 %/min (70-100-70 %) | 53 | 0 | 0,26 | 0,00032 | 0,00076 | cumple |
| Rampa 1 %/min | 33 | 0 | 0,54 | 0,00062 | 0,0015 | cumple |
| Rampa 5 %/min | 17 | 0 | 2,26 | 0,0020 | 0,0068 | cumple |
| Rampa 10 %/min | 15 | 0 | 3,70 | 0,0037 | 0,016 | cumple |

### 7.2 Con ruido a los sigma iniciales (informativo; M3 calibra)

| Escenario | Filas | Violaciones por regla |
|---|---|---|
| Equilibrio 100 % | 4000 | 0 en las 7 |
| Rampas 70-100 % (mezcla de cargas) | 4000 | 2 en `energy_balance`; 0 en las demas |

Estadisticos con ruido (equilibrio 100 %): residuo de energia media 0,0095,
sigma ~0,012 (maximo 0,044); `P_th - flux` sigma 0,40 pp (maximo 1,41);
desajuste de masa sigma efectiva 0,0100 (maximo 0,037); `dT_avg/dt` entre
muestras sigma 0,00118 K/s (maximo 0,0043).

### 7.3 Regla de energia con ruido, por nivel de carga (20000 filas por nivel, equilibrio)

| Carga | Delta T verdadero | Violaciones regla 1 | Tasa |
|---|---|---|---|
| 1,00 | 29,7 K | 0 | 0 |
| 0,85 | 25,3 K | 4 | 0,02 % |
| 0,70 | 20,9 K | 21 | 0,11 % |
| 0,50 | 15,0 K | 349 | 1,7 % |
| 0,30 | 9,0 K | 2934 | 14,7 % |
| 0,20 | 6,0 K | 6460 | 32,3 % |

Las otras seis reglas: 0 violaciones en todos los niveles. Con sigma de termopar
0,3 K el residuo de energia tiene sigma ~1,2 % a plena carga y crece
al bajar la carga (sigma x1,3 a 70 %, x1,8 a 50 %, x4,3 a 20 %): el umbral del TDD (5 %) es 4,2 sigma a 100 %, 3,2 a 70 % y 2,4 a 50 %.
Decision para F3-M3 (no se toma aqui): bajar el sigma de los termopares,
restringir el rango de carga de operacion normal, o no evaluar la regla por
debajo de un suelo de potencia.

## 8. Inconsistencias del TDD

| # | TDD | Problema | Que se hace |
|---|---|---|---|
| 1 | 5.2: `Cp = 4182` | Es agua a ~20 C. A 307,5 C y 155 bar `cp = 5664` (tabla, IAPWS-97): con 4182 `Q_removed` queda 26 % corto y la regla 1 marcaria 26 % de desbalance con la planta sana | `cp` de la tabla versionada (D8) |
| 2 | 5.2: `q_gen = flux * 3.2e-11 * 2.5e18` | `3,2e-11 J x 2,5e18 /s = 8,0e7 W = 80 MW` por unidad de flujo, 37,5 veces menos que los 3000 MW del TDD/preflight; con flujo 1 el desbalance seria 3650 % | `Q_gen = P0 x flux / 100`, con `P0` de `plant_parameters.yaml` |
| 3 | App. B y 5.4: 10 C/s; 3.3: 50 C/s | Dos limites para "dT/dt" | 10 C/s sobre el estado verdadero (D9); el 50 del validador no se toca |
| 4 | App. B: `T < T_sat(P)`; 5.4: `P < P_sat(T)` | Ley frente a violacion: difieren en el borde (3.6) | Se sigue 5.4, operador estricto |
| 5 | App. B: "Pearson correlation monitor" (potencia/flujo); 5.4: `abs(P_th - k flux) > umbral` | Una es una caracteristica estadistica, la otra una regla instantanea | Se implementa la de 5.4; la correlacion queda como feature |
| 6 | App. B: masa "hard rule check on flow sensors"; 5.4 no la lista | Solo hay seis reglas en 5.4 | Septima regla `mass_balance`, solo en cuasi-estacionario |
| 7 | 5.4: `dT/dt > 10` | Sin valor absoluto ni cadencia | Magnitud; a 180 s inobservable (seccion 5) |
| 8 | 5.2 indexa `[flux, flow, t_in, t_out, flow_out]` | El orden de columnas condicionara Fase 4 (C9) | Fuera de M2; M7 lo fija en el relevo de Fase 4 |

## 9. Matriz de violaciones esperadas

Leyenda: `0*` = medido en F3-M2 (cero violaciones); `H+` = **hipotesis** de que
la regla se viola; `H0` = hipotesis de que no se viola; `?` = sin hipotesis
fundada; `[M4]`, `[M5]` = hito donde se mide. **Nada de H+/H0/? es una medicion.**
Se sustituyen por la tasa medida en F3-M4 a F3-M6.

| Escenario | 1 energia | 2 orden | 3 dT/dt | 4 caudal | 5 pot./flujo | 6 ebullicion | 7 masa |
|---|---|---|---|---|---|---|---|
| Normal, estado verdadero | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| Normal con ruido inicial | 0* (2/4000 en rampas) | 0* | 0* | 0* | 0* | 0* | 0* |
| Proceso: excursion de temperatura del nucleo [M4] | ? | H0 | H0 | H0 | ? | H+ (si `T_out` > `T_sat`) | H0 |
| Proceso: reduccion de caudal [M4] | ? | H0 | H0 | H0 | ? | ? | H0 (las dos piernas bajan juntas) |
| Proceso: formacion de vacio [M4] | ? | H0 | H0 | H0 | ? | H+ (con 2 K de retraso respecto al inicio del vacio) | ? |
| Instrumento: deriva [M5] | H0 hasta ~3 K sobre un canal de T_out (la mediana de 2 reparte b/2 y 5 % de 29,7 K = 1,5 K) | H0 | H0 | H0 | H0 | H0 | H0 |
| Instrumento: bloqueado [M5] | ? | ? | H0 | ? (si queda en rango) | ? | ? | ? |
| Instrumento: pico de ruido [M5] | H0 | H0 | H0 (1800 K a 180 s) | H+ (si el pico supera 4950 en un canal de caudal) | H0 | H0 | H0 |
| Instrumento: sesgo [M5] | H+ (a partir de ~3 K en un termopar de T_out) | H0 | H0 | ? | ? | ? | ? |
| Instrumento: intermitente [M5] | ? | ? | ? | ? | ? | ? | ? |

## 10. Limitaciones y preguntas abiertas

- **Baja carga**: la regla 1 es relativa a `Delta T` y con sigma de termopar
  0,3 K se vuelve ruidosa por debajo de ~70 % de carga (7.3); la regla 2 solo se
  degrada cerca de potencia nula (0 violaciones hasta carga 0,20). F3-M3
  decide sigma, rango de operacion o suelo de potencia; aqui no se inventa ese
  parametro.
- **Votacion**: la mediana enmascara un unico canal averiado en las reglas de
  relacion. Si M5 necesita que un canal defectuoso viole una regla, habra que
  evaluar por canal; la decision es de M5 con la matriz delante.
- **Cota fisica a 180 s**: la regla 3 es inobservable; la deteccion de saltos de
  temperatura la hace el validador de Fase 2 por rango.
- **Disparo de bomba**: el modelo de F3-M1 apaga el reactor por vacio y lleva
  `T_s` a ~185 C (bajo la rejilla de 200 C). Las constraints lo tratan como
  "fuera de tabla, no evaluado" en vez de abortar; M4 debe decidir el rango de
  validez de ese fallo.
- **Umbrales**: `power_flux_threshold_percent` cubre la rampa normal de 5 %/min
  pero no la de 10 %/min; M3 define el programa de carga y recalibra.
