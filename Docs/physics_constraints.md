# Constraints fisicos de ReactorGuard (F3-M2, calibrado en F3-M3, matriz de F3-M4)

Implementacion: `data/simulation/physics_constraints.py`. Tests:
`tests/unit/test_physics_constraints.py`. Modelo contra el que se verifican:
`data/simulation/thermal_model.py` (F3-M1). Limites: bloque `limits:` de
`configs/plant_parameters.yaml`. Roles: bloque `roles:` de `configs/sim_sensors.yaml`.

Todas las cifras de este documento son medidas con el codigo del repositorio en
F3-M2, F3-M3 (2026-10-11) y F3-M4 (2026-10-13) salvo las marcadas como
**hipotesis**. Las secciones 1 a 10 son de F3-M2 (con notas de F3-M3 donde una
decision se cerro); las mediciones de F3-M3 estan en la seccion 11 y las de los
fallos de proceso de F3-M4 en la 12. El punto de operacion
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

Ninguno alcanza 10 C/s: en estos tres transitorios (aplicados desde el
equilibrio) la derivada verdadera tampoco llega al limite. (El disparo de bomba
lleva al modelo fuera de su validez fisica: el reactor se apaga por el vacio y la
pierna fria baja de 200 C.) **Matiz de F3-M4**: el vacio SI supera 10 C/s sobre el
estado verdadero (maximo 283 C/s en el nivel alto, 12.5): la regla 3 se viola
sobre la derivada instantanea y sigue siendo inobservable entre muestras de 180 s.

## 6. Limites y valores

Fijados en F3-M2 con una regla explicita (**pico del estado verdadero en una
rampa normal de 5 %/min + 5 sigma del estadistico con el ruido del catalogo**,
redondeado hacia arriba) y **recalibrados en F3-M3** con el ruido y el programa
de carga definitivos (seccion 11.4): ninguno cambio de valor. La rampa de
5 %/min es `simulation.max_ramp_percent_per_min`.

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
Decision que se dejo para F3-M3 y que se cerro alli: **sigma de los 4 termopares
de lazo = 0,15 K** y rango de carga 70-100 % (seccion 11.1). Las cifras de esta
seccion 7.3 son las del sigma inicial de 0,30 K.

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
| 8 | 5.2 indexa `[flux, flow, t_in, t_out, flow_out]` | El orden de columnas condicionara Fase 4 (C9) | Fuera de M2; M7 lo fija en las notas de arranque de Fase 4 |

## 9. Matriz de violaciones esperadas

Leyenda: `0*` = medido (cero violaciones); `n/N (p %)` = medido: violaciones sobre
muestras evaluables de la ventana del fallo, tasa y retardo en muestras (`dK`: la
primera violacion llega K muestras de 180 s despues del inicio); `k/14 sens.` (solo
filas de instrumento) = sensores, de los 14 que algun rol de las constraints lee, que violan la
regla, y `n/N` esta agregado solo sobre ellos. Las filas de proceso son de F3-M4 (seccion 12);
las de instrumento, de F3-M5 (seccion 13), con la mediana del rol (votada). La seccion
13.3 trae las mismas celdas por canal y la 13.4 los grupos de sensores que violan cada
regla. Ya no queda ninguna hipotesis (H+/H0/?) en la tabla.

| Escenario | 1 energia | 2 orden | 3 dT/dt | 4 caudal | 5 pot./flujo | 6 ebullicion | 7 masa |
|---|---|---|---|---|---|---|---|
| Normal, estado verdadero | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| Normal con ruido inicial (F3-M2) | 0* (2/4000 en rampas) | 0* | 0* | 0* | 0* | 0* | 0* |
| Normal con ruido calibrado (F3-M3, 34 runs, 51.000 instantes) | 0* (peor residuo 3,92 %) | 0* | 0* | 0* | 0* | 0* | 0* (peor 4,41 %) |
| Proceso: excursion de temperatura, nivel bajo/medio/alto (F3-M4, 453 muestras por nivel) | 0* | 0* | 0* | 0* | 0* | 0* (margen minimo a T_sat 12,9 K en el alto) | 0* |
| Proceso: reduccion de caudal, nivel bajo/medio/alto (F3-M4, 423 muestras por nivel) | 0* | 0* | 0* | 0* | 0* | 0* (margen minimo 3,9 K en el alto) | 0* verdadero; 4/423 (0,9 %) con ruido en el alto |
| Proceso: vacio, nivel bajo (F3-M4, 453 muestras) | 107/453 (23,6 %) d46 | 0* | 0* (inobservable a 180 s) | 0* | 79/453 (17,4 %) d46 | 0* (el vacio incipiente no llega a T_sat) | 0* |
| Proceso: vacio, nivel medio (F3-M4, 453 muestras) | 361/453 (79,7 %) d12 | 0* | 0* (inobservable a 180 s) | 87/453 (19,2 %) d32 | 361/453 (79,7 %) d13 | 176/453 (38,9 %) d16 | 185/360 (51,4 %) d23 |
| Proceso: vacio, nivel alto (F3-M4, 453 muestras) | 386/453 (85,2 %) d10 | 0* | 0* (inobservable a 180 s) | 82/453 (18,1 %) d13 | 386/453 (85,2 %) d10 | 365/453 (80,6 %) d9 | 207/318 (65,1 %) d12 |
| Instrumento: deriva bajo (F3-M5, votada) | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| Instrumento: deriva medio (F3-M5, votada) | 4/14 sens.: 1347/1920 (70,2 %) d44 | 0* | 0* | 4/14 sens.: 181/1920 (9,4 %) d141 | 1/14 sens.: 439/480 (91,5 %) d18 | 0* | 4/14 sens.: 275/1920 (14,3 %) d139 |
| Instrumento: deriva alto (F3-M5, votada) | 6/14 sens.: 304/396 (76,8 %) d0 | 4/14 sens.: 125/320 (39,1 %) d7 | 0* | 4/14 sens.: 265/320 (82,8 %) d1 | 1/14 sens.: 80/80 (100,0 %) d0 | 4/14 sens.: 103/254 (40,6 %) d7 | 4/14 sens.: 312/320 (97,5 %) d1 |
| Instrumento: bloqueado bajo (F3-M5, votada) | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| Instrumento: bloqueado medio (F3-M5, votada) | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| Instrumento: bloqueado alto (F3-M5, votada) | 3/14 sens.: 238/1440 (16,5 %) d171 | 0* | 0* | 0* | 1/14 sens.: 356/480 (74,2 %) d60 | 0* | 0* |
| Instrumento: pico bajo (F3-M5, votada) | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| Instrumento: pico medio (F3-M5, votada) | 4/14 sens.: 6/8 (75,0 %) d0 | 0* | 0* | 4/14 sens.: 4/8 (50,0 %) d0 | 1/14 sens.: 2/2 (100,0 %) d0 | 0* | 4/14 sens.: 8/8 (100,0 %) d0 |
| Instrumento: pico alto (F3-M5, votada) | 4/14 sens.: 8/8 (100,0 %) d0 | 0* | 0* | 4/14 sens.: 4/8 (50,0 %) d0 | 1/14 sens.: 2/2 (100,0 %) d0 | 0* | 4/14 sens.: 8/8 (100,0 %) d0 |
| Instrumento: sesgo bajo (F3-M5, votada) | 2/14 sens.: 59/3600 (1,6 %) d176 | 0* | 0* | 0* | 1/14 sens.: 1799/1800 (99,9 %) d0 | 0* | 0* |
| Instrumento: sesgo medio (F3-M5, votada) | 4/14 sens.: 6211/7200 (86,3 %) d0 | 0* | 0* | 0* | 1/14 sens.: 1800/1800 (100,0 %) d0 | 0* | 3/14 sens.: 4/5400 (0,1 %) d314 |
| Instrumento: sesgo alto (F3-M5, votada) | 4/14 sens.: 7200/7200 (100,0 %) d0 | 0* | 0* | 4/14 sens.: 3558/7200 (49,4 %) d0 | 1/14 sens.: 1800/1800 (100,0 %) d0 | 0* | 4/14 sens.: 5876/7200 (81,6 %) d0 |
| Instrumento: intermitente bajo (F3-M5, votada) | 4/14 sens.: 70/960 (7,3 %) d0 | 0* | 0* | 4/14 sens.: 14/960 (1,5 %) d0 | 1/14 sens.: 8/240 (3,3 %) d0 | 0* | 4/14 sens.: 25/960 (2,6 %) d0 |
| Instrumento: intermitente medio (F3-M5, votada) | 4/14 sens.: 285/1920 (14,8 %) d0 | 0* | 0* | 4/14 sens.: 184/1920 (9,6 %) d0 | 1/14 sens.: 82/480 (17,1 %) d0 | 0* | 4/14 sens.: 313/1920 (16,3 %) d0 |
| Instrumento: intermitente alto (F3-M5, votada) | 4/14 sens.: 1444/2880 (50,1 %) d0 | 0* | 0* | 4/14 sens.: 653/2880 (22,7 %) d0 | 1/14 sens.: 325/720 (45,1 %) d0 | 0* | 4/14 sens.: 1442/2880 (50,1 %) d0 |

## 10. Limitaciones y preguntas abiertas

- **Baja carga**: la regla 1 es relativa a `Delta T` y con sigma de termopar
  0,3 K se vuelve ruidosa por debajo de ~70 % de carga (7.3); la regla 2 solo se
  degrada cerca de potencia nula (0 violaciones hasta carga 0,20). **Cerrado en
  F3-M3** (11.1): sigma de lazo 0,15 K dentro de 70-100 %. Por debajo de ~50 % de
  carga la regla 1 no se ha medido con el sigma nuevo: no es operacion normal
  del dataset.
- **Votacion**: la mediana enmascara un unico canal averiado en las reglas de
  relacion. **Medido en F3-M5** (13.3-13.5): votada, los 3 canales de flujo neutronico
  quedan ocultos y los termopares (2 canales) llegan diluidos a la mitad; por canal
  se ven. La decision de votar o no en el Rule Engine queda para Fase 4 (T7.3); el
  dataset guarda las lecturas por sensor, de modo que ambas evaluaciones son posibles.
- **Cota fisica a 180 s**: la regla 3 es inobservable; la deteccion de saltos de
  temperatura la hace el validador de Fase 2 por rango.
- **Disparo de bomba**: el modelo de F3-M1 apaga el reactor por vacio y lleva
  `T_s` a ~185 C (bajo la rejilla de 200 C). Las constraints lo tratan como
  "fuera de tabla, no evaluado" en vez de abortar. **Cerrado en F3-M4** (12.5): la
  reduccion de caudal es valida hasta una demanda de 0,45; por debajo (acantilado
  entre 0,41 y 0,40 a carga 1,0) el fallo ya es una ebullicion y es el escenario de
  vacio.
- **Umbrales**: `power_flux_threshold_percent` cubre la rampa normal de 5 %/min
  pero no la de 10 %/min. F3-M3 fija la rampa maxima normal en 5 %/min
  (`simulation.max_ramp_percent_per_min`), de modo que 10 %/min queda fuera de la
  operacion normal; si F3-M4 o F3-M5 quieren una rampa mas rapida como
  escenario habra que subir el umbral o tratarla como violacion esperada.

## 11. Mediciones de F3-M3

Semillas: dataset `openmc_seed = 42` (34 runs de 1500 pasos a 180 s = 51.000
instantes, 1.377.000 lecturas), spans `SPANS_SEED = 20260905`, estudio de cadencia
`CADENCE_STUDY_SEED = 20261012`. Herramienta: `data/generators/simulation_study.py`
(`python -m data.generators.simulation_study report --params ... --output ...`); el
informe JSON de cada cifra sale de ahi.

### 11.1 Decision heredada de F3-M2: regla de energia a baja carga

Regla 1 sobre mediciones normales, 34 runs (51.000 instantes), carga 70-100 %,
solo cambia el sigma de los 4 termopares de lazo (TC-HL-1/2, TC-CL-1/2: los de
`T_out` y `T_in`). Los TC-CORE no participan en ninguna regla y quedan a 0,3 K.

| sigma de lazo [K] | Violaciones regla 1 (de 51.000) | Residuo maximo medido | Distancia minima al umbral (5 %) |
|---|---|---|---|
| 0,30 (inicial) | **22** | 6,43 % | -1,43 pp |
| 0,20 | 0 | 4,69 % | 0,31 pp |
| **0,15 (elegido)** | **0** | **3,92 %** | **1,08 pp** |
| 0,10 | 0 | 3,47 % | 1,53 pp |

Sondeo previo (100.000 muestras por celda, equilibrio, sigma 0,30 K): violaciones
a carga 1,00 / 0,90 / 0,80 / 0,70 = 1 / 14 / 38 / 144. **Acotar la carga no basta**:
incluso a plena carga el sigma inicial viola la regla.

Opciones consideradas:

| Opcion | Resultado | Decision |
|---|---|---|
| Bajar el sigma de lazo a 0,15 K | 0 violaciones, residuo maximo 3,92 % | **Elegida** |
| Acotar la carga (>= 0,90) con 0,30 K | 14 violaciones a 0,90; 1 a 1,00 | Descartada: no llega a 0 |
| Suelo de potencia para la regla 1 | No hace falta en 70-100 % | Descartada: cambia la semantica de la regla de F3-M2 |

Justificacion fisica de 0,15 K: los termopares de pierna caliente/fria de un PWR
son de lazo estrecho y alta precision (en planta real, RTD de 0,1-0,2 K); el
contrato `SensorReading` no tiene tipo RTD y se declaran `thermocouple`. Es un
**supuesto de ingenieria**, no un dato de una planta concreta.

### 11.2 Reglas sobre mediciones normales: distancia minima al umbral

Catalogo final, 34 runs, 51.000 instantes. `margen` = distancia con signo al umbral
(`rule_margins`): positiva = dentro de lo admisible.

| Regla | Violaciones | Evaluables | Umbral | Margen minimo | Margen medio | Peor valor medido |
|---|---|---|---|---|---|---|
| 1 energia | 0 | 51.000 | 5 % | 0,0108 | 0,0431 | 3,92 % |
| 2 orden T | 0 | 51.000 | `T_out - T_in = 0` | 20,5 K | 25,3 K | `T_out - T_in` = 20,5 K |
| 3 dT/dt | 0 | 50.966 | 10 C/s | 9,986 C/s | 9,998 C/s | 0,014 C/s |
| 4 caudal | 0 | 51.000 | [0; 4950] kg/s por canal | 235 kg/s | 404 kg/s | 4715 kg/s |
| 5 pot./flujo | 0 | 51.000 | 4,5 pp | 2,18 pp | 4,18 pp | 2,32 pp |
| 6 ebullicion | 0 | 51.000 | `T_sat(P)` | 23,7 K | 26,6 K | `T_out` a 23,7 K de `T_sat` |
| 7 masa | 0 | 50.966 | 0,055 | 0,0109 | 0,0470 | 4,41 % |

Sigma de la regla (estimado desde el residuo medio, `media(|x|) = 0,798 sigma`):
energia 0,87 %, masa 1,00 %. El umbral queda a **5,8 sigma** (energia) y
**5,5 sigma** (masa) de la media; el peor de 51.000 instantes llega a 4,5 sigma y
4,4 sigma. Las 34 filas no evaluables de las reglas 3 y 7 son la primera de cada
run (sin derivada).

### 11.3 Estudio de cadencia (D1)

Escenario normal identico en tiempo fisico a 1, 10, 60 y 180 s: 6 h simuladas,
calentamiento de 30 min, esperas entre rampas de 20-40 min, rampas de hasta
5 %/min; 13-19 % de las muestras estan en rampa. Los umbrales de las reglas y
los spans del validador **no se recalibran** entre cadencias. Un estudio previo
sin rampas (calentamiento de 15 h) se descarto: no ejercitaba ninguna dinamica.

| dt [s] | Muestras | Lecturas | Muestras con alguna regla violada (valores verdaderos) | max abs(dT/dt) del estado [C/s] | max abs(dT/dt) de `T_out`/`T_in` [C/s] | Salto minimo observable [K] |
|---|---|---|---|---|---|---|
| 1 | 21.601 | 583.227 | **0** | 0,217 | 0,0144 | 10 |
| 10 | 2.161 | 58.347 | **0** | 0,218 | 0,0142 | 100 |
| 60 | 361 | 9.747 | **0** | 0,211 | 0,0133 | 600 |
| 180 | 121 | 3.267 | **0** | 0,176 | 0,0097 | 1.800 |

La maxima derivada verdadera (0,22 C/s, en el combustible, que es el nodo rapido)
es 46 veces menor que el limite de 10 C/s: la regla 3 no se dispara en
operacion normal a ninguna cadencia, y a 180 s ni siquiera puede (C5: 1.800 K).
**Observable como detector desde `dt <= 16 s`** (toda la tabla, 160 K, cabe en un
salto de 10 x dt).

Trampa de la regla de masa (medida): el ruido de `dT_avg/dt` por diferencias
finitas escala como 1/dt.

| dt [s] | sigma de `dT_avg/dt` medida [K/s] | Umbral `quasi_steady_dt_avg_k_s` [K/s] | Muestras en que la regla de masa es evaluable (medidas) | idem (valores verdaderos) |
|---|---|---|---|---|
| 1 | 0,105 | 0,02 | **15,0 %** | 100 % |
| 10 | 0,0104 | 0,02 | 94,9 % | 99,95 % |
| 60 | 0,0021 | 0,02 | 99,7 % | 99,7 % |
| 180 | 0,0012 | 0,02 | 99,2 % | 99,2 % |

(La sigma incluye senal ademas de ruido; a cadencias finas domina el ruido.) Con
el sigma de lazo inicial de 0,30 K el valor a 1 s seria el doble (0,21 K/s;
escalado aritmetico, no medido). **Decision**: no se filtra ni se escala el umbral
con dt. La cadencia es un parametro fijo del dataset (D1) y cambiarla obliga a
recalibrar mas de un umbral (seccion 11.4); `quasi_steady_dt_avg_k_s` queda
**calibrado a 180 s** y esta tabla cuantifica el coste de ir mas fino (a 1 s la
regla de masa se pierde casi entera, a menos que se promedie `dT_avg` sobre >= 10 s).

Validador de Fase 2 (`SensorValidator(load_sensor_spans("configs/sim_sensor_spans.yaml"))`,
alimentado con `readings_from_frame`, **sin recalibrar nada**):

| dt [s] | Alertas / 1000 lecturas | En carga constante | En rampa | `kalman_anomaly` | `noise_spike` | `sensor_drift` |
|---|---|---|---|---|---|---|
| 1 | 1085,8 | 1085,4 | 1088,8 | 52.256 | 509.723 | 71.312 |
| 10 | 510,9 | 510,1 | 516,6 | 5.070 | 19.619 | 5.121 |
| 60 | 111,1 | 110,0 | 117,3 | 853 | 51 | 179 |
| 180 | 90,3 | 89,9 | 91,8 | 295 | 0 | 0 |

Lectura:

- **El detector de tasa (`noise_spike`) escala con dt**: su umbral es una fraccion
  del sobre por segundo, y a 1-10 s el ruido de un sensor excede ese umbral en
  casi toda lectura. A 180 s no alerta (0 de 3.267): la cadencia de D1 es la
  unica de las cuatro compatible con ese umbral.
- **A 180 s todas las alertas son `kalman_anomaly` y de los 5 canales de caudal**
  (FT-HL-1/2, FT-CL-1/2, FT-FW-1: 54-62 de 121 lecturas cada uno, ~49 %). Los otros
  22 sensores dan **0** alertas a 180 s (rango, bloqueo, tasa y Kalman). Causa
  probable (por leer el codigo, no medida con un experimento de control): el
  `KalmanResidualDetector` usa `observation_noise = 1.0` fija **en unidades de
  ingenieria**, y un caudal en kg/s con sigma 45 kg/s excede esa escala. Es una
  limitacion del validador de Fase 2, que F3 no toca; queda como deuda.
- Las alertas no dependen de la rampa (89,9 en carga constante frente a 91,8 en
  rampa a 180 s): el Kalman no ve la dinamica de carga como anomalia.

### 11.4 Recalibracion de los limites (`limits:`)

Regla de F3-M2: pico del estado verdadero en rampa de 5 %/min + 5 sigma del
estadistico con el ruido definitivo.

| Clave | Valor F3-M2 | Cota con el ruido definitivo | Valor F3-M3 | Evidencia |
|---|---|---|---|---|
| `mass_balance_tolerance` | 0,055 | 0,0020 + 5 x 0,0100 = 0,052 | 0,055 (sin cambio) | sigma de caudal no cambia; peor medido 4,41 % |
| `quasi_steady_dt_avg_k_s` | 0,02 K/s | 0,0055 + 5 x 0,00058 = 0,0084 (cota inferior) | 0,02 (sin cambio) | evaluable en 99,9 % de 51.000 instantes |
| `power_flux_threshold_percent` | 4,5 pp | 2,26 + 5 x 0,40 = 4,26 | 4,5 (sin cambio) | peor medido 2,32 pp (programa de 5 %/min) |
| `pump_max_fraction` | 1,10 | - | 1,10 (sin cambio) | peor canal 4715 kg/s de 4950 |
| `power_flux_k`, `n_loops`, `conservation_tolerance`, `max_dt_c_per_s`, `energy_imbalance_fraction` | - | - | sin cambio | - |

La bajada del sigma de lazo hace **mas holgada** la cota inferior de
`quasi_steady_dt_avg_k_s` (de 0,0114 a 0,0084) pero no obliga a moverlo.

### 11.5 Determinismo y rendimiento

| Medida | Resultado |
|---|---|
| Mismo run en el mismo proceso: `DataFrame.equals` | **cierto** (40.500 filas) |
| Mismo run en el mismo proceso: bytes de parquet | **identicos** |
| Mismo run en un proceso nuevo (sha256 del parquet) | **identico** (`28d8799d...`) |
| Integracion por paso (modelo + observables + diagnostico) | 2,1 ms (1500 pasos en 3,12 s) |
| Ensamblado (ruido, cuentas ADC, frame largo) por run de 1500 pasos | 0,37 s |
| Escritura de parquet por run | 0,13 s |
| **Pasos/s extremo a extremo** (1 run, mismo proceso) | **414 pasos/s** (11.178 lecturas/s) |
| Dataset completo (34 runs, `main`) | **146 s**, 1.377.000 lecturas, **75,2 MB** (54,6 B/lectura) |

Los bytes de parquet son reproducibles con `pyarrow 24.0.0` y `pandas 2.3.3`; no
se ha verificado entre versiones de `pyarrow`. Maquina: i5-1135G7 sin otra carga;
no se detecto suspension de Windows durante las ejecuciones.

## 12. Mediciones de F3-M4: fallos de proceso

Implementacion: `data/simulation/process_faults.py`. Niveles: bloque `fault_tiers:` de
`configs/plant_parameters.yaml`. Herramienta de la matriz:
`python -m data.generators.simulation_study faults --params params.yaml --output ... --repeats 3`
(`fault_run`, `fault_matrix`, `summarize_fault_matrix`). Semilla maestra 42, cadencia
180 s, runs de 1500 pasos con el programa de carga normal (70-100 %, 5 %/min) y el
fallo superpuesto desde el paso 300 (`warmup_fraction`).

Los fallos de proceso perturban las ECUACIONES (`ModelInputs`), no la medicion:
`corrupted` es `False` y `process_fault`/`process_severity` etiquetan TODOS los
sensores de cada paso de la ventana, desde `injected_at`. La severidad sale de
`FAULT_SEVERITY` (TDD 2.2: excursion CRITICAL, caudal HIGH, vacio CRITICAL).

### 12.1 Perturbaciones

| Tipo | Entrada del modelo | Cronologia por defecto (`fault_tiers`) |
|---|---|---|
| `core_temp_excursion` | `reactivity_insertion` en rampa lineal de 30 min a `reactivity_pcm`; `rod_control_enabled = False` mientras dura la ventana | 10 pasos de rampa, 120 a plena magnitud, 20 de recuperacion |
| `coolant_flow_reduction` | `pump_demand = 1 + e(t) (A sin(2 pi t / 300 s) - (1 - f)(1 - exp(-t / 300 s)))` | sin rampa (la dinamica es la constante de 5 min), 120 y 20 pasos |
| `void_formation` | `leak_bar = e(t) x fuga(carga, margen)`, con la fuga calculada para que `T_sat(P) - T_out` valga el margen objetivo en lazo abierto | 10 pasos de rampa, 120 y 20 pasos |

`e(t)` es la envolvente (`ProcessFaultSpec.envelope`): 0 antes de `injected_step`, rampa
de entrada, 1 durante la persistencia y rampa de salida. `hold_steps = None` deja el
fallo hasta el final del run. El run acaba con operacion normal tras la ventana.

**El nivel del vacio es un margen, no una fuga en bar.** La misma fuga da
consecuencias distintas a 70 y a 100 % de carga (T_out cambia 6 K): con 42 bar el
nivel que a carga 1,0 ebullia (margen -5,6 K) a 0,7 llegaba a +4,6 K y no formaba
vacio. Por eso `fault_tiers` guarda `saturation_margin_k` y `leak_for_margin` calcula la
fuga sobre el equilibrio de la carga vigente.

### 12.2 Niveles medidos (carga constante 1,0, sin ruido, fallo persistente)

Cada nivel esta definido por su objetivo medible; la banda `expect:` del YAML se
comprueba en `tests/unit/test_process_faults.py`.

| Tipo | Nivel | Parametro | T_out pico [C] | Subida [K] | Margen min. a T_sat [K] | Caudal min. [kg/s] | n pico | n min. | Vacio pico | P min./max. [bar] |
|---|---|---|---|---|---|---|---|---|---|---|
| excursion | bajo | 100 pcm | 323,53 | 2,84 | 22,20 | 18000 | 1,073 | 1,000 | 0 | 155,0 / 156,9 |
| excursion | medio | 300 pcm | 329,15 | 8,47 | 18,43 | 18000 | 1,220 | 1,000 | 0 | 155,0 / 160,7 |
| excursion | alto | 600 pcm | 337,44 | 16,75 | 12,87 | 18000 | 1,439 | 1,000 | 0 | 155,0 / 166,3 |
| caudal | bajo | demanda 0,75 | 325,87 | 5,19 | 18,92 | 13335 | 1,001 | 0,998 | 0 | 155,0 / 155,0 |
| caudal | medio | demanda 0,60 | 331,34 | 10,65 | 13,46 | 10470 | 1,002 | 0,996 | 0 | 155,0 / 155,0 |
| caudal | alto | demanda 0,45 | 340,88 | 20,19 | 3,94 | 7604 | 1,005 | 0,993 | 0 | 154,9 / 155,1 |
| vacio | bajo | margen +1,5 K | 320,69 | 0,00 | 1,88 | 18000 | 1,015 | 0,977 | 0,003 | 116,1 / 155,0 |
| vacio | medio | margen 0,0 K | 321,63 | 0,94 | -1,65 | 18000 | 1,028 | 0,389 | 0,161 | 112,3 / 155,0 |
| vacio | alto | margen -8,0 K | 324,86 | 4,17 | -13,19 | 18000 | 3,099 | 0,168 | 0,410 | 92,9 / 155,0 |

El mismo vacio a carga 0,7 (el nivel significa lo mismo): bajo margen 1,21 K, vacio
0,020, n min. 0,596; medio margen -3,32 K, vacio 0,169, n min. 0,241; alto margen
-12,00 K, vacio 0,441, n min. 0,087.

La respuesta del vacio sigue evolucionando con la ventana (el ciclo de la seccion
12.4 se desarrolla): el pico de vacio del nivel alto pasa de 0,351 a 25 muestras a
0,377 a 40 y 0,410 a 60. La medida de nivel usa 5 muestras de calentamiento y 60 de
respuesta (`measure_level_response`).

### 12.3 Matriz de violaciones medida

Tres repeticiones por celda (`run_index` 0, 1 y 2: otra traza de carga y otro ruido),
ventana de cada run = inicio hasta el fin de la recuperacion (151 muestras en excursion
y vacio, 141 en caudal). Tasas sobre las muestras evaluables de la ventana; `dK` =
retardo en muestras de 180 s hasta la primera violacion (minimo entre repeticiones).

**Sobre valores verdaderos**

| Celda | 1 energia | 2 orden | 3 dT/dt | 4 caudal | 5 pot./flujo | 6 ebullicion | 7 masa |
|---|---|---|---|---|---|---|---|
| excursion bajo, medio, alto | 0/453 | 0/453 | 0/453 | 0/453 | 0/453 | 0/453 | 0/453 |
| caudal bajo, medio, alto | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 |
| vacio bajo | 107/453 (23,6 %) d46 | 0 | 0 | 0 | 79/453 (17,4 %) d46 | 0 | 0 |
| vacio medio | 361/453 (79,7 %) d12 | 0 | 0 | 87/453 (19,2 %) d32 | 361/453 (79,7 %) d13 | 176/453 (38,9 %) d16 | 185/360 (51,4 %) d23 |
| vacio alto | 386/453 (85,2 %) d10 | 0 | 0 | 82/453 (18,1 %) d13 | 386/453 (85,2 %) d10 | 365/453 (80,6 %) d9 | 207/318 (65,1 %) d12 |

**Sobre mediciones con el ruido del catalogo** (umbrales sin recalibrar)

| Celda | 1 energia | 2 orden | 3 dT/dt | 4 caudal | 5 pot./flujo | 6 ebullicion | 7 masa |
|---|---|---|---|---|---|---|---|
| excursion bajo, medio | 0/453 | 0/453 | 0/453 | 0/453 | 0/453 | 0/453 | 0/453 |
| excursion alto | 0/434 | 0/453 | 0/453 | 0/453 | 0/453 | 0/434 | 0/453 |
| caudal bajo, medio | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 |
| caudal alto | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 | 0/423 | 4/423 (0,9 %) d56 |
| vacio bajo | 107/453 (23,6 %) d38 | 0 | 0 | 0 | 76/453 (16,8 %) d54 | 0 | 0 |
| vacio medio | 361/453 (79,7 %) d12 | 0 | 0 | 96/453 (21,2 %) d32 | 360/453 (79,5 %) d13 | 179/453 (39,5 %) d17 | 186/360 (51,7 %) d21 |
| vacio alto | 388/453 (85,7 %) d10 | 0 | 0 | 86/453 (19,0 %) d13 | 384/453 (84,8 %) d10 | 364/453 (80,4 %) d9 | 194/316 (61,4 %) d12 |

**Fuera de la ventana**: 0 violaciones en las 9 celdas, sobre valores verdaderos y
sobre mediciones, tanto antes de `injected_at` como en la cola de operacion normal
posterior a la recuperacion (30 muestras): ninguna regla se dispara fuera de lo que el
fallo explica.

Lectura:

- **Excursion y reduccion de caudal no violan ninguna de las 7 reglas en ningun
  nivel** a 180 s. No es un defecto de umbrales: la energia se conserva (peor cierre
  por intervalo 2,4e-12) y el balance `Q_gen - Q_removed` queda siempre dentro del 5 %
  (0 de 453 muestras con desbalance instantaneo del nucleo > 5 %, el maximo medido con
  rampa de 3 min a 500 pcm fue 4,1 %). La firma "balance fuera de 5 % durante el
  transitorio" de la hipotesis inicial **no se alcanza con magnitudes validas**: los
  nodos termicos tienen poca capacidad (tau de decenas de segundos) y a 180 s todo es
  cuasi-estatico. Estos dos fallos solo se ven por el cambio de la senal (nivel alto:
  T_out +15 a +18 K en la matriz y hasta +20 K a carga 1,0, caudal -58 %, n hasta
  +38 % en la excursion), no por una regla.
- **El vacio es el unico que las reglas ven**: las reglas 1 y 5 (energia y potencia/
  flujo) se violan en el 80-85 % de la ventana a partir de 10-13 muestras (30-40 min);
  la 6 (ebullicion) solo cuando T_out cruza T_sat, a las 9 muestras en el nivel alto y
  a las 16 en el medio, y nunca en el bajo; la 7 (masa) en el 51-65 % de las muestras
  evaluables; la 4 (caudal) en el 18-19 %.
- **La regla 4 solo dispara en los canales de pierna caliente**: el caudal de
  pierna caliente es `caudal + insurge`, y el insurge del presurizador llega a
  +2964 kg/s con el vacio (FT-HL-1/2 hasta 5241 kg/s frente al tope de 4950; las
  piernas frias quedan en 4500). Es un efecto del presurizador, no de que la bomba se
  desboque.
- **La regla 3 no dispara con datos muestreados** (maxima |dT/dt| por diferencia
  finita 0,09 C/s, 110 veces bajo el limite) aunque el estado verdadero SI la viola:
  la derivada instantanea pasa de 10 C/s en 364 de 453 muestras del vacio alto (maximo
  283 C/s) y en 341 del medio. Es C5: a 180 s la regla 3 es inobservable.
- **Un falso positivo por ruido relativo** (`mass_balance`, 4/423 con caudal 0,42):
  las reglas relativas pierden margen cuando el caudal cae, porque el ruido absoluto del
  canal de caudal pesa mas sobre una `m` menor. El estado verdadero no las viola.
- **Vacio bajo = incipiente**: hay vacio (0,016-0,017) y n baja (0,68-0,75) pero T_out
  no cruza T_sat (margen 0,63 a 1,28 K). Las reglas 1 y 5 lo ven de forma intermitente
  (17-24 %) y tarde (46 muestras, 2,3 h, o mas): depende de que una rampa de carga
  empuje el sistema por encima del umbral del vacio.
- **Excursion alta**: con el programa de carga y las barras congeladas la presion
  llega al techo de 170 bar en una repeticion (66 de 453 muestras limitadas); la
  presion MEDIDA supera 170 bar en 19 muestras y deja esas filas "no evaluadas" en las
  reglas 1 y 6 (0/434).

### 12.4 Observabilidad: la oscilacion de la bomba y el ciclo del vacio

- **Oscilacion de la bomba.** Periodo 300 s, amplitud 1, 2 y 3 % de la demanda segun
  el nivel. A 180 s el periodo minimo representable es 360 s (Nyquist,
  `min_observable_period_s`): una oscilacion de 300 s se **aliasa a un periodo
  aparente de 450 s** (`aliased_period_s`, comprobado con la FFT de una sinusoide
  muestreada). Toda oscilacion mas rapida que ~6 min es inobservable como tal a esta
  cadencia: el estado verdadero la tiene y la muestra no puede distinguirla de una
  lenta. La reduccion de caudal se integra con 4 sub-pasos de 45 s por muestra (6,7
  puntos por periodo) para que el aliasing sea el real y no un artefacto de la rejilla.
  La bomba la deja pasar casi entera (tau de 10 s: ganancia teorica 0,98).
- **Ciclo del vacio.** El colapso de n y la oscilacion salen **enteramente de
  `alpha_void_per_fraction = -0,15`** (coeficiente provisional, `LITERATURE_PLACEHOLDER`
  hasta F3-M8): con el coeficiente a cero n se queda en 1,000 en los tres niveles y el
  margen del nivel medio es +0,51 K en lugar de -1,65 K. Con el coeficiente real
  del modelo, el vacio nivel alto muestra n entre 0,15 y 3,8 entre muestras
  (relajacion: el vacio apaga la potencia, T_c baja, el vacio desaparece y n
  vuelve). El signo de la realimentacion es el correcto (n baja al crecer el vacio) y el
  tamano del ciclo es dato de F3-M8, no de F3-M4.

### 12.5 Rango de validez y limitador de presion

| Fallo | Limite | Evidencia |
|---|---|---|
| excursion | `MAX_REACTIVITY_PCM = 700` | a 800 pcm con barras congeladas la presion llega a 170 bar; 700 pcm da n 1,51 y T_out 340 C |
| caudal | `MIN_PUMP_DEMAND = 0,45` | a carga 1,0, demanda 0,41: margen 0,7 K y vacio 0,016; demanda 0,40: margen -17,9 K, vacio 0,30 y n 0,17 (el acantilado); 0,45 con la oscilacion maxima: margen 3,1 K sin vacio |
| vacio | `MIN_SATURATION_MARGIN_K = -16` | a -16 K y carga 0,7 la presion llega a 61,2 bar, al borde de la tabla |

**Hallazgo: el modelo necesitaba un limitador de presion.** Un run de vacio nivel alto
con recuperacion corta saco la presion de la tabla de agua (173,9 bar) cuando el pico
de potencia empujaba T_avg justo al retirar la fuga, y la medicion de la matriz con la
cronologia por defecto (120 pasos, recuperacion de 20) abortaba con el mismo
`ValueError`. La tabla no extrapola, y un dataset no puede abortar a mitad de un run.
`thermal_model.py` acota ahora el objetivo de presion al rango de la tabla (60-170 bar):
el techo es fisico (valvulas de alivio del presurizador a ~170 bar), el suelo es un
**limite de validez** del modelo de agua liquida. No actua en operacion normal: los
digests de parquet de los runs normales son identicos a los de F3-M3
(`28d8799d...` del run 0 y `b0b53e5b...` del run 3 de 200 pasos). La matriz cuenta las
muestras de la ventana con la presion a menos de 0,05 bar de un limite
(`pressure_limited_steps`): 66 en la excursion alta y 0 en el resto.

### 12.6 Coste

Medido en la corrida de la matriz (media de 3 runs por celda), por run de 481 pasos
(471 en caudal: hasta el fin de la ventana mas 30 de cola):

| Celda | Tiempo por run |
|---|---|
| excursion (bajo / medio y alto) | ~2 s / ~9 s |
| caudal (4 sub-pasos) | ~4 s |
| vacio bajo / medio / alto | ~22 s / ~50 s / ~59 s |

La matriz completa (9 celdas x 3 repeticiones) suma 492 s de integracion y
evaluacion (~8 min). El vacio es caro por la
no suavidad de la ebullicion: hasta 26.000 evaluaciones de la derivada por paso en el
ciclo, frente a ~40 en operacion normal. Relajar la tolerancia (rtol 1e-6) lo abarata
x1,8 pero cambia n en 1,3e-3: no se hizo.

## 13. Mediciones de F3-M5: fallos de instrumento, planificador y cobertura

Implementacion: `data/simulation/instrument_faults.py` (los 5 fallos, funciones puras sobre la
serie MEDIDA tras el ruido), `data/simulation/scenario_planner.py` (presupuesto D11),
`data/simulation/coverage.py` (rejilla y particiones). Niveles y topes:
`configs/plant_parameters.yaml` (`instrument_faults:`, `scenario_planner:`). Herramientas:
`python -m data.generators.simulation_study instrument|plan ...`. Semilla maestra 42, cadencia
180 s, runs de 1500 pasos. **Los niveles y los topes son propuestas, no medidas contra datos
reales**; lo medido aqui es lo que producen.

Un fallo de instrumento no toca las ecuaciones: `true_value` es el de la planta,
`corrupted = True` marca las lecturas cuyo valor difiere de la medida sin fallo y
`quality` sigue en `good` (el instrumento no sabe que miente). La ventana etiquetada de
un episodio va de su primera a su ultima muestra afectada.

### 13.1 Niveles (`instrument_faults:`)

| Tipo | Bajo | Medio | Alto |
|---|---|---|---|
| deriva | 0,0005 anchos/h, 900 pasos | 0,1 anchos/h, 240 pasos | 13-17 anchos/h, 40 pasos |
| bloqueado | 12 pasos | 48 pasos | 240 pasos |
| pico | 5-7 sigma | 15-25 sigma | 50-70 sigma |
| sesgo | 0,10-0,20 anchos | 0,35-0,50 anchos | 1,00-1,50 anchos |
| intermitente | p_enter 0,02 / p_exit 0,50, 120 pasos | 0,05 / 0,30, 240 pasos | 0,15 / 0,15, 360 pasos |

"Ancho" es el ancho del sobre calibrado del sensor (`SensorSpan`). Con el sobre estrecho
del PWR (~9 K en un termopar) 15 anchos/h son ~6,8 K por muestra: es lo que pedia el plan
(banda 13-17 de Fase 2), pero es una deriva fisicamente muy rapida.

### 13.2 Metodologia de la matriz

`instrument_matrix`: UN run normal (indice 0, 1500 pasos) integrado y medido una vez; cada
celda es esa misma medicion con UN episodio sobre UN sensor, con el punto medio del rango de
magnitud del nivel, empezando en el paso 600 (40 %) y en los dos sentidos (+1, -1). Se
manipulan los 14 sensores que algun rol de las constraints lee mas un sensor de control
fuera de los roles (`TC-CORE-01`). Cada celda se evalua dos veces:

- **Votada**: con la mediana del rol, como el Rule Engine de F3-M2. Con 2 canales (T_in,
  T_out, caudales por pierna, presion) la mediana es la media: un sesgo `b` en un canal llega
  como `b/2`. Con 3 canales (flujo neutronico) la mediana lo enmascara por completo.
- **Por canal**: con el sensor aislado como unico canal de sus roles
  (`single_channel_roles`), que es lo que veria una regla sin votacion.

Las tasas son violaciones sobre muestras evaluables de la ventana del episodio, **agregadas
solo sobre los sensores que violan la regla** ("k/14 sens."); `dK` es el retardo minimo en
muestras de 180 s. `0*` = cero violaciones en los 14 sensores y los dos sentidos. Fuera de los
roles (sensor de control y sensores sin rol) hubo 0 violaciones de las 7 reglas en las 15
celdas, votadas y por canal: manipular un sensor que ninguna regla lee no la viola.

### 13.3 Matriz por canal (sensor aislado)

La matriz votada esta en la seccion 9. Esta es la misma medicion sin votacion:

| Celda | 1 energia | 2 orden | 3 dT/dt | 4 caudal | 5 pot./flujo | 6 ebullicion | 7 masa |
|---|---|---|---|---|---|---|---|
| deriva bajo | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| deriva medio | 11/14 sens.: 4098/5280 (77,6 %) d18 | 0* | 0* | 4/14 sens.: 181/1920 (9,4 %) d141 | 4/14 sens.: 1755/1920 (91,4 %) d16 | 0* | 4/14 sens.: 1058/1920 (55,1 %) d42 |
| deriva alto | 13/14 sens.: 662/709 (93,4 %) d0 | 4/14 sens.: 144/320 (45,0 %) d3 | 0* | 4/14 sens.: 265/320 (82,8 %) d1 | 4/14 sens.: 320/320 (100,0 %) d0 | 4/14 sens.: 100/213 (46,9 %) d3 | 4/14 sens.: 259/261 (99,2 %) d0 |
| bloqueado bajo | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| bloqueado medio | 0* | 0* | 0* | 0* | 0* | 0* | 0* |
| bloqueado alto | 7/14 sens.: 1482/3360 (44,1 %) d59 | 0* | 0* | 0* | 4/14 sens.: 1408/1920 (73,3 %) d60 | 0* | 0* |
| pico bajo | 4/14 sens.: 7/8 (87,5 %) d0 | 0* | 0* | 0* | 0* | 0* | 4/14 sens.: 5/8 (62,5 %) d0 |
| pico medio | 11/14 sens.: 21/22 (95,5 %) d0 | 0* | 0* | 4/14 sens.: 4/8 (50,0 %) d0 | 4/14 sens.: 8/8 (100,0 %) d0 | 0* | 4/14 sens.: 8/8 (100,0 %) d0 |
| pico alto | 11/14 sens.: 22/22 (100,0 %) d0 | 0* | 0* | 4/14 sens.: 4/8 (50,0 %) d0 | 4/14 sens.: 8/8 (100,0 %) d0 | 0* | 4/14 sens.: 8/8 (100,0 %) d0 |
| sesgo bajo | 11/14 sens.: 8853/19800 (44,7 %) d0 | 0* | 0* | 0* | 4/14 sens.: 7196/7200 (99,9 %) d0 | 0* | 2/14 sens.: 3/3600 (0,1 %) d276 |
| sesgo medio | 11/14 sens.: 14871/19800 (75,1 %) d0 | 0* | 0* | 0* | 4/14 sens.: 7200/7200 (100,0 %) d0 | 0* | 4/14 sens.: 1399/7200 (19,4 %) d0 |
| sesgo alto | 11/14 sens.: 19800/19800 (100,0 %) d0 | 0* | 0* | 4/14 sens.: 3558/7200 (49,4 %) d0 | 4/14 sens.: 7200/7200 (100,0 %) d0 | 0* | 4/14 sens.: 7200/7200 (100,0 %) d0 |
| intermitente bajo | 11/14 sens.: 141/2640 (5,3 %) d0 | 0* | 0* | 4/14 sens.: 14/960 (1,5 %) d0 | 4/14 sens.: 54/960 (5,6 %) d0 | 0* | 4/14 sens.: 25/960 (2,6 %) d0 |
| intermitente medio | 11/14 sens.: 815/5280 (15,4 %) d0 | 0* | 0* | 4/14 sens.: 184/1920 (9,6 %) d0 | 4/14 sens.: 299/1920 (15,6 %) d0 | 0* | 4/14 sens.: 313/1920 (16,3 %) d0 |
| intermitente alto | 11/14 sens.: 3959/7920 (50,0 %) d0 | 0* | 0* | 4/14 sens.: 653/2880 (22,7 %) d0 | 4/14 sens.: 1398/2880 (48,5 %) d0 | 0* | 4/14 sens.: 1442/2880 (50,1 %) d0 |

### 13.4 Que sensores violan que regla (V = votada, C = por canal)

Grupos: `Q_cl`/`Q_hl` caudal pierna fria/caliente, `T_in`/`T_out` termopares, `flux`
(3 canales neutronicos), `P_th` potencia termica, `P` presion del presurizador.

- deriva bajo: V[-] | C[-]
- deriva medio: V[energy_balance: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]
- deriva alto: V[energy_balance: P,T_in,T_out; temperature_ordering: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; boiling: P,T_out; mass_balance: Q_cl,Q_hl] | C[energy_balance: P,Q_cl,Q_hl,T_in,T_out,flux; temperature_ordering: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; boiling: P,T_out; mass_balance: Q_cl,Q_hl]
- bloqueado bajo: V[-] | C[-]
- bloqueado medio: V[-] | C[-]
- bloqueado alto: V[energy_balance: T_in,T_out; power_flux: P_th] | C[energy_balance: T_in,T_out,flux; power_flux: P_th,flux]
- pico bajo: V[-] | C[energy_balance: Q_cl,Q_hl; mass_balance: Q_cl,Q_hl]
- pico medio: V[energy_balance: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]
- pico alto: V[energy_balance: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]
- sesgo bajo: V[energy_balance: T_out; power_flux: P_th] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; power_flux: P_th,flux; mass_balance: Q_cl]
- sesgo medio: V[energy_balance: T_in,T_out; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]
- sesgo alto: V[energy_balance: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]
- intermitente bajo: V[energy_balance: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]
- intermitente medio: V[energy_balance: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]
- intermitente alto: V[energy_balance: T_in,T_out; flow_range: Q_cl,Q_hl; power_flux: P_th; mass_balance: Q_cl,Q_hl] | C[energy_balance: Q_cl,Q_hl,T_in,T_out,flux; flow_range: Q_cl,Q_hl; power_flux: P_th,flux; mass_balance: Q_cl,Q_hl]

### 13.5 Lectura de la matriz

- **Las reglas 3 (dT/dt) y 6 (ebullicion) casi nunca disparan**: la 3 no dispara en ninguna
  de las 15 celdas (C5: a 180 s es inobservable) y la 6 solo con la deriva alta (que saca
  la presion y T_out del sobre). La regla 2 (orden) solo con la deriva alta.
- **Votada vs por canal**: la votacion oculta por completo los 3 canales de flujo
  neutronico (por canal violan las reglas 1 y 5 en todas las celdas visibles salvo el pico bajo) y diluye a
  la mitad los termopares. El **sesgo bajo** (0,10-0,20 anchos) apenas se ve votado en los
  termopares (59/3600 en la regla 1, solo en T_out); donde se ve es en la potencia termica,
  que tiene un unico canal y no se vota (regla 5: 1799/1800). Por canal se ve en la regla 1
  de 11 de los 14 sensores.
  Es una decision de diseno del Rule Engine de Fase 4 (T7.3): votar esconde los fallos de un
  canal; no votar los expone, pero tambien expone cada regla al ruido de un solo canal (no
  medido aqui).
- **Invisibles a las 7 reglas, votadas y por canal**: la deriva baja (0,0005 anchos/h) y
  el bloqueado bajo y medio (12 y 48 pasos). Son las celdas que deben aparecer como falsos
  negativos esperados del Rule Engine.
- **Pico**: una sola muestra; el pico bajo (5-7 sigma) solo lo ven por canal los
  caudalimetros (reglas 1 y 7). Votado, el medio y el alto violan las reglas 1, 4, 5 y 7
  en el instante del pico.
- **Intermitente**: viola las reglas 1, 4, 5 y 7 (donde su valor invalido cae en el
  carril `span.min`/`span.max`) con tasas que suben con la fraccion invalida (1,5 a 50 %
  de las muestras evaluables de los sensores que la violan, votada); es lo que las hipotesis "?" de la seccion 9 no sabian.
- **Hipotesis frente a medida**: confirmadas la regla 3 (H0) en el pico, la regla 4 (H+)
  en el pico sobre un canal de caudal y la regla 1 (H+) en el sesgo. La deriva no es H0
  en todos los niveles: el nivel medio (0,1 anchos/h durante 240 pasos = 1,2 anchos de
  recorrido) viola la regla 1 en el 70 % de la ventana de 4 de los 14 sensores.

### 13.6 Deteccion del validador de Fase 2 (informativa)

`instrument_detection`: cada episodio inyectado en un run entero (2 repeticiones, 6 grupos
de sensor = 12 episodios por celda; 120 para el pico) y pasado por `SensorValidator` tal
como esta, **sin recalibrar nada**. "Excl. caudal" descarta los 2 episodios de caudal por
repeticion: el Kalman de Fase 2 (`observation_noise = 1,0`) alerta en ~48 % de las
lecturas normales de esos canales (480 por 1000), una deuda conocida que no se aborda aqui.
Falsos positivos fuera de caudal y de las ventanas inyectadas: de 0,0 a 1,65 por 1000 lecturas.

| Celda | Episodios detectados (excl. caudal) | Lecturas marcadas / etiquetadas | Retardo (muestras, los 12 episodios) | Detectores que disparan (lecturas) |
|---|---|---|---|---|
| deriva bajo | 0/10 | 875/10800 (solo Kalman de caudal) | - | - |
| deriva medio | 10/10 | 2070/2880 | mediana 62,5, max. 74 | fuera de rango 1971, Kalman 235 |
| deriva alto | 10/10 | 480/480 | 0 | deriva 452, fuera de rango 479, pico 16, Kalman 44 |
| bloqueado bajo | 10/10 | 74/144 | mediana 6 | bloqueo 72 |
| bloqueado medio | 10/10 | 506/576 | mediana 6 | bloqueo 504 |
| bloqueado alto | 10/10 | 2810/2880 | mediana 6 | bloqueo 2808 |
| pico bajo | 3/100 | 23/120 | 0 | fuera de rango 21, Kalman 13, pico 1, deriva 6 |
| pico medio | 68/100 | 88/120 | 0 | fuera de rango 88, pico 50, deriva 46, Kalman 13 |
| pico alto | 100/100 | 120/120 | 0 | fuera de rango 110, Kalman 113, pico 100, deriva 87 |
| sesgo bajo | 0/10 | 875/10800 (solo Kalman de caudal) | - | - |
| sesgo medio | 10/10 | 4225/10800 | mediana 0,5, max. 5 | fuera de rango 3541, Kalman 880 |
| sesgo alto | 10/10 | 10800/10800 | 0 | fuera de rango 10800, Kalman 883 |
| intermitente bajo | 10/10 | 259/1440 | 0 | fuera de rango 94, Kalman 197, pico 82, deriva 67, bloqueo 2 |
| intermitente medio | 10/10 | 782/2880 | 0 | fuera de rango 418, Kalman 477, pico 270, deriva 209, bloqueo 45 |
| intermitente alto | 10/10 | 2724/4320 | 0 | fuera de rango 2166, Kalman 854, pico 678, deriva 513, bloqueo 636 |

Falsos negativos esperados confirmados: **deriva baja y sesgo bajo no los ve ningun
detector** (0/10 fuera de caudal; las 875 lecturas marcadas son los falsos positivos del
Kalman de caudal) y el pico bajo se ve en 3/100 episodios. El nivel medio de la deriva no es
sub-detectable en su ventana de 240 pasos: acumula 1,2 anchos y sale del rango (retardo
mediana 62,5 muestras); solo el nivel bajo (0,02 anchos en 900 pasos) es invisible. Las
etiquetas de nivel del plan hablan del ritmo, no de la visibilidad final.

### 13.7 Presupuesto y capacidad (D11)

Con `params.yaml` (34 runs x 1500 pasos x 27 sensores = 1.377.000 lecturas,
`fault_injection_rate` 0,40 y pesos 0,30/0,30/0,25/0,15) el objetivo es 550.800 lecturas
etiquetadas. **El planificador falla con `ValueError`** (no fuerza densidades): necesita 397
runs frente a 32 disponibles. Cuotas y capacidad por run bajo los topes:

| Severidad | Cuota [lecturas] | Tipos | Capacidad por run | Runs necesarios |
|---|---|---|---|---|
| CRITICAL | 165.240 | excursion, vacio, bloqueado | 49.500 | 12 |
| HIGH | 165.240 | caudal, sesgo, deriva | 36.277 | 15 |
| MEDIUM | 137.700 | intermitente | 2.160 | 64 |
| LOW | 82.620 | pico | 270 | 306 |

Capacidad maxima por run y tipo bajo los topes (`max_window_fraction` 0,60,
`max_sensor_fraction` 0,34, `max_spike_rate` 0,02): excursion, caudal y vacio 24.300 (el 60 %
del run en 27 sensores); sesgo 8.437; deriva 3.540; intermitente 2.160; bloqueado 900;
pico 270.

**El limite es estructural**: MEDIUM y LOW dependen de un unico tipo de instrumento cada
uno (intermitente y pico) y un episodio de instrumento etiqueta un solo sensor. Tasa maxima
alcanzable por reparto de severidad (biseccion sobre la misma regla del planificador):

| Pesos (crit./alto/medio/bajo) | Tasa maxima | CRITICAL sobre todas las lecturas |
|---|---|---|
| 0,30 / 0,30 / 0,25 / 0,15 (D11) | 1,44 % | 0,43 % |
| 0,50 / 0,38 / 0,10 / 0,02 | 7,84 % | 3,92 % |
| 0,55 / 0,38 / 0,06 / 0,01 | 15,69 % | 8,63 % |
| 0,60 / 0,37 / 0,025 / 0,005 | 21,36 % | 12,82 % |

Ninguna alcanza el 40 %, y solo las dos ultimas dan el 5 % de CRITICAL (`critical_min_fraction`).
Cota por aritmetica (no medida con el planificador): con 2 runs normales y 3 runs por cada
uno de los 5 tipos de instrumento, quedan 17 runs de proceso x 24.300 = 413.100 lecturas
(30,0 %) mas ~45.900 de instrumento (3,3 %): **el 40 % no cabe con estos topes con ningun
reparto de severidades**.

**Relajar los topes con los pesos de D11 tampoco llega** (tasa maxima, CRITICAL):

| `max_sensor_fraction` | `max_spike_rate` | Tasa maxima | CRITICAL |
|---|---|---|---|
| 0,34 | 0,02 | 1,44 % | 0,43 % |
| 0,34 | 0,10 | 4,39 % | 1,32 % |
| 0,50 | 0,02 | 2,08 % | 0,62 % |
| 0,50 | 0,10 | 6,34 % | 1,90 % |
| 1,00 | 0,02 | 4,31 % | 1,29 % |
| 1,00 | 0,10 | 13,18 % | 3,95 % |

`max_window_fraction` (0,60, 0,80 y 0,90) **no cambia ninguna de estas cifras** con los
pesos de D11: el limite lo ponen el pico y el intermitente, no la ventana de proceso.
Con los pesos 0,55/0,38/0,06/0,01 subir los topes si ayuda: 22,66 % con ventana 0,80 y sensores
0,50 (CRITICAL 12,46 %) y 39,18 % con ventana 0,90, todos los sensores y picos 0,10
(CRITICAL 21,55 %), a costa de runs con casi todo el catalogo y la mayor parte del run etiquetados.

### 13.8 Dataset planificado medido (tasa 0,12, pesos 0,55/0,38/0,06/0,01)

Generado entero (34 runs, 1.377.000 lecturas) con `ReactorSimulator.simulate` +
`build_run_frame`, sin tocar `params.yaml`:

| Magnitud | Plan | Frame |
|---|---|---|
| Lecturas etiquetadas | 168.958 (12,27 %) | 168.958 (12,27 %), **34/34 runs coinciden** |
| CRITICAL / HIGH / MEDIUM / LOW | 91.062 / 66.283 / 9.960 / 1.653 | idem: 6,61 / 4,81 / 0,72 / 0,12 % |
| Runs por tipo | sesgo 3, caudal 3, excursion 3, deriva 3, intermitente 5, pico 7, bloqueado 3, vacio 3, normal 4 | (los del plan) |
| Cobertura de tipos | 8/8 (tambien en train, val y test) | 8/8 |
| Rejilla (k_min 10, n_min 1) | 144/144 en el dataset; train 75,0 %, val 38,2 %, test 40,3 % | los mismos conjuntos de modos, lecturas y episodios (plan == frame en las 4 particiones) |

- `quality = good` en todas las lecturas de todos los runs; `true_value` preservado.
- `corrupted`: 0 en proceso (el fallo perturba las ecuaciones); en instrumento coincide con la
  etiqueta salvo en el **intermitente: 3.294 de 9.960 lecturas etiquetadas (33,1 %)** estan
  realmente corruptas, el resto son lecturas validas dentro de la ventana del episodio.
- Residuo de conservacion por intervalo: peor 3,9e-12 en los 34 runs (0 fallos de
  tolerancia); sin pasos pegados al limitador de presion (suelo 60, techo 170 bar) con las
  ventanas planificadas; presion de 87,6 a 167,2 bar.
- Tamano: 75,6 MB en 34 parquets (2,2 MB por run); el dataset normal medido en F3-M3 pesaba
  75,2 MB.
- Reparto por severidad fuera de objetivo: el planificador reparte cada severidad entre
  sus tipos proporcionalmente a su capacidad por run (M5-c) y las cuotas se redondean a
  episodios/ventanas enteras: sesgo, deriva y bloqueado salen un 9 a 17 % por encima de su cuota
  (17.137 frente a 14.603; 7.080 frente a 6.127; 1.800 frente a 1.652).

### 13.9 Barrido de (k_min_readings, n_min_episodes) (D6)

Sobre el plan anterior, sin simular. Fraccion de la rejilla (144 modos = 8 tipos x 6 grupos
de sensor x 3 niveles) cubierta:

| n_min | k_min | Dataset | Train | Val | Test |
|---|---|---|---|---|---|
| 1 | 1 | 100,0 % | 75,0 % | 39,6 % | 40,3 % |
| 1 | 10 | 100,0 % | 75,0 % | 38,2 % | 40,3 % |
| 1 | 20 | 95,8 % | 72,2 % | 28,5 % | 29,2 % |
| 1 | 50 | 87,5 % | 64,6 % | 26,4 % | 25,7 % |
| 1 | 100 | 75,0 % | 52,8 % | 26,4 % | 25,0 % |
| 1 | 500 | 59,0 % | 36,1 % | 12,5 % | 11,1 % |
| 2 | 10 | 25,0 % | 18,1 % | 10,4 % | 13,2 % |
| 3 | 10 | 16,7 % | 12,5 % | 10,4 % | 12,5 % |

El modo con menos lecturas es el bloqueado bajo (12 lecturas por modo: 12 pasos en 1 sensor
x 1 episodio), justo sobre la ventana de 8 muestras del detector; el siguiente es la deriva
alta (40). Cualquier `k_min_readings` <= 12 deja la rejilla al 100 %; se mantiene 10
(margen sobre los 8 de la ventana). `n_min_episodes = 1` es lo que el presupuesto sostiene: con
2 episodios por modo la rejilla cae al 25 %, porque cada modo tiene un unico episodio en los
32 runs. **La puerta D6 (8/8 tipos y 95 % de la rejilla) se cumple en el dataset entero,
pero val y test ven solo el 38-40 % de la rejilla**: la particion 70/15/15 es por tipo
sobre la linea temporal de sus runs concatenados (`partition_of`), y con 3 runs por tipo y un
nivel por run cada nivel cae casi entero en una particion, aunque los 8 tipos estan en las tres.

### 13.10 Coste y validez de las ventanas largas

Tiempos de pared con `simulate` + `build_run_frame` + parquet, en una maquina con
suspensiones de Windows (Kernel-Power 506/507): los runs 8 (excursion) y 12 (caudal) quedaron
**contaminados por suspensiones y no se usan**.

| Tipo | Ventana [pasos] | Tiempo por run |
|---|---|---|
| normal, instrumento | - | 9 a 19 s |
| caudal | 519-520 | 35 a 41 s |
| excursion | 551 | 10 y 49 s |
| vacio | 551 | 838 s (medio), 457 s (bajo), 834 s (alto) |

El vacio con la ventana planificada cuesta de 14 a 21 veces lo que M4 midio con la ventana
por defecto (22/50/59 s con 151 pasos) y el coste por paso tambien sube (1,5 s/paso frente a
0,33): el ciclo del vacio se desarrolla con la ventana. Los 3 runs de vacio suman 2.129 s de
los 2.560 s medidos sin los dos runs contaminados (83 %); con ellos el dataset completo ronda
los 43 min de pared, no los 146 s del normal de F3-M3. El sobrecoste de M5 es ~1 s por run
(el mismo comando `digest` del run 0 tarda 22,5 s en el commit de F3-M4 y 23,5 s en el de
F3-M5, arranque incluido). Los 9-14 s por run de esta medicion no son comparables con los
4,3 s medios de M3: aqui se escribe parquet, se calculan etiquetas y la maquina estaba mas
cargada.

Validez de la ventana maxima (900 pasos = `max_window_fraction` 0,60), un run por celda, mismo
procedimiento:

| Celda | Ventana [pasos] | P min./max. [bar] | Pasos en el limitador | Residuo max. | Fallos | Tiempo |
|---|---|---|---|---|---|---|
| core_temp_excursion high | 900 | 153,4 / 170,0 | 250 | 2,95e-12 | 0 | 110 s |
| coolant_flow_reduction high | 900 | 153,1 / 155,4 | 0 | 2,92e-15 | 0 | 28 s |
| core_temp_excursion low | 900 | 153,4 / 160,2 | 0 | 1,64e-15 | 0 | 7 s |
| coolant_flow_reduction low | 900 | 153,1 / 155,4 | 0 | 3,00e-15 | 0 | 41 s |
| void_formation low | 900 | 108,5 / 154,9 | 0 | 1,73e-14 | 0 | 409 s |
| void_formation high | 900 | 91,4 / 154,9 | 0 | 2,39e-14 | 0 | 853 s |

Las 6 celdas medidas quedan dentro de la tabla de agua (60-170 bar) y conservan la energia
(residuo maximo 2,9e-12, 0 fallos), pero **la excursion alta con 900 pasos opera 250 pasos
contra el techo de 170 bar** del limitador de presion (F3-M4, M4-3): el estado esta saturado
y la respuesta deja de ser la del fallo. Con las ventanas planificadas (551) ningun run toca el
limitador. La excursion media a 900 pasos no se midio. El coste del vacio no crece con la
ventana: 853 s a 900 pasos frente a 834 s a 551 (alto) y 409 s frente a 457 s (bajo); sin
suspensiones de Windows en el intervalo de esta medicion.

El rango de validez de M4 se midio con 60-120 pasos; las ventanas planificadas llegan a 551.

### 13.11 Digests (M5-h)

`injected_at_step` y `fault_level` son columnas nuevas del frame largo (nulas en los runs
normales): las 23 columnas anteriores son identicas elemento a elemento (run 0 y run 3 con
200 pasos, comparados contra el commit de F3-M4 en un worktree aparte), pero los bytes del
parquet cambian.

| Run | Digest F3-M4 | Digest F3-M5 |
|---|---|---|
| 0 (1500 pasos) | `28d8799d90c9e1d77d9daffbda1e943836c7436bcbc2fa1d30e8d973a23934e0` | `d60169ad0a229ec86a7d4a13bad0a4398fd29fd150d47cf7acfbda344554be8f` |
| 0 (200 pasos) | `d551c4cad15c01040bc784e32168be15b0fc3c58f5c72441674abb840fcf37cd` | `93dd7bccef83859c3272d5c7242f7b59499baddfd62407dbd68c66edac7da5a2` |
| 3 (200 pasos) | `b0b53e5b01c91ef962f26ff6173253a1464024cfcce5c1bb5df81000a9f90b6e` | `d0a222158483feb057e6bba824fd2ee61386cc333216519f66b8f671a7f02693` |
