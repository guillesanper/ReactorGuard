# ReactorGuard - Fase 2: streaming de lecturas

*Referencia tecnica del camino Kafka de la Fase 2 (T3.3, T3.5, T4.3, T4.5, T4.6). Estado verificado contra el
codigo tras el hito M9 (2026-10-11).*

Este documento describe lo que el codigo hace y por que, y lo separa de lo que esta **medido**. Cada cifra
indica donde se obtuvo. Nada de lo que aqui se describe se ha desplegado en GCP: los criterios 1 y 3 de la
Fase 2 solo se cierran con una medicion en el cluster (ver [seccion 9](#9-que-no-cierra-esta-fase)).

Documentos relacionados: [Progreso](ReactorGuard_Progreso.md) (estado por tarea y criterios),
[Handoff_Paso4](Handoff_Paso4.md) (guia de despliegue en GCP).

---

## 1. Flujo y topologia

```
 data/processed/tep/fault_type=NN/readings.parquet            (22 runs, 550.160 lecturas)
        |  ParquetDirSource | StorageSource (GCS)
        v
 tep_streamer  (1 replica)        key = sensor_id, header run-id = fault_type=NN/loop=K
        |
        v   sensor-readings-raw   12 particiones, RF 3, min.insync 2, retencion 7 d
 sensor-validator (2 replicas, hasta 12)    grupo de consumo: sensor-validator
        |   SensorValidator: 5 detectores con estado por sensor
        |   flush del productor  ->  commit sincrono del offset (at-least-once)
        +--> sensor-validated     12 particiones, RF 3, retencion 7 d     (TODAS las lecturas, con quality)
        +--> anomaly-alerts        3 particiones, RF 3, retencion 30 d    (una AlertEvent por fallo)
```

| Topic | Particiones | Escribe | Lee | Valor |
|---|---:|---|---|---|
| `sensor-readings-raw` | 12 | `reactorguard-ingestion` | `reactorguard-validator` (grupo `sensor-validator`) | `SensorReading` JSON |
| `sensor-validated` | 12 | `reactorguard-validator` | `reactorguard-detector` (grupo `anomaly-detector`, Fase 4) | `SensorReading` JSON con `quality` enriquecida |
| `anomaly-alerts` | 3 | `reactorguard-validator` (y `reactorguard-detector`, Fase 4) | nadie todavia (Alertmanager y dashboard, Fase 8) | `AlertEvent` JSON |
| `bench-throughput` | 12 | `reactorguard-benchmark` | `reactorguard-benchmark` | mensajes del benchmark (1 h de retencion) |

Los KafkaUser y sus ACL estan en `k8s/base/kafka/kafka-users.yaml`; un test (`test_k8s_manifests.py`) exige
que cada topic y grupo que usa `params.yaml: streaming` este cubierto por el usuario correcto.

Entornos:

| Entorno | Broker | Seguridad | Quien lo levanta |
|---|---|---|---|
| Local | un nodo KRaft en `127.0.0.1:9092`, Redis en `127.0.0.1:6380` | PLAINTEXT | `infra/local/Invoke-LocalStack.ps1 -Up` |
| Cluster | Strimzi, 3 brokers | mTLS (puerto 9093), un KafkaUser por servicio | `kubectl apply -k k8s/base/` |

El perfil `observability` del stack local anade Prometheus (`127.0.0.1:9091`) y Grafana
(`127.0.0.1:3000`) con el dashboard `data-pipeline` ya aprovisionado.

---

## 2. Invariantes (D1 a D14)

Son las decisiones de diseno del plan de Fase 2 que, si se rompen, no producen un error sino resultados
silenciosamente incorrectos. Cada una indica donde se garantiza y que test la protege.

| # | Invariante | Donde se garantiza | Que la protege |
|---|---|---|---|
| D1 | **La clave de particion es `sensor_id`** (bytes UTF-8) con el particionador por defecto (murmur2). Los detectores acumulan estado por sensor: sin clave, el stuck de `TEP-XMV-04` deja de detectarse sin error ni log. | `TEPStreamer` (clave en cada `send`), `ValidationConsumer._publish` (conserva la clave) | `test_stream_invariants.py`: XMV-04/d21 con 1, 2 y 3 consumidores da **exactamente** los 473 pares de la validacion offline; el test espejo con clave round-robin falla la deteccion |
| D2 | **Cabecera `run-id`** (`fault_type=NN/loop=K`) en cada mensaje. El consumidor reinicia el estado de un sensor cuando cambia. Red de seguridad: si el timestamp retrocede, tambien. | `transport.HEADER_RUN_ID`, `tep_streamer.format_run_id`, `ValidationConsumer._check_continuity` | `test_tep_streamer.py`, `test_validation_consumer.py`; metrica `detector_state_resets_total{reason}` |
| D3 | **Se publican todas las lecturas** en `sensor-validated`, con `quality` enriquecida (`SUSPECT`/`BAD`). Los consumidores posteriores filtran por `is_usable`. Descartar lo sospechoso ocultaria evidencia. | `ValidationConsumer._publish` | `test_validation_consumer.py` (rango critico sale `BAD` y se publica igualmente) |
| D4 | **At-least-once**: procesar lote -> `producer.flush()` verificando cada resultado -> `consumer.commit()` sincrono. Nunca exactly-once; los duplicados se eliminan aguas abajo con `reading_id`/`alert_id` (uuid5 deterministas). | `ValidationConsumer.checkpoint` | `test_validation_consumer.py`: la secuencia `send* < flush < commit` y que un `flush` que falla no produce commit |
| D5 | **Mensaje venenoso** (no deserializable): se registra (topic, particion, offset, hash), se cuenta y se **salta con commit**. No hay DLQ en esta fase. | `ValidationConsumer._deserialize` | `test_validation_consumer.py`; `reactorguard_poison_messages_total` |
| D6 | **Un rebalanceo pierde el estado** de los sensores de las particiones perdidas; se descarta explicitamente. No se persiste estado. | `ValidationConsumer.on_partitions_revoked/lost/_release` | `test_validation_consumer.py`; `consumer_rebalances_total`, `detector_state_resets_total{reason="rebalance"}` |
| D7 | **Grupo de consumo `sensor-validator`** en codigo, ACL y dashboard. | `params.yaml: streaming.consumer_group`, `kafka-users.yaml` | `test_k8s_manifests.py` (cobertura de ACL) |
| D8 | **mTLS en el cluster, PLAINTEXT en local.** La seguridad se inyecta por `KafkaConnectionSettings.from_env()`; los tunables viven en `params.yaml`; endpoints y credenciales solo en variables de entorno. | `data/streaming/kafka_settings.py` | `test_kafka_settings.py` (SSL sin ficheros falla; las credenciales no salen en `repr`) |
| D9 | **Un `CollectorRegistry` por instancia** (no el global) y un unico modulo con todos los nombres. El lag lo calcula el propio consumidor. Etiquetas de baja cardinalidad (nunca `sensor_id`). | `data/streaming/metrics.py` | `test_stream_metrics.py`, `test_grafana_dashboard.py` |
| D10 | **Puertos**: `/health` y `/ready` en 8000, `/metrics` en 9090, en ambos servicios. | `data/streaming/observability.py`, `params.yaml: streaming` | `test_service_http_server.py`, `test_k8s_workloads.py` |
| D11 | **Puertos y adaptadores**: `Protocol` para productor, consumidor y almacen de blobs; adaptadores finos sobre `kafka-python` y `google-cloud-storage`; fakes en memoria para los tests. | `data/streaming/transport.py`, `kafka_adapters.py`, `data/storage/blob_store.py` | todos los tests de servicio corren sin broker ni red |
| D12 | **Parametros** en secciones `streaming`, `storage`, `feature_store` y `streamer` de `params.yaml`, cada una con su loader `*_params.py` (dataclass `frozen`, `_require`). | `*_params.py` | `test_streaming_params.py`, `test_storage_params.py`, `test_feature_store_params.py`, `test_tep_streamer_params.py` |
| D13 | **`AlertEvent` es un esquema propio**; `SensorReading` no se modifica. | `data/validation/alert_event.py` | `test_alert_event.py` |
| D14 | **Metricas `reactorguard_*`**; `sensor_faults_total{fault_type}` coincide con la etiqueta que prevee la Fase 8 del TDD. | `data/streaming/metrics.py` | `test_stream_metrics.py` |

Dos matices respecto al plan original, decididos durante la implementacion:

- **El productor usa idempotencia con `max_in_flight_requests=5`**, no `max_in_flight=1` como decia el plan.
  Con idempotencia, kafka-python conserva el orden por particion con hasta 5 peticiones en vuelo
  (`data/streaming/kafka_adapters.py`, cabecera del modulo). Es la configuracion medida en el benchmark local.
- **El listener de rebalanceo no hace flush ni commit.** En `kafka-python 3.x` los callbacks corren en el hilo
  de IO de los latidos y `commit()` desde ahi lanza `RuntimeError`. No hace falta: el unico punto de entrada
  al rejoin es `poll()` y cada lote se confirma por completo antes del siguiente `poll()`, de modo que al
  revocar nunca queda trabajo sin confirmar (`validation_consumer.py`, cabecera del modulo).

---

## 3. Contrato de mensajes

### 3.1 Clave, valor y cabeceras

| Campo | Valor |
|---|---|
| Clave | `sensor_id` en UTF-8, p. ej. `TEP-XMV-04`. En `sensor-validated` y `anomaly-alerts` se conserva |
| Valor (`raw`, `validated`) | `SensorReading.to_kafka_bytes()`: JSON UTF-8 del contrato canonico (`data/schemas/sensor_reading.py`) |
| Valor (`alerts`) | `AlertEvent.to_kafka_bytes()`: JSON UTF-8 |
| Cabecera `run-id` | ASCII, `fault_type=NN/loop=K`. `NN` es el fichero del TEP (00 a 21) y `K` el numero de vuelta cuando `streamer.loop` es `true`. El consumidor la copia a `sensor-validated` y `anomaly-alerts` |

Un mensaje sin cabecera es valido: el consumidor cae en la red de seguridad del retroceso de timestamp.

Una lectura real de `TEP-XMV-04` en d21 (el unico transmisor congelado de verdad del dataset):

```json
{
  "reading_id": "c7b4a278-7954-5894-b5da-59a73180a13a",
  "timestamp": "2000-01-01T00:00:00Z",
  "plant_id": "TEP-PLANT-01",
  "sensor": {"id": "TEP-XMV-04", "type": "position", "location": "core", "elevation_m": 78.0},
  "measurement": {"value": 61.302, "unit": "percent", "quality": "good", "raw_counts": 32218},
  "metadata": {"calibration_date": "2023-06-01", "last_maintenance": "2023-12-01", "drift_coefficient": 0.0001},
  "is_usable": true
}
```

`reading_id` es un `uuid5` de `plant_id|fault_type|sensor_id|timestamp`: determinista y unico en las 550.160
lecturas. El `fault_type` forma parte de la clave porque los 22 ficheros reinician el reloj en
`2000-01-01`.

### 3.2 `quality` e `is_usable`

El validador **escribe** `quality`, nunca la lee. `quality` describe la fiabilidad del INSTRUMENTO, no el
estado del proceso: el adaptador emite `good` para los 22 ficheros aunque 21 describan perturbaciones del
proceso.

| `quality` a la salida | Cuando |
|---|---|
| `good` | ningun detector disparo |
| `suspect` | hay fallos, ninguno `CRITICAL` |
| `bad` | algun fallo `CRITICAL` |
| `missing` | no llega senal (no lo emite el validador) |

`is_usable` es `false` para `bad` y `missing`. Una lectura `suspect` es utilizable con cautela. Los
consumidores posteriores **deben** filtrar por `is_usable`: la lectura mala se publica igualmente (D3).

### 3.3 `AlertEvent`

Un evento por cada `SensorFault` de cada lectura. Ejemplo (el bloque `source` es ilustrativo):

```json
{
  "alert_id": "8a9397ad-9781-53b6-afde-be3437951ea0",
  "reading_id": "30417012-5f76-59e6-b6e2-687a7f5f4fce",
  "sensor_id": "TEP-XMV-04",
  "plant_id": "TEP-PLANT-01",
  "run_id": "fault_type=21/loop=0",
  "fault_type": "stuck",
  "severity": "MEDIUM",
  "confidence": 0.5,
  "detected_at": "2000-01-01T00:21:00Z",
  "detector": "StuckValueDetector",
  "evidence": {"run_length": 8, "window_size": 8, "value": 61.302, "tolerance": 0.0},
  "source": {"topic": "sensor-readings-raw", "partition": 7, "offset": 1234}
}
```

| Campo | Significado |
|---|---|
| `alert_id` | `uuid5(reading_id|fault_type|detector[|occurrence])`. Idempotente: un reproceso at-least-once produce el mismo id |
| `fault_type` | `stuck`, `noise_spike`, `bias_out_of_range`, `drift_correlated`, `sensor_drift`, `kalman_anomaly` |
| `severity` | `LOW`, `MEDIUM`, `HIGH`, `CRITICAL` (urgencia operativa, no probabilidad) |
| `confidence` | en [0, 1]; heuristica del detector, **no** probabilidad calibrada |
| `detected_at` | timestamp de la lectura que disparo, no hora de reloj |
| `evidence` | las cifras que llevaron a la deteccion (`SensorFault.evidence`) |
| `source` | topic, particion y offset de la lectura cruda: permite volver al dato original |

Severidad por detector, segun el codigo (`sensor_validator.py`): rango `CRITICAL` si el valor sale del span
calibrado, `HIGH` si excede el 25 % del ancho del sobre de alarma, `MEDIUM` en el resto; stuck escala con la
longitud de la racha (`MEDIUM` por debajo de 2 ventanas, `HIGH` desde 2, `CRITICAL` desde 4); ruido y deriva
`MEDIUM` o `HIGH`; deriva correlada `MEDIUM`; `kalman_anomaly` `MEDIUM`, y `HIGH` mientras el filtro no esta
inicializado. El plan decia "severidad baja" para `kalman_anomaly`: no es asi en el codigo y se anota como
observacion para revisarla (no se ha tocado, porque afecta al significado de las alertas).

---

## 4. Semantica de commit

```
poll (max 500 registros)
  -> por registro, en orden de offset dentro de cada particion:
       deserializar   (veneno: contar, registrar, saltar)
       continuidad     (run-id distinto o timestamp que retrocede -> reset del sensor)
       validar
       send(sensor-validated) y send(anomaly-alerts) por cada fallo
       pending[particion] = offset + 1
  -> producer.flush(30 s)      lanza PublishError si el timeout vence o algun envio fallo
  -> consumer.commit(pending)  sincrono, hasta 3 reintentos; despues CommitError
```

**Convenio**: `commit` recibe el **siguiente** offset a leer (ultimo procesado mas uno), igual que Kafka y que
`end_offsets`. Con ese convenio el lag de una particion es `end_offsets[p] - position(p)`.

| Fallo | Que pasa | Consecuencia |
|---|---|---|
| El proceso muere antes del `flush` | el lote no se confirma | se reprocesa entero al reiniciar; los mensajes ya enviados vuelven a salir |
| Muere entre `flush` y `commit` | el lote esta publicado pero no confirmado | se reprocesa: **duplicados** en `sensor-validated` y `anomaly-alerts` |
| `commit` falla | se reintenta `commit_retries` veces y luego `CommitError` | el pod reinicia sin avanzar; `commit_failures_total` cuenta cada intento fallido |
| `flush` falla | `PublishError`, `produce_errors_total`, el lote no se confirma | se reprocesa al reiniciar |
| Mensaje no deserializable | se salta **con** commit | se pierde ese mensaje; queda en el log con su hash |

Los duplicados son esperables y la deduplicacion es responsabilidad del consumidor aguas abajo, por
`reading_id` (lecturas) y `alert_id` (alertas).

**Tiempos.** El peor caso de un lote es `flush_timeout_s + (commit_retries + 1) * commit_timeout_ms` =
30 s + 4 x 30 s = **150 s**, por debajo de `max_poll_interval_ms` (300 s), o Kafka expulsaria al consumidor del
grupo a mitad de lote; el loader de `streaming_params.py` lo valida. El computo de un lote es despreciable
(validador p99 0,23 ms por lectura: 500 lecturas, unos 115 ms). `terminationGracePeriodSeconds` es 180 s en
el validador (cubre el peor caso) y 90 s en el streamer.

**Cierre ordenado.** `SIGTERM`/`SIGINT` activan un evento; el bucle termina el lote en curso, hace commit y
cierra. El exit code del servicio es 0 correcto, 1 error de ejecucion, 2 configuracion invalida.

---

## 5. Rebalanceo y estado

Los detectores acumulan estado **en memoria** por sensor (racha de stuck, ultimo valor, filtro de Kalman,
ventana de correlacion). Un rebalanceo lo pierde para las particiones que cambian de dueno.

| Concepto | Valor |
|---|---|
| Que se descarta | el estado de **todos** los sensores vistos en las particiones revocadas o perdidas, no solo los del ultimo lote |
| Por que se descarta | si la particion vuelve mas tarde, el estado obsoleto produciria resultados falsos |
| Recalentamiento de stuck | 8 muestras iguales = **24 min** a 180 s por muestra. Un sensor congelado no se detecta de nuevo hasta entonces |
| Como se ve | `reactorguard_consumer_rebalances_total` y `reactorguard_detector_state_resets_total{reason="rebalance"}` |
| Persistencia del estado | **no existe**. Persistirlo es un rediseno, no un ajuste |

**Paralelismo util.** El maximo teorico son 12 consumidores (uno por particion). El real es menor: con
`murmur2` sobre las 52 claves del TEP, **las particiones 5 y 9 quedan vacias** y el reparto es
`[5, 3, 5, 8, 8, 0, 6, 6, 5, 0, 4, 2]` sensores por particion (calculado con el `murmur2` de kafka-python, el
mismo que usa `InMemoryBroker`). Hay por tanto 10 consumidores utiles, y la particion mas cargada recibe 8 de
52 sensores (15,4 % del trafico frente al 8,3 % ideal).

**Actualizaciones.** El Deployment del validador usa `RollingUpdate` con `maxUnavailable: 1` y `maxSurge: 0`:
sustituye un pod cada vez y sin superar las replicas, de modo que cada paso del rollout provoca un
rebalanceo y un recalentamiento de 24 min en las particiones afectadas. Un `PodDisruptionBudget` con
`minAvailable: 1` evita que un drain deje el grupo sin consumidores.

**Serie de lag.** Al revocar o perder una particion, `_release` elimina su serie de
`reactorguard_consumer_lag`; una replica que cede una particion no sigue exponiendo su ultimo valor. El
dashboard usa `max by (partition)` igualmente, como defensa durante la ventana en que dos replicas la
reportan.

---

## 6. Como leer cada metrica

Todas llevan el prefijo `reactorguard_` y se exponen en `:9090/metrics` de cada pod. Las etiquetas son de baja
cardinalidad; nunca `sensor_id`.

| Metrica | Tipo | Etiquetas | Que mide y como leerla |
|---|---|---|---|
| `stream_messages_produced_total` | contador | `topic` | Mensajes entregados al productor, tambien por el consumidor (`sensor-validated`, `anomaly-alerts`). En `topic="sensor-readings-raw"` es el streamer |
| `stream_produce_errors_total` | contador | | Envios rechazados o `flush` con fallos. Cualquier incremento es un lote sin confirmar |
| `stream_produce_latency_seconds` | histograma | | Del primer `send` del lote hasta que vuelve su `flush`. No se grafica hoy |
| `readings_consumed_total` | contador | | Lecturas leidas del topic crudo. Debe igualar a `messages_produced{topic="sensor-validated"}` (D3) |
| `readings_validated_total` | contador | `quality` | Resultado de la validacion: `good`, `suspect`, `bad` |
| `sensor_faults_total` | contador | `fault_type` | Fallos de sensor detectados. Una lectura puede contar varios |
| `alerts_published_total` | contador | `severity` | Eventos publicados en `anomaly-alerts`; misma poblacion que `sensor_faults_total` |
| `poison_messages_total` | contador | | No deserializables saltados. Debe ser 0 |
| `validation_latency_seconds` | histograma | | Segundos por lectura en el validador. Buckets de 50 us a 250 ms; por encima cae en `+Inf` |
| `batch_size` | histograma | | Registros por lote confirmado. Su `_count` cuenta commits efectivos (solo se actualiza tras un checkpoint correcto) |
| `consumer_lag` | gauge | `partition` | `end_offset - position`, tras cada lote. Lo calcula el consumidor, no un exporter |
| `consumer_rebalances_total` | contador | | Reasignaciones vistas por la replica |
| `detector_state_resets_total` | contador | `reason` | `run_change` (cambio de fichero), `time_regression` (red de seguridad), `rebalance` |
| `commit_failures_total` | contador | | Intentos de commit fallidos. Un reintento exitoso tambien cuenta |
| `consumer_heartbeat_timestamp_seconds` | gauge | | Unix time de la ultima iteracion del bucle; base de `/health` |

### 6.1 La ventana de `rate()` y la cadencia de 180 s

El TDD y los prompts originales dan por hecha una muestra por segundo; el TEP entrega **un lote de 52
lecturas cada 180 s** (0,29 msg/s de media en modo realtime). Esto cambia como leer cualquier `rate()`:

- **Medido en M8**: con scrape de 15 s, `rate(...[1m])` **no** devuelve vacio. Da 0 en el 80 % de las
  evaluaciones y, cuando hay un lote dentro de la ventana, picos de 1,16 msg/s frente a una media real de
  0,29. Con `[5m]` da 0 en el 44 % de las evaluaciones. La serie solo queda vacia con scrape >= 60 s.
  El defecto es por tanto **un valor enganoso, no una ausencia visible**, que es peor.
- Por eso el dashboard expone la ventana como variable `rate_window` (`1m|5m|15m|1h`, **defecto 15m**) y
  usa `increase()` para los conteos de fallos y alertas.
- Una ventana de al menos **dos intervalos de muestreo (6 min)** es el minimo para que el valor represente
  la tasa y no el instante del lote. 15 m da 5 intervalos.

### 6.2 Latidos

`consumer_heartbeat_timestamp_seconds` lo da tanto el streamer como el validador. El silencio legitimo mas
largo es el del streamer en modo realtime, 180 s entre lotes; la espera se trocea en tramos de 30 s con
latido entre ellos. `/health` devuelve 503 a los `staleness_seconds` = **540 s** (3 intervalos): tolera dos
lotes perdidos antes de que el kubelet reinicie el pod. Con los 30 s heredados del TDD el pod se reiniciaria
en cada espera.

`/ready` es distinto: el validador esta listo cuando se ha unido al grupo y no esta rebalanceando. Un pod sin
particiones (mas pods que particiones utiles) esta listo: simplemente no tiene trabajo.

---

## 7. Runbook de las condiciones de Fase 2

Todavia **no existen reglas de alerta** de Prometheus (T12.1, Fase 8). Los umbrales siguientes son los colores
del dashboard `data-pipeline` y la base de esas reglas. Los comandos `kubectl` son de referencia para cuando
exista el cluster; ninguno se ha ejecutado contra uno.

| Senal | Umbral (dashboard) | Significado | Diagnostico | Accion |
|---|---|---|---|---|
| Lag total | amarillo > 500 | mas de un `poll` completo pendiente | `Lag por particion`; `batch_size`; latencia p99; replicas frente a particiones utiles (10) | subir replicas hasta 10; si una sola particion crece, es la mas cargada (8 sensores): no se arregla con replicas |
| Lag fijo y no nulo con el consumidor vivo | | un consumidor atascado en un lote o sin asignacion | edad del latido; `kubectl logs` del pod; `/ready` | si hay `PublishError` o `CommitError`, ver filas siguientes; si no hay asignacion, hay mas pods que particiones utiles |
| Edad del latido | amarillo 180 s, rojo 540 s | el bucle no itera; a 540 s `/health` da 503 y el pod reinicia | CPU y memoria del pod; si el streamer esta en `realtime`, 180 s es normal | rojo repetido: revisar OOM, `max_poll_interval`, red hacia el broker |
| Fallos de commit | rojo >= 1 | el grupo es inestable (reasignacion durante el lote) aunque no se haya perdido nada | rebalanceos en la misma ventana; pods reiniciandose | estabilizar replicas; si persiste, comprobar `max_poll_interval_ms` y `commit_timeout_ms` |
| Errores de publicacion | rojo >= 1 | un `send` o un `flush` fallo; el lote se reprocesa | broker y `min.insync.replicas` (2 de 3); ACL de escritura del KafkaUser | si es transitorio no hace falta nada: es at-least-once; si se repite, ver estado del cluster Kafka |
| Mensajes venenosos | rojo >= 1 | llego un valor no deserializable; se salto con commit | `kubectl logs`: topic, particion, offset y hash sha256 | localizar al productor que lo emitio. **No hay DLQ**: el contenido solo esta en el log. Rescatarlo antes de que expire la retencion (7 d) |
| Rebalanceos | no hay umbral | cada uno reinicia el estado de detectores | coincide con un despliegue o un cambio de replicas? Si no: reinicios por OOM o expulsion del grupo | sin causa externa, tratarlo como inestabilidad. Cada uno cuesta 24 min de recalentamiento de stuck |
| Resets `time_regression` | esperado 0 | el reloj de un sensor retrocedio sin cambio de `run-id` | origen: un productor sin cabecera `run-id`, o un reenvio fuera de orden | revisar el productor; el consumidor ya ha reiniciado el estado |
| Resets `run_change` | esperado: uno por sensor y por fichero | cambio de run (o vuelta del bucle) | normal en el TEP: 22 ficheros | ninguna |
| `quality` `bad` o `suspect` alto | | muchas lecturas con fallos de instrumento | ver `Fallos por fault_type` y `Fallos menos frecuentes` | ver [seccion 8](#8-dimensionado-de-alertas): en los runs con fallo de proceso del TEP es esperable |
| Alerta `CRITICAL` de `stuck` o `bias_out_of_range` | | valor fuera del span calibrado o racha >= 4 ventanas | la alerta lleva `evidence` y `source` (particion y offset) | inspeccionar el sensor; en el TEP, comparar con el `run_id` |
| Paneles vacios | | la ventana de `rate()` es menor que dos intervalos, o no hay scrape de los pods | variable `rate_window`; target de Prometheus | subir a 15m; ver deuda de despliegue (no hay Prometheus ni scrape de los pods en el cluster) |
| Feature nula en lectura online | | `stale_sensors`: el valor supera 540 s de antiguedad (frescura de `FeastClient`); `missing_sensors`: la clave caduco en Redis (`key_ttl_seconds` = 540 s) o nunca se escribio | los dos campos de `OnlineFeatures` | comprobar la materializacion y el productor del sensor; no imputar en silencio. Un Redis inalcanzable lanza `ConnectionError`, no devuelve nulos |

---

## 8. Dimensionado de alertas

El handoff de Paso 4 dimensiono las alertas con el 3,59 % de falsos positivos del residual de Kalman
(medido solo en operacion normal, sobre sensores intactos de d00): unas **19.700** alertas sobre 550.160
lecturas. Ese numero **no es el del sistema completo**. Medido con el validador sobre los 22 runs:

| Magnitud | Valor medido |
|---|---:|
| Lecturas validadas | 550.160 |
| **Alertas emitidas** | **100.934 (18,3 %)** |
| `bias_out_of_range` | 70.225 |
| `stuck` | 15.765 |
| `kalman_anomaly` | 11.518 |
| `sensor_drift` | 1.920 |
| `noise_spike` | 1.506 |
| `quality` `good` | 462.478 |
| `quality` `suspect` | 55.905 |
| `quality` `bad` | 31.777 |

Cuatro consecuencias:

1. `anomaly-alerts` (3 particiones) debe dimensionarse con ~100.000 eventos por pasada del dataset, no con
   ~20.000. Al ritmo medido del validador en local (unas 2,2 mil lecturas/s) y con un 18,3 % de lecturas
   que generan alertas, serian del orden de 400 eventos/s en ese topic (cifra derivada de dos medidas, no
   medida directamente).
2. **La causa no esta investigada.** La hipotesis probable es que 21 de los 22 ficheros son perturbaciones
   de proceso que sacan los sensores de su sobre de operacion normal, con lo que `bias_out_of_range` mide
   proceso y no instrumento. Se documenta como cifra medida, no como defecto.
3. **`stuck` merece revision**: 15.765 alertas contrastan con las 480 muestras realmente congeladas de
   d21 (el unico positivo real del dataset). La validacion offline sobre d00 degradado da precision 1,0000 y
   recall 0,81, de modo que la discrepancia no la explica el detector en condiciones controladas. No se ha
   investigado en que runs y sensores aparecen esas alertas.
4. La precision del criterio 2 (1,0000) es sobre inyeccion sintetica en d00 y sobre el unico positivo real; no
   dice nada sobre la tasa de alertas de los 21 runs de proceso.

`kalman_anomaly` es senal de apoyo y no lleva gate: `drift_correlated` no se evalua porque
`CrossCorrelationChecker` viene sin parejas base.

---

## 9. Que no cierra esta fase

| Criterio | Estado | Evidencia disponible | Para cerrarlo |
|---|---|---|---|
| 1. Throughput > 50.000 msg/s | **NO MEDIDO** | local: 13.993 msg/s con un proceso Python y un broker de un nodo (informativo) | `Invoke-KafkaTests.ps1` en el cluster |
| 2. Stuck precision > 95 % | **CUMPLIDO** | 1,0000; 27 tests ejecutados, 0 saltados (`Verify-Phase2.ps1`) | |
| 3. Latencia de features < 100 ms | **NO MEDIDO** | local, informativo (ver Progreso) | `benchmark_feature_latency.py --environment cluster` |
| 4. `dvc repro` desde cero | **CUMPLIDO** | `dvc status` limpio, `dvc.lock` completo y reproduccion desde cero en una copia temporal con las 8 salidas bit a bit iguales (`Verify-Phase2.ps1 -FromScratch`, 80 s) | |

Un criterio solo es CUMPLIDO con evidencia del entorno que el criterio exige. `Verify-Phase2.ps1` lo impone:
una medicion `environment: local` deja el criterio en NO MEDIDO aunque supere el umbral.

Limitaciones conocidas (deuda que pasa a la Fase 3 o posterior):

- **Sin DLQ.** Un mensaje venenoso se pierde salvo por el log.
- **Estado de detectores sin persistir.** Cada rebalanceo cuesta 24 min de recalentamiento.
- **Sin camino de features en linea.** `FeaturePipeline.transform()` es por lotes sobre una ventana contigua;
  el servicio de validacion no calcula features.
- **Fuente del streamer en el cluster.** No hay remoto DVC; el parquet debe subirse al bucket con
  `gsutil rsync data/processed/tep gs://reactorguard-data-raw-dev/tep/`.
- **Sin Grafana ni Prometheus en el cluster** (T2.6 de la Fase 1 sin hacer) ni scrape de los pods en 9090.
- **Imagen unica y grande** (`api/Dockerfile` arrastra torch para streamer y validador).
- **Un solo proceso productor.** El techo medido de un proceso de kafka-python esta entre 12.000 y 17.000
  msg/s; llegar a 50.000 en el cluster exigira fragmentar sensores entre procesos productores (cada sensor
  siempre en uno solo, para no romper el orden) o un productor mas rapido. No se ha anticipado.
