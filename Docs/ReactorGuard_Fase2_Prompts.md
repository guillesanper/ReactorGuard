# ReactorGuard — Fase 2: Plan Detallado con Prompts
**Data Pipeline + Sensor Validator · Semanas 3–4**

---

## Cómo usar este documento

Cada subtarea tiene:
- **Qué hace**: descripción del resultado esperado
- **Prerequisitos**: qué debe estar listo antes
- **Prompt**: listo para pegar directamente en Claude
- **Verificación**: cómo comprobar que está bien hecho

El orden es el orden de ejecución. No saltes tareas.

---

## SEMANA 3 — TEP Dataset + Schema + Sensor Validator

---

### T3.1 — Schema de sensor reading (Pydantic)

**Qué hace**: Define el contrato de datos de todo el sistema. Cada mensaje que viaje por Kafka, cada lectura que entre al validador y a los modelos ML, tendrá este formato. Es la fuente de verdad del schema del TDD sección 4.2.

**Prerequisitos**: Fase 1 completada. Repositorio con estructura base de T1.1.

**Prompt**:
```
Crea el schema completo de datos de sensor para ReactorGuard en data/schemas/sensor_reading.py.

El schema es la fuente de verdad del TDD sección 4.2. Debe cubrir exactamente este JSON:

{
  "reading_id": "uuid-v4",
  "timestamp": "2024-01-15T14:32:18.001Z",   <- ms precision
  "plant_id": "REACTOR-01",
  "sensor": {
    "id": "TC-CORE-12",
    "type": "thermocouple | flux_detector | pressure | flow | position",
    "location": "primary_loop | secondary_loop | core | containment",
    "elevation_m": 3.45
  },
  "measurement": {
    "value": 312.4,
    "unit": "celsius | bar | kg_s | percent | normalized",
    "quality": "good | suspect | bad | missing",
    "raw_counts": 4092
  },
  "metadata": {
    "calibration_date": "2024-01-01",
    "last_maintenance": "2023-12-15",
    "drift_coefficient": 0.0012
  }
}

Requisitos de implementación:

1. Usa Pydantic v2 con model_validator y field_validator donde tenga sentido física:
   - elevation_m debe ser >= -10.0 y <= 100.0 (rango físico del edificio del reactor)
   - raw_counts debe ser 0-65535 (ADC de 16 bits)
   - drift_coefficient debe ser >= 0.0
   - Si quality == "missing", value puede ser None
   - Si quality == "bad", el sensor no debe usarse para inferencia (añadir propiedad is_usable)

2. Añade estos Enums explícitos: SensorType, SensorLocation, MeasurementUnit, QualityFlag

3. Añade una propiedad computed is_usable: bool que retorna False si quality es "bad" o "missing"

4. Añade un método de clase from_kafka_bytes(data: bytes) -> SensorReading que deserializa desde JSON bytes (para el consumer de Kafka)

5. Añade un método to_kafka_bytes() -> bytes que serializa a JSON bytes (para el producer)

6. Añade un método to_feature_dict() -> dict que retorna solo los campos numéricos relevantes para ML:
   {"sensor_id", "timestamp_unix", "value", "raw_counts", "drift_coefficient", "elevation_m"}

7. Crea data/schemas/__init__.py exportando todas las clases

8. Crea tests/unit/test_sensor_reading.py con casos de test:
   - Schema válido completo: OK
   - quality="missing" con value=None: OK
   - quality="missing" con value=312.4: OK (permitido, el sensor puede tener valor y calidad missing)
   - raw_counts=70000: falla (fuera de rango ADC)
   - drift_coefficient=-0.001: falla (negativo)
   - is_usable=False cuando quality="bad"
   - from_kafka_bytes ↔ to_kafka_bytes roundtrip sin pérdida
   - to_feature_dict retorna exactamente los campos esperados

Añade docstrings en cada clase y método.
```

**Verificación**:
```bash
pytest tests/unit/test_sensor_reading.py -v
# Todos los tests en verde
python -c "from data.schemas import SensorReading; print('Schema OK')"
```

---

### T3.2 — Descarga y exploración del dataset TEP

**Qué hace**: Descarga el dataset Tennessee Eastman Process, lo explora para entender su estructura, y lo adapta al schema de ReactorGuard. TEP es el estándar de referencia para detección de anomalías industriales: 52 variables de proceso, 21 tipos de fallo.

**Prerequisitos**: T3.1 completada. DVC inicializado (`dvc init` en el repo).

**Prompt**:
```
Crea los scripts de descarga, exploración y adaptación del dataset Tennessee Eastman Process (TEP) para ReactorGuard.

El TEP tiene esta estructura:
- Archivos: d00.dat (normal), d01.dat a d21.dat (21 tipos de fallo)
- Variables: 52 columnas — 41 variables de proceso medidas + 11 variables de manipulación
- Frecuencia: 1 muestra cada 3 minutos en el dataset original
- URL: https://github.com/camaramm/tennessee-eastman-profBraatz

Las 52 variables del TEP mapean de esta forma a los sensores de ReactorGuard:
- Variables 1-22: variables de proceso (temperatura, presión, flujo) → sensor type thermocouple/pressure/flow
- Variables 23-41: variables de análisis (composición) → sensor type normalized
- Variables 42-52: variables de manipulación (actuadores) → sensor type position
Los 21 fault types del TEP son análogos a los del TDD: faults 1-7 son step changes (bias faults), 8-12 son random variations (noise), 13 es slow drift, 14-15 son sticking faults (stuck sensor), 16-21 son faults desconocidos.

Genera estos archivos:

1. data/generators/tep_downloader.py:
   - Función download_tep(output_dir: str) que descarga los 22 archivos .dat desde el repositorio GitHub
   - Usa requests con retry (3 intentos, backoff exponencial)
   - Muestra progress bar con tqdm
   - Verifica checksums MD5 de los archivos descargados
   - Si los archivos ya existen, no re-descarga (idempotente)

2. data/generators/tep_explorer.py:
   - Función explore_tep(data_dir: str) -> dict que carga todos los archivos y produce un reporte:
     - Shape de cada dataset (normal vs. cada fault)
     - Estadísticas básicas: mean, std, min, max por variable
     - Distribución de valores faltantes
     - Correlaciones entre variables (matriz de Pearson)
   - Genera data/reports/tep_exploration.json con el reporte
   - Genera data/reports/tep_correlations.csv con la matriz de correlación

3. data/generators/tep_adapter.py:
   - Clase TEPAdapter que convierte filas del TEP a objetos SensorReading:
     - Mapea cada columna del TEP a un sensor_id (ej. columna 0 → "TEP-XMEAS-01")
     - Asigna sensor types según el mapeo anterior
     - El timestamp se genera sintéticamente: start_time + row_index * 3_minutos
     - El fault type del archivo determina la quality flag:
       - d00 (normal): quality="good"
       - d01-d21 (faults): quality="suspect" para todas las lecturas del periodo de fallo
     - plant_id = "TEP-PLANT-01"
   - Método adapt_file(filepath: str, fault_type: int) -> List[SensorReading]
   - Método adapt_all(data_dir: str) -> pd.DataFrame con todos los readings adaptados

4. Script data/generators/prepare_tep.sh:
   - Ejecuta download + explore + adapt
   - Guarda el resultado en data/raw/tep/ como parquet particionado por fault_type
   - Imprime estadísticas finales: total de readings, distribución por fault_type

5. Actualiza dvc.yaml con stage download_tep:
   - cmd: python data/generators/tep_downloader.py
   - outs: data/raw/tep/

6. tests/unit/test_tep_adapter.py:
   - Test que la adaptación de una fila del TEP produce un SensorReading válido
   - Test que el fault_type correcto mapea a quality="suspect"
   - Test que los timestamps son monotónicamente crecientes
```

**Verificación**:
```bash
dvc repro download_tep
ls data/raw/tep/  # 22 archivos .dat descargados
python data/generators/tep_explorer.py  # Genera tep_exploration.json sin errores
pytest tests/unit/test_tep_adapter.py -v
```

---

### T3.3 — TEP Streamer: CSV → Kafka producer

**Qué hace**: Reproduce el dataset TEP como si fuera un stream en tiempo real publicando en Kafka. Es el simulador de la fuente de datos para todas las pruebas de la Fase 2 y la base del throughput benchmark.

**Prerequisitos**: T3.1, T3.2 completadas. Kafka cluster de Fase 1 operativo.

**Prompt**:
```
Crea el TEP Streamer para ReactorGuard: el componente que lee el dataset TEP y lo publica en Kafka simulando un stream de sensores en tiempo real.

El streamer debe ser configurable: puede correr en modo "real-time" (respeta los timestamps originales del TEP, 1 mensaje cada 3 minutos) o en modo "fast" (publica tan rápido como puede, para benchmarking de throughput).

Genera estos archivos:

1. data/generators/tep_streamer.py — clase principal TEPStreamer:

   class TEPStreamer:
     def __init__(self, kafka_bootstrap: str, topic: str, data_dir: str, speed_multiplier: float = 1.0)
     - speed_multiplier=1.0 → tiempo real, speed_multiplier=0.0 → máxima velocidad (sin sleep)
     - Inicializa KafkaProducer con:
       - bootstrap_servers: el bootstrap del cluster
       - value_serializer: SensorReading.to_kafka_bytes
       - acks='all' (espera confirmación de todos los ISR — durabilidad máxima)
       - retries=3, retry_backoff_ms=100
       - compression_type='lz4' (reduce bandwidth ~4x para datos de sensor)
       - batch_size=16384, linger_ms=5 (micro-batching para throughput)

   def stream_file(self, filepath: str, fault_type: int, loop: bool = False)
     - Lee el archivo TEP fila a fila usando TEPAdapter
     - Publica cada SensorReading en el topic con key=sensor_id (para que mensajes del mismo sensor vayan siempre a la misma partición)
     - Si speed_multiplier > 0: sleep entre mensajes según timestamps
     - Si loop=True: cuando acaba el archivo, vuelve a empezar (útil para demos)
     - Métricas: mensajes enviados, bytes enviados, latencia de produce(), errores

   def stream_all(self, interleave: bool = True)
     - Si interleave=True: mezcla todos los fault types para simular una planta real
     - Si interleave=False: un fault type a la vez (para testing)

   def get_metrics(self) -> dict
     - Retorna: messages_sent, bytes_sent, errors, avg_produce_latency_ms, throughput_msg_per_sec

2. data/generators/tep_streamer_service.py — entry point para correr como servicio K8s:
   - Lee configuración de variables de entorno: KAFKA_BOOTSTRAP, KAFKA_TOPIC, DATA_DIR, SPEED_MULTIPLIER
   - Expone /health y /metrics (Prometheus) en puerto 8001
   - Graceful shutdown con signal handlers (SIGTERM → termina la iteración actual y cierra el producer)
   - Logs estructurados en JSON (para Cloud Logging)

3. k8s/base/ingestion/tep-streamer-deployment.yaml:
   - Deployment en namespace reactorguard-ingestion
   - 1 réplica (singleton — solo queremos un stream por planta)
   - ConfigMap con KAFKA_BOOTSTRAP, DATA_DIR
   - Secret reference para credenciales si las hubiera
   - Resources: request 256m CPU / 512Mi RAM, limit 500m / 1Gi

4. tests/unit/test_tep_streamer.py:
   - Test con KafkaProducer mockeado (unittest.mock)
   - Verifica que stream_file() produce exactamente N mensajes para N filas del TEP
   - Verifica que el key de cada mensaje es el sensor_id
   - Verifica que speed_multiplier=0.0 no hace sleep
   - Verifica que get_metrics() retorna throughput correcto

5. tests/integration/test_tep_streamer_integration.py:
   - Test de integración real contra Kafka del cluster
   - Produce 1000 mensajes con speed_multiplier=0.0
   - Consumer verifica que llegan los 1000 mensajes
   - Mide throughput real y verifica > 1000 msg/s en modo fast

Añade logging estructurado en JSON en todos los métodos principales.
```

**Verificación**:
```bash
pytest tests/unit/test_tep_streamer.py -v  # Tests unitarios con mock
pytest tests/integration/test_tep_streamer_integration.py -v  # Contra Kafka real
# El test de integración debe imprimir throughput > 1000 msg/s en modo fast
```

---

### T3.4 — Sensor Validator: los 5 detectores

**Qué hace**: Implementa el pre-filtro de sensores que clasifica cada lectura ANTES de que llegue a cualquier modelo ML. Es la pieza más crítica de la arquitectura: un sensor roto nunca debe contaminar los modelos.

**Prerequisitos**: T3.1 completada.

**Prompt**:
```
Implementa el Sensor Validator de ReactorGuard: el componente que clasifica el estado de cada sensor antes de que sus datos lleguen a los modelos ML.

Según el TDD sección 3.3, hay 5 detectores. Cada uno detecta un tipo distinto de fallo de sensor.

Genera data/validation/sensor_validator.py con estas clases:

---

1. StuckValueDetector:
   - Mantiene un buffer rolling de las últimas N lecturas por sensor_id
   - Calcula la varianza del buffer
   - Si varianza < threshold durante >= min_stuck_samples consecutivos → fallo stuck
   - Parámetros: window_size=20, variance_threshold=0.001, min_stuck_samples=10
   - Método: check(reading: SensorReading) -> Optional[SensorFault]
   - El detector es stateful: mantiene estado por sensor_id en un dict interno

2. RateOfChangeDetector:
   - Mantiene el último valor por sensor_id y su timestamp
   - Calcula dX/dt = (current_value - last_value) / elapsed_seconds
   - Compara con physical_limits: dict de sensor_type → max_rate_per_second
     - thermocouple: 50.0 °C/s
     - pressure: 10.0 bar/s
     - flow: 100.0 kg_s/s
     - flux_detector: 1e15 neutrons/cm²/s²  (alta por naturaleza)
     - position: 10.0 percent/s
   - Si |dX/dt| > physical_limit → fallo noise_spike
   - Método: check(reading: SensorReading) -> Optional[SensorFault]

3. RangeValidator:
   - Valida que el valor esté dentro de los límites físicos del tipo de sensor
   - physical_ranges: dict de sensor_type → (min_value, max_value)
     - thermocouple: (0.0, 700.0) °C (agua en condiciones normales de reactor)
     - pressure: (0.0, 200.0) bar
     - flow: (0.0, 2000.0) kg/s
     - flux_detector: (0.0, 1e18) n/cm²/s
     - position: (0.0, 100.0) %
   - Si value fuera de rango → fallo bias_out_of_range
   - Stateless (no necesita historial)
   - Método: check(reading: SensorReading) -> Optional[SensorFault]

4. CrossCorrelationChecker:
   - Mantiene ventanas rolling de pares de sensores correlacionados
   - baseline_correlations: dict de (sensor_id_a, sensor_id_b) → expected_pearson
     (se carga desde un archivo JSON de calibración; por ahora acepta un dict vacío → sin checks)
   - Calcula la correlación de Pearson en la ventana actual
   - Si |pearson_actual - pearson_expected| > correlation_threshold=0.3 → fallo drift_correlated
   - Parámetros: window_size=100, correlation_threshold=0.3
   - Método: check_pair(id_a: str, id_b: str) -> Optional[SensorFault]
   - Método: update(reading: SensorReading) para actualizar el buffer del sensor

5. KalmanResidualDetector:
   - Un filtro de Kalman por sensor_id (instanciado lazy al primer reading)
   - Modelo de Kalman de orden 1: estado = [valor, velocidad], observación = [valor]
     - F (transición de estado): [[1, dt], [0, 1]]
     - H (observación): [[1, 0]]
     - Q (ruido de proceso): ajustable, default 0.1 * I
     - R (ruido de observación): ajustable, default 1.0
   - Calcula el residual = |measurement - prediction|
   - Si residual > k_sigma * sigma_predicted → fallo kalman_anomaly
   - Parámetros: k_sigma=3.0, process_noise=0.1, observation_noise=1.0
   - Auto-reinicia el filtro si el residual es > 10*k_sigma (evita divergencia permanente)
   - Método: check(reading: SensorReading) -> Optional[SensorFault]

---

Además genera:

6. data/validation/sensor_fault.py — dataclass SensorFault:
   - sensor_id: str
   - fault_type: Literal["stuck", "noise_spike", "bias_out_of_range", "drift_correlated", "kalman_anomaly"]
   - severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
   - confidence: float (0.0-1.0)
   - detected_at: datetime
   - detector: str (nombre del detector que lo detectó)
   - evidence: dict (valores que llevaron a la detección, para explicabilidad)

7. data/validation/sensor_validator.py — clase orquestadora SensorValidator:
   - Agrega los 5 detectores
   - Método validate(reading: SensorReading) -> ValidationResult
   - ValidationResult tiene: is_valid: bool, faults: List[SensorFault], enriched_reading: SensorReading
     (el enriched_reading tiene el quality flag actualizado si se detecta un fallo)
   - Expone métricas Prometheus: contador por fault_type, latencia del validate()

8. tests/unit/test_sensor_validator.py con al menos 3 casos de test por detector:
   - StuckValueDetector: stream de valores idénticos → detecta stuck; stream variable → no detecta
   - RateOfChangeDetector: salto de 100°C en 1s → detecta spike; cambio gradual → no detecta
   - RangeValidator: valor -5°C en thermocouple → detecta; valor 300°C → no detecta
   - KalmanResidualDetector: valor normal → residual bajo; valor aberrante → residual alto → detecta
   - SensorValidator orquestador: reading con stuck Y out of range → ambos fallos en la lista

Añade type hints completos y docstrings en todos los métodos públicos.
```

**Verificación**:
```bash
pytest tests/unit/test_sensor_validator.py -v --tb=short
# Todos los tests en verde
# Cobertura de código > 80%: pytest --cov=data/validation/
```

---

### T3.5 — Kafka Consumer: pipeline de validación

**Qué hace**: Construye el consumidor Kafka que conecta todo: lee de `sensor-readings-raw`, aplica el SensorValidator, y publica el resultado enriquecido en `sensor-validated`. Es el primer servicio de producción del sistema.

**Prerequisitos**: T3.3 (Streamer) y T3.4 (Validator) completadas. Kafka operativo.

**Prompt**:
```
Crea el Kafka Consumer de validación de sensores para ReactorGuard.

Este servicio consume del topic sensor-readings-raw, aplica el SensorValidator, y publica los resultados en sensor-validated. Es el primer microservicio real de producción.

Genera estos archivos:

1. data/validation/validation_consumer.py — clase ValidationConsumer:

   class ValidationConsumer:
     def __init__(self, kafka_bootstrap: str, group_id: str = "sensor-validator-group")
     - KafkaConsumer configurado con:
       - group_id: permite múltiples instancias del consumer (cada una lee particiones distintas)
       - auto_offset_reset='earliest' (en dev) / 'latest' (en prod)
       - enable_auto_commit=False (commit manual para garantizar at-least-once processing)
       - max_poll_records=500 (micro-batch de 500 mensajes)
       - session_timeout_ms=30000
       - heartbeat_interval_ms=10000
     - KafkaProducer para sensor-validated con misma config que T3.3
     - SensorValidator instanciado con los 5 detectores
     - Métricas Prometheus: messages_consumed, messages_validated_ok, faults_detected (por tipo), processing_latency

   def process_batch(self, records: List[ConsumerRecord]) -> ProcessingResult:
     - Para cada record:
       1. Deserializar: SensorReading.from_kafka_bytes(record.value)
       2. Validar: result = self.validator.validate(reading)
       3. Si result.is_valid: publicar enriched_reading en sensor-validated
       4. Si result.faults: publicar también en anomaly-alerts con severity del peor fault
       5. Registrar métricas
     - Retorna ProcessingResult con counts y latencias
     - Commit offset solo después de que todos los mensajes del batch estén publicados

   def run(self):
     - Loop principal: poll → process_batch → commit
     - Graceful shutdown con signal handler: completa el batch actual antes de cerrar
     - Logging estructurado por batch: N messages processed, M faults detected

2. data/validation/validation_service.py — entry point:
   - Lee config de env vars: KAFKA_BOOTSTRAP, CONSUMER_GROUP_ID, INPUT_TOPIC, OUTPUT_TOPIC
   - Health check en /health puerto 8000: retorna {"status": "ok", "lag": N} donde N es el consumer lag
   - Métricas Prometheus en /metrics puerto 8000
   - Llama a ValidationConsumer.run()

3. k8s/base/ingestion/sensor-validator-deployment.yaml:
   - Namespace: reactorguard-ingestion
   - ServiceAccount: sensor-validator (creado en Fase 1 T2.2)
   - 2 réplicas (cada una consume particiones distintas del topic — con 12 particiones, 2 réplicas es suficiente para empezar)
   - ConfigMap con KAFKA_BOOTSTRAP, TOPIC names
   - Liveness probe: /health cada 30s
   - Resources: request 500m CPU / 1Gi RAM, limit 1000m / 2Gi
   - PodDisruptionBudget: minAvailable=1

4. tests/integration/test_validation_pipeline.py:
   - Test end-to-end: publica 100 mensajes en sensor-readings-raw con el TEP Streamer
   - Espera a que el consumer procese (poll hasta que consumer lag = 0)
   - Lee de sensor-validated y verifica que llegaron exactamente 100 mensajes (o menos si alguno era inválido)
   - Para un mensaje con stuck sensor simulado: verifica que aparece en anomaly-alerts con fault_type="stuck"
   - Verifica la latencia end-to-end (from raw → validated) < 100ms p99

Añade un diagrama de flujo en el docstring de la clase con el ciclo completo: raw → validate → validated/alerts.
```

**Verificación**:
```bash
# Levantar el consumer localmente:
KAFKA_BOOTSTRAP=localhost:9092 python data/validation/validation_service.py &
# En otra terminal, publicar mensajes de prueba:
python -c "from tests.integration.test_validation_pipeline import *; run_quick_test()"
# Verificar consumer lag = 0:
kubectl exec -n kafka-operator kafka-pod -- kafka-consumer-groups.sh \
  --bootstrap-server localhost:9092 --describe --group sensor-validator-group
pytest tests/integration/test_validation_pipeline.py -v
```

---

### T3.6 — Tests unitarios del Sensor Validator con datos TEP reales

**Qué hace**: Valida que los detectores funcionan sobre datos reales del TEP, no solo sobre datos sintéticos. Calcula las métricas de precisión del criterio de éxito: precision > 95% en stuck sensor.

**Prerequisitos**: T3.2 (TEP descargado), T3.4 (Validator implementado).

**Prompt**:
```
Crea la suite de evaluación del Sensor Validator sobre el dataset TEP para ReactorGuard.

El criterio de éxito de la Fase 2 es: "stuck sensor precision > 95% en test set TEP".
Necesitamos evaluar los 5 detectores sobre datos TEP reales y calcular las métricas.

El TEP tiene fault types que mapean a nuestros detectores:
- Faults 14, 15: sticking valve → stuck sensor
- Faults 1-7: step changes → bias / range faults
- Fault 13: slow drift → kalman / drift
- Faults 8-12: random variation → noise spike (en algunos)
- d00 (normal): sin fallos → todos los detectores deben retornar is_valid=True

Genera estos archivos:

1. tests/unit/test_validator_on_tep.py — evaluación de métricas sobre TEP:

   Clase TEPValidatorEvaluator:
   - Carga todos los archivos TEP adaptados desde data/raw/tep/
   - Para cada archivo (fault type) y cada lectura, ejecuta el SensorValidator
   - Construye una matriz de confusión por detector:
     - TP: fallo detectado cuando hay fallo real (según fault type del archivo TEP)
     - FP: fallo detectado cuando el archivo es d00 (normal)
     - TN: no fallo detectado en d00
     - FN: no fallo detectado cuando hay fallo real

   Función evaluate_stuck_detector():
   - Archivos positivos: d14, d15 (sticking faults)
   - Archivos negativos: d00 (normal)
   - Calcula precision, recall, F1 del StuckValueDetector
   - El test FALLA si precision < 0.95

   Función evaluate_range_detector():
   - Archivos positivos: d01-d07 (step changes que pueden salir del rango)
   - Calcula métricas del RangeValidator

   Función evaluate_all_detectors() -> dict:
   - Ejecuta evaluación para todos los detectores
   - Retorna DataFrame con precision/recall/F1 por detector
   - Guarda en tests/results/validator_metrics_tep.json
   - Imprime tabla de resultados

2. tests/unit/test_validator_edge_cases.py — casos límite adicionales:
   - Sensor que alterna entre good y stuck: detector debe detectar el periodo stuck
   - Sensor con ruido muy alto: RangeValidator no debe dar falsos positivos si el valor sigue en rango
   - Primer reading de un sensor nuevo: Kalman debe inicializar sin error
   - Reading con gap de tiempo grande (>1 hora): RateOfChange no debe fallar por el gap
   - Secuencia de 1000 readings normales seguidos de stuck: detecta el onset del stuck en < 15 readings

3. Script tests/run_validator_evaluation.sh:
   - Ejecuta ambas suites de tests
   - Genera el JSON de métricas
   - Imprime tabla de resultados:
     Detector               Precision  Recall   F1
     StuckValueDetector     0.97       0.89     0.93  ✅ (≥0.95)
     RateOfChangeDetector   0.91       0.95     0.93
     RangeValidator         0.99       0.72     0.84
     ...
   - Falla con código 1 si StuckValueDetector precision < 0.95

Incluye comentarios explicando por qué la precision del stuck detector es el criterio principal (los falsos positivos de stuck son muy costosos operacionalmente).
```

**Verificación**:
```bash
bash tests/run_validator_evaluation.sh
# StuckValueDetector precision ≥ 0.95 → ✅
# JSON de métricas en tests/results/validator_metrics_tep.json
pytest tests/unit/test_validator_edge_cases.py -v
```

---

## SEMANA 4 — Feature Engineering + Feast + DVC

---

### T4.1 — Filtro de Kalman online por sensor

**Qué hace**: Implementa el filtro de Kalman que corre en línea (actualiza su estado con cada nueva lectura). Es la base del feature `kalman_residual` y del KalmanResidualDetector del Validator. Al separarlo en su propio módulo evitamos duplicación entre ambos usos.

**Prerequisitos**: T3.1 completada.

**Prompt**:
```
Implementa el filtro de Kalman online para ReactorGuard en ml/features/kalman.py.

El filtro de Kalman estima el estado "verdadero" de un sensor dado lecturas ruidosas. El residual (diferencia entre medida y predicción) es la señal de anomalía.

Especificaciones:

1. Clase OnlineKalmanFilter:
   - Modelo de orden 1 (posición + velocidad): estado = [x, ẋ]
   - Matrices del modelo:
     - F (transición): [[1, dt], [0, 1]] — la velocidad persiste, la posición cambia con velocidad
     - H (observación): [[1, 0]] — solo observamos la posición, no la velocidad
     - Q (ruido de proceso): process_noise * [[dt³/3, dt²/2], [dt²/2, dt]] — ruido de aceleración
     - R (ruido de observación): [[observation_noise]] — ruido del sensor
     - P (covarianza inicial): np.eye(2) * 1000 — alta incertidumbre inicial
   
   def __init__(self, process_noise: float = 0.1, observation_noise: float = 1.0)
   
   def initialize(self, initial_value: float, initial_velocity: float = 0.0)
     - Establece el estado inicial x = [initial_value, initial_velocity]
     - Resetea P a identidad * 1000
   
   def predict(self, dt: float) -> Tuple[float, float]:
     - Paso de predicción: x_pred = F @ x, P_pred = F @ P @ F.T + Q
     - Retorna (predicted_value, predicted_uncertainty_sigma)
   
   def update(self, measurement: float) -> KalmanResult:
     - Paso de corrección: calcula ganancia K, actualiza x y P
     - Retorna KalmanResult con:
       - predicted_value: float (la predicción antes de la corrección)
       - corrected_value: float (la estimación tras incorporar la medida)
       - residual: float (measurement - predicted_value)
       - innovation_sigma: float (sqrt de la varianza del residual)
       - normalized_residual: float (residual / innovation_sigma) — para detección de anomalías
       - is_initialized: bool
   
   def step(self, measurement: float, timestamp: datetime) -> KalmanResult:
     - Calcula dt desde el último timestamp
     - Llama predict(dt) + update(measurement)
     - Actualiza el timestamp interno
     - Si no está inicializado: inicializa con measurement y retorna residual=0
     - Auto-reinicio: si |normalized_residual| > 10, reinicializa (evita divergencia permanente)

2. Clase KalmanFilterBank:
   - Gestiona un OnlineKalmanFilter por sensor_id
   - Crea filtros lazy (al primer reading de cada sensor)
   - def process(self, reading: SensorReading) -> KalmanResult
   - def reset_sensor(self, sensor_id: str) — reinicializa el filtro de un sensor específico
   - def get_all_states(self) -> dict — retorna el estado actual de todos los filtros (para serialización)

3. Dataclass KalmanResult:
   - predicted_value, corrected_value, residual: float
   - innovation_sigma, normalized_residual: float
   - is_anomaly: bool (|normalized_residual| > k_sigma=3.0)
   - is_initialized: bool

4. tests/unit/test_kalman.py:
   - Test convergencia: tras 50 steps con señal constante + ruido gaussiano, el residual debe ser < 2*sigma
   - Test detección de step change: un salto abrupto de 50 unidades da normalized_residual > 3
   - Test drift gradual: deriva de 0.01 unidades/step durante 100 steps → residual crece gradualmente
   - Test auto-reinicio: residual > 10*sigma dispara reinicio (is_initialized vuelve a False por 1 step)
   - Test KalmanFilterBank: 3 sensores distintos con filtros independientes
   - Test dt variable: steps con intervalos de tiempo irregulares no causan errores

Implementa con numpy. Añade docstrings con las ecuaciones matemáticas explícitas en LaTeX-style.
```

**Verificación**:
```bash
pytest tests/unit/test_kalman.py -v
# Test de convergencia: el filtro debe converger en < 50 steps sobre señal estacionaria
python -c "
from ml.features.kalman import OnlineKalmanFilter
import numpy as np
kf = OnlineKalmanFilter()
from datetime import datetime, timedelta
t = datetime.now()
for i in range(100):
    r = kf.step(300.0 + np.random.normal(0, 1), t + timedelta(seconds=i))
print(f'Residual tras 100 steps: {r.residual:.3f} (debe ser < 3)')
"
```

---

### T4.2 — Feature Pipeline completo

**Qué hace**: Implementa todas las features del TDD sección 5.5. Transforma las series temporales crudas en el vector de features que consumirán los modelos ML. Es el componente más voluminoso de la Fase 2.

**Prerequisitos**: T4.1 (Kalman) completada. T3.1 (Schema) completada.

**Prompt**:
```
Implementa el Feature Pipeline completo de ReactorGuard según el TDD sección 5.5.

El pipeline transforma readings crudos de sensores en un vector de features numéricas para los modelos ML.

Genera ml/features/pipeline.py con la clase FeaturePipeline:

FEATURES A IMPLEMENTAR (exactamente como el TDD sección 5.5):

Grupo 1 — Rolling statistics (requieren ventana temporal):
  - rolling_mean_{sensor}__60s, rolling_mean_{sensor}__5min, rolling_mean_{sensor}__1h
  - rolling_std_{sensor}__60s, rolling_std_{sensor}__5min
  Implementación: mantener deque con los readings de los últimos N segundos por sensor_id.
  Usar deque con maxlen para eficiencia de memoria.

Grupo 2 — Rate of change:
  - rate_of_change_{sensor}__1s: dX/dt con dt=1s (diferencia con reading anterior)
  - rate_of_change_{sensor}__10s: pendiente lineal de los últimos 10s (regresión lineal de 1 variable)
  Implementación: guardar historial de (timestamp, value) por sensor.

Grupo 3 — Kalman features:
  - kalman_residual_{sensor}: residual normalizado del filtro Kalman (de T4.1)
  - kalman_uncertainty_{sensor}: innovation_sigma del filtro Kalman

Grupo 4 — Correlación cruzada:
  - cross_correlation_{sensor_i}_{sensor_j}__5min: Pearson en ventana de 5 min
  Solo para pares de sensores previamente definidos en correlation_pairs config.
  Implementación: mantener ventanas sincronizadas por pares de sensores.

Grupo 5 — Estadísticas de distribución:
  - variance_ratio_{sensor}: std_60s / std_1h (ratio de volatilidad corto/largo)
  - zero_crossing_rate_{sensor}__60s: cuántas veces la señal cruza su media en 60s

Grupo 6 — Fault indicators:
  - stuck_score_{sensor}__10s: rolling variance en 10s (directamente el valor, sin umbral)
  (el umbral lo aplica el detector, aquí solo calculamos el score)

Grupo 7 — Contexto operacional:
  - hour_of_day: int (0-23)
  - day_of_week: int (0-6)
  - power_level_pct: float (del reading de position sensors — promedio de barras de control)

Clase FeaturePipeline:

  def __init__(self, correlation_pairs: List[Tuple[str,str]] = None, kalman_config: dict = None)
  
  def update(self, reading: SensorReading) -> Optional[FeatureVector]:
    - Actualiza todos los buffers internos con el nuevo reading
    - Si no hay suficiente historial para calcular todas las features (ventana de 1h no llena):
      retorna None para ese sensor hasta tener suficientes datos
    - Retorna FeatureVector con todos los features calculados para ese sensor_id

  def update_batch(self, readings: List[SensorReading]) -> List[FeatureVector]:
    - Procesa un batch de readings en orden cronológico
    - Retorna solo los FeatureVectors completos (descarta los que no tienen historia suficiente)

  def get_feature_names(self) -> List[str]:
    - Retorna los nombres de todas las features en el mismo orden que el vector

Dataclass FeatureVector:
  - sensor_id: str
  - timestamp: datetime
  - features: np.ndarray  (vector numérico listo para modelos ML)
  - feature_names: List[str]
  - reading_id: str  (para trazabilidad)
  - def to_dict(self) -> dict

ADEMÁS:
- ml/features/batch_featurizer.py: procesa un DataFrame entero de readings (para offline/training)
  - Lee parquet de data/raw/tep/, aplica FeaturePipeline fila a fila respetando orden cronológico
  - Guarda resultado en data/processed/features/ como parquet particionado por sensor_id

- tests/unit/test_feature_pipeline.py:
  - Test que rolling_mean con 10 readings = media exacta de esos 10 valores
  - Test que variance_ratio > 1 cuando hay un pico de volatilidad reciente
  - Test que stuck_score ≈ 0 cuando los últimos 10 readings son idénticos
  - Test que zero_crossing_rate = 0 para señal constante, > 0 para señal oscilatoria
  - Test que kalman_residual crece cuando hay un salto abrupto
  - Test update_batch procesa correctamente 1000 readings en < 1 segundo

Añade type hints completos. El pipeline debe ser thread-safe (usa threading.Lock para los buffers compartidos).
```

**Verificación**:
```bash
pytest tests/unit/test_feature_pipeline.py -v
# Test de rendimiento: 1000 readings en < 1 segundo
python ml/features/batch_featurizer.py --input data/raw/tep/ --output data/processed/features/
ls data/processed/features/  # Parquet files por sensor
```

---

### T4.3 — Feast Feature Store: configuración y feature views

**Qué hace**: Configura Feast para servir las features tanto en modo offline (para entrenamiento) como online (para inferencia en tiempo real). Unifica la definición de features entre entrenamiento y producción.

**Prerequisitos**: T4.2 completada. Buckets GCS de Fase 1 disponibles.

**Prompt**:
```
Configura el Feature Store Feast para ReactorGuard.

Feast necesita: una fuente de datos offline (GCS con parquet), una fuente de datos online (Redis o GCS), y la definición de las feature views.

Genera estos archivos:

1. ml/features/feast/feature_store.yaml — configuración principal de Feast:
   project: reactorguard
   registry: gs://reactorguard-data-processed/feast/registry.db
   provider: gcp
   online_store:
     type: redis
     connection_string: "redis://redis-service.reactorguard-ingestion:6379"
     # Nota: Redis se desplegará en K8s (ver T4.4)
   offline_store:
     type: bigquery  # alternativa: file (para desarrollo local)
     dataset: reactorguard_features
   entity_key_serialization_version: 2

2. ml/features/feast/entities.py — entidades Feast:
   - sensor_entity: entity con join_key="sensor_id", value_type=STRING
   - plant_entity: entity con join_key="plant_id", value_type=STRING

3. ml/features/feast/feature_views.py — feature views:
   
   SensorRollingFeaturesView:
   - Entidad: sensor_entity
   - Source: FileSource apuntando a gs://reactorguard-data-processed/features/
   - TTL: 1 hora (features más viejas de 1h no son válidas para inferencia)
   - Features: rolling_mean_60s, rolling_std_60s, rolling_mean_5min, rolling_std_5min,
               rolling_mean_1h, rate_of_change_1s, rate_of_change_10s
   
   SensorKalmanFeaturesView:
   - Entidad: sensor_entity
   - TTL: 30 segundos (el residual Kalman se vuelve stale rápido)
   - Features: kalman_residual, kalman_uncertainty
   
   SensorFaultIndicatorsView:
   - Entidad: sensor_entity
   - TTL: 1 minuto
   - Features: stuck_score_10s, variance_ratio, zero_crossing_rate_60s
   
   PlantContextView:
   - Entidad: plant_entity
   - TTL: 5 minutos
   - Features: hour_of_day, day_of_week, power_level_pct

4. ml/features/feast/feature_service.py — FeatureService que agrupa los views para inferencia:
   - SensorAnalysisFeatureService: agrupa los 4 views anteriores
   - Este es el "contrato" que la API usará para pedir features en tiempo real

5. ml/features/feast_client.py — wrapper de alto nivel para usar Feast:
   class FeastClient:
     def __init__(self, repo_path: str)
     
     def get_online_features(self, sensor_ids: List[str]) -> pd.DataFrame:
       - Llama a feast store.get_online_features() con el FeatureService
       - Retorna DataFrame con una fila por sensor y todas las features como columnas
       - Timeout: 10ms (para cumplir el target de latencia de la API)
     
     def get_offline_features(self, sensor_ids: List[str], start_time: datetime, end_time: datetime) -> pd.DataFrame:
       - Para entrenamiento: recupera historial completo
     
     def materialize_incremental(self):
       - Actualiza el online store con los datos más recientes del offline store

6. k8s/base/ingestion/redis-deployment.yaml:
   - Redis 7 en namespace reactorguard-ingestion (para Feast online store)
   - 1 réplica (Redis standalone, suficiente para dev)
   - PVC de 5Gi para persistencia
   - Resources: request 256m CPU / 512Mi, limit 500m / 1Gi

7. tests/integration/test_feast.py:
   - Test que feast apply no falla (registra las definiciones)
   - Test get_online_features devuelve DataFrame con las features correctas tras materialize
   - Test que TTL se respeta: features materializadas hace >1h retornan NaN para rolling features
   - Benchmark: get_online_features para 10 sensores en < 10ms

Incluye un script ml/features/feast/setup.sh que ejecute feast apply y materialize para el entorno de dev.
```

**Verificación**:
```bash
cd ml/features/feast && feast apply  # Debe registrar todas las feature views sin errores
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
pytest tests/integration/test_feast.py -v
# Benchmark latencia < 10ms para get_online_features
```

---

### T4.4 — Pipeline DVC: stages simulate → featurize → split

**Qué hace**: Define el DAG de DVC que hace reproducible todo el pipeline de datos: desde los parámetros de configuración hasta los splits de train/val/test listos para los modelos. Si cambias cualquier parámetro, `dvc repro` sabe exactamente qué re-ejecutar.

**Prerequisitos**: T3.2 (TEP adaptado), T4.2 (Feature Pipeline) completadas. DVC inicializado con remote GCS.

**Prompt**:
```
Configura el pipeline DVC completo para ReactorGuard.

DVC define el pipeline como un DAG en dvc.yaml. Cada stage tiene cmd, deps (dependencias) y outs (outputs). Si una dep cambia, DVC invalida ese stage y todos los downstream.

Genera estos archivos:

1. dvc.yaml — el DAG completo con estos stages en orden:

   Stage 1: download_tep
   - cmd: python data/generators/tep_downloader.py --output-dir data/raw/tep/
   - params: ninguno (siempre descarga la misma versión)
   - outs: data/raw/tep/ (cached en DVC remote = GCS)
   - frozen: true (no re-ejecutar si ya está descargado)

   Stage 2: adapt_tep
   - cmd: python data/generators/tep_adapter.py --input data/raw/tep/ --output data/raw/tep_adapted/
   - deps: data/raw/tep/, data/generators/tep_adapter.py
   - params: params.yaml#tep.start_time, params.yaml#tep.end_time
   - outs: data/raw/tep_adapted/ (parquet con SensorReadings adaptados)

   Stage 3: featurize
   - cmd: python ml/features/batch_featurizer.py --input data/raw/tep_adapted/ --output data/processed/features/
   - deps: data/raw/tep_adapted/, ml/features/pipeline.py, ml/features/kalman.py
   - params: params.yaml#features.correlation_pairs, params.yaml#features.kalman_config
   - outs: data/processed/features/

   Stage 4: split
   - cmd: python data/generators/train_val_test_split.py
   - deps: data/processed/features/
   - params: params.yaml#split.train_ratio, params.yaml#split.val_ratio, params.yaml#split.seed
   - outs: data/processed/train/, data/processed/val/, data/processed/test/
   - metrics: data/processed/split_stats.json (cuántos samples en cada split, distribución de fault types)

2. params.yaml completo con todos los parámetros:
   tep:
     start_time: "2024-01-01T00:00:00"
     end_time: "2024-12-31T23:59:59"
   features:
     kalman_config:
       process_noise: 0.1
       observation_noise: 1.0
       k_sigma: 3.0
     correlation_pairs: []  # se rellenará en Fase 3
     windows_seconds: [60, 300, 3600]
   split:
     train_ratio: 0.70
     val_ratio: 0.15
     seed: 42
     stratify_by: fault_type  # splits estratificados para balancear clases
   simulation:
     n_samples: 100000
     fault_injection_rate: 0.40
     openmc_seed: 42

3. data/generators/train_val_test_split.py:
   - Lee data/processed/features/ (todos los parquet)
   - Split estratificado por fault_type: 70% train, 15% val, 15% test
   - Respeta orden temporal: train < val < test (no mezcla tiempos)
   - Guarda en data/processed/train/, val/, test/ como parquet
   - Genera split_stats.json: n_samples, fault_type_distribution por split
   - Imprime advertencia si alguna clase tiene < 100 samples en test

4. .dvc/config:
   - Remote: gs://reactorguard-data-processed/dvc-cache
   - autostage: true

5. Script data/generators/verify_pipeline.sh:
   - Ejecuta dvc repro --dry (muestra qué stages se ejecutarían sin ejecutar)
   - Ejecuta dvc repro
   - Verifica que data/processed/train/, val/, test/ existen y tienen parquet files
   - Ejecuta dvc metrics show para mostrar las estadísticas del split
   - Verifica reproducibilidad: ejecuta dvc repro de nuevo y verifica que ningún stage se re-ejecuta (todo cacheado)

6. tests/unit/test_dvc_pipeline.py:
   - Test que dvc.yaml es válido: dvc dag sin errores
   - Test que split estratificado tiene todas las clases de fault_type en cada split
   - Test que la ratio train/val/test es correcta (±1%)
   - Test que los timestamps de train < val < test (no hay data leakage temporal)

Añade comentarios en dvc.yaml explicando por qué el orden temporal importa en el split (data leakage).
```

**Verificación**:
```bash
dvc repro  # Ejecuta el pipeline completo
dvc metrics show  # Muestra estadísticas del split
dvc repro  # Segunda ejecución: "All stages are up-to-date" (reproducibilidad)
pytest tests/unit/test_dvc_pipeline.py -v
ls data/processed/train data/processed/val data/processed/test  # Parquet files
```

---

### T4.5 — Particionamiento GCS y estructura de almacenamiento

**Qué hace**: Establece la estructura exacta de particionamiento en GCS según el TDD sección 4.4 y crea las utilidades para leer/escribir datos respetando esa estructura. Esto es crítico para que Feast y DVC puedan leer los datos eficientemente.

**Prerequisitos**: T1.5 (buckets GCS) de Fase 1, T3.1 (schema) completadas.

**Prompt**:
```
Implementa las utilidades de almacenamiento GCS con el particionamiento del TDD para ReactorGuard.

El TDD sección 4.4 define esta estructura:
gs://reactorguard-data-raw/
  plant=REACTOR-01/year=2024/month=01/day=15/hour=14/
    readings_20240115T14.parquet   <- particionado por location de sensor también

gs://reactorguard-data-raw/simulation/fault_type=stuck/seed=42/
    sim_run_001.parquet

gs://reactorguard-data-processed/
  features/window=60s/
  validation/    <- quality flags del sensor validator

gs://reactorguard-models/
  pinn/v1/
  bnn/v1/
  conformal/v1/

Genera estos archivos:

1. data/storage/gcs_client.py — GCSStorageClient:
   class GCSStorageClient:
     def __init__(self, project_id: str = "reactorguard-platform")
     - Usa google-cloud-storage con Workload Identity (no API key)
     
     def write_sensor_readings(self, readings: List[SensorReading], plant_id: str, timestamp: datetime)
     - Construye el path: gs://reactorguard-data-raw/plant={plant_id}/year={Y}/month={M:02d}/day={D:02d}/hour={H:02d}/
     - Nombre de archivo: readings_{timestamp.strftime('%Y%m%dT%H')}.parquet
     - Convierte a DataFrame y guarda como parquet con PyArrow
     - Añade metadata en el parquet: schema_version, created_at, record_count
     
     def write_simulation_data(self, df: pd.DataFrame, fault_type: str, seed: int, run_id: int)
     - Path: gs://reactorguard-data-raw/simulation/fault_type={fault_type}/seed={seed}/sim_run_{run_id:03d}.parquet
     
     def write_features(self, features_df: pd.DataFrame, window_seconds: int)
     - Path: gs://reactorguard-data-processed/features/window={window_seconds}s/
     - Parquet particionado por sensor_id
     
     def write_model(self, model_bytes: bytes, model_type: str, version: str)
     - Path: gs://reactorguard-models/{model_type}/{version}/model.pkl
     - También escribe metadata.json con: version, created_at, training_metrics
     
     def read_sensor_readings(self, plant_id: str, start_time: datetime, end_time: datetime) -> pd.DataFrame
     - Lista los paths en el rango de tiempo
     - Lee todos los parquets en paralelo (ThreadPoolExecutor, max_workers=4)
     - Retorna DataFrame concatenado y ordenado por timestamp
     
     def read_features(self, window_seconds: int, sensor_ids: List[str] = None) -> pd.DataFrame
     
     def list_simulation_runs(self, fault_type: str = None) -> List[dict]

2. data/storage/local_cache.py — LocalCache para desarrollo sin GCS:
   - Misma interfaz que GCSStorageClient pero guarda en /tmp/reactorguard-cache/
   - Útil para tests y desarrollo local sin acceso a GCP
   - Factory function: get_storage_client(use_local: bool = False) -> StorageClient

3. data/storage/parquet_utils.py:
   - Función sensor_readings_to_parquet(readings: List[SensorReading]) -> bytes
   - Función parquet_to_sensor_readings(parquet_bytes: bytes) -> List[SensorReading]
   - Schema PyArrow explícito que coincide con SensorReading (para compatibilidad entre versiones)
   - Compresión: snappy (buena ratio speed/size para datos numéricos)

4. tests/unit/test_gcs_partitioning.py:
   - Test que write_sensor_readings genera el path correcto para un timestamp dado
   - Test que read_sensor_readings con rango de 2 horas lee los 2 parquets correctos
   - Test roundtrip: escribir 100 readings → leer → mismos datos
   - Test con LocalCache: mismos tests pero sin GCS (para CI sin credenciales GCP)

5. tests/integration/test_gcs_integration.py (requiere GCS real):
   - Escribe 1000 readings en GCS
   - Lee con read_sensor_readings y verifica que llegan todos
   - Verifica que el particionamiento está correcto en el bucket
   - Limpia los datos de prueba al final

Usa GOOGLE_CLOUD_PROJECT env var para el project_id. Añade type hints completos.
```

**Verificación**:
```bash
# Test unitario con LocalCache (sin GCS):
pytest tests/unit/test_gcs_partitioning.py -v
# Test de integración (requiere GCS):
pytest tests/integration/test_gcs_integration.py -v
# Verificar estructura en GCS:
gsutil ls gs://reactorguard-data-raw/  # Debe mostrar plant= partitions
```

---

### T4.6 — Dashboard Grafana: métricas del pipeline de datos

**Qué hace**: Crea el dashboard de Grafana que muestra la salud del pipeline de datos en tiempo real: throughput de Kafka, lag del consumer, tasa de fallos de sensores detectados y latencia del feature pipeline.

**Prerequisitos**: T2.6 (Prometheus) de Fase 1, T3.5 (Consumer con métricas) completadas.

**Prompt**:
```
Crea el dashboard de Grafana para el pipeline de datos de ReactorGuard (Fase 2).

El dashboard debe mostrar en tiempo real la salud de todos los componentes de ingestión.

Genera observability/grafana/dashboards/data-pipeline.json — dashboard JSON de Grafana con estos paneles:

Fila 1 — Throughput del pipeline:
1. Panel "Kafka Ingestion Rate" (Graph): mensajes/segundo publicados en sensor-readings-raw.
   Query: rate(kafka_server_BrokerTopicMetrics_MessagesInPerSec_Count{topic="sensor-readings-raw"}[1m])

2. Panel "Validation Throughput" (Graph): mensajes/segundo procesados por el consumer.
   Query: rate(reactorguard_messages_validated_total[1m])
   Dos series: validated_ok y total (para ver el ratio de fallos)

3. Panel "Consumer Lag" (Stat): lag actual del consumer group sensor-validator-group.
   Query: sum(kafka_consumergroup_lag{consumergroup="sensor-validator-group"})
   Threshold: verde < 100, amarillo < 1000, rojo >= 1000

Fila 2 — Fallos de sensores detectados:
4. Panel "Fault Detection Rate by Type" (Bar chart): fallos detectados por tipo en la última hora.
   Query: increase(reactorguard_sensor_faults_total[1h]) by (fault_type)
   Colores: stuck=naranja, noise_spike=amarillo, bias=rojo, drift=azul, kalman=morado

5. Panel "Sensor Health Status" (Pie chart): distribución de quality flags en los últimos 5 min.
   Query: sum(reactorguard_readings_total[5m]) by (quality)
   Colores: good=verde, suspect=amarillo, bad=rojo, missing=gris

6. Panel "Top Faulty Sensors" (Table): top 10 sensores con más fallos en la última hora.
   Columnas: sensor_id, fault_count, most_common_fault_type

Fila 3 — Latencia y rendimiento:
7. Panel "Validation Latency" (Heatmap): distribución de latencia del validator por minuto.
   Query: reactorguard_validation_latency_seconds_bucket

8. Panel "Feature Pipeline Latency p99" (Graph): latencia del cálculo de features.
   Query: histogram_quantile(0.99, rate(reactorguard_feature_pipeline_latency_seconds_bucket[5m]))
   Línea de alerta en 0.1s (100ms — el umbral del criterio de éxito)

9. Panel "GCS Write Latency" (Stat): latencia media de escritura en GCS.
   Query: rate(reactorguard_gcs_write_seconds_sum[5m]) / rate(reactorguard_gcs_write_seconds_count[5m])

Variables del dashboard:
- plant_id: dropdown con todos los plant_ids detectados en las métricas
- time_range: los estándares de Grafana (last 1h, 6h, 24h)
- refresh: auto, cada 10 segundos

TAMBIÉN genera:
- observability/grafana/dashboards/kustomization.yaml que incluye el dashboard como ConfigMap
- observability/grafana/provisioning/dashboards.yaml — configuración de provisionado de dashboards en Grafana

Asegúrate de que el JSON es válido. Usa un datasource llamado "prometheus" que debe existir en Grafana.
```

**Verificación**:
```bash
kubectl apply -k observability/grafana/
# Abrir Grafana en port-forward:
kubectl port-forward svc/grafana -n reactorguard-observability 3000:3000
# Ir a http://localhost:3000 → Dashboards → Data Pipeline
# Verificar que los paneles cargan sin errores de query
```

---

### T4.7 — Verificación completa de la Fase 2

**Qué hace**: Script de cierre que ejecuta todos los criterios de éxito de la Fase 2 y genera el reporte de estado para poder pasar a la Fase 3.

**Prerequisitos**: Todas las tareas T3.1 a T4.6 completadas.

**Prompt**:
```
Crea el script de verificación y cierre de la Fase 2 de ReactorGuard.

Debe verificar los 4 criterios de éxito del plan de fases y todos los entregables declarados.

Genera infra/scripts/verify_phase2.sh con estas verificaciones:

CRITERIO 1: Throughput Kafka pipeline > 50,000 readings/s sostenidos
- Ejecutar el TEP Streamer en modo fast durante 30 segundos
- Medir throughput real con el benchmark de T3.3
- Comparar contra umbral de 50,000 msg/s
- BLOCKING: sin este throughput no se puede procesar una planta real

CRITERIO 2: Stuck sensor precision > 95%
- Leer el archivo tests/results/validator_metrics_tep.json (generado en T3.6)
- Extraer la precision del StuckValueDetector
- Comparar contra 0.95
- BLOCKING: un stuck sensor no detectado puede esconder un evento real

CRITERIO 3: Feature pipeline latency < 100ms por ventana de 60s
- Ejecutar el benchmark de T4.2: procesar 1000 readings y medir tiempo por feature vector
- Calcular p99 de latencia
- Comparar contra 100ms
- BLOCKING: la latencia total de la API no puede ser > 50ms; features deben ser < 20ms

CRITERIO 4: dvc repro featurize reproducible desde cero sin errores
- Borrar data/processed/features/ y data/processed/train,val,test
- Ejecutar dvc repro --no-cache (fuerza re-ejecución)
- Verificar que termina sin errores
- Verificar que los splits tienen el número esperado de samples
- WARNING (no blocking): si el pipeline tarda > 10 minutos

ENTREGABLES adicionales (WARNING si fallan, no blocking):
- TEP Streamer publicando en Kafka: kubectl get pods -n reactorguard-ingestion → Running
- Sensor validator con tests unitarios: pytest tests/unit/test_sensor_validator.py → 0 failures
- Feature pipeline implementado: pytest tests/unit/test_feature_pipeline.py → 0 failures
- Feast configurado: feast apply sin errores
- GCS con datos particionados: gsutil ls gs://reactorguard-data-processed/features/ → archivos parquet
- Dashboard Grafana cargando: curl http://grafana:3000/api/dashboards/home → HTTP 200

Formato de salida:
=== FASE 2: DATA PIPELINE + SENSOR VALIDATOR ===

CRITERIOS DE ÉXITO (BLOCKING):
✅ Throughput: 73,420 msg/s (> 50,000)
✅ Stuck precision: 0.97 (> 0.95)
✅ Feature latency p99: 42ms (< 100ms)
✅ DVC reproducible: pipeline completo en 8m32s

ENTREGABLES (WARNING):
✅ TEP Streamer: Running (2/2 pods)
✅ Sensor Validator tests: 24/24 passing
✅ Feature Pipeline tests: 18/18 passing
✅ Feast: 4 feature views registradas
✅ GCS partitioned data: 847 parquet files
✅ Grafana dashboard: loaded

4/4 criterios blocking cumplidos
6/6 entregables completos

Estado: LISTA PARA FASE 3 ✅
Informe guardado en docs/phase2_completion_report.md
```

**Verificación**:
```bash
bash infra/scripts/verify_phase2.sh
# Exit code 0 si todos los criterios blocking pasan
# docs/phase2_completion_report.md generado con timestamp
```

---

## Resumen de la Fase 2

| # | Tarea | Semana | Tiempo estimado | Output principal |
|---|-------|--------|-----------------|-----------------|
| T3.1 | Schema SensorReading (Pydantic) | 3 | 2h | data/schemas/sensor_reading.py + tests |
| T3.2 | Descarga y exploración TEP | 3 | 2h | data/raw/tep/ + tep_adapter.py |
| T3.3 | TEP Streamer → Kafka producer | 3 | 3h | tep_streamer.py + deployment K8s |
| T3.4 | Sensor Validator: 5 detectores | 3 | 4h | sensor_validator.py + tests unitarios |
| T3.5 | Kafka Consumer: pipeline validación | 3 | 3h | validation_consumer.py + deployment |
| T3.6 | Evaluación Validator sobre TEP | 3 | 2h | validator_metrics_tep.json (precision > 95%) |
| T4.1 | Filtro de Kalman online | 4 | 2h | ml/features/kalman.py + tests |
| T4.2 | Feature Pipeline completo (10 features) | 4 | 4h | ml/features/pipeline.py + batch_featurizer |
| T4.3 | Feast Feature Store | 4 | 3h | feast config + feature views + Redis |
| T4.4 | Pipeline DVC: simulate→featurize→split | 4 | 2h | dvc.yaml + train/val/test splits |
| T4.5 | Particionamiento GCS | 4 | 2h | data/storage/gcs_client.py |
| T4.6 | Dashboard Grafana: data pipeline | 4 | 1.5h | data-pipeline.json dashboard |
| T4.7 | Verificación cierre Fase 2 | 4 | 0.5h | Reporte 4/4 criterios cumplidos |

**Tiempo total estimado: ~31 horas** distribuidas en 2 semanas.

---

## Dependencias entre tareas

```
T3.1 (Schema)
  ├── T3.2 (TEP download + adapter)
  │     └── T3.3 (Streamer) ──────────────────┐
  │     └── T3.6 (Evaluación en TEP)           │
  │                                            ▼
  ├── T3.4 (Sensor Validator) ─────────── T3.5 (Consumer) → T4.6 (Dashboard)
  │
  └── T4.1 (Kalman)
        └── T4.2 (Feature Pipeline)
              ├── T4.3 (Feast)
              ├── T4.4 (DVC pipeline) ── T4.5 (GCS partitioning)
              └── T4.7 (Verificación Fase 2)
```

---

*ReactorGuard Fase 2 — Plan Detallado con Prompts · v1.0*
