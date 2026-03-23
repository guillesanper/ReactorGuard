"""
tests/integration/test_kafka_connectivity.py
Test de conectividad básica con el cluster Kafka de ReactorGuard.

Verifica que:
  1. El productor puede conectar al bootstrap server y enviar mensajes.
  2. El consumidor recibe exactamente los 100 mensajes enviados.
  3. El contenido de cada mensaje es correcto (schema de sensor reading simplificado).
  4. Imprime la latencia promedio de produce → consume.

Schema del mensaje (compatible con sensor_reading.py TDD sección 4.2):
  {
    "sensor_id":   str,   # e.g. "TC-101"
    "timestamp":   float, # Unix epoch en segundos
    "value":       float, # lectura del sensor
    "unit":        str,   # unidad física ("°C", "bar", "kg/s", ...)
    "quality":     int,   # 0=bad, 1=uncertain, 192=good (estándar OPC-UA)
    "sequence_id": int    # número de secuencia del mensaje
  }

Uso:
  # Desde dentro del pod kafka-client (kubectl exec):
  pip install kafka-python
  python test_kafka_connectivity.py
"""

import json
import logging
import time
import uuid
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = "reactorguard-cluster-kafka-bootstrap.kafka-operator:9092"
TOPIC = "sensor-readings-raw"
NUM_MESSAGES = 100
# Group ID único por ejecución para que siempre se lea desde el inicio
CONSUMER_GROUP = f"connectivity-test-{uuid.uuid4().hex[:8]}"

# Sensores simplificados del reactor TEP
SENSOR_IDS = ["TC-101", "TC-102", "FI-201", "PI-301", "LI-401"]
UNITS = {"TC": "°C", "FI": "kg/s", "PI": "bar", "LI": "m"}


def _make_sensor_reading(sequence_id: int) -> dict[str, Any]:
    """Genera un mensaje de lectura de sensor compatible con el TDD sección 4.2."""
    sensor_id = SENSOR_IDS[sequence_id % len(SENSOR_IDS)]
    prefix = sensor_id.split("-")[0]
    return {
        "sensor_id": sensor_id,
        "timestamp": time.time(),
        "value": round(300.0 + (sequence_id % 50) * 0.5, 3),
        "unit": UNITS.get(prefix, "unknown"),
        "quality": 192,  # OPC-UA: Good
        "sequence_id": sequence_id,
    }


def produce_messages(num_messages: int) -> dict[int, float]:
    """
    Produce `num_messages` mensajes al topic y devuelve un dict
    {sequence_id: timestamp_produce} para calcular la latencia.
    """
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",          # esperar confirmación de todas las ISR réplicas
        retries=3,
        request_timeout_ms=5000,
    )

    produce_times: dict[int, float] = {}

    log.info("Produciendo %d mensajes al topic '%s'...", num_messages, TOPIC)
    for i in range(num_messages):
        message = _make_sensor_reading(i)
        produce_times[i] = time.perf_counter()
        producer.send(
            topic=TOPIC,
            key=message["sensor_id"],
            value=message,
        )

    producer.flush()
    producer.close()
    log.info("Producción completada: %d mensajes enviados.", num_messages)
    return produce_times


def consume_messages(num_messages: int) -> list[dict[str, Any]]:
    """
    Consume hasta `num_messages` mensajes desde el inicio del topic.
    Devuelve la lista de mensajes consumidos con timestamp de recepción.
    """
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        auto_offset_reset="earliest",   # leer desde el principio
        enable_auto_commit=False,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        consumer_timeout_ms=10_000,     # timeout si no hay mensajes nuevos
        max_poll_records=num_messages,
    )

    received: list[dict[str, Any]] = []
    log.info("Consumiendo mensajes (group_id=%s)...", CONSUMER_GROUP)

    for record in consumer:
        payload = record.value
        payload["_received_at"] = time.perf_counter()
        received.append(payload)
        if len(received) >= num_messages:
            break

    consumer.close()
    log.info("Consumidos %d mensajes.", len(received))
    return received


def verify_messages(sent_count: int, received: list[dict[str, Any]]) -> bool:
    """Verifica integridad: cantidad y campos obligatorios."""
    ok = True

    if len(received) != sent_count:
        log.error("❌ Conteo incorrecto: enviados=%d, recibidos=%d", sent_count, len(received))
        ok = False
    else:
        log.info("✅ Conteo correcto: %d/%d mensajes", len(received), sent_count)

    required_fields = {"sensor_id", "timestamp", "value", "unit", "quality", "sequence_id"}
    for i, msg in enumerate(received):
        missing = required_fields - set(msg.keys())
        if missing:
            log.error("❌ Mensaje %d: faltan campos %s", i, missing)
            ok = False
            break

    if ok:
        log.info("✅ Schema correcto: todos los mensajes tienen los campos requeridos.")

    return ok


def report_latency(received: list[dict[str, Any]], produce_times: dict[int, float]) -> float:
    """Calcula y reporta la latencia promedio produce→consume."""
    latencies_ms: list[float] = []

    for msg in received:
        seq = msg.get("sequence_id")
        if seq is not None and seq in produce_times:
            received_at = msg.get("_received_at", 0.0)
            lat_ms = (received_at - produce_times[seq]) * 1000
            latencies_ms.append(lat_ms)

    if not latencies_ms:
        log.warning("No se pudo calcular la latencia (sequence_ids no coinciden).")
        return 0.0

    avg_ms = sum(latencies_ms) / len(latencies_ms)
    log.info("Latencia promedio produce→consume: %.2f ms (n=%d)", avg_ms, len(latencies_ms))
    return avg_ms


def main() -> None:
    log.info("=" * 60)
    log.info("ReactorGuard — Test de conectividad Kafka")
    log.info("Bootstrap: %s", BOOTSTRAP_SERVERS)
    log.info("Topic: %s | Mensajes: %d", TOPIC, NUM_MESSAGES)
    log.info("=" * 60)

    try:
        produce_times = produce_messages(NUM_MESSAGES)
        received = consume_messages(NUM_MESSAGES)
        passed = verify_messages(NUM_MESSAGES, received)
        avg_latency = report_latency(received, produce_times)

        log.info("=" * 60)
        status = "✅ PASSED" if passed else "❌ FAILED"
        log.info("Resultado: %s | Latencia promedio: %.2f ms", status, avg_latency)
        log.info("=" * 60)

        if not passed:
            raise SystemExit(1)

    except KafkaError as exc:
        log.error("❌ Error de conexión con Kafka: %s", exc)
        log.error("Verificar que el cluster está en estado READY:")
        log.error("  kubectl get kafka -n kafka-operator")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
