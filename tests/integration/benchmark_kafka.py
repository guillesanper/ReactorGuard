"""
tests/integration/benchmark_kafka.py
Benchmark de latencia y throughput de Kafka para ReactorGuard.

Criterio de éxito (TDD Fase 1):
  latencia p99 produce→consume < 10ms en red interna del cluster K8s.

Metodología:
  - Produce 10.000 mensajes de ~1KB (tamaño típico sensor reading + metadata).
  - El consumidor los recibe en paralelo (thread daemon).
  - La latencia se mide como: tiempo en que el consumer recibe el mensaje
    menos el tiempo en que el producer llamó a send() (perf_counter).
  - Se usan headers Kafka para pasar el timestamp del producer al consumer
    sin contaminar el payload de negocio.

Salida:
  - Estadísticas por consola (p50, p95, p99, p99.9, throughput).
  - Archivo JSON: tests/results/kafka_benchmark.json
  - Exit code 1 si p99 > 10ms (criterio de fallo de la Fase 1).

Uso:
  # Desde dentro del pod kafka-client:
  pip install kafka-python
  python benchmark_kafka.py [--messages 10000] [--output /results/kafka_benchmark.json]
"""

import argparse
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Configuración ────────────────────────────────────────────────────────────
BOOTSTRAP_SERVERS = "reactorguard-cluster-kafka-bootstrap.kafka-operator:9092"
TOPIC = "sensor-readings-raw"
P99_THRESHOLD_MS = 10.0       # Criterio de éxito Fase 1: p99 < 10ms
MESSAGE_SIZE_BYTES = 1024     # ~1KB por mensaje (sensor reading con metadata)
DEFAULT_NUM_MESSAGES = 10_000
CONSUMER_GROUP = f"benchmark-{uuid.uuid4().hex[:8]}"

# ─── Tipos de datos ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    num_messages: int = 0
    duration_seconds: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    p999_ms: float = 0.0
    throughput_msg_per_sec: float = 0.0
    throughput_mb_per_sec: float = 0.0
    passed: bool = False
    failure_reason: str = ""


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_1kb_payload(sequence_id: int) -> bytes:
    """
    Genera un payload JSON de aproximadamente 1KB.
    El tamaño real de un sensor reading incluye:
      - 52 sensores × ~15 bytes cada uno + metadata OpenTelemetry
    El padding asegura exactamente ~1024 bytes para el benchmark.
    """
    base = {
        "sensor_id": f"TC-{100 + (sequence_id % 52):03d}",
        "timestamp": time.time(),
        "value": round(300.0 + (sequence_id % 100) * 0.1, 6),
        "unit": "°C",
        "quality": 192,
        "sequence_id": sequence_id,
        "reactor_id": "reactor-01",
        "plant_id": "reactorguard-platform",
        "scan_cycle": sequence_id // 52,
        # Metadata de trazabilidad (OpenTelemetry compatible)
        "trace_id": uuid.uuid4().hex,
        "span_id": uuid.uuid4().hex[:16],
    }
    base_bytes = json.dumps(base).encode("utf-8")
    # Padding para alcanzar ~1KB exactos
    padding_needed = max(0, MESSAGE_SIZE_BYTES - len(base_bytes) - 12)
    base["_pad"] = "x" * padding_needed
    return json.dumps(base).encode("utf-8")


def _percentile(data: list[float], pct: float) -> float:
    """Calcula el percentil `pct` (0-100) de una lista ordenada."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = (pct / 100) * (len(sorted_data) - 1)
    lower = int(index)
    upper = min(lower + 1, len(sorted_data) - 1)
    fraction = index - lower
    return sorted_data[lower] * (1 - fraction) + sorted_data[upper] * fraction


# ─── Lógica principal del benchmark ───────────────────────────────────────────

class KafkaBenchmark:
    """
    Benchmark de latencia end-to-end Kafka.

    El producer y el consumer corren concurrentemente:
      - El producer envía mensajes con el timestamp en un header Kafka.
      - El consumer lee los mensajes y calcula la diferencia.
    Esto evita la deriva de reloj entre perf_counter() del producer y del consumer
    porque ambos corren en el mismo proceso Python.
    """

    def __init__(self, num_messages: int = DEFAULT_NUM_MESSAGES):
        self.num_messages = num_messages
        self._latencies: list[float] = []
        self._consume_done = threading.Event()
        self._produce_done = threading.Event()
        self._lock = threading.Lock()

    def _consume_worker(self) -> None:
        """Thread daemon que consume mensajes y registra latencias."""
        consumer = KafkaConsumer(
            TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            group_id=CONSUMER_GROUP,
            auto_offset_reset="latest",   # sólo mensajes nuevos del benchmark
            enable_auto_commit=False,
            consumer_timeout_ms=15_000,
        )

        count = 0
        try:
            for record in consumer:
                received_at = time.perf_counter()

                # El producer embedó el timestamp en los headers
                produce_ts: float | None = None
                for key, value in record.headers:
                    if key == "produce_ts":
                        produce_ts = float(value.decode("utf-8"))
                        break

                if produce_ts is not None:
                    lat_ms = (received_at - produce_ts) * 1000.0
                    with self._lock:
                        self._latencies.append(lat_ms)

                count += 1
                if count >= self.num_messages:
                    break
        finally:
            consumer.close()
            self._consume_done.set()
            log.debug("Consumer terminado: %d mensajes recibidos.", count)

    def run(self) -> BenchmarkResult:
        result = BenchmarkResult(num_messages=self.num_messages)

        # Iniciar el consumer en un thread daemon antes de producir
        consumer_thread = threading.Thread(target=self._consume_worker, daemon=True)
        consumer_thread.start()

        # Breve pausa para que el consumer se suscriba antes de que el producer envíe
        time.sleep(2)

        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: v,   # ya es bytes
            acks="all",
            linger_ms=0,       # sin batching: mide latencia real sin acumulación artificial
            request_timeout_ms=5000,
        )

        log.info("Produciendo %d mensajes de ~%dB...", self.num_messages, MESSAGE_SIZE_BYTES)
        produce_start = time.perf_counter()

        for i in range(self.num_messages):
            payload = _make_1kb_payload(i)
            produce_ts = time.perf_counter()
            producer.send(
                topic=TOPIC,
                value=payload,
                headers=[("produce_ts", str(produce_ts).encode("utf-8"))],
            )

        producer.flush()
        producer.close()

        produce_end = time.perf_counter()
        result.duration_seconds = produce_end - produce_start

        log.info(
            "Producción completada en %.2fs. Esperando al consumer...",
            result.duration_seconds,
        )

        # Esperar a que el consumer termine (máx 20s)
        self._consume_done.wait(timeout=20)
        consumer_thread.join(timeout=5)

        with self._lock:
            result.latencies_ms = list(self._latencies)

        return result


def compute_stats(result: BenchmarkResult) -> BenchmarkResult:
    """Calcula los percentiles y throughput a partir de las latencias recopiladas."""
    lats = result.latencies_ms

    if not lats:
        result.failure_reason = "No se recibieron mensajes."
        result.passed = False
        return result

    result.p50_ms  = _percentile(lats, 50)
    result.p95_ms  = _percentile(lats, 95)
    result.p99_ms  = _percentile(lats, 99)
    result.p999_ms = _percentile(lats, 99.9)

    n = len(lats)
    duration = result.duration_seconds if result.duration_seconds > 0 else 1.0
    result.throughput_msg_per_sec = n / duration
    result.throughput_mb_per_sec  = (n * MESSAGE_SIZE_BYTES) / (duration * 1024 * 1024)

    if result.p99_ms > P99_THRESHOLD_MS:
        result.passed = False
        result.failure_reason = (
            f"p99 ({result.p99_ms:.2f}ms) excede el umbral de {P99_THRESHOLD_MS}ms. "
            "Verificar carga del cluster y configuración de linger_ms."
        )
    else:
        result.passed = True

    return result


def print_report(result: BenchmarkResult) -> None:
    """Imprime el reporte de benchmark formateado."""
    print("\n" + "=" * 62)
    print("  ReactorGuard — Kafka Benchmark Report")
    print("=" * 62)
    print(f"  Mensajes producidos : {result.num_messages:,}")
    print(f"  Mensajes recibidos  : {len(result.latencies_ms):,}")
    print(f"  Duración producción : {result.duration_seconds:.2f}s")
    print()
    print("  Latencia produce → consume:")
    print(f"    p50   : {result.p50_ms:7.2f} ms")
    print(f"    p95   : {result.p95_ms:7.2f} ms")
    p99_label = "✅ < 10ms" if result.p99_ms < P99_THRESHOLD_MS else f"❌ > {P99_THRESHOLD_MS}ms"
    print(f"    p99   : {result.p99_ms:7.2f} ms  {p99_label}")
    print(f"    p99.9 : {result.p999_ms:7.2f} ms")
    print()
    print("  Throughput:")
    print(f"    {result.throughput_msg_per_sec:,.0f} msg/s")
    print(f"    {result.throughput_mb_per_sec:.2f} MB/s")
    print()
    if result.passed:
        print("   RESULTADO: PASSED — p99 dentro del criterio Fase 1 (< 10ms)")
    else:
        print(f"  RESULTADO: FAILED — {result.failure_reason}")
    print("=" * 62 + "\n")


def save_json(result: BenchmarkResult, output_path: str) -> None:
    """Guarda los resultados en JSON para archivar y comparar entre runs."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bootstrap_servers": BOOTSTRAP_SERVERS,
        "topic": TOPIC,
        "num_messages": result.num_messages,
        "messages_received": len(result.latencies_ms),
        "duration_seconds": round(result.duration_seconds, 4),
        "latency_ms": {
            "p50":   round(result.p50_ms, 4),
            "p95":   round(result.p95_ms, 4),
            "p99":   round(result.p99_ms, 4),
            "p99_9": round(result.p999_ms, 4),
        },
        "throughput": {
            "msg_per_sec": round(result.throughput_msg_per_sec, 2),
            "mb_per_sec":  round(result.throughput_mb_per_sec, 4),
        },
        "criteria": {
            "p99_threshold_ms": P99_THRESHOLD_MS,
            "passed": result.passed,
            "failure_reason": result.failure_reason,
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    log.info("Resultados guardados en: %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark de Kafka para ReactorGuard.")
    parser.add_argument("--messages", type=int, default=DEFAULT_NUM_MESSAGES,
                        help=f"Número de mensajes a producir (default: {DEFAULT_NUM_MESSAGES})")
    parser.add_argument("--output", type=str, default="tests/results/kafka_benchmark.json",
                        help="Ruta del archivo JSON de resultados.")
    args = parser.parse_args()

    log.info("Iniciando benchmark de Kafka (n=%d, threshold p99 < %.0fms)...",
             args.messages, P99_THRESHOLD_MS)

    try:
        bench = KafkaBenchmark(num_messages=args.messages)
        result = bench.run()
        result = compute_stats(result)
        print_report(result)
        save_json(result, args.output)

        if not result.passed:
            raise SystemExit(1)

    except KafkaError as exc:
        log.error(" Error de conexión con Kafka: %s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
