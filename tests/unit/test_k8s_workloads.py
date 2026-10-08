"""Comprobaciones estaticas de las cargas de trabajo de streaming (M6).

El streamer, el validador y Redis se despliegan en un cluster que todavia no existe, asi
que el unico sitio donde sus manifiestos se contrastan con el codigo que ejecutan es
este. Cada test enlaza un valor del manifiesto con su fuente de verdad (params.yaml, el
loader de configuracion, Terraform, kafka-cluster.yaml, los scripts) en lugar de repetir
constantes: una constante repetida seria una copia mas que puede quedarse obsoleta.

Lo que NO se comprueba aqui, porque exige un cluster: que el pod arranque, que mTLS
contra Strimzi funcione, que los recursos solicitados basten. Esos puntos quedan
anotados en los propios manifiestos como no medidos.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import pytest
import yaml

from data.generators.tep_params import load_tep_params
from data.generators.tep_streamer_params import (
    DataSource,
    StreamerParams,
    StreamMode,
    apply_env_overrides,
    load_streamer_params,
    validate_against_streaming,
)
from data.storage.storage_params import load_storage_params
from data.streaming import kafka_settings
from data.streaming.streaming_params import StreamingParams, load_streaming_params
from tests.integration import benchmark_kafka as kafka_bench
from tests.integration import test_kafka_connectivity as latency_tool
from tests.integration.benchmark_report import build_report, write_report
from tests.unit.test_k8s_manifests import (
    K8S_BASE,
    K8S_ROOT,
    PARAMS_PATH,
    REPO_ROOT,
    _all_documents,
    _find,
    _read,
    _terraform_local,
)

NAMESPACE = "reactorguard-ingestion"
SYNC_SCRIPT = REPO_ROOT / "infra" / "scripts" / "Sync-KafkaCredentials.ps1"
KAFKA_TESTS_SCRIPT = REPO_ROOT / "tests" / "integration" / "Invoke-KafkaTests.ps1"
BENCHMARK_POD_MANIFEST = K8S_ROOT / "tools" / "kafka-benchmark-pod.yaml"
DOCKERFILE = REPO_ROOT / "api" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
WORKLOAD_KINDS = ("Deployment", "StatefulSet")

# Servicio -> (KafkaUser cuyo certificado monta, replicas esperadas).
_STREAMING_SERVICES: dict[str, tuple[str, int]] = {
    "tep-streamer": ("reactorguard-ingestion", 1),
    "sensor-validator": ("reactorguard-validator", 2),
}
_SECRET_NAME_HINTS = ("PASSWORD", "SECRET", "TOKEN", "API_KEY")

# Ejercita la capa 2 de Sync-KafkaCredentials.ps1 con un certificado autofirmado real.
# Cada linea de salida (CLAVE=valor) la comprueba el test; no se toca ningun cluster.
_PURE_FUNCTIONS_HARNESS = r"""
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
. "__SCRIPT__"

$rsa = [System.Security.Cryptography.RSA]::Create(2048)
$hash = [System.Security.Cryptography.HashAlgorithmName]::SHA256
$padding = [System.Security.Cryptography.RSASignaturePadding]::Pkcs1
$request = [System.Security.Cryptography.X509Certificates.CertificateRequest]::new(
    "CN=t", $rsa, $hash, $padding)
$now = [DateTimeOffset]::UtcNow
$cert = $request.CreateSelfSigned($now, $now.AddDays(100))
$enc = { param($t) [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes($t)) }
$secret = @{ metadata = @{ name = "s" }; data = @{
    "user.crt" = (& $enc $cert.ExportCertificatePem())
    "user.key" = (& $enc $rsa.ExportPkcs8PrivateKeyPem())
    "user.p12" = "AAAA" } } | ConvertTo-Json -Depth 5 | ConvertFrom-Json

$data = Select-SecretData -Secret $secret -Keys $UserSecretKeys
Write-Output ("KEYS=" + ($data.Keys -join ","))
$json = New-SecretManifestJson -Name s -Namespace n -Data $data -SourceNamespace kafka-operator
Write-Output ("TYPE=" + ($json | ConvertFrom-Json).type)
$left = (Get-CertificateNotAfter -Base64 $data["user.crt"]) - [DateTime]::UtcNow
Write-Output ("DAYS=" + [math]::Round($left.TotalDays))
Write-Output ("CA=" + (Get-ClusterCaSecretName -Cluster c))
try { Select-SecretData -Secret $secret -Keys @("absent") } catch { Write-Output "MISSING=ok" }
$bad = '{"metadata":{"name":"x"},"data":{"user.crt":"AAAA"}}' | ConvertFrom-Json
try { Select-SecretData -Secret $bad -Keys @("user.crt") } catch { Write-Output "NOTPEM=ok" }
"""


# Ejercita la capa 2 de Invoke-KafkaTests.ps1 contra un informe real de build_report.
_RUNNER_HARNESS = r"""
$ErrorActionPreference = "Stop"
. "__SCRIPT__"
$report = Read-BenchmarkReport -JsonPath "__REPORT__"
Write-Output ("LINE=" + (Format-ReportLine -Report $report))
Write-Output ("MISSING=" + ($null -eq (Read-BenchmarkReport -JsonPath "nope")))
$one = Get-ThroughputArguments -TopicName t -Duration 5 -OutputPath /w/o.json `
    -ParamsPath /app/params.yaml
Write-Output ("THROUGHPUT=" + ($one -join " "))
$two = Get-LatencyArguments -TopicName t -Count 7 -OutputPath /w/l.json
Write-Output ("LATENCY=" + ($two -join " "))
"""


def _workloads() -> list[dict[str, Any]]:
    """Return every Deployment and StatefulSet under k8s/."""
    return [doc for _, doc in _all_documents() if doc.get("kind") in WORKLOAD_KINDS]


def _benchmark_pod() -> dict[str, Any]:
    """Return the ephemeral benchmark Pod (k8s/tools, outside the kustomize base)."""
    return _find("Pod", "kafka-benchmark", NAMESPACE)


def _hardened_workloads() -> list[dict[str, Any]]:
    """Return the workloads that must meet the hardening baseline.

    Incluye el pod de benchmark. Queda fuera kafka-client-pod.yaml (imagen de Confluent,
    pod de depuracion que su propia cabecera declara solo para desarrollo).
    """
    return [*_workloads(), _benchmark_pod()]


def _pod_spec(workload: dict[str, Any]) -> dict[str, Any]:
    """Return the pod spec of a workload (a Pod carries it directly under spec)."""
    if workload["kind"] == "Pod":
        return dict(workload["spec"])
    return dict(workload["spec"]["template"]["spec"])


def _workload_id(workload: dict[str, Any]) -> str:
    """Return a readable pytest id for a workload."""
    return f"{workload['kind']}/{workload['metadata']['name']}"


def _service(name: str) -> dict[str, Any]:
    """Return the Deployment of a streaming service."""
    return _find("Deployment", name, NAMESPACE)


def _container(workload: dict[str, Any]) -> dict[str, Any]:
    """Return the single container of a workload.

    Args:
        workload: A Deployment or StatefulSet.

    Returns:
        The container.

    Raises:
        AssertionError: If the pod does not run exactly one container.
    """
    containers = _pod_spec(workload)["containers"]
    assert len(containers) == 1, f"{_workload_id(workload)}: se esperaba un solo contenedor"
    return dict(containers[0])


def _env(container: dict[str, Any]) -> dict[str, str]:
    """Return the literal environment variables of a container (valueFrom ones excluded)."""
    return {item["name"]: item["value"] for item in container.get("env", []) if "value" in item}


def _secret_behind(workload: dict[str, Any], file_path: str) -> tuple[str, str]:
    """Resolve which Secret and key provides a file mounted into the container.

    Args:
        workload: The Deployment that mounts the file.
        file_path: Absolute path inside the container.

    Returns:
        The (secret name, secret key) that the kubelet projects at that path.
    """
    container = _container(workload)
    path = PurePosixPath(file_path)
    mounts = {mount["mountPath"]: mount["name"] for mount in container["volumeMounts"]}
    volume_name = mounts[str(path.parent)]
    volume = next(v for v in _pod_spec(workload)["volumes"] if v["name"] == volume_name)
    item = next(i for i in volume["secret"]["items"] if i["path"] == path.name)
    return str(volume["secret"]["secretName"]), str(item["key"])


def _tls_listener() -> dict[str, Any]:
    """Return the mTLS listener of the Kafka cluster."""
    cluster = _find("Kafka", "reactorguard-cluster")
    listeners = [
        listener
        for listener in cluster["spec"]["kafka"]["listeners"]
        if listener.get("tls") and listener["authentication"]["type"] == "tls"
    ]
    assert len(listeners) == 1, "Se esperaba un unico listener mTLS en el cluster Kafka"
    return dict(listeners[0])


class TestWorkloadHardening:
    """Every workload runs unprivileged, read-only and with bounded resources."""

    @pytest.mark.parametrize("workload", _hardened_workloads(), ids=_workload_id)
    def test_pod_security_context(self, workload: dict[str, Any]) -> None:
        """Pod no-root con el perfil seccomp por defecto y sin token de la API montado."""
        pod = _pod_spec(workload)
        context = pod["securityContext"]
        assert context["runAsNonRoot"] is True
        assert context["seccompProfile"] == {"type": "RuntimeDefault"}
        assert pod["automountServiceAccountToken"] is False

    @pytest.mark.parametrize("workload", _hardened_workloads(), ids=_workload_id)
    def test_container_security_context(self, workload: dict[str, Any]) -> None:
        """Sin escalada de privilegios, raiz de solo lectura y sin capabilities."""
        context = _container(workload)["securityContext"]
        assert context["allowPrivilegeEscalation"] is False
        assert context["readOnlyRootFilesystem"] is True
        assert context["capabilities"] == {"drop": ["ALL"]}

    @pytest.mark.parametrize("workload", _hardened_workloads(), ids=_workload_id)
    def test_resources_are_bounded(self, workload: dict[str, Any]) -> None:
        """requests y limits de CPU y memoria, o un pod sin limite desaloja a sus vecinos."""
        resources = _container(workload)["resources"]
        for section in ("requests", "limits"):
            assert {"cpu", "memory"} <= set(resources[section]), (
                f"{_workload_id(workload)}: falta cpu o memory en resources.{section}"
            )

    @pytest.mark.parametrize("workload", _workloads(), ids=_workload_id)
    def test_probes_are_declared(self, workload: dict[str, Any]) -> None:
        """Sin sondas, un proceso colgado sigue recibiendo trafico y no se reinicia."""
        container = _container(workload)
        assert "livenessProbe" in container
        assert "readinessProbe" in container

    @pytest.mark.parametrize("workload", _hardened_workloads(), ids=_workload_id)
    def test_no_literal_credentials_in_the_environment(self, workload: dict[str, Any]) -> None:
        """Una variable con aspecto de credencial solo puede venir de un Secret."""
        for item in _container(workload).get("env", []):
            if any(hint in item["name"] for hint in _SECRET_NAME_HINTS):
                assert "value" not in item and "secretKeyRef" in item.get("valueFrom", {}), (
                    f"{_workload_id(workload)}: {item['name']} lleva un valor literal"
                )

    def test_no_secret_is_committed(self) -> None:
        """Los Secrets se crean fuera de banda o con Sync-KafkaCredentials.ps1, nunca aqui."""
        secrets = [
            doc["metadata"]["name"] for _, doc in _all_documents() if doc.get("kind") == "Secret"
        ]
        assert not secrets, f"Hay Secrets versionados en k8s/: {secrets}"


class TestStreamingWorkloads:
    """The streamer and the validator agree with the code they run."""

    @pytest.mark.parametrize(("name", "expected"), sorted(_STREAMING_SERVICES.items()))
    def test_replicas_and_strategy(self, name: str, expected: tuple[str, int]) -> None:
        """Un streamer (Recreate) y dos validadores. Dos streamers duplicarian cada lectura."""
        deployment = _service(name)
        assert deployment["spec"]["replicas"] == expected[1]
        if name == "tep-streamer":
            assert deployment["spec"]["strategy"]["type"] == "Recreate"

    @pytest.mark.parametrize("name", sorted(_STREAMING_SERVICES))
    def test_entrypoint_module_exists(self, name: str) -> None:
        """`python -m <modulo>` debe apuntar a un modulo real del repositorio."""
        command = _container(_service(name))["command"]
        assert command[:2] == ["python", "-m"]
        assert importlib.util.find_spec(command[2]) is not None, f"No existe el modulo {command[2]}"

    @pytest.mark.parametrize("name", sorted(_STREAMING_SERVICES))
    def test_params_file_comes_from_the_config_map(self, name: str) -> None:
        """`--params` debe leer el fichero que el ConfigMap proyecta en el volumen."""
        workload = _service(name)
        container = _container(workload)
        args = container["args"]
        assert args[0] == "--params"
        mounts = {mount["name"]: mount["mountPath"] for mount in container["volumeMounts"]}
        volume = next(v for v in _pod_spec(workload)["volumes"] if "configMap" in v)
        config_map = _find("ConfigMap", volume["configMap"]["name"], NAMESPACE)
        expected = f"{mounts[volume['name']]}/params.yaml"
        assert args[1] == expected
        assert "params.yaml" in config_map["data"]

    @pytest.mark.parametrize("name", sorted(_STREAMING_SERVICES))
    def test_ports_and_probes_follow_params(self, name: str) -> None:
        """Probes sobre el puerto de salud y metricas sobre el de params.yaml."""
        streaming = load_streaming_params(PARAMS_PATH)
        container = _container(_service(name))
        ports = {port["name"]: port["containerPort"] for port in container["ports"]}
        assert ports == {"health": streaming.health_port, "metrics": streaming.metrics_port}
        assert container["livenessProbe"]["httpGet"] == {"path": "/health", "port": "health"}
        assert container["readinessProbe"]["httpGet"] == {"path": "/ready", "port": "health"}

    def test_grace_periods_cover_the_worst_batch(self) -> None:
        """terminationGracePeriodSeconds debe superar el peor lote y el cierre ordenado.

        Si recalibras los timeouts de params.yaml, este test te obliga a recalibrar el
        manifiesto: un SIGKILL a mitad de lote deja un commit a medias y reprocesa.
        """
        streaming = load_streaming_params(PARAMS_PATH)
        streamer = load_streamer_params(PARAMS_PATH)
        close_s = streaming.close_timeout_ms / 1000.0
        validator_worst = streaming.worst_case_batch_seconds + close_s
        streamer_worst = streamer.wait_slice_s + streaming.flush_timeout_s + close_s
        assert _pod_spec(_service("sensor-validator"))["terminationGracePeriodSeconds"] > (
            validator_worst
        )
        assert _pod_spec(_service("tep-streamer"))["terminationGracePeriodSeconds"] > (
            streamer_worst
        )

    @pytest.mark.parametrize("name", sorted(_STREAMING_SERVICES))
    def test_service_account_exists_and_streamer_reaches_gcs(self, name: str) -> None:
        """La KSA existe; la que lee GCS debe llevar Workload Identity."""
        account = _pod_spec(_service(name))["serviceAccountName"]
        service_account = _find("ServiceAccount", account, NAMESPACE)
        annotations = service_account["metadata"].get("annotations", {})
        assert "iam.gke.io/gcp-service-account" in annotations, (
            f"La KSA {account} no tiene Workload Identity: {name} no podria leer GCS"
        )

    @pytest.mark.parametrize("name", sorted(_STREAMING_SERVICES))
    def test_image_lives_in_the_terraform_project(self, name: str) -> None:
        """Una imagen de otro proyecto da ImagePullBackOff sin pista en el manifiesto."""
        image = _container(_service(name))["image"]
        assert image.startswith(f"gcr.io/{_terraform_local('project_id')}/reactorguard-api"), image

    @pytest.mark.parametrize(("name", "expected"), sorted(_STREAMING_SERVICES.items()))
    def test_mtls_wiring(self, name: str, expected: tuple[str, int]) -> None:
        """Bootstrap, protocolo y ficheros TLS deben llevar al listener y al KafkaUser correctos."""
        user = expected[0]
        workload = _service(name)
        env = _env(_container(workload))
        cluster = _find("Kafka", "reactorguard-cluster")
        listener = _tls_listener()

        host, _, port = env["KAFKA_BOOTSTRAP"].rpartition(":")
        assert int(port) == listener["port"], "El bootstrap no apunta al listener mTLS"
        # Con check_hostname el nombre debe estar en los SAN de los brokers: el servicio
        # de bootstrap en su forma <servicio>.<namespace>.svc.
        namespace = cluster["metadata"]["namespace"]
        assert host == f"{cluster['metadata']['name']}-kafka-bootstrap.{namespace}.svc"
        assert env["KAFKA_SECURITY_PROTOCOL"] == "SSL"

        assert _secret_behind(workload, env["KAFKA_SSL_CERTFILE"]) == (user, "user.crt")
        assert _secret_behind(workload, env["KAFKA_SSL_KEYFILE"]) == (user, "user.key")
        ca_secret = f"{cluster['metadata']['name']}-cluster-ca-cert"
        assert _secret_behind(workload, env["KAFKA_SSL_CAFILE"]) == (ca_secret, "ca.crt")
        _find("KafkaUser", user, namespace)

    @pytest.mark.parametrize("name", sorted(_STREAMING_SERVICES))
    def test_kafka_env_names_exist_in_the_settings_loader(self, name: str) -> None:
        """Una variable KAFKA_* mal escrita se ignora en silencio y cambia el comportamiento."""
        known = {
            value
            for attr, value in vars(kafka_settings).items()
            if attr.startswith("ENV_") and isinstance(value, str)
        }
        used = {key for key in _env(_container(_service(name))) if key.startswith("KAFKA_")}
        assert used <= known, f"Variables KAFKA_* desconocidas: {sorted(used - known)}"

    def test_streamer_overrides_are_accepted_by_the_code(self) -> None:
        """Los valores de STREAM_MODE, SPEED_MULTIPLIER y DATA_SOURCE los acepta el loader."""
        env = _env(_container(_service("tep-streamer")))
        params = apply_env_overrides(load_streamer_params(PARAMS_PATH), env)
        assert params.mode is StreamMode.REALTIME
        assert params.data_source is DataSource.GCS
        assert params.speed_multiplier > 0

    def test_validator_pdb_protects_the_consumer_group(self) -> None:
        """minAvailable 1 sobre las etiquetas del pod y menos que las replicas."""
        pdb = _find("PodDisruptionBudget", "sensor-validator", NAMESPACE)
        deployment = _service("sensor-validator")
        assert pdb["spec"]["minAvailable"] == 1
        selector = pdb["spec"]["selector"]["matchLabels"]
        pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
        assert selector.items() <= pod_labels.items()
        assert deployment["spec"]["replicas"] > pdb["spec"]["minAvailable"]

    def test_consumer_group_replicas_do_not_exceed_partitions(self) -> None:
        """Mas replicas que particiones dejaria consumidores ociosos (el tope util)."""
        topic = _find("KafkaTopic", load_streaming_params(PARAMS_PATH).raw_topic)
        assert _service("sensor-validator")["spec"]["replicas"] <= topic["spec"]["partitions"]


class TestParamsConfigMap:
    """The cluster's params.yaml is a faithful copy of the repository's."""

    # Diferencias de cluster deliberadas con params.yaml (ver params-configmap.yaml).
    _CLUSTER_OVERRIDES: dict[tuple[str, str], Any] = {("streamer", "loop"): True}

    @staticmethod
    def _cluster_params() -> dict[str, Any]:
        """Return the params.yaml document embedded in the ConfigMap."""
        config_map = _find("ConfigMap", "reactorguard-params", NAMESPACE)
        return dict(yaml.safe_load(config_map["data"]["params.yaml"]))

    @pytest.mark.parametrize("section", ["tep", "streaming", "storage", "streamer"])
    def test_section_matches_the_repository(self, section: str) -> None:
        """Salvo los overrides declarados, la seccion es identica a la de params.yaml."""
        repo = yaml.safe_load(_read(PARAMS_PATH))[section]
        expected = dict(repo)
        for (override_section, key), value in self._CLUSTER_OVERRIDES.items():
            if override_section == section:
                expected[key] = value
        assert self._cluster_params()[section] == expected, (
            f"La seccion '{section}' del ConfigMap difiere de params.yaml. Copia el valor "
            "cambiado, o declara el override de cluster en _CLUSTER_OVERRIDES."
        )

    def test_only_the_sections_the_services_read(self) -> None:
        """Una seccion de mas (training, simulation) no pinta nada en el cluster."""
        assert set(self._cluster_params()) == {"tep", "streaming", "storage", "streamer"}

    def test_overrides_actually_differ_from_the_repository(self) -> None:
        """Un override que ya coincide con params.yaml es ruido que oculta la divergencia real."""
        repo = yaml.safe_load(_read(PARAMS_PATH))
        for (section, key), value in self._CLUSTER_OVERRIDES.items():
            assert repo[section][key] != value, f"El override {section}.{key} ya no cambia nada"

    def test_real_loaders_accept_the_embedded_file(self, tmp_path: Path) -> None:
        """Los cuatro loaders y la validacion cruzada deben aceptar el fichero tal cual."""
        path = tmp_path / "params.yaml"
        path.write_text(_find("ConfigMap", "reactorguard-params", NAMESPACE)["data"]["params.yaml"])
        streaming: StreamingParams = load_streaming_params(path)
        streamer: StreamerParams = load_streamer_params(path)
        validate_against_streaming(streamer, streaming)
        assert streamer.loop is True
        assert load_tep_params(path).sample_interval_minutes == 3
        assert load_storage_params(path).raw_bucket.startswith("reactorguard-data-raw-")

    def test_span_table_ships_in_the_image(self) -> None:
        """tep.spans_path es relativo al directorio de trabajo: debe estar en la imagen."""
        spans_path = load_tep_params(PARAMS_PATH).spans_path
        assert (REPO_ROOT / spans_path).is_file()
        assert f"COPY {spans_path.parent.as_posix()}/" in _read(DOCKERFILE)


class TestRedis:
    """The Feast online store: authenticated, persistent and reachable by label."""

    @staticmethod
    def _stateful_set() -> dict[str, Any]:
        """Return the Redis StatefulSet."""
        return _find("StatefulSet", "redis", NAMESPACE)

    def test_password_comes_from_a_secret_and_never_from_the_arguments(self) -> None:
        """La contrasena en los argumentos del proceso se veria en /proc/<pid>/cmdline."""
        container = _container(self._stateful_set())
        sources = {
            item["name"]: item["valueFrom"]["secretKeyRef"]
            for item in container["env"]
            if "secretKeyRef" in item.get("valueFrom", {})
        }
        assert sources["REDIS_PASSWORD"] == {"name": "redis-credentials", "key": "password"}
        assert sources["REDISCLI_AUTH"] == {"name": "redis-credentials", "key": "password"}
        arguments = " ".join(container["args"])
        assert "--requirepass" not in arguments
        assert "$REDIS_PASSWORD" in arguments

    def test_config_includes_a_file_the_container_can_write(self) -> None:
        """La raiz es de solo lectura: el fichero de auth debe caer en un emptyDir montado."""
        stateful_set = self._stateful_set()
        config = _find("ConfigMap", "redis-config", NAMESPACE)["data"]["redis.conf"]
        included = re.search(r"^include (\S+)$", config, re.MULTILINE)
        assert included, "redis.conf no incluye el fichero de autenticacion"
        directory = str(PurePosixPath(included.group(1)).parent)
        container = _container(stateful_set)
        mounts = {mount["mountPath"]: mount["name"] for mount in container["volumeMounts"]}
        volumes = _pod_spec(stateful_set)["volumes"]
        volume = next(v for v in volumes if v["name"] == mounts[directory])
        assert "emptyDir" in volume
        assert f"> {included.group(1)}" in " ".join(container["args"])
        assert "noeviction" in config

    def test_data_is_persistent(self) -> None:
        """Un PVC en /data: sin el, cada reinicio vacia el online store."""
        stateful_set = self._stateful_set()
        claims = stateful_set["spec"]["volumeClaimTemplates"]
        assert [claim["metadata"]["name"] for claim in claims] == ["redis-data"]
        mounts = {m["name"]: m["mountPath"] for m in _container(stateful_set)["volumeMounts"]}
        assert mounts["redis-data"] == "/data"

    def test_service_selects_the_pods_the_network_policies_target(self) -> None:
        """La etiqueta app.kubernetes.io/name=redis es la que usan las NetworkPolicies."""
        stateful_set = self._stateful_set()
        service = _find("Service", "redis-service", NAMESPACE)
        labels = stateful_set["spec"]["template"]["metadata"]["labels"]
        assert service["spec"]["selector"].items() <= labels.items()
        assert labels["app.kubernetes.io/name"] == "redis"
        assert stateful_set["spec"]["serviceName"] == service["metadata"]["name"]
        assert [port["port"] for port in service["spec"]["ports"]] == [6379]


class TestSyncKafkaCredentialsScript:
    """Sync-KafkaCredentials.ps1 copies exactly what the workloads mount."""

    @staticmethod
    def _default(pattern: str) -> str:
        """Return the first capture group of a pattern over the script text."""
        match = re.search(pattern, _read(SYNC_SCRIPT))
        assert match, f"Sync-KafkaCredentials.ps1: no encuentro {pattern!r}"
        return match.group(1)

    @classmethod
    def _array(cls, variable: str) -> set[str]:
        """Return the quoted strings of a `$variable = @(...)` assignment in the script."""
        body = cls._default(rf"\${variable}\s*=\s*@\(([^)]*)\)")
        return set(re.findall(r'"([^"]+)"', body))

    def test_namespaces_and_cluster_match_the_manifests(self) -> None:
        """Origen = namespace del Kafka, destino = el de los Deployments, mismo cluster."""
        cluster = _find("Kafka", "reactorguard-cluster")
        assert self._default(r'\$SourceNamespace\s*=\s*"([^"]+)"') == (
            cluster["metadata"]["namespace"]
        )
        assert self._default(r'\$TargetNamespace\s*=\s*"([^"]+)"') == NAMESPACE
        assert self._default(r'\$ClusterName\s*=\s*"([^"]+)"') == cluster["metadata"]["name"]

    def test_copies_every_user_the_workloads_mount(self) -> None:
        """Un KafkaUser que se monta y no se copia deja el pod en ContainerCreating."""
        copied = self._array("KafkaUsers")
        mounted = {user for user, _ in _STREAMING_SERVICES.values()}
        benchmark = next(
            volume["secret"]["secretName"]
            for volume in _pod_spec(_benchmark_pod())["volumes"]
            if volume["name"] == "kafka-user"
        )
        assert copied == mounted | {benchmark}

    def test_copies_the_keys_the_workloads_read(self) -> None:
        """Las claves copiadas deben cubrir lo que proyectan los volumenes de los Deployments."""
        user_keys = self._array("UserSecretKeys")
        ca_keys = self._array("CaSecretKeys")
        workloads = [_service(name) for name in _STREAMING_SERVICES] + [_benchmark_pod()]
        for workload in workloads:
            name = _workload_id(workload)
            for volume in _pod_spec(workload)["volumes"]:
                if "secret" not in volume:
                    continue
                keys = {item["key"] for item in volume["secret"]["items"]}
                is_ca = volume["secret"]["secretName"].endswith("-ca-cert")
                wanted = ca_keys if is_ca else user_keys
                assert keys <= wanted, f"{name}: {volume['name']} lee claves que no se copian"

    def test_follows_the_script_conventions(self) -> None:
        """PowerShell 7, cuatro capas y solo ASCII (sin emojis ni caracteres especiales)."""
        text = _read(SYNC_SCRIPT)
        assert text.startswith("#Requires -Version 7.0")
        for layer in ("CAPA 1", "CAPA 2", "CAPA 3", "CAPA 4"):
            assert layer in text, f"Falta la {layer}"
        assert text.isascii()

    @pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell 7 no esta instalado")
    def test_pure_functions_behave(self, tmp_path: Path) -> None:
        """Capa 2 ejecutada de verdad: filtra claves, valida PEM y lee la caducidad."""
        harness = tmp_path / "harness.ps1"
        harness.write_text(
            _PURE_FUNCTIONS_HARNESS.replace("__SCRIPT__", str(SYNC_SCRIPT)), encoding="utf-8"
        )
        result = subprocess.run(  # noqa: S603  (pwsh resuelto con shutil.which, script propio)
            [str(shutil.which("pwsh")), "-NoProfile", "-File", str(harness)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        lines = set(result.stdout.split())
        assert {"KEYS=user.crt,user.key", "TYPE=Opaque", "DAYS=100"} <= lines
        assert {"CA=c-cluster-ca-cert", "MISSING=ok", "NOTPEM=ok"} <= lines


class TestDockerBuildContext:
    """The image carries what the services read at runtime, and nothing heavy."""

    @staticmethod
    def _copy_sources() -> list[tuple[int, str]]:
        """Return (line number, source) for every COPY from the build context."""
        sources: list[tuple[int, str]] = []
        for number, line in enumerate(_read(DOCKERFILE).splitlines(), start=1):
            if not line.startswith("COPY ") or "--from" in line:
                continue
            *origins, _destination = line.split()[1:]
            sources.extend((number, origin) for origin in origins)
        return sources

    def test_runtime_configuration_is_copied(self) -> None:
        """params.yaml y configs/ se leen por ruta relativa; sin ellos el validador no arranca."""
        copied = {source for _, source in self._copy_sources()}
        assert {"params.yaml", "configs/"} <= copied

    def test_builder_has_the_readme_hatchling_requires(self) -> None:
        """pyproject declara `readme = README.md`: sin el, hatchling aborta el build."""
        assert 'readme = "README.md"' in _read(REPO_ROOT / "pyproject.toml")
        assert "README.md" in {source for _, source in self._copy_sources()}

    def test_every_copy_source_exists_and_is_allowed_by_dockerignore(self) -> None:
        """El .dockerignore es una lista de permitidos: lo que se copia debe readmitirse."""
        ignore = _read(DOCKERIGNORE).splitlines()
        allowed = {line[1:].strip() for line in ignore if line.startswith("!")}
        for number, source in self._copy_sources():
            assert (REPO_ROOT / source).exists(), f"Dockerfile:{number}: {source} no existe"
            assert source in allowed, f"Dockerfile:{number}: {source} no esta readmitido"

    def test_dockerignore_keeps_the_datasets_out_of_the_image(self) -> None:
        """data/processed son cientos de MB de parquet que `COPY data/` arrastraria."""
        ignore = _read(DOCKERIGNORE).splitlines()
        assert "*" in ignore
        for excluded in ("data/raw", "data/processed"):
            assert excluded in ignore, f".dockerignore no excluye {excluded}"
        assert ignore.index("data/processed") > ignore.index("!data/")

    def test_base_kustomization_lists_the_workloads(self) -> None:
        """Un Deployment fuera del kustomization nunca se despliega con `apply -k`."""
        resources = yaml.safe_load(_read(K8S_BASE / "kustomization.yaml"))["resources"]
        for manifest in (
            "ingestion/tep-streamer-deployment.yaml",
            "ingestion/sensor-validator-deployment.yaml",
            "ingestion/redis.yaml",
            "ingestion/params-configmap.yaml",
        ):
            assert manifest in resources


class TestBenchmarkRunner:
    """Invoke-KafkaTests.ps1, the benchmark Pod and the code they run agree."""

    @staticmethod
    def _default(variable: str) -> str:
        """Return the first quoted value assigned to a script variable."""
        match = re.search(rf'\${variable}\s*=\s*"([^"]+)"', _read(KAFKA_TESTS_SCRIPT))
        assert match, f"Invoke-KafkaTests.ps1 no declara ${variable}"
        return match.group(1)

    @staticmethod
    def _array(variable: str) -> list[str]:
        """Return the quoted strings of a `$variable = @(...)` assignment."""
        match = re.search(rf"\${variable}\s*=\s*@\(([^)]*)\)", _read(KAFKA_TESTS_SCRIPT))
        assert match, f"Invoke-KafkaTests.ps1 no declara ${variable}"
        return re.findall(r'"([^"]+)"', match.group(1))

    def test_pod_and_namespace_match_the_manifest(self) -> None:
        """El script debe crear, esperar y borrar el pod que el manifiesto declara."""
        pod = _benchmark_pod()
        assert self._default("PodName") == pod["metadata"]["name"]
        namespace = re.search(
            r'\[string\]\$Namespace\s*=\s*"([^"]+)"', _read(KAFKA_TESTS_SCRIPT)
        )
        assert namespace
        assert namespace.group(1) == pod["metadata"]["namespace"]
        assert self._default("RemoteWorkDir") == _container(pod)["workingDir"]
        assert BENCHMARK_POD_MANIFEST.is_file()
        resources = yaml.safe_load(_read(K8S_BASE / "kustomization.yaml"))["resources"]
        assert not any("kafka-benchmark-pod" in entry for entry in resources), (
            "El pod de benchmark es una herramienta: no debe desplegarse con apply -k"
        )

    def test_pod_has_a_deadline_so_a_forgotten_pod_cleans_itself(self) -> None:
        """Si el script muere sin limpiar, activeDeadlineSeconds evita un pod olvidado."""
        spec = _pod_spec(_benchmark_pod())
        assert 0 < spec["activeDeadlineSeconds"] <= 3600
        assert spec["restartPolicy"] == "Never"

    def test_pod_mounts_the_secrets_the_script_checks(self) -> None:
        """Los Secrets que el script exige existentes son los que el pod monta."""
        mounted = {
            volume["secret"]["secretName"]
            for volume in _pod_spec(_benchmark_pod())["volumes"]
            if "secret" in volume
        }
        assert set(self._array("CredentialSecrets")) == mounted

    def test_pod_uses_the_benchmark_identity_over_mtls(self) -> None:
        """Bootstrap mTLS y certificado del KafkaUser de benchmark, no el de produccion."""
        pod = _benchmark_pod()
        env = _env(_container(pod))
        cluster = _find("Kafka", "reactorguard-cluster")
        assert int(env["KAFKA_BOOTSTRAP"].rpartition(":")[2]) == _tls_listener()["port"]
        assert env["KAFKA_SECURITY_PROTOCOL"] == "SSL"
        user = "reactorguard-benchmark"
        assert _secret_behind(pod, env["KAFKA_SSL_CERTFILE"]) == (user, "user.crt")
        assert _secret_behind(pod, env["KAFKA_SSL_KEYFILE"]) == (user, "user.key")
        ca_secret = f"{cluster['metadata']['name']}-cluster-ca-cert"
        assert _secret_behind(pod, env["KAFKA_SSL_CAFILE"]) == (ca_secret, "ca.crt")
        _find("KafkaUser", user)

    def test_topic_is_the_declared_benchmark_topic(self) -> None:
        """El topic por defecto del script y del test de latencia es un KafkaTopic real."""
        topic = self._default("Topic")
        _find("KafkaTopic", topic)
        assert latency_tool.DEFAULT_TOPIC == topic
        assert topic != load_streaming_params(PARAMS_PATH).raw_topic

    def test_staged_files_exist_and_are_closed_under_imports(self) -> None:
        """Lo que se copia debe existir y bastar: cada import de tests.* debe ir en la lista."""
        staged = self._array("StagedFiles")
        for relative in staged:
            assert (REPO_ROOT / relative).is_file(), f"No existe {relative}"
        modules = {
            relative[: -len(".py")].replace("/", ".")
            for relative in staged
            if not relative.endswith("__init__.py")
        }
        for relative in staged:
            source = _read(REPO_ROOT / relative)
            for imported in re.findall(r"^from (tests\.[\w.]+) import", source, re.MULTILINE):
                assert imported in modules, f"{relative} importa {imported}, que no se copia"

    def test_remote_commands_run_real_modules_and_inputs(self) -> None:
        """Los modulos que se ejecutan existen; el parquet y params.yaml estan donde se leen."""
        text = _read(KAFKA_TESTS_SCRIPT)
        for module in re.findall(r'"(tests\.integration\.\w+)"', text):
            assert importlib.util.find_spec(module) is not None, f"No existe {module}"
        assert self._default("PayloadRelative") == kafka_bench._PAYLOAD_PARTITION.as_posix()
        dockerfile = _read(DOCKERFILE)
        assert "WORKDIR /app" in dockerfile
        assert "COPY params.yaml ./params.yaml" in dockerfile
        assert self._default("RemoteParams") == "/app/params.yaml"

    def test_follows_the_script_conventions(self) -> None:
        """PowerShell 7, cuatro capas y solo ASCII."""
        text = _read(KAFKA_TESTS_SCRIPT)
        assert text.startswith("#Requires -Version 7.0")
        for layer in ("CAPA 1", "CAPA 2", "CAPA 3", "CAPA 4"):
            assert layer in text, f"Falta la {layer}"
        assert text.isascii()

    def test_never_leaves_the_pod_behind(self) -> None:
        """El borrado va en un finally, o un fallo a mitad dejaria el pod corriendo."""
        assert re.search(r"finally\s*\{\s*Remove-BenchmarkPod", _read(KAFKA_TESTS_SCRIPT))

    @pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell 7 no esta instalado")
    def test_pure_layer_reads_the_report_format_python_writes(self, tmp_path: Path) -> None:
        """Capa 2 ejecutada de verdad contra un informe real de build_report."""
        report_path = tmp_path / "report.json"
        write_report(
            report_path,
            build_report(
                criterion="kafka_latency",
                environment="cluster",
                value=3.5,
                unit="milliseconds",
                threshold=10.0,
                direction="at_most",
                details={},
                commit="abc1234",
            ),
        )
        harness = tmp_path / "harness.ps1"
        harness.write_text(
            _RUNNER_HARNESS.replace("__SCRIPT__", str(KAFKA_TESTS_SCRIPT)).replace(
                "__REPORT__", str(report_path)
            ),
            encoding="utf-8",
        )
        result = subprocess.run(  # noqa: S603  (pwsh resuelto con shutil.which, script propio)
            [str(shutil.which("pwsh")), "-NoProfile", "-File", str(harness)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        output = result.stdout
        assert "LINE=kafka_latency: 3.5 milliseconds (umbral <= 10; entorno: cluster)" in output
        assert "dentro del umbral" in output
        assert "MISSING=True" in output
        assert "THROUGHPUT=python -m tests.integration.benchmark_kafka" in output
        assert "--environment cluster" in output
        assert "--topic t --duration 5 --output /w/o.json --params /app/params.yaml" in output
        assert "LATENCY=python -m tests.integration.test_kafka_connectivity" in output
        assert "--messages 7 --output /w/l.json" in output
