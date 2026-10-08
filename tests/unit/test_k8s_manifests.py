"""Comprobaciones estaticas de los manifiestos de k8s y de los scripts de infra.

POR QUE EXISTE ESTE MODULO. Los YAML de `k8s/` y los defaults de los scripts de
`infra/scripts/` no los ejecuta nadie hasta que hay un cluster GKE encendido, es
decir, hasta que el error ya cuesta dinero por hora. Los dos fallos que motivaron
estos tests estaban ambos en esa zona ciega:

  1. Tres separadores `---` ausentes en network-policies.yaml fusionaban cada
     `allow-<ns>` con el `deny-all-<ns>` siguiente. De las 8 NetworkPolicies
     declaradas sobrevivian 5, y las que se perdian eran justo las que ABREN
     trafico: los namespaces quedaban con el deny-all y sin su allow, o sea
     incomunicados hasta para DNS. El fichero se leia bien a ojo porque la
     cabecera de seccion en comentarios parece un separador y no lo es.

  2. Las anotaciones de Workload Identity y los defaults de $ProjectId apuntaban
     a un proyecto GCP inexistente. Sintoma en produccion: 403 de los pods contra
     GCS y Secret Manager, sin nada en los manifiestos que delate la causa.

De ahi el criterio de diseno de este modulo: **nada de constantes propias**. Todo
lo que se espera se extrae de `infra/terraform/`, que es quien CREA los recursos.
Un test que repitiese el project ID a mano seria una novena copia del dato que
justamente se descontrolo, y podria quedarse obsoleto en la misma direccion que
el codigo que vigila.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.streaming.streaming_params import StreamingParams, load_streaming_params

REPO_ROOT = Path(__file__).resolve().parents[2]

K8S_ROOT = REPO_ROOT / "k8s"
K8S_BASE = K8S_ROOT / "base"
PARAMS_PATH = REPO_ROOT / "params.yaml"
TERRAFORM_DEV_MAIN = REPO_ROOT / "infra" / "terraform" / "environments" / "dev" / "main.tf"
TERRAFORM_IAM_MAIN = REPO_ROOT / "infra" / "terraform" / "modules" / "iam" / "main.tf"
INSTALL_KAFKA_SCRIPT = REPO_ROOT / "infra" / "scripts" / "Install-Kafka.ps1"
KAFKA_TESTS_SCRIPT = REPO_ROOT / "tests" / "integration" / "Invoke-KafkaTests.ps1"

MANIFESTS = sorted(K8S_ROOT.rglob("*.yaml"))

WORKLOAD_IDENTITY_ANNOTATION = "iam.gke.io/gcp-service-account"

# Puerto del JMX Prometheus Exporter de los brokers: lo fija Strimzi (puerto nombrado
# tcp-prometheus), no es configurable desde kafka-cluster.yaml.
JMX_EXPORTER_PORT = 9404
# Servidor de metadatos de GKE bajo Calico: sin esta ruta Workload Identity no obtiene token.
GKE_METADATA_SERVER_CIDR = "169.254.169.252/32"
GKE_METADATA_SERVER_PORT = 988
# Unicos destinos abiertos (sin `to`) que una NetworkPolicy puede declarar: HTTPS saliente
# hacia las APIs de Google / alertas externas y DNS.
_OPEN_EGRESS_ALLOWED = frozenset({("TCP", 443), ("UDP", 53), ("TCP", 53)})

# Scripts cuyo default de proyecto/region debe coincidir con Terraform, y el
# patron con el que cada uno declara la variable.
_PROJECT_PATTERNS: dict[str, str] = {
    "infra/scripts/Verify-Infra.ps1": r"\$ProjectId\s*=\s*\"([^\"]+)\"",
    "infra/scripts/Load-Secrets.ps1": r"\$ProjectId\s*=\s*\"([^\"]+)\"",
    "infra/scripts/Test-WorkloadIdentity.ps1": r"\$ProjectId\s*=\s*\"([^\"]+)\"",
    "infra/scripts/bootstrap.ps1": r"\$PROJECT_ID\s*=\s*\"([^\"]+)\"",
    "infra/scripts/bootstrap.sh": r"^PROJECT_ID=\"([^\"]+)\"",
}

_REGION_PATTERNS: dict[str, str] = {
    "infra/scripts/Verify-Infra.ps1": r"\$Region\s*=\s*\"([^\"]+)\"",
    "infra/scripts/bootstrap.ps1": r"\$REGION\s*=\s*\"([^\"]+)\"",
    "infra/scripts/bootstrap.sh": r"^REGION=\"([^\"]+)\"",
}


def _read(path: Path) -> str:
    """Return the UTF-8 text of a repository file.

    Args:
        path: File to read.

    Returns:
        The decoded contents.
    """
    return path.read_text(encoding="utf-8")


def _terraform_local(name: str) -> str:
    """Extract a value from the locals block of the dev Terraform environment.

    Terraform is the authority: it is what actually creates the project's
    resources, so the manifests and scripts must agree with it and not the other
    way round.

    Args:
        name: Local variable name, such as project_id or region.

    Returns:
        The literal string assigned to that local.

    Raises:
        AssertionError: If the local is not declared exactly once.
    """
    pattern = rf"^\s*{name}\s*=\s*\"([^\"]+)\""
    matches = re.findall(pattern, _read(TERRAFORM_DEV_MAIN), re.MULTILINE)
    assert len(matches) == 1, (
        f"Se esperaba un unico local {name} en {TERRAFORM_DEV_MAIN.name}, hay {len(matches)}"
    )
    return matches[0]


def _terraform_service_accounts() -> set[str]:
    """Return the account_id of every service account the IAM module creates.

    Returns:
        The set of account ids, without the @project.iam.gserviceaccount.com suffix.
    """
    pattern = r"^\s*account_id\s*=\s*\"([^\"]+)\""
    return set(re.findall(pattern, _read(TERRAFORM_IAM_MAIN), re.MULTILINE))


def _documents(path: Path) -> list[dict[str, object]]:
    """Parse a manifest into its non-empty YAML documents.

    Args:
        path: Manifest to parse.

    Returns:
        The list of parsed documents, skipping empty ones.
    """
    return [doc for doc in yaml.safe_load_all(_read(path)) if doc]


def _annotated_service_accounts() -> list[tuple[Path, str, str]]:
    """Collect every Workload Identity annotation declared under k8s/.

    Returns:
        Triples of (manifest path, ServiceAccount name, annotated GSA email).
    """
    found: list[tuple[Path, str, str]] = []
    for path in MANIFESTS:
        for doc in _documents(path):
            if not isinstance(doc, dict) or doc.get("kind") != "ServiceAccount":
                continue
            metadata = doc.get("metadata")
            if not isinstance(metadata, dict):
                continue
            annotations = metadata.get("annotations")
            if not isinstance(annotations, dict):
                continue
            email = annotations.get(WORKLOAD_IDENTITY_ANNOTATION)
            if isinstance(email, str):
                found.append((path, str(metadata.get("name")), email))
    return found


def test_manifests_are_discovered() -> None:
    """The glob must find manifests, otherwise every test below passes vacuously."""
    assert MANIFESTS, f"No se encontro ningun YAML bajo {K8S_ROOT}"


class TestManifestsParse:
    """Every manifest must be readable, and no resource may vanish while parsing."""

    @pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: str(p.relative_to(K8S_ROOT)))
    def test_manifest_parses(self, path: Path) -> None:
        """A manifest that does not parse would only fail at kubectl apply time."""
        assert _documents(path), f"{path} no contiene ningun documento YAML"

    @pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: str(p.relative_to(K8S_ROOT)))
    def test_no_documents_merged_by_a_missing_separator(self, path: Path) -> None:
        """Each top-level apiVersion must start its own document.

        Un `---` ausente no rompe el parseo: funde dos documentos en uno y las
        claves repetidas se pisan, de modo que el recurso de arriba desaparece
        callado. Contar las lineas `apiVersion:` en columna 0 y compararlas con
        los documentos parseados detecta exactamente ese caso.

        Limitacion asumida: el conteo supone un `apiVersion:` sin indentar por
        documento, que es como estan escritos todos los manifiestos del repo.
        """
        declared = [
            number
            for number, line in enumerate(_read(path).splitlines(), start=1)
            if line.startswith("apiVersion:")
        ]
        parsed = _documents(path)
        assert len(declared) == len(parsed), (
            f"{path.relative_to(REPO_ROOT)}: {len(declared)} lineas 'apiVersion:' "
            f"(lineas {declared}) pero {len(parsed)} documentos parseados. "
            "Falta un separador '---' y hay recursos fusionandose en silencio."
        )


class TestWorkloadIdentityAnnotations:
    """The KSA annotations must name service accounts that Terraform really creates."""

    def test_annotations_exist(self) -> None:
        """Without annotations there is nothing to check and no Workload Identity."""
        assert _annotated_service_accounts(), (
            "Ninguna ServiceAccount lleva anotacion de Workload Identity"
        )

    @pytest.mark.parametrize(
        ("path", "ksa", "email"),
        _annotated_service_accounts(),
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_annotation_targets_the_terraform_project(
        self, path: Path, ksa: str, email: str
    ) -> None:
        """A GSA in another project yields opaque 403s, never a manifest error."""
        expected_domain = f"@{_terraform_local('project_id')}.iam.gserviceaccount.com"
        assert email.endswith(expected_domain), (
            f"{path.relative_to(REPO_ROOT)}: la KSA {ksa} apunta a {email}, "
            f"que no pertenece al proyecto que crea Terraform ({expected_domain})"
        )

    @pytest.mark.parametrize(
        ("path", "ksa", "email"),
        _annotated_service_accounts(),
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_annotation_names_an_existing_service_account(
        self, path: Path, ksa: str, email: str
    ) -> None:
        """The account must be one the IAM module declares, not an invented name."""
        account = email.split("@", 1)[0]
        accounts = _terraform_service_accounts()
        assert account in accounts, (
            f"{path.relative_to(REPO_ROOT)}: la KSA {ksa} apunta a la GSA {account!r}, "
            f"que Terraform no crea. Existentes: {sorted(accounts)}"
        )


class TestNetworkPolicyPairs:
    """Under a deny-all baseline, a namespace without its allow policy is isolated."""

    @staticmethod
    def _policies_by_namespace() -> dict[str, set[str]]:
        """Group NetworkPolicy names by the namespace they apply to.

        Returns:
            Mapping of namespace to the set of policy names declared for it.
        """
        grouped: dict[str, set[str]] = {}
        for path in MANIFESTS:
            for doc in _documents(path):
                if not isinstance(doc, dict) or doc.get("kind") != "NetworkPolicy":
                    continue
                metadata = doc.get("metadata")
                if not isinstance(metadata, dict):
                    continue
                namespace = str(metadata.get("namespace"))
                grouped.setdefault(namespace, set()).add(str(metadata.get("name")))
        return grouped

    def test_policies_exist(self) -> None:
        """A repo with no NetworkPolicy would pass the pairing test vacuously."""
        assert self._policies_by_namespace(), "No hay ninguna NetworkPolicy declarada"

    def test_every_denied_namespace_has_an_allow_policy(self) -> None:
        """deny-all sin su allow deja el namespace sin trafico, ni siquiera DNS.

        Es el sintoma exacto del separador ausente: el allow se fusiona con el
        deny siguiente y desaparece, dejando el namespace incomunicado sin que
        ningun fichero lo diga.
        """
        isolated = {
            namespace: sorted(names)
            for namespace, names in self._policies_by_namespace().items()
            if any(name.startswith("deny-all") for name in names)
            and not any(name.startswith("allow") for name in names)
        }
        assert not isolated, (
            f"Namespaces con deny-all y sin ninguna policy allow: {isolated}. "
            "Sus pods arrancarian pero no podrian abrir ninguna conexion."
        )


class TestInfraScriptDefaults:
    """Script defaults must match Terraform, which is what creates the resources."""

    @pytest.mark.parametrize(("relative", "pattern"), sorted(_PROJECT_PATTERNS.items()))
    def test_project_default_matches_terraform(self, relative: str, pattern: str) -> None:
        """A wrong default sends every unqualified invocation to the wrong project."""
        path = REPO_ROOT / relative
        found = re.findall(pattern, _read(path), re.MULTILINE)
        assert found, f"{relative}: no se encontro ninguna asignacion de proyecto con {pattern!r}"
        expected = _terraform_local("project_id")
        assert set(found) == {expected}, (
            f"{relative}: declara {sorted(set(found))}, Terraform usa {expected!r}"
        )

    @pytest.mark.parametrize(("relative", "pattern"), sorted(_REGION_PATTERNS.items()))
    def test_region_default_matches_terraform(self, relative: str, pattern: str) -> None:
        """Una region equivocada hace que Verify-Infra.ps1 no encuentre el cluster.

        El fallo es peor que un error: el script reporta "no desplegado" con la
        infraestructura viva, porque busca en otra region y alli no hay nada.
        """
        path = REPO_ROOT / relative
        found = re.findall(pattern, _read(path), re.MULTILINE)
        assert found, f"{relative}: no se encontro ninguna asignacion de region con {pattern!r}"
        expected = _terraform_local("region")
        assert set(found) == {expected}, (
            f"{relative}: declara {sorted(set(found))}, Terraform usa {expected!r}"
        )


# ---------------------------------------------------------------------------
# M6: Kafka, kustomize y NetworkPolicies
# ---------------------------------------------------------------------------


def _all_documents() -> list[tuple[Path, dict[str, Any]]]:
    """Return every YAML document under k8s/ together with the file it came from.

    Returns:
        Pairs of (manifest path, parsed document), skipping non-mapping documents.
    """
    return [
        (path, doc) for path in MANIFESTS for doc in _documents(path) if isinstance(doc, dict)
    ]


def _find(kind: str, name: str, namespace: str | None = None) -> dict[str, Any]:
    """Return the single manifest document of a kind and name.

    Args:
        kind: Resource kind, such as ConfigMap or KafkaUser.
        name: metadata.name to look for.
        namespace: When given, metadata.namespace must also match.

    Returns:
        The parsed document.

    Raises:
        AssertionError: If there is not exactly one match.
    """
    matches = [
        doc
        for _, doc in _all_documents()
        if doc.get("kind") == kind
        and doc.get("metadata", {}).get("name") == name
        and (namespace is None or doc["metadata"].get("namespace") == namespace)
    ]
    assert len(matches) == 1, f"Se esperaba un unico {kind}/{name}, hay {len(matches)}"
    return matches[0]


def _kustomized_files(directory: Path) -> set[Path]:
    """Collect the manifest files a kustomization pulls in, following directories.

    Args:
        directory: Directory holding a kustomization.yaml.

    Returns:
        Resolved paths of every file reachable from that kustomization.
    """
    kustomization = yaml.safe_load(_read(directory / "kustomization.yaml"))
    found: set[Path] = set()
    for entry in kustomization.get("resources", []):
        target = (directory / entry).resolve()
        if target.is_dir():
            found |= _kustomized_files(target)
        else:
            found.add(target)
    return found


def _kafka_topic_names() -> set[str]:
    """Return the name of every KafkaTopic declared under k8s/."""
    return {
        doc["metadata"]["name"] for _, doc in _all_documents() if doc.get("kind") == "KafkaTopic"
    }


class TestDocumentSeparators:
    """Each document must start behind its own `---`, not only parse as one."""

    @pytest.mark.parametrize("path", MANIFESTS, ids=lambda p: str(p.relative_to(K8S_ROOT)))
    def test_every_block_between_separators_is_one_document(self, path: Path) -> None:
        """Cada bloque entre `---` lleva un unico `apiVersion:` en columna 0.

        Complementa al test de conteo global: dos recursos fusionados en un bloque
        dejan ese bloque con dos `apiVersion:`, y un `---` sobrante o ausente rompe
        la igualdad entre bloques con contenido y documentos parseados.
        """
        chunks = re.split(r"^---[ \t]*$", _read(path), flags=re.MULTILINE)
        populated = [
            chunk
            for chunk in chunks
            if any(
                line.strip() and not line.lstrip().startswith("#") for line in chunk.splitlines()
            )
        ]
        for chunk in populated:
            declared = [line for line in chunk.splitlines() if line.startswith("apiVersion:")]
            assert len(declared) == 1, (
                f"{path.relative_to(REPO_ROOT)}: un bloque entre separadores declara "
                f"{len(declared)} lineas 'apiVersion:' en columna 0 (se esperaba 1)."
            )
        assert len(populated) == len(_documents(path)), (
            f"{path.relative_to(REPO_ROOT)}: {len(populated)} bloques con contenido pero "
            f"{len(_documents(path))} documentos parseados."
        )


class TestKustomizationCoverage:
    """`kubectl apply -k k8s/base/` must reach every manifest of the base."""

    def test_base_kustomization_includes_every_manifest(self) -> None:
        """Un manifiesto fuera del kustomization no se despliega y nadie lo nota.

        Es el hueco que tenia el repo: kafka/ existia pero `apply -k` no creaba el
        cluster, ni los topics, ni los usuarios.
        """
        expected = {
            path.resolve()
            for path in K8S_BASE.rglob("*.yaml")
            if path.name != "kustomization.yaml"
        }
        reachable = _kustomized_files(K8S_BASE)
        missing = sorted(str(p.relative_to(REPO_ROOT)) for p in expected - reachable)
        assert not missing, f"Manifiestos fuera de k8s/base/kustomization.yaml: {missing}"

    def test_every_kustomization_resource_exists(self) -> None:
        """Una entrada que apunta a un fichero inexistente rompe el apply entero."""
        for kustomization in K8S_BASE.rglob("kustomization.yaml"):
            document = yaml.safe_load(_read(kustomization))
            for entry in document.get("resources", []):
                assert (kustomization.parent / entry).exists(), (
                    f"{kustomization.relative_to(REPO_ROOT)}: el recurso '{entry}' no existe"
                )


class TestKafkaAcls:
    """Every topic and group the code uses must be covered by the right KafkaUser."""

    @staticmethod
    def _acls_by_user() -> dict[str, dict[tuple[str, str], set[str]]]:
        """Group the ACL operations of each KafkaUser by (resource type, resource name).

        Returns:
            Mapping of user name to {(type, name): operations}.
        """
        users: dict[str, dict[tuple[str, str], set[str]]] = {}
        for _, doc in _all_documents():
            if doc.get("kind") != "KafkaUser":
                continue
            acls: dict[tuple[str, str], set[str]] = {}
            for acl in doc["spec"]["authorization"]["acls"]:
                assert acl["resource"]["patternType"] == "literal", (
                    f"{doc['metadata']['name']}: ACL con patron no literal"
                )
                key = (acl["resource"]["type"], acl["resource"]["name"])
                acls.setdefault(key, set()).update(acl["operations"])
            users[doc["metadata"]["name"]] = acls
        return users

    @staticmethod
    def _benchmark_topic() -> str:
        """Return the topic the cluster benchmark writes to, as its runner declares it."""
        match = re.search(r'\$Topic\s*=\s*"([^"]+)"', _read(KAFKA_TESTS_SCRIPT))
        assert match, "Invoke-KafkaTests.ps1 no declara $Topic"
        return match.group(1)

    @classmethod
    def _expected(
        cls, streaming: StreamingParams
    ) -> dict[str, dict[tuple[str, str], set[str]]]:
        """Return the ACLs each service needs, derived from params.yaml: streaming.

        El rol de cada usuario es el uso que hace el codigo: el streamer solo
        produce en raw; el validador consume raw con su grupo y produce en validated
        y en anomaly-alerts.

        Args:
            streaming: Resolved streaming parameters.

        Returns:
            The exact ACL set expected per user (least privilege in both directions).
        """
        return {
            "reactorguard-ingestion": {("topic", streaming.raw_topic): {"Write", "Describe"}},
            "reactorguard-validator": {
                ("topic", streaming.raw_topic): {"Read", "Describe"},
                ("group", streaming.consumer_group): {"Read"},
                ("topic", streaming.validated_topic): {"Write", "Describe"},
                ("topic", streaming.alerts_topic): {"Write", "Describe"},
            },
            # Solo su topic: una medicion de throughput no debe poder escribir en raw.
            "reactorguard-benchmark": {
                ("topic", cls._benchmark_topic()): {"Write", "Read", "Describe"},
            },
        }

    @pytest.mark.parametrize(
        "user", ["reactorguard-ingestion", "reactorguard-validator", "reactorguard-benchmark"]
    )
    def test_acls_match_what_the_code_uses(self, user: str) -> None:
        """Ni un permiso de menos (el lote no se confirma) ni uno de mas.

        Sin Write en anomaly-alerts, el primer AlertEvent falla en el flush y el
        validador reintenta el mismo lote para siempre. Un permiso de mas rompe el
        minimo privilegio sobre los datos de una planta nuclear.
        """
        streaming = load_streaming_params(PARAMS_PATH)
        actual = self._acls_by_user()[user]
        expected = self._expected(streaming)[user]
        missing = {key: sorted(ops - actual.get(key, set())) for key, ops in expected.items()}
        extra = {key: sorted(ops - expected.get(key, set())) for key, ops in actual.items()}
        assert actual == expected, (
            f"ACL de {user} distintas de las que usa el codigo. "
            f"Faltan: { {k: v for k, v in missing.items() if v} }. "
            f"Sobran: { {k: v for k, v in extra.items() if v} }."
        )

    def test_every_streaming_topic_is_declared(self) -> None:
        """Un topic usado por el codigo y sin KafkaTopic no existe: el productor se bloquea."""
        streaming = load_streaming_params(PARAMS_PATH)
        used = {streaming.raw_topic, streaming.validated_topic, streaming.alerts_topic}
        assert used <= _kafka_topic_names(), (
            f"Topics sin KafkaTopic: {sorted(used - _kafka_topic_names())}"
        )

    def test_benchmark_topic_replicates_like_the_real_one(self) -> None:
        """Un topic de benchmark con otra replicacion daria una cifra no representativa."""
        raw = _find("KafkaTopic", load_streaming_params(PARAMS_PATH).raw_topic)
        bench = _find("KafkaTopic", self._benchmark_topic())
        for key in ("partitions", "replicas"):
            assert bench["spec"][key] == raw["spec"][key], f"bench-throughput difiere en {key}"
        assert bench["spec"]["config"]["min.insync.replicas"] == (
            raw["spec"]["config"]["min.insync.replicas"]
        )

    def test_every_acl_topic_is_declared(self) -> None:
        """Una ACL sobre un topic inexistente es un typo que nadie ve hasta el cluster."""
        declared = _kafka_topic_names()
        for user, acls in self._acls_by_user().items():
            topics = {name for kind, name in acls if kind == "topic"}
            assert topics <= declared, (
                f"{user}: ACL sobre topics no declarados {sorted(topics - declared)}"
            )


class TestNetworkPolicyRules:
    """The egress and ingress rules M6 adds, and the ceiling on what may be open."""

    @staticmethod
    def _spec(namespace: str, name: str) -> dict[str, Any]:
        """Return the spec of a NetworkPolicy.

        Args:
            namespace: Namespace the policy applies to.
            name: Policy name.

        Returns:
            The policy spec.
        """
        return dict(_find("NetworkPolicy", name, namespace)["spec"])

    @staticmethod
    def _ports(rule: dict[str, Any]) -> set[tuple[str, int]]:
        """Return the (protocol, port) pairs a rule opens.

        Args:
            rule: One ingress or egress rule.

        Returns:
            The set of pairs; empty when the rule lists no ports.
        """
        return {(port.get("protocol", "TCP"), port["port"]) for port in rule.get("ports", [])}

    @classmethod
    def _has_rule(
        cls,
        namespace: str,
        policy: str,
        direction: str,
        port: int,
        matches_peer: Callable[[dict[str, Any]], bool],
    ) -> bool:
        """Say whether a policy has a TCP rule on a port toward a matching peer.

        Args:
            namespace: Namespace of the policy.
            policy: Policy name.
            direction: "ingress" or "egress".
            port: TCP port the rule must open.
            matches_peer: Predicate over one `from`/`to` peer.

        Returns:
            True when some rule opens that port to a peer the predicate accepts.
        """
        peer_key = "from" if direction == "ingress" else "to"
        for rule in cls._spec(namespace, policy).get(direction, []):
            if ("TCP", port) not in cls._ports(rule):
                continue
            if any(matches_peer(peer) for peer in rule.get(peer_key, [])):
                return True
        return False

    @staticmethod
    def _namespace(peer: dict[str, Any]) -> str | None:
        """Return the `name` label a peer's namespaceSelector requires, if any."""
        labels = (peer.get("namespaceSelector") or {}).get("matchLabels") or {}
        value = labels.get("name")
        return str(value) if value is not None else None

    @staticmethod
    def _pod_app(peer: dict[str, Any]) -> str | None:
        """Return the app.kubernetes.io/name a peer's podSelector requires, if any."""
        labels = (peer.get("podSelector") or {}).get("matchLabels") or {}
        value = labels.get("app.kubernetes.io/name")
        return str(value) if value is not None else None

    def test_no_cidr_opens_the_whole_internet(self) -> None:
        """Ninguna policy declara 0.0.0.0/0 ni ::/0 como ipBlock."""
        offenders = [
            (doc["metadata"]["name"], peer["ipBlock"]["cidr"])
            for _, doc in _all_documents()
            if doc.get("kind") == "NetworkPolicy"
            for direction, peer_key in (("ingress", "from"), ("egress", "to"))
            for rule in doc["spec"].get(direction, [])
            for peer in rule.get(peer_key, [])
            if peer.get("ipBlock", {}).get("cidr") in {"0.0.0.0/0", "::/0"}
        ]
        assert not offenders, f"ipBlock abiertos a todo internet: {offenders}"

    def test_open_egress_is_limited_to_https_and_dns(self) -> None:
        """Un egress sin `to` sale a cualquier destino: solo vale para 443 y DNS.

        Una regla sin `to` y sin `ports` abriria TODO el trafico saliente. Lo unico
        abierto a cualquier destino es lo que ya existia: HTTPS (APIs de Google,
        alertas externas) y DNS.
        """
        for _, doc in _all_documents():
            if doc.get("kind") != "NetworkPolicy":
                continue
            for rule in doc["spec"].get("egress", []):
                if rule.get("to"):
                    continue
                ports = self._ports(rule)
                name = f"{doc['metadata']['namespace']}/{doc['metadata']['name']}"
                assert ports, f"{name}: regla de egress sin destino ni puertos (abre todo)"
                excess = sorted(ports - _OPEN_EGRESS_ALLOWED)
                assert not excess, (
                    f"{name}: egress abierto a cualquier destino en {excess}; "
                    f"solo se admiten {sorted(_OPEN_EGRESS_ALLOWED)}"
                )

    def test_ingress_rules_always_name_a_source(self) -> None:
        """Una regla de ingress sin `from` acepta trafico de cualquier origen."""
        for _, doc in _all_documents():
            if doc.get("kind") != "NetworkPolicy":
                continue
            for rule in doc["spec"].get("ingress", []):
                assert rule.get("from"), (
                    f"{doc['metadata']['namespace']}/{doc['metadata']['name']}: "
                    "regla de ingress sin 'from'"
                )

    def test_namespace_selectors_reference_labeled_namespaces(self) -> None:
        """Un `name:` sin namespace con esa etiqueta deja la regla sin efecto, en silencio."""
        labeled = {
            doc["metadata"]["labels"]["name"]
            for _, doc in _all_documents()
            if doc.get("kind") == "Namespace"
        }
        for _, doc in _all_documents():
            if doc.get("kind") != "NetworkPolicy":
                continue
            for direction, peer_key in (("ingress", "from"), ("egress", "to")):
                for rule in doc["spec"].get(direction, []):
                    for peer in rule.get(peer_key, []):
                        target = self._namespace(peer)
                        assert target is None or target in labeled, (
                            f"{doc['metadata']['name']}: namespaceSelector name={target!r} "
                            f"sin namespace etiquetado. Existentes: {sorted(labeled)}"
                        )

    def test_ingestion_reaches_google_apis_and_the_metadata_server(self) -> None:
        """El streamer lee GCS: necesita el 443 y el servidor de metadatos de Workload Identity."""
        spec = self._spec("reactorguard-ingestion", "allow-ingestion")
        open_https = [
            rule
            for rule in spec["egress"]
            if not rule.get("to") and ("TCP", 443) in self._ports(rule)
        ]
        assert open_https, "ingestion no tiene egress 443"
        assert self._has_rule(
            "reactorguard-ingestion",
            "allow-ingestion",
            "egress",
            GKE_METADATA_SERVER_PORT,
            lambda peer: peer.get("ipBlock", {}).get("cidr") == GKE_METADATA_SERVER_CIDR,
        ), "ingestion no alcanza el servidor de metadatos de GKE: Workload Identity no daria token"

    @pytest.mark.parametrize("port", [9092, 9093])
    def test_ingestion_egress_to_kafka(self, port: int) -> None:
        """El 9093 (mTLS) debe estar abierto hacia kafka-operator, ademas del 9092."""
        assert self._has_rule(
            "reactorguard-ingestion",
            "allow-ingestion",
            "egress",
            port,
            lambda peer: self._namespace(peer) == "kafka-operator",
        ), f"ingestion no tiene egress a kafka-operator:{port}"

    def test_redis_inside_ingestion(self) -> None:
        """Redis 6379 dentro de reactorguard-ingestion, en las dos direcciones."""
        assert self._has_rule(
            "reactorguard-ingestion",
            "allow-ingestion",
            "egress",
            6379,
            lambda peer: self._pod_app(peer) == "redis",
        ), "ingestion no tiene egress a los pods de Redis"
        assert self._has_rule(
            "reactorguard-ingestion",
            "allow-ingestion",
            "ingress",
            6379,
            lambda peer: peer.get("podSelector") == {},
        ), "ingestion no acepta trafico a Redis desde su propio namespace"

    def test_ml_reaches_redis_and_ingestion_accepts_it(self) -> None:
        """ml -> Redis: egress en ml (solo los pods de Redis) e ingress en ingestion."""
        assert self._has_rule(
            "reactorguard-ml",
            "allow-ml",
            "egress",
            6379,
            lambda peer: self._namespace(peer) == "reactorguard-ingestion"
            and self._pod_app(peer) == "redis",
        ), "ml no tiene egress 6379 a los pods de Redis de ingestion"
        assert self._has_rule(
            "reactorguard-ingestion",
            "allow-ingestion",
            "ingress",
            6379,
            lambda peer: self._namespace(peer) == "reactorguard-ml",
        ), "ingestion no acepta 6379 desde reactorguard-ml"

    def test_observability_scrapes_kafka_jmx(self) -> None:
        """observability -> kafka-operator:9404: egress en observability e ingress en kafka."""
        assert self._has_rule(
            "reactorguard-observability",
            "allow-observability",
            "egress",
            JMX_EXPORTER_PORT,
            lambda peer: self._namespace(peer) == "kafka-operator",
        ), "observability no puede salir a kafka-operator:9404"
        assert self._has_rule(
            "kafka-operator",
            "allow-kafka",
            "ingress",
            JMX_EXPORTER_PORT,
            lambda peer: self._namespace(peer) == "reactorguard-observability",
        ), "kafka-operator no acepta el JMX desde observability"

    def test_kafka_operator_reaches_the_api_server(self) -> None:
        """El operador Strimzi reconcilia contra la API: ruta al master privado de Terraform."""
        master_cidr = _terraform_local("master_ipv4_cidr")
        assert self._has_rule(
            "kafka-operator",
            "allow-kafka",
            "egress",
            443,
            lambda peer: peer.get("ipBlock", {}).get("cidr") == master_cidr,
        ), (
            f"kafka-operator no tiene egress 443 a {master_cidr} (master_ipv4_cidr de "
            "Terraform): el operador y el Entity Operator no podrian hablar con la API"
        )


class TestKafkaMetricsConfigMap:
    """kafka-cluster.yaml references a ConfigMap that must exist, or brokers do not start."""

    @staticmethod
    def _config_map() -> tuple[dict[str, str], dict[str, Any]]:
        """Return the metricsConfig reference of the Kafka resource and the ConfigMap it names.

        Returns:
            The reference (name, key, namespace) and the referenced ConfigMap document.
        """
        cluster = _find("Kafka", "reactorguard-cluster")
        ref = cluster["spec"]["kafka"]["metricsConfig"]["valueFrom"]["configMapKeyRef"]
        namespace = cluster["metadata"]["namespace"]
        reference = {"name": ref["name"], "key": ref["key"], "namespace": namespace}
        return reference, _find("ConfigMap", ref["name"], namespace)

    def test_referenced_config_map_exists_in_the_cluster_namespace(self) -> None:
        """Sin el ConfigMap Strimzi no genera la configuracion de los brokers."""
        reference, config_map = self._config_map()
        assert reference["key"] in config_map["data"], (
            f"El ConfigMap {reference['name']} no tiene la clave {reference['key']}"
        )

    def test_exporter_config_is_valid_and_its_patterns_compile(self) -> None:
        """Un patron que no compila hace fallar el JMX exporter al arrancar el broker."""
        reference, config_map = self._config_map()
        config = yaml.safe_load(config_map["data"][reference["key"]])
        assert config["lowercaseOutputName"] is True
        assert config["rules"], "La configuracion del exporter no tiene reglas"
        for rule in config["rules"]:
            assert "name" in rule, f"Regla sin 'name': {rule}"
            re.compile(rule["pattern"])

    @pytest.mark.parametrize(
        "mbean",
        [
            "kafka.server<type=ReplicaManager, name=UnderReplicatedPartitions><>Value",
            "kafka.server<type=BrokerTopicMetrics, name=MessagesInPerSec><>Count",
            "kafka.controller<type=KafkaController, name=ActiveControllerCount><>Value",
        ],
    )
    def test_exporter_covers_the_dashboard_metrics(self, mbean: str) -> None:
        """Las reglas deben cubrir ISR, particiones sub-replicadas y mensajes por segundo."""
        reference, config_map = self._config_map()
        config = yaml.safe_load(config_map["data"][reference["key"]])
        assert any(re.match(rule["pattern"], mbean) for rule in config["rules"]), (
            f"Ninguna regla del exporter cubre {mbean}"
        )


class TestInstallKafkaScript:
    """Install-Kafka.ps1 must apply the Kafka manifests in an order that works."""

    _MANIFESTS = (
        "kafka-metrics.yaml",
        "kafka-cluster.yaml",
        "kafka-topics.yaml",
        "kafka-users.yaml",
    )

    def test_applies_every_kafka_manifest(self) -> None:
        """Si el script no aplica kafka-users.yaml, ningun cliente tiene identidad ni ACL."""
        text = _read(INSTALL_KAFKA_SCRIPT)
        for manifest in self._MANIFESTS:
            assert f'"{manifest}"' in text, f"Install-Kafka.ps1 no aplica {manifest}"

    def test_metrics_before_cluster_and_users_after_topics(self) -> None:
        """El ConfigMap antes del cluster (o los brokers no arrancan); usuarios al final."""
        text = _read(INSTALL_KAFKA_SCRIPT)
        position = {name: text.index(f'"{name}"') for name in self._MANIFESTS}
        assert position["kafka-metrics.yaml"] < position["kafka-cluster.yaml"]
        assert position["kafka-cluster.yaml"] < position["kafka-topics.yaml"]
        assert position["kafka-topics.yaml"] < position["kafka-users.yaml"]
        orchestration = text[text.rindex("Install-StrimziOperator") :]
        assert orchestration.index("Deploy-KafkaTopics") < orchestration.index("Deploy-KafkaUsers")

    def test_waits_for_the_users_declared_in_the_manifest(self) -> None:
        """Esperar a usuarios que el manifiesto no declara haria colgar el script."""
        declared = {
            doc["metadata"]["name"] for _, doc in _all_documents() if doc.get("kind") == "KafkaUser"
        }
        block = re.search(r"\$expectedUsers\s*=\s*@\(([^)]*)\)", _read(INSTALL_KAFKA_SCRIPT))
        assert block, "Install-Kafka.ps1 no declara $expectedUsers"
        waited = set(re.findall(r'"([^"]+)"', block.group(1)))
        assert waited == declared, (
            f"Install-Kafka espera {sorted(waited)}, el manifiesto declara {sorted(declared)}"
        )
