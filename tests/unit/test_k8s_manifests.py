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
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

K8S_ROOT = REPO_ROOT / "k8s"
TERRAFORM_DEV_MAIN = REPO_ROOT / "infra" / "terraform" / "environments" / "dev" / "main.tf"
TERRAFORM_IAM_MAIN = REPO_ROOT / "infra" / "terraform" / "modules" / "iam" / "main.tf"

MANIFESTS = sorted(K8S_ROOT.rglob("*.yaml"))

WORKLOAD_IDENTITY_ANNOTATION = "iam.gke.io/gcp-service-account"

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
