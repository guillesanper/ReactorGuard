"""Coherencia del dashboard de Grafana del pipeline de datos (T4.6) con el codigo.

POR QUE EXISTE ESTE MODULO. Un dashboard es JSON que nadie ejecuta hasta que alguien
abre Grafana, y el fallo tipico es silencioso: un panel que cita una metrica que no
existe (o una etiqueta mal escrita) no da error, se queda vacio, y un panel vacio se
lee como "todo en calma". Estos tests cierran esa clase de error comparando el JSON
contra la fuente de verdad:

  - los nombres de metrica y de etiqueta, contra `StreamMetrics` (lo que el servicio
    expone de verdad, incluidos los sufijos `_total`, `_bucket`, `_sum` y `_count`);
  - los valores de etiqueta (`quality`, `fault_type`, `severity`, `reason`, `topic`),
    contra los enums del codigo y `params.yaml`;
  - las ventanas, el refresco y los umbrales, contra la cadencia del TEP (180 s) y los
    parametros de streaming.

Como en test_k8s_manifests.py, el modulo no define constantes propias de los datos que
vigila: todo se extrae de la fuente que lo produce.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from data.generators.tep_params import load_tep_params
from data.schemas.sensor_reading import QualityFlag
from data.streaming.metrics import RESET_REASONS, StreamMetrics
from data.streaming.streaming_params import load_streaming_params
from data.validation.sensor_fault import FaultType, Severity

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "params.yaml"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
GRAFANA_DIR = REPO_ROOT / "observability" / "grafana"
DASHBOARDS_DIR = GRAFANA_DIR / "dashboards"
DASHBOARD_PATH = DASHBOARDS_DIR / "data-pipeline.json"
KUSTOMIZATION_PATH = DASHBOARDS_DIR / "kustomization.yaml"
PROVISIONING_PATH = GRAFANA_DIR / "provisioning" / "dashboards.yaml"
COMPOSE_PATH = REPO_ROOT / "infra" / "local" / "docker-compose.yaml"
PROMETHEUS_CONFIG_PATH = REPO_ROOT / "infra" / "local" / "prometheus" / "prometheus.yml"
NAMESPACES_PATH = REPO_ROOT / "k8s" / "base" / "namespaces.yaml"

RATE_WINDOW_VARIABLE = "rate_window"
DATASOURCE_VARIABLE = "ds"
# Etiquetas que Prometheus anade o el servicio usa sin declararlas en la metrica.
IMPLICIT_LABELS = frozenset({"le", "job", "instance"})
# Grafana limita el uid de un dashboard a 40 caracteres.
UID_MAX_LENGTH = 40
GRID_COLUMNS = 24


# ---------------------------------------------------------------------------
# Fuente de verdad: lo que StreamMetrics expone
# ---------------------------------------------------------------------------

_SERIES_SUFFIXES = {
    "counter": ("_total", "_created"),
    "gauge": ("",),
    "histogram": ("_bucket", "_sum", "_count", "_created"),
}


def exposed_series() -> dict[str, frozenset[str]]:
    """Map each series name Prometheus will see to the label names it carries.

    Returns:
        Series name (with its `_total`, `_bucket`, `_sum` or `_count` suffix) to the
        set of labels on it.
    """
    metrics = StreamMetrics()
    types = {family.name: family.type for family in metrics.registry.collect()}
    series: dict[str, frozenset[str]] = {}
    for collector in vars(metrics).values():
        # _name y _labelnames no son API publica de prometheus_client, pero no hay otra
        # forma de leer las etiquetas de una metrica etiquetada sin ninguna muestra.
        family = getattr(collector, "_name", None)
        if family not in types:
            continue
        labels = frozenset(getattr(collector, "_labelnames", ()))
        for suffix in _SERIES_SUFFIXES[types[family]]:
            extra = frozenset({"le"}) if suffix == "_bucket" else frozenset()
            series[f"{family}{suffix}"] = labels | extra
    return series


def label_values() -> dict[str, frozenset[str]]:
    """Return the closed set of values each enumerable label can take.

    Returns:
        Label name to its legal values, taken from the enums and params.yaml.
    """
    streaming = load_streaming_params(PARAMS_PATH)
    return {
        "quality": frozenset(member.value for member in QualityFlag),
        "fault_type": frozenset(member.value for member in FaultType),
        "severity": frozenset(member.value for member in Severity),
        "reason": frozenset(RESET_REASONS),
        "topic": frozenset(
            {streaming.raw_topic, streaming.validated_topic, streaming.alerts_topic}
        ),
    }


# ---------------------------------------------------------------------------
# Analisis minimo de PromQL (suficiente para estas expresiones, no un parser)
# ---------------------------------------------------------------------------

_BRACES_RE = re.compile(r"\{[^{}]*\}")
_MATCHER_RE = re.compile(r'([A-Za-z_]\w*)\s*(=~|!~|!=|=)\s*"((?:[^"\\]|\\.)*)"')
_GROUPING_RE = re.compile(r"\b(?:by|without|on|ignoring)\s*\(([^)]*)\)")
_RANGE_RE = re.compile(r"\[([^\]]*)\]")
_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_TEMPLATE_RE = re.compile(r"\$\{?[A-Za-z_]\w*\}?")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b")
_IDENT_RE = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")

# Funciones, operadores y palabras clave de PromQL que el dashboard usa. Un
# identificador que no este aqui ni en las series expuestas se trata como una metrica
# inexistente: el test falla del lado seguro y obliga a ampliar esta lista a mano.
_PROMQL_WORDS = frozenset(
    {
        "rate", "irate", "increase", "delta", "sum", "avg", "min", "max", "count",
        "topk", "bottomk", "quantile", "histogram_quantile", "round", "time", "vector",
        "abs", "ceil", "floor", "clamp_min", "clamp_max", "or", "and", "unless", "by",
        "without", "on", "ignoring", "group_left", "group_right", "bool", "offset",
    }
)  # fmt: skip


@dataclass(frozen=True)
class ParsedExpr:
    """The pieces of a PromQL expression the coherence checks look at."""

    metrics: tuple[str, ...]
    matchers: tuple[tuple[str, str, str], ...]  # (label, operator, value)
    group_labels: tuple[str, ...]
    ranges: tuple[str, ...]


def parse_promql(expr: str) -> ParsedExpr:
    """Extract metric names, label matchers, grouping labels and range windows.

    Args:
        expr: A PromQL expression, possibly with Grafana template variables.

    Returns:
        The parsed pieces.
    """
    matchers = tuple(
        (label, operator, value)
        for block in _BRACES_RE.findall(expr)
        for label, operator, value in _MATCHER_RE.findall(block)
    )
    group_labels = tuple(
        label.strip()
        for group in _GROUPING_RE.findall(expr)
        for label in group.split(",")
        if label.strip()
    )
    ranges = tuple(_RANGE_RE.findall(expr))

    rest = _BRACES_RE.sub(" ", expr)
    for pattern in (_STRING_RE, _GROUPING_RE, _RANGE_RE, _TEMPLATE_RE, _NUMBER_RE):
        rest = pattern.sub(" ", rest)
    metrics = tuple(ident for ident in _IDENT_RE.findall(rest) if ident not in _PROMQL_WORDS)
    return ParsedExpr(metrics, matchers, group_labels, ranges)


def unknown_metrics(expr: str) -> list[str]:
    """Return the metric names of an expression that StreamMetrics does not expose.

    Args:
        expr: A PromQL expression.

    Returns:
        The unknown names; empty when every name exists.
    """
    exposed = exposed_series()
    return [name for name in parse_promql(expr).metrics if name not in exposed]


def unknown_labels(expr: str) -> list[str]:
    """Return the labels used by an expression that none of its metrics carries.

    Args:
        expr: A PromQL expression.

    Returns:
        The unknown label names; empty when every label exists.
    """
    exposed = exposed_series()
    parsed = parse_promql(expr)
    known = set(IMPLICIT_LABELS)
    for name in parsed.metrics:
        known |= exposed.get(name, frozenset())
    used = {label for label, _, _ in parsed.matchers} | set(parsed.group_labels)
    return sorted(used - known)


def invalid_label_values(expr: str) -> list[str]:
    """Return matcher values that are not legal for an enumerable label.

    Un `=~"a|b"` se descompone en sus alternativas. Tambien se comprueban `!=` y
    `!~`: un valor mal escrito en una exclusion no excluye nada y pasa inadvertido.

    Args:
        expr: A PromQL expression.

    Returns:
        Descriptions of the offending matchers; empty when all are legal.
    """
    legal = label_values()
    problems = []
    for label, operator, value in parse_promql(expr).matchers:
        if label not in legal:
            continue
        alternatives = value.split("|") if operator in ("=~", "!~") else [value]
        problems.extend(
            f'{label}{operator}"{alternative}"'
            for alternative in alternatives
            if alternative not in legal[label]
        )
    return problems


# ---------------------------------------------------------------------------
# Carga del dashboard
# ---------------------------------------------------------------------------


def _load_dashboard() -> dict[str, Any]:
    loaded = json.loads(DASHBOARD_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


DASHBOARD = _load_dashboard()
ALL_PANELS: list[dict[str, Any]] = DASHBOARD["panels"]
PANELS: list[dict[str, Any]] = [panel for panel in ALL_PANELS if panel["type"] != "row"]
PANEL_IDS = [panel["title"] for panel in PANELS]
QUERIES: list[tuple[str, str]] = [
    (panel["title"], target["expr"]) for panel in PANELS for target in panel["targets"]
]
QUERY_IDS = [f"{title}[{index}]" for index, (title, _) in enumerate(QUERIES)]


def _panel(title: str) -> dict[str, Any]:
    matches = [panel for panel in PANELS if panel["title"] == title]
    assert len(matches) == 1, f"No hay exactamente un panel titulado {title!r}"
    return matches[0]


def _variable(name: str) -> dict[str, Any]:
    matches = [var for var in DASHBOARD["templating"]["list"] if var["name"] == name]
    assert len(matches) == 1, f"No hay exactamente una variable {name!r}"
    return matches[0]


def _seconds(duration: str) -> int:
    """Convert a Prometheus or Grafana duration such as 15s, 1m or 1h to seconds."""
    match = re.fullmatch(r"(\d+)([smh])", duration)
    assert match is not None, f"Duracion no reconocida: {duration!r}"
    return int(match.group(1)) * {"s": 1, "m": 60, "h": 3600}[match.group(2)]


def _step_value(panel: dict[str, Any], color: str) -> float:
    steps = panel["fieldConfig"]["defaults"]["thresholds"]["steps"]
    values = [step["value"] for step in steps if step["color"] == color]
    assert len(values) == 1, f"Se esperaba un unico umbral {color!r} en {panel['title']!r}"
    return float(values[0])


# ---------------------------------------------------------------------------
# El fichero y su identidad
# ---------------------------------------------------------------------------


class TestFile:
    def test_is_valid_json_with_panels(self) -> None:
        assert PANELS
        assert DASHBOARD["schemaVersion"] >= 38

    def test_is_ascii_only(self) -> None:
        # Sin emojis ni tildes en ningun fichero del repo (estandar del proyecto).
        assert DASHBOARD_PATH.read_text(encoding="utf-8").isascii()

    def test_is_not_ignored_by_git(self) -> None:
        # La regla *.json del .gitignore tapa todo JSON salvo excepcion explicita.
        rules = GITIGNORE_PATH.read_text(encoding="utf-8").splitlines()
        assert "!observability/**/*.json" in rules


class TestIdentity:
    def test_uid_is_well_formed(self) -> None:
        uid = DASHBOARD["uid"]
        assert re.fullmatch(r"[A-Za-z0-9_-]+", uid)
        assert len(uid) <= UID_MAX_LENGTH

    def test_uid_is_unique_across_dashboards(self) -> None:
        uids = [
            json.loads(path.read_text(encoding="utf-8"))["uid"]
            for path in sorted(DASHBOARDS_DIR.glob("*.json"))
        ]
        assert DASHBOARD["uid"] in uids
        assert len(uids) == len(set(uids)), f"uid repetido: {uids}"

    def test_panel_ids_are_unique_integers(self) -> None:
        ids = [panel["id"] for panel in ALL_PANELS]
        assert all(isinstance(value, int) for value in ids)
        assert len(ids) == len(set(ids))

    def test_panel_titles_are_unique(self) -> None:
        assert len(PANEL_IDS) == len(set(PANEL_IDS))

    def test_panels_fit_the_grid_without_overlap(self) -> None:
        rects = [
            (p["gridPos"]["x"], p["gridPos"]["y"], p["gridPos"]["w"], p["gridPos"]["h"])
            for p in ALL_PANELS
        ]
        for x, _, w, _ in rects:
            assert 0 <= x and x + w <= GRID_COLUMNS
        for i, (x1, y1, w1, h1) in enumerate(rects):
            for x2, y2, w2, h2 in rects[i + 1 :]:
                overlap = x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1
                assert not overlap, f"Paneles solapados en ({x1},{y1}) y ({x2},{y2})"


class TestDatasource:
    def test_variable_selects_prometheus(self) -> None:
        variable = _variable(DATASOURCE_VARIABLE)
        assert variable["type"] == "datasource"
        assert variable["query"] == "prometheus"

    @pytest.mark.parametrize("panel", PANELS, ids=PANEL_IDS)
    def test_panel_and_targets_use_the_variable(self, panel: dict[str, Any]) -> None:
        expected = {"type": "prometheus", "uid": f"${{{DATASOURCE_VARIABLE}}}"}
        assert panel["datasource"] == expected
        assert all(target["datasource"] == expected for target in panel["targets"])


# ---------------------------------------------------------------------------
# Las consultas contra StreamMetrics
# ---------------------------------------------------------------------------


class TestQueries:
    def test_every_panel_has_a_query(self) -> None:
        for panel in PANELS:
            assert panel["targets"], f"{panel['title']!r} no tiene consultas"
            assert all(target["expr"].strip() for target in panel["targets"])

    @pytest.mark.parametrize(("title", "expr"), QUERIES, ids=QUERY_IDS)
    def test_every_metric_exists(self, title: str, expr: str) -> None:
        assert parse_promql(expr).metrics, f"{title!r}: la expresion no cita ninguna metrica"
        assert unknown_metrics(expr) == [], f"{title!r} cita metricas que no se exponen"

    @pytest.mark.parametrize(("title", "expr"), QUERIES, ids=QUERY_IDS)
    def test_only_project_metrics_are_cited(self, title: str, expr: str) -> None:
        # No hay Kafka Exporter ni JMX raspado (ni metricas de Feast): un panel sobre
        # kafka_* o feast_* estaria vacio (decision D9 y nota de M7).
        foreign = [n for n in parse_promql(expr).metrics if not n.startswith("reactorguard_")]
        assert foreign == [], f"{title!r} cita metricas ajenas al proyecto: {foreign}"

    @pytest.mark.parametrize(("title", "expr"), QUERIES, ids=QUERY_IDS)
    def test_every_label_exists(self, title: str, expr: str) -> None:
        assert unknown_labels(expr) == [], f"{title!r} usa etiquetas inexistentes"

    @pytest.mark.parametrize(("title", "expr"), QUERIES, ids=QUERY_IDS)
    def test_label_values_are_legal(self, title: str, expr: str) -> None:
        assert invalid_label_values(expr) == [], f"{title!r} usa valores de etiqueta ilegales"

    @pytest.mark.parametrize(("title", "expr"), QUERIES, ids=QUERY_IDS)
    def test_every_range_uses_the_window_variable(self, title: str, expr: str) -> None:
        ranges = parse_promql(expr).ranges
        assert set(ranges) <= {f"${RATE_WINDOW_VARIABLE}"}, (
            f"{title!r} fija una ventana: usa ${RATE_WINDOW_VARIABLE}"
        )

    def test_the_streamer_and_consumer_topics_are_both_charted(self) -> None:
        topics = {
            value
            for _, expr in QUERIES
            for label, _, value in parse_promql(expr).matchers
            if label == "topic"
        }
        assert topics == set(label_values()["topic"])

    def test_color_overrides_cover_each_enum_exactly(self) -> None:
        # Un panel agrupado por una etiqueta enumerable tiene un color por valor: si
        # alguien anade un FaultType o una quality, este test pide su color.
        legal = label_values()
        checked = 0
        for panel in PANELS:
            legends = {target["legendFormat"] for target in panel["targets"]}
            for label, values in legal.items():
                if legends != {f"{{{{{label}}}}}"}:
                    continue
                named = {o["matcher"]["options"] for o in panel["fieldConfig"]["overrides"]}
                assert named == values, f"{panel['title']!r}: colores {named} != {values}"
                checked += 1
        assert checked >= 4  # quality, fault_type (x2), severity, reason


# ---------------------------------------------------------------------------
# Ventana, refresco y umbrales calibrados a la cadencia del TEP
# ---------------------------------------------------------------------------


class TestCadence:
    @staticmethod
    def _interval_seconds() -> int:
        return load_tep_params(PARAMS_PATH).sample_interval_minutes * 60

    def test_window_options_are_the_requested_ones(self) -> None:
        variable = _variable(RATE_WINDOW_VARIABLE)
        assert variable["type"] == "custom"
        assert [opt["value"] for opt in variable["options"]] == ["1m", "5m", "15m", "1h"]
        assert variable["query"] == "1m,5m,15m,1h"

    def test_default_window_is_15m_and_selected(self) -> None:
        variable = _variable(RATE_WINDOW_VARIABLE)
        assert variable["current"]["value"] == "15m"
        selected = [opt["value"] for opt in variable["options"] if opt["selected"]]
        assert selected == ["15m"]

    def test_default_window_holds_several_tep_samples(self) -> None:
        # Con una muestra cada 180 s, una ventana de 1 m contiene como mucho una
        # (rate() oscila entre 0 y un pico): la de defecto debe abarcar varias.
        default = _seconds(_variable(RATE_WINDOW_VARIABLE)["current"]["value"])
        assert default >= 3 * self._interval_seconds()

    def test_refresh_is_between_scrape_and_sample_interval(self) -> None:
        config = yaml.safe_load(PROMETHEUS_CONFIG_PATH.read_text(encoding="utf-8"))
        scrape = _seconds(config["global"]["scrape_interval"])
        refresh = _seconds(DASHBOARD["refresh"])
        assert scrape <= refresh <= self._interval_seconds()
        assert DASHBOARD["refresh"] in DASHBOARD["timepicker"]["refresh_intervals"]

    def test_lag_threshold_is_one_full_poll(self) -> None:
        max_poll_records = load_streaming_params(PARAMS_PATH).max_poll_records
        for title in ("Lag total", "Lag por particion"):
            assert _step_value(_panel(title), "yellow") == max_poll_records

    def test_heartbeat_thresholds_are_interval_and_staleness(self) -> None:
        panel = _panel("Edad del latido")
        assert _step_value(panel, "yellow") == self._interval_seconds()
        assert _step_value(panel, "red") == load_streaming_params(PARAMS_PATH).staleness_seconds

    def test_alarm_stats_turn_red_at_the_first_event(self) -> None:
        for title in ("Fallos de commit", "Mensajes venenosos", "Errores de publicacion"):
            assert _step_value(_panel(title), "red") == 1


# ---------------------------------------------------------------------------
# Despliegue: ConfigMap para el sidecar y provisionado local
# ---------------------------------------------------------------------------


class TestDeployment:
    @staticmethod
    def _kustomization() -> dict[str, Any]:
        return yaml.safe_load(KUSTOMIZATION_PATH.read_text(encoding="utf-8"))

    def test_config_map_carries_the_dashboard_file(self) -> None:
        generators = self._kustomization()["configMapGenerator"]
        assert len(generators) == 1
        assert generators[0]["files"] == [DASHBOARD_PATH.name]

    def test_config_map_has_the_sidecar_label_and_a_stable_name(self) -> None:
        options = self._kustomization()["generatorOptions"]
        assert options["labels"] == {"grafana_dashboard": "1"}
        assert options["disableNameSuffixHash"] is True

    def test_namespace_exists_in_the_base_manifests(self) -> None:
        namespaces = {
            doc["metadata"]["name"]
            for doc in yaml.safe_load_all(NAMESPACES_PATH.read_text(encoding="utf-8"))
            if doc and doc.get("kind") == "Namespace"
        }
        assert self._kustomization()["namespace"] in namespaces

    def test_compose_mounts_what_the_provider_declares(self) -> None:
        compose = yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))
        mounts: dict[Path, str] = {}
        for volume in compose["services"]["grafana"]["volumes"]:
            host, container, *_ = volume.split(":")
            mounts[(COMPOSE_PATH.parent / host).resolve()] = container

        provisioning = yaml.safe_load(PROVISIONING_PATH.read_text(encoding="utf-8"))
        provider = provisioning["providers"][0]
        assert mounts[PROVISIONING_PATH.resolve()].startswith(
            "/etc/grafana/provisioning/dashboards/"
        )
        assert mounts[DASHBOARDS_DIR.resolve()] == provider["options"]["path"]
        assert provider["type"] == "file"


# ---------------------------------------------------------------------------
# El propio analizador tiene dientes
# ---------------------------------------------------------------------------


class TestAnalyzer:
    def test_parses_a_histogram_quantile(self) -> None:
        parsed = parse_promql(
            "histogram_quantile(0.99, sum by (le) "
            "(rate(reactorguard_validation_latency_seconds_bucket[$rate_window])))"
        )
        assert parsed.metrics == ("reactorguard_validation_latency_seconds_bucket",)
        assert parsed.group_labels == ("le",)
        assert parsed.ranges == ("$rate_window",)

    def test_parses_matchers_and_ignores_quoted_values_as_names(self) -> None:
        parsed = parse_promql('sum(rate(reactorguard_x_total{topic="sensor-readings-raw"}[5m]))')
        assert parsed.metrics == ("reactorguard_x_total",)
        assert parsed.matchers == (("topic", "=", "sensor-readings-raw"),)

    def test_flags_a_metric_that_is_not_exposed(self) -> None:
        assert unknown_metrics("sum(kafka_consumergroup_lag)") == ["kafka_consumergroup_lag"]

    def test_flags_a_suffix_the_metric_type_does_not_have(self) -> None:
        # batch_size es un histograma: no tiene _total; consumer_lag es un gauge: no
        # tiene _bucket.
        assert unknown_metrics("reactorguard_batch_size_total") != []
        assert unknown_metrics("reactorguard_consumer_lag_bucket") != []
        assert unknown_metrics("reactorguard_batch_size_bucket") == []
        assert unknown_metrics("reactorguard_poison_messages_total") == []

    def test_flags_a_label_the_metric_does_not_carry(self) -> None:
        expr = "sum by (sensor_id) (reactorguard_readings_validated_total)"
        assert unknown_labels(expr) == ["sensor_id"]
        assert unknown_labels("sum by (quality) (reactorguard_readings_validated_total)") == []

    def test_flags_an_illegal_label_value(self) -> None:
        assert invalid_label_values('reactorguard_readings_validated_total{quality="GOOD"}') == [
            'quality="GOOD"'
        ]
        assert invalid_label_values('x{fault_type=~"stuck|nope"}') == ['fault_type=~"nope"']
        assert invalid_label_values('x{fault_type!="kalman_anomaly"}') == []

    def test_exposed_series_cover_every_registered_metric(self) -> None:
        exposed = exposed_series()
        assert "reactorguard_stream_messages_produced_total" in exposed
        assert exposed["reactorguard_readings_validated_total"] == {"quality"}
        assert exposed["reactorguard_validation_latency_seconds_bucket"] == {"le"}
        assert exposed["reactorguard_consumer_lag"] == {"partition"}
