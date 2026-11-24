"""Unit tests for ml.training.checkpoint (contrato de model.pt, convencion 5 y H5).

El checkpoint se escribe y se lee con `weights_only=True`; sus bytes no dependen
del directorio ni de la ejecucion, solo del contenido y del nombre del fichero.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from ml.models.pinn import ReactorPINN
from ml.training.checkpoint import (
    CARD_FILE,
    CONTRACT_VERSION,
    MODEL_FILE,
    checkpoint_document,
    load_checkpoint,
    save_checkpoint,
    write_json,
)
from ml.training.pinn_data import CONTEXT_NAMES, NormalizationStats

_TAGS = ("A", "B", "C")


def _stats() -> NormalizationStats:
    return NormalizationStats(
        tags=_TAGS,
        sensor_mean=(1.0, 2.0, 3.0),
        sensor_std=(0.5, 0.25, 2.0),
        context_mean=(0.9, 0.0, 0.1, -0.1),
        context_std=(0.1, 0.2, 0.7, 0.7),
    )


def _model(seed: int = 0) -> ReactorPINN:
    generator = torch.Generator().manual_seed(seed)
    model = ReactorPINN(len(_TAGS), hidden=8, n_hidden=2, dropout=0.1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return model


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestDocument:
    """What model.pt carries."""

    def test_keys_and_primitives(self) -> None:
        """Contract version, kwargs, weights, column order and z-score."""
        document = checkpoint_document(_model(), _stats())
        assert document["contract_version"] == CONTRACT_VERSION
        assert document["model_kwargs"] == {
            "n_sensors": 3,
            "n_context": 4,
            "hidden": 8,
            "n_hidden": 2,
            "dropout": 0.1,
        }
        assert document["sensor_tags"] == list(_TAGS)
        assert document["context_names"] == list(CONTEXT_NAMES)
        assert document["sensor_std"] == [0.5, 0.25, 2.0]
        assert all(isinstance(v, torch.Tensor) for v in document["state_dict"].values())

    def test_width_mismatch_is_rejected(self) -> None:
        """A model for 4 sensors cannot be saved with statistics for 3."""
        with pytest.raises(ValueError, match="stats describe"):
            checkpoint_document(ReactorPINN(4, hidden=8), _stats())


class TestRoundTrip:
    """save -> load with weights_only=True."""

    def test_load_rebuilds_the_same_network(self, tmp_path: Path) -> None:
        """Same predictions and statistics after reloading, in eval mode."""
        model = _model()
        path = save_checkpoint(tmp_path / "out" / MODEL_FILE, model, _stats())
        loaded = load_checkpoint(path)
        model.eval()
        x = torch.randn(5, model.n_inputs, generator=torch.Generator().manual_seed(1))
        torch.testing.assert_close(loaded.model(x), model(x))
        assert loaded.stats == _stats()
        assert not loaded.model.training

    def test_plain_torch_load_with_weights_only(self, tmp_path: Path) -> None:
        """The file only holds tensors and primitives."""
        path = save_checkpoint(tmp_path / MODEL_FILE, _model(), _stats())
        document = torch.load(path, weights_only=True)
        assert set(document) >= {"state_dict", "model_kwargs", "sensor_mean"}

    def test_missing_file(self, tmp_path: Path) -> None:
        """A missing checkpoint is a FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            load_checkpoint(tmp_path / MODEL_FILE)

    def test_foreign_document_is_rejected(self, tmp_path: Path) -> None:
        """A bare state_dict is not a PINN checkpoint."""
        path = tmp_path / MODEL_FILE
        torch.save(_model().state_dict(), path)
        with pytest.raises(ValueError, match="does not follow"):
            load_checkpoint(path)

    def test_other_contract_version_is_rejected(self, tmp_path: Path) -> None:
        """A version bump means the reader must be updated."""
        document = checkpoint_document(_model(), _stats())
        document["contract_version"] = CONTRACT_VERSION + 1
        path = tmp_path / MODEL_FILE
        torch.save(document, path)
        with pytest.raises(ValueError, match="contract version"):
            load_checkpoint(path)

    def test_other_context_is_rejected(self, tmp_path: Path) -> None:
        """The context names are part of the contract."""
        document = checkpoint_document(_model(), _stats())
        document["context_names"] = ["a", "b", "c", "d"]
        path = tmp_path / MODEL_FILE
        torch.save(document, path)
        with pytest.raises(ValueError, match="context"):
            load_checkpoint(path)

    def test_statistics_of_the_wrong_size_are_rejected(self, tmp_path: Path) -> None:
        """One mean per sensor."""
        document = checkpoint_document(_model(), _stats())
        document["sensor_mean"] = [1.0]
        path = tmp_path / MODEL_FILE
        torch.save(document, path)
        with pytest.raises(ValueError, match="sensor_mean"):
            load_checkpoint(path)

    def test_tags_must_match_the_model(self, tmp_path: Path) -> None:
        """The sensor order has one tag per model output."""
        document = checkpoint_document(_model(), _stats())
        for key in ("sensor_mean", "sensor_std"):
            document[key] = [1.0, 1.0]
        document["sensor_tags"] = ["A", "B"]
        path = tmp_path / MODEL_FILE
        torch.save(document, path)
        with pytest.raises(ValueError, match="sensors"):
            load_checkpoint(path)


class TestByteStability:
    """H5: same content -> same bytes, whatever the directory or the call."""

    def test_two_saves_in_two_directories_are_identical(self, tmp_path: Path) -> None:
        """The zip has fixed dates and a content-derived serialization id."""
        first = save_checkpoint(tmp_path / "a" / MODEL_FILE, _model(), _stats())
        second = save_checkpoint(tmp_path / "b" / MODEL_FILE, _model(), _stats())
        assert _digest(first) == _digest(second)

    def test_other_weights_give_other_bytes(self, tmp_path: Path) -> None:
        """The digest does follow the content."""
        first = save_checkpoint(tmp_path / "a" / MODEL_FILE, _model(0), _stats())
        second = save_checkpoint(tmp_path / "b" / MODEL_FILE, _model(1), _stats())
        assert _digest(first) != _digest(second)


class TestWriteJson:
    """model_card.json and the metrics: sorted keys, LF, no NaN."""

    def test_sorted_with_trailing_newline(self, tmp_path: Path) -> None:
        """Keys are sorted and the file ends in a newline."""
        path = write_json({"b": 1, "a": {"d": 2, "c": 3}}, tmp_path / "x" / CARD_FILE)
        text = path.read_bytes().decode("utf-8")
        assert text.endswith("}\n") and "\r" not in text
        assert list(json.loads(text)) == ["a", "b"]
        assert text.index('"c"') < text.index('"d"')

    def test_nan_is_rejected(self, tmp_path: Path) -> None:
        """A NaN metric is a bug, not a value to publish."""
        with pytest.raises(ValueError):
            write_json({"a": float("nan")}, tmp_path / "x.json")
