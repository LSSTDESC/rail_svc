"""Tests for CLI remote load, read_slice, and download commands.

Uses Click's CliRunner with mocked remote_sync operations.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rail_svc import models
from rail_svc.cli.remote.base import dataset_group, estimates_group, model_group


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_dataset_ops():
    """Mock remote_sync.dataset() to return a mock ops object."""
    mock_ops = MagicMock()
    mock_ops.ctx.response_class.col_names_for_table = ["id_", "name", "path"]
    with patch("rail_svc.cli.remote.base.remote_sync.dataset", return_value=mock_ops):
        yield mock_ops


@pytest.fixture
def mock_estimates_ops():
    """Mock remote_sync.estimates() to return a mock ops object."""
    mock_ops = MagicMock()
    mock_ops.ctx.response_class.col_names_for_table = ["id_", "name", "path"]
    with patch("rail_svc.cli.remote.base.remote_sync.estimates", return_value=mock_ops):
        yield mock_ops


@pytest.fixture
def mock_model_ops():
    """Mock remote_sync.model() to return a mock ops object."""
    mock_ops = MagicMock()
    mock_ops.ctx.response_class.col_names_for_table = ["id_", "name", "path"]
    with patch("rail_svc.cli.remote.base.remote_sync.model", return_value=mock_ops):
        yield mock_ops


class TestDatasetLoad:
    """Test remote dataset load CLI command."""

    def test_load_with_fields(self, runner, mock_dataset_ops, tmp_path):
        """Test loading a dataset with KEY=VALUE fields."""
        data_file = tmp_path / "catalog.hdf5"
        data_file.write_bytes(b"data")

        mock_result = models.Dataset(
            id_=1, name="ds", path=str(data_file), n_objects=100, is_collection=False, catalog_tag_id=1
        )
        mock_dataset_ops.load.return_value = mock_result

        result = runner.invoke(
            dataset_group,
            ["load", "--path", str(data_file), "--output", "json", "catalog_tag_name=lsst"],
        )

        assert result.exit_code == 0
        assert "Successfully loaded dataset" in result.output
        mock_dataset_ops.load.assert_called_once()

    def test_load_from_json(self, runner, mock_dataset_ops, tmp_path):
        """Test loading a dataset from a JSON file."""
        data_file = tmp_path / "catalog.hdf5"
        data_file.write_bytes(b"data")

        json_file = tmp_path / "params.json"
        json_file.write_text(json.dumps({"catalog_tag_name": "lsst"}))

        mock_result = models.Dataset(
            id_=1, name="ds", path=str(data_file), n_objects=100, is_collection=False, catalog_tag_id=1
        )
        mock_dataset_ops.load.return_value = mock_result

        result = runner.invoke(
            dataset_group,
            ["load", "--path", str(data_file), "--from-json", str(json_file), "--output", "json"],
        )

        assert result.exit_code == 0

    def test_load_error(self, runner, mock_dataset_ops, tmp_path):
        """Test load error handling."""
        data_file = tmp_path / "bad.hdf5"
        data_file.write_bytes(b"bad")

        mock_dataset_ops.load.side_effect = RuntimeError("Connection refused")

        result = runner.invoke(
            dataset_group,
            ["load", "--path", str(data_file), "--output", "json", "catalog_tag_name=x"],
        )

        assert result.exit_code != 0
        assert "Error loading dataset" in result.output


class TestDatasetReadSlice:
    """Test remote dataset read-slice CLI command."""

    def test_read_slice_json(self, runner, mock_dataset_ops):
        """Test reading a dataset slice with JSON output."""
        mock_dataset_ops.read_slice.return_value = {"mag_g": [22.5, 23.1]}

        result = runner.invoke(dataset_group, ["read-slice", "--output", "json", "1"])

        assert result.exit_code == 0

    def test_read_slice_with_slice_option(self, runner, mock_dataset_ops):
        """Test reading with --slice option."""
        mock_dataset_ops.read_slice.return_value = {"flux": [1.0]}

        result = runner.invoke(dataset_group, ["read-slice", "--slice", "0:5", "--output", "json", "3"])

        assert result.exit_code == 0

    def test_read_slice_error(self, runner, mock_dataset_ops):
        """Test read_slice error handling."""
        mock_dataset_ops.read_slice.side_effect = RuntimeError("Not found")

        result = runner.invoke(dataset_group, ["read-slice", "--output", "json", "999"])

        assert result.exit_code != 0


class TestDatasetDownload:
    """Test remote dataset download CLI command."""

    def test_download_success(self, runner, mock_dataset_ops):
        """Test successful download."""
        mock_dataset_ops.download.return_value = Path("/downloads/dataset_1.hdf5")

        result = runner.invoke(dataset_group, ["download", "1"])

        assert result.exit_code == 0
        assert "Successfully downloaded" in result.output

    def test_download_with_output_path(self, runner, mock_dataset_ops):
        """Test download with custom output path."""
        mock_dataset_ops.download.return_value = Path("/custom/output.hdf5")

        result = runner.invoke(dataset_group, ["download", "1", "--output-path", "/custom/output.hdf5"])

        assert result.exit_code == 0

    def test_download_error(self, runner, mock_dataset_ops):
        """Test download error handling."""
        mock_dataset_ops.download.side_effect = RuntimeError("Server error")

        result = runner.invoke(dataset_group, ["download", "1"])

        assert result.exit_code != 0
        assert "Error downloading" in result.output


class TestEstimatesLoad:
    """Test remote estimates load CLI command."""

    def test_load_with_fields(self, runner, mock_estimates_ops, tmp_path):
        """Test loading estimates with KEY=VALUE fields."""
        data_file = tmp_path / "est.hdf5"
        data_file.write_bytes(b"data")

        mock_result = models.Estimates(
            id_=1, name="est", path=str(data_file), n_objects=100, dataset_id=1, estimator_id=1
        )
        mock_estimates_ops.load.return_value = mock_result

        result = runner.invoke(
            estimates_group,
            ["load", "--path", str(data_file), "--output", "json", "dataset_name=ds", "estimator_name=bpz"],
        )

        assert result.exit_code == 0
        assert "Successfully loaded estimates" in result.output

    def test_load_error(self, runner, mock_estimates_ops, tmp_path):
        """Test estimates load error handling."""
        data_file = tmp_path / "bad.hdf5"
        data_file.write_bytes(b"bad")

        mock_estimates_ops.load.side_effect = ValueError("Bad format")

        result = runner.invoke(
            estimates_group,
            ["load", "--path", str(data_file), "--output", "json", "dataset_name=ds", "estimator_name=x"],
        )

        assert result.exit_code != 0


class TestEstimatesReadSlice:
    """Test remote estimates read-slice CLI command."""

    def test_read_slice(self, runner, mock_estimates_ops):
        """Test reading estimates slice."""
        mock_data = MagicMock()
        mock_data.to_json.return_value = '{"z_pdf": [0.1, 0.5]}'
        mock_estimates_ops.read_slice.return_value = mock_data

        result = runner.invoke(estimates_group, ["read-slice", "--output", "json", "1"])

        assert result.exit_code == 0

    def test_read_slice_error(self, runner, mock_estimates_ops):
        """Test read_slice error handling."""
        mock_estimates_ops.read_slice.side_effect = FileNotFoundError("Not found")

        result = runner.invoke(estimates_group, ["read-slice", "--output", "json", "999"])

        assert result.exit_code != 0


class TestEstimatesDownload:
    """Test remote estimates download CLI command."""

    def test_download_success(self, runner, mock_estimates_ops):
        """Test successful download."""
        mock_estimates_ops.download.return_value = Path("/downloads/est_1.hdf5")

        result = runner.invoke(estimates_group, ["download", "1"])

        assert result.exit_code == 0
        assert "Successfully downloaded" in result.output


class TestModelLoad:
    """Test remote model load CLI command."""

    def test_load_with_fields(self, runner, mock_model_ops, tmp_path):
        """Test loading a model with KEY=VALUE fields."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"model")

        mock_result = models.Model(
            id_=1, name="rf_model", path=str(model_file), algo_id=1, catalog_tag_id=1
        )
        mock_model_ops.load.return_value = mock_result

        result = runner.invoke(
            model_group,
            ["load", "--path", str(model_file), "--output", "json", "algo_name=RF", "catalog_tag_name=lsst"],
        )

        assert result.exit_code == 0
        assert "Successfully loaded model" in result.output

    def test_load_error(self, runner, mock_model_ops, tmp_path):
        """Test model load error handling."""
        model_file = tmp_path / "bad.pkl"
        model_file.write_bytes(b"bad")

        mock_model_ops.load.side_effect = ValueError("CatalogTag mismatch")

        result = runner.invoke(
            model_group,
            ["load", "--path", str(model_file), "--output", "json", "algo_name=X", "catalog_tag_name=Y"],
        )

        assert result.exit_code != 0


class TestModelDownload:
    """Test remote model download CLI command."""

    def test_download_success(self, runner, mock_model_ops):
        """Test successful download."""
        mock_model_ops.download.return_value = Path("/downloads/model_1.pkl")

        result = runner.invoke(model_group, ["download", "1"])

        assert result.exit_code == 0
        assert "Successfully downloaded" in result.output
