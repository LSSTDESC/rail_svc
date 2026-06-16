"""Tests for CLI local load and read_slice commands (dataset, estimates, model).

Uses Click's CliRunner with mocked local_sync operations.
"""

import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from rail_svc import models
from rail_svc.cli.local.base import dataset_group, estimates_group, model_group


@pytest.fixture
def runner():
    return CliRunner()


class TestDatasetLoad:
    """Test dataset load CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_with_fields(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test loading a dataset with KEY=VALUE fields."""
        data_file = tmp_path / "catalog.hdf5"
        data_file.write_bytes(b"data")

        mock_result = models.Dataset(
            id_=1, name="my_ds", path=str(data_file), n_objects=100, is_collection=False, catalog_tag_id=1
        )
        mock_local_sync.dataset.load.return_value = mock_result
        mock_local_sync.dataset.ctx.response_class.col_names_for_table = ["id_", "name", "path"]

        result = runner.invoke(
            dataset_group,
            ["load", "--path", str(data_file), "--output", "json", "catalog_tag_name=lsst"],
        )

        assert result.exit_code == 0
        assert "Successfully loaded dataset" in result.output
        mock_local_sync.dataset.load.assert_called_once()

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_from_json(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test loading a dataset from a JSON file."""
        data_file = tmp_path / "catalog.hdf5"
        data_file.write_bytes(b"data")

        json_file = tmp_path / "params.json"
        json_file.write_text(json.dumps({"catalog_tag_name": "lsst", "is_collection": False}))

        mock_result = models.Dataset(
            id_=1, name="my_ds", path=str(data_file), n_objects=100, is_collection=False, catalog_tag_id=1
        )
        mock_local_sync.dataset.load.return_value = mock_result
        mock_local_sync.dataset.ctx.response_class.col_names_for_table = ["id_", "name"]

        result = runner.invoke(
            dataset_group,
            ["load", "--path", str(data_file), "--from-json", str(json_file), "--output", "json"],
        )

        assert result.exit_code == 0
        assert "Successfully loaded dataset" in result.output

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_no_validate(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test loading a dataset with --no-validate."""
        data_file = tmp_path / "catalog.hdf5"
        data_file.write_bytes(b"data")

        mock_result = models.Dataset(
            id_=1, name="ds", path=str(data_file), n_objects=50, is_collection=False, catalog_tag_id=1
        )
        mock_local_sync.dataset.load.return_value = mock_result
        mock_local_sync.dataset.ctx.response_class.col_names_for_table = ["id_", "name"]

        result = runner.invoke(
            dataset_group,
            [
                "load",
                "--path",
                str(data_file),
                "--no-validate",
                "--output",
                "json",
                "catalog_tag_name=test",
                "n_objects=50",
            ],
        )

        assert result.exit_code == 0
        call_kwargs = mock_local_sync.dataset.load.call_args
        assert call_kwargs[1]["validate"] is False

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_error(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test load error handling."""
        data_file = tmp_path / "bad.hdf5"
        data_file.write_bytes(b"bad")

        mock_local_sync.dataset.load.side_effect = ValueError("Invalid file")

        result = runner.invoke(
            dataset_group,
            ["load", "--path", str(data_file), "--output", "json", "catalog_tag_name=x"],
        )

        assert result.exit_code != 0
        assert "Error loading dataset" in result.output


class TestDatasetReadSlice:
    """Test dataset read-slice CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_read_slice_json(self, mock_local_sync, mock_init, runner):
        """Test reading a dataset slice with JSON output."""
        mock_local_sync.dataset.read_slice.return_value = {"mag_g": [22.5, 23.1]}

        result = runner.invoke(dataset_group, ["read-slice", "--output", "json", "1"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "mag_g" in output

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_read_slice_with_slice(self, mock_local_sync, mock_init, runner):
        """Test reading a dataset slice with --slice option."""
        mock_local_sync.dataset.read_slice.return_value = {"flux": [1.0, 2.0]}

        result = runner.invoke(dataset_group, ["read-slice", "--slice", "0:2", "--output", "json", "5"])

        assert result.exit_code == 0

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_read_slice_error(self, mock_local_sync, mock_init, runner):
        """Test read_slice error handling."""
        mock_local_sync.dataset.read_slice.side_effect = FileNotFoundError("Dataset file not found")

        result = runner.invoke(dataset_group, ["read-slice", "--output", "json", "999"])

        assert result.exit_code != 0


class TestEstimatesLoad:
    """Test estimates load CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_with_fields(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test loading estimates with KEY=VALUE fields."""
        data_file = tmp_path / "estimates.hdf5"
        data_file.write_bytes(b"data")

        mock_result = models.Estimates(
            id_=1, name="est", path=str(data_file), n_objects=100, dataset_id=1, estimator_id=1
        )
        mock_local_sync.estimates.load.return_value = mock_result
        mock_local_sync.estimates.ctx.response_class.col_names_for_table = ["id_", "name"]

        result = runner.invoke(
            estimates_group,
            ["load", "--path", str(data_file), "--output", "json", "dataset_name=ds", "estimator_name=bpz"],
        )

        assert result.exit_code == 0
        assert "Successfully loaded estimates" in result.output

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_error(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test estimates load error handling."""
        data_file = tmp_path / "bad.hdf5"
        data_file.write_bytes(b"bad")

        mock_local_sync.estimates.load.side_effect = ValueError("Bad format")

        result = runner.invoke(
            estimates_group,
            ["load", "--path", str(data_file), "--output", "json", "dataset_name=ds", "estimator_name=bpz"],
        )

        assert result.exit_code != 0
        assert "Error loading estimates" in result.output


class TestEstimatesReadSlice:
    """Test estimates read-slice CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_read_slice_json(self, mock_local_sync, mock_init, runner):
        """Test reading estimates slice with JSON output."""
        mock_local_sync.estimates.read_slice.return_value = {"z_pdf": [0.1, 0.5, 0.3]}

        result = runner.invoke(estimates_group, ["read-slice", "--output", "json", "1"])

        assert result.exit_code == 0

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_read_slice_error(self, mock_local_sync, mock_init, runner):
        """Test read_slice error handling."""
        mock_local_sync.estimates.read_slice.side_effect = FileNotFoundError("File not found")

        result = runner.invoke(estimates_group, ["read-slice", "--output", "json", "999"])

        assert result.exit_code != 0


class TestModelLoad:
    """Test model load CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_with_fields(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test loading a model with KEY=VALUE fields."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"model")

        mock_result = models.Model(id_=1, name="my_model", path=str(model_file), algo_id=1, catalog_tag_id=1)
        mock_local_sync.model.load.return_value = mock_result
        mock_local_sync.model.ctx.response_class.col_names_for_table = ["id_", "name"]

        result = runner.invoke(
            model_group,
            [
                "load",
                "--path",
                str(model_file),
                "--output",
                "json",
                "algo_name=BPZEstimator",
                "catalog_tag_name=lsst",
            ],
        )

        assert result.exit_code == 0
        assert "Successfully loaded model" in result.output

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_with_link(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test loading a model with --load-type link."""
        model_file = tmp_path / "model.pkl"
        model_file.write_bytes(b"model")

        mock_result = models.Model(
            id_=1, name="linked", path="models/linked_model.pkl", algo_id=1, catalog_tag_id=1
        )
        mock_local_sync.model.load.return_value = mock_result
        mock_local_sync.model.ctx.response_class.col_names_for_table = ["id_", "name"]

        result = runner.invoke(
            model_group,
            [
                "load",
                "--path",
                str(model_file),
                "--load-type",
                "link",
                "--output",
                "json",
                "algo_name=RF",
                "catalog_tag_name=lsst",
            ],
        )

        assert result.exit_code == 0

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.local_sync")
    def test_load_error(self, mock_local_sync, mock_init, runner, tmp_path):
        """Test model load error handling."""
        model_file = tmp_path / "bad.pkl"
        model_file.write_bytes(b"bad")

        mock_local_sync.model.load.side_effect = ValueError("CatalogTag mismatch")

        result = runner.invoke(
            model_group,
            ["load", "--path", str(model_file), "--output", "json", "algo_name=X", "catalog_tag_name=Y"],
        )

        assert result.exit_code != 0
        assert "Error loading model" in result.output
