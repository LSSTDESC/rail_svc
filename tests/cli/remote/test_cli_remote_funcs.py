"""Tests for CLI remote funcs commands.

Uses Click's CliRunner with mocked remote_sync.funcs to test CLI behavior.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rail_svc import models
from rail_svc.cli.remote.funcs import funcs_group


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_sync_funcs():
    """Mock remote_sync.funcs() to return a mock SyncRemoteFuncs."""
    mock = MagicMock()
    with patch("rail_svc.cli.remote.funcs.remote_sync.funcs", return_value=mock):
        yield mock


class TestEstimatePdf:
    """Test estimate-pdf CLI command."""

    def test_success_json(self, runner, mock_sync_funcs):
        """Test successful PDF estimation with JSON output."""
        mock_sync_funcs.estimate_pdf.return_value = {"z": [0.1, 0.5], "pdf": [0.3, 0.7]}

        result = runner.invoke(
            funcs_group,
            ["estimate-pdf", "--estimator-id", "1", "--dataset-id", "2", "--row", "5", "--output", "json"],
        )

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert "z" in output

    def test_success_table(self, runner, mock_sync_funcs):
        """Test successful PDF estimation with table output."""
        mock_sync_funcs.estimate_pdf.return_value = {"z": [0.1], "pdf": [1.0]}

        result = runner.invoke(
            funcs_group,
            ["estimate-pdf", "--estimator-id", "1", "--dataset-id", "2", "--row", "0", "--output", "table"],
        )

        assert result.exit_code == 0

    def test_error_handling(self, runner, mock_sync_funcs):
        """Test error handling."""
        mock_sync_funcs.estimate_pdf.side_effect = RuntimeError("Connection failed")

        result = runner.invoke(
            funcs_group,
            ["estimate-pdf", "--estimator-id", "1", "--dataset-id", "2", "--row", "0", "--output", "json"],
        )

        assert result.exit_code != 0


class TestGetEstimatorsForDataset:
    """Test get-estimators-for-dataset CLI command."""

    def test_success(self, runner, mock_sync_funcs):
        """Test successful estimator retrieval."""
        mock_sync_funcs.get_estimators_for_dataset.return_value = [
            {"id_": 1, "name": "bpz", "config": {}, "model_id": 1},
        ]

        result = runner.invoke(
            funcs_group,
            ["get-estimators-for-dataset", "--dataset-id", "5", "--output", "json"],
        )

        assert result.exit_code == 0


class TestGetDatasetAndEstimates:
    """Test get-dataset-and-estimates CLI command."""

    def test_success(self, runner, mock_sync_funcs):
        """Test successful dataset+estimates retrieval."""
        mock_dataset = models.Dataset(
            id_=1, name="ds", path="/d.hdf5", n_objects=100, is_collection=False, catalog_tag_id=1
        )
        mock_response = models.GetDatasetAndEstimatesResponse(dataset=mock_dataset, estimates={})
        mock_sync_funcs.get_dataset_and_estimates.return_value = mock_response

        result = runner.invoke(
            funcs_group,
            ["get-dataset-and-estimates", "--dataset-id", "1", "--output", "json"],
        )

        assert result.exit_code == 0


class TestEstimateEnsemble:
    """Test estimate-ensemble CLI command."""

    def test_success(self, runner, mock_sync_funcs):
        """Test successful ensemble estimation."""
        mock_response = MagicMock()
        mock_response.message = "Estimation complete: /output/est.hdf5"
        mock_sync_funcs.estimate_ensemble.return_value = mock_response

        result = runner.invoke(
            funcs_group,
            [
                "estimate-ensemble",
                "--estimator-id",
                "1",
                "--dataset-id",
                "2",
                "--output-path",
                "/output/est.hdf5",
            ],
        )

        assert result.exit_code == 0
        assert "Estimation complete" in result.output


class TestLoadCatalogYaml:
    """Test load-catalog-yaml CLI command."""

    def test_success(self, runner, mock_sync_funcs, tmp_path):
        """Test successful catalog YAML loading."""
        yaml_file = tmp_path / "catalog.yaml"
        yaml_file.write_text("# test")
        filter_dir = tmp_path / "filters"
        filter_dir.mkdir()

        mock_response = MagicMock()
        mock_response.bands = [
            {"id_": 1, "name": "g", "band_wavelengths": [400.0], "band_transmission": [0.9]}
        ]
        mock_response.catalog_tags = [{"id_": 1, "name": "lsst"}]
        mock_response.catalog_band_assocs = [
            {
                "id_": 1,
                "catalog_tag_id": 1,
                "band_id": 1,
                "mag_column_name": "mag_g",
                "mag_err_column_name": "err_g",
            }
        ]
        mock_sync_funcs.load_catalog_yaml.return_value = mock_response

        result = runner.invoke(
            funcs_group,
            [
                "load-catalog-yaml",
                "--catalog-yaml",
                str(yaml_file),
                "--filter-dir",
                str(filter_dir),
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0


class TestGetDataAndEstimatesData:
    """Test get-data-and-estimates-data CLI command."""

    def test_success_json(self, runner, mock_sync_funcs):
        """Test successful retrieval with JSON output."""
        mock_response = MagicMock()
        mock_response.data = {"mag_g": [20.1], "mag_r": [19.5]}
        mock_response.estimates_dict = {"bpz": {"z": [0.1], "pdf": [1.0]}}
        mock_sync_funcs.get_data_and_estimates_data.return_value = mock_response

        result = runner.invoke(
            funcs_group,
            ["get-data-and-estimates-data", "--dataset-id", "1", "--row", "42", "--output", "json"],
        )

        assert result.exit_code == 0

    def test_success_table(self, runner, mock_sync_funcs):
        """Test successful retrieval with table output."""
        mock_response = MagicMock()
        mock_response.data = {"mag_g": [20.1]}
        mock_response.estimates_dict = {"bpz": "some data"}
        mock_sync_funcs.get_data_and_estimates_data.return_value = mock_response

        result = runner.invoke(
            funcs_group,
            ["get-data-and-estimates-data", "--dataset-id", "1", "--row", "0", "--output", "table"],
        )

        assert result.exit_code == 0


class TestEstimatePdfForSlice:
    """Test estimate-pdf-for-slice CLI command."""

    def test_success_json(self, runner, mock_sync_funcs):
        """Test successful slice estimation with JSON output."""
        mock_sync_funcs.estimate_pdf_for_slice.return_value = {"z": [0.1], "pdf": [1.0]}

        result = runner.invoke(
            funcs_group,
            [
                "estimate-pdf-for-slice",
                "--estimator-id",
                "1",
                "--dataset-id",
                "2",
                "--slice",
                "0:10",
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0

    def test_success_table(self, runner, mock_sync_funcs):
        """Test successful slice estimation with table output."""
        mock_sync_funcs.estimate_pdf_for_slice.return_value = {"z": [0.1], "pdf": [1.0]}

        result = runner.invoke(
            funcs_group,
            [
                "estimate-pdf-for-slice",
                "--estimator-id",
                "1",
                "--dataset-id",
                "2",
                "--slice",
                "0:10",
                "--output",
                "table",
            ],
        )

        assert result.exit_code == 0


class TestEstimateDataset:
    """Test estimate-dataset CLI command."""

    def test_success(self, runner, mock_sync_funcs):
        """Test successful dataset estimation."""
        mock_sync_funcs.estimate_dataset.return_value = {
            "id_": 1,
            "name": "est",
            "path": "/est.hdf5",
            "n_objects": 100,
            "dataset_id": 1,
            "estimator_id": 1,
        }

        result = runner.invoke(
            funcs_group,
            [
                "estimate-dataset",
                "--estimator-id",
                "1",
                "--dataset-id",
                "2",
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0


class TestCreateMatchedDataset:
    """Test create-matched-dataset CLI command."""

    def test_success(self, runner, mock_sync_funcs, tmp_path):
        """Test successful matched dataset creation."""
        path_file = tmp_path / "matched.hdf5"
        path_file.write_bytes(b"data")

        mock_sync_funcs.create_matched_dataset.return_value = {
            "id_": 10,
            "name": "matched",
            "path": str(path_file),
            "n_objects": 1000,
            "is_collection": True,
            "catalog_tag_id": 1,
        }

        result = runner.invoke(
            funcs_group,
            [
                "create-matched-dataset",
                "--matched-dataset-name",
                "matched",
                "--catalog-tag-name",
                "lsst",
                "--component-dataset-names",
                "comp1",
                "--component-dataset-names",
                "comp2",
                "--path",
                str(path_file),
                "--n-objects",
                "1000",
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0
