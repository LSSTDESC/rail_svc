"""Tests for CLI local funcs commands.

Uses Click's CliRunner with mocked local_sync.funcs to test CLI behavior.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from rail_svc.cli.local.funcs import funcs_group


@pytest.fixture
def runner():
    return CliRunner()


class TestEstimatePdf:
    """Test estimate-pdf CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.estimate_pdf")
    def test_success_json(self, mock_estimate, mock_init, runner):
        """Test successful PDF estimation with JSON output."""
        mock_estimate.return_value = {"z": [0.1, 0.5], "pdf": [0.3, 0.7]}

        result = runner.invoke(
            funcs_group,
            ["estimate-pdf", "--estimator-id", "1", "--dataset-id", "2", "--row", "5", "--output", "json"],
        )

        assert result.exit_code == 0
        mock_estimate.assert_called_once()

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.estimate_pdf")
    def test_error_handling(self, mock_estimate, mock_init, runner):
        """Test error handling."""
        mock_estimate.side_effect = ValueError("Estimator not found")

        result = runner.invoke(
            funcs_group,
            ["estimate-pdf", "--estimator-id", "999", "--dataset-id", "1", "--row", "0", "--output", "json"],
        )

        assert result.exit_code != 0


class TestEstimateEnsemble:
    """Test estimate-ensemble CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.estimate_ensemble")
    def test_success(self, mock_estimate, mock_init, runner, tmp_path):
        """Test successful ensemble estimation."""
        mock_estimate.return_value = Path("/output/estimates.hdf5")

        result = runner.invoke(
            funcs_group,
            [
                "estimate-ensemble",
                "--estimator-id", "1",
                "--dataset-id", "2",
                "--output-path", str(tmp_path / "est.hdf5"),
            ],
        )

        assert result.exit_code == 0
        mock_estimate.assert_called_once()


class TestGetEstimatorsForDataset:
    """Test get-estimators-for-dataest CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.get_estimators_for_dataest")
    def test_success(self, mock_get, mock_init, runner):
        """Test successful estimator retrieval."""
        from rail_svc import models

        mock_get.return_value = [
            models.Estimator(id_=1, name="bpz", config={}, model_id=1),
        ]

        result = runner.invoke(
            funcs_group,
            ["get-estimators-for-dataest", "--dataset-id", "5", "--output", "json"],
        )

        assert result.exit_code == 0


class TestGetDatasetAndEstimates:
    """Test get-dataset-and-estimates CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.get_dataset_and_estimates")
    def test_success(self, mock_get, mock_init, runner):
        """Test successful dataset+estimates retrieval."""
        from rail_svc import models

        mock_dataset = models.Dataset(
            id_=1, name="ds", path="/d.hdf5", n_objects=100, is_collection=False, catalog_tag_id=1
        )
        mock_estimates = [
            models.Estimates(id_=1, name="e1", path="/e.hdf5", n_objects=100, dataset_id=1, estimator_id=1)
        ]
        mock_get.return_value = (mock_dataset, mock_estimates)

        result = runner.invoke(
            funcs_group,
            ["get-dataset-and-estimates", "--dataset-id", "1", "--output", "json"],
        )

        assert result.exit_code == 0


class TestCreateMatchedDataset:
    """Test create-matched-dataset CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.create_matched_dataset")
    def test_success(self, mock_create, mock_init, runner, tmp_path):
        """Test successful matched dataset creation."""
        from rail_svc import models

        path_file = tmp_path / "matched.hdf5"
        path_file.write_bytes(b"data")

        mock_dataset = models.Dataset(
            id_=10, name="matched", path=str(path_file), n_objects=1000, is_collection=True, catalog_tag_id=1
        )
        mock_assocs = [
            models.DatasetAssoc(id_=1, name="matched_c1", matched_dataset_id=10, component_dataset_id=2)
        ]
        mock_create.return_value = (mock_dataset, mock_assocs)

        result = runner.invoke(
            funcs_group,
            [
                "create-matched-dataset",
                "--matched-dataset-name", "matched",
                "--catalog-tag-name", "lsst",
                "--component-dataset-names", "comp1",
                "--component-dataset-names", "comp2",
                "--path", str(path_file),
                "--n-objects", "1000",
                "--output", "json",
            ],
        )

        assert result.exit_code == 0


class TestLoadCatalogYaml:
    """Test load-catalog-yaml CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.load_catalog_yaml")
    def test_success(self, mock_load, mock_init, runner, tmp_path):
        """Test successful catalog YAML loading."""
        yaml_file = tmp_path / "catalog.yaml"
        yaml_file.write_text("bands: []")

        mock_load.return_value = ([], [], [])

        result = runner.invoke(
            funcs_group,
            ["load-catalog-yaml", "--catalog-yaml", str(yaml_file), "--output", "json"],
        )

        assert result.exit_code == 0
        mock_load.assert_called_once()


class TestGetDataAndEstimatesData:
    """Test get-data-and-estimates-data CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.get_data_and_estimates_data")
    def test_success_json(self, mock_get, mock_init, runner):
        """Test successful retrieval with JSON output."""
        mock_get.return_value = ({"mag_g": [22.5]}, {"bpz": {"z": [0.5]}})

        result = runner.invoke(
            funcs_group,
            ["get-data-and-estimates-data", "--dataset-id", "1", "--row", "0", "--output", "json"],
        )

        assert result.exit_code == 0

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.get_data_and_estimates_data")
    def test_success_table(self, mock_get, mock_init, runner):
        """Test successful retrieval with table output."""
        mock_get.return_value = ({"mag_g": [22.5]}, {})

        result = runner.invoke(
            funcs_group,
            ["get-data-and-estimates-data", "--dataset-id", "1", "--row", "0", "--output", "table"],
        )

        assert result.exit_code == 0


class TestEstimatePdfForSlice:
    """Test estimate-pdf-for-slice CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.estimate_pdf_for_slice")
    def test_success(self, mock_est, mock_init, runner):
        """Test successful slice estimation."""
        mock_est.return_value = {"z": [0.1, 0.5], "pdf": [0.3, 0.7]}

        result = runner.invoke(
            funcs_group,
            [
                "estimate-pdf-for-slice",
                "--estimator-id", "1",
                "--dataset-id", "2",
                "--slice", "0:10",
                "--output", "json",
            ],
        )

        assert result.exit_code == 0
        mock_est.assert_called_once()


class TestEstimateDataset:
    """Test estimate-dataset CLI command."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.local_sync.funcs.estimate_dataset")
    def test_success(self, mock_est, mock_init, runner):
        """Test successful dataset estimation."""
        from rail_svc import models

        mock_est.return_value = models.Estimates(
            id_=1, name="est", path="/e.hdf5", n_objects=100, dataset_id=1, estimator_id=1
        )

        result = runner.invoke(
            funcs_group,
            ["estimate-dataset", "--estimator-id", "1", "--dataset-id", "2", "--output", "json"],
        )

        assert result.exit_code == 0
