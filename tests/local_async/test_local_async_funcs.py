from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import qp

from rail_svc import models
from rail_svc.local_async import funcs as api_funcs
from rail_svc.rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper


@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.begin = MagicMock()
    session.begin.return_value.__aenter__ = AsyncMock()
    session.begin.return_value.__aexit__ = AsyncMock()
    return session


@pytest.fixture
def mock_get_session(mock_session):
    """Mock the get_session context manager."""
    with patch("rail_svc.local_async.base.get_session") as mock:
        mock.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock.return_value.__aexit__ = AsyncMock()
        yield mock


class TestEstimationFunctions:
    """Tests for estimation-related functions."""

    @pytest.mark.asyncio
    async def test_build_pdf_estimation_wrapper(self, mock_get_session, mock_session):
        """Test building PDF estimation wrapper."""
        estimator_id = 123
        expected_wrapper = MagicMock(spec=CatEstimatorPdfWrapper)

        with patch("rail_svc.db_oper.wrappers.build_pdf_estimation_wrapper") as mock_build:
            mock_build.return_value = expected_wrapper

            result = await api_funcs.build_pdf_estimation_wrapper(estimator_id)

            assert result == expected_wrapper
            mock_build.assert_called_once_with(mock_session, estimator_id)

    @pytest.mark.asyncio
    async def test_build_ensemble_estimation_wrapper(self, mock_get_session, mock_session):
        """Test building ensemble estimation wrapper."""
        estimator_id = 456
        expected_wrapper = MagicMock(spec=CatEstimatorEnsembleWrapper)

        with patch("rail_svc.db_oper.wrappers.build_ensemble_estimation_wrapper") as mock_build:
            mock_build.return_value = expected_wrapper

            result = await api_funcs.build_ensemble_estimation_wrapper(estimator_id)

            assert result == expected_wrapper
            mock_build.assert_called_once_with(mock_session, estimator_id)

    @pytest.mark.asyncio
    async def test_estimate_pdf(self, mock_get_session, mock_session):
        """Test PDF estimation."""
        estimator_id = 123
        dataset_id = 456
        row = 10
        expected_ensemble = MagicMock(spec=qp.Ensemble)

        with patch("rail_svc.db_oper.estimation_funcs.estimate_pdf") as mock_estimate:
            mock_estimate.return_value = expected_ensemble

            result = await api_funcs.estimate_pdf(estimator_id, dataset_id, row)

            assert result == expected_ensemble
            mock_estimate.assert_called_once_with(mock_session, estimator_id, dataset_id, row)

    @pytest.mark.asyncio
    async def test_estimate_ensemble(self, mock_get_session, mock_session):
        """Test ensemble estimation."""
        estimator_id = 123
        dataset_id = 456
        output_path = Path("/tmp/output.hdf5")
        expected_path = Path("/tmp/output.hdf5")

        with patch("rail_svc.db_oper.estimation_funcs.estimate_ensemble") as mock_estimate:
            mock_estimate.return_value = expected_path

            result = await api_funcs.estimate_ensemble(estimator_id, dataset_id, output_path)

            assert result == expected_path
            mock_estimate.assert_called_once_with(mock_session, estimator_id, dataset_id, output_path)

    @pytest.mark.asyncio
    async def test_build_cat_estimator_pdf_wrappers_for_dataset(self, mock_get_session, mock_session):
        """Test building PDF wrappers for dataset."""
        dataset_id = 789
        expected_wrappers = [
            MagicMock(spec=CatEstimatorPdfWrapper),
            MagicMock(spec=CatEstimatorPdfWrapper),
        ]

        with patch(
            "rail_svc.db_oper.estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset"
        ) as mock_build:
            mock_build.return_value = expected_wrappers

            result = await api_funcs.build_cat_estimator_pdf_wrappers_for_dataset(dataset_id)

            assert result == expected_wrappers
            mock_build.assert_called_once_with(mock_session, dataset_id)

    @pytest.mark.asyncio
    async def test_build_cat_estimator_ensemble_wrappers_for_dataset(self, mock_get_session, mock_session):
        """Test building ensemble wrappers for dataset."""
        dataset_id = 789
        expected_wrappers = [
            MagicMock(spec=CatEstimatorEnsembleWrapper),
            MagicMock(spec=CatEstimatorEnsembleWrapper),
        ]

        with patch(
            "rail_svc.db_oper.estimation_funcs.build_cat_estimator_ensemble_wrappers_for_dataset"
        ) as mock_build:
            mock_build.return_value = expected_wrappers

            result = await api_funcs.build_cat_estimator_ensemble_wrappers_for_dataset(dataset_id)

            assert result == expected_wrappers
            mock_build.assert_called_once_with(mock_session, dataset_id)


class TestCatalogFunctions:
    """Tests for catalog-related functions."""

    @pytest.mark.skip(reason="should load real file")
    @pytest.mark.asyncio
    async def test_load_catalog_yaml(self, mock_get_session, mock_session):
        """Test loading catalog from YAML."""
        catalog_yaml = Path("/path/to/catalog.yaml")
        filter_dir = Path("/path/to/filters")

        db_bands = [MagicMock(), MagicMock()]
        db_catalog_tags = [MagicMock()]
        db_models = [MagicMock(), MagicMock(), MagicMock()]

        pydantic_bands = [
            models.Band(
                id_=1,
                name="g",
                band_wavelengths=[400.0, 500.0, 600.0],
                band_transmission=[0.1, 0.9, 0.1],
            ),
            models.Band(
                id_=2,
                name="r",
                band_wavelengths=[500.0, 600.0, 700.0],
                band_transmission=[0.1, 0.9, 0.1],
            ),
        ]
        pydantic_tags = [models.CatalogTag(id_=1, name="test")]
        pydantic_assocs = [
            models.CatalogBandAssoc(
                id_=1,
                catalog_tag_id=1,
                band_id=1,
                mag_column_name="mag_g",
                mag_err_column_name="mag_err_g",
            ),
            models.CatalogBandAssoc(
                id_=2,
                catalog_tag_id=1,
                band_id=2,
                mag_column_name="mag_r",
                mag_err_column_name="mag_err_r",
            ),
            models.CatalogBandAssoc(
                id_=3,
                catalog_tag_id=1,
                band_id=3,
                mag_column_name="mag_i",
                mag_err_column_name="mag_err_i",
            ),
        ]
        with (
            patch("rail_svc.db_oper.catalog_funcs.load_catalog_yaml", new_callable=AsyncMock) as mock_load,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_band_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_tag_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_assoc_convert,
        ):

            mock_load.return_value = (db_bands, db_catalog_tags, db_models)
            mock_band_convert.return_value = pydantic_bands
            mock_tag_convert.return_value = pydantic_tags
            mock_assoc_convert.return_value = pydantic_assocs

            result = await api_funcs.load_catalog_yaml(catalog_yaml, filter_dir)

            assert result == (pydantic_bands, pydantic_tags, pydantic_assocs)
            mock_load.assert_called_once_with(mock_session, catalog_yaml, filter_dir)
            mock_band_convert.assert_called_once_with(db_bands)
            mock_tag_convert.assert_called_once_with(db_catalog_tags)
            mock_assoc_convert.assert_called_once_with(db_models)
            mock_session.begin.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_catalog_yaml_no_filter_dir(self, mock_get_session, mock_session):
        """Test loading catalog from YAML without filter directory."""
        catalog_yaml = Path("/path/to/catalog.yaml")

        db_bands = []
        db_catalog_tags = []
        db_models = []

        with (
            patch("rail_svc.db_oper.catalog_funcs.load_catalog_yaml", new_callable=AsyncMock) as mock_load,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_band_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_tag_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_assoc_convert,
        ):

            mock_load.return_value = (db_bands, db_catalog_tags, db_models)
            mock_band_convert.return_value = []
            mock_tag_convert.return_value = []
            mock_assoc_convert.return_value = []

            result = await api_funcs.load_catalog_yaml(catalog_yaml)

            assert result == ([], [], [])
            mock_load.assert_called_once_with(mock_session, catalog_yaml)
            mock_session.begin.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_catalog_row(self, mock_get_session, mock_session):
        """Test getting catalog row."""
        dataset_id = 123
        row = 5
        expected_data = {"flux": np.array([1.0, 2.0, 3.0]), "flux_err": np.array([0.1, 0.2, 0.3])}

        with patch("rail_svc.db_oper.catalog_funcs.get_catalog_row") as mock_get:
            mock_get.return_value = expected_data

            result = await api_funcs.get_catalog_row(dataset_id, row)

            assert result == expected_data
            mock_get.assert_called_once_with(mock_session, dataset_id, row)

    @pytest.mark.asyncio
    async def test_get_estimates_row(self, mock_get_session, mock_session):
        """Test getting estimates row."""
        estimates_id = 456
        row = 10
        expected_data = {"z_mean": np.array([0.5]), "z_mode": np.array([0.48])}

        with patch("rail_svc.db_oper.catalog_funcs.get_estimates_row") as mock_get:
            mock_get.return_value = expected_data

            result = await api_funcs.get_estimates_row(estimates_id, row)

            assert result == expected_data
            mock_get.assert_called_once_with(mock_session, estimates_id, row)

    @pytest.mark.skip(reason="Not working")
    @pytest.mark.asyncio
    async def test_get_dataset_and_estimates(self, mock_get_session, mock_session):
        """Test getting dataset and its estimates."""
        dataset_id = 789

        db_dataset = MagicMock()
        db_estimates = [MagicMock(), MagicMock()]

        pydantic_dataset = models.Dataset(
            id_=789,
            name="test_dataset",
            path=None,
            n_objects=500,
            is_collection=False,
            catalog_tag_id=1,
        )
        pydantic_estimates = [
            models.Estimates(id_=1, name="est1", dataset_id=789, estimator_id=1, path="/path/est1"),
            models.Estimates(id_=2, name="est2", dataset_id=789, estimator_id=2, path="/path/est2"),
        ]

        with (
            patch("rail_svc.db_oper.catalog_funcs.get_dataset_and_estimates") as mock_get,
            patch("rail_svc.local_async.base.to_pydantic") as mock_dataset_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_estimates_convert,
        ):

            mock_get.return_value = (db_dataset, db_estimates)
            mock_dataset_convert.return_value = pydantic_dataset
            mock_estimates_convert.return_value = pydantic_estimates

            result = await api_funcs.get_dataset_and_estimates(dataset_id)

            assert result == (pydantic_dataset, pydantic_estimates)
            mock_get.assert_called_once_with(mock_session, dataset_id)
            mock_dataset_convert.assert_called_once_with(db_dataset)
            mock_estimates_convert.assert_called_once_with(db_estimates)
            mock_session.begin.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_and_estimates_data(self, mock_get_session, mock_session):
        """Test getting data and estimates data."""
        dataset_id = 123
        row = 5

        catalog_data = {"flux": np.array([1.0, 2.0])}
        estimates_data = {"est1": MagicMock(spec=qp.Ensemble), "est2": MagicMock(spec=qp.Ensemble)}
        expected_result = (catalog_data, estimates_data)

        with patch("rail_svc.db_oper.catalog_funcs.get_data_and_estimates_data") as mock_get:
            mock_get.return_value = expected_result

            result = await api_funcs.get_data_and_estimates_data(dataset_id, row)

            assert result == expected_result
            mock_get.assert_called_once_with(mock_session, dataset_id, row)

    @pytest.mark.skip(reason="Not working")
    @pytest.mark.asyncio
    async def test_create_matched_dataset(self, mock_get_session, mock_session):
        """Test creating matched dataset."""
        matched_dataset_name = "matched_ds"
        catalog_tag_name = "test_tag"
        component_dataset_names = ["ds1", "ds2", "ds3"]
        path = "/path/to/matched"
        n_objects = 1000

        db_matched_dataset = MagicMock()
        db_dataset_assocs = [MagicMock(), MagicMock(), MagicMock()]

        pydantic_dataset = models.Dataset(
            id_=1,
            name=matched_dataset_name,
            path=path,
            n_objects=n_objects,
            is_collection=True,
            catalog_tag_id=1,
        )
        pydantic_assocs = [
            models.DatasetAssoc(
                id_=1,
                name="matched_ds_ds1",
                matched_dataset_id=1,
                component_dataset_id=10,
            ),
            models.DatasetAssoc(
                id_=2,
                name="matched_ds_ds2",
                matched_dataset_id=1,
                component_dataset_id=11,
            ),
            models.DatasetAssoc(
                id_=3,
                name="matched_ds_ds3",
                matched_dataset_id=1,
                component_dataset_id=12,
            ),
        ]

        with (
            patch("rail_svc.db_oper.catalog_funcs.create_matched_dataset") as mock_create,
            patch("rail_svc.local_async.base.to_pydantic") as mock_dataset_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_assoc_convert,
        ):

            mock_create.return_value = (db_matched_dataset, db_dataset_assocs)
            mock_dataset_convert.return_value = pydantic_dataset
            mock_assoc_convert.return_value = pydantic_assocs

            result = await api_funcs.create_matched_dataset(
                matched_dataset_name=matched_dataset_name,
                catalog_tag_name=catalog_tag_name,
                component_dataset_names=component_dataset_names,
                path=path,
                n_objects=n_objects,
            )

            assert result == (pydantic_dataset, pydantic_assocs)
            mock_create.assert_called_once_with(
                mock_session,
                matched_dataset_name=matched_dataset_name,
                catalog_tag_name=catalog_tag_name,
                component_dataset_names=component_dataset_names,
                path=path,
                n_objects=n_objects,
            )
            mock_dataset_convert.assert_called_once_with(db_matched_dataset)
            mock_assoc_convert.assert_called_once_with(db_dataset_assocs)
            mock_session.begin.assert_called_once()

    @pytest.mark.skip(reason="Not working")
    @pytest.mark.asyncio
    async def test_create_matched_dataset_no_path(self, mock_get_session, mock_session):
        """Test creating matched dataset without path."""
        matched_dataset_name = "matched_ds"
        catalog_tag_name = "test_tag"
        component_dataset_names = ["ds1"]
        n_objects = 100

        db_matched_dataset = MagicMock()
        db_dataset_assocs = [MagicMock()]

        pydantic_dataset = models.Dataset(
            id_=1,
            name=matched_dataset_name,
            path=None,
            n_objects=n_objects,
            is_collection=True,
            catalog_tag_id=1,
        )
        pydantic_assocs = [
            models.DatasetAssoc(
                id_=1,
                name="matched_ds_ds1",
                matched_dataset_id=1,
                component_dataset_id=10,
            )
        ]

        with (
            patch("rail_svc.db_oper.catalog_funcs.create_matched_dataset") as mock_create,
            patch("rail_svc.local_async.base.to_pydantic") as mock_dataset_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_assoc_convert,
        ):

            mock_create.return_value = (db_matched_dataset, db_dataset_assocs)
            mock_dataset_convert.return_value = pydantic_dataset
            mock_assoc_convert.return_value = pydantic_assocs

            result = await api_funcs.create_matched_dataset(
                matched_dataset_name=matched_dataset_name,
                catalog_tag_name=catalog_tag_name,
                component_dataset_names=component_dataset_names,
                path=None,
                n_objects=n_objects,
            )

            assert result == (pydantic_dataset, pydantic_assocs)
            mock_create.assert_called_once_with(
                mock_session,
                matched_dataset_name=matched_dataset_name,
                catalog_tag_name=catalog_tag_name,
                component_dataset_names=component_dataset_names,
                path=None,
                n_objects=n_objects,
            )
            mock_session.begin.assert_called_once()


class TestDecoratorBehavior:
    """Tests for the with_transaction decorator behavior."""

    @pytest.mark.asyncio
    async def test_decorator_creates_session(self):
        """Test that decorator creates and manages session."""
        mock_session = AsyncMock()

        with patch("rail_svc.local_async.base.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock()

            with patch("rail_svc.db_oper.estimation_funcs.estimate_pdf") as mock_estimate:
                mock_estimate.return_value = MagicMock(spec=qp.Ensemble)

                await api_funcs.estimate_pdf(1, 2, 3)

                # Verify session was created
                mock_get_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_decorator_passes_session_to_function(self):
        """Test that decorator passes session as first argument."""
        mock_session = AsyncMock()

        with patch("rail_svc.local_async.base.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock()

            with patch("rail_svc.db_oper.catalog_funcs.get_catalog_row") as mock_get_row:
                mock_get_row.return_value = {}

                await api_funcs.get_catalog_row(dataset_id=123, row=5)

                # Verify session was passed as first argument
                mock_get_row.assert_called_once_with(mock_session, dataset_id=123, row=5)

    @pytest.mark.asyncio
    async def test_decorator_propagates_exceptions(self):
        """Test that decorator propagates exceptions from wrapped function."""
        mock_session = AsyncMock()

        with patch("rail_svc.local_async.base.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch("rail_svc.db_oper.catalog_funcs.get_catalog_row") as mock_get_row:
                mock_get_row.side_effect = ValueError("Test error")

                with pytest.raises(ValueError, match="Test error"):
                    await api_funcs.get_catalog_row(dataset_id=123, row=5)

    @pytest.mark.asyncio
    async def test_decorator_handles_transaction_rollback(self):
        """Test that with_session_transaction handles errors properly."""
        mock_session = AsyncMock()
        mock_begin_context = MagicMock()
        mock_begin_context.__aenter__ = AsyncMock()
        mock_begin_context.__aexit__ = AsyncMock(return_value=False)
        mock_session.begin = MagicMock(return_value=mock_begin_context)

        with patch("rail_svc.local_async.base.get_session") as mock_get_session:
            mock_get_session.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_get_session.return_value.__aexit__ = AsyncMock(return_value=False)

            with patch(
                "rail_svc.db_oper.catalog_funcs.load_catalog_yaml", new_callable=AsyncMock
            ) as mock_load:
                mock_load.side_effect = RuntimeError("Database error")

                with pytest.raises(RuntimeError, match="Database error"):
                    await api_funcs.load_catalog_yaml("test.yaml")

                # Verify begin was called (with_session_transaction)
                mock_session.begin.assert_called_once()
                # Verify __aexit__ was called (transaction cleanup)
                mock_begin_context.__aexit__.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_estimate_pdf_with_zero_row(self, mock_get_session, mock_session):
        """Test estimate_pdf with row=0."""
        with patch("rail_svc.db_oper.estimation_funcs.estimate_pdf") as mock_estimate:
            mock_estimate.return_value = MagicMock(spec=qp.Ensemble)

            _result = await api_funcs.estimate_pdf(1, 2, 0)

            mock_estimate.assert_called_once_with(mock_session, 1, 2, 0)

    @pytest.mark.asyncio
    async def test_estimate_ensemble_with_string_path(self, mock_get_session, mock_session):
        """Test estimate_ensemble with string path."""
        output_path = "/tmp/output.hdf5"

        with patch("rail_svc.db_oper.estimation_funcs.estimate_ensemble") as mock_estimate:
            mock_estimate.return_value = Path(output_path)

            result = await api_funcs.estimate_ensemble(1, 2, output_path)

            assert isinstance(result, Path)
            mock_estimate.assert_called_once_with(mock_session, 1, 2, output_path)

    @pytest.mark.asyncio
    async def test_estimate_ensemble_with_path_object(self, mock_get_session, mock_session):
        """Test estimate_ensemble with Path object."""
        output_path = Path("/tmp/output.hdf5")

        with patch("rail_svc.db_oper.estimation_funcs.estimate_ensemble") as mock_estimate:
            mock_estimate.return_value = output_path

            result = await api_funcs.estimate_ensemble(1, 2, output_path)

            assert result == output_path
            mock_estimate.assert_called_once_with(mock_session, 1, 2, output_path)

    @pytest.mark.skip(reason="Not working")
    @pytest.mark.asyncio
    async def test_create_matched_dataset_empty_component_list(self, mock_get_session, mock_session):
        """Test create_matched_dataset with empty component list."""
        with (
            patch("rail_svc.db_oper.catalog_funcs.create_matched_dataset") as mock_create,
            patch("rail_svc.local_async.base.to_pydantic") as mock_dataset_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_assoc_convert,
        ):

            mock_create.return_value = (MagicMock(), [])
            mock_dataset_convert.return_value = models.Dataset(
                id_=1,
                name="test",
                path=None,
                n_objects=0,
                is_collection=True,
                catalog_tag_id=1,
            )
            mock_assoc_convert.return_value = []

            result = await api_funcs.create_matched_dataset(
                matched_dataset_name="test",
                catalog_tag_name="tag",
                component_dataset_names=[],
                path=None,
                n_objects=0,
            )

            assert result[1] == []

    @pytest.mark.asyncio
    async def test_load_catalog_yaml_returns_empty_lists(self, mock_get_session, mock_session):
        """Test load_catalog_yaml when database returns empty lists."""
        catalog_yaml = Path("/path/to/catalog.yaml")

        with (
            patch("rail_svc.db_oper.catalog_funcs.load_catalog_yaml") as mock_load,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_band_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_tag_convert,
            patch("rail_svc.local_async.base.to_pydantic_list") as mock_assoc_convert,
        ):

            mock_load.return_value = ([], [], [])
            mock_band_convert.return_value = []
            mock_tag_convert.return_value = []
            mock_assoc_convert.return_value = []

            result = await api_funcs.load_catalog_yaml(catalog_yaml)

            assert result == ([], [], [])
            assert isinstance(result[0], list)
            assert isinstance(result[1], list)
            assert isinstance(result[2], list)

    @pytest.mark.asyncio
    async def test_get_catalog_row_empty_dict(self, mock_get_session, mock_session):
        """Test get_catalog_row returning empty dictionary."""
        with patch("rail_svc.db_oper.catalog_funcs.get_catalog_row") as mock_get:
            mock_get.return_value = {}

            result = await api_funcs.get_catalog_row(1, 0)

            assert result == {}
            assert isinstance(result, dict)
