"""Unit tests for rail_svc.db_oper.catalog_funcs"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from rail_svc import db
from rail_svc.db_oper import catalog_funcs
from rail_svc.models import (BandCreate, CatalogBandAssocCreate,
                             CatalogTagCreate)


@pytest.fixture
def mock_session():
    """Create a mock AsyncSession"""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def sample_catalog_yaml():
    """Create a temporary catalog YAML file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("# Sample catalog YAML\n")
        yaml_path = Path(f.name)

    yield yaml_path

    # Cleanup
    yaml_path.unlink()


class TestLoadCatalogYaml:
    """Tests for load_catalog_yaml function"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.catalog_band_assoc.create_rows")
    @patch("rail_svc.db_oper.catalog_funcs.catalog_tag.create_rows")
    @patch("rail_svc.db_oper.catalog_funcs.band.create_rows")
    @patch("rail_svc.rail_funcs.catalog_funcs.load_catalog_yaml")
    async def test_load_catalog_yaml_successful(
        self,
        mock_rail_load,
        mock_band_create,
        mock_tag_create,
        mock_assoc_create,
        mock_session,
        sample_catalog_yaml,
    ):
        """Test successful loading of catalog YAML"""
        # Setup mock returns from rail_funcs
        mock_bands = [
            BandCreate(name="g", band_wavelengths=[100.0], band_transmission=[0.5]),
            BandCreate(name="r", band_wavelengths=[200.0], band_transmission=[0.6]),
        ]
        mock_tags = [CatalogTagCreate(name="test_catalog")]
        mock_assocs = [
            CatalogBandAssocCreate(
                mag_column_name="mag_g",
                mag_err_column_name="err_g",
                band_name="g",
                catalog_tag_name="test_catalog",
            )
        ]
        mock_rail_load.return_value = (mock_bands, mock_tags, mock_assocs)

        # Setup mock DB returns
        mock_db_bands = [Mock(spec=db.Band), Mock(spec=db.Band)]
        mock_db_tags = [Mock(spec=db.CatalogTag)]
        mock_db_assocs = [Mock(spec=db.CatalogBandAssoc)]

        mock_band_create.return_value = mock_db_bands
        mock_tag_create.return_value = mock_db_tags
        mock_assoc_create.return_value = mock_db_assocs

        # Execute
        bands, tags, assocs = await catalog_funcs.load_catalog_yaml(mock_session, sample_catalog_yaml)

        # Verify
        assert bands == mock_db_bands
        assert tags == mock_db_tags
        assert assocs == mock_db_assocs

        mock_rail_load.assert_called_once_with(sample_catalog_yaml, None)
        mock_band_create.assert_called_once()
        mock_tag_create.assert_called_once()
        mock_assoc_create.assert_called_once()

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.catalog_band_assoc.create_rows")
    @patch("rail_svc.db_oper.catalog_funcs.catalog_tag.create_rows")
    @patch("rail_svc.db_oper.catalog_funcs.band.create_rows")
    @patch("rail_svc.rail_funcs.catalog_funcs.load_catalog_yaml")
    async def test_load_catalog_yaml_with_filter_dir(
        self,
        mock_rail_load,
        mock_band_create,
        mock_tag_create,
        mock_assoc_create,
        mock_session,
        sample_catalog_yaml,
    ):
        """Test loading catalog YAML with custom filter directory"""
        filter_dir = Path("/custom/filters")

        mock_rail_load.return_value = ([], [], [])
        mock_band_create.return_value = []
        mock_tag_create.return_value = []
        mock_assoc_create.return_value = []

        await catalog_funcs.load_catalog_yaml(mock_session, sample_catalog_yaml, filter_dir=filter_dir)

        mock_rail_load.assert_called_once_with(sample_catalog_yaml, filter_dir)


class TestGetCatalogRow:
    """Tests for get_catalog_row function"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_get_catalog_row_successful(
        self, mock_anyio_path, mock_dataset_get, mock_rail_get_row, mock_config, mock_session
    ):
        """Test successful retrieval of catalog row"""
        # Setup mocks
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/absolute/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        expected_data = {"mag_g": np.array([22.5])}
        mock_rail_get_row.return_value = expected_data

        # Execute
        result = await catalog_funcs.get_catalog_row(mock_session, 1, 42)

        # Verify
        assert result == expected_data
        mock_dataset_get.assert_called_once_with(mock_session, 1)
        mock_rail_get_row.assert_called_once()

        # Verify path construction
        call_args = mock_rail_get_row.call_args[0]
        assert "catalog.hdf5" in str(call_args[0])

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_get_catalog_row_with_different_row_index(
        self, mock_anyio_path, mock_dataset_get, mock_config, mock_session
    ):
        """Test retrieving different row indices"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        with patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row") as mock_rail_get:
            await catalog_funcs.get_catalog_row(mock_session, 1, 100)

            # Verify row index is passed correctly
            assert mock_rail_get.call_args[0][1] == 100


class TestGetEstimatesRow:
    """Tests for get_estimates_row function"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_estimates_row")
    @patch("rail_svc.db_oper.catalog_funcs.estimates.get_row")
    @patch("anyio.Path")
    async def test_get_estimates_row_successful(
        self, mock_anyio_path, mock_estimates_get, mock_rail_get_estimates, mock_config, mock_session
    ):
        """Test successful retrieval of estimates row"""
        # Setup mocks
        mock_estimates_obj = Mock(spec=db.Estimates)
        mock_estimates_obj.path = "estimates.hdf5"
        mock_estimates_get.return_value = mock_estimates_obj

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/absolute/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        expected_data = Mock(spec=qp.Ensemble)
        mock_rail_get_estimates.return_value = expected_data

        # Execute
        result = await catalog_funcs.get_estimates_row(mock_session, 1, 42)

        # Verify
        assert result == expected_data
        mock_estimates_get.assert_called_once_with(mock_session, 1)
        mock_rail_get_estimates.assert_called_once()


class TestGetDatasetAndEstimates:
    """Tests for get_dataset_and_estimates function"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.estimates.find_by")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    async def test_get_dataset_and_estimates_successful(
        self, mock_dataset_get, mock_estimates_find, mock_session
    ):
        """Test successful retrieval of dataset and its estimates"""
        # Setup mocks
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.id_ = 1
        mock_dataset_get.return_value = mock_dataset

        mock_estimate1 = Mock(spec=db.Estimates)
        mock_estimate2 = Mock(spec=db.Estimates)

        # Return a list that will be converted by the function
        mock_estimates_find.return_value = [mock_estimate1, mock_estimate2]

        # Execute
        dataset, estimates_list = await catalog_funcs.get_dataset_and_estimates(mock_session, 1)

        # Verify
        assert dataset == mock_dataset
        assert len(estimates_list) == 2
        assert estimates_list[0] == mock_estimate1
        assert estimates_list[1] == mock_estimate2

        mock_dataset_get.assert_called_once_with(mock_session, 1)
        mock_estimates_find.assert_called_once_with(mock_session, dataset_id=1)

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.estimates.find_by")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    async def test_get_dataset_and_estimates_no_estimates(
        self, mock_dataset_get, mock_estimates_find, mock_session
    ):
        """Test retrieval when dataset has no estimates"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.id_ = 1
        mock_dataset_get.return_value = mock_dataset

        # Return empty list
        mock_estimates_find.return_value = []

        dataset, estimates_list = await catalog_funcs.get_dataset_and_estimates(mock_session, 1)

        assert dataset == mock_dataset
        assert estimates_list == []


class TestGetDataAndEstimatesData:
    """Tests for get_data_and_estimates_data function"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_estimates_row")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.estimator.get_row")
    @patch("rail_svc.db_oper.catalog_funcs.get_dataset_and_estimates")
    @patch("anyio.Path")
    async def test_get_data_and_estimates_data_successful(
        self,
        mock_anyio_path,
        mock_get_dataset_estimates,
        mock_estimator_get,
        mock_get_catalog,
        mock_get_estimates,
        mock_config,
        mock_session,
    ):
        """Test successful retrieval of data and estimates"""
        # Setup dataset and estimates
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"

        mock_estimate = Mock(spec=db.Estimates)
        mock_estimate.estimator_id = 10
        mock_estimate.path = "estimates.hdf5"

        mock_get_dataset_estimates.return_value = (mock_dataset, [mock_estimate])

        # Setup estimator
        mock_estimator = Mock(spec=db.Estimator)
        mock_estimator.name = "BPZ"
        mock_estimator_get.return_value = mock_estimator

        # Setup config and paths
        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        # Setup return data
        catalog_data = {"mag_g": np.array([22.5])}
        mock_get_catalog.return_value = catalog_data

        estimates_data = Mock(spec=qp.Ensemble)
        mock_get_estimates.return_value = estimates_data

        # Execute
        data, estimates_dict = await catalog_funcs.get_data_and_estimates_data(mock_session, 1, 42)

        # Verify
        assert data == catalog_data
        assert "BPZ" in estimates_dict
        assert estimates_dict["BPZ"] == estimates_data

        mock_get_dataset_estimates.assert_called_once_with(mock_session, 1)
        mock_estimator_get.assert_called_once_with(mock_session, 10)

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.get_dataset_and_estimates")
    @patch("anyio.Path")
    async def test_get_data_and_estimates_data_no_estimates(
        self, mock_anyio_path, mock_get_dataset_estimates, mock_get_catalog, mock_config, mock_session
    ):
        """Test retrieval when dataset has no estimates"""
        # Setup dataset with no estimates
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"

        mock_get_dataset_estimates.return_value = (mock_dataset, [])

        # Setup config and paths
        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        # Setup catalog data
        catalog_data = {"mag_g": np.array([22.5])}
        mock_get_catalog.return_value = catalog_data

        # Execute
        data, estimates_dict = await catalog_funcs.get_data_and_estimates_data(mock_session, 1, 42)

        # Verify
        assert data == catalog_data
        assert estimates_dict == {}

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_estimates_row")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.estimator.get_row")
    @patch("rail_svc.db_oper.catalog_funcs.get_dataset_and_estimates")
    @patch("anyio.Path")
    async def test_get_data_and_estimates_data_multiple_estimates(
        self,
        mock_anyio_path,
        mock_get_dataset_estimates,
        mock_estimator_get,
        mock_get_catalog,
        mock_get_estimates,
        mock_config,
        mock_session,
    ):
        """Test retrieval with multiple estimates"""
        # Setup dataset with multiple estimates
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"

        mock_estimate1 = Mock(spec=db.Estimates)
        mock_estimate1.estimator_id = 10
        mock_estimate1.path = "estimates1.hdf5"

        mock_estimate2 = Mock(spec=db.Estimates)
        mock_estimate2.estimator_id = 20
        mock_estimate2.path = "estimates2.hdf5"

        mock_get_dataset_estimates.return_value = (mock_dataset, [mock_estimate1, mock_estimate2])

        # Setup estimators
        mock_estimator1 = Mock(spec=db.Estimator)
        mock_estimator1.name = "BPZ"

        mock_estimator2 = Mock(spec=db.Estimator)
        mock_estimator2.name = "FlexZBoost"

        mock_estimator_get.side_effect = [mock_estimator1, mock_estimator2]

        # Setup config
        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        # Setup return data
        catalog_data = {"mag_g": np.array([22.5])}
        mock_get_catalog.return_value = catalog_data

        estimates_data1 = Mock(spec=qp.Ensemble)
        estimates_data2 = Mock(spec=qp.Ensemble)
        mock_get_estimates.side_effect = [estimates_data1, estimates_data2]

        # Execute
        data, estimates_dict = await catalog_funcs.get_data_and_estimates_data(mock_session, 1, 42)

        # Verify
        assert data == catalog_data
        assert len(estimates_dict) == 2
        assert estimates_dict["BPZ"] == estimates_data1
        assert estimates_dict["FlexZBoost"] == estimates_data2


class TestCreateMatchedDataset:
    """Tests for create_matched_dataset function"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_successful(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test successful creation of matched dataset"""
        # Setup mocks
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.name = "matched_dataset"
        mock_dataset_create.return_value = mock_dataset

        mock_assoc1 = Mock(spec=db.DatasetAssoc)
        mock_assoc2 = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.side_effect = [mock_assoc1, mock_assoc2]

        # Execute
        dataset, assoc_list = await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="matched_dataset",
            catalog_tag_name="lsst_dp0",
            component_dataset_names=["component1", "component2"],
            path="matched.hdf5",
            n_objects=1000,
        )

        # Verify
        assert dataset == mock_dataset
        assert len(assoc_list) == 2
        assert assoc_list[0] == mock_assoc1
        assert assoc_list[1] == mock_assoc2

        # Verify dataset creation
        mock_dataset_create.assert_called_once_with(
            mock_session,
            name="matched_dataset",
            catalog_tag_name="lsst_dp0",
            path="matched.hdf5",
            n_objects=1000,
            is_collection=True,
            validate_file=False,
        )

        # Verify associations created
        assert mock_assoc_create.call_count == 2

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_with_none_path(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test creation of matched dataset with None path"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        mock_assoc = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.return_value = mock_assoc

        dataset, assoc_list = await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="matched",
            catalog_tag_name="test",
            component_dataset_names=["comp1"],
            path=None,
            n_objects=100,
        )

        # Verify path=None was passed
        call_kwargs = mock_dataset_create.call_args[1]
        assert call_kwargs["path"] is None

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_empty_components(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test creation of matched dataset with no components"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        dataset, assoc_list = await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="matched",
            catalog_tag_name="test",
            component_dataset_names=[],
            path="matched.hdf5",
            n_objects=0,
        )

        assert dataset == mock_dataset
        assert assoc_list == []
        mock_assoc_create.assert_not_called()

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_assoc_naming(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test that associations are named correctly"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        mock_assoc = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.return_value = mock_assoc

        await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="my_matched",
            catalog_tag_name="test",
            component_dataset_names=["comp1", "comp2"],
            path="matched.hdf5",
            n_objects=100,
        )

        # Verify association names
        calls = mock_assoc_create.call_args_list
        assert len(calls) == 2

        # First association
        assert calls[0][1]["name"] == "my_matched_comp1"
        assert calls[0][1]["matched_dataset_name"] == "my_matched"
        assert calls[0][1]["component_dataset_name"] == "comp1"

        # Second association
        assert calls[1][1]["name"] == "my_matched_comp2"
        assert calls[1][1]["matched_dataset_name"] == "my_matched"
        assert calls[1][1]["component_dataset_name"] == "comp2"

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_many_components(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test creation with many component datasets"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        mock_assoc = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.return_value = mock_assoc

        component_names = [f"component_{i}" for i in range(10)]

        dataset, assoc_list = await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="large_matched",
            catalog_tag_name="test",
            component_dataset_names=component_names,
            path="large.hdf5",
            n_objects=5000,
        )

        assert len(assoc_list) == 10
        assert mock_assoc_create.call_count == 10


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    async def test_get_catalog_row_dataset_not_found(self, mock_dataset_get, mock_session):
        """Test error when dataset is not found"""
        mock_dataset_get.side_effect = Exception("Dataset not found")

        with pytest.raises(Exception, match="Dataset not found"):
            await catalog_funcs.get_catalog_row(mock_session, 999, 0)

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.estimates.get_row")
    async def test_get_estimates_row_estimates_not_found(self, mock_estimates_get, mock_session):
        """Test error when estimates are not found"""
        mock_estimates_get.side_effect = Exception("Estimates not found")

        with pytest.raises(Exception, match="Estimates not found"):
            await catalog_funcs.get_estimates_row(mock_session, 999, 0)

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_creation_fails(self, mock_dataset_create, mock_session):
        """Test error handling when dataset creation fails"""
        mock_dataset_create.side_effect = Exception("Creation failed")

        with pytest.raises(Exception, match="Creation failed"):
            await catalog_funcs.create_matched_dataset(
                mock_session,
                matched_dataset_name="test",
                catalog_tag_name="test",
                component_dataset_names=["comp1"],
                path="test.hdf5",
                n_objects=100,
            )


class TestIntegration:
    """Integration-style tests"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_get_catalog_row_path_construction(
        self, mock_anyio_path, mock_dataset_get, mock_rail_get, mock_config, mock_session
    ):
        """Test that paths are correctly constructed"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "subdir/catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        mock_config.storage.archive = "/base/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/base/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        mock_rail_get.return_value = {}

        await catalog_funcs.get_catalog_row(mock_session, 1, 0)

        # Verify the path passed to rail_funcs includes both archive and dataset path
        call_path = mock_rail_get.call_args[0][0]
        assert "archive" in str(call_path)
        assert "catalog.hdf5" in str(call_path)

    @pytest.mark.asyncio
    @patch("rail_svc.rail_funcs.catalog_funcs.load_catalog_yaml")
    async def test_load_catalog_yaml_data_flow(self, mock_rail_load, mock_session, sample_catalog_yaml):
        """Test that data flows correctly through load_catalog_yaml"""
        # Create realistic mock data
        band_create = BandCreate(name="g", band_wavelengths=[400.0, 500.0], band_transmission=[0.5, 0.8])
        tag_create = CatalogTagCreate(name="test_catalog")
        assoc_create = CatalogBandAssocCreate(
            mag_column_name="mag_g",
            mag_err_column_name="err_g",
            band_name="g",
            catalog_tag_name="test_catalog",
        )

        mock_rail_load.return_value = ([band_create], [tag_create], [assoc_create])

        with patch("rail_svc.db_oper.catalog_funcs.band.create_rows") as mock_band:
            with patch("rail_svc.db_oper.catalog_funcs.catalog_tag.create_rows") as mock_tag:
                with patch("rail_svc.db_oper.catalog_funcs.catalog_band_assoc.create_rows") as mock_assoc:
                    mock_band.return_value = [Mock(spec=db.Band)]
                    mock_tag.return_value = [Mock(spec=db.CatalogTag)]
                    mock_assoc.return_value = [Mock(spec=db.CatalogBandAssoc)]

                    await catalog_funcs.load_catalog_yaml(mock_session, sample_catalog_yaml)

                    # Verify model_dump was called on each create object
                    band_call = mock_band.call_args[0][1][0]
                    assert band_call["name"] == "g"
                    assert band_call["band_wavelengths"] == [400.0, 500.0]

                    tag_call = mock_tag.call_args[0][1][0]
                    assert tag_call["name"] == "test_catalog"

                    assoc_call = mock_assoc.call_args[0][1][0]
                    assert assoc_call["mag_column_name"] == "mag_g"


class TestPathHandling:
    """Tests for path handling across different functions"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_absolute_path_conversion(
        self, mock_anyio_path, mock_dataset_get, mock_rail_get, mock_config, mock_session
    ):
        """Test that relative paths are converted to absolute"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        mock_config.storage.archive = "relative/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/absolute/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        mock_rail_get.return_value = {}

        await catalog_funcs.get_catalog_row(mock_session, 1, 0)

        # Verify absolute was called
        mock_anyio_path_instance.absolute.assert_called_once()

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_estimates_row")
    @patch("rail_svc.db_oper.catalog_funcs.estimates.get_row")
    @patch("anyio.Path")
    async def test_estimates_path_handling(
        self, mock_anyio_path, mock_estimates_get, mock_rail_get, mock_config, mock_session
    ):
        """Test path handling for estimates"""
        mock_estimates_obj = Mock(spec=db.Estimates)
        mock_estimates_obj.path = "nested/dir/estimates.hdf5"
        mock_estimates_get.return_value = mock_estimates_obj

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        mock_rail_get.return_value = Mock(spec=qp.Ensemble)

        await catalog_funcs.get_estimates_row(mock_session, 1, 0)

        # Verify path includes nested directories
        call_path = mock_rail_get.call_args[0][0]
        assert "nested" in str(call_path)
        assert "estimates.hdf5" in str(call_path)


class TestDataTypes:
    """Tests for proper data type handling"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_catalog_row_returns_dict_with_arrays(
        self, mock_anyio_path, mock_dataset_get, mock_rail_get, mock_config, mock_session
    ):
        """Test that catalog row returns proper numpy array dict"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        expected_data = {
            "mag_g": np.array([22.5, 23.1]),
            "mag_r": np.array([21.8, 22.3]),
            "redshift": np.array([0.5, 0.6]),
        }
        mock_rail_get.return_value = expected_data

        result = await catalog_funcs.get_catalog_row(mock_session, 1, 0)

        assert isinstance(result, dict)
        assert "mag_g" in result
        assert isinstance(result["mag_g"], np.ndarray)

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_estimates_row")
    @patch("rail_svc.db_oper.catalog_funcs.estimates.get_row")
    @patch("anyio.Path")
    async def test_estimates_row_returns_qp_ensemble(
        self, mock_anyio_path, mock_estimates_get, mock_rail_get, mock_config, mock_session
    ):
        """Test that estimates row returns qp.Ensemble"""
        mock_estimates_obj = Mock(spec=db.Estimates)
        mock_estimates_obj.path = "estimates.hdf5"
        mock_estimates_get.return_value = mock_estimates_obj

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        expected_ensemble = Mock(spec=qp.Ensemble)
        mock_rail_get.return_value = expected_ensemble

        result = await catalog_funcs.get_estimates_row(mock_session, 1, 0)

        assert isinstance(result, Mock)
        assert result._spec_class == qp.Ensemble


class TestAsyncBehavior:
    """Tests for async/await behavior"""

    @pytest.mark.asyncio
    async def test_all_functions_are_async(self):
        """Test that all public functions are async"""
        import inspect

        functions = [
            catalog_funcs.load_catalog_yaml,
            catalog_funcs.get_catalog_row,
            catalog_funcs.get_estimates_row,
            catalog_funcs.get_dataset_and_estimates,
            catalog_funcs.get_data_and_estimates_data,
            catalog_funcs.create_matched_dataset,
        ]

        for func in functions:
            assert inspect.iscoroutinefunction(func), f"{func.__name__} is not async"

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    async def test_awaitable_calls(self, mock_dataset_get, mock_session):
        """Test that async calls are properly awaited"""
        mock_dataset_get.return_value = Mock(spec=db.Dataset, path="test.hdf5")

        # This should not raise any errors about unawaited coroutines
        with patch("rail_svc.db_oper.catalog_funcs.global_config"):
            with patch("anyio.Path") as mock_anyio:
                mock_anyio_instance = AsyncMock()
                mock_anyio_instance.absolute = AsyncMock(return_value="/archive")
                mock_anyio.return_value = mock_anyio_instance

                with patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row"):
                    await catalog_funcs.get_catalog_row(mock_session, 1, 0)


class TestValidationFlags:
    """Tests for dataset creation validation flags"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_matched_dataset_validation_disabled(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test that matched dataset creation disables file validation"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="matched",
            catalog_tag_name="test",
            component_dataset_names=[],
            path="matched.hdf5",
            n_objects=100,
        )

        # Verify validate_file=False was passed
        call_kwargs = mock_dataset_create.call_args[1]
        assert call_kwargs["validate_file"] is False

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_matched_dataset_is_collection_flag(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test that matched dataset is marked as collection"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="matched",
            catalog_tag_name="test",
            component_dataset_names=[],
            path="matched.hdf5",
            n_objects=100,
        )

        # Verify is_collection=True was passed
        call_kwargs = mock_dataset_create.call_args[1]
        assert call_kwargs["is_collection"] is True


class TestSpecialCases:
    """Tests for special cases and unusual inputs"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_with_special_chars(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test creation with special characters in names"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        mock_assoc = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.return_value = mock_assoc

        await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="matched-dataset_v2.0",
            catalog_tag_name="test",
            component_dataset_names=["comp-1_final"],
            path="matched.hdf5",
            n_objects=100,
        )

        # Verify names with special characters were handled
        assoc_call = mock_assoc_create.call_args[1]
        assert assoc_call["name"] == "matched-dataset_v2.0_comp-1_final"

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_estimates_row")
    @patch("rail_svc.db_oper.catalog_funcs.estimator.get_row")
    @patch("rail_svc.db_oper.catalog_funcs.get_dataset_and_estimates")
    @patch("anyio.Path")
    async def test_get_data_with_duplicate_estimator_names(
        self,
        mock_anyio_path,
        mock_get_dataset_estimates,
        mock_estimator_get,
        mock_get_estimates,
        mock_get_catalog,
        mock_config,
        mock_session,
    ):
        """Test handling when multiple estimates have same estimator name"""
        # This tests dict key overwriting behavior
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"

        mock_estimate1 = Mock(spec=db.Estimates)
        mock_estimate1.estimator_id = 10
        mock_estimate1.path = "estimates1.hdf5"

        mock_estimate2 = Mock(spec=db.Estimates)
        mock_estimate2.estimator_id = 11
        mock_estimate2.path = "estimates2.hdf5"

        mock_get_dataset_estimates.return_value = (mock_dataset, [mock_estimate1, mock_estimate2])

        # Both estimators have same name
        mock_estimator = Mock(spec=db.Estimator)
        mock_estimator.name = "BPZ"
        mock_estimator_get.return_value = mock_estimator

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        mock_get_catalog.return_value = {}
        mock_get_estimates.side_effect = [Mock(spec=qp.Ensemble), Mock(spec=qp.Ensemble)]

        data, estimates_dict = await catalog_funcs.get_data_and_estimates_data(mock_session, 1, 0)

        # Only one entry should exist (last one overwrites)
        assert len(estimates_dict) == 1
        assert "BPZ" in estimates_dict

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_zero_objects(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test creation with zero objects"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        mock_assoc = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.return_value = mock_assoc

        dataset, assocs = await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="empty_matched",
            catalog_tag_name="test",
            component_dataset_names=["comp1"],
            path="empty.hdf5",
            n_objects=0,
        )

        # Verify n_objects=0 was passed
        call_kwargs = mock_dataset_create.call_args[1]
        assert call_kwargs["n_objects"] == 0

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_get_catalog_row_zero_index(
        self, mock_anyio_path, mock_dataset_get, mock_rail_get, mock_config, mock_session
    ):
        """Test retrieving row at index 0"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        mock_rail_get.return_value = {}

        await catalog_funcs.get_catalog_row(mock_session, 1, 0)

        # Verify row 0 was passed correctly
        assert mock_rail_get.call_args[0][1] == 0


class TestErrorPropagation:
    """Tests for proper error propagation from lower layers"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_rail_funcs_error_propagates(
        self, mock_anyio_path, mock_dataset_get, mock_rail_get, mock_config, mock_session
    ):
        """Test that errors from rail_funcs layer propagate"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        mock_config.storage.archive = "/archive"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/archive")
        mock_anyio_path.return_value = mock_anyio_path_instance

        mock_rail_get.side_effect = ValueError("Invalid row index")

        with pytest.raises(ValueError, match="Invalid row index"):
            await catalog_funcs.get_catalog_row(mock_session, 1, -1)

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_assoc_creation_error_stops_process(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test that association creation error stops the process"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        # First assoc succeeds, second fails
        mock_assoc = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.side_effect = [mock_assoc, Exception("DB error")]

        with pytest.raises(Exception, match="DB error"):
            await catalog_funcs.create_matched_dataset(
                mock_session,
                matched_dataset_name="matched",
                catalog_tag_name="test",
                component_dataset_names=["comp1", "comp2"],
                path="matched.hdf5",
                n_objects=100,
            )


class TestReturnTypes:
    """Tests for correct return type annotations and actual returns"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.catalog_band_assoc.create_rows")
    @patch("rail_svc.db_oper.catalog_funcs.catalog_tag.create_rows")
    @patch("rail_svc.db_oper.catalog_funcs.band.create_rows")
    @patch("rail_svc.rail_funcs.catalog_funcs.load_catalog_yaml")
    async def test_load_catalog_yaml_return_types(
        self,
        mock_rail_load,
        mock_band_create,
        mock_tag_create,
        mock_assoc_create,
        mock_session,
        sample_catalog_yaml,
    ):
        """Test that load_catalog_yaml returns correct types"""
        mock_rail_load.return_value = ([], [], [])

        mock_bands = [Mock(spec=db.Band)]
        mock_tags = [Mock(spec=db.CatalogTag)]
        mock_assocs = [Mock(spec=db.CatalogBandAssoc)]

        mock_band_create.return_value = mock_bands
        mock_tag_create.return_value = mock_tags
        mock_assoc_create.return_value = mock_assocs

        result = await catalog_funcs.load_catalog_yaml(mock_session, sample_catalog_yaml)

        assert isinstance(result, tuple)
        assert len(result) == 3
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)
        assert isinstance(result[2], list)

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.estimates.find_by")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    async def test_get_dataset_and_estimates_return_types(
        self, mock_dataset_get, mock_estimates_find, mock_session
    ):
        """Test return types from get_dataset_and_estimates"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.id_ = 1
        mock_dataset_get.return_value = mock_dataset

        # Return a list instead of async generator
        mock_estimates_find.return_value = [Mock(spec=db.Estimates)]

        result = await catalog_funcs.get_dataset_and_estimates(mock_session, 1)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Mock)  # Dataset
        assert isinstance(result[1], list)  # List of Estimates

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset_assoc.create_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.create_row")
    async def test_create_matched_dataset_return_types(
        self, mock_dataset_create, mock_assoc_create, mock_session
    ):
        """Test return types from create_matched_dataset"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset_create.return_value = mock_dataset

        mock_assoc = Mock(spec=db.DatasetAssoc)
        mock_assoc_create.return_value = mock_assoc

        result = await catalog_funcs.create_matched_dataset(
            mock_session,
            matched_dataset_name="matched",
            catalog_tag_name="test",
            component_dataset_names=["comp1"],
            path="matched.hdf5",
            n_objects=100,
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], Mock)  # Dataset
        assert isinstance(result[1], list)  # List of DatasetAssoc


class TestSessionUsage:
    """Tests for proper AsyncSession usage"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.band.create_rows")
    @patch("rail_svc.rail_funcs.catalog_funcs.load_catalog_yaml")
    async def test_session_passed_to_db_operations(
        self, mock_rail_load, mock_band_create, mock_session, sample_catalog_yaml
    ):
        """Test that session is passed to all DB operations"""
        mock_rail_load.return_value = (
            [BandCreate(name="g", band_wavelengths=[100.0], band_transmission=[0.5])],
            [],
            [],
        )
        mock_band_create.return_value = []

        with patch("rail_svc.db_oper.catalog_funcs.catalog_tag.create_rows") as mock_tag:
            with patch("rail_svc.db_oper.catalog_funcs.catalog_band_assoc.create_rows") as mock_assoc:
                mock_tag.return_value = []
                mock_assoc.return_value = []

                await catalog_funcs.load_catalog_yaml(mock_session, sample_catalog_yaml)

                # Verify session was passed to all create_rows calls
                assert mock_band_create.call_args[0][0] == mock_session
                assert mock_tag.call_args[0][0] == mock_session
                assert mock_assoc.call_args[0][0] == mock_session

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    async def test_session_passed_to_get_operations(self, mock_dataset_get, mock_session):
        """Test that session is passed to get operations"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "test.hdf5"
        mock_dataset_get.return_value = mock_dataset

        with patch("rail_svc.db_oper.catalog_funcs.global_config"):
            with patch("anyio.Path") as mock_anyio:
                mock_anyio_instance = AsyncMock()
                mock_anyio_instance.absolute = AsyncMock(return_value="/archive")
                mock_anyio.return_value = mock_anyio_instance

                with patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row"):
                    await catalog_funcs.get_catalog_row(mock_session, 1, 0)

                    # Verify session was passed
                    assert mock_dataset_get.call_args[0][0] == mock_session


class TestConfigUsage:
    """Tests for proper global_config usage"""

    @pytest.mark.asyncio
    @patch("rail_svc.db_oper.catalog_funcs.global_config")
    @patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row")
    @patch("rail_svc.db_oper.catalog_funcs.dataset.get_row")
    @patch("anyio.Path")
    async def test_archive_path_from_config(
        self, mock_anyio_path, mock_dataset_get, mock_rail_get, mock_config, mock_session
    ):
        """Test that archive path is read from global config"""
        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.path = "catalog.hdf5"
        mock_dataset_get.return_value = mock_dataset

        # Set specific archive path
        mock_config.storage.archive = "/custom/archive/path"
        mock_anyio_path_instance = AsyncMock()
        mock_anyio_path_instance.absolute = AsyncMock(return_value="/custom/archive/path")
        mock_anyio_path.return_value = mock_anyio_path_instance

        mock_rail_get.return_value = {}

        await catalog_funcs.get_catalog_row(mock_session, 1, 0)

        # Verify the archive path from config was used
        mock_anyio_path.assert_called_with("/custom/archive/path")
