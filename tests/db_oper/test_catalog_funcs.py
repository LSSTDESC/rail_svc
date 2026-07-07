"""Integration tests for rail_svc.db_oper.catalog_funcs"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from rail_svc.db_oper import catalog_funcs
from rail_svc.models import BandCreate, CatalogBandAssocCreate, CatalogTagCreate


class TestGetDatasetAndEstimates:
    """Tests for get_dataset_and_estimates — pure DB operations."""

    @pytest.mark.asyncio
    async def test_returns_dataset_and_estimates(self, session, sample_dataset, sample_estimates):
        dataset, estimates_list = await catalog_funcs.get_dataset_and_estimates(session, sample_dataset.id_)

        assert dataset.id_ == sample_dataset.id_
        assert len(estimates_list) == 1
        assert estimates_list[0].id_ == sample_estimates.id_

    @pytest.mark.asyncio
    async def test_no_estimates(self, session, sample_dataset):
        dataset, estimates_list = await catalog_funcs.get_dataset_and_estimates(session, sample_dataset.id_)

        assert dataset.id_ == sample_dataset.id_
        assert estimates_list == []

    @pytest.mark.asyncio
    async def test_dataset_not_found(self, session):
        with pytest.raises(KeyError):
            await catalog_funcs.get_dataset_and_estimates(session, 99999)


class TestCreateMatchedDataset:
    """Tests for create_matched_dataset — DB writes with real session."""

    @pytest.mark.asyncio
    async def test_creates_dataset_and_associations(
        self, session, sample_catalog_tag, component_dataset_1, component_dataset_2
    ):
        dataset, assoc_list = await catalog_funcs.create_matched_dataset(
            session,
            matched_dataset_name="my_matched",
            catalog_tag_name=sample_catalog_tag.name,
            component_dataset_names=[component_dataset_1.name, component_dataset_2.name],
            path="matched/my_matched.hdf5",
            n_objects=11000,
        )

        assert dataset.name == "my_matched"
        assert dataset.is_collection is True
        assert dataset.n_objects == 11000
        assert dataset.catalog_tag_id == sample_catalog_tag.id_
        assert len(assoc_list) == 2

    @pytest.mark.asyncio
    async def test_association_naming(self, session, sample_catalog_tag, component_dataset_1):
        dataset, assoc_list = await catalog_funcs.create_matched_dataset(
            session,
            matched_dataset_name="matched-v2",
            catalog_tag_name=sample_catalog_tag.name,
            component_dataset_names=[component_dataset_1.name],
            path="matched.hdf5",
            n_objects=5000,
        )

        assert assoc_list[0].name == f"matched-v2_{component_dataset_1.name}"

    @pytest.mark.asyncio
    async def test_empty_components(self, session, sample_catalog_tag):
        dataset, assoc_list = await catalog_funcs.create_matched_dataset(
            session,
            matched_dataset_name="empty_matched",
            catalog_tag_name=sample_catalog_tag.name,
            component_dataset_names=[],
            path="matched/empty.hdf5",
            n_objects=0,
        )

        assert dataset.name == "empty_matched"
        assert assoc_list == []

    @pytest.mark.asyncio
    async def test_nonexistent_catalog_tag(self, session):
        with pytest.raises(ValueError, match="CatalogTag with name 'nonexistent' not found"):
            await catalog_funcs.create_matched_dataset(
                session,
                matched_dataset_name="test",
                catalog_tag_name="nonexistent",
                component_dataset_names=[],
                path="test.hdf5",
                n_objects=0,
            )


class TestGetCatalogRow:
    """Tests for get_catalog_row — real DB, mock file I/O."""

    @pytest.mark.asyncio
    async def test_reads_from_correct_path(self, session, sample_dataset, tmp_path):
        expected = {"mag_g": np.array([22.5])}

        with (
            patch("rail_svc.db_oper.catalog_funcs.global_config") as mock_config,
            patch("anyio.Path") as mock_anyio_path,
            patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row", return_value=expected) as mock_read,
        ):
            mock_config.storage.archive = str(tmp_path)
            mock_anyio_instance = MagicMock()
            mock_anyio_instance.absolute = AsyncMock(return_value=str(tmp_path))
            mock_anyio_path.return_value = mock_anyio_instance

            result = await catalog_funcs.get_catalog_row(session, sample_dataset.id_, 42)

            assert result == expected
            call_path = mock_read.call_args[0][0]
            assert sample_dataset.path in str(call_path)
            assert mock_read.call_args[0][1] == 42

    @pytest.mark.asyncio
    async def test_dataset_not_found(self, session):
        with pytest.raises(KeyError):
            await catalog_funcs.get_catalog_row(session, 99999, 0)


class TestGetEstimatesRow:
    """Tests for get_estimates_row — real DB, mock file I/O."""

    @pytest.mark.asyncio
    async def test_reads_from_correct_path(self, session, sample_estimates, tmp_path):
        expected_ensemble = {"z": np.array([0.5])}

        with (
            patch("rail_svc.db_oper.catalog_funcs.global_config") as mock_config,
            patch("anyio.Path") as mock_anyio_path,
            patch(
                "rail_svc.rail_funcs.catalog_funcs.get_estimates_row", return_value=expected_ensemble
            ) as mock_read,
        ):
            mock_config.storage.archive = str(tmp_path)
            mock_anyio_instance = MagicMock()
            mock_anyio_instance.absolute = AsyncMock(return_value=str(tmp_path))
            mock_anyio_path.return_value = mock_anyio_instance

            result = await catalog_funcs.get_estimates_row(session, sample_estimates.id_, 7)

            assert result == expected_ensemble
            call_path = mock_read.call_args[0][0]
            assert sample_estimates.path in str(call_path)
            assert mock_read.call_args[0][1] == 7


class TestGetDataAndEstimatesData:
    """Tests for get_data_and_estimates_data — real DB, mock file I/O."""

    @pytest.mark.asyncio
    async def test_returns_catalog_and_estimates(
        self, session, sample_dataset, sample_estimates, sample_estimator, tmp_path
    ):
        catalog_data = {"mag_g": np.array([22.5])}
        estimates_data = {"z": np.array([0.5])}

        with (
            patch("rail_svc.db_oper.catalog_funcs.global_config") as mock_config,
            patch("anyio.Path") as mock_anyio_path,
            patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row", return_value=catalog_data),
            patch("rail_svc.rail_funcs.catalog_funcs.get_estimates_row", return_value=estimates_data),
        ):
            mock_config.storage.archive = str(tmp_path)
            mock_anyio_instance = MagicMock()
            mock_anyio_instance.absolute = AsyncMock(return_value=str(tmp_path))
            mock_anyio_path.return_value = mock_anyio_instance

            data, est_dict = await catalog_funcs.get_data_and_estimates_data(session, sample_dataset.id_, 0)

            assert data == catalog_data
            assert sample_estimator.name in est_dict
            assert est_dict[sample_estimator.name] == estimates_data

    @pytest.mark.asyncio
    async def test_no_estimates_returns_empty_dict(self, session, sample_dataset, tmp_path):
        catalog_data = {"mag_g": np.array([22.5])}

        with (
            patch("rail_svc.db_oper.catalog_funcs.global_config") as mock_config,
            patch("anyio.Path") as mock_anyio_path,
            patch("rail_svc.rail_funcs.catalog_funcs.get_catalog_row", return_value=catalog_data),
        ):
            mock_config.storage.archive = str(tmp_path)
            mock_anyio_instance = MagicMock()
            mock_anyio_instance.absolute = AsyncMock(return_value=str(tmp_path))
            mock_anyio_path.return_value = mock_anyio_instance

            data, est_dict = await catalog_funcs.get_data_and_estimates_data(session, sample_dataset.id_, 0)

            assert data == catalog_data
            assert est_dict == {}


class TestLoadCatalogYaml:
    """Tests for load_catalog_yaml — real DB, mock YAML parser."""

    @pytest.mark.asyncio
    async def test_persists_bands_tags_assocs(self, session, tmp_path):
        yaml_path = tmp_path / "catalog.yaml"
        yaml_path.write_text("# placeholder")

        band_creates = [BandCreate(name="g", band_wavelengths=[400.0, 500.0], band_transmission=[0.5, 0.8])]
        tag_creates = [CatalogTagCreate(name="test_catalog")]
        assoc_creates = [
            CatalogBandAssocCreate(
                mag_column_name="mag_g",
                mag_err_column_name="err_g",
                band_name="g",
                catalog_tag_name="test_catalog",
            )
        ]

        with patch(
            "rail_svc.rail_funcs.catalog_funcs.load_catalog_yaml",
            return_value=(band_creates, tag_creates, assoc_creates),
        ):
            bands, tags, assocs = await catalog_funcs.load_catalog_yaml(session, yaml_path)

        assert len(bands) == 1
        assert bands[0].name == "g"
        assert len(tags) == 1
        assert tags[0].name == "test_catalog"
        assert len(assocs) == 1
        assert assocs[0].mag_column_name == "mag_g"

    @pytest.mark.asyncio
    async def test_passes_filter_dir(self, session, tmp_path):
        yaml_path = tmp_path / "catalog.yaml"
        yaml_path.write_text("# placeholder")
        filter_dir = tmp_path / "filters"

        band_creates = [BandCreate(name="r", band_wavelengths=[600.0], band_transmission=[0.9])]
        tag_creates = [CatalogTagCreate(name="filter_test_tag")]
        assoc_creates = [
            CatalogBandAssocCreate(
                mag_column_name="mag_r",
                mag_err_column_name="err_r",
                band_name="r",
                catalog_tag_name="filter_test_tag",
            )
        ]

        with patch(
            "rail_svc.rail_funcs.catalog_funcs.load_catalog_yaml",
            return_value=(band_creates, tag_creates, assoc_creates),
        ) as mock_parser:
            await catalog_funcs.load_catalog_yaml(session, yaml_path, filter_dir=filter_dir)

        mock_parser.assert_called_once_with(yaml_path, filter_dir)
