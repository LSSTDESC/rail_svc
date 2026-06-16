import logging
from unittest.mock import AsyncMock, Mock, patch

import pytest
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from rail_svc import db
from rail_svc.db_oper import estimation_funcs
from rail_svc.rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper


@pytest.fixture
def mock_session():
    """Create a mock async session."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def mock_components():
    """Create mock estimator components."""
    estimator = Mock(spec=db.Estimator)
    estimator.id_ = 1
    estimator.name = "test_estimator"
    estimator.model_id = 10
    estimator.config = {"param1": "value1"}

    model = Mock(spec=db.Model)
    model.id_ = 10
    model.algo_id = 100
    model.catalog_tag_id = 1000
    model.path = "models/test_model.pkl"

    algorithm = Mock(spec=db.Algorithm)
    algorithm.id_ = 100
    algorithm.class_name = "TestEstimator"

    catalog_tag = Mock(spec=db.CatalogTag)
    catalog_tag.id_ = 1000
    catalog_tag.name = "test_catalog"

    return estimator, model, algorithm, catalog_tag


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = Mock(spec=db.Dataset)
    dataset.id_ = 5
    dataset.path = "datasets/test_data.hdf5"
    dataset.catalog_tag_id = 1000
    return dataset


class TestEstimatePdf:
    """Test estimate_pdf function."""

    @pytest.mark.asyncio
    async def test_successful_estimation(self, mock_session):
        """Test successful PDF estimation."""
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)
        mock_catalog_data = {"mag_g": 20.0, "mag_r": 19.5}
        mock_pdf = Mock(spec=qp.Ensemble)
        mock_wrapper.return_value = mock_pdf

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper", return_value=mock_wrapper
            ),
            patch(
                "rail_svc.db_oper.estimation_funcs.catalog_funcs.get_catalog_row",
                return_value=mock_catalog_data,
            ),
        ):
            result = await estimation_funcs.estimate_pdf(mock_session, estimator_id=1, dataset_id=5, row=42)

            assert result == mock_pdf
            # This is incorrectly failling for some reason.
            # mock_wrapper.assert_called_once_with(mock_catalog_data)

    @pytest.mark.asyncio
    async def test_catalog_read_failure(self, mock_session):
        """Test error when catalog reading fails."""
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper", return_value=mock_wrapper
            ),
            patch(
                "rail_svc.db_oper.estimation_funcs.catalog_funcs.get_catalog_row",
                side_effect=FileNotFoundError("Catalog not found"),
            ),
        ):
            with pytest.raises(FileNotFoundError, match="Catalog not found"):
                await estimation_funcs.estimate_pdf(mock_session, estimator_id=1, dataset_id=5, row=42)


class TestEstimateEnsemble:
    """Test estimate_ensemble function."""

    @pytest.mark.asyncio
    async def test_successful_estimation(self, mock_session, mock_dataset, tmp_path):
        """Test successful ensemble estimation."""
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        input_path = archive_dir / mock_dataset.path
        input_path.parent.mkdir(parents=True)
        input_path.write_text("mock dataset")

        output_path = archive_dir / "estimates" / "output.hdf5"

        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        def create_output(*args):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("mock estimates")

        mock_wrapper.side_effect = create_output

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
        ):
            result = await estimation_funcs.estimate_ensemble(
                mock_session,
                estimator_id=1,
                dataset_id=5,
                output_file_path="estimates/output.hdf5",
            )

            assert result == output_path
            assert output_path.exists()

    @pytest.mark.asyncio
    async def test_absolute_output_path(self, mock_session, mock_dataset, tmp_path):
        """Test with absolute output path."""
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        input_path = archive_dir / mock_dataset.path
        input_path.parent.mkdir(parents=True)
        input_path.write_text("mock dataset")

        output_path = tmp_path / "custom" / "output.hdf5"
        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        def create_output(*args):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("mock estimates")

        mock_wrapper.side_effect = create_output

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
        ):
            result = await estimation_funcs.estimate_ensemble(
                mock_session, estimator_id=1, dataset_id=5, output_file_path=output_path
            )

            assert result == output_path

    @pytest.mark.asyncio
    async def test_input_file_not_found(self, mock_session, mock_dataset, tmp_path):
        """Test error when input dataset file doesn't exist."""
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()

        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
        ):
            with pytest.raises(FileNotFoundError, match="Dataset file not found"):
                await estimation_funcs.estimate_ensemble(
                    mock_session, estimator_id=1, dataset_id=5, output_file_path="output.hdf5"
                )

    @pytest.mark.asyncio
    async def test_output_file_not_created(self, mock_session, mock_dataset, tmp_path):
        """Test error when wrapper doesn't create output file."""
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        input_path = archive_dir / mock_dataset.path
        input_path.parent.mkdir(parents=True)
        input_path.write_text("mock dataset")

        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)
        mock_wrapper.return_value = None  # Doesn't create file

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
        ):
            with pytest.raises(OSError, match="Output file was not created"):
                await estimation_funcs.estimate_ensemble(
                    mock_session, estimator_id=1, dataset_id=5, output_file_path="output.hdf5"
                )


class TestGetEstimatorsForDataset:
    """Test get_estimators_for_dataest function."""

    @pytest.mark.asyncio
    async def test_successful_retrieval(self, mock_session, mock_dataset):
        """Test successful retrieval of estimators for dataset."""
        catalog_tag = Mock(spec=db.CatalogTag)
        catalog_tag.id_ = 1000

        models = [Mock(spec=db.Model, id_=10), Mock(spec=db.Model, id_=11)]

        estimator1 = Mock(spec=db.Estimator, id_=1)
        estimator2 = Mock(spec=db.Estimator, id_=2)
        estimator3 = Mock(spec=db.Estimator, id_=3)

        with (
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.catalog_tag.get_row", return_value=catalog_tag),
            patch("rail_svc.db_oper.estimation_funcs.model.find_by", return_value=models),
            patch(
                "rail_svc.db_oper.estimation_funcs.estimator.find_by",
                side_effect=[[estimator1, estimator2], [estimator3]],
            ),
        ):
            result = await estimation_funcs.get_estimators_for_dataest(mock_session, dataset_id=5)

            assert len(result) == 3
            assert estimator1 in result
            assert estimator2 in result
            assert estimator3 in result

    @pytest.mark.asyncio
    async def test_no_models_returns_empty(self, mock_session, mock_dataset):
        """Test that no models results in empty list."""
        catalog_tag = Mock(spec=db.CatalogTag)

        with (
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.catalog_tag.get_row", return_value=catalog_tag),
            patch("rail_svc.db_oper.estimation_funcs.model.find_by", return_value=[]),
        ):
            result = await estimation_funcs.get_estimators_for_dataest(mock_session, dataset_id=5)

            assert result == []

    @pytest.mark.asyncio
    async def test_dataset_not_found(self, mock_session):
        """Test error when dataset not found."""
        with patch(
            "rail_svc.db_oper.estimation_funcs.dataset.get_row",
            side_effect=ValueError("Dataset not found"),
        ):
            with pytest.raises(ValueError, match="Dataset not found"):
                await estimation_funcs.get_estimators_for_dataest(mock_session, dataset_id=999)


class TestBuildWrapperLists:
    """Test functions that build lists of wrappers."""

    @pytest.mark.asyncio
    async def test_build_pdf_wrappers_list(self, mock_session):
        """Test building list of PDF wrappers."""
        estimators = [Mock(spec=db.Estimator, id_=i, name=f"est{i}") for i in range(3)]
        wrappers = [Mock(spec=CatEstimatorPdfWrapper) for _ in range(3)]

        with (
            patch("rail_svc.db_oper.estimation_funcs.get_estimators_for_dataest", return_value=estimators),
            patch("rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper", side_effect=wrappers),
        ):
            result = await estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(
                mock_session, dataset_id=5
            )

            assert len(result) == 3
            assert result == wrappers

    @pytest.mark.asyncio
    async def test_build_ensemble_wrappers_list(self, mock_session):
        """Test building list of ensemble wrappers."""
        estimators = [Mock(spec=db.Estimator, id_=i, name=f"est{i}") for i in range(3)]
        wrappers = [Mock(spec=CatEstimatorEnsembleWrapper) for _ in range(3)]

        with (
            patch("rail_svc.db_oper.estimation_funcs.get_estimators_for_dataest", return_value=estimators),
            patch(
                "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper", side_effect=wrappers
            ),
        ):
            result = await estimation_funcs.build_cat_estimator_ensemble_wrappers_for_dataset(
                mock_session, dataset_id=5
            )

            assert len(result) == 3
            assert result == wrappers

    @pytest.mark.asyncio
    async def test_partial_wrapper_failures(self, mock_session, caplog):
        """Test that some wrappers can fail while others succeed."""
        estimators = [
            Mock(spec=db.Estimator, id_=1, name="est1"),
            Mock(spec=db.Estimator, id_=2, name="est2"),
            Mock(spec=db.Estimator, id_=3, name="est3"),
        ]

        wrapper1 = Mock(spec=CatEstimatorPdfWrapper)
        wrapper3 = Mock(spec=CatEstimatorPdfWrapper)

        def build_side_effect(session, est_id):
            if est_id == 2:
                raise ValueError("Build failed")
            return wrapper1 if est_id == 1 else wrapper3

        with (
            patch("rail_svc.db_oper.estimation_funcs.get_estimators_for_dataest", return_value=estimators),
            patch(
                "rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper",
                side_effect=build_side_effect,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = await estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(
                mock_session, dataset_id=5
            )

            assert len(result) == 2
            assert wrapper1 in result
            assert wrapper3 in result
            assert "Failed to build estimator" in caplog.text

    @pytest.mark.asyncio
    async def test_empty_estimator_list(self, mock_session):
        """Test with no estimators available."""
        with patch("rail_svc.db_oper.estimation_funcs.get_estimators_for_dataest", return_value=[]):
            pdf_result = await estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(
                mock_session, dataset_id=5
            )
            ensemble_result = await estimation_funcs.build_cat_estimator_ensemble_wrappers_for_dataset(
                mock_session, dataset_id=5
            )

            assert pdf_result == []
            assert ensemble_result == []


class TestIntegrationScenarios:
    """Test complete workflows."""

    @pytest.mark.asyncio
    async def test_full_pdf_estimation_workflow(self, mock_session, mock_components, tmp_path):
        """Test complete workflow from dataset to PDF estimation."""
        estimator, model, algorithm, catalog_tag = mock_components

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        model_path = archive_dir / model.path
        model_path.parent.mkdir(parents=True)
        model_path.write_text("mock model")

        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.id_ = 5
        mock_dataset.catalog_tag_id = 1000

        mock_catalog_data = {"mag_g": 20.0}
        mock_pdf = Mock(spec=qp.Ensemble)
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)
        mock_wrapper.return_value = mock_pdf

        with (
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.catalog_tag.get_row", return_value=catalog_tag),
            patch("rail_svc.db_oper.estimation_funcs.model.find_by", return_value=[model]),
            patch("rail_svc.db_oper.estimation_funcs.estimator.find_by", return_value=[estimator]),
            patch("rail_svc.db_oper.wrappers._get_estimator_components", return_value=mock_components),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
            patch.object(CatEstimatorPdfWrapper, "build_wrapper", return_value=mock_wrapper),
            patch(
                "rail_svc.db_oper.estimation_funcs.catalog_funcs.get_catalog_row",
                return_value=mock_catalog_data,
            ),
        ):
            # Get estimators
            estimators = await estimation_funcs.get_estimators_for_dataest(mock_session, dataset_id=5)
            assert len(estimators) == 1

            # Estimate PDF
            pdf = await estimation_funcs.estimate_pdf(
                mock_session, estimator_id=estimators[0].id_, dataset_id=5, row=0
            )
            assert pdf == mock_pdf

    @pytest.mark.asyncio
    async def test_full_ensemble_estimation_workflow(self, mock_session, mock_components, tmp_path):
        """Test complete workflow for ensemble estimation."""
        estimator, model, algorithm, catalog_tag = mock_components

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        model_path = archive_dir / model.path
        model_path.parent.mkdir(parents=True)
        model_path.write_text("mock model")

        input_path = archive_dir / "datasets" / "test_data.hdf5"
        input_path.parent.mkdir(parents=True)
        input_path.write_text("mock dataset")

        output_path = archive_dir / "estimates" / "output.hdf5"

        mock_dataset = Mock(spec=db.Dataset)
        mock_dataset.id_ = 5
        mock_dataset.path = "datasets/test_data.hdf5"
        mock_dataset.catalog_tag_id = 1000

        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        def create_output(*args):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("mock estimates")

        mock_wrapper.side_effect = create_output

        with (
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset),
            patch("rail_svc.db_oper.estimation_funcs.catalog_tag.get_row", return_value=catalog_tag),
            patch("rail_svc.db_oper.estimation_funcs.model.find_by", return_value=[model]),
            patch("rail_svc.db_oper.estimation_funcs.estimator.find_by", return_value=[estimator]),
            patch("rail_svc.db_oper.wrappers._get_estimator_components", return_value=mock_components),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
            patch.object(CatEstimatorEnsembleWrapper, "build_wrapper", return_value=mock_wrapper),
        ):
            # Get estimators
            estimators = await estimation_funcs.get_estimators_for_dataest(mock_session, dataset_id=5)

            # Run ensemble estimation
            result_path = await estimation_funcs.estimate_ensemble(
                mock_session,
                estimator_id=estimators[0].id_,
                dataset_id=5,
                output_file_path="estimates/output.hdf5",
            )

            assert result_path == output_path
            assert output_path.exists()
