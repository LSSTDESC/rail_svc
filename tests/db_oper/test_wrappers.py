"""Integration tests for rail_svc.db_oper.wrappers using real in-memory SQLite."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from rail_svc.db import Algorithm, CatalogTag, Estimator, Model
from rail_svc.db_oper.wrappers import (
    WrapperType,
    _build_estimation_wrapper,
    _get_estimator_components,
    build_ensemble_estimation_wrapper,
    build_pdf_estimation_wrapper,
)
from rail_svc.rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper
from tests.conftest import create


class TestGetEstimatorComponents:
    """Test _get_estimator_components against real DB."""

    @pytest.mark.asyncio
    async def test_fetches_all_components(
        self, session, sample_estimator, sample_model, sample_algorithm, sample_catalog_tag
    ):
        """Test that all related components are fetched correctly."""
        estimator_obj, model_obj, algo_obj, catalog_tag_obj = await _get_estimator_components(
            session, sample_estimator.id_
        )

        assert estimator_obj.id_ == sample_estimator.id_
        assert estimator_obj.name == sample_estimator.name
        assert model_obj.id_ == sample_model.id_
        assert algo_obj.id_ == sample_algorithm.id_
        assert catalog_tag_obj.id_ == sample_catalog_tag.id_

    @pytest.mark.asyncio
    async def test_invalid_estimator_id(self, session):
        """Test error when estimator doesn't exist."""
        with pytest.raises(Exception):
            await _get_estimator_components(session, 99999)

    @pytest.mark.asyncio
    async def test_returns_correct_types(self, session, sample_estimator):
        """Test that returned objects are the correct ORM types."""
        from rail_svc import db

        estimator_obj, model_obj, algo_obj, catalog_tag_obj = await _get_estimator_components(
            session, sample_estimator.id_
        )

        assert isinstance(estimator_obj, db.Estimator)
        assert isinstance(model_obj, db.Model)
        assert isinstance(algo_obj, db.Algorithm)
        assert isinstance(catalog_tag_obj, db.CatalogTag)


@pytest.fixture
async def wrapper_fixtures(session, tmp_path):
    """Create DB objects with relative model path and matching tmp files."""
    algo = await create(session, Algorithm(name="wrapper_algo", class_name="rail.estimation.SOMEstimator"))
    cat_tag = await create(session, CatalogTag(name="wrapper_catalog"))
    mdl = await create(
        session,
        Model(name="wrapper_model", path="models/knn.pkl", algo_id=algo.id_, catalog_tag_id=cat_tag.id_),
    )
    est = await create(
        session,
        Estimator(
            name="wrapper_estimator", config={"n_neighbors": 5, "weights": "distance"}, model_id=mdl.id_
        ),
    )

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    model_file = archive_dir / "models" / "knn.pkl"
    model_file.parent.mkdir(parents=True)
    model_file.write_text("mock model")

    return {
        "algorithm": algo,
        "catalog_tag": cat_tag,
        "model": mdl,
        "estimator": est,
        "archive_dir": archive_dir,
        "model_file": model_file,
    }


def _patch_anyio_absolute(archive_dir):
    """Return a patch for anyio.Path(...).absolute() to return archive_dir."""
    mock_anyio_path_instance = Mock()
    mock_anyio_path_instance.absolute = AsyncMock(return_value=archive_dir)
    return patch("rail_svc.db_oper.wrappers.anyio.Path", return_value=mock_anyio_path_instance)


class TestBuildEstimationWrapper:
    """Test _build_estimation_wrapper with real DB + mocked filesystem/RAIL."""

    @pytest.mark.asyncio
    async def test_builds_pdf_wrapper(self, session, wrapper_fixtures):
        fx = wrapper_fixtures
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)

        with (
            patch("rail_svc.db_oper.wrappers.global_config") as mock_config,
            _patch_anyio_absolute(fx["archive_dir"]),
            patch.object(CatEstimatorPdfWrapper, "build_wrapper", return_value=mock_wrapper) as mock_build,
        ):
            mock_config.storage.archive = str(fx["archive_dir"])
            result = await _build_estimation_wrapper(session, fx["estimator"].id_, WrapperType.PDF)

            assert result is mock_wrapper
            mock_build.assert_called_once_with(
                "wrapper_estimator",
                "rail.estimation.SOMEstimator",
                fx["model_file"],
                "wrapper_catalog",
                n_neighbors=5,
                weights="distance",
            )

    @pytest.mark.asyncio
    async def test_builds_ensemble_wrapper(self, session, wrapper_fixtures):
        fx = wrapper_fixtures
        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        with (
            patch("rail_svc.db_oper.wrappers.global_config") as mock_config,
            _patch_anyio_absolute(fx["archive_dir"]),
            patch.object(
                CatEstimatorEnsembleWrapper, "build_wrapper", return_value=mock_wrapper
            ) as mock_build,
        ):
            mock_config.storage.archive = str(fx["archive_dir"])
            result = await _build_estimation_wrapper(session, fx["estimator"].id_, WrapperType.ENSEMBLE)

            assert result is mock_wrapper
            mock_build.assert_called_once_with(
                "wrapper_estimator",
                "rail.estimation.SOMEstimator",
                fx["model_file"],
                "wrapper_catalog",
                n_neighbors=5,
                weights="distance",
            )

    @pytest.mark.asyncio
    async def test_model_file_not_found(self, session, wrapper_fixtures):
        fx = wrapper_fixtures
        fx["model_file"].unlink()

        with (
            patch("rail_svc.db_oper.wrappers.global_config") as mock_config,
            _patch_anyio_absolute(fx["archive_dir"]),
        ):
            mock_config.storage.archive = str(fx["archive_dir"])
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                await _build_estimation_wrapper(session, fx["estimator"].id_, WrapperType.PDF)

    @pytest.mark.asyncio
    async def test_none_config_defaults_to_empty_dict(self, session, wrapper_fixtures):
        fx = wrapper_fixtures
        estimator_no_config = await create(
            session,
            Estimator(name="no_config_estimator", config=None, model_id=fx["model"].id_),
        )

        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)

        with (
            patch("rail_svc.db_oper.wrappers.global_config") as mock_config,
            _patch_anyio_absolute(fx["archive_dir"]),
            patch.object(CatEstimatorPdfWrapper, "build_wrapper", return_value=mock_wrapper) as mock_build,
        ):
            mock_config.storage.archive = str(fx["archive_dir"])
            result = await _build_estimation_wrapper(session, estimator_no_config.id_, WrapperType.PDF)

            assert result is mock_wrapper
            mock_build.assert_called_once_with(
                "no_config_estimator",
                "rail.estimation.SOMEstimator",
                fx["model_file"],
                "wrapper_catalog",
            )

    @pytest.mark.asyncio
    async def test_invalid_estimator_id_propagates(self, session):
        with pytest.raises(Exception):
            await _build_estimation_wrapper(session, 99999, WrapperType.PDF)


class TestBuildPdfEstimationWrapper:
    """Test build_pdf_estimation_wrapper convenience function."""

    @pytest.mark.asyncio
    async def test_returns_pdf_wrapper(self, session, wrapper_fixtures):
        fx = wrapper_fixtures
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)

        with (
            patch("rail_svc.db_oper.wrappers.global_config") as mock_config,
            _patch_anyio_absolute(fx["archive_dir"]),
            patch.object(CatEstimatorPdfWrapper, "build_wrapper", return_value=mock_wrapper),
        ):
            mock_config.storage.archive = str(fx["archive_dir"])
            result = await build_pdf_estimation_wrapper(session, fx["estimator"].id_)

        assert result is mock_wrapper


class TestBuildEnsembleEstimationWrapper:
    """Test build_ensemble_estimation_wrapper convenience function."""

    @pytest.mark.asyncio
    async def test_returns_ensemble_wrapper(self, session, wrapper_fixtures):
        fx = wrapper_fixtures
        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        with (
            patch("rail_svc.db_oper.wrappers.global_config") as mock_config,
            _patch_anyio_absolute(fx["archive_dir"]),
            patch.object(CatEstimatorEnsembleWrapper, "build_wrapper", return_value=mock_wrapper),
        ):
            mock_config.storage.archive = str(fx["archive_dir"])
            result = await build_ensemble_estimation_wrapper(session, fx["estimator"].id_)

        assert result is mock_wrapper
