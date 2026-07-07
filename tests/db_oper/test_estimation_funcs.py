"""Integration tests for rail_svc.db_oper.estimation_funcs"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import qp

from rail_svc.db_oper import estimation_funcs
from rail_svc.rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper


class TestGetEstimatorsForDataset:
    """Tests for get_estimators_for_dataest — pure integration, no mocks."""

    @pytest.mark.asyncio
    async def test_finds_estimator_through_model_chain(
        self, session, sample_dataset, sample_model, sample_estimator
    ):
        # Chain: sample_dataset.catalog_tag_id == sample_model.catalog_tag_id
        #        sample_estimator.model_id == sample_model.id_
        result = await estimation_funcs.get_estimators_for_dataest(session, sample_dataset.id_)

        assert len(result) == 1
        assert result[0].id_ == sample_estimator.id_

    @pytest.mark.asyncio
    async def test_no_models_returns_empty(self, session, sample_dataset):
        # sample_dataset has catalog_tag, but no model references that catalog_tag yet
        # (sample_model fixture not loaded)
        result = await estimation_funcs.get_estimators_for_dataest(session, sample_dataset.id_)

        assert result == []

    @pytest.mark.asyncio
    async def test_dataset_not_found(self, session):
        with pytest.raises(KeyError):
            await estimation_funcs.get_estimators_for_dataest(session, 99999)

    @pytest.mark.asyncio
    async def test_multiple_estimators(
        self, session, sample_dataset, sample_model, multiple_estimators
    ):
        result = await estimation_funcs.get_estimators_for_dataest(session, sample_dataset.id_)

        assert len(result) == 3
        ids = {e.id_ for e in result}
        for est in multiple_estimators:
            assert est.id_ in ids


class TestEstimatePdf:
    """Tests for estimate_pdf — mock wrapper + file I/O, real DB."""

    @pytest.mark.asyncio
    async def test_successful_estimation(self, session, sample_dataset, sample_estimator):
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)
        mock_pdf = Mock(spec=qp.Ensemble)
        mock_wrapper.return_value = mock_pdf
        catalog_data = {"mag_g": 20.0, "mag_r": 19.5}

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch(
                "rail_svc.db_oper.estimation_funcs.catalog_funcs.get_catalog_row",
                return_value=catalog_data,
            ),
        ):
            result = await estimation_funcs.estimate_pdf(
                session,
                estimator_id=sample_estimator.id_,
                dataset_id=sample_dataset.id_,
                row=42,
            )

        assert result == mock_pdf
        mock_wrapper.assert_called_once()

    @pytest.mark.asyncio
    async def test_catalog_read_failure(self, session, sample_dataset, sample_estimator):
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch(
                "rail_svc.db_oper.estimation_funcs.catalog_funcs.get_catalog_row",
                side_effect=FileNotFoundError("Catalog not found"),
            ),
        ):
            with pytest.raises(FileNotFoundError):
                await estimation_funcs.estimate_pdf(
                    session,
                    estimator_id=sample_estimator.id_,
                    dataset_id=sample_dataset.id_,
                    row=0,
                )


class TestEstimateEnsemble:
    """Tests for estimate_ensemble — mock wrapper, mock dataset.get_row for path control."""

    @pytest.mark.asyncio
    async def test_successful_estimation(self, session, sample_estimator, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        dataset_rel = "datasets/test.hdf5"
        input_path = archive_dir / dataset_rel
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("mock dataset")

        output_rel = "estimates/output.hdf5"
        output_path = archive_dir / output_rel

        mock_dataset_obj = Mock()
        mock_dataset_obj.path = dataset_rel

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
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset_obj),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
        ):
            result = await estimation_funcs.estimate_ensemble(
                session,
                estimator_id=sample_estimator.id_,
                dataset_id=1,
                output_file_path=output_rel,
            )

        assert result == output_path
        assert output_path.exists()

    @pytest.mark.asyncio
    async def test_input_file_not_found(self, session, sample_estimator, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()

        mock_dataset_obj = Mock()
        mock_dataset_obj.path = "datasets/missing.hdf5"

        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset_obj),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
        ):
            with pytest.raises(FileNotFoundError):
                await estimation_funcs.estimate_ensemble(
                    session,
                    estimator_id=sample_estimator.id_,
                    dataset_id=1,
                    output_file_path="output.hdf5",
                )

    @pytest.mark.asyncio
    async def test_output_not_created(self, session, sample_estimator, tmp_path):
        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        dataset_rel = "datasets/test.hdf5"
        input_path = archive_dir / dataset_rel
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("mock dataset")

        mock_dataset_obj = Mock()
        mock_dataset_obj.path = dataset_rel

        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)
        mock_wrapper.return_value = None

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper",
                return_value=mock_wrapper,
            ),
            patch("rail_svc.db_oper.estimation_funcs.dataset.get_row", return_value=mock_dataset_obj),
            patch("rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(archive_dir)),
            patch("anyio.Path.absolute", return_value=archive_dir),
        ):
            with pytest.raises(OSError, match="Output file was not created"):
                await estimation_funcs.estimate_ensemble(
                    session,
                    estimator_id=sample_estimator.id_,
                    dataset_id=1,
                    output_file_path="output.hdf5",
                )


class TestBuildWrapperLists:
    """Tests for wrapper list construction — mock wrappers, real estimator lookup."""

    @pytest.mark.asyncio
    async def test_builds_pdf_wrappers(self, session, sample_dataset, sample_model, sample_estimator):
        mock_wrapper = Mock(spec=CatEstimatorPdfWrapper)

        with patch(
            "rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper",
            return_value=mock_wrapper,
        ):
            result = await estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(
                session, sample_dataset.id_
            )

        assert len(result) == 1
        assert result[0] == mock_wrapper

    @pytest.mark.asyncio
    async def test_builds_ensemble_wrappers(
        self, session, sample_dataset, sample_model, sample_estimator
    ):
        mock_wrapper = Mock(spec=CatEstimatorEnsembleWrapper)

        with patch(
            "rail_svc.db_oper.estimation_funcs.build_ensemble_estimation_wrapper",
            return_value=mock_wrapper,
        ):
            result = await estimation_funcs.build_cat_estimator_ensemble_wrappers_for_dataset(
                session, sample_dataset.id_
            )

        assert len(result) == 1
        assert result[0] == mock_wrapper

    @pytest.mark.asyncio
    async def test_partial_failures_are_skipped(
        self, session, sample_dataset, sample_model, multiple_estimators, caplog
    ):
        def side_effect(session, est_id):
            if est_id == multiple_estimators[1].id_:
                raise ValueError("Model file missing")
            return Mock(spec=CatEstimatorPdfWrapper)

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.build_pdf_estimation_wrapper",
                side_effect=side_effect,
            ),
            caplog.at_level(logging.WARNING),
        ):
            result = await estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(
                session, sample_dataset.id_
            )

        assert len(result) == 2
        assert "Failed to build estimator" in caplog.text

    @pytest.mark.asyncio
    async def test_no_estimators_returns_empty(self, session, sample_dataset):
        result = await estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(
            session, sample_dataset.id_
        )

        assert result == []


class TestEstimateDataset:
    """Tests for estimate_dataset — real DB, mock ensemble estimation."""

    @pytest.mark.asyncio
    async def test_creates_estimates_record(
        self, session, sample_dataset, sample_estimator, tmp_path
    ):
        estimates_name = f"{sample_dataset.name}__{sample_estimator.name}"
        estimates_rel = f"estimates/{estimates_name}.hdf5"
        estimates_path = tmp_path / estimates_rel
        estimates_path.parent.mkdir(parents=True, exist_ok=True)
        estimates_path.write_text("mock data")

        with (
            patch(
                "rail_svc.db_oper.estimation_funcs.estimate_ensemble",
                return_value=estimates_path,
            ),
            patch(
                "rail_svc.db_oper.estimation_funcs.global_config.storage.archive", str(tmp_path)
            ),
            patch.object(
                estimation_funcs.estimates, "_validate_path_security",
                return_value=estimates_path,
            ),
            patch.object(
                estimation_funcs.estimates, "validate_data_for_path",
                return_value=sample_dataset.n_objects,
            ),
        ):
            result = await estimation_funcs.estimate_dataset(
                session,
                estimator_id=sample_estimator.id_,
                dataset_id=sample_dataset.id_,
            )

        assert result.name == estimates_name
        assert result.dataset_id == sample_dataset.id_
        assert result.estimator_id == sample_estimator.id_
        assert result.n_objects == sample_dataset.n_objects

    @pytest.mark.asyncio
    async def test_returns_existing_estimates(self, session, sample_dataset, sample_estimates):
        result = await estimation_funcs.estimate_dataset(
            session,
            estimator_id=sample_estimates.estimator_id,
            dataset_id=sample_dataset.id_,
        )

        assert result.id_ == sample_estimates.id_

    @pytest.mark.asyncio
    async def test_raises_if_exists_and_flag_set(self, session, sample_dataset, sample_estimates):
        with pytest.raises(ValueError, match="already exist"):
            await estimation_funcs.estimate_dataset(
                session,
                estimator_id=sample_estimates.estimator_id,
                dataset_id=sample_dataset.id_,
                raise_if_exists=True,
            )
