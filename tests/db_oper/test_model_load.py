"""Tests for Model load operations."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from macon.common import LoadType
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from rail_svc import models
from rail_svc.db_oper.model import model


@pytest.fixture
def mock_get_session(engine):
    """Provide a get_session callable that yields fresh sessions from the test engine."""
    factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    @asynccontextmanager
    async def _get_session():
        async with factory() as sess:
            yield sess

    return _get_session


class TestModelLoad:
    """Test ModelOperations.load."""

    @pytest.mark.asyncio
    async def test_load_without_validation(
        self, mock_get_session, sample_algorithm, sample_catalog_tag, tmp_path
    ):
        """Test loading a model with validate_file=False."""
        source_file = tmp_path / "model.pkl"
        source_file.write_bytes(b"pickled model")

        with patch("rail_svc.db_oper.model.get_session", mock_get_session):
            result = await model.load(
                name="test_model",
                orig_path=str(source_file),
                algo_name=sample_algorithm.name,
                catalog_tag_name=sample_catalog_tag.name,
                load_type=LoadType.in_place,
                validate_file=False,
            )

            assert isinstance(result, models.Model)
            assert result.name == "test_model"
            assert result.algo_id == sample_algorithm.id_
            assert result.catalog_tag_id == sample_catalog_tag.id_
            assert "model.pkl" in result.path

    @pytest.mark.asyncio
    async def test_load_with_copy(self, mock_get_session, sample_algorithm, sample_catalog_tag, tmp_path):
        """Test loading a model with copy creates file in archive."""
        source_file = tmp_path / "rf_model.pkl"
        source_file.write_bytes(b"random forest model")

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "models").mkdir()

        with (
            patch("rail_svc.db_oper.model.get_session", mock_get_session),
            patch("macon.config.config.storage.archive", str(archive_dir)),
        ):
            result = await model.load(
                name="copied_model",
                orig_path=str(source_file),
                algo_name=sample_algorithm.name,
                catalog_tag_name=sample_catalog_tag.name,
                load_type=LoadType.copy,
                validate_file=False,
            )

            assert result.name == "copied_model"
            copied_path = archive_dir / "models" / "copied_model_rf_model.pkl"
            assert copied_path.exists()
            assert copied_path.read_bytes() == b"random forest model"

    @pytest.mark.asyncio
    async def test_load_with_link(self, mock_get_session, sample_algorithm, sample_catalog_tag, tmp_path):
        """Test loading a model with symlink."""
        source_file = tmp_path / "linked_model.pkl"
        source_file.write_bytes(b"model data")

        archive_dir = tmp_path / "archive"
        archive_dir.mkdir()
        (archive_dir / "models").mkdir()

        with (
            patch("rail_svc.db_oper.model.get_session", mock_get_session),
            patch("macon.config.config.storage.archive", str(archive_dir)),
        ):
            result = await model.load(
                name="linked_model",
                orig_path=str(source_file),
                algo_name=sample_algorithm.name,
                catalog_tag_name=sample_catalog_tag.name,
                load_type=LoadType.link,
                validate_file=False,
            )

            assert result.name == "linked_model"
            link_path = archive_dir / "models" / "linked_model_linked_model.pkl"
            assert link_path.is_symlink()

    @pytest.mark.asyncio
    async def test_load_with_validation(
        self, mock_get_session, sample_algorithm, sample_catalog_tag, tmp_path
    ):
        """Test loading a model with validation enabled (catalog_tag matches)."""
        source_file = tmp_path / "validated_model.pkl"
        source_file.write_bytes(b"valid model")

        mock_rail_model = MagicMock()
        mock_rail_model.catalog_tag = sample_catalog_tag.name
        mock_rail_model.creation_class_name = None

        with (
            patch("rail_svc.db_oper.model.get_session", mock_get_session),
            patch("rail_svc.db_oper.model.RailModel.read", return_value=mock_rail_model),
            patch("anyio.Path.resolve", new_callable=AsyncMock, return_value=source_file),
        ):
            result = await model.load(
                name="validated_model",
                orig_path=str(source_file),
                algo_name=sample_algorithm.name,
                catalog_tag_name=sample_catalog_tag.name,
                load_type=LoadType.in_place,
                validate_file=True,
            )

            assert isinstance(result, models.Model)
            assert result.name == "validated_model"

    @pytest.mark.asyncio
    async def test_load_validation_catalog_tag_mismatch(
        self, mock_get_session, sample_algorithm, sample_catalog_tag, tmp_path
    ):
        """Test that validation fails on catalog tag mismatch."""
        source_file = tmp_path / "bad_model.pkl"
        source_file.write_bytes(b"model with wrong tag")

        mock_rail_model = MagicMock()
        mock_rail_model.catalog_tag = "wrong_catalog"
        mock_rail_model.creation_class_name = None

        with (
            patch("rail_svc.db_oper.model.get_session", mock_get_session),
            patch("rail_svc.db_oper.model.RailModel.read", return_value=mock_rail_model),
            patch("anyio.Path.resolve", new_callable=AsyncMock, return_value=source_file),
        ):
            with pytest.raises(ValueError, match="CatalogTag mismatch"):
                await model.load(
                    name="bad_model",
                    orig_path=str(source_file),
                    algo_name=sample_algorithm.name,
                    catalog_tag_name=sample_catalog_tag.name,
                    load_type=LoadType.in_place,
                    validate_file=True,
                )


class TestConvertInformerToEstimator:
    """Test the _convert_informer_to_estimator helper."""

    def test_standard_pattern(self):
        """Test standard Informer -> Estimator conversion."""
        assert model._convert_informer_to_estimator("RandomForestInformer") == "RandomForestEstimator"

    def test_another_pattern(self):
        """Test another Informer -> Estimator conversion."""
        assert model._convert_informer_to_estimator("BPZInformer") == "BPZEstimator"

    def test_unknown_pattern_raises(self):
        """Test that non-Informer suffix raises ValueError."""
        with pytest.raises(ValueError, match="Cannot convert"):
            model._convert_informer_to_estimator("SomeRandomClass")
