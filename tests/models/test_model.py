"""Unit tests for the Model Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.model import Model, ModelBase, ModelCreate


class TestModelBase:
    """Tests for ModelBase model"""

    def test_valid_model_base(self):
        """Test creating a valid ModelBase"""
        model = ModelBase(
            name="test_model",
            path="/models/test_model.pkl",
        )
        assert model.name == "test_model"
        assert model.path == "/models/test_model.pkl"

    def test_model_base_missing_name(self):
        """Test that name is required"""
        with pytest.raises(ValidationError) as exc_info:
            ModelBase(path="/some/path.pkl")
        assert "name" in str(exc_info.value)

    def test_model_base_missing_path(self):
        """Test that path is required"""
        with pytest.raises(ValidationError) as exc_info:
            ModelBase(name="test")
        assert "path" in str(exc_info.value)


class TestModelCreate:
    """Tests for ModelCreate model"""

    def test_valid_model_create(self):
        """Test creating a valid ModelCreate"""
        model = ModelCreate(
            name="new_model",
            path="/models/new_model.pkl",
            algo_name="KNearNeighEstimator",
            catalog_tag_name="lsst_dp02",
        )
        assert model.name == "new_model"
        assert model.path == "/models/new_model.pkl"
        assert model.algo_name == "KNearNeighEstimator"
        assert model.catalog_tag_name == "lsst_dp02"

    def test_model_create_missing_algo_name(self):
        """Test that algo_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            ModelCreate(
                name="test",
                path="/path.pkl",
                catalog_tag_name="lsst",
            )
        assert "algo_name" in str(exc_info.value)

    def test_model_create_missing_catalog_tag_name(self):
        """Test that catalog_tag_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            ModelCreate(
                name="test",
                path="/path.pkl",
                algo_name="SOMEstimator",
            )
        assert "catalog_tag_name" in str(exc_info.value)


class TestModel:
    """Tests for Model model"""

    def test_valid_model(self):
        """Test creating a valid Model with all fields"""
        model = Model(
            id=1,
            name="trained_model",
            path="/models/trained_model.pkl",
            algo_id=2,
            catalog_tag_id=3,
        )
        assert model.id == 1
        assert model.name == "trained_model"
        assert model.path == "/models/trained_model.pkl"
        assert model.algo_id == 2
        assert model.catalog_tag_id == 3

    def test_model_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=0,
                name="test",
                path="/path.pkl",
                algo_id=1,
                catalog_tag_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=-1,
                name="test",
                path="/path.pkl",
                algo_id=1,
                catalog_tag_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_model_algo_id_must_be_positive(self):
        """Test that algo_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=1,
                name="test",
                path="/path.pkl",
                algo_id=0,
                catalog_tag_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=1,
                name="test",
                path="/path.pkl",
                algo_id=-5,
                catalog_tag_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_model_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=1,
                name="test",
                path="/path.pkl",
                algo_id=1,
                catalog_tag_id=0,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=1,
                name="test",
                path="/path.pkl",
                algo_id=1,
                catalog_tag_id=-3,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_model_missing_id(self):
        """Test that id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Model(
                name="test",
                path="/path.pkl",
                algo_id=1,
                catalog_tag_id=1,
            )
        assert "id" in str(exc_info.value)

    def test_model_missing_algo_id(self):
        """Test that algo_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=1,
                name="test",
                path="/path.pkl",
                catalog_tag_id=1,
            )
        assert "algo_id" in str(exc_info.value)

    def test_model_missing_catalog_tag_id(self):
        """Test that catalog_tag_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Model(
                id=1,
                name="test",
                path="/path.pkl",
                algo_id=1,
            )
        assert "catalog_tag_id" in str(exc_info.value)

    def test_model_from_attributes(self):
        """Test that from_attributes config works"""
        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 5
            name = "orm_model"
            path = "/orm/model.pkl"
            algo_id = 10
            catalog_tag_id = 15

        orm_obj = MockORMObject()
        model = Model.model_validate(orm_obj)
        assert model.id == 5
        assert model.name == "orm_model"
        assert model.path == "/orm/model.pkl"
        assert model.algo_id == 10
        assert model.catalog_tag_id == 15

    def test_model_col_names_for_table(self):
        """Test that col_names_for_table class variable is set correctly"""
        expected_cols = ["id", "name", "algo_id", "catalog_tag_id", "path"]
        assert Model.col_names_for_table == expected_cols

    def test_model_field_descriptions(self):
        """Test that field descriptions are set"""
        schema = Model.model_json_schema()
        assert "Unique name for this model" in schema["properties"]["name"]["description"]
        assert "File path to the stored model" in schema["properties"]["path"]["description"]
        assert "Foreign key referencing Algorithm.id" in schema["properties"]["algo_id"]["description"]
        assert "Foreign key referencing CatalogTag.id" in schema["properties"]["catalog_tag_id"]["description"]
