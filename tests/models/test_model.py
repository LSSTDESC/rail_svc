"""Unit tests for Model Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.model import Model, ModelBase, ModelCreate


class TestModelBase:
    """Tests for ModelBase model"""

    def test_valid_model_base(self):
        """Test creating valid ModelBase instance"""
        model = ModelBase(name="knn_model_v1", path="/models/knn_v1.pkl")
        assert model.name == "knn_model_v1"
        assert model.path == "/models/knn_v1.pkl"

    def test_missing_name_raises_error(self):
        """Test that missing name raises ValidationError"""
        with pytest.raises(ValidationError):
            ModelBase(path="/models/test.pkl")

    def test_missing_path_raises_error(self):
        """Test that missing path raises ValidationError"""
        with pytest.raises(ValidationError):
            ModelBase(name="test_model")

    def test_missing_all_fields_raises_error(self):
        """Test that missing all fields raises ValidationError"""
        with pytest.raises(ValidationError):
            ModelBase()

    def test_empty_string_name(self):
        """Test that empty string is allowed for name"""
        model = ModelBase(name="", path="/path/to/model")
        assert model.name == ""

    def test_empty_string_path(self):
        """Test that empty string is allowed for path"""
        model = ModelBase(name="test", path="")
        assert model.path == ""


class TestModelCreate:
    """Tests for ModelCreate model"""

    def test_valid_model_create(self):
        """Test creating valid ModelCreate instance"""
        model = ModelCreate(
            name="new_model",
            path="/models/new_model.pkl",
            algo_name="SOMEstimator",
            catalog_tag_name="lsst_dp02",
        )
        assert model.name == "new_model"
        assert model.path == "/models/new_model.pkl"
        assert model.algo_name == "SOMEstimator"
        assert model.catalog_tag_name == "lsst_dp02"

    def test_missing_algo_name(self):
        """Test that missing algo_name raises ValidationError"""
        with pytest.raises(ValidationError):
            ModelCreate(name="test", path="/path", catalog_tag_name="test_tag")

    def test_missing_catalog_tag_name(self):
        """Test that missing catalog_tag_name raises ValidationError"""
        with pytest.raises(ValidationError):
            ModelCreate(name="test", path="/path", algo_name="test_algo")

    def test_inherits_base_validation(self):
        """Test that ModelCreate inherits base validation"""
        with pytest.raises(ValidationError):
            ModelCreate(algo_name="test_algo", catalog_tag_name="test_tag")


class TestModel:
    """Tests for Model model"""

    def test_valid_model(self):
        """Test creating valid Model instance"""
        model = Model(
            id_=1, name="production_model", path="/models/prod_model.pkl", algo_id=5, catalog_tag_id=10
        )
        assert model.id_ == 1
        assert model.name == "production_model"
        assert model.path == "/models/prod_model.pkl"
        assert model.algo_id == 5
        assert model.catalog_tag_id == 10

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            Model(id_=0, name="test", path="/path", algo_id=1, catalog_tag_id=1)

        with pytest.raises(ValidationError):
            Model(id_=-1, name="test", path="/path", algo_id=1, catalog_tag_id=1)

    def test_algo_id_must_be_positive(self):
        """Test that algo_id must be greater than 0"""
        with pytest.raises(ValidationError):
            Model(id_=1, name="test", path="/path", algo_id=0, catalog_tag_id=1)

        with pytest.raises(ValidationError):
            Model(id_=1, name="test", path="/path", algo_id=-1, catalog_tag_id=1)

    def test_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        with pytest.raises(ValidationError):
            Model(id_=1, name="test", path="/path", algo_id=1, catalog_tag_id=0)

        with pytest.raises(ValidationError):
            Model(id_=1, name="test", path="/path", algo_id=1, catalog_tag_id=-1)

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 99
            name = "orm_model"
            path = "/models/orm.pkl"
            algo_id = 7
            catalog_tag_id = 12

        model = Model.model_validate(MockORMObject())
        assert model.id_ == 99
        assert model.name == "orm_model"
        assert model.path == "/models/orm.pkl"
        assert model.algo_id == 7
        assert model.catalog_tag_id == 12

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        expected = ["id_", "name", "algo_id", "catalog_tag_id", "path"]
        assert Model.col_names_for_table == expected

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        model = Model(id_=1, name="test", path="/path", algo_id=1, catalog_tag_id=1)
        assert "col_names_for_table" not in model.model_dump()


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_model_to_dict(self):
        """Test converting Model to dict"""
        model = Model(
            id_=42, name="serialize_test", path="/models/serialize.pkl", algo_id=15, catalog_tag_id=25
        )
        data = model.model_dump()
        assert data["id_"] == 42
        assert data["name"] == "serialize_test"
        assert data["path"] == "/models/serialize.pkl"
        assert data["algo_id"] == 15
        assert data["catalog_tag_id"] == 25

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = Model(id_=7, name="json_test", path="/models/json.pkl", algo_id=3, catalog_tag_id=8)
        json_str = original.model_dump_json()
        restored = Model.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.name == original.name
        assert restored.path == original.path
        assert restored.algo_id == original.algo_id
        assert restored.catalog_tag_id == original.catalog_tag_id

    def test_create_model_serialization(self):
        """Test serialization of ModelCreate"""
        create = ModelCreate(
            name="test_model", path="/models/test.pkl", algo_name="KNNEstimator", catalog_tag_name="des_y6"
        )
        data = create.model_dump()
        assert data["name"] == "test_model"
        assert data["path"] == "/models/test.pkl"
        assert data["algo_name"] == "KNNEstimator"
        assert data["catalog_tag_name"] == "des_y6"

    def test_base_model_serialization(self):
        """Test serialization of ModelBase"""
        base = ModelBase(name="base_model", path="/models/base.pkl")
        data = base.model_dump()
        assert data["name"] == "base_model"
        assert data["path"] == "/models/base.pkl"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_model_with_various_path_formats(self):
        """Test Model with different path formats"""
        # Relative path
        m1 = ModelBase(name="test1", path="models/relative.pkl")
        assert m1.path == "models/relative.pkl"

        # Absolute path
        m2 = ModelBase(name="test2", path="/absolute/path/model.pkl")
        assert m2.path == "/absolute/path/model.pkl"

        # URL
        m3 = ModelBase(name="test3", path="s3://bucket/model.pkl")
        assert m3.path == "s3://bucket/model.pkl"

    def test_model_with_different_file_extensions(self):
        """Test Model with various file extensions"""
        for ext in [".pkl", ".yaml", ".json", ".h5", ".pt", ".onnx"]:
            model = ModelBase(name=f"test{ext}", path=f"/models/model{ext}")
            assert model.path.endswith(ext)

    def test_minimum_valid_ids(self):
        """Test Model with minimum valid id values"""
        model = Model(id_=1, name="test", path="/path", algo_id=1, catalog_tag_id=1)
        assert model.id_ == 1
        assert model.algo_id == 1
        assert model.catalog_tag_id == 1

    def test_large_id_values(self):
        """Test Model with large id values"""
        model = Model(id_=999999999, name="test", path="/path", algo_id=888888888, catalog_tag_id=777777777)
        assert model.id_ == 999999999
        assert model.algo_id == 888888888
        assert model.catalog_tag_id == 777777777
