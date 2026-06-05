"""Unit tests for Estimator Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.estimator import Estimator, EstimatorBase, EstimatorCreate


class TestEstimatorBase:
    """Tests for EstimatorBase model"""

    def test_valid_estimator_base(self):
        """Test creating valid EstimatorBase instance"""
        estimator = EstimatorBase(name="knn_estimator", config={"n_neighbors": 5, "weights": "distance"})
        assert estimator.name == "knn_estimator"
        assert estimator.config == {"n_neighbors": 5, "weights": "distance"}

    def test_config_can_be_none(self):
        """Test that config can be None"""
        estimator = EstimatorBase(name="test", config=None)
        assert estimator.config is None

    def test_config_defaults_to_none(self):
        """Test that config defaults to None when not provided"""
        estimator = EstimatorBase(name="test")
        assert estimator.config is None

    def test_missing_name_raises_error(self):
        """Test that missing name raises ValidationError"""
        with pytest.raises(ValidationError):
            EstimatorBase()

    def test_empty_string_name(self):
        """Test that empty string is allowed for name"""
        estimator = EstimatorBase(name="")
        assert estimator.name == ""

    def test_empty_dict_config(self):
        """Test that empty dict is allowed for config"""
        estimator = EstimatorBase(name="test", config={})
        assert estimator.config == {}

    def test_nested_config_dict(self):
        """Test that nested dictionaries are allowed in config"""
        estimator = EstimatorBase(
            name="test", config={"model_params": {"layers": [10, 20]}, "training": {"epochs": 100}}
        )
        assert estimator.config["model_params"]["layers"] == [10, 20]


class TestEstimatorCreate:
    """Tests for EstimatorCreate model"""

    def test_valid_estimator_create(self):
        """Test creating valid EstimatorCreate instance"""
        estimator = EstimatorCreate(
            name="new_estimator", config={"param1": "value1"}, model_name="trained_model_v1"
        )
        assert estimator.name == "new_estimator"
        assert estimator.config == {"param1": "value1"}
        assert estimator.model_name == "trained_model_v1"

    def test_missing_model_name(self):
        """Test that missing model_name raises ValidationError"""
        with pytest.raises(ValidationError):
            EstimatorCreate(name="test")

    def test_inherits_base_validation(self):
        """Test that EstimatorCreate inherits base validation"""
        with pytest.raises(ValidationError):
            EstimatorCreate(model_name="test_model")

    def test_create_with_none_config(self):
        """Test creating EstimatorCreate with None config"""
        estimator = EstimatorCreate(name="test", config=None, model_name="model")
        assert estimator.config is None


class TestEstimator:
    """Tests for Estimator model"""

    def test_valid_estimator(self):
        """Test creating valid Estimator instance"""
        estimator = Estimator(id_=1, name="production_estimator", config={"learning_rate": 0.001}, model_id=5)
        assert estimator.id_ == 1
        assert estimator.name == "production_estimator"
        assert estimator.config == {"learning_rate": 0.001}
        assert estimator.model_id == 5

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            Estimator(id_=0, name="test", model_id=1)

        with pytest.raises(ValidationError):
            Estimator(id_=-1, name="test", model_id=1)

    def test_model_id_must_be_positive(self):
        """Test that model_id must be greater than 0"""
        with pytest.raises(ValidationError):
            Estimator(id_=1, name="test", model_id=0)

        with pytest.raises(ValidationError):
            Estimator(id_=1, name="test", model_id=-1)

    def test_estimator_with_none_config(self):
        """Test creating Estimator with None config"""
        estimator = Estimator(id_=1, name="test", config=None, model_id=1)
        assert estimator.config is None

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 99
            name = "orm_estimator"
            config = {"param": "value"}
            model_id = 7

        estimator = Estimator.model_validate(MockORMObject())
        assert estimator.id_ == 99
        assert estimator.name == "orm_estimator"
        assert estimator.config == {"param": "value"}
        assert estimator.model_id == 7

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        expected = ["id_", "name", "model_id"]
        assert Estimator.col_names_for_table == expected

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        estimator = Estimator(id_=1, name="test", model_id=1)
        assert "col_names_for_table" not in estimator.model_dump()


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_estimator_to_dict(self):
        """Test converting Estimator to dict"""
        estimator = Estimator(
            id_=42, name="serialize_test", config={"batch_size": 32, "epochs": 10}, model_id=15
        )
        data = estimator.model_dump()
        assert data["id_"] == 42
        assert data["name"] == "serialize_test"
        assert data["config"] == {"batch_size": 32, "epochs": 10}
        assert data["model_id"] == 15

    def test_estimator_with_none_config_serialization(self):
        """Test serialization when config is None"""
        estimator = Estimator(id_=1, name="no_config", config=None, model_id=1)
        data = estimator.model_dump()
        assert data["config"] is None

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = Estimator(id_=7, name="json_test", config={"key": "value"}, model_id=3)
        json_str = original.model_dump_json()
        restored = Estimator.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.name == original.name
        assert restored.config == original.config
        assert restored.model_id == original.model_id

    def test_create_model_serialization(self):
        """Test serialization of EstimatorCreate"""
        create = EstimatorCreate(name="test_estimator", config={"param": "value"}, model_name="test_model")
        data = create.model_dump()
        assert data["name"] == "test_estimator"
        assert data["config"] == {"param": "value"}
        assert data["model_name"] == "test_model"

    def test_complex_config_serialization(self):
        """Test serialization of complex nested config"""
        config = {"optimizer": {"type": "adam", "lr": 0.001}, "layers": [64, 128, 256], "dropout": 0.5}
        estimator = Estimator(id_=1, name="complex_test", config=config, model_id=1)
        json_str = estimator.model_dump_json()
        restored = Estimator.model_validate_json(json_str)
        assert restored.config == config
