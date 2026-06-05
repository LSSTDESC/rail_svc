"""Unit tests for Estimates Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.estimates import Estimates, EstimatesBase, EstimatesCreate


class TestEstimatesBase:
    """Tests for EstimatesBase model"""

    def test_valid_estimates_base(self):
        """Test creating valid EstimatesBase instance"""
        estimates = EstimatesBase(name="test_estimates", path="/data/estimates.hdf5")
        assert estimates.name == "test_estimates"
        assert estimates.path == "/data/estimates.hdf5"

    def test_path_can_be_none(self):
        """Test that path can be None"""
        estimates = EstimatesBase(name="test", path=None)
        assert estimates.path is None

    def test_path_defaults_to_none(self):
        """Test that path defaults to None when not provided"""
        estimates = EstimatesBase(name="test")
        assert estimates.path is None

    def test_missing_name_raises_error(self):
        """Test that missing name raises ValidationError"""
        with pytest.raises(ValidationError):
            EstimatesBase()

    def test_empty_string_name(self):
        """Test that empty string is allowed for name"""
        estimates = EstimatesBase(name="")
        assert estimates.name == ""


class TestEstimatesCreate:
    """Tests for EstimatesCreate model"""

    def test_valid_estimates_create(self):
        """Test creating valid EstimatesCreate instance"""
        estimates = EstimatesCreate(
            name="new_estimates",
            path="/data/new_estimates.hdf5",
            estimator_name="knn_estimator",
            dataset_name="validation_set",
        )
        assert estimates.name == "new_estimates"
        assert estimates.path == "/data/new_estimates.hdf5"
        assert estimates.estimator_name == "knn_estimator"
        assert estimates.dataset_name == "validation_set"

    def test_missing_estimator_name(self):
        """Test that missing estimator_name raises ValidationError"""
        with pytest.raises(ValidationError):
            EstimatesCreate(name="test", dataset_name="test_dataset")

    def test_missing_dataset_name(self):
        """Test that missing dataset_name raises ValidationError"""
        with pytest.raises(ValidationError):
            EstimatesCreate(name="test", estimator_name="test_estimator")

    def test_inherits_base_validation(self):
        """Test that EstimatesCreate inherits base validation"""
        with pytest.raises(ValidationError):
            EstimatesCreate(estimator_name="test_estimator", dataset_name="test_dataset")

    def test_create_with_none_path(self):
        """Test creating EstimatesCreate with None path"""
        estimates = EstimatesCreate(
            name="test", path=None, estimator_name="estimator", dataset_name="dataset"
        )
        assert estimates.path is None


class TestEstimates:
    """Tests for Estimates model"""

    def test_valid_estimates(self):
        """Test creating valid Estimates instance"""
        estimates = Estimates(
            id_=1,
            name="production_estimates",
            path="/data/prod_estimates.hdf5",
            estimator_id=5,
            dataset_id=10,
        )
        assert estimates.id_ == 1
        assert estimates.name == "production_estimates"
        assert estimates.path == "/data/prod_estimates.hdf5"
        assert estimates.estimator_id == 5
        assert estimates.dataset_id == 10

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            Estimates(id_=0, name="test", estimator_id=1, dataset_id=1)

        with pytest.raises(ValidationError):
            Estimates(id_=-1, name="test", estimator_id=1, dataset_id=1)

    def test_estimator_id_must_be_positive(self):
        """Test that estimator_id must be greater than 0"""
        with pytest.raises(ValidationError):
            Estimates(id_=1, name="test", estimator_id=0, dataset_id=1)

        with pytest.raises(ValidationError):
            Estimates(id_=1, name="test", estimator_id=-1, dataset_id=1)

    def test_dataset_id_must_be_positive(self):
        """Test that dataset_id must be greater than 0"""
        with pytest.raises(ValidationError):
            Estimates(id_=1, name="test", estimator_id=1, dataset_id=0)

        with pytest.raises(ValidationError):
            Estimates(id_=1, name="test", estimator_id=1, dataset_id=-1)

    def test_estimates_with_none_path(self):
        """Test creating Estimates with None path"""
        estimates = Estimates(id_=1, name="test", path=None, estimator_id=1, dataset_id=1)
        assert estimates.path is None

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 99
            name = "orm_estimates"
            path = "/data/orm_estimates.hdf5"
            estimator_id = 7
            dataset_id = 12

        estimates = Estimates.model_validate(MockORMObject())
        assert estimates.id_ == 99
        assert estimates.name == "orm_estimates"
        assert estimates.estimator_id == 7
        assert estimates.dataset_id == 12

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        estimates = Estimates(id_=1, name="test", estimator_id=1, dataset_id=1)
        assert "col_names_for_table" not in estimates.model_dump()


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_estimates_to_dict(self):
        """Test converting Estimates to dict"""
        estimates = Estimates(
            id_=42, name="serialize_test", path="/data/serialize.hdf5", estimator_id=15, dataset_id=25
        )
        data = estimates.model_dump()
        assert data["id_"] == 42
        assert data["name"] == "serialize_test"
        assert data["path"] == "/data/serialize.hdf5"
        assert data["estimator_id"] == 15
        assert data["dataset_id"] == 25

    def test_estimates_with_none_path_serialization(self):
        """Test serialization when path is None"""
        estimates = Estimates(id_=1, name="no_path", path=None, estimator_id=1, dataset_id=1)
        data = estimates.model_dump()
        assert data["path"] is None

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = Estimates(id_=7, name="json_test", path="/data/json.hdf5", estimator_id=3, dataset_id=8)
        json_str = original.model_dump_json()
        restored = Estimates.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.name == original.name
        assert restored.path == original.path
        assert restored.estimator_id == original.estimator_id
        assert restored.dataset_id == original.dataset_id

    def test_create_model_serialization(self):
        """Test serialization of EstimatesCreate"""
        create = EstimatesCreate(
            name="test_estimates",
            path="/data/test.hdf5",
            estimator_name="test_estimator",
            dataset_name="test_dataset",
        )
        data = create.model_dump()
        assert data["name"] == "test_estimates"
        assert data["path"] == "/data/test.hdf5"
        assert data["estimator_name"] == "test_estimator"
        assert data["dataset_name"] == "test_dataset"
