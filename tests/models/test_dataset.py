"""Unit tests for Dataset Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.dataset import Dataset, DatasetBase, DatasetCreate


class TestDatasetBase:
    """Tests for DatasetBase model"""

    def test_valid_dataset_base(self):
        """Test creating valid DatasetBase instance"""
        dataset = DatasetBase(name="test_dataset", path="/data/test.hdf5", n_objects=10000)
        assert dataset.name == "test_dataset"
        assert dataset.path == "/data/test.hdf5"
        assert dataset.n_objects == 10000
        assert dataset.is_collection is False

    def test_path_can_be_none(self):
        """Test that path can be None"""
        dataset = DatasetBase(name="test", path=None, n_objects=100)
        assert dataset.path is None

    def test_path_defaults_to_none(self):
        """Test that path defaults to None when not provided"""
        dataset = DatasetBase(name="test", n_objects=100)
        assert dataset.path is None

    def test_is_collection_defaults_to_false(self):
        """Test that is_collection defaults to False"""
        dataset = DatasetBase(name="test", n_objects=100)
        assert dataset.is_collection is False

    def test_is_collection_can_be_true(self):
        """Test that is_collection can be set to True"""
        dataset = DatasetBase(name="test", n_objects=100, is_collection=True)
        assert dataset.is_collection is True

    def test_missing_required_fields(self):
        """Test that missing required fields raises ValidationError"""
        with pytest.raises(ValidationError):
            DatasetBase()

        with pytest.raises(ValidationError):
            DatasetBase(name="test")

        with pytest.raises(ValidationError):
            DatasetBase(n_objects=100)

    def test_n_objects_can_be_zero(self):
        """Test that n_objects can be zero"""
        dataset = DatasetBase(name="empty", n_objects=0)
        assert dataset.n_objects == 0

    def test_negative_n_objects_allowed(self):
        """Test that negative n_objects is allowed (no validation constraint)"""
        dataset = DatasetBase(name="test", n_objects=-1)
        assert dataset.n_objects == -1


class TestDatasetCreate:
    """Tests for DatasetCreate model"""

    def test_valid_dataset_create(self):
        """Test creating valid DatasetCreate instance"""
        dataset = DatasetCreate(
            name="new_dataset", path="/data/new.hdf5", n_objects=5000, catalog_tag_name="lsst_dp02"
        )
        assert dataset.name == "new_dataset"
        assert dataset.catalog_tag_name == "lsst_dp02"
        assert dataset.validate_file is False

    def test_validate_file_defaults_to_false(self):
        """Test that validate_file defaults to False"""
        dataset = DatasetCreate(name="test", n_objects=100, catalog_tag_name="test_tag")
        assert dataset.validate_file is False

    def test_validate_file_can_be_true(self):
        """Test that validate_file can be set to True"""
        dataset = DatasetCreate(name="test", n_objects=100, catalog_tag_name="test_tag", validate_file=True)
        assert dataset.validate_file is True

    def test_missing_catalog_tag_name(self):
        """Test that missing catalog_tag_name raises ValidationError"""
        with pytest.raises(ValidationError):
            DatasetCreate(name="test", n_objects=100)

    def test_inherits_base_validation(self):
        """Test that DatasetCreate inherits base validation"""
        with pytest.raises(ValidationError):
            DatasetCreate(catalog_tag_name="test_tag")


class TestDataset:
    """Tests for Dataset model"""

    def test_valid_dataset(self):
        """Test creating valid Dataset instance"""
        dataset = Dataset(
            id_=1, name="production_dataset", path="/data/prod.hdf5", n_objects=1000000, catalog_tag_id=5
        )
        assert dataset.id_ == 1
        assert dataset.name == "production_dataset"
        assert dataset.catalog_tag_id == 5
        assert dataset.n_objects == 1000000

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            Dataset(id_=0, name="test", n_objects=100, catalog_tag_id=1)

        with pytest.raises(ValidationError):
            Dataset(id_=-1, name="test", n_objects=100, catalog_tag_id=1)

    def test_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        with pytest.raises(ValidationError):
            Dataset(id_=1, name="test", n_objects=100, catalog_tag_id=0)

        with pytest.raises(ValidationError):
            Dataset(id_=1, name="test", n_objects=100, catalog_tag_id=-1)

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 99
            name = "orm_dataset"
            path = "/data/orm.hdf5"
            n_objects = 50000
            is_collection = False
            catalog_tag_id = 3

        dataset = Dataset.model_validate(MockORMObject())
        assert dataset.id_ == 99
        assert dataset.name == "orm_dataset"
        assert dataset.catalog_tag_id == 3

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        expected = ["id_", "name", "n_objects", "catalog_tag_id", "path"]
        assert Dataset.col_names_for_table == expected

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        dataset = Dataset(id_=1, name="test", n_objects=100, catalog_tag_id=1)
        assert "col_names_for_table" not in dataset.model_dump()


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_dataset_to_dict(self):
        """Test converting Dataset to dict"""
        dataset = Dataset(
            id_=42,
            name="serialize_test",
            path="/data/test.hdf5",
            n_objects=7500,
            is_collection=True,
            catalog_tag_id=10,
        )
        data = dataset.model_dump()
        assert data["id_"] == 42
        assert data["name"] == "serialize_test"
        assert data["path"] == "/data/test.hdf5"
        assert data["n_objects"] == 7500
        assert data["is_collection"] is True
        assert data["catalog_tag_id"] == 10

    def test_dataset_with_none_path_serialization(self):
        """Test serialization when path is None"""
        dataset = Dataset(id_=1, name="no_path", path=None, n_objects=100, catalog_tag_id=1)
        data = dataset.model_dump()
        assert data["path"] is None

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = Dataset(id_=7, name="json_test", path="/data/json.hdf5", n_objects=12345, catalog_tag_id=2)
        json_str = original.model_dump_json()
        restored = Dataset.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.name == original.name
        assert restored.path == original.path
        assert restored.n_objects == original.n_objects
        assert restored.catalog_tag_id == original.catalog_tag_id

    def test_create_model_serialization(self):
        """Test serialization of DatasetCreate"""
        create = DatasetCreate(
            name="test_dataset", n_objects=500, catalog_tag_name="test_tag", validate_file=True
        )
        data = create.model_dump()
        assert data["catalog_tag_name"] == "test_tag"
        assert data["validate_file"] is True
