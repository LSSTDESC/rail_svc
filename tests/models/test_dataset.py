"""Unit tests for the Dataset Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_pz_service.models.dataset import Dataset, DatasetBase, DatasetCreate


class TestDatasetBase:
    """Tests for DatasetBase model"""

    def test_valid_dataset_base(self):
        """Test creating a valid DatasetBase"""
        dataset = DatasetBase(
            name="test_dataset",
            path="/path/to/data.parquet",
            n_objects=1000,
        )
        assert dataset.name == "test_dataset"
        assert dataset.path == "/path/to/data.parquet"
        assert dataset.n_objects == 1000
        assert dataset.data is None

    def test_dataset_base_with_data(self):
        """Test creating DatasetBase with data dict"""
        data_dict = {"mag_r": [20.1, 20.5], "mag_i": [19.8, 20.2]}
        dataset = DatasetBase(
            name="small_dataset",
            data=data_dict,
            n_objects=2,
        )
        assert dataset.data == data_dict
        assert dataset.path is None

    def test_dataset_base_minimal(self):
        """Test creating DatasetBase with minimal fields"""
        dataset = DatasetBase(name="minimal", n_objects=None)
        assert dataset.name == "minimal"
        assert dataset.path is None
        assert dataset.data is None
        assert dataset.n_objects is None

    def test_dataset_base_missing_name(self):
        """Test that name is required"""
        with pytest.raises(ValidationError) as exc_info:
            DatasetBase(n_objects=100)
        assert "name" in str(exc_info.value)


class TestDatasetCreate:
    """Tests for DatasetCreate model"""

    def test_valid_dataset_create(self):
        """Test creating a valid DatasetCreate"""
        dataset = DatasetCreate(
            name="new_dataset",
            path="/data/catalog.parquet",
            n_objects=5000,
            catalog_tag_name="lsst_dp02",
            validate_file=True,
        )
        assert dataset.name == "new_dataset"
        assert dataset.catalog_tag_name == "lsst_dp02"
        assert dataset.validate_file is True

    def test_dataset_create_default_validate(self):
        """Test that validate_file defaults to False"""
        dataset = DatasetCreate(
            name="test",
            n_objects=10,
            catalog_tag_name="lsst",
        )
        assert dataset.validate_file is False

    def test_dataset_create_missing_catalog_tag(self):
        """Test that catalog_tag_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            DatasetCreate(name="test", n_objects=10)
        assert "catalog_tag_name" in str(exc_info.value)


class TestDataset:
    """Tests for Dataset model"""

    def test_valid_dataset(self):
        """Test creating a valid Dataset with all fields"""
        dataset = Dataset(
            id=1,
            name="full_dataset",
            path="/data/full.parquet",
            n_objects=10000,
            catalog_tag_id=5,
        )
        assert dataset.id == 1
        assert dataset.name == "full_dataset"
        assert dataset.catalog_tag_id == 5

    def test_dataset_with_data_dict(self):
        """Test Dataset with inline data"""
        data = {"mag_g": [21.0], "mag_r": [20.5]}
        dataset = Dataset(
            id=2,
            name="inline_data",
            data=data,
            n_objects=1,
            catalog_tag_id=3,
        )
        assert dataset.data == data
        assert dataset.path is None

    def test_dataset_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Dataset(
                id=0,
                name="test",
                n_objects=10,
                catalog_tag_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Dataset(
                id=-1,
                name="test",
                n_objects=10,
                catalog_tag_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_dataset_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Dataset(
                id=1,
                name="test",
                n_objects=10,
                catalog_tag_id=0,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_dataset_from_attributes(self):
        """Test that from_attributes config works"""
        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 10
            name = "orm_dataset"
            path = "/orm/path.parquet"
            data = None
            n_objects = 500
            catalog_tag_id = 2

        orm_obj = MockORMObject()
        dataset = Dataset.model_validate(orm_obj)
        assert dataset.id == 10
        assert dataset.name == "orm_dataset"
        assert dataset.catalog_tag_id == 2

    def test_dataset_col_names_for_table(self):
        """Test that col_names_for_table class variable is set correctly"""
        expected_cols = ["id", "name", "n_objects", "catalog_tag_id", "path"]
        assert Dataset.col_names_for_table == expected_cols
