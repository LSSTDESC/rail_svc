"""Unit tests for DatasetAssoc Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.dataset_assoc import DatasetAssoc, DatasetAssocBase, DatasetAssocCreate


class TestDatasetAssocBase:
    """Tests for DatasetAssocBase model"""

    def test_valid_dataset_assoc_base(self):
        """Test creating valid DatasetAssocBase instance"""
        assoc = DatasetAssocBase(name="test_association")
        assert assoc.name == "test_association"

    def test_missing_name_raises_error(self):
        """Test that missing name raises ValidationError"""
        with pytest.raises(ValidationError):
            DatasetAssocBase()

    def test_empty_string_name(self):
        """Test that empty string is allowed for name"""
        assoc = DatasetAssocBase(name="")
        assert assoc.name == ""


class TestDatasetAssocCreate:
    """Tests for DatasetAssocCreate model"""

    def test_valid_dataset_assoc_create(self):
        """Test creating valid DatasetAssocCreate instance"""
        assoc = DatasetAssocCreate(
            name="photo_spec_match",
            matched_dataset_name="photometric_catalog",
            component_dataset_name="spectroscopic_catalog",
        )
        assert assoc.name == "photo_spec_match"
        assert assoc.matched_dataset_name == "photometric_catalog"
        assert assoc.component_dataset_name == "spectroscopic_catalog"

    def test_missing_matched_dataset_name(self):
        """Test that missing matched_dataset_name raises ValidationError"""
        with pytest.raises(ValidationError):
            DatasetAssocCreate(name="test", component_dataset_name="component")

    def test_missing_component_dataset_name(self):
        """Test that missing component_dataset_name raises ValidationError"""
        with pytest.raises(ValidationError):
            DatasetAssocCreate(name="test", matched_dataset_name="matched")

    def test_inherits_base_validation(self):
        """Test that DatasetAssocCreate inherits base validation"""
        with pytest.raises(ValidationError):
            DatasetAssocCreate(matched_dataset_name="matched", component_dataset_name="component")


class TestDatasetAssoc:
    """Tests for DatasetAssoc model"""

    def test_valid_dataset_assoc(self):
        """Test creating valid DatasetAssoc instance"""
        assoc = DatasetAssoc(id_=1, name="main_association", matched_dataset_id=10, component_dataset_id=20)
        assert assoc.id_ == 1
        assert assoc.name == "main_association"
        assert assoc.matched_dataset_id == 10
        assert assoc.component_dataset_id == 20

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            DatasetAssoc(id_=0, name="test", matched_dataset_id=1, component_dataset_id=1)

        with pytest.raises(ValidationError):
            DatasetAssoc(id_=-1, name="test", matched_dataset_id=1, component_dataset_id=1)

    def test_matched_dataset_id_must_be_positive(self):
        """Test that matched_dataset_id must be greater than 0"""
        with pytest.raises(ValidationError):
            DatasetAssoc(id_=1, name="test", matched_dataset_id=0, component_dataset_id=1)

        with pytest.raises(ValidationError):
            DatasetAssoc(id_=1, name="test", matched_dataset_id=-1, component_dataset_id=1)

    def test_component_dataset_id_must_be_positive(self):
        """Test that component_dataset_id must be greater than 0"""
        with pytest.raises(ValidationError):
            DatasetAssoc(id_=1, name="test", matched_dataset_id=1, component_dataset_id=0)

        with pytest.raises(ValidationError):
            DatasetAssoc(id_=1, name="test", matched_dataset_id=1, component_dataset_id=-1)

    def test_same_matched_and_component_ids_allowed(self):
        """Test that same dataset can be both matched and component"""
        assoc = DatasetAssoc(id_=1, name="self_match", matched_dataset_id=5, component_dataset_id=5)
        assert assoc.matched_dataset_id == assoc.component_dataset_id

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 99
            name = "orm_association"
            matched_dataset_id = 15
            component_dataset_id = 25

        assoc = DatasetAssoc.model_validate(MockORMObject())
        assert assoc.id_ == 99
        assert assoc.name == "orm_association"
        assert assoc.matched_dataset_id == 15
        assert assoc.component_dataset_id == 25

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        expected = [
            "id_",
            "name",
            "n_objects",
            "matched_dataset_id",
            "component_dataset_id",
            "path",
        ]
        assert DatasetAssoc.col_names_for_table == expected

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        assoc = DatasetAssoc(id_=1, name="test", matched_dataset_id=1, component_dataset_id=1)
        assert "col_names_for_table" not in assoc.model_dump()


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_dataset_assoc_to_dict(self):
        """Test converting DatasetAssoc to dict"""
        assoc = DatasetAssoc(id_=42, name="serialize_test", matched_dataset_id=100, component_dataset_id=200)
        data = assoc.model_dump()
        assert data["id_"] == 42
        assert data["name"] == "serialize_test"
        assert data["matched_dataset_id"] == 100
        assert data["component_dataset_id"] == 200

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = DatasetAssoc(id_=7, name="json_test", matched_dataset_id=50, component_dataset_id=60)
        json_str = original.model_dump_json()
        restored = DatasetAssoc.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.name == original.name
        assert restored.matched_dataset_id == original.matched_dataset_id
        assert restored.component_dataset_id == original.component_dataset_id

    def test_create_model_serialization(self):
        """Test serialization of DatasetAssocCreate"""
        create = DatasetAssocCreate(
            name="test_assoc", matched_dataset_name="matched_data", component_dataset_name="component_data"
        )
        data = create.model_dump()
        assert data["name"] == "test_assoc"
        assert data["matched_dataset_name"] == "matched_data"
        assert data["component_dataset_name"] == "component_data"
