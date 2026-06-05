"""Unit tests for CatalogTag Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.catalog_tag import CatalogTag, CatalogTagBase, CatalogTagCreate


class TestCatalogTagBase:
    """Tests for CatalogTagBase model"""

    def test_valid_catalog_tag_base(self):
        """Test creating valid CatalogTagBase instance"""
        tag = CatalogTagBase(name="lsst_dp02")
        assert tag.name == "lsst_dp02"

    def test_missing_name_raises_error(self):
        """Test that missing name raises ValidationError"""
        with pytest.raises(ValidationError):
            CatalogTagBase()

    def test_empty_string_name(self):
        """Test that empty string is allowed for name"""
        tag = CatalogTagBase(name="")
        assert tag.name == ""


class TestCatalogTagCreate:
    """Tests for CatalogTagCreate model"""

    def test_valid_catalog_tag_create(self):
        """Test creating valid CatalogTagCreate instance"""
        tag = CatalogTagCreate(name="hsc_pdr3")
        assert tag.name == "hsc_pdr3"

    def test_inherits_base_validation(self):
        """Test that CatalogTagCreate inherits base validation"""
        with pytest.raises(ValidationError):
            CatalogTagCreate()


class TestCatalogTag:
    """Tests for CatalogTag model"""

    def test_valid_catalog_tag(self):
        """Test creating valid CatalogTag instance"""
        tag = CatalogTag(id_=1, name="des_y6")
        assert tag.id_ == 1
        assert tag.name == "des_y6"

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            CatalogTag(id_=0, name="test")

        with pytest.raises(ValidationError):
            CatalogTag(id_=-1, name="test")

    def test_minimum_valid_id(self):
        """Test minimum valid id_ value"""
        tag = CatalogTag(id_=1, name="test")
        assert tag.id_ == 1

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 42
            name = "orm_catalog"

        tag = CatalogTag.model_validate(MockORMObject())
        assert tag.id_ == 42
        assert tag.name == "orm_catalog"

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        assert CatalogTag.col_names_for_table == ["id_", "name"]

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        tag = CatalogTag(id_=1, name="test")
        assert "col_names_for_table" not in tag.model_dump()


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_catalog_tag_to_dict(self):
        """Test converting CatalogTag to dict"""
        tag = CatalogTag(id_=99, name="sdss_dr17")
        data = tag.model_dump()
        assert data == {"id_": 99, "name": "sdss_dr17"}

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = CatalogTag(id_=7, name="euclid_dr1")
        json_str = original.model_dump_json()
        restored = CatalogTag.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.name == original.name

    def test_base_model_serialization(self):
        """Test serialization of CatalogTagBase"""
        base = CatalogTagBase(name="test_catalog")
        data = base.model_dump()
        assert data == {"name": "test_catalog"}
