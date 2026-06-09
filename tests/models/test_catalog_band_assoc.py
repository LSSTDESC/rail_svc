"""Unit tests for CatalogBandAssoc Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.catalog_band_assoc import (CatalogBandAssoc,
                                                CatalogBandAssocBase,
                                                CatalogBandAssocCreate)


class TestCatalogBandAssocBase:
    """Tests for CatalogBandAssocBase model"""

    def test_valid_catalog_band_assoc_base(self):
        """Test creating valid CatalogBandAssocBase instance"""
        assoc = CatalogBandAssocBase(mag_column_name="g_mag", mag_err_column_name="g_mag_err")
        assert assoc.mag_column_name == "g_mag"
        assert assoc.mag_err_column_name == "g_mag_err"

    def test_missing_required_fields(self):
        """Test that missing required fields raises ValidationError"""
        with pytest.raises(ValidationError):
            CatalogBandAssocBase()

        with pytest.raises(ValidationError):
            CatalogBandAssocBase(mag_column_name="g_mag")

        with pytest.raises(ValidationError):
            CatalogBandAssocBase(mag_err_column_name="g_mag_err")

    def test_empty_string_column_names(self):
        """Test that empty strings are allowed for column names"""
        assoc = CatalogBandAssocBase(mag_column_name="", mag_err_column_name="")
        assert assoc.mag_column_name == ""
        assert assoc.mag_err_column_name == ""


class TestCatalogBandAssocCreate:
    """Tests for CatalogBandAssocCreate model"""

    def test_valid_catalog_band_assoc_create(self):
        """Test creating valid CatalogBandAssocCreate instance"""
        assoc = CatalogBandAssocCreate(
            mag_column_name="r_mag",
            mag_err_column_name="r_mag_err",
            catalog_tag_name="lsst_dp02",
            band_name="r_band",
        )
        assert assoc.mag_column_name == "r_mag"
        assert assoc.mag_err_column_name == "r_mag_err"
        assert assoc.catalog_tag_name == "lsst_dp02"
        assert assoc.band_name == "r_band"

    def test_missing_catalog_tag_name(self):
        """Test that missing catalog_tag_name raises ValidationError"""
        with pytest.raises(ValidationError):
            CatalogBandAssocCreate(
                mag_column_name="g_mag", mag_err_column_name="g_mag_err", band_name="g_band"
            )

    def test_missing_band_name(self):
        """Test that missing band_name raises ValidationError"""
        with pytest.raises(ValidationError):
            CatalogBandAssocCreate(
                mag_column_name="g_mag", mag_err_column_name="g_mag_err", catalog_tag_name="lsst_dp02"
            )

    def test_inherits_base_validation(self):
        """Test that CatalogBandAssocCreate inherits base validation"""
        with pytest.raises(ValidationError):
            CatalogBandAssocCreate(catalog_tag_name="test", band_name="test_band")


class TestCatalogBandAssoc:
    """Tests for CatalogBandAssoc model"""

    def test_valid_catalog_band_assoc(self):
        """Test creating valid CatalogBandAssoc instance"""
        assoc = CatalogBandAssoc(
            id_=1, catalog_tag_id=10, band_id=5, mag_column_name="i_mag", mag_err_column_name="i_mag_err"
        )
        assert assoc.id_ == 1
        assert assoc.catalog_tag_id == 10
        assert assoc.band_id == 5
        assert assoc.mag_column_name == "i_mag"

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            CatalogBandAssoc(
                id_=0, catalog_tag_id=1, band_id=1, mag_column_name="test", mag_err_column_name="test_err"
            )

        with pytest.raises(ValidationError):
            CatalogBandAssoc(
                id_=-1, catalog_tag_id=1, band_id=1, mag_column_name="test", mag_err_column_name="test_err"
            )

    def test_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        with pytest.raises(ValidationError):
            CatalogBandAssoc(
                id_=1, catalog_tag_id=0, band_id=1, mag_column_name="test", mag_err_column_name="test_err"
            )

    def test_band_id_must_be_positive(self):
        """Test that band_id must be greater than 0"""
        with pytest.raises(ValidationError):
            CatalogBandAssoc(
                id_=1, catalog_tag_id=1, band_id=0, mag_column_name="test", mag_err_column_name="test_err"
            )

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 99
            catalog_tag_id = 10
            band_id = 5
            mag_column_name = "z_mag"
            mag_err_column_name = "z_mag_err"

        assoc = CatalogBandAssoc.model_validate(MockORMObject())
        assert assoc.id_ == 99
        assert assoc.catalog_tag_id == 10
        assert assoc.band_id == 5

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        expected = ["id_", "catalog_tag_id", "band_id", "mag_column_name"]
        assert CatalogBandAssoc.col_names_for_table == expected

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        assoc = CatalogBandAssoc(
            id_=1, catalog_tag_id=1, band_id=1, mag_column_name="test", mag_err_column_name="test_err"
        )
        assert "col_names_for_table" not in assoc.model_dump()


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_catalog_band_assoc_to_dict(self):
        """Test converting CatalogBandAssoc to dict"""
        assoc = CatalogBandAssoc(
            id_=42, catalog_tag_id=100, band_id=50, mag_column_name="y_mag", mag_err_column_name="y_mag_err"
        )
        data = assoc.model_dump()
        assert data["id_"] == 42
        assert data["catalog_tag_id"] == 100
        assert data["band_id"] == 50
        assert data["mag_column_name"] == "y_mag"
        assert data["mag_err_column_name"] == "y_mag_err"

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = CatalogBandAssoc(
            id_=7, catalog_tag_id=15, band_id=8, mag_column_name="u_mag", mag_err_column_name="u_mag_err"
        )
        json_str = original.model_dump_json()
        restored = CatalogBandAssoc.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.catalog_tag_id == original.catalog_tag_id
        assert restored.band_id == original.band_id
        assert restored.mag_column_name == original.mag_column_name

    def test_create_model_serialization(self):
        """Test serialization of CatalogBandAssocCreate"""
        create = CatalogBandAssocCreate(
            mag_column_name="test_mag",
            mag_err_column_name="test_mag_err",
            catalog_tag_name="test_catalog",
            band_name="test_band",
        )
        data = create.model_dump()
        assert "catalog_tag_name" in data
        assert "band_name" in data
        assert data["catalog_tag_name"] == "test_catalog"
        assert data["band_name"] == "test_band"
