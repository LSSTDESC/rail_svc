"""Unit tests for the CatalogBandAssoc Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.catalog_band_assoc import (
    CatalogBandAssoc,
    CatalogBandAssocBase,
    CatalogBandAssocCreate,
)


class TestCatalogBandAssocBase:
    """Tests for CatalogBandAssocBase model"""

    def test_valid_catalog_band_assoc_base(self):
        """Test creating a valid CatalogBandAssocBase"""
        assoc = CatalogBandAssocBase(limiting_mag=24.5)
        assert assoc.limiting_mag == 24.5

    def test_catalog_band_assoc_base_integer_magnitude(self):
        """Test that integer magnitudes are accepted and converted to float"""
        assoc = CatalogBandAssocBase(limiting_mag=25)
        assert assoc.limiting_mag == 25
        assert isinstance(assoc.limiting_mag, (int, float))

    def test_catalog_band_assoc_base_realistic_magnitudes(self):
        """Test with realistic limiting magnitude values"""
        realistic_mags = [21.0, 23.5, 24.8, 26.2, 27.5]
        
        for mag in realistic_mags:
            assoc = CatalogBandAssocBase(limiting_mag=mag)
            assert assoc.limiting_mag == mag

    def test_catalog_band_assoc_base_missing_limiting_mag(self):
        """Test that limiting_mag is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocBase()
        assert "limiting_mag" in str(exc_info.value)

    def test_catalog_band_assoc_base_must_be_positive(self):
        """Test that limiting_mag must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocBase(limiting_mag=0)
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocBase(limiting_mag=-5.0)
        assert "greater than 0" in str(exc_info.value)

    def test_catalog_band_assoc_base_invalid_type(self):
        """Test that limiting_mag must be numeric"""
        with pytest.raises(ValidationError):
            CatalogBandAssocBase(limiting_mag={})

        with pytest.raises(ValidationError):
            CatalogBandAssocBase(limiting_mag=None)

    def test_catalog_band_assoc_base_edge_values(self):
        """Test edge case magnitude values"""
        # Very bright limiting magnitude (unlikely but valid)
        bright = CatalogBandAssocBase(limiting_mag=15.0)
        assert bright.limiting_mag == 15.0

        # Very faint limiting magnitude
        faint = CatalogBandAssocBase(limiting_mag=30.0)
        assert faint.limiting_mag == 30.0

        # Very small positive value
        tiny = CatalogBandAssocBase(limiting_mag=0.001)
        assert tiny.limiting_mag == 0.001


class TestCatalogBandAssocCreate:
    """Tests for CatalogBandAssocCreate model"""

    def test_valid_catalog_band_assoc_create(self):
        """Test creating a valid CatalogBandAssocCreate"""
        assoc = CatalogBandAssocCreate(
            limiting_mag=24.7,
            catalog_tag_name="lsst_dp02",
            band_name="g_band",
        )
        assert assoc.limiting_mag == 24.7
        assert assoc.catalog_tag_name == "lsst_dp02"
        assert assoc.band_name == "g_band"

    def test_catalog_band_assoc_create_missing_catalog_tag_name(self):
        """Test that catalog_tag_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocCreate(
                limiting_mag=24.5,
                band_name="r_band",
            )
        assert "catalog_tag_name" in str(exc_info.value)

    def test_catalog_band_assoc_create_missing_band_name(self):
        """Test that band_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocCreate(
                limiting_mag=24.5,
                catalog_tag_name="lsst_dp02",
            )
        assert "band_name" in str(exc_info.value)

    def test_catalog_band_assoc_create_inherits_validation(self):
        """Test that CatalogBandAssocCreate inherits validation from CatalogBandAssocBase"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocCreate(
                limiting_mag=0,
                catalog_tag_name="lsst",
                band_name="g",
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocCreate(
                limiting_mag=-10.0,
                catalog_tag_name="lsst",
                band_name="g",
            )
        assert "greater than 0" in str(exc_info.value)

    def test_catalog_band_assoc_create_realistic_lsst_bands(self):
        """Test with realistic LSST catalog and band combinations"""
        lsst_bands = [
            ("lsst_dp02", "u_band", 23.9),
            ("lsst_dp02", "g_band", 25.0),
            ("lsst_dp02", "r_band", 24.7),
            ("lsst_dp02", "i_band", 24.0),
            ("lsst_dp02", "z_band", 23.3),
            ("lsst_dp02", "y_band", 22.1),
        ]
        
        for catalog, band, lim_mag in lsst_bands:
            assoc = CatalogBandAssocCreate(
                limiting_mag=lim_mag,
                catalog_tag_name=catalog,
                band_name=band,
            )
            assert assoc.catalog_tag_name == catalog
            assert assoc.band_name == band
            assert assoc.limiting_mag == lim_mag


class TestCatalogBandAssoc:
    """Tests for CatalogBandAssoc model"""

    def test_valid_catalog_band_assoc(self):
        """Test creating a valid CatalogBandAssoc with all fields"""
        assoc = CatalogBandAssoc(
            id=1,
            limiting_mag=24.5,
            catalog_tag_id=5,
            band_id=3,
        )
        assert assoc.id == 1
        assert assoc.limiting_mag == 24.5
        assert assoc.catalog_tag_id == 5
        assert assoc.band_id == 3

    def test_catalog_band_assoc_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=0,
                limiting_mag=24.5,
                catalog_tag_id=1,
                band_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=-1,
                limiting_mag=24.5,
                catalog_tag_id=1,
                band_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_catalog_band_assoc_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=24.5,
                catalog_tag_id=0,
                band_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=24.5,
                catalog_tag_id=-5,
                band_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_catalog_band_assoc_band_id_must_be_positive(self):
        """Test that band_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=24.5,
                catalog_tag_id=1,
                band_id=0,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=24.5,
                catalog_tag_id=1,
                band_id=-3,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_catalog_band_assoc_missing_id(self):
        """Test that id is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                limiting_mag=24.5,
                catalog_tag_id=1,
                band_id=1,
            )
        assert "id" in str(exc_info.value)

    def test_catalog_band_assoc_missing_catalog_tag_id(self):
        """Test that catalog_tag_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=24.5,
                band_id=1,
            )
        assert "catalog_tag_id" in str(exc_info.value)

    def test_catalog_band_assoc_missing_band_id(self):
        """Test that band_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=24.5,
                catalog_tag_id=1,
            )
        assert "band_id" in str(exc_info.value)

    def test_catalog_band_assoc_inherits_limiting_mag_validation(self):
        """Test that CatalogBandAssoc inherits limiting_mag validation"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=0,
                catalog_tag_id=1,
                band_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(
                id=1,
                limiting_mag=-5.0,
                catalog_tag_id=1,
                band_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_catalog_band_assoc_from_attributes(self):
        """Test that from_attributes config works"""
        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 10
            limiting_mag = 25.2
            catalog_tag_id = 3
            band_id = 7

        orm_obj = MockORMObject()
        assoc = CatalogBandAssoc.model_validate(orm_obj)
        assert assoc.id == 10
        assert assoc.limiting_mag == 25.2
        assert assoc.catalog_tag_id == 3
        assert assoc.band_id == 7

    def test_catalog_band_assoc_col_names_for_table(self):
        """Test that col_names_for_table class variable is set correctly"""
        expected_cols = ["id", "catalog_tag_id", "band_id", "limiting_mag"]
        assert CatalogBandAssoc.col_names_for_table == expected_cols

    def test_catalog_band_assoc_field_descriptions(self):
        """Test that field descriptions are set"""
        schema = CatalogBandAssoc.model_json_schema()
        assert "Five sigma limiting magnitude" in schema["properties"]["limiting_mag"]["description"]
        assert "Foreign key referencing CatalogTag.id" in schema["properties"]["catalog_tag_id"]["description"]
        assert "Foreign key referencing Band.id" in schema["properties"]["band_id"]["description"]

    def test_catalog_band_assoc_realistic_survey_data(self):
        """Test with realistic survey catalog/band associations"""
        # Simulate LSST DP0.2 limiting magnitudes
        lsst_dp02_bands = [
            (1, 1, 23.9),  # u band
            (1, 2, 25.0),  # g band
            (1, 3, 24.7),  # r band
            (1, 4, 24.0),  # i band
            (1, 5, 23.3),  # z band
            (1, 6, 22.1),  # y band
        ]
        
        for idx, (cat_id, band_id, lim_mag) in enumerate(lsst_dp02_bands, start=1):
            assoc = CatalogBandAssoc(
                id=idx,
                catalog_tag_id=cat_id,
                band_id=band_id,
                limiting_mag=lim_mag,
            )
            assert assoc.catalog_tag_id == cat_id
            assert assoc.band_id == band_id
            assert assoc.limiting_mag == lim_mag

    def test_catalog_band_assoc_multiple_catalogs_same_band(self):
        """Test same band in different catalogs with different limiting mags"""
        # g-band in different surveys
        g_band_id = 2
        
        surveys = [
            (1, "LSST", 25.0),
            (2, "HSC", 26.5),
            (3, "DES", 24.3),
            (4, "SDSS", 23.0),
        ]
        
        for idx, (cat_id, name, lim_mag) in enumerate(surveys, start=1):
            assoc = CatalogBandAssoc(
                id=idx,
                catalog_tag_id=cat_id,
                band_id=g_band_id,
                limiting_mag=lim_mag,
            )
            assert assoc.band_id == g_band_id
            assert assoc.limiting_mag == lim_mag

    def test_catalog_band_assoc_json_serialization(self):
        """Test that CatalogBandAssoc can be serialized to/from JSON"""
        original = CatalogBandAssoc(
            id=15,
            limiting_mag=24.8,
            catalog_tag_id=3,
            band_id=5,
        )
        
        # Serialize to JSON
        json_str = original.model_dump_json()
        
        # Deserialize from JSON
        restored = CatalogBandAssoc.model_validate_json(json_str)
        
        assert restored.id == original.id
        assert restored.limiting_mag == original.limiting_mag
        assert restored.catalog_tag_id == original.catalog_tag_id
        assert restored.band_id == original.band_id

    def test_catalog_band_assoc_floating_point_precision(self):
        """Test that floating point limiting magnitudes preserve precision"""
        precise_mag = 24
