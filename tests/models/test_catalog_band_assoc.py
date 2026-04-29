"""Unit tests for CatalogBandAssoc Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.catalog_band_assoc import CatalogBandAssoc, CatalogBandAssocBase, CatalogBandAssocCreate


class TestCatalogBandAssocBase:
    """Tests for CatalogBandAssocBase model"""

    def test_valid_creation(self):
        """Test creating a valid CatalogBandAssocBase instance"""
        data = {"band_alias": "g_band"}
        model = CatalogBandAssocBase(**data)
        assert model.band_alias == "g_band"

    def test_missing_required_field(self):
        """Test that missing band_alias raises ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocBase()

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["loc"] == ("band_alias",)
        assert errors[0]["type"] == "missing"

    def test_empty_string_allowed(self):
        """Test that empty string is allowed for band_alias"""
        data = {"band_alias": ""}
        model = CatalogBandAssocBase(**data)
        assert model.band_alias == ""

    def test_field_description(self):
        """Test that field has correct description"""
        field_info = CatalogBandAssocBase.model_fields["band_alias"]
        assert field_info.description == "Name given to Band in the Catalog"


class TestCatalogBandAssocCreate:
    """Tests for CatalogBandAssocCreate model"""

    def test_valid_creation(self):
        """Test creating a valid CatalogBandAssocCreate instance"""
        data = {
            "band_alias": "g_band",
            "catalog_tag_name": "LSST_DP0",
            "band_name": "LSST_g"
        }
        model = CatalogBandAssocCreate(**data)
        assert model.band_alias == "g_band"
        assert model.catalog_tag_name == "LSST_DP0"
        assert model.band_name == "LSST_g"

    def test_inherits_from_base(self):
        """Test that CatalogBandAssocCreate inherits from CatalogBandAssocBase"""
        assert issubclass(CatalogBandAssocCreate, CatalogBandAssocBase)

    def test_missing_catalog_tag_name(self):
        """Test that missing catalog_tag_name raises ValidationError"""
        data = {
            "band_alias": "g_band",
            "band_name": "LSST_g"
        }
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocCreate(**data)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("catalog_tag_name",) for e in errors)

    def test_missing_band_name(self):
        """Test that missing band_name raises ValidationError"""
        data = {
            "band_alias": "g_band",
            "catalog_tag_name": "LSST_DP0"
        }
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocCreate(**data)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("band_name",) for e in errors)

    def test_all_fields_missing(self):
        """Test that all missing fields raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssocCreate()

        errors = exc_info.value.errors()
        assert len(errors) == 3
        field_names = {e["loc"][0] for e in errors}
        assert field_names == {"band_alias", "catalog_tag_name", "band_name"}

    def test_field_descriptions(self):
        """Test that fields have correct descriptions"""
        fields = CatalogBandAssocCreate.model_fields
        assert fields["catalog_tag_name"].description == "Name of the associated CatalogTag"
        assert fields["band_name"].description == "Name of the associated Band"


class TestCatalogBandAssoc:
    """Tests for CatalogBandAssoc model"""

    def test_valid_creation(self):
        """Test creating a valid CatalogBandAssoc instance"""
        data = {
            "band_alias": "g_band",
            "id": 1,
            "catalog_tag_id": 10,
            "band_id": 5
        }
        model = CatalogBandAssoc(**data)
        assert model.band_alias == "g_band"
        assert model.id == 1
        assert model.catalog_tag_id == 10
        assert model.band_id == 5

    def test_inherits_from_base(self):
        """Test that CatalogBandAssoc inherits from CatalogBandAssocBase"""
        assert issubclass(CatalogBandAssoc, CatalogBandAssocBase)

    def test_from_attributes_config(self):
        """Test that model has from_attributes configuration"""
        assert CatalogBandAssoc.model_config["from_attributes"] is True

    def test_col_names_for_table_class_var(self):
        """Test that col_names_for_table ClassVar is correctly defined"""
        expected_cols = ["id", "catalog_tag_id", "band_id", "band_alias"]
        assert CatalogBandAssoc.col_names_for_table == expected_cols

    def test_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        data = {
            "band_alias": "g_band",
            "id": 0,
            "catalog_tag_id": 10,
            "band_id": 5
        }
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(**data)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("id",) and e["type"] == "greater_than"
            for e in errors
        )

    def test_id_negative_not_allowed(self):
        """Test that negative id is not allowed"""
        data = {
            "band_alias": "g_band",
            "id": -1,
            "catalog_tag_id": 10,
            "band_id": 5
        }
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(**data)

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("id",) for e in errors)

    def test_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        data = {
            "band_alias": "g_band",
            "id": 1,
            "catalog_tag_id": 0,
            "band_id": 5
        }
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(**data)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("catalog_tag_id",) and e["type"] == "greater_than"
            for e in errors
        )

    def test_band_id_must_be_positive(self):
        """Test that band_id must be greater than 0"""
        data = {
            "band_alias": "g_band",
            "id": 1,
            "catalog_tag_id": 10,
            "band_id": 0
        }
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(**data)

        errors = exc_info.value.errors()
        assert any(
            e["loc"] == ("band_id",) and e["type"] == "greater_than"
            for e in errors
        )

    def test_field_descriptions(self):
        """Test that fields have correct descriptions"""
        fields = CatalogBandAssoc.model_fields
        assert fields["catalog_tag_id"].description == "Foreign key referencing CatalogTag.id"
        assert fields["band_id"].description == "Foreign key referencing Band.id"

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc()

        errors = exc_info.value.errors()
        assert len(errors) == 4
        field_names = {e["loc"][0] for e in errors}
        assert field_names == {"band_alias", "id", "catalog_tag_id", "band_id"}

    @pytest.mark.parametrize("field_name,invalid_value", [
        ("id", 0),
        ("id", -5),
        ("catalog_tag_id", 0),
        ("catalog_tag_id", -10),
        ("band_id", 0),
        ("band_id", -3),
    ])
    def test_invalid_integer_fields(self, field_name, invalid_value):
        """Test various invalid values for integer fields"""
        data = {
            "band_alias": "g_band",
            "id": 1,
            "catalog_tag_id": 10,
            "band_id": 5,
            field_name: invalid_value
        }
        with pytest.raises(ValidationError) as exc_info:
            CatalogBandAssoc(**data)

        errors = exc_info.value.errors()
        assert any(e["loc"] == (field_name,) for e in errors)

    def test_model_dump(self):
        """Test that model can be dumped to dict"""
        data = {
            "band_alias": "g_band",
            "id": 1,
            "catalog_tag_id": 10,
            "band_id": 5
        }
        model = CatalogBandAssoc(**data)
        dumped = model.model_dump()

        assert dumped["band_alias"] == "g_band"
        assert dumped["id"] == 1
        assert dumped["catalog_tag_id"] == 10
        assert dumped["band_id"] == 5

    def test_model_json_serialization(self):
        """Test that model can be serialized to JSON"""
        data = {
            "band_alias": "g_band",
            "id": 1,
            "catalog_tag_id": 10,
            "band_id": 5
        }
        model = CatalogBandAssoc(**data)
        json_str = model.model_dump_json()

        assert '"band_alias":"g_band"' in json_str
        assert '"id":1' in json_str
        assert '"catalog_tag_id":10' in json_str
        assert '"band_id":5' in json_str
