"""Unit tests for Band Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.band import Band, BandBase, BandCreate


class TestBandBase:
    """Tests for BandBase model"""

    def test_valid_band_base(self):
        """Test creating valid BandBase instance"""
        band = BandBase(
            name="test_band",
            band_wavelengths=[400.0, 500.0, 600.0],
            band_transmission=[0.1, 0.9, 0.2]
        )
        assert band.name == "test_band"
        assert band.band_wavelengths == [400.0, 500.0, 600.0]
        assert band.band_transmission == [0.1, 0.9, 0.2]

    def test_empty_wavelengths_raises_error(self):
        """Test that empty wavelengths array raises ValidationError"""
        with pytest.raises(ValidationError, match="Array must not be empty"):
            BandBase(
                name="test",
                band_wavelengths=[],
                band_transmission=[0.5]
            )

    def test_empty_transmission_raises_error(self):
        """Test that empty transmission array raises ValidationError"""
        with pytest.raises(ValidationError, match="Array must not be empty"):
            BandBase(
                name="test",
                band_wavelengths=[500.0],
                band_transmission=[]
            )

    def test_mismatched_array_lengths_raises_error(self):
        """Test that mismatched array lengths raise ValidationError"""
        with pytest.raises(ValidationError, match="must have same length"):
            BandBase(
                name="test",
                band_wavelengths=[400.0, 500.0, 600.0],
                band_transmission=[0.1, 0.9]
            )

    def test_single_element_arrays(self):
        """Test that single-element arrays are valid"""
        band = BandBase(
            name="test",
            band_wavelengths=[500.0],
            band_transmission=[0.8]
        )
        assert len(band.band_wavelengths) == 1
        assert len(band.band_transmission) == 1

    def test_missing_required_fields(self):
        """Test that missing required fields raises ValidationError"""
        with pytest.raises(ValidationError):
            BandBase()


class TestBandCreate:
    """Tests for BandCreate model"""

    def test_band_create_inherits_validation(self):
        """Test that BandCreate inherits validation from BandBase"""
        band = BandCreate(
            name="create_test",
            band_wavelengths=[450.0, 550.0],
            band_transmission=[0.3, 0.7]
        )
        assert band.name == "create_test"
        
        with pytest.raises(ValidationError):
            BandCreate(
                name="test",
                band_wavelengths=[],
                band_transmission=[0.5]
            )


class TestBand:
    """Tests for Band model"""

    def test_valid_band(self):
        """Test creating valid Band instance"""
        band = Band(
            id_=1,
            name="g_band",
            band_wavelengths=[400.0, 500.0, 600.0],
            band_transmission=[0.1, 0.95, 0.15]
        )
        assert band.id_ == 1
        assert band.name == "g_band"

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            Band(
                id_=0,
                name="test",
                band_wavelengths=[500.0],
                band_transmission=[0.5]
            )

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""
        class MockORMObject:
            id_ = 5
            name = "orm_band"
            band_wavelengths = [450.0, 550.0]
            band_transmission = [0.4, 0.6]
        
        band = Band.model_validate(MockORMObject())
        assert band.id_ == 5
        assert band.name == "orm_band"

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        assert Band.col_names_for_table == ["id_", "name"]

    def test_col_names_not_instance_attribute(self):
        """Test that col_names_for_table is not an instance attribute"""
        band = Band(
            id_=1,
            name="test",
            band_wavelengths=[500.0],
            band_transmission=[0.5]
        )
        assert "col_names_for_table" not in band.model_dump()


class TestArrayValidation:
    """Specific tests for array validation edge cases"""

    @pytest.mark.parametrize("length", [1, 2, 10, 100])
    def test_various_matching_lengths(self, length):
        """Test arrays of various matching lengths"""
        wavelengths = [float(i) for i in range(length)]
        transmission = [0.5] * length
        band = BandBase(
            name="test",
            band_wavelengths=wavelengths,
            band_transmission=transmission
        )
        assert len(band.band_wavelengths) == length
        assert len(band.band_transmission) == length

    def test_wavelengths_can_be_any_float(self):
        """Test that wavelengths accept any float values"""
        band = BandBase(
            name="test",
            band_wavelengths=[0.1, 1e6, -100.0],
            band_transmission=[0.0, 0.5, 1.0]
        )
        assert band.band_wavelengths == [0.1, 1e6, -100.0]

    def test_transmission_can_be_any_float(self):
        """Test that transmission accepts any float values"""
        band = BandBase(
            name="test",
            band_wavelengths=[500.0, 600.0],
            band_transmission=[-0.1, 2.0]  # Values outside [0,1] are allowed
        )
        assert band.band_transmission == [-0.1, 2.0]


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_band_to_dict(self):
        """Test converting Band to dict"""
        band = Band(
            id_=42,
            name="r_band",
            band_wavelengths=[600.0, 700.0],
            band_transmission=[0.2, 0.8]
        )
        data = band.model_dump()
        assert data["id_"] == 42
        assert data["name"] == "r_band"
        assert data["band_wavelengths"] == [600.0, 700.0]
        assert data["band_transmission"] == [0.2, 0.8]

    def test_json_serialization(self):
        """Test JSON serialization round-trip"""
        original = Band(
            id_=7,
            name="i_band",
            band_wavelengths=[700.0, 800.0, 900.0],
            band_transmission=[0.1, 0.9, 0.3]
        )
        json_str = original.model_dump_json()
        restored = Band.model_validate_json(json_str)
        assert restored.id_ == original.id_
        assert restored.name == original.name
        assert restored.band_wavelengths == original.band_wavelengths
        assert restored.band_transmission == original.band_transmission


class TestInheritanceRelationships:
    """Tests for model inheritance structure"""

    def test_band_create_is_subclass_of_base(self):
        """Test that BandCreate inherits from BandBase"""
        assert issubclass(BandCreate, BandBase)

    def test_band_is_subclass_of_base(self):
        """Test that Band inherits from BandBase"""
        assert issubclass(Band, BandBase)

    def test_band_has_additional_id_field(self):
        """Test that Band adds id_ field"""
        base_fields = set(BandBase.model_fields.keys())
        band_fields = set(Band.model_fields.keys())
        assert band_fields == base_fields | {"id_"}


class TestFieldDescriptions:
    """Tests for field descriptions and metadata"""

    def test_name_field_description(self):
        """Test that name field has correct description"""
        field_info = BandBase.model_fields["name"]
        assert field_info.description == "Unique name for this band"

    def test_wavelengths_field_description(self):
        """Test that band_wavelengths field has correct description"""
        field_info = BandBase.model_fields["band_wavelengths"]
        assert field_info.description == "Wavelengths for band transmission grid"

    def test_transmission_field_description(self):
        """Test that band_transmission field has correct description"""
        field_info = BandBase.model_fields["band_transmission"]
        assert field_info.description == "Transmission at given wavelengths"
