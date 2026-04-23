"""Unit tests for the Band Pydantic models"""

import pytest

import numpy as np

from pydantic import ValidationError

from rail_svc.models.band import Band, BandBase, BandCreate


class TestBandBase:
    """Tests for BandBase model"""

    def test_valid_band_base(self):
        """Test creating a valid BandBase"""
        wavelengths = [400.0, 450.0, 500.0, 550.0, 600.0]
        transmission = [0.1, 0.5, 0.9, 0.5, 0.1]
        
        band = BandBase(
            name="g_band",
            band_wavelengths=wavelengths,
            band_transmission=transmission,
        )
        assert band.name == "g_band"
        assert band.band_wavelengths == wavelengths
        assert band.band_transmission == transmission

    def test_band_base_single_point(self):
        """Test creating BandBase with single wavelength point"""
        band = BandBase(
            name="monochromatic",
            band_wavelengths=[500.0],
            band_transmission=[1.0],
        )
        assert len(band.band_wavelengths) == 1
        assert len(band.band_transmission) == 1

    def test_band_base_many_points(self):
        """Test creating BandBase with many wavelength points"""
        wavelengths = list(range(300, 1100, 10))  # 80 points
        transmission = [0.5] * len(wavelengths)
        
        band = BandBase(
            name="wide_band",
            band_wavelengths=wavelengths,
            band_transmission=transmission,
        )
        assert len(band.band_wavelengths) == 80
        assert len(band.band_transmission) == 80

    def test_band_base_missing_name(self):
        """Test that name is required"""
        with pytest.raises(ValidationError) as exc_info:
            BandBase(
                band_wavelengths=[500.0],
                band_transmission=[1.0],
            )
        assert "name" in str(exc_info.value)

    def test_band_base_missing_wavelengths(self):
        """Test that band_wavelengths is required"""
        with pytest.raises(ValidationError) as exc_info:
            BandBase(
                name="test",
                band_transmission=[1.0],
            )
        assert "band_wavelengths" in str(exc_info.value)

    def test_band_base_missing_transmission(self):
        """Test that band_transmission is required"""
        with pytest.raises(ValidationError) as exc_info:
            BandBase(
                name="test",
                band_wavelengths=[500.0],
            )
        assert "band_transmission" in str(exc_info.value)

    def test_band_base_empty_wavelengths(self):
        """Test that wavelengths array cannot be empty"""
        with pytest.raises(ValidationError) as exc_info:
            BandBase(
                name="test",
                band_wavelengths=[],
                band_transmission=[],
            )
        assert "must not be empty" in str(exc_info.value)

    def test_band_base_empty_transmission(self):
        """Test that transmission array cannot be empty"""
        with pytest.raises(ValidationError) as exc_info:
            BandBase(
                name="test",
                band_wavelengths=[500.0],
                band_transmission=[],
            )
        assert "must not be empty" in str(exc_info.value)

    def test_band_base_mismatched_lengths(self):
        """Test that wavelengths and transmission must have same length"""
        with pytest.raises(ValidationError) as exc_info:
            BandBase(
                name="test",
                band_wavelengths=[400.0, 500.0, 600.0],
                band_transmission=[0.5, 0.9],  # Too short
            )
        assert "same length" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            BandBase(
                name="test",
                band_wavelengths=[400.0, 500.0],
                band_transmission=[0.5, 0.9, 0.5],  # Too long
            )
        assert "same length" in str(exc_info.value)

    def test_band_base_accepts_integers(self):
        """Test that integer values are accepted and converted to float"""
        band = BandBase(
            name="int_band",
            band_wavelengths=[400, 500, 600],
            band_transmission=[0, 1, 0],
        )
        # Pydantic should coerce to float
        assert all(isinstance(w, (int, float)) for w in band.band_wavelengths)
        assert all(isinstance(t, (int, float)) for t in band.band_transmission)

    def test_band_base_realistic_filter(self):
        """Test with realistic LSST filter data"""
        # Simplified g-band transmission curve
        wavelengths = [400, 425, 450, 475, 500, 525, 550]
        transmission = [0.0, 0.3, 0.7, 0.9, 0.8, 0.4, 0.0]
        
        band = BandBase(
            name="lsst_g",
            band_wavelengths=wavelengths,
            band_transmission=transmission,
        )
        assert band.name == "lsst_g"
        assert max(band.band_transmission) == 0.9
        assert min(band.band_transmission) == 0.0


class TestBandCreate:
    """Tests for BandCreate model"""

    def test_valid_band_create(self):
        """Test creating a valid BandCreate"""
        band = BandCreate(
            name="r_band",
            band_wavelengths=[500, 600, 700],
            band_transmission=[0.1, 0.8, 0.1],
        )
        assert band.name == "r_band"
        assert len(band.band_wavelengths) == 3

    def test_band_create_inherits_validation(self):
        """Test that BandCreate inherits validation from BandBase"""
        with pytest.raises(ValidationError) as exc_info:
            BandCreate(
                name="test",
                band_wavelengths=[500],
                band_transmission=[],
            )
        assert "must not be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            BandCreate(
                name="test",
                band_wavelengths=[500, 600],
                band_transmission=[0.5],
            )
        assert "same length" in str(exc_info.value)


class TestBand:
    """Tests for Band model"""

    def test_valid_band(self):
        """Test creating a valid Band with all fields"""
        band = Band(
            id=1,
            name="i_band",
            band_wavelengths=[700, 750, 800, 850],
            band_transmission=[0.0, 0.5, 0.9, 0.0],
        )
        assert band.id == 1
        assert band.name == "i_band"
        assert len(band.band_wavelengths) == 4
        assert len(band.band_transmission) == 4

    def test_band_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Band(
                id=0,
                name="test",
                band_wavelengths=[500],
                band_transmission=[1.0],
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Band(
                id=-1,
                name="test",
                band_wavelengths=[500],
                band_transmission=[1.0],
            )
        assert "greater than 0" in str(exc_info.value)

    def test_band_missing_id(self):
        """Test that id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Band(
                name="test",
                band_wavelengths=[500],
                band_transmission=[1.0],
            )
        assert "id" in str(exc_info.value)

    def test_band_inherits_array_validation(self):
        """Test that Band inherits array validation from BandBase"""
        with pytest.raises(ValidationError) as exc_info:
            Band(
                id=1,
                name="test",
                band_wavelengths=[],
                band_transmission=[],
            )
        assert "must not be empty" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Band(
                id=1,
                name="test",
                band_wavelengths=[500, 600],
                band_transmission=[0.5],
            )
        assert "same length" in str(exc_info.value)

    def test_band_from_attributes(self):
        """Test that from_attributes config works"""
        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 5
            name = "z_band"
            band_wavelengths = [800, 850, 900, 950, 1000]
            band_transmission = [0.0, 0.4, 0.8, 0.4, 0.0]

        orm_obj = MockORMObject()
        band = Band.model_validate(orm_obj)
        assert band.id == 5
        assert band.name == "z_band"
        assert band.band_wavelengths == [800, 850, 900, 950, 1000]
        assert band.band_transmission == [0.0, 0.4, 0.8, 0.4, 0.0]

    def test_band_col_names_for_table(self):
        """Test that col_names_for_table class variable is set correctly"""
        expected_cols = ["id", "name"]
        assert Band.col_names_for_table == expected_cols

    def test_band_field_descriptions(self):
        """Test that field descriptions are set"""
        schema = Band.model_json_schema()
        assert "Unique name for this band" in schema["properties"]["name"]["description"]
        assert "Wavelengths for band transmission grid" in schema["properties"]["band_wavelengths"]["description"]
        assert "Transmission at given wavelengths" in schema["properties"]["band_transmission"]["description"]

    def test_band_realistic_lsst_filters(self):
        """Test with realistic LSST filter examples"""
        lsst_filters = [
            ("u", [320, 350, 380, 400], [0.0, 0.5, 0.8, 0.0]),
            ("g", [400, 450, 500, 550], [0.0, 0.7, 0.9, 0.0]),
            ("r", [550, 600, 650, 700], [0.0, 0.8, 0.9, 0.0]),
            ("i", [700, 750, 800, 850], [0.0, 0.7, 0.8, 0.0]),
            ("z", [850, 900, 950, 1000], [0.0, 0.6, 0.7, 0.0]),
            ("y", [950, 1000, 1050, 1100], [0.0, 0.5, 0.6, 0.0]),
        ]
        
        for idx, (filter_name, wavelengths, transmission) in enumerate(lsst_filters, start=1):
            band = Band(
                id=idx,
                name=f"lsst_{filter_name}",
                band_wavelengths=wavelengths,
                band_transmission=transmission,
            )
            assert band.name == f"lsst_{filter_name}"
            assert len(band.band_wavelengths) == len(band.band_transmission)

    def test_band_json_serialization(self):
        """Test that Band can be serialized to/from JSON"""
        original = Band(
            id=1,
            name="test_band",
            band_wavelengths=[400.0, 500.0, 600.0],
            band_transmission=[0.1, 0.9, 0.1],
        )
        
        # Serialize to JSON
        json_str = original.model_dump_json()
        
        # Deserialize from JSON
        restored = Band.model_validate_json(json_str)
        
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.band_wavelengths == original.band_wavelengths
        assert restored.band_transmission == original.band_transmission

    def test_band_with_numpy_conversion(self):
        """Test that Band works with NumPy array conversion"""
        import numpy as np
        
        # Create with numpy arrays (converted to lists)
        wavelengths_np = np.array([400, 500, 600])
        transmission_np = np.array([0.1, 0.9, 0.1])
        
        band = Band(
            id=1,
            name="numpy_band",
            band_wavelengths=wavelengths_np.tolist(),
            band_transmission=transmission_np.tolist(),
        )
        
        # Verify can convert back to numpy
        wavelengths_restored = np.array(band.band_wavelengths)
        transmission_restored = np.array(band.band_transmission)
        
        assert np.array_equal(wavelengths_restored, wavelengths_np)
        assert np.array_equal(transmission_restored, transmission_np)
