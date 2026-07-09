"""Tests for SedCreate model validators."""

import pytest
from pydantic import ValidationError

from rail_svc.models import SedCreate


class TestSedCreateValidators:
    """Test SedCreate field validators."""

    def test_valid_creation(self):
        sed = SedCreate(name="test", sed_wavelengths=[100.0, 200.0], sed_values=[0.1, 0.5])
        assert sed.name == "test"

    def test_empty_wavelengths_raises(self):
        with pytest.raises(ValidationError, match="Array must not be empty"):
            SedCreate(name="test", sed_wavelengths=[], sed_values=[0.1])

    def test_empty_values_raises(self):
        with pytest.raises(ValidationError, match="Array must not be empty"):
            SedCreate(name="test", sed_wavelengths=[100.0], sed_values=[])

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValidationError, match="sed_values must have same length"):
            SedCreate(name="test", sed_wavelengths=[100.0, 200.0], sed_values=[0.1])
