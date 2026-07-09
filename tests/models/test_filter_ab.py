"""Tests for FilterABCreate model validators."""

import pytest
from pydantic import ValidationError

from rail_svc.models import FilterABCreate


class TestFilterABCreateValidators:
    """Test FilterABCreate field validators."""

    def test_valid_creation(self):
        fab = FilterABCreate(
            name="test", redshifts=[0.0, 0.5], fluxes=[1.5, 2.3], band_name="g", sed_name="elliptical"
        )
        assert fab.name == "test"

    def test_empty_redshifts_raises(self):
        with pytest.raises(ValidationError, match="Array must not be empty"):
            FilterABCreate(name="test", redshifts=[], fluxes=[1.5], band_name="g", sed_name="e")

    def test_empty_fluxes_raises(self):
        with pytest.raises(ValidationError, match="Array must not be empty"):
            FilterABCreate(name="test", redshifts=[0.0], fluxes=[], band_name="g", sed_name="e")

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValidationError, match="fluxes must have same length"):
            FilterABCreate(name="test", redshifts=[0.0, 0.5], fluxes=[1.5], band_name="g", sed_name="e")
