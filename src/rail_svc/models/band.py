"""Pydantic model for the Band"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class BandBase(BaseModel):
    """Band parameters that are in DB tables and also used to create new rows"""

    #: Name for this Band, unique
    name: str = Field(..., description="Unique name for this band")

    #: Wavelength grid
    band_wavelengths: list[float] = Field(..., description="Wavelengths for band transmission grid")

    #: Transmission at given wavelengths
    band_transmission: list[float] = Field(..., description="Transmission at given wavelengths")

    @field_validator("band_wavelengths", "band_transmission")
    @classmethod
    def validate_non_empty(cls, v: list[float]) -> list[float]:
        """Ensure arrays are not empty"""
        if len(v) == 0:
            raise ValueError("Array must not be empty")
        return v

    @field_validator("band_transmission")
    @classmethod
    def validate_same_length(cls, v: list[float], info: ValidationInfo) -> list[float]:
        """Ensure transmission and wavelength arrays have same length"""
        if "band_wavelengths" in info.data:
            if len(v) != len(info.data["band_wavelengths"]):
                raise ValueError("band_transmission must have same length as band_wavelengths")
        return v


class BandCreate(BandBase):
    """Band Parameters that are used to create new rows but not in DB tables"""


class Band(BandBase):
    """Information about a particular filter, in particular the transmission curve.

    Stores the wavelength-dependent transmission function that defines
    a photometric band/filter.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
    ]

    #: primary key
    id_: int = Field(..., gt=0)
