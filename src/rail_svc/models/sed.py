"""Pydantic model for the Band"""

from typing import ClassVar

from pydantic import (BaseModel, ConfigDict, Field, ValidationInfo,
                      field_validator)


class SedBase(BaseModel):
    """Sed parameters that are in DB tables and also used to create new rows"""

    #: Name for this Band, unique
    name: str = Field(..., description="Unique name for this band")

    #: Wavelength grid
    sed_wavelengths: list[float] = Field(..., description="Wavelengths for sed grid")

    #: SED at given wavelengths
    sed_values: list[float] = Field(..., description="SED at given wavelengths")

    @field_validator("sed_wavelengths", "sed_values")
    @classmethod
    def validate_non_empty(cls, v: list[float]) -> list[float]:
        """Ensure arrays are not empty"""
        if len(v) == 0:
            raise ValueError("Array must not be empty")
        return v

    @field_validator("sed_values")
    @classmethod
    def validate_same_length(cls, v: list[float], info: ValidationInfo) -> list[float]:
        """Ensure sed values and wavelength arrays have same length"""
        if "sed_wavelengths" in info.data:
            if len(v) != len(info.data["sed_wavelengths"]):
                raise ValueError("sed_values must have same length as sed_wavelengths")
        return v


class SedCreate(SedBase):
    """Sed Parameters that are used to create new rows but not in DB tables"""


class Sed(SedBase):
    """Information about a particular sed

    Stores the wavelength-dependent sed function that defines.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
    ]

    #: primary key
    id_: int = Field(..., gt=0)
