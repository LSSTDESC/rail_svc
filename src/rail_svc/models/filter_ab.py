"""Pydantic model for the FilterAB"""

from typing import ClassVar

from pydantic import (BaseModel, ConfigDict, Field, ValidationInfo,
                      field_validator)


class FilterABBase(BaseModel):
    """FilterAB parameters that are in DB tables and also used to create new rows"""

    #: Name for this FilterAB, unique
    name: str = Field(..., description="Unique name for this band")

    #: Redshift grid
    redshifts: list[float] = Field(..., description="Redshfit grid")

    #: Fluxes at given redshifts
    fluxes: list[float] = Field(..., description="Fluxes at given wavelengths")

    @field_validator("redshifts", "fluxes")
    @classmethod
    def validate_non_empty(cls, v: list[float]) -> list[float]:
        """Ensure arrays are not empty"""
        if len(v) == 0:
            raise ValueError("Array must not be empty")
        return v

    @field_validator("fluxes")
    @classmethod
    def validate_same_length(cls, v: list[float], info: ValidationInfo) -> list[float]:
        """Ensure transmission and wavelength arrays have same length"""
        if "redshifts" in info.data:
            if len(v) != len(info.data["redshifts"]):
                raise ValueError("fluxes must have same length as redshifts")
        return v


class FilterABCreate(FilterABBase):
    """FilterAB Parameters that are used to create new rows but not in DB tables"""

    #: Name of the Band
    band_name: str = Field(..., description="Name of the associated Band")

    #: Name of the Sed
    sed_name: str = Field(..., description="Name of the sed Band")
    

class FilterAB(FilterABBase):
    """Information about a particular filter AB fluxes as function of redshift

    Stores the redshift-dependent flux in a particular filter
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
    ]

    #: primary key
    id_: int = Field(..., gt=0)

    #: foreign key into band table
    band_id: int = Field(..., gt=0, description="Foreign key referencing Band.id_")

    #: foreign key into sed table
    sed_id: int = Field(..., gt=0, description="Foreign key referencing Sed.id_")

    
