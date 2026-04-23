"""Pydantic model for the CatalogTag"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CatalogTagBase(BaseModel):
    """CatalogTag parameters that are in DB tables and also used to create new rows"""

    #: Name for this CatalogTag, unique
    name: str = Field(..., description="Unique name for this catalog tag")

    #: Name for the python class implementing the catalog tag
    class_name: str = Field(..., description="Fully qualified Python class name")

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        # Ensure it's a valid dotted path (e.g., "rail.utils.catalog_utils.LsstCatalog")
        if not all(part.isidentifier() for part in v.split(".")):
            raise ValueError("class_name must be a valid Python module path")
        return v


class CatalogTagCreate(CatalogTagBase):
    """CatalogTag Parameters that are used to create new rows but not in DB tables"""


class CatalogTag(CatalogTagBase):
    """Defines what kind of catalog we are analyzing data from.
    Specifically what to expect for the names of the magnitude columns.

    This is implemented in the `rail.utils.catalog_utils` module,
    which uses a catalog tag to set the default parameters for
    RAIL modules to match the catalog.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = ["id", "name", "class_name"]

    #: primary key
    id: int = Field(..., gt=0)
