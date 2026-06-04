"""Pydantic model for the CatalogTag"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class CatalogTagBase(BaseModel):
    """CatalogTag parameters that are in DB tables and also used to create new rows"""

    #: Name for this CatalogTag, unique
    name: str = Field(..., description="Unique name for this catalog tag")


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
    col_names_for_table: ClassVar[list[str]] = ["id_", "name"]

    #: primary key
    id_: int = Field(..., gt=0)
