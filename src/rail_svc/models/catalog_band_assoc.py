"""Pydantic model for the CatalogBandAssoc"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class CatalogBandAssocBase(BaseModel):
    """CatalogBandAssoc parameters that are in DB tables and also used to create new rows"""

    #: What the band magntitude is called in the catalog
    mag_column_name: str = Field(..., description="Name given to magnitude column in catalog")

    #: What the band magntitude error is called in the catalog tag
    mag_err_column_name: str = Field(..., description="Name given to magnitude error column in catalog")


class CatalogBandAssocCreate(CatalogBandAssocBase):
    """CatalogBandAssoc Parameters that are used to create new rows but not in DB tables"""

    #: Name of the CatalogTag
    catalog_tag_name: str = Field(..., description="Name of the associated CatalogTag")

    #: Name of the Band
    band_name: str = Field(..., description="Name of the associated Band")


class CatalogBandAssoc(CatalogBandAssocBase):
    """Defines an association between a CatalogTag and a Band

    This is basically an association that says a Band is present in a CatalogTag
    and the average five sigma limiting magnitude for the Band in that CatalogTag.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "catalog_tag_id",
        "band_id",
        "mag_column_name",
    ]

    #: primary key
    id_: int = Field(..., gt=0)

    #: foreign key into catalog_tag table
    catalog_tag_id: int = Field(..., gt=0, description="Foreign key referencing CatalogTag.id_")

    #: foreign key into band table
    band_id: int = Field(..., gt=0, description="Foreign key referencing Band.id_")
