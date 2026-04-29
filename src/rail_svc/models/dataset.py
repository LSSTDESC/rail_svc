"""Pydantic model for the Dataset"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class DatasetBase(BaseModel):
    """Dataset parameters that are in DB tables and also used to create new rows"""

    #: Name for this Dataset, unique
    name: str = Field(..., description="Unique name for this dataset")

    #: Path to the relevant file (could be None)
    path: str = Field(..., description="File path to the stored dataset")

    #: Number of objects in the dataset
    n_objects: int = Field(..., description="Number of objects in the dataset")


class DatasetCreate(DatasetBase):
    """Dataset Parameters that are used to create new rows but not in DB tables"""

    #: Name of the associated catalog tag
    catalog_tag_name: str = Field(..., description="Name of the associated catalog tag")

    #: Validate the files before loading
    validate_file: bool = Field(False, description="Whether to validate the file before loading")


class Dataset(DatasetBase):
    """Magnitude data about set of objects that can be used to obtain
    p(z) estimates.

    It is associated with a `CatalogTag` that defines which columns
    names to expect.

    It is stored in a file
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "n_objects",
        "catalog_tag_id",
        "path",
    ]

    #: primary key
    id_: int = Field(..., gt=0)

    #: foreign key into catalog_tag table
    catalog_tag_id: int = Field(..., gt=0, description="Foreign key referencing CatalogTag.id_")
