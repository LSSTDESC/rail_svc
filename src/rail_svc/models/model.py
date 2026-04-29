"""Pydantic model for the Model"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ModelBase(BaseModel):
    """Model parameters that are in DB tables and also used to create new rows"""

    #: Name for this Model, unique
    name: str = Field(..., description="Unique name for this model")

    #: path to associated file
    path: str = Field(..., description="File path to the stored model")


class ModelCreate(ModelBase):
    """Model Parameters that are used to create new rows but not in DB tables"""

    #: Name of the algorithm
    algo_name: str = Field(..., description="Name of the associated algorithm")

    #: Name of the associated catalog tag
    catalog_tag_name: str = Field(..., description="Name of the associated catalog tag")


class Model(ModelBase):
    """Specific ML model that is trained to work with a specific `Algorithm`.
    On a particular type of data (`CatalogTag`)

    Typically a `Model` is stored as a pickle or yaml file.

    The `rail.core.model.Model` class provides a standard wrapper to store meta
    data such as the name of the python class that created the model, and
    the applicable `CatalogTag` to use the model with.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "algo_id",
        "catalog_tag_id",
        "path",
    ]

    #: primary key
    id_: int = Field(..., gt=0)

    #: foreign key into algorithm table
    algo_id: int = Field(..., gt=0, description="Foreign key referencing Algorithm.id_")

    #: foreign key into catalog_tag table
    catalog_tag_id: int = Field(..., gt=0, description="Foreign key referencing CatalogTag.id_")
