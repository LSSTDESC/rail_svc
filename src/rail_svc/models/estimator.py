"""Pydantic model for the Estimator"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class EstimatorBase(BaseModel):
    """Estimator parameters that are in DB tables and also used to create new rows"""

    #: Name for this Estimator, unique
    name: str = Field(..., description="Unique name for this estimator")

    #: Configuration parameters for this estimator
    config: dict | None = Field(None, description="Configuration parameters for this estimator")


class EstimatorCreate(EstimatorBase):
    """Estimator Parameters that are used to create new rows but not in DB tables"""

    #: Name of the model, unique
    model_name: str = Field(..., description="Name of the associated model")


class Estimator(EstimatorBase):
    """Combination of an `Algorithm` to run a trained `Model` to apply to the
    data, and any specific configuration overrides.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "algo_id",
        "catalog_tag_id",
        "model_id",
    ]

    #: primary key
    id_: int = Field(..., gt=0)

    #: foreign key into algorithm table
    algo_id: int = Field(..., gt=0, description="Foreign key referencing Algorithm.id_")

    #: foreign key into catalog_tag table
    catalog_tag_id: int = Field(..., gt=0, description="Foreign key referencing CatalogTag.id_")

    #: foreign key into model table
    model_id: int = Field(..., gt=0, description="Foreign key referencing Model.id_")
