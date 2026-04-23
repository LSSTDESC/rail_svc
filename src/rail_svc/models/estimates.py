"""Pydantic model for the Estimates"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class EstimatesBase(BaseModel):
    """Estimates parameters that are in DB tables and also used to create new rows"""

    #: path to the output file
    qp_file_path: str | None = Field(None, description="Path to the output qp ensemble file")


class EstimatesCreate(EstimatesBase):
    """Estimates Parameters that are used to create new rows but not in DB tables"""

    #: Name of the estimator
    estimator_name: str = Field(..., description="Name of the associated estimator")

    #: Name of the dataset
    dataset_name: str = Field(..., description="Name of the associated dataset")


class Estimates(EstimatesBase):
    """Returns per-galaxy p(z) for all objects in a particular Dataset
    using a specific Estimator.

    The output p(z) distribution is stored in a qp ensemble file.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id",
        "estimator_id",
        "dataset_id",
        "qp_file_path",
    ]

    #: primary key
    id: int = Field(..., gt=0)

    #: foreign key into estimator table
    estimator_id: int = Field(..., gt=0, description="Foreign key referencing Estimator.id")

    #: foreign key into dataset table
    dataset_id: int = Field(..., gt=0, description="Foreign key referencing Dataset.id")
