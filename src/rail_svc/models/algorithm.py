"""Pydantic model for an Algorithm"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AlgorithmBase(BaseModel):
    """Algorithm parameters that are in DB tables and also used to create new rows"""

    #: Name for this Algorithm, unique
    name: str = Field(..., description="Unique name for this algorithm")

    #: Name for the python class implementing the algorithm
    class_name: str = Field(..., description="Fully qualified Python class name")

    @field_validator("class_name")
    @classmethod
    def validate_class_name(cls, v: str) -> str:
        # Ensure it's a valid dotted path (e.g., "rail.estimation.SOMEstimator")
        if not all(part.isidentifier() for part in v.split(".")):
            raise ValueError("class_name must be a valid Python module path")
        return v


class AlgorithmCreate(AlgorithmBase):
    """Algorithm Parameters that are used to create new rows but not in DB tables"""


class Algorithm(AlgorithmBase):
    """Algorithm is wrapper for a specific RAIL class
    that implements a particular p(z) estimation algorithm.

    This just defines the particular python class implementing
    the algorithm.  The selection of a particular instance of the
    training `Model` and any non-default parameters used to
    initialize an `Estimator` are handled in their own classes.
    """

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = ["id", "name", "class_name"]

    #: primary key
    id: int = Field(..., gt=0)
