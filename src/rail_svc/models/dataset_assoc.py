"""Pydantic model for the Dataset"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class DatasetAssocBase(BaseModel):
    """DatasetAssoc parameters that are in DB tables and also used to create new rows"""

    #: Name for this DatasetAssoc, unique
    name: str = Field(..., description="Unique name for this dataset association")


class DatasetAssocCreate(DatasetAssocBase):
    """DatasetAssoc Parameters that are used to create new rows but not in DB tables"""

    #: Name of the matched dataset
    matched_dataset_name: str = Field(..., description="Name of the matched dataset")

    #: Name of the compoment dataset
    component_dataset_name: str = Field(..., description="Name of the component dataset")


class DatasetAssoc(DatasetAssocBase):
    """Associations between datasets"""

    model_config = ConfigDict(from_attributes=True)

    #: column names to use when printing the table
    col_names_for_table: ClassVar[list[str]] = [
        "id_",
        "name",
        "n_objects",
        "matched_dataset_id",
        "component_dataset_id",
        "path",
    ]

    #: primary key
    id_: int = Field(..., gt=0)

    #: foreign key into dataset table for matched catalog
    matched_dataset_id: int = Field(..., gt=0, description="Foreign key referencing matched dataset.id_")

    #: foreign key into dataset table for component catalog
    component_dataset_id: int = Field(..., gt=0, description="Foreign key referencing matched component.id_")
