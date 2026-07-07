"""Database model for DatasetAssoc table

This module defines the association table that links matched datasets with their
component datasets. The association represents a relationship where a "matched"
dataset has been created by matching/correlating objects from a "component" dataset.

Business Logic:
--------------
- **Matched Dataset**: The output dataset containing matched/correlated objects
- **Component Dataset**: An input dataset that contributed objects to the matching process
- **Many-to-Many Relationship**: A matched dataset can be built from multiple component
  datasets, and a component dataset can contribute to multiple matched datasets
- **Example**: If you cross-match catalog A with catalog B to create matched catalog C,
  then C is the "matched_dataset" and both A and B would have separate DatasetAssoc
  records as "component_dataset" entries pointing to C

Constraints:
-----------
- Each association must have a unique name
- A dataset cannot be associated with itself (matched_dataset_id != component_dataset_id)
- Each (matched_dataset_id, component_dataset_id) pair must be unique to prevent
  duplicate associations
"""

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel
from sqlalchemy import CheckConstraint, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from macon.db.base import Base

if TYPE_CHECKING:
    from .dataset import Dataset

logger = structlog.get_logger(__name__)


class DatasetAssoc(Base):
    """Association between matched and component datasets.

    This table implements a many-to-many relationship between datasets, specifically
    tracking which component datasets were used to create a matched dataset.

    Attributes
    ----------
    id_ : int
        Primary key
    name : str
        Unique human-readable name for this association (e.g., "gaia_dr3_to_sdss_match")
    matched_dataset_id : int
        Foreign key to the dataset that is the result of matching
    component_dataset_id : int
        Foreign key to the dataset that was used as input for matching
    matched_dataset : Dataset
        Read-only relationship to the matched dataset
    component_dataset : Dataset
        Read-only relationship to the component dataset

    Examples
    --------
    If you match Gaia DR3 with SDSS to create a new matched catalog:
    - matched_dataset: The new matched catalog
    - component_dataset: Either Gaia DR3 or SDSS (you'd have two rows)
    """

    __tablename__ = "dataset_assoc"

    # Table-level constraints
    __table_args__ = (
        # Prevent a dataset from being matched with itself
        CheckConstraint(
            "matched_dataset_id != component_dataset_id", name="ck_dataset_assoc_no_self_reference"
        ),
        # Prevent duplicate associations between the same two datasets
        UniqueConstraint(
            "matched_dataset_id", "component_dataset_id", name="uq_dataset_assoc_matched_component"
        ),
    )

    #: Primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    #: Unique name for this association
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: Foreign key to the matched dataset (the output of the matching process)
    matched_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id_", ondelete="CASCADE"),
        index=True,
    )

    #: Foreign key to the component dataset (an input to the matching process)
    component_dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id_", ondelete="CASCADE"),
        index=True,
    )

    #: Relationship to matched dataset (read-only)
    matched_dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        foreign_keys=[matched_dataset_id],
        viewonly=True,
    )

    #: Relationship to component dataset (read-only)
    component_dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        foreign_keys=[component_dataset_id],
        viewonly=True,
    )

    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        """Pydantic model for creating DatasetAssoc instances.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for creating new associations
        """
        return models.DatasetAssocCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Pydantic model for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for DatasetAssoc
        """
        return models.DatasetAssoc

    @classmethod
    def class_string(cls) -> str:
        """Return the table name identifier.

        Returns
        -------
        str
            The string 'dataset_assoc' for use in help functions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        """Detailed string representation for debugging.

        Returns
        -------
        str
            A comprehensive representation including all key fields
        """
        return (
            f"DatasetAssoc(name={self.name!r}, id_={self.id_}, "
            f"matched_dataset_id={self.matched_dataset_id}, "
            f"component_dataset_id={self.component_dataset_id})"
        )

    def __str__(self) -> str:
        """Simple string representation.

        Returns
        -------
        str
            The association name
        """
        return self.name
