"""Database model for Estimates table"""

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from macon.db.base import Base

if TYPE_CHECKING:
    from .dataset import Dataset
    from .estimator import Estimator

logger = structlog.get_logger(__name__)


class Estimates(Base):
    """Estimates model representing a collection of p(z) estimates

    An Estimates record is associated with a Dataset and an Estimator and
    references a file containing the actual probability distribution data.
    """

    __tablename__ = "estimates"

    #: primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this Estimates
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: Number of objects in the estimates
    n_objects: Mapped[int] = mapped_column()

    #: Path to the relevant file
    path: Mapped[str] = mapped_column(unique=True)

    #: foreign key into dataset table
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("dataset.id_", ondelete="CASCADE"),
        index=True,
    )

    #: foreign key into estimator table
    estimator_id: Mapped[int] = mapped_column(
        ForeignKey("estimator.id_", ondelete="CASCADE"),
        index=True,
    )

    # Relationship - read-only access to associated dataset
    dataset: Mapped["Dataset"] = relationship(
        "Dataset",
        back_populates="estimates",
        viewonly=True,
    )

    # Relationship - read-only access to associated estimator
    estimator: Mapped["Estimator"] = relationship(
        "Estimator",
        back_populates="estimates",
        viewonly=True,
    )

    # Pydantic integration
    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        """Pydantic model used to create rows in this table.

        Subclasses must implement this to specify their associated
        Pydantic model for creation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class
        """
        return models.EstimatesCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Estimates
        """
        return models.Estimates

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'estimates' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        return (
            f"Estimates(name={self.name!r}, id_={self.id_}, "
            f"n_objects={self.n_objects}, dataset_id={self.dataset_id}, "
            f"estimator_id={self.estimator_id}, "
            f"path={self.path!r})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the Estimates.

        Returns
        -------
        str
            Just the estimates name
        """
        return self.name
