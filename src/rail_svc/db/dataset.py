"""Database model for Dataset table"""

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from .base import Base

if TYPE_CHECKING:
    from .catalog_tag import CatalogTag
    from .estimates import Estimates

logger = structlog.get_logger(__name__)


class Dataset(Base):
    """Dataset model representing a collection of astronomical objects.

    A Dataset is associated with a CatalogTag and references a file
    containing the actual data.
    """

    __tablename__ = "dataset"

    #: primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this Dataset
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: Number of objects in the dataset
    n_objects: Mapped[int] = mapped_column()

    #: Path to the relevant file
    path: Mapped[str] = mapped_column(unique=True)

    #: foreign key into catalog_tag table
    catalog_tag_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_tag.id_", ondelete="CASCADE"),
        index=True,
    )

    # Relationship - read-only access to associated catalog_tag
    catalog_tag: Mapped["CatalogTag"] = relationship(
        "CatalogTag",
        back_populates="datasets",
        viewonly=True,
    )

    estimates: Mapped[list["Estimates"]] = relationship(
        "Estimates",
        back_populates="dataset",
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
        return models.DatasetCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Dataset
        """
        return models.Dataset

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'dataset' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        return (
            f"Dataset(name={self.name!r}, id_={self.id_}, "
            f"n_objects={self.n_objects}, catalog_tag_id={self.catalog_tag_id}, "
            f"path={self.path!r})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the Dataset.

        Returns
        -------
        str
            Just the dataset name
        """
        return self.name
