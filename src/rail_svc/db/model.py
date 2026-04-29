"""Database model for Model table"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from .algorithm import Algorithm
from .base import Base
from .catalog_tag import CatalogTag

if TYPE_CHECKING:
    from .estimator import Estimator

logger = structlog.get_logger(__name__)


class Model(Base):
    """Model representing a trained machine learning model.

    A Model is associated with an Algorithm and CatalogTag, and references
    a file containing the serialized model data.
    """

    __tablename__ = "model"

    #: primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    #: Name for this Model, unique
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: Path to the relevant file
    path: Mapped[str] = mapped_column()

    #: foreign key into `Algorithm` table
    algo_id: Mapped[int] = mapped_column(
        ForeignKey("algorithm.id_", ondelete="CASCADE"),
        index=True,
    )

    #: foreign key into catalog_tag table
    catalog_tag_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_tag.id_", ondelete="CASCADE"),
        index=True,
    )

    # Relationship - read-only access to associated algorithm
    algo: Mapped[Algorithm] = relationship(
        "Algorithm",
        back_populates="models",
        viewonly=True,
    )

    # Relationship - read-only access to associated catalog_tag
    catalog_tag: Mapped[CatalogTag] = relationship(
        "CatalogTag",
        back_populates="models",
        viewonly=True,
    )

    # Relationship - read-only access to associated Estimators
    estimators: Mapped[list[Estimator]] = relationship(
        "Estimator",
        back_populates="model",
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
        return models.ModelCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Model
        """
        return models.Model

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'model' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        return (
            f"Model(name={self.name!r}, id_={self.id_}, "
            f"algo_id={self.algo_id}, catalog_tag_id={self.catalog_tag_id}, "
            f"path={self.path!r})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the Model.

        Returns
        -------
        str
            Just the model name
        """
        return self.name
