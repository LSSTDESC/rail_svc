"""Database model for Estimator table"""

from typing import Any, TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from .base import Base

if TYPE_CHECKING:
    from .model import Model
    from .algorithm import Algorithm
    from .catalog_tag import CatalogTag


class Estimator(Base):
    """Estimator configuration for machine learning models.

    An Estimator represents a specific configuration of a machine learning
    algorithm that can be used to train models. It captures the hyperparameters
    and settings needed to create a trainable model instance.

    The Estimator is associated with a Model, and through that relationship
    has access to the Algorithm and CatalogTag. This normalized design
    ensures data consistency.

    Attributes
    ----------
    id_ : int
        Primary key, auto-incrementing unique identifier
    name : str
        Unique name for this estimator configuration
    model_id : int
        Foreign key to the Model this estimator is associated with
    config : Dict[str, Any] | None
        JSON-serialized configuration parameters (hyperparameters, etc.)
    model : Model
        The associated Model instance

    Notes
    -----
    To access the Algorithm or CatalogTag, use the model relationship:
        estimator.model.algo
        estimator.model.catalog_tag
    """

    __tablename__ = "estimator"

    # Primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this estimator
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    # Foreign key to model (which has algo_id and catalog_tag_id)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id_", ondelete="CASCADE"),
        index=True,
    )

    # Configuration stored as JSON
    config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    # Relationship - read-only access to associated model
    model: Mapped["Model"] = relationship(
        "Model",
        back_populates="estimators",
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
        return models.EstimatorCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Estimator
        """
        return models.Estimator

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'estimator' for use in help functions and descriptions
        """
        return cls.__tablename__

    # Convenience properties for accessing related data
    @property
    def algo_id(self) -> int:
        """Get the algorithm ID from the associated model.

        Returns
        -------
        int
            The algorithm ID
        """
        return self.model.algo_id

    @property
    def catalog_tag_id(self) -> int:
        """Get the catalog tag ID from the associated model.

        Returns
        -------
        int
            The catalog tag ID
        """
        return self.model.catalog_tag_id

    @property
    def algo(self) -> "Algorithm":
        """Get the associated Algorithm via the model.

        Returns
        -------
        Algorithm
            The algorithm instance
        """
        return self.model.algo

    @property
    def catalog_tag(self) -> "CatalogTag":
        """Get the associated CatalogTag via the model.

        Returns
        -------
        CatalogTag
            The catalog tag instance
        """
        return self.model.catalog_tag

    @property
    def algo_name(self) -> str:
        """Get the name from the associated algorithm.

        Returns
        -------
        str
            The algorithm name
        """
        return self.algo.name

    @property
    def catalog_tag_name(self) -> str:
        """Get the name from the associated catalog tag

        Returns
        -------
        str
            The catalog tag name
        """
        return self.catalog_tag.name

    def __repr__(self) -> str:
        """Return a detailed string representation of the Estimator.

        Returns
        -------
        str
            String showing id_, name, and model_id
        """
        return f"Estimator(id_={self.id_}, name='{self.name}', model_id={self.model_id})"

    def __str__(self) -> str:
        """Return a simple string representation of the Estimator.

        Returns
        -------
        str
            Just the estimator name
        """
        return self.name
