"""Database model for Estimator table"""

from typing import Any

from pydantic import BaseModel
from sqlalchemy import JSON, String
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from .base import Base


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
    id : int
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
    id: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this estimator
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    # Foreign key to model (which has algo_id and catalog_tag_id)
    model_id: Mapped[int] = mapped_column(
        ForeignKey("model.id", ondelete="CASCADE"),
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
            String showing id, name, and model_id
        """
        return f"Estimator(id={self.id}, name='{self.name}', model_id={self.model_id})"

    def __str__(self) -> str:
        """Return a simple string representation of the Estimator.

        Returns
        -------
        str
            Just the estimator name
        """
        return self.name

    @classmethod
    async def get_create_kwargs(
        cls,
        session: async_scoped_session,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare keyword arguments for creating an Estimator.

        This method handles the logic of looking up a model by either
        ID or name. The algorithm and catalog tag are accessed through
        the model relationship.

        Parameters
        ----------
        session : async_scoped_session
            Database session for queries
        **kwargs : Any
            Must contain 'name' and either 'model_id' or 'model_name'.
            May optionally contain 'config'.

        Returns
        -------
        dict[str, Any]
            Complete keyword arguments ready for row creation, including:
            - name
            - config
            - model_id

        Raises
        ------
        RAILMissingRowCreateInputError
            If required fields are missing or invalid

        Examples
        --------
        >>> # Create by model_id
        >>> kwargs = await Estimator.get_create_kwargs(
        ...     session,
        ...     name="my_estimator",
        ...     model_id=123,
        ...     config={"learning_rate": 0.01}
        ... )

        >>> # Create by model_name
        >>> kwargs = await Estimator.get_create_kwargs(
        ...     session,
        ...     name="my_estimator",
        ...     model_name="baseline_model",
        ...     config={"max_depth": 10}
        ... )
        """
        # Validate required field
        if "name" not in kwargs:
            raise KeyError(
                "Missing required field 'name' to create Estimator"
            )

        name = kwargs["name"]
        config = kwargs.get("config", {})

        # Get model either by ID or by name
        model_id = kwargs.get("model_id")

        if model_id is None:
            # Must provide model_name if model_id not provided
            model_name = kwargs.get("model_name")
            if model_name is None:
                raise KeyError(
                    "Either 'model_id' or 'model_name' must be provided to create Estimator"
                )

            # Look up model by name
            from .model import Model
            model = await Model.get_row_by_name(session, model_name)
            model_id = model.id

        # Build kwargs (no need to fetch model if we already have model_id)
        return {
            "name": name,
            "config": config,
            "model_id": model_id,
        }
