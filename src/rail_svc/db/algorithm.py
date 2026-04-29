"""Database model for Algorithm table"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .. import models
from .base import Base

if TYPE_CHECKING:
    from .estimator import Estimator
    from .model import Model


class Algorithm(Base):
    """Algorithm database model.

    Represents a machine learning algorithm that can be used to create
    estimators and models. Each algorithm has a unique name and references
    a Python class that implements the algorithm logic.

    Attributes
    ----------
    id_ : int
        Primary key, auto-incrementing unique identifier
    name : str
        Unique name for this algorithm (e.g., 'random_forest', 'xgboost')
    class_name : str
        Fully qualified Python class name implementing the algorithm
        (e.g., 'sklearn.ensemble.RandomForestClassifier')
    estimators : list[Estimator]
        Read-only list of estimators using this algorithm
    models : list[Model]
        Read-only list of trained models using this algorithm
    """

    __tablename__ = "algorithm"

    # Primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # Algorithm identification
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    # Python class implementing the algorithm
    class_name: Mapped[str] = mapped_column(String(512))

    # Relationships - view-only access to associated objects
    estimators: Mapped[list["Estimator"]] = relationship(
        "Estimator",
        back_populates="algorithm",
        viewonly=True,
    )

    models: Mapped[list["Model"]] = relationship(
        "Model",
        back_populates="algorithm",
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
        return models.AlgorithmCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Algorithm
        """
        return models.Algorithm

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'algorithm' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        """Return a detailed string representation of the Algorithm.

        Returns
        -------
        str
            String showing id_, name, and class_name
        """
        return f"Algorithm(id_={self.id_}, name='{self.name}', class_name='{self.class_name}')"

    def __str__(self) -> str:
        """Return a simple string representation of the Algorithm.

        Returns
        -------
        str
            Just the algorithm name
        """
        return self.name
