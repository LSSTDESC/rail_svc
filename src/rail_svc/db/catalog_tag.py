"""Database model for CatalogTag table"""

from typing import TYPE_CHECKING, Dict, Any

from pydantic import BaseModel
from sqlalchemy import String, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .. import models
from .base import Base
from .row import RowMixin

if TYPE_CHECKING:
    from .dataset import Dataset
    from .estimator import Estimator
    from .model import Model


class CatalogTag(Base, RowMixin):
    """Catalog tag for organizing ML artifacts.
    
    CatalogTags provide a way to categorize and organize datasets, models,
    and estimators. They can represent different projects, experiments,
    use cases, or any other organizational structure.
    
    Attributes
    ----------
    id : int
        Primary key, auto-incrementing unique identifier
    name : str
        Unique name for this catalog tag (e.g., 'project_alpha', 'experiment_001')
    description : str | None
        Optional human-readable description of this tag
    metadata : Dict[str, Any] | None
        Optional JSON metadata for additional tag information
    estimators : list[Estimator]
        Read-only list of estimators tagged with this catalog tag
    models : list[Model]
        Read-only list of models tagged with this catalog tag
    datasets : list[Dataset]
        Read-only list of datasets tagged with this catalog tag
    
    Examples
    --------
    >>> tag = CatalogTag(
    ...     name="production_models",
    ...     description="Models currently deployed in production",
    ...     metadata={"environment": "prod", "team": "ml-ops"}
    ... )
    """

    __tablename__ = "catalog_tag"

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this catalog tag
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    # Optional description
    description: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
    )

    # Optional metadata for additional information
    metadata: Mapped[Dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Relationships - read-only access to tagged objects
    estimators: Mapped[list["Estimator"]] = relationship(
        "Estimator",
        back_populates="catalog_tag",
        viewonly=True,
    )

    models: Mapped[list["Model"]] = relationship(
        "Model",
        back_populates="catalog_tag",
        viewonly=True,
    )

    datasets: Mapped[list["Dataset"]] = relationship(
        "Dataset",
        back_populates="catalog_tag",
        viewonly=True,
    )

    # Pydantic integration
    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.
        
        Returns
        -------
        type[BaseModel]
            The Pydantic model class for CatalogTag
        """
        return models.CatalogTag

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.
        
        Returns
        -------
        str
            The string 'catalog_tag' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        """Return a detailed string representation of the CatalogTag.
        
        Returns
        -------
        str
            String showing id, name, and description
        """
        desc = f", description='{self.description[:50]}...'" if self.description else ""
        return f"CatalogTag(id={self.id}, name='{self.name}'{desc})"

    def __str__(self) -> str:
        """Return a simple string representation of the CatalogTag.
        
        Returns
        -------
        str
            Just the catalog tag name
        """
        return self.name
