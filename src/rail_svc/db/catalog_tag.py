"""Database model for CatalogTag table"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .. import models
from .base import Base

if TYPE_CHECKING:
    from .dataset import Dataset
    from .estimator import Estimator
    from .model import Model


class CatalogTag(Base):
    """Catalog tag


    Attributes
    ----------
    id_ : int
        Primary key, auto-incrementing unique identifier
    name : str
        Unique name for this catalog tag (e.g., 'roman', 'rubin')
    estimators : list[Estimator]
        Read-only list of estimators tagged with this catalog tag
    models : list[Model]
        Read-only list of models tagged with this catalog tag
    datasets : list[Dataset]
        Read-only list of datasets tagged with this catalog tag

    Examples
    --------
    >>> tag = CatalogTag(
    ...     name="roman",
    ...     metadata={"redshift_col": "redshift", "object_id_col": "object_id"}
    ... )
    """

    __tablename__ = "catalog_tag"

    # Primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this catalog tag
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

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
    def pydantic_create_class(cls) -> type[BaseModel]:
        """Pydantic model used to create rows in this table.

        Subclasses must implement this to specify their associated
        Pydantic model for creation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class
        """
        return models.CatalogTagCreate

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
            String showing id_, name, and description
        """
        return f"CatalogTag(id_={self.id_}, name='{self.name})"

    def __str__(self) -> str:
        """Return a simple string representation of the CatalogTag.

        Returns
        -------
        str
            Just the catalog tag name
        """
        return self.name
