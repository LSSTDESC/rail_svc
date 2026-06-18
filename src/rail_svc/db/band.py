"""Database model for Band table"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .. import models
from .base import Base

if TYPE_CHECKING:
    from .catalog_band_assoc import CatalogBandAssoc
    from .filter_ab import FilterAB

class Band(Base):
    """Catalog tag


    Attributes
    ----------
    id_ : int
        Primary key, auto-incrementing unique identifier
    name : str
        Unique name for this base tag (e.g., 'lsst_u')

    Examples
    --------
    >>> tag = Band(
    ...     name="production_models",
    ...     metadata={"environment": "prod", "team": "ml-ops"}
    ... )
    """

    __tablename__ = "band"

    # Primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this catalog tag
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    # Wavelength grid
    band_wavelengths: Mapped[list[float]] = mapped_column(JSON)

    #: Transmission at given wavelengths
    band_transmission: Mapped[list[float]] = mapped_column(JSON)

    # Relationships - read-only access to tagged objects
    catalog_assocs: Mapped[list["CatalogBandAssoc"]] = relationship(
        "CatalogBandAssoc",
        back_populates="band",
        viewonly=True,
    )

    filter_abs: Mapped[list["FilterAB"]] = relationship(
        "FilterAB",
        back_populates="band",
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
        return models.BandCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for Band
        """
        return models.Band

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'band' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        """Return a detailed string representation of the Band.

        Returns
        -------
        str
            String showing id_, name, and description
        """
        return f"Band(id_={self.id_}, name='{self.name}')"

    def __str__(self) -> str:
        """Return a simple string representation of the Band.

        Returns
        -------
        str
            Just the bad name
        """
        return self.name
