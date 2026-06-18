"""Database model for FilterAB table"""

from typing import TYPE_CHECKING

from pydantic import BaseModel
from sqlalchemy import JSON, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .. import models
from .base import Base

if TYPE_CHECKING:
    from .band import Band
    from .sed import Sed


class FilterAB(Base):
    """Filter fluxes


    Attributes
    ----------
    id_ : int
        Primary key, auto-incrementing unique identifier
    name : str
        Unique name for this set of fliterAB fluexes

    """

    __tablename__ = "filter_ab"

    # Primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # Unique name for this catalog tag
    name: Mapped[str] = mapped_column(String(255), index=True, unique=True)

    #: foreign key into band table
    band_id: Mapped[int] = mapped_column(
        ForeignKey("band.id_", ondelete="CASCADE"),
        index=True,
    )

    #: foreign key into sed table
    sed_id: Mapped[int] = mapped_column(
        ForeignKey("sed.id_", ondelete="CASCADE"),
        index=True,
    )
    
    # Redshift grid
    redshifts: Mapped[list[float]] = mapped_column(JSON)
    
    # Fluxes at given redshifts
    fluxes: Mapped[list[float]] = mapped_column(JSON)

    # Relationships - read-only access to tagged objects
    band: Mapped["Band"] = relationship(
        "Band",
        back_populates="filter_abs",
        viewonly=True,
    )

    sed: Mapped["Sed"] = relationship(
        "Sed",
        back_populates="filter_abs",
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
        return models.FilterABCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for FilterAB
        """
        return models.FilterAB

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'filter_ab' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        """Return a detailed string representation of the FilterAB.

        Returns
        -------
        str
            String showing id_, name, and description
        """
        return f"FilterAB(id_={self.id_}, name='{self.name}')"

    def __str__(self) -> str:
        """Return a simple string representation of the FilterAB.

        Returns
        -------
        str
            Just the name
        """
        return self.name
