"""Database model for CatalogBandAssoc association table"""

from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from macon.db.base import Base

if TYPE_CHECKING:
    from .band import Band
    from .catalog_tag import CatalogTag

logger = structlog.get_logger(__name__)


class CatalogBandAssoc(Base):
    """Association table linking CatalogTags to Bands with aliases

    This model represents the many-to-many relationship between catalog tags
    and photometric bands, including the alias name used for each band in
    the specific catalog.
    """

    __tablename__ = "catalog_band_assoc"

    __table_args__ = (
        UniqueConstraint("catalog_tag_id", "band_id", name="uq_catalog_band"),
        UniqueConstraint("catalog_tag_id", "mag_column_name", name="uq_catalog_mag_column_name"),
        UniqueConstraint("catalog_tag_id", "mag_err_column_name", name="uq_catalog_mag_err_column_name"),
    )

    # Primary key
    id_: Mapped[int] = mapped_column(primary_key=True)

    # What the band is called in the catalog tag
    mag_column_name: Mapped[str] = mapped_column(String(255))

    #: What the band magntitude error is called in the catalog tag
    mag_err_column_name: Mapped[str] = mapped_column(String(255))

    #: foreign key into catalog_tag table
    catalog_tag_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_tag.id_", ondelete="CASCADE"),
        index=True,
    )

    #: foreign key into band table
    band_id: Mapped[int] = mapped_column(
        ForeignKey("band.id_", ondelete="CASCADE"),
        index=True,
    )

    # Relationships - read-only access to associated objects
    catalog_tag: Mapped["CatalogTag"] = relationship(
        "CatalogTag",
        back_populates="band_assocs",
        viewonly=True,
    )

    band: Mapped["Band"] = relationship(
        "Band",
        back_populates="catalog_assocs",
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
        return models.CatalogBandAssocCreate

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Return the Pydantic model class for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class for CatalogBandAssoc
        """
        return models.CatalogBandAssoc

    @classmethod
    def class_string(cls) -> str:
        """Return the class identifier string.

        Returns
        -------
        str
            The string 'catalog_band_assoc' for use in help functions and descriptions
        """
        return cls.__tablename__

    def __repr__(self) -> str:
        """Return a detailed string representation of the CatalogBandAssoc.

        Returns
        -------
        str
            String showing id_, catalog_tag_id, band_id, and alias
        """
        return (
            f"CatalogBandAssoc(id_={self.id_}, "
            f"catalog_tag_id={self.catalog_tag_id}, "
            f"mag_column_name={self.mag_column_name}, "
            f"mag_err_column_name={self.mag_err_column_name})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the CatalogBandAssoc.

        Returns
        -------
        str
            The band alias for this association
        """
        return self.mag_column_name
