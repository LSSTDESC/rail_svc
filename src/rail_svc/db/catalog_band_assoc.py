"""Database model for CatalogBandAssoc association table"""

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel
from sqlalchemy import String, UniqueConstraint
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.schema import ForeignKey

from .. import models
from .base import Base
from .row import RowMixin

if TYPE_CHECKING:
    from .band import Band
    from .catalog_tag import CatalogTag

logger = structlog.get_logger(__name__)


class CatalogBandAssoc(Base, RowMixin):
    """Association table linking CatalogTags to Bands with aliases

    This model represents the many-to-many relationship between catalog tags
    and photometric bands, including the alias name used for each band in
    the specific catalog.
    """

    __tablename__ = "catalog_band_assoc"

    __table_args__ = (
        UniqueConstraint('catalog_tag_id', 'band_alias', name='uq_catalog_band_alias'),
    )

    # Primary key
    id: Mapped[int] = mapped_column(primary_key=True)

    # What the band is called in the catalog tag
    band_alias: Mapped[str] = mapped_column(String(255))

    #: foreign key into catalog_tag table
    catalog_tag_id: Mapped[int] = mapped_column(
        ForeignKey("catalog_tag.id", ondelete="CASCADE"),
        index=True,
    )

    #: foreign key into band table
    band_id: Mapped[int] = mapped_column(
        ForeignKey("band.id", ondelete="CASCADE"),
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
            String showing id, catalog_tag_id, band_id, and alias
        """
        return (
            f"CatalogBandAssoc(id={self.id}, "
            f"catalog_tag_id={self.catalog_tag_id}, "
            f"band_id={self.band_id}, "
            f"band_alias={self.band_alias!r})"
        )

    def __str__(self) -> str:
        """Return a simple string representation of the CatalogBandAssoc.

        Returns
        -------
        str
            The band alias for this association
        """
        return self.band_alias

    @classmethod
    async def get_create_kwargs(
        cls,
        session: async_scoped_session,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating a CatalogBandAssoc instance.

        Parameters
        ----------
        session
            Database session
        **kwargs
            Must include 'band_alias'
            Should include either 'catalog_tag_id' or 'catalog_tag_name'
            Should include either 'band_id' or 'band_name'

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for CatalogBandAssoc creation

        Raises
        ------
        KeyError
            If required parameters are missing
        """
        # Validate required field
        if "band_alias" not in kwargs:
            logger.warning(
                "Missing required field to create CatalogBandAssoc",
                table=cls.__name__,
                missing_field="band_alias",
            )
            raise KeyError(
                "Missing required field 'band_alias' to create CatalogBandAssoc"
            )
        band_alias = kwargs["band_alias"]

        # Get catalog_tag either by ID or by name
        catalog_tag_id = kwargs.get("catalog_tag_id")

        if catalog_tag_id is None:
            catalog_tag_name = kwargs.get("catalog_tag_name")
            if catalog_tag_name is None:
                logger.warning(
                    "Missing catalog_tag identifier to create CatalogBandAssoc",
                    table=cls.__name__,
                )
                raise KeyError(
                    "Either 'catalog_tag_id' or 'catalog_tag_name' must be provided "
                    "to create CatalogBandAssoc"
                )

            # Look up catalog_tag by name
            from .catalog_tag import CatalogTag
            catalog_tag = await CatalogTag.get_row_by_name(session, catalog_tag_name)
            catalog_tag_id = catalog_tag.id

        # Get band either by ID or by name
        band_id = kwargs.get("band_id")

        if band_id is None:
            band_name = kwargs.get("band_name")
            if band_name is None:
                logger.warning(
                    "Missing band identifier to create CatalogBandAssoc",
                    table=cls.__name__,
                )
                raise KeyError(
                    "Either 'band_id' or 'band_name' must be provided "
                    "to create CatalogBandAssoc"
                )

            # Look up band by name
            from .band import Band
            band = await Band.get_row_by_name(session, band_name)
            band_id = band.id

        # Build kwargs
        return {
            "band_alias": band_alias,
            "catalog_tag_id": catalog_tag_id,
            "band_id": band_id,
        }
