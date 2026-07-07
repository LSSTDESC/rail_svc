"""FilterAB table operations.

Provides CRUD operations and Pydantic conversions for the FilterAB table,
with automatic lookup of band and sed foreign keys by ID or name.
"""

import logging
from typing import Any

from macon import db_funcs
from macon.db_oper.base import TableContext, TableOperations
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db, models

logger = logging.getLogger(__name__)

__all__ = ["FilterABOperations", "filter_ab"]


class FilterABOperations(TableOperations[db.FilterAB, models.FilterAB, models.FilterABCreate]):
    """Create operations for FilterAB table.

    Handles automatic lookup of band and sed by either ID or name.
    """

    async def get_create_kwargs(
        self,
        session: AsyncSession,
        name: str | None = None,
        redshifts: list[float] | None = None,
        fluxes: list[float] | None = None,
        band_id: int | None = None,
        band_name: str | None = None,
        sed_id: int | None = None,
        sed_name: str | None = None,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating a FilterAB instance.

        Resolves foreign keys by looking up band and sed by name
        if IDs are not provided.

        Parameters
        ----------
        session
            Database session
        name
            Unique name for this FilterAB entry
        redshifts
            Redshift grid values
        fluxes
            Flux values at given redshifts
        band_id
            ID of the band (provide this OR band_name)
        band_name
            Name of the band (provide this OR band_id)
        sed_id
            ID of the sed (provide this OR sed_name)
        sed_name
            Name of the sed (provide this OR sed_id)
        **extra_kwargs
            Additional fields to pass through to model

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for FilterAB creation

        Raises
        ------
        ValueError
            If neither ID nor name provided for band or sed
        NoResultFound
            If band or sed lookup by name fails
        """
        band_id, _ = await db_funcs.read.lookup_by_id_or_name(
            db.Band,
            session,
            band_id,
            band_name,
        )

        sed_id, _ = await db_funcs.read.lookup_by_id_or_name(
            db.Sed,
            session,
            sed_id,
            sed_name,
        )

        return {
            "name": name,
            "redshifts": redshifts,
            "fluxes": fluxes,
            "band_id": band_id,
            "sed_id": sed_id,
            **extra_kwargs,
        }


# Module-level singleton
filter_ab: FilterABOperations = FilterABOperations(TableContext.from_db_class(db.FilterAB))
