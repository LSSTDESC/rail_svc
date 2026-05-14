"""Operations for DatasetAssoc table.

This module provides CRUD operations for dataset associations, which link
matched datasets with their component datasets.
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from .. import db, db_funcs, models
from .base import TableContext, TableOperations

logger = logging.getLogger(__name__)

__all__ = ["DatasetAssocOperations", "dataset_assoc"]


class DatasetAssocOperations(
    TableOperations[db.DatasetAssoc, models.DatasetAssoc, models.DatasetAssocCreate]
):
    """Operations for managing dataset associations.

    This class provides methods for creating and managing associations between
    matched datasets and their component datasets.
    """

    async def get_create_kwargs(
        self,
        session: AsyncSession,
        *,
        name: str | None = None,
        matched_dataset_id: int | None = None,
        matched_dataset_name: str | None = None,
        component_dataset_id: int | None = None,
        component_dataset_name: str | None = None,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating a DatasetAssoc instance.

        Resolves matched and component dataset foreign keys by ID or name,
        and validates that the datasets are not the same.

        Parameters
        ----------
        session :
            Database session
        name :
            Unique name for this association (e.g., "gaia_sdss_match_component_1")
        matched_dataset_id :
            ID of the matched dataset (provide this OR matched_dataset_name)
        matched_dataset_name :
            Name of the matched dataset (provide this OR matched_dataset_id)
        component_dataset_id :
            ID of the component dataset (provide this OR component_dataset_name)
        component_dataset_name :
            Name of the component dataset (provide this OR component_dataset_id)
        **extra_kwargs : Any
            Additional fields (currently unused, reserved for future extension)

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for DatasetAssoc creation, containing:
            - name: str - The association name
            - matched_dataset_id: int - Resolved matched dataset ID
            - component_dataset_id: int - Resolved component dataset ID

        Raises
        ------
        ValueError
            If neither matched_dataset_id nor matched_dataset_name provided,
            if matched dataset not found,
            if neither component_dataset_id nor component_dataset_name provided,
            if component dataset not found,
            if matched_dataset_id equals component_dataset_id (self-reference)

        Examples
        --------
        Create association by dataset IDs:

        >>> kwargs = await ops.get_create_kwargs(
        ...     session,
        ...     name="gaia_sdss_match_gaia_component",
        ...     matched_dataset_id=100,
        ...     component_dataset_id=50
        ... )

        Create association by dataset names:

        >>> kwargs = await ops.get_create_kwargs(
        ...     session,
        ...     name="gaia_sdss_match_sdss_component",
        ...     matched_dataset_name="gaia_sdss_matched",
        ...     component_dataset_name="sdss_dr16"
        ... )
        """
        # Log any unused kwargs for debugging
        if extra_kwargs:
            logger.warning(f"Ignoring extra kwargs in DatasetAssoc creation: {list(extra_kwargs.keys())}")

        # 1. Resolve matched dataset foreign key
        try:
            matched_dataset_id, matched_dataset_obj = await db_funcs.read.lookup_by_id_or_name(
                db.Dataset,
                session,
                matched_dataset_id,
                matched_dataset_name,
            )
        except ValueError as e:
            raise ValueError(f"Failed to resolve matched dataset: {e}") from e

        # 2. Resolve component dataset foreign key
        try:
            component_dataset_id, component_dataset_obj = await db_funcs.read.lookup_by_id_or_name(
                db.Dataset,
                session,
                component_dataset_id,
                component_dataset_name,  # FIXED: was component_dataset_mame
            )
        except ValueError as e:
            raise ValueError(f"Failed to resolve component dataset: {e}") from e

        # 3. Validate no self-reference (belt-and-suspenders with DB constraint)
        if matched_dataset_id == component_dataset_id:
            raise ValueError(
                f"A dataset cannot be associated with itself. "
                f"matched_dataset_id ({matched_dataset_id}) == "
                f"component_dataset_id ({component_dataset_id})"
            )

        # 4. Build final kwargs
        result = {
            "name": name,
            "matched_dataset_id": matched_dataset_id,
            "component_dataset_id": component_dataset_id,
        }

        logger.debug(
            f"Prepared DatasetAssoc creation kwargs: {result}",
            extra={
                "matched_dataset": matched_dataset_obj.name if matched_dataset_obj else None,
                "component_dataset": component_dataset_obj.name if component_dataset_obj else None,
            },
        )

        return result


# Module-level singleton
dataset_assoc: DatasetAssocOperations = DatasetAssocOperations(TableContext.from_db_class(db.DatasetAssoc))
