"""Estimator operations.

This module provides operations for creating Estimator records with automatic
lookup of the associated Model by ID or name.

"""

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from macon import db_funcs
from .. import db, models
from macon.db_oper.base import TableContext, TableOperations

__all__ = ["EstimatorOperations", "estimator"]


class EstimatorOperations(TableOperations[db.Estimator, models.Estimator, models.EstimatorCreate]):
    """Create operations for Estimator table.

    Handles automatic lookup of Model by either ID or name.
    """

    async def get_create_kwargs(
        self,
        session: AsyncSession,
        *,
        model_id: int | None = None,
        model_name: str | None = None,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating an Estimator instance.

        Resolves the model foreign key by looking up Model by name
        if model_id is not provided.

        Parameters
        ----------
        session
            Database session
        model_id
            ID of the model (provide this OR model_name)
        model_name
            Name of the model (provide this OR model_id)
        **extra_kwargs
            Additional fields to pass through to Estimator model.
            Common fields include:
            - name (str): Estimator name
            - description (str): Human-readable description
            - version (str): Version identifier
            - hyperparameters (dict): Model hyperparameters

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for Estimator creation, including
            resolved model_id and any extra fields

        Raises
        ------
        ValueError
            If neither model_id nor model_name provided, or if
            model lookup by name fails

        Examples
        --------
        By model ID:

        >>> kwargs = await creator.get_create_kwargs(
        ...     session,
        ...     model_id=123,
        ...     name="My Estimator",
        ...     version="1.0.0"
        ... )

        By model name:

        >>> kwargs = await creator.get_create_kwargs(
        ...     session,
        ...     model_name="RandomForest_v1",
        ...     name="RF Estimator",
        ...     description="Photo-z estimator using random forest"
        ... )
        """
        # Resolve model_id
        model_id, _ = await db_funcs.read.lookup_by_id_or_name(
            db.Model,
            session,
            model_id,
            model_name,
        )

        # Build final kwargs
        return {
            "model_id": model_id,
            **extra_kwargs,
        }


# Module-level singleton
estimator: EstimatorOperations = EstimatorOperations(TableContext.from_db_class(db.Estimator))
