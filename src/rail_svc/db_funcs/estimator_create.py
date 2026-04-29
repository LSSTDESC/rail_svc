"""Estimator creation operations.

This module provides operations for creating Estimator records with automatic
lookup of the associated Model by ID or name.

Examples
--------
Create estimator by model name:

>>> from myapp.operations.estimator import estimator_creator
>>>
>>> async with session.begin():
...     estimator = await estimator_creator.create_row(
...         session,
...         model_name="RandomForest_v1",
...         name="RF_Estimator_1",
...         description="Random forest photo-z estimator"
...     )

Create estimator by model ID:

>>> async with session.begin():
...     estimator = await estimator_creator.create_row(
...         session,
...         model_id=123,
...         name="RF_Estimator_2",
...         version="1.0.0"
...     )
"""

import logging
from typing import Any

from sqlalchemy.ext.asyncio import async_scoped_session

from .. import db_funcs
from ..db import Estimator, Model
from .row_create import RowCreateBase, RowCreateContext

logger = logging.getLogger(__name__)

__all__ = ["EstimatorCreate", "estimator_creator"]


class EstimatorCreate(RowCreateBase[Estimator]):
    """Create operations for Estimator table.

    Handles automatic lookup of Model by either ID or name.
    """

    async def get_create_kwargs(
        self,
        session: async_scoped_session,
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
            Model,
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
estimator_creator: EstimatorCreate = EstimatorCreate(RowCreateContext.from_db_class(Estimator))
