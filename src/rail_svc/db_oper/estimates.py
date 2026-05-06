"""Estimates operations.

This module provides operations for creating Estimates records with automatic
catalog tag resolution and file validation.


Notes
-----
File validation includes:
- Path traversal protection
- File existence check
- Data format validation
- Object count extraction
"""

from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

import qp

from .. import db, db_funcs, models
from .base import TableContext, FileValidatedOperations


__all__ = ["EstimatesOperations", "estimates"]


class EstimatesOperations(FileValidatedOperations[db.Estimates, models.Estimates, models.EstimatesCreate]):
    """Create operations for Estimates table.

    Handles automatic lookup of dataset and estimator by either ID or name,
    with optional file validation and object counting.
    """

    async def get_create_kwargs(
        self,
        session: AsyncSession,
        *,
        path: str | None = None,
        dataset_id: int | None = None,
        dataset_name: str | None = None,
        estimator_id: int | None = None,
        estimator_name: str | None = None,
        validate_file: bool = True,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating an Estimates instance.

        Resolves foreign keys for dataset and estimator by ID or name,
        validates file path security, and optionally validates file content.

        Parameters
        ----------
        session
            Database session
        path
            Path to estimates file relative to archive directory.
            If None, n_objects must be provided in extra_kwargs.
        dataset_id
            ID of the dataset (provide this OR dataset_name)
        dataset_name
            Name of the dataset (provide this OR dataset_id)
        estimator_id
            ID of the estimator (provide this OR estimator_name)
        estimator_name
            Name of the estimator (provide this OR estimator_id)
        validate_file
            Whether to validate file exists and count objects.
            If False, n_objects must be provided in extra_kwargs.
        **extra_kwargs
            Additional fields to pass through to model.
            May include 'n_objects' if validate_file=False.

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for Estimates creation, including:
            - path: str | None
            - dataset_id: int
            - estimator_id: int
            - n_objects: int
            - any fields from extra_kwargs

        Raises
        ------
        ValueError
            If neither dataset_id nor dataset_name provided,
            if neither estimator_id nor estimator_name provided,
            if path escapes archive directory,
            if file validation fails,
            or if n_objects not provided when validate_file=False
        NoResultFound
            If dataset or estimator lookup by name fails
        FileNotFoundError
            If estimates file doesn't exist when validate_file=True

        Examples
        --------
        With file validation:
        
        >>> kwargs = await ops.get_create_kwargs(
        ...     session,
        ...     path="estimates/pzflow_results.hdf5",
        ...     dataset_name="LSST_Y1",
        ...     estimator_name="PZFlow",
        ...     validate_file=True
        ... )
        >>> # kwargs includes n_objects from file

        Without validation:

        >>> kwargs = await ops.get_create_kwargs(
        ...     session,
        ...     path="estimates/custom.hdf5",
        ...     dataset_id=123,
        ...     estimator_id=456,
        ...     n_objects=5000,
        ...     validate_file=False
        ... )
        """
        
        # 1. Resolve dataset_id and estimator_id foreign keys
        dataset_id, dataset_obj = await db_funcs.read.lookup_by_id_or_name(
            db.Dataset, session, dataset_id, dataset_name,
            need_object=validate_file and path is not None
        )

        estimator_id, _ = await db_funcs.read.lookup_by_id_or_name(
            db.Estimator,
            session,
            estimator_id,
            estimator_name,
        )

        # 2. Process path and determine n_objects
        n_objects = await self._process_path(path, dataset_obj, validate_file, extra_kwargs)

        # 3. Build final kwargs
        result = {
            "path": path,
            "dataset_id": dataset_id,
            "estimator_id": estimator_id,
            "n_objects": n_objects,
        }

        # Add extra kwargs, but remove n_objects if present (we set it above)
        extra_kwargs_filtered = {k: v for k, v in extra_kwargs.items() if k != "n_objects"}
        result.update(extra_kwargs_filtered)

        return result

    def get_file_length(self, path: Path) -> int:
        """Extract number of objects from qp ensemble file.
    
        Parameters
        ----------
        path : Path
            Absolute path to the estimates file
        
        Returns
        -------
        int
            Number of objects in the ensemble
        
        """
        return qp.data_length(str(path))    

    
# Module-level singleton
estimates: EstimatesOperations = EstimatesOperations(TableContext.from_db_class(db.Estimates))
