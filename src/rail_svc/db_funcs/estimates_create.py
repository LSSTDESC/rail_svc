"""Estimates creation operations.

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

import asyncio
import logging
from pathlib import Path
from typing import Any

from sqlalchemy.ext.asyncio import async_scoped_session

import qp

from .. import db_funcs
from ..db import Dataset, Estimates, Estimator
from .row_create import RowCreateBase, RowCreateContext

logger = logging.getLogger(__name__)

__all__ = ["EstimatesCreate", "estimates_creator"]


class EstimatesCreate(RowCreateBase[Estimates]):
    """Create operations for Estimates table.

    Handles automatic lookup of dataset and estimator by either ID or name,
    with optional file validation and object counting.
    """

    async def get_create_kwargs(
        self,
        session: async_scoped_session,
        *,
        path: str | None = None,
        dataset_id: int | None = None,
        dataset_name: str | None = None,
        estimator_id: int | None = None,
        estimator_name: str | None = None,
        validate_file: bool = True,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        # 1. Resolve dataset_id and estimator_id foreign keys
        dataset_id, dataset_obj = await db_funcs.read.lookup_by_id_or_name(
            Dataset, session, dataset_id, dataset_name,
            need_object=validate_file and path is not None
        )

        estimator_id, _ = await db_funcs.read.lookup_by_id_or_name(
            Estimator,
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

    async def _process_path(
        self,
        path: str | None,
        dataset: Dataset | None,
        validate_file: bool,
        extra_kwargs: dict[str, Any],
    ) -> int:
        """Process estimates path and determine n_objects.

        Parameters
        ----------
        path
            Estimates file path (relative to archive)
        dataset
            Dataset object for validation (may be None)
        validate_file
            Whether to validate file
        extra_kwargs
            Additional kwargs that may contain n_objects

        Returns
        -------
        int
            Number of objects in estimates

        Raises
        ------
        ValueError
            If path invalid, validation fails, or n_objects not provided
            when needed
        FileNotFoundError
            If file doesn't exist when validation enabled
        """
        if path is None:
            # No path - must have n_objects in extra_kwargs
            n_objects = extra_kwargs.get("n_objects")
            if n_objects is None:
                logger.warning(
                    "No path or n_objects provided",
                    table=self.ctx.db_class.__name__,
                )
                raise ValueError("Either 'path' or 'n_objects' must be provided")
            return n_objects

        if not validate_file:
            # Path provided but validation disabled - use provided n_objects
            n_objects = extra_kwargs.get("n_objects")
            if n_objects is None:
                logger.warning(
                    "File validation disabled but n_objects not provided",
                    table=self.ctx.db_class.__name__,
                    path=path,
                )
                raise ValueError("When validate_file=False, 'n_objects' must be provided")
            return n_objects

        # Validate path and file
        fullpath = self._validate_path_security(path)
        n_objects = await self.validate_data_for_path(fullpath, dataset)

        # Check against user-provided value if present
        user_n_objects = extra_kwargs.get("n_objects")
        if user_n_objects is not None and user_n_objects != n_objects:
            logger.warning(
                "Provided n_objects doesn't match file content",
                table=self.ctx.db_class.__name__,
                path=path,
                provided=user_n_objects,
                actual=n_objects,
            )

        return n_objects

    async def validate_data_for_path(
        self,
        path: Path,
        dataset: Dataset | None = None,
    ) -> int:
        """Validate that data file exists and can be read.

        This method performs synchronous I/O in an executor to avoid
        blocking the event loop.

        Parameters
        ----------
        path
            Absolute path to the estimates file
        dataset
            Dataset object for future validation (currently unused
            but reserved for validating data matches catalog schema)

        Returns
        -------
        int
            Number of objects in the estimates

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file cannot be read or has invalid format

        Notes
        -----
        Future enhancement: Use catalog_tag to validate that the data
        columns match the expected schema for this catalog.
        """
        # Reserved for future use: validate data matches catalog_tag schema
        _ = dataset

        # Check file exists
        if not path.exists():
            logger.error(
                "Estimates file not found",
                table=self.ctx.db_class.__name__,
                path=str(path),
            )
            raise FileNotFoundError(f"Estimates file {path} not found")

        # Read file in executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            n_objects = await loop.run_in_executor(None, qp.data_length, str(path))
        except OSError as exc:
            # File system errors
            logger.error(
                "Failed to read estimates file",
                table=self.ctx.db_class.__name__,
                path=str(path),
                error=str(exc),
                error_type="io_error",
            )
            raise ValueError(f"Could not read data from {path}: {exc}") from exc
        except ValueError as exc:
            # Data format errors
            logger.error(
                "Invalid data format in estimates file",
                table=self.ctx.db_class.__name__,
                path=str(path),
                error=str(exc),
                error_type="format_error",
            )
            raise ValueError(f"Invalid data format in {path}: {exc}") from exc
        except Exception as exc:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(
                "Unexpected error reading estimates file",
                table=self.ctx.db_class.__name__,
                path=str(path),
            )
            raise ValueError(f"Unexpected error reading {path}: {exc}") from exc

        logger.debug(
            "Estimates file validated",
            table=self.ctx.db_class.__name__,
            path=str(path),
            n_objects=n_objects,
        )

        return n_objects


# Module-level singleton
estimates_creator: EstimatesCreate = EstimatesCreate(RowCreateContext.from_db_class(Estimates))
