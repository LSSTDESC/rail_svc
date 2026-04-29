"""Dataset creation operations.

This module provides operations for creating Dataset records with automatic
catalog tag resolution and file validation.

Examples
--------
Create with file validation:

>>> from myapp.operations.dataset import dataset_creator
>>>
>>> async with session.begin():
...     dataset = await dataset_creator.create_row(
...         session,
...         path="catalogs/sdss_dr16.h5",
...         catalog_tag_name="SDSS_DR16",
...         validate_file=True
...     )

Create without validation:

>>> async with session.begin():
...     dataset = await dataset_creator.create_row(
...         session,
...         path="catalogs/custom.h5",
...         catalog_tag_id=123,
...         n_objects=1000,
...         validate_file=False
...     )

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
import tables_io

from .. import db_funcs
from ..db import CatalogTag, Dataset
from .row_create import RowCreateBase, RowCreateContext

logger = logging.getLogger(__name__)

__all__ = ["DatasetCreate", "dataset_creator"]


class DatasetCreate(RowCreateBase[Dataset]):
    """Create operations for Dataset table.

    Handles automatic lookup of catalog_tag by either ID or name,
    with optional file validation and object counting.
    """

    async def get_create_kwargs(
        self,
        session: async_scoped_session,
        *,
        path: str | None = None,
        catalog_tag_id: int | None = None,
        catalog_tag_name: str | None = None,
        validate_file: bool = True,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating a Dataset instance.

        Resolves catalog_tag foreign key by ID or name, validates file
        path security, and optionally validates file content.

        Parameters
        ----------
        session
            Database session
        path
            Path to dataset file relative to archive directory.
            If None, n_objects must be provided in extra_kwargs.
        catalog_tag_id
            ID of the catalog tag (provide this OR catalog_tag_name)
        catalog_tag_name
            Name of the catalog tag (provide this OR catalog_tag_id)
        validate_file
            Whether to validate file exists and count objects.
            If False, n_objects must be provided in extra_kwargs.
        **extra_kwargs
            Additional fields to pass through to model.
            May include 'n_objects' if validate_file=False.

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for Dataset creation, including:
            - path: str | None
            - catalog_tag_id: int
            - n_objects: int
            - any fields from extra_kwargs

        Raises
        ------
        ValueError
            If neither catalog_tag_id nor catalog_tag_name provided,
            if catalog_tag not found,
            if path escapes archive directory,
            if file validation fails,
            or if n_objects not provided when validate_file=False
        FileNotFoundError
            If dataset file doesn't exist when validate_file=True

        Examples
        --------
        With file validation:

        >>> kwargs = await creator.get_create_kwargs(
        ...     session,
        ...     path="catalogs/sdss.h5",
        ...     catalog_tag_name="SDSS_DR16",
        ...     validate_file=True
        ... )
        >>> # kwargs includes n_objects from file

        Without validation:

        >>> kwargs = await creator.get_create_kwargs(
        ...     session,
        ...     path="catalogs/custom.h5",
        ...     catalog_tag_id=123,
        ...     n_objects=5000,
        ...     validate_file=False
        ... )
        """
        # 1. Resolve catalog_tag foreign key
        catalog_tag_id, catalog_tag_obj = await db_funcs.read.lookup_by_id_or_name(
            CatalogTag, session, catalog_tag_id, catalog_tag_name,
            need_object=validate_file and path is not None
        )

        # 2. Process path and determine n_objects
        n_objects = await self._process_path(path, catalog_tag_obj, validate_file, extra_kwargs)

        # 3. Build final kwargs
        result = {
            "path": path,
            "catalog_tag_id": catalog_tag_id,
            "n_objects": n_objects,
        }

        # Add extra kwargs, but remove n_objects if present (we set it above)
        extra_kwargs_filtered = {k: v for k, v in extra_kwargs.items() if k != "n_objects"}
        result.update(extra_kwargs_filtered)

        return result

    async def _process_path(
        self,
        path: str | None,
        catalog_tag: CatalogTag | None,
        validate_file: bool,
        extra_kwargs: dict[str, Any],
    ) -> int:
        """Process dataset path and determine n_objects.

        Parameters
        ----------
        path
            Dataset file path (relative to archive)
        catalog_tag
            CatalogTag object for validation (may be None)
        validate_file
            Whether to validate file
        extra_kwargs
            Additional kwargs that may contain n_objects

        Returns
        -------
        int
            Number of objects in dataset

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
        n_objects = await self.validate_data_for_path(fullpath, catalog_tag)

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
        catalog_tag: CatalogTag | None = None,
    ) -> int:
        """Validate that data file exists and can be read.

        This method performs synchronous I/O in an executor to avoid
        blocking the event loop.

        Parameters
        ----------
        path
            Absolute path to the dataset file
        catalog_tag
            CatalogTag object for future validation (currently unused
            but reserved for validating data matches catalog schema)

        Returns
        -------
        int
            Number of objects in the dataset

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
        _ = catalog_tag

        # Check file exists
        if not path.exists():
            logger.error(
                "Dataset file not found",
                table=self.ctx.db_class.__name__,
                path=str(path),
            )
            raise FileNotFoundError(f"Dataset file {path} not found")

        # Read file in executor to avoid blocking
        loop = asyncio.get_event_loop()
        try:
            n_objects = await loop.run_in_executor(None, tables_io.hdf5.get_input_data_length, str(path))
        except OSError as exc:
            # File system errors
            logger.error(
                "Failed to read dataset file",
                table=self.ctx.db_class.__name__,
                path=str(path),
                error=str(exc),
                error_type="io_error",
            )
            raise ValueError(f"Could not read data from {path}: {exc}") from exc
        except ValueError as exc:
            # Data format errors
            logger.error(
                "Invalid data format in dataset file",
                table=self.ctx.db_class.__name__,
                path=str(path),
                error=str(exc),
                error_type="format_error",
            )
            raise ValueError(f"Invalid data format in {path}: {exc}") from exc
        except Exception as exc:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(
                "Unexpected error reading dataset file",
                table=self.ctx.db_class.__name__,
                path=str(path),
            )
            raise ValueError(f"Unexpected error reading {path}: {exc}") from exc

        logger.debug(
            "Dataset file validated",
            table=self.ctx.db_class.__name__,
            path=str(path),
            n_objects=n_objects,
        )

        return n_objects


# Module-level singleton
dataset_creator: DatasetCreate = DatasetCreate(RowCreateContext.from_db_class(Dataset))
