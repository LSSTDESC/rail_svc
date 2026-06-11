"""Dataset operations.

This module provides operations for creating Dataset records with automatic
catalog tag resolution and file validation.

Notes
-----
File validation includes:
- Path traversal protection
- File existence check
- Data format validation
- Object count extraction
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tables_io
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db, db_funcs, models
from ..rail_funcs.catalog_funcs import read_multi_catalog_slice, read_single_catalog_slice
from .base import FileValidatedOperations, TableContext
from .dataset_assoc import dataset_assoc

logger = logging.getLogger(__name__)

__all__ = ["DatasetOperations", "dataset"]


class DatasetOperations(FileValidatedOperations[db.Dataset, models.Dataset, models.DatasetCreate]):
    """Create operations for Dataset table.

    Handles automatic lookup of catalog_tag by either ID or name,
    with optional file validation and object counting.
    """

    async def get_create_kwargs(
        self,
        session: AsyncSession,
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
            db.CatalogTag,
            session,
            catalog_tag_id,
            catalog_tag_name,
            need_object=validate_file and path is not None,
        )

        # 2. Process path and determine n_objects
        n_objects = await self._process_path(
            path, catalog_tag_obj, validate_file=validate_file, extra_kwargs=extra_kwargs
        )

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

    def get_file_length(self, path: Path) -> int:
        """Extract number of objects from hdf5 file.

        Parameters
        ----------
        path : Path
            Absolute path to the dataset file

        Returns
        -------
        int
            Number of objects in file

        """
        return tables_io.hdf5.get_input_data_length(str(path))

    def get_subdirectory(self) -> str:
        """Get the subdirectory to store files in"""
        return "datasets"

    async def read_slice(
        self,
        session: AsyncSession,
        row: int,
        the_slice: slice | int | None = None,
    ) -> dict[str, np.ndarray]:

        the_dataset = await self.get_row(session, row)
        the_compontent_paths: dict[str, str | Path] = {}
        if the_dataset.is_collection:
            dataset_assocs = await dataset_assoc.find_by(session, matched_dataset_id=the_dataset.id_)
            for dataset_assoc_ in dataset_assocs:
                component_dataset = await dataset.get_row(session, row=dataset_assoc_.component_dataset_id)
                the_compontent_paths[component_dataset.name] = component_dataset.path
            return read_multi_catalog_slice(the_dataset.path, the_compontent_paths, the_slice)
        return read_single_catalog_slice(the_dataset.path, the_slice)


# Module-level singleton
dataset: DatasetOperations = DatasetOperations(TableContext.from_db_class(db.Dataset))
