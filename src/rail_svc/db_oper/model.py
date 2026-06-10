"""Model operations.

This module provides operations for creating Model records with automatic
lookup of associated Algorithm and CatalogTag by ID or name, plus optional
model file validation.

"""

import asyncio
import logging
from pathlib import Path
from typing import Any, override

import anyio
from rail.core.model import Model as RailModel
from sqlalchemy.ext.asyncio import AsyncSession

from ..common import LoadType, handle_file
from .. import db, db_funcs, models
from .base import TableContext, TableOperations

logger = logging.getLogger(__name__)

__all__ = ["ModelOperations", "model"]


# Mapping from Informer class names to Estimator class names
# This is only needed for cases that don't follow the
# XxxInformer -> XxxEstimator pattern
INFORMER_TO_ESTIMATOR_MAP: dict[str, str] = {"dummy": "dummy"}


class ModelOperations(TableOperations[db.Model, models.Model, models.ModelCreate]):
    """Create operations for Model table.

    Handles automatic lookup of Algorithm and CatalogTag by either ID or name,
    with optional model file validation.
    """

    @override
    async def get_create_kwargs(
        self,
        session: AsyncSession,
        name: str | None = None,
        path: str | None = None,
        algo_id: int | None = None,
        algo_name: str | None = None,
        catalog_tag_id: int | None = None,
        catalog_tag_name: str | None = None,
        validate_file: bool = True,
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating a Model instance.

        Resolves Algorithm and CatalogTag foreign keys by looking up by name
        if IDs are not provided. Optionally validates the model file.

        Parameters
        ----------
        session
            Database session
        name
            Model name (required)
        path
            Path to model file relative to archive directory (required)
        algo_id
            Algorithm ID (provide this OR algo_name)
        algo_name
            Algorithm name (provide this OR algo_id)
        catalog_tag_id
            CatalogTag ID (provide this OR catalog_tag_name)
        catalog_tag_name
            CatalogTag name (provide this OR catalog_tag_id)
        validate_file
            Whether to validate model file exists and is compatible with
            the algorithm and catalog tag. If True, the model file will be
            loaded and checked.
        **extra_kwargs
            Additional fields to pass through to Model model

        Returns
        -------
        dict[str, Any]
            Validated kwargs ready for Model creation, including resolved
            foreign keys and any extra fields

        Raises
        ------
        ValueError
            If required parameters missing, if neither ID nor name provided
            for algorithm or catalog_tag, if model file validation fails,
            or if path escapes archive directory
        FileNotFoundError
            If model file doesn't exist when validate_file=True

        Examples
        --------
        With validation by names:

        >>> kwargs = await creator.get_create_kwargs(
        ...     session,
        ...     name="RF_Model",
        ...     path="models/rf_v1.pkl",
        ...     algo_name="RandomForestEstimator",
        ...     catalog_tag_name="SDSS_DR16",
        ...     validate_file=True
        ... )

        Without validation by IDs:

        >>> kwargs = await creator.get_create_kwargs(
        ...     session,
        ...     name="Quick_Model",
        ...     path="models/quick.pkl",
        ...     algo_id=123,
        ...     catalog_tag_id=456,
        ...     validate_file=False
        ... )
        """
        # Resolve algorithm foreign key
        algo_id, algo_obj = await db_funcs.read.lookup_by_id_or_name(
            db.Algorithm, session, algo_id, algo_name, need_object=validate_file
        )

        # Resolve catalog_tag foreign key
        catalog_tag_id, catalog_tag_obj = await db_funcs.read.lookup_by_id_or_name(
            db.CatalogTag, session, catalog_tag_id, catalog_tag_name, need_object=validate_file
        )

        assert path

        # Validate file if requested
        if validate_file:  # pragma: no cover
            assert algo_obj
            assert catalog_tag_obj

            fullpath = self._validate_path_security(path)
            await self.validate_model(fullpath, algo_obj, catalog_tag_obj)

        # Build final kwargs, excluding processed parameters
        processed_fields = {
            "name",
            "path",
            "algo_id",
            "algo_name",
            "catalog_tag_id",
            "catalog_tag_name",
            "validate_file",
        }
        remaining_kwargs = {k: v for k, v in extra_kwargs.items() if k not in processed_fields}

        return {
            "name": name,
            "path": path,
            "algo_id": algo_id,
            "catalog_tag_id": catalog_tag_id,
            **remaining_kwargs,
        }

    async def validate_model(
        self,
        path: Path,
        algo: db.Algorithm,
        catalog_tag: db.CatalogTag,
    ) -> None:
        """Validate that the model is appropriate for the Algorithm and CatalogTag.

        Performs I/O in executor to avoid blocking the event loop.

        Parameters
        ----------
        path
            Absolute path to the model file
        algo
            Algorithm object to validate against
        catalog_tag
            CatalogTag object to validate against

        Raises
        ------
        FileNotFoundError
            If the model file doesn't exist
        ValueError
            If the model doesn't match the algorithm or catalog tag,
            or if the file cannot be read

        Notes
        -----
        This method performs synchronous file I/O in an executor to
        avoid blocking the async event loop.
        """
        async_path = anyio.Path(path)
        if not await async_path.exists():
            logger.error(
                "Model file not found:",
            )
            raise FileNotFoundError(f"Model file {path} not found")

        # Read model in executor (blocking I/O)
        loop = asyncio.get_event_loop()

        try:
            the_model = await loop.run_in_executor(None, RailModel.read, str(path))
        except Exception as exc:
            logger.error(
                "Failed to read model file",
            )
            raise ValueError(f"Could not read model from {path}: {exc}") from exc

        # Validate catalog tag
        if the_model.catalog_tag:
            if the_model.catalog_tag != catalog_tag.name:
                logger.error(
                    "CatalogTag mismatch {path} {the_model.catalog_tag} {catalog_tag.name}",
                )
                raise ValueError(
                    f"CatalogTag mismatch: model has '{the_model.catalog_tag}' "
                    f"but expected '{catalog_tag.name}'"
                )

        # Validate algorithm
        if the_model.creation_class_name:
            expected_estimator_class = self._convert_informer_to_estimator(the_model.creation_class_name)

            if algo.class_name != expected_estimator_class:
                logger.error(
                    "Algorithm class mismatch",
                )
                raise ValueError(
                    f"Algorithm mismatch: model expects '{expected_estimator_class}' "
                    f"but got '{algo.class_name}'"
                )

        logger.info(
            "Model validation successful {path}",
        )

    def _convert_informer_to_estimator(self, informer_class_name: str) -> str:
        """Convert Informer class name to Estimator class name.

        Uses a mapping to convert between the two naming conventions.
        If the class name follows the pattern XxxInformer, it will be
        converted to XxxEstimator. Otherwise, uses the explicit mapping.

        Parameters
        ----------
        informer_class_name
            Informer class name (e.g., "RandomForestInformer")

        Returns
        -------
        str
            Estimator class name (e.g., "RandomForestEstimator")

        Raises
        ------
        ValueError
            If class name doesn't follow expected pattern and is not
            in the mapping

        Examples
        --------
        >>> creator._convert_informer_to_estimator("RandomForestInformer")
        'RandomForestEstimator'

        >>> creator._convert_informer_to_estimator("CustomInformer")
        'CustomEstimator'
        """
        # Try explicit mapping first
        if informer_class_name in INFORMER_TO_ESTIMATOR_MAP:
            return INFORMER_TO_ESTIMATOR_MAP[informer_class_name]

        # Fall back to pattern-based conversion
        if informer_class_name.endswith("Informer"):
            estimator_class_name = informer_class_name[:-8] + "Estimator"
            return estimator_class_name

        # No conversion possible
        logger.warning(
            "Unknown Informer class name pattern {informer_class_name}",
        )
        raise ValueError(
            f"Cannot convert Informer class name '{informer_class_name}' to Estimator. "
            f"Class name should end with 'Informer' or be in INFORMER_TO_ESTIMATOR_MAP."
        )

    async def load(
        self,
        name: str,
        orig_path: Path | str,
        algo_name: str,
        catalog_tag_name: str,
        load_type: LoadType = LoadType.in_place,
        *,
        validate_file: bool = True,
    ) -> models.Model:
        """Load a model file into the system.

        Creates a Model database record and handles the model file according to
        the specified load type (in-place, symlink, or copy). Automatically looks
        up the Algorithm and CatalogTag by name and optionally validates that the
        model file is compatible with them BEFORE any file operations or database
        modifications occur.

        Parameters
        ----------
        name
            Name for the model record
        orig_path
            Path to the original model file (can be absolute or relative)
        algo_name
            Name of the Algorithm this model uses
        catalog_tag_name
            Name of the CatalogTag this model is trained on
        load_type
            How to handle the file:
            - LoadType.in_place: Use file at its current location
            - LoadType.link: Create symbolic link in archive
            - LoadType.copy: Copy file to archive
        validate_file
            Whether to validate the model file by loading it and checking
            compatibility with the algorithm and catalog tag before any
            file operations or database changes

        Returns
        -------
        models.Model
            The created Model pydantic model

        Raises
        ------
        FileNotFoundError
            If the original file doesn't exist
        ValueError
            If the model validation fails (when validate_file=True), or if
            the Algorithm or CatalogTag names are not found in the database
        PermissionError
            If there are insufficient permissions for file operations
        OSError
            If there are filesystem errors during copy/link operations

        Examples
        --------
        Load a model by copying to archive with validation:

        >>> model_obj = await model.load(
        ...     "my_rf_model",
        ...     "/data/models/random_forest_v1.pkl",
        ...     "RandomForestEstimator",
        ...     "SDSS_DR16",
        ...     load_type=LoadType.copy,
        ...     validate_file=True
        ... )
        >>> print(f"Created model {model_obj.id} at {model_obj.path}")
        Created model 123 at models/my_rf_model_random_forest_v1.pkl

        Load a model in-place without validation:

        >>> model_obj = await model.load(
        ...     "quick_model",
        ...     "/scratch/temp_model.pkl",
        ...     "SimpleEstimator",
        ...     "test_catalog",
        ...     load_type=LoadType.in_place,
        ...     validate_file=False
        ... )

        Load a model with a symbolic link:

        >>> model_obj = await model.load(
        ...     "shared_model",
        ...     "/shared/models/production.pkl",
        ...     "DeepLearningEstimator",
        ...     "LSST_Y1",
        ...     load_type=LoadType.link
        ... )

        Notes
        -----
        - Validation occurs BEFORE any file operations (copy/link) or database
          records are created, so invalid models will not pollute the filesystem
          or database
        - When using LoadType.copy or LoadType.link, the file is placed in the
          'models/' subdirectory of the archive with the pattern:
          'models/{name}_{basename}'
        - When validate_file=True, the model file is loaded using RailModel.read()
          and checked against the Algorithm and CatalogTag
        - The function manages its own database transaction
        """
        # Validate FIRST, before any file operations or database changes
        if validate_file:
            # Need to look up Algorithm and CatalogTag for validation
            async with get_session() as session:
                # Look up algorithm
                algo_id, algo_obj = await db_funcs.read.lookup_by_id_or_name(
                    db.Algorithm, session, None, algo_name, need_object=True
                )
                if algo_obj is None:
                    raise ValueError(f"Algorithm '{algo_name}' not found in database")

                # Look up catalog tag
                catalog_tag_id, catalog_tag_obj = await db_funcs.read.lookup_by_id_or_name(
                    db.CatalogTag, session, None, catalog_tag_name, need_object=True
                )
                if catalog_tag_obj is None:
                    raise ValueError(f"CatalogTag '{catalog_tag_name}' not found in database")

                # Validate the original model file before any operations
                validation_path = Path(orig_path).resolve()
                logger.info(f"Validating model file before loading: {validation_path}")
                await self.validate_model(validation_path, algo_obj, catalog_tag_obj)
                logger.info(f"Model validation successful, proceeding with load")

        # Generate archive path based on original filename
        basename = os.path.basename(orig_path)
        archive_path = Path("models") / f"{name}_{basename}"

        # Handle the file according to load_type (only after validation passes)
        output_path = handle_file(orig_path, archive_path, load_type)

        # Create the database record (validation passed or was skipped)
        async with get_session() as session:
            async with session.begin():
                new_model = await self.create_row(
                    session,
                    name=name,
                    path=str(output_path),
                    algo_name=algo_name,
                    catalog_tag_name=catalog_tag_name,
                    validate_file=False,  # Already validated above
                )
                return self.to_pydantic(new_model)    

# Module-level singleton
model: ModelOperations = ModelOperations(TableContext.from_db_class(db.Model))
