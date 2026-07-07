import logging
from enum import Enum
from pathlib import Path
from typing import cast

import anyio
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db
from macon.config import config as global_config
from ..rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper
from .algorithm import algorithm
from .catalog_tag import catalog_tag
from .estimator import estimator
from .model import model

logger = logging.getLogger(__name__)


class WrapperType(Enum):
    """Types of estimation wrappers."""

    PDF = "pdf"
    ENSEMBLE = "ensemble"


async def _get_estimator_components(
    session: AsyncSession,
    estimator_id: int,
) -> tuple[db.Estimator, db.Model, db.Algorithm, db.CatalogTag]:
    """
    Fetch an estimator and all its related components.

    This function retrieves the estimator, its model, algorithm, and catalog tag.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    estimator_id : int
        ID of the estimator to fetch.

    Returns
    -------
    estimator_obj :
        The estimator record.
    model_obj :
        The associated model record.
    algo_obj :
        The algorithm used by the model.
    catalog_tag_obj :
        The catalog tag for the model.

    Raises
    ------
    ValueError
        If estimator_id is invalid or components are not found.
    """
    logger.debug(f"Fetching estimator {estimator_id} and related components")

    try:
        # Fetch estimator
        estimator_obj = await estimator.get_row(session, estimator_id)

        # Fetch model
        model_obj = await model.get_row(session, estimator_obj.model_id)

        # Fetch algorithm
        algo_obj = await algorithm.get_row(session, model_obj.algo_id)

        # Fetch catalog tag
        catalog_tag_obj = await catalog_tag.get_row(session, model_obj.catalog_tag_id)

        logger.debug(
            f"Successfully fetched estimator '{estimator_obj.name}' "
            f"(algo: {algo_obj.class_name}, catalog: {catalog_tag_obj.name})"
        )

        return (estimator_obj, model_obj, algo_obj, catalog_tag_obj)

    except Exception as e:
        logger.error(f"Failed to fetch estimator components: {e}")
        raise


async def _build_estimation_wrapper(
    session: AsyncSession,
    estimator_id: int,
    wrapper_type: WrapperType,
) -> CatEstimatorPdfWrapper | CatEstimatorEnsembleWrapper:
    """
    Build an estimation wrapper (PDF or Ensemble) for a given estimator.

    This is a unified function that handles both wrapper types to avoid
    code duplication.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    estimator_id : int
        ID of the estimator to build wrapper for.
    wrapper_type : WrapperType
        Type of wrapper to build (PDF or ENSEMBLE).

    Returns
    -------
    CatEstimatorPdfWrapper or CatEstimatorEnsembleWrapper
        The constructed estimation wrapper.

    Raises
    ------
    ValueError
        If inputs are invalid or required data is missing.
    FileNotFoundError
        If the model file doesn't exist.
    """
    logger.info(f"Building {wrapper_type.value} estimation wrapper for estimator {estimator_id}")

    try:
        # Fetch all required components
        estimator_obj, model_obj, algo_obj, catalog_tag_obj = await _get_estimator_components(
            session, estimator_id
        )

        # Get archive directory
        archive_dir = Path(await anyio.Path(global_config.storage.archive).absolute())

        # Construct model path
        model_path = archive_dir / model_obj.path

        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Get config parameters
        config_params = estimator_obj.config
        if config_params is None:
            config_params = {}

        logger.debug(
            f"Building wrapper: name={estimator_obj.name}, "
            f"class={algo_obj.class_name}, model={model_path}, "
            f"catalog={catalog_tag_obj.name}"
        )

        wrapper: CatEstimatorPdfWrapper | CatEstimatorEnsembleWrapper | None = None

        # Build the appropriate wrapper type
        if wrapper_type == WrapperType.PDF:
            wrapper = CatEstimatorPdfWrapper.build_wrapper(
                estimator_obj.name,
                algo_obj.class_name,
                model_path,
                catalog_tag_obj.name,
                **config_params,
            )
        elif wrapper_type == WrapperType.ENSEMBLE:
            wrapper = CatEstimatorEnsembleWrapper.build_wrapper(
                estimator_obj.name,
                algo_obj.class_name,
                model_path,
                catalog_tag_obj.name,
                **config_params,
            )
        else:  # pragma: no cover
            raise ValueError(f"Unknown wrapper type: {wrapper_type}")
        assert wrapper is not None

        logger.info(f"Successfully built {wrapper_type.value} wrapper for estimator {estimator_id}")
        return wrapper

    except (ValueError, FileNotFoundError):
        raise
    except Exception as uexc:
        logger.error(f"Failed to build {wrapper_type.value} wrapper: {uexc}")
        raise


async def build_pdf_estimation_wrapper(
    session: AsyncSession,
    estimator_id: int,
) -> CatEstimatorPdfWrapper:
    """
    Build a PDF estimation wrapper for single-object photo-z estimation.

    This wrapper is used for estimating photo-z PDFs for individual objects
    or small batches.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    estimator_id : int
        ID of the estimator to use.

    Returns
    -------
    CatEstimatorPdfWrapper
        Configured wrapper ready for estimation.

    Raises
    ------
    ValueError
        If estimator_id is invalid or required data is missing.
    FileNotFoundError
        If the model file doesn't exist.

    Examples
    --------
    >>> wrapper = await build_pdf_estimation_wrapper(session, estimator_id=1)
    >>> pdf = wrapper(catalog_data)
    """
    return cast(
        CatEstimatorPdfWrapper, await _build_estimation_wrapper(session, estimator_id, WrapperType.PDF)
    )


async def build_ensemble_estimation_wrapper(
    session: AsyncSession,
    estimator_id: int,
) -> CatEstimatorEnsembleWrapper:
    """
    Build an ensemble estimation wrapper for batch photo-z estimation.

    This wrapper is used for estimating photo-z PDFs for entire catalogs,
    processing data in batches for efficiency.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    estimator_id : int
        ID of the estimator to use.

    Returns
    -------
    CatEstimatorEnsembleWrapper
        Configured wrapper ready for batch estimation.

    Raises
    ------
    ValueError
        If estimator_id is invalid or required data is missing.
    FileNotFoundError
        If the model file doesn't exist.

    Examples
    --------
    >>> wrapper = await build_ensemble_estimation_wrapper(session, estimator_id=1)
    >>> results = wrapper(input_catalog_path, output_path)
    """
    return cast(
        CatEstimatorEnsembleWrapper,
        await _build_estimation_wrapper(session, estimator_id, WrapperType.ENSEMBLE),
    )
