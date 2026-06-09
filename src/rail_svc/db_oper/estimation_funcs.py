import logging
from enum import Enum
from pathlib import Path
from typing import cast

import anyio
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db
from ..config import config as global_config
from ..rail_funcs.estimation_funcs import (CatEstimatorEnsembleWrapper,
                                           CatEstimatorPdfWrapper)
from . import catalog_funcs
from .algorithm import algorithm
from .catalog_tag import catalog_tag
from .dataset import dataset
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


async def estimate_pdf(
    session: AsyncSession,
    estimator_id: int,
    dataset_id: int,
    row: int,
) -> qp.Ensemble:
    """
    Estimate the photo-z PDF for a single catalog object.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    estimator_id : int
        ID of the estimator to use for estimation.
    dataset_id : int
        ID of the dataset containing the object.
    row : int
        Zero-based row index of the object in the catalog.

    Returns
    -------
    qp.Ensemble
        Probability distribution (photo-z PDF) for the requested object.

    Raises
    ------
    ValueError
        If any ID or row index is invalid.
    FileNotFoundError
        If model or catalog files don't exist.
    IOError
        If files cannot be read.

    Examples
    --------
    >>> pdf = await estimate_pdf(session, estimator_id=1, dataset_id=5, row=42)
    >>> z_mode = pdf.mode()[0]
    >>> print(f"Photo-z mode: {z_mode:.3f}")

    Notes
    -----
    This function is suitable for interactive analysis or small batches.
    For processing entire catalogs, use estimate_ensemble instead.
    """
    logger.info(f"Estimating PDF for dataset {dataset_id}, row {row} " f"using estimator {estimator_id}")

    try:
        # Build the wrapper
        wrapper = await build_pdf_estimation_wrapper(session, estimator_id)
        logger.debug("PDF wrapper built successfully")

        # Get the catalog data
        catalog_data = await catalog_funcs.get_catalog_row(session, dataset_id, row)
        logger.debug(f"Retrieved catalog data with {len(catalog_data)} columns")

        # Run the estimation
        pdf_result = wrapper(catalog_data)
        logger.info(f"Successfully estimated PDF for row {row}")

        return pdf_result

    except (OSError, ValueError, FileNotFoundError):
        raise
    except Exception as uexc:
        logger.error(f"Failed to estimate PDF: {uexc}")
        raise


async def estimate_ensemble(
    session: AsyncSession,
    estimator_id: int,
    dataset_id: int,
    output_file_path: str | Path,
) -> Path:
    """
    Estimate photo-z PDFs for an entire catalog (batch processing).

    This function processes all objects in a catalog dataset, generating
    photo-z posterior distributions and saving them to an output file.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    estimator_id : int
        ID of the estimator to use for estimation.
    dataset_id : int
        ID of the dataset (catalog) to process.
    output_file_path : str or Path
        Path where the output estimates file should be written.
        Can be absolute or relative to the archive directory.

    Returns
    -------
    Path
        Absolute path to the created estimates file.

    Raises
    ------
    ValueError
        If any ID is invalid or paths are invalid.
    FileNotFoundError
        If model or catalog files don't exist.
    IOError
        If files cannot be read or written.

    Examples
    --------
    >>> output_path = await estimate_ensemble(
    ...     session,
    ...     estimator_id=1,
    ...     dataset_id=5,
    ...     output_file_path="estimates/lsst_dp0_photo_z.hdf5"
    ... )
    >>> print(f"Estimates saved to: {output_path}")

    Notes
    -----
    This function processes data in batches for memory efficiency.
    For large catalogs, this is much faster than calling estimate_pdf
    repeatedly.

    The output file format is qp (HDF5) containing photo-z posteriors
    for all objects in the input catalog.
    """
    # Convert to Path if necessary
    if not isinstance(output_file_path, Path):
        output_file_path = Path(output_file_path)

    logger.info(
        f"Estimating ensemble for dataset {dataset_id} "
        f"using estimator {estimator_id}, output: {output_file_path}"
    )

    try:
        # Build the wrapper
        wrapper = await build_ensemble_estimation_wrapper(session, estimator_id)
        logger.debug("Ensemble wrapper built successfully")

        # Get the dataset
        dataset_obj = await dataset.get_row(session, dataset_id)
        logger.debug(f"Found dataset: {dataset_obj.path}")

        # Get archive directory
        archive_dir = Path(await anyio.Path(global_config.storage.archive).absolute())

        # Construct input catalog path
        input_path = archive_dir / dataset_obj.path

        if not input_path.exists():
            logger.error(f"Dataset file not found: {input_path}")
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        # Determine output path
        # If output_file_path is absolute, use it directly
        # If relative, put it in archive directory
        if output_file_path.is_absolute():
            final_output_path = output_file_path
        else:
            final_output_path = archive_dir / output_file_path

        # Create parent directories if they don't exist
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if output file already exists
        if final_output_path.exists():  # pragma: no cover
            logger.warning(f"Output file already exists and will be overwritten: {final_output_path}")

        logger.info(f"Processing catalog: {input_path}")
        logger.info(f"Output will be written to: {final_output_path}")

        # Run the estimation
        _result = wrapper(input_path, final_output_path)

        # Verify output was created
        if not final_output_path.exists():
            logger.error(f"Estimation completed but output file not found: {final_output_path}")
            raise OSError(f"Output file was not created: {final_output_path}")

        file_size = final_output_path.stat().st_size
        logger.info(
            f"Successfully created estimates file: {final_output_path} " f"({file_size / 1024 / 1024:.2f} MB)"
        )

        return final_output_path

    except (OSError, ValueError, FileNotFoundError):
        raise
    except Exception as uexc:
        logger.error(f"Failed to estimate ensemble: {uexc}")
        raise


async def get_estimators_for_dataest(
    session: AsyncSession,
    dataset_id: int,
) -> list[db.Estimator]:

    all_estimators: list[db.Estimator] = []

    try:
        # Get the associationed dataset
        the_dataset = await dataset.get_row(session, dataset_id)

        # Get the associated catalog_tag
        the_catalog_tag = await catalog_tag.get_row(session, the_dataset.catalog_tag_id)

        # Get all models that use that catalog tag
        the_models = await model.find_by(session, catalog_tag_id=the_catalog_tag.id_)

        # For each model, get all the estimators
        for a_model in the_models:
            all_estimators += await estimator.find_by(session, model_id=a_model.id_)

        return all_estimators

    except ValueError:
        raise
    except Exception as uexc:
        logger.error(f"Failed to get estimators: {uexc}")
        raise


async def build_cat_estimator_pdf_wrappers_for_dataset(
    session: AsyncSession,
    dataset_id: int,
) -> list[CatEstimatorPdfWrapper]:
    the_estimators = await get_estimators_for_dataest(session, dataset_id)
    ret_list = []
    for estimator_ in the_estimators:
        try:
            ret_list.append(await build_pdf_estimation_wrapper(session, estimator_.id_))
        except Exception as exc:
            logger.warning(f"Failed to build estimator {estimator} because {exc}")

    return ret_list


async def build_cat_estimator_ensemble_wrappers_for_dataset(
    session: AsyncSession,
    dataset_id: int,
) -> list[CatEstimatorEnsembleWrapper]:
    the_estimators = await get_estimators_for_dataest(session, dataset_id)
    ret_list = []
    for estimator_ in the_estimators:
        try:
            ret_list.append(await build_ensemble_estimation_wrapper(session, estimator_.id_))
        except Exception as uexc:
            logger.warning(f"Failed to build estimator {estimator} because {uexc}")

    return ret_list
