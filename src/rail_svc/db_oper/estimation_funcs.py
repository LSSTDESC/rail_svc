import logging
from pathlib import Path

import anyio
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db, rail_funcs
from macon.config import config as global_config
from ..rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper
from . import catalog_funcs
from .wrappers import build_pdf_estimation_wrapper, build_ensemble_estimation_wrapper
from .catalog_tag import catalog_tag
from .dataset import dataset
from .estimator import estimator
from .estimates import estimates
from .model import model

logger = logging.getLogger(__name__)


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
    logger.info(f"Estimating PDF for dataset {dataset_id}, row {row} using estimator {estimator_id}")

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
            f"Successfully created estimates file: {final_output_path} ({file_size / 1024 / 1024:.2f} MB)"
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


async def estimate_pdf_for_slice(
    session: AsyncSession,
    estimator_id: int,
    dataset_id: int,
    the_slice: slice | int | None,
    *,
    recompute_if_exists: bool = False,
) -> qp.Ensemble:

    existing = await estimates.find_by(session, estimator_id=estimator_id, dataset_id=dataset_id)
    if existing and not recompute_if_exists:
        return rail_funcs.catalog_funcs.read_estimates_slice(existing[0].path, the_slice)

    logger.info(f"Estimating PDF for dataset {dataset_id}, slice {the_slice} using estimator {estimator_id}")

    try:
        # Build the wrapper
        wrapper = await build_pdf_estimation_wrapper(session, estimator_id)
        logger.debug("PDF wrapper built successfully")

        # Get the catalog data
        catalog_data = await dataset.read_slice(session, dataset_id, the_slice)
        logger.debug(f"Retrieved catalog data with {len(catalog_data)} columns")

        # Run the estimation
        pdf_result = wrapper(catalog_data)
        logger.info(f"Successfully estimated PDF for slice {the_slice}")

        return pdf_result

    except (OSError, ValueError, FileNotFoundError):
        raise
    except Exception as uexc:
        logger.error(f"Failed to estimate PDF: {uexc}")
        raise


async def estimate_dataset(
    session: AsyncSession,
    estimator_id: int,
    dataset_id: int,
    *,
    raise_if_exists: bool = False,
) -> db.Estimates:

    existing = await estimates.find_by(session, estimator_id=estimator_id, dataset_id=dataset_id)
    if existing and raise_if_exists:
        raise ValueError(
            f"Estimates for dataset {dataset_id} and estimator_id {estimator_id} "
            f"already exist: {existing[0]}."
        )
    if existing:
        return existing[0]

    estimator_obj = await estimator.get_row(session, estimator_id)
    dataset_obj = await dataset.get_row(session, dataset_id)

    estimates_name = f"{dataset_obj.name}__{estimator_obj.name}"
    estimates_path = Path(global_config.storage.archive) / "estimates" / f"{estimates_name}.hdf5"

    await estimate_ensemble(session, estimator_id, dataset_id, output_file_path=estimates_path)
    output_estimates = await estimates.create_row(
        session,
        name=estimates_name,
        path=str(estimates_path),
        n_objects=dataset_obj.n_objects,
        estimator_name=estimator_obj.name,
        dataset_name=dataset_obj.name,
    )
    return output_estimates
