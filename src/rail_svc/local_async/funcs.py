from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db_oper, models
from ..db.session import get_session
from ..rail_funcs.estimation_funcs import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper


def with_transaction(func: Callable) -> Callable:
    """Decorator that wraps a function with session transaction management."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Callable:
        async with get_session() as session:
            async with session.begin():
                return await func(session, *args, **kwargs)

    return wrapper


@with_transaction
async def build_pdf_estimation_wrapper(session: AsyncSession, estimator_id: int) -> CatEstimatorPdfWrapper:
    return await db_oper.estimation_funcs.build_pdf_estimation_wrapper(session, estimator_id)


@with_transaction
async def build_ensemble_estimation_wrapper(
    session: AsyncSession, estimator_id: int
) -> CatEstimatorEnsembleWrapper:
    return await db_oper.estimation_funcs.build_ensemble_estimation_wrapper(session, estimator_id)


@with_transaction
async def estimate_pdf(session: AsyncSession, estimator_id: int, dataset_id: int, row: int) -> qp.Ensemble:
    return await db_oper.estimation_funcs.estimate_pdf(session, estimator_id, dataset_id, row)


@with_transaction
async def estimate_ensemble(
    session: AsyncSession, estimator_id: int, dataset_id: int, output_file_path: str | Path
) -> Path:
    return await db_oper.estimation_funcs.estimate_ensemble(
        session, estimator_id, dataset_id, output_file_path
    )


@with_transaction
async def load_catalog_yaml(
    session: AsyncSession, catalog_yaml: Path, filter_dir: Path | None = None
) -> tuple[list[models.Band], list[models.CatalogTag], list[models.CatalogBandAssoc]]:
    db_bands, db_catalog_tags, db_models = await db_oper.catalog_funcs.load_catalog_yaml(
        session, catalog_yaml, filter_dir
    )
    return (
        db_oper.band.to_pydantic_list(db_bands),
        db_oper.catalog_tag.to_pydantic_list(db_catalog_tags),
        db_oper.catalog_band_assoc.to_pydantic_list(db_models),
    )


@with_transaction
async def get_catalog_row(session: AsyncSession, dataset_id: int, row: int) -> dict[str, np.ndarray]:
    return await db_oper.catalog_funcs.get_catalog_row(session, dataset_id, row)


@with_transaction
async def get_estimates_row(session: AsyncSession, estimates_id: int, row: int) -> dict[str, np.ndarray]:
    return await db_oper.catalog_funcs.get_estimates_row(session, estimates_id, row)


@with_transaction
async def get_dataset_and_estimates(
    session: AsyncSession, dataset_id: int
) -> tuple[models.Dataset, list[models.Estimates]]:
    db_dataset, db_estimates = await db_oper.catalog_funcs.get_dataset_and_estimates(session, dataset_id)
    return (
        db_oper.dataset.to_pydantic(db_dataset),
        db_oper.estimates.to_pydantic_list(db_estimates),
    )


@with_transaction
async def get_data_and_estimates_data(
    session: AsyncSession, dataset_id: int, row: int
) -> tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]:
    return await db_oper.catalog_funcs.get_data_and_estimates_data(session, dataset_id, row)


@with_transaction
async def create_matched_dataset(
    session: AsyncSession,
    matched_dataset_name: str,
    catalog_tag_name: str,
    component_dataset_names: list[str],
    path: str | None,
    n_objects: int,
) -> tuple[models.Dataset, list[models.DatasetAssoc]]:
    db_matched_dataset, db_dataset_assocs = await db_oper.catalog_funcs.create_matched_dataset(
        session,
        matched_dataset_name=matched_dataset_name,
        catalog_tag_name=catalog_tag_name,
        component_dataset_names=component_dataset_names,
        path=path,
        n_objects=n_objects,
    )
    return (
        db_oper.dataset.to_pydantic(db_matched_dataset),
        db_oper.dataset_assoc.to_pydantic_list(db_dataset_assocs),
    )


@with_transaction
async def build_cat_estimator_pdf_wrappers_for_dataset(
    session: AsyncSession, dataset_id: int
) -> list[CatEstimatorPdfWrapper]:
    return await db_oper.estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(session, dataset_id)


@with_transaction
async def build_cat_estimator_ensemble_wrappers_for_dataset(
    session: AsyncSession, dataset_id: int
) -> list[CatEstimatorEnsembleWrapper]:
    return await db_oper.estimation_funcs.build_cat_estimator_ensemble_wrappers_for_dataset(
        session, dataset_id
    )
