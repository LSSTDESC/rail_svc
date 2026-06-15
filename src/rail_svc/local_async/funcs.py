from pathlib import Path
from typing import Any

import numpy as np
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db_oper, models
from ..rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper
from .base import with_session, with_session_transaction, to_pydantic_list


@with_session
async def build_pdf_estimation_wrapper(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> CatEstimatorPdfWrapper:
    return await db_oper.wrappers.build_pdf_estimation_wrapper(session, *args, **kwargs)


@with_session
async def build_ensemble_estimation_wrapper(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> CatEstimatorEnsembleWrapper:
    return await db_oper.wrappers.build_ensemble_estimation_wrapper(session, *args, **kwargs)


@with_session
async def estimate_pdf(session: AsyncSession, *args: Any, **kwargs: Any) -> qp.Ensemble:
    return await db_oper.estimation_funcs.estimate_pdf(session, *args, **kwargs)


@with_session
async def estimate_ensemble(session: AsyncSession, *args: Any, **kwargs: Any) -> Path:
    return await db_oper.estimation_funcs.estimate_ensemble(session, *args, **kwargs)


@with_session
@to_pydantic_list
async def get_estimators_for_dataest(session: AsyncSession, *args: Any, **kwargs: Any) -> Any:
    return await db_oper.estimation_funcs.get_estimators_for_dataest(session, *args, **kwargs)


@with_session_transaction
async def load_catalog_yaml(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> tuple[list[models.Band], list[models.CatalogTag], list[models.CatalogBandAssoc]]:
    db_bands, db_catalog_tags, db_models = await db_oper.catalog_funcs.load_catalog_yaml(
        session, *args, **kwargs
    )
    return (
        db_oper.band.to_pydantic_list(db_bands),
        db_oper.catalog_tag.to_pydantic_list(db_catalog_tags),
        db_oper.catalog_band_assoc.to_pydantic_list(db_models),
    )


@with_session
async def get_catalog_row(session: AsyncSession, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    return await db_oper.catalog_funcs.get_catalog_row(session, *args, **kwargs)


@with_session
async def get_estimates_row(session: AsyncSession, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    return await db_oper.catalog_funcs.get_estimates_row(session, *args, **kwargs)


@with_session
async def get_dataset_and_estimates(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> tuple[models.Dataset, list[models.Estimates]]:
    db_dataset, db_estimates = await db_oper.catalog_funcs.get_dataset_and_estimates(session, *args, **kwargs)
    return (
        db_oper.dataset.to_pydantic(db_dataset),
        db_oper.estimates.to_pydantic_list(db_estimates),
    )


@with_session
async def get_data_and_estimates_data(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]:
    return await db_oper.catalog_funcs.get_data_and_estimates_data(session, *args, **kwargs)


@with_session_transaction
async def create_matched_dataset(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> tuple[models.Dataset, list[models.DatasetAssoc]]:
    db_matched_dataset, db_dataset_assocs = await db_oper.catalog_funcs.create_matched_dataset(
        session, *args, **kwargs
    )
    return (
        db_oper.dataset.to_pydantic(db_matched_dataset),
        db_oper.dataset_assoc.to_pydantic_list(db_dataset_assocs),
    )


@with_session
async def build_cat_estimator_pdf_wrappers_for_dataset(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> list[CatEstimatorPdfWrapper]:
    return await db_oper.estimation_funcs.build_cat_estimator_pdf_wrappers_for_dataset(
        session, *args, **kwargs
    )


@with_session
async def build_cat_estimator_ensemble_wrappers_for_dataset(
    session: AsyncSession, *args: Any, **kwargs: Any
) -> list[CatEstimatorEnsembleWrapper]:
    return await db_oper.estimation_funcs.build_cat_estimator_ensemble_wrappers_for_dataset(
        session, *args, **kwargs
    )


@with_session
async def estimate_pdf_for_slice(session: AsyncSession, *args: Any, **kwargs: Any) -> qp.Ensemble:
    return await db_oper.estimation_funcs.estimate_pdf_for_slice(session, *args, **kwargs)


@with_session_transaction
@to_pydantic_list
async def estimate_dataset(session: AsyncSession, *args: Any, **kwargs: Any) -> models.Estimates:
    return await db_oper.estimation_funcs.estimate_dataset(session, *args, **kwargs)  # type: ignore
