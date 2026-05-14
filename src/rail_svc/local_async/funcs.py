from pathlib import Path

import numpy as np
import qp

from .. import db_oper, models
from ..db.session import get_session
from ..rail_funcs.estimation_funcs import (CatEstimatorEnsembleWrapper,
                                           CatEstimatorPdfWrapper)


async def build_pdf_estimation_wrapper(
    estimator_id: int,
) -> CatEstimatorPdfWrapper:
    async with get_session() as session:
        async with session.begin():
            return await db_oper.estimation_funcs.build_pdf_estimation_wrapper(session, estimator_id)


async def build_ensemble_estimation_wrapper(
    estimator_id: int,
) -> CatEstimatorEnsembleWrapper:
    async with get_session() as session:
        async with session.begin():
            return await db_oper.estimation_funcs.build_ensemble_estimation_wrapper(session, estimator_id)


async def estimate_pdf(
    estimator_id: int,
    dataset_id: int,
    row: int,
) -> qp.Ensemble:
    async with get_session() as session:
        async with session.begin():
            return await db_oper.estimation_funcs.estimate_pdf(session, estimator_id, dataset_id, row)


async def estimate_ensemble(
    estimator_id: int,
    dataset_id: int,
    output_file_path: str | Path,
) -> Path:
    async with get_session() as session:
        async with session.begin():
            return await db_oper.estimation_funcs.estimate_ensemble(
                session, estimator_id, dataset_id, output_file_path
            )


async def load_catalog_yaml(
    catalog_yaml: Path, filter_dir: Path | None = None
) -> tuple[list[models.Band], list[models.CatalogTag], list[models.CatalogBandAssoc]]:
    async with get_session() as session:
        async with session.begin():
            db_bands, db_catalog_tags, db_models = await db_oper.catalog_funcs.load_catalog_yaml(
                session, catalog_yaml, filter_dir
            )
            return (
                db_oper.band.to_pydantic_list(db_bands),
                db_oper.catalog_tag.to_pydantic_list(db_catalog_tags),
                db_oper.catalog_band_assoc.to_pydantic_list(db_models),
            )


async def get_catalog_row(
    dataset_id: int,
    row: int,
) -> dict[str, np.ndarray]:
    async with get_session() as session:
        async with session.begin():
            return await db_oper.catalog_funcs.get_catalog_row(session, dataset_id, row)


async def get_estimates_row(
    estimates_id: int,
    row: int,
) -> dict[str, np.ndarray]:
    async with get_session() as session:
        async with session.begin():
            return await db_oper.catalog_funcs.get_estimates_row(session, estimates_id, row)


async def get_dataset_and_estimates(
    dataset_id: int,
) -> tuple[models.Dataset, list[models.Estimates]]:
    async with get_session() as session:
        async with session.begin():
            db_dataset, db_estimates = await db_oper.catalog_funcs.get_dataset_and_estimates(
                session, dataset_id
            )
            return (
                db_oper.dataset.to_pydantic(db_dataset),
                db_oper.estimates.to_pydantic_list(db_estimates),
            )


async def create_matched_dataset(
    matched_dataset_name: str,
    component_dataset_names: list[str],
    path: str | None,
    n_objects: int,
) -> tuple[models.Dataset, list[models.DatasetAssoc]]:
    async with get_session() as session:
        async with session.begin():
            db_matched_dataset, db_dataset_assocs = await db_oper.catalog_funcs.create_matched_dataset(
                session,
                matched_dataset_name=matched_dataset_name,
                component_dataset_names=component_dataset_names,
                path=path,
                n_objects=n_objects,
            )
            return (
                db_oper.dataset.to_pydantic(db_matched_dataset),
                db_oper.dataset_assoc.to_pydantic_list(db_dataset_assocs),
            )
