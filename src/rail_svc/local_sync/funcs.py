import asyncio
from pathlib import Path

import numpy as np
import qp

from .. import local_async, models
from ..rail_funcs.estimation_funcs import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper


def build_pdf_estimation_wrapper(
    estimator_id: int,
) -> CatEstimatorPdfWrapper:
    return asyncio.run(local_async.funcs.build_pdf_estimation_wrapper(estimator_id))


def build_ensemble_estimation_wrapper(
    estimator_id: int,
) -> CatEstimatorEnsembleWrapper:
    return asyncio.run(local_async.funcs.build_ensemble_estimation_wrapper(estimator_id))


def estimate_pdf(
    estimator_id: int,
    dataset_id: int,
    row: int,
) -> qp.Ensemble:
    return asyncio.run(local_async.funcs.estimate_pdf(estimator_id, dataset_id, row))


def estimate_ensemble(
    estimator_id: int,
    dataset_id: int,
    output_file_path: str | Path,
) -> Path:
    return asyncio.run(local_async.funcs.estimate_ensemble(estimator_id, dataset_id, output_file_path))


def load_catalog_yaml(
    catalog_yaml: Path, filter_dir: Path | None = None
) -> tuple[list[models.Band], list[models.CatalogTag], list[models.CatalogBandAssoc]]:
    return asyncio.run(local_async.funcs.load_catalog_yaml(catalog_yaml, filter_dir))


def get_catalog_row(
    dataset_id: int,
    row: int,
) -> dict[str, np.ndarray]:
    return asyncio.run(local_async.funcs.get_catalog_row(dataset_id, row))


def get_estimates_row(
    estimates_id: int,
    row: int,
) -> dict[str, np.ndarray]:
    return asyncio.run(local_async.funcs.get_estimates_row(estimates_id, row))


def get_dataset_and_estimates(
    dataset_id: int,
) -> tuple[models.Dataset, list[models.Estimates]]:
    return asyncio.run(local_async.funcs.get_dataset_and_estimates(dataset_id))


def get_data_and_estimates_data(
    dataset_id: int,
    row: int,
) -> tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]:
    return asyncio.run(local_async.funcs.get_data_and_estimates_data(dataset_id, row))


def create_matched_dataset(
    matched_dataset_name: str,
    catalog_tag_name: str,
    component_dataset_names: list[str],
    path: str | None,
    n_objects: int,
) -> tuple[models.Dataset, list[models.DatasetAssoc]]:
    return asyncio.run(
        local_async.funcs.create_matched_dataset(
            matched_dataset_name,
            catalog_tag_name,
            component_dataset_names,
            path,
            n_objects,
        )
    )
