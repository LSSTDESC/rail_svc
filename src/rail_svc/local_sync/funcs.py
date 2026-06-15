import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import qp

from .. import local_async, models, db
from ..rail_funcs.wrappers import CatEstimatorEnsembleWrapper, CatEstimatorPdfWrapper


def build_pdf_estimation_wrapper(*args: Any, **kwargs: Any) -> CatEstimatorPdfWrapper:
    return asyncio.run(local_async.funcs.build_pdf_estimation_wrapper(*args, **kwargs))


def build_ensemble_estimation_wrapper(*args: Any, **kwargs: Any) -> CatEstimatorEnsembleWrapper:
    return asyncio.run(local_async.funcs.build_ensemble_estimation_wrapper(*args, **kwargs))


def estimate_pdf(*args: Any, **kwargs: Any) -> qp.Ensemble:
    return asyncio.run(local_async.funcs.estimate_pdf(*args, **kwargs))


def estimate_ensemble(*args: Any, **kwargs: Any) -> Path:
    return asyncio.run(local_async.funcs.estimate_ensemble(*args, **kwargs))


def get_estimators_for_dataest(*args: Any, **kwargs: Any) -> list[db.Estimator]:
    return asyncio.run(local_async.funcs.get_estimators_for_dataest(*args, **kwargs))


def load_catalog_yaml(
    *args: Any, **kwargs: Any
) -> tuple[list[models.Band], list[models.CatalogTag], list[models.CatalogBandAssoc]]:
    return asyncio.run(local_async.funcs.load_catalog_yaml(*args, **kwargs))


def get_catalog_row(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    return asyncio.run(local_async.funcs.get_catalog_row(*args, **kwargs))


def get_estimates_row(*args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
    return asyncio.run(local_async.funcs.get_estimates_row(*args, **kwargs))


def get_dataset_and_estimates(*args: Any, **kwargs: Any) -> tuple[models.Dataset, list[models.Estimates]]:
    return asyncio.run(local_async.funcs.get_dataset_and_estimates(*args, **kwargs))


def get_data_and_estimates_data(
    *args: Any, **kwargs: Any
) -> tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]:
    return asyncio.run(local_async.funcs.get_data_and_estimates_data(*args, **kwargs))


def create_matched_dataset(*args: Any, **kwargs: Any) -> tuple[models.Dataset, list[models.DatasetAssoc]]:
    return asyncio.run(local_async.funcs.create_matched_dataset(*args, **kwargs))


def build_cat_estimator_pdf_wrappers_for_dataset(*args: Any, **kwargs: Any) -> list[CatEstimatorPdfWrapper]:
    return asyncio.run(local_async.funcs.build_cat_estimator_pdf_wrappers_for_dataset(*args, **kwargs))


def build_cat_estimator_ensemble_wrappers_for_dataset(
    *args: Any, **kwargs: Any
) -> list[CatEstimatorEnsembleWrapper]:
    return asyncio.run(local_async.funcs.build_cat_estimator_ensemble_wrappers_for_dataset(*args, **kwargs))


def estimate_pdf_for_slice(*args: Any, **kwargs: Any) -> qp.Ensemble:
    return asyncio.run(local_async.funcs.estimate_pdf_for_slice(*args, **kwargs))


def estimate_dataset(*args: Any, **kwargs: Any) -> models.Estimates:
    return asyncio.run(local_async.funcs.estimate_dataset(*args, **kwargs))
