"""Remote operations for database tables.

This module provides pre-configured AsyncRemoteOperations instances
for each database table. Use these for making async API calls to the
remote service.

Examples
--------
>>> async with algorithm as ops:
...     result = await ops.get_row(1)

>>> # Or for single operations (less efficient):
>>> result = await algorithm.get_row(1)
"""

from .. import models
from ..config import config as global_config
from .base import AsyncRemoteOperations

BASE_URL = global_config.client.service_url

algorithm: AsyncRemoteOperations[models.Algorithm, models.AlgorithmCreate] = AsyncRemoteOperations(
    BASE_URL, "algorithm", models.Algorithm, models.AlgorithmCreate
)

band: AsyncRemoteOperations[models.Band, models.BandCreate] = AsyncRemoteOperations(
    BASE_URL, "band", models.Band, models.BandCreate
)

catalog_band_assoc: AsyncRemoteOperations[models.CatalogBandAssoc, models.CatalogBandAssocCreate] = (
    AsyncRemoteOperations(
        BASE_URL, "catalog_band_assoc", models.CatalogBandAssoc, models.CatalogBandAssocCreate
    )
)

catalog_tag: AsyncRemoteOperations[models.CatalogTag, models.CatalogTagCreate] = AsyncRemoteOperations(
    BASE_URL, "catalog_tag", models.CatalogTag, models.CatalogTagCreate
)

dataset: AsyncRemoteOperations[models.Dataset, models.DatasetCreate] = AsyncRemoteOperations(
    BASE_URL, "dataset", models.Dataset, models.DatasetCreate
)

dataset_assoc: AsyncRemoteOperations[models.DatasetAssoc, models.DatasetAssocCreate] = AsyncRemoteOperations(
    BASE_URL, "dataset_assoc", models.DatasetAssoc, models.DatasetAssocCreate
)

estimates: AsyncRemoteOperations[models.Estimates, models.EstimatesCreate] = AsyncRemoteOperations(
    BASE_URL, "estimates", models.Estimates, models.EstimatesCreate
)

estimator: AsyncRemoteOperations[models.Estimator, models.EstimatorCreate] = AsyncRemoteOperations(
    BASE_URL, "estimator", models.Estimator, models.EstimatorCreate
)

model: AsyncRemoteOperations[models.Model, models.ModelCreate] = AsyncRemoteOperations(
    BASE_URL, "model", models.Model, models.ModelCreate
)

__all__ = [
    "algorithm",
    "band",
    "catalog_band_assoc",
    "catalog_tag",
    "dataset",
    "dataset_assoc",
    "estimates",
    "estimator",
    "model",
]
