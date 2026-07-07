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

>>> # Using extended dataset operations:
>>> async with dataset as ops:
...     loaded = await ops.load(path="/data/file.hdf5", load_type=LoadType.link)
...     data = await ops.read_slice(row_id=loaded.id, start=0, stop=100)
"""

from .. import models
from ..client.base import (
    RemoteDatasetOperations,
    RemoteEstimatesOperations,
    RemoteModelOperations,
)
from macon.config import config as global_config
from .base import (
    AsyncRemoteDatasetOperations,
    AsyncRemoteEstimatesOperations,
    AsyncRemoteModelOperations,
    AsyncRemoteOperations,
)
from .funcs import AsyncRemoteFuncs

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

dataset: AsyncRemoteDatasetOperations = AsyncRemoteDatasetOperations(
    BASE_URL, "dataset", models.Dataset, models.DatasetCreate, client_class=RemoteDatasetOperations
)

dataset_assoc: AsyncRemoteOperations[models.DatasetAssoc, models.DatasetAssocCreate] = AsyncRemoteOperations(
    BASE_URL, "dataset_assoc", models.DatasetAssoc, models.DatasetAssocCreate
)

estimates: AsyncRemoteEstimatesOperations = AsyncRemoteEstimatesOperations(
    BASE_URL, "estimates", models.Estimates, models.EstimatesCreate, client_class=RemoteEstimatesOperations
)

estimator: AsyncRemoteOperations[models.Estimator, models.EstimatorCreate] = AsyncRemoteOperations(
    BASE_URL, "estimator", models.Estimator, models.EstimatorCreate
)

model: AsyncRemoteModelOperations = AsyncRemoteModelOperations(
    BASE_URL, "model", models.Model, models.ModelCreate, client_class=RemoteModelOperations
)

filter_ab: AsyncRemoteOperations[models.FilterAB, models.FilterABCreate] = AsyncRemoteOperations(
    BASE_URL, "filter_ab", models.FilterAB, models.FilterABCreate
)

sed: AsyncRemoteOperations[models.Sed, models.SedCreate] = AsyncRemoteOperations(
    BASE_URL, "sed", models.Sed, models.SedCreate
)

funcs: AsyncRemoteFuncs = AsyncRemoteFuncs(
    BASE_URL,
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
    "filter_ab",
    "model",
    "sed",
    "funcs",
]
