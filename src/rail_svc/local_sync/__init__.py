from .. import local_async
from .base import (AlgorithmSyncOperations, BandSyncOperations,
                   CatalogBandAssocOperations, CatalogTagSyncOperations,
                   DatasetSyncOperations, EstimatesSyncOperations,
                   EstimatorSyncOperations, ModelSyncOperations)

algorithm = AlgorithmSyncOperations(local_async.algorithm)
band = BandSyncOperations(local_async.band)
catalog_band_assoc = CatalogBandAssocOperations(local_async.catalog_band_assoc)
catalog_tag = CatalogTagSyncOperations(local_async.catalog_tag)
dataset = DatasetSyncOperations(local_async.dataset)
estimates = EstimatesSyncOperations(local_async.estimates)
estimator = EstimatorSyncOperations(local_async.estimator)
model = ModelSyncOperations(local_async.model)

__all__ = [
    "algorithm",
    "band",
    "catalog_band_assoc",
    "catalog_tag",
    "dataset",
    "estimates",
    "estimator",
    "model",
]
