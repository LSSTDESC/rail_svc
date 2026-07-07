"""Domain-specific sync operations for rail_svc tables."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import qp

from macon.local_sync.base import SyncOperations, sync_wrapper

__all__ = ["SyncOperations", "sync_wrapper"]

from .. import db, models
from ..local_async.rail_svc import (
    DatasetLocalOperations,
    EstimatesLocalOperations,
    ModelLocalOperations,
)


class AlgorithmSyncOperations(SyncOperations[db.Algorithm, models.Algorithm, models.AlgorithmCreate]):
    """Operations on local DB for Algorithm table."""


class BandSyncOperations(SyncOperations[db.Band, models.Band, models.BandCreate]):
    """Operations on local DB for Band table."""


class CatalogBandAssocOperations(
    SyncOperations[db.CatalogBandAssoc, models.CatalogBandAssoc, models.CatalogBandAssocCreate]
):
    """Operations on local DB for CatalogBandAssoc table."""


class CatalogTagSyncOperations(SyncOperations[db.CatalogTag, models.CatalogTag, models.CatalogTagCreate]):
    """Operations on local DB for CatalogTag table."""


class DatasetSyncOperations(SyncOperations[db.Dataset, models.Dataset, models.DatasetCreate]):
    """Operations on local DB for Dataset table."""

    @sync_wrapper(DatasetLocalOperations.load)
    def load(self, *args: Any, **kwargs: Any) -> models.Dataset:
        return cast(DatasetLocalOperations, self.async_ops).load(*args, **kwargs)  # type: ignore

    @sync_wrapper(DatasetLocalOperations.read_slice)
    def read_slice(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        return cast(DatasetLocalOperations, self.async_ops).read_slice(*args, **kwargs)  # type: ignore


class DatasetAssocSyncOperations(
    SyncOperations[db.DatasetAssoc, models.DatasetAssoc, models.DatasetAssocCreate]
):
    """Operations on local DB for DatasetAssoc table."""


class EstimatesSyncOperations(SyncOperations[db.Estimates, models.Estimates, models.EstimatesCreate]):
    """Operations on local DB for Estimates table."""

    @sync_wrapper(EstimatesLocalOperations.load)
    def load(self, *args: Any, **kwargs: Any) -> models.Estimates:
        return cast(EstimatesLocalOperations, self.async_ops).load(*args, **kwargs)  # type: ignore

    @sync_wrapper(EstimatesLocalOperations.read_slice)
    def read_slice(self, *args: Any, **kwargs: Any) -> qp.Ensemble:
        return cast(EstimatesLocalOperations, self.async_ops).read_slice(*args, **kwargs)


class EstimatorSyncOperations(SyncOperations[db.Estimator, models.Estimator, models.EstimatorCreate]):
    """Operations on local DB for Estimator table."""


class ModelSyncOperations(SyncOperations[db.Model, models.Model, models.ModelCreate]):
    """Operations on local DB for Model table."""

    @sync_wrapper(ModelLocalOperations.load)
    def load(self, *args: Any, **kwargs: Any) -> models.Model:
        return cast(ModelLocalOperations, self.async_ops).load(*args, **kwargs)  # type: ignore


class FilterABSyncOperations(SyncOperations[db.FilterAB, models.FilterAB, models.FilterABCreate]):
    """Operations on local DB for FilterAB table."""


class SedSyncOperations(SyncOperations[db.Sed, models.Sed, models.SedCreate]):
    """Operations on local DB for Sed table."""
