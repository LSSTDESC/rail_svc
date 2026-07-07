"""Domain-specific sync remote operations for rail_svc tables."""

from __future__ import annotations

from typing import ClassVar

from macon.remote_sync.base import SyncRemoteOperations

__all__ = ["SyncRemoteOperations"]

from .. import models


class AlgorithmSyncRemoteOperations(SyncRemoteOperations[models.Algorithm, models.AlgorithmCreate]):
    """Sync wrapper for remote operations on Algorithm table."""


class BandSyncRemoteOperations(SyncRemoteOperations[models.Band, models.BandCreate]):
    """Sync wrapper for remote operations on Band table."""


class CatalogBandAssocSyncRemoteOperations(
    SyncRemoteOperations[models.CatalogBandAssoc, models.CatalogBandAssocCreate]
):
    """Sync wrapper for remote operations on CatalogBandAssoc table."""


class CatalogTagSyncRemoteOperations(SyncRemoteOperations[models.CatalogTag, models.CatalogTagCreate]):
    """Sync wrapper for remote operations on CatalogTag table."""


class DatasetSyncRemoteOperations(SyncRemoteOperations[models.Dataset, models.DatasetCreate]):
    """Sync wrapper for remote operations on Dataset table."""

    _extra_methods: ClassVar[list[str]] = ["load", "read_slice", "download"]


class DatasetAssocSyncRemoteOperations(SyncRemoteOperations[models.DatasetAssoc, models.DatasetAssocCreate]):
    """Sync wrapper for remote operations on DatasetAssoc table."""


class EstimatesSyncRemoteOperations(SyncRemoteOperations[models.Estimates, models.EstimatesCreate]):
    """Sync wrapper for remote operations on Estimates table."""

    _extra_methods: ClassVar[list[str]] = ["load", "read_slice", "download"]


class EstimatorSyncRemoteOperations(SyncRemoteOperations[models.Estimator, models.EstimatorCreate]):
    """Sync wrapper for remote operations on Estimator table."""


class ModelSyncRemoteOperations(SyncRemoteOperations[models.Model, models.ModelCreate]):
    """Sync wrapper for remote operations on Model table."""

    _extra_methods: ClassVar[list[str]] = ["load", "download"]


class FilterABSyncRemoteOperations(SyncRemoteOperations[models.FilterAB, models.FilterABCreate]):
    """Sync wrapper for remote operations on FilterAB table."""


class SedSyncRemoteOperations(SyncRemoteOperations[models.Sed, models.SedCreate]):
    """Sync wrapper for remote operations on Sed table."""
