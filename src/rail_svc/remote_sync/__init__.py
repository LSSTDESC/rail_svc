"""Sync wrapper instances for remote table operations.

This module provides factory functions to create synchronous wrappers
around async remote operations for each database table.

Examples
--------
>>> ops = algorithm()
>>> result = ops.get_row(1)

>>> # With custom configuration:
>>> ops = algorithm(timeout=60.0, auth_token="...")
>>> result = ops.get_row(1)
"""

from typing import Any

from .. import remote_async
from .base import (
    AlgorithmSyncRemoteOperations,
    BandSyncRemoteOperations,
    CatalogBandAssocSyncRemoteOperations,
    CatalogTagSyncRemoteOperations,
    DatasetAssocSyncRemoteOperations,
    DatasetSyncRemoteOperations,
    EstimatesSyncRemoteOperations,
    EstimatorSyncRemoteOperations,
    ModelSyncRemoteOperations,
)
from .funcs import SyncRemoteFuncs


def algorithm() -> AlgorithmSyncRemoteOperations:
    """Create sync remote operations for algorithm table."""
    return AlgorithmSyncRemoteOperations(remote_async.algorithm)


def band() -> BandSyncRemoteOperations:
    """Create sync remote operations for band table."""
    return BandSyncRemoteOperations(remote_async.band)


def catalog_band_assoc() -> CatalogBandAssocSyncRemoteOperations:
    """Create sync remote operations for catalog_band_assoc table."""
    return CatalogBandAssocSyncRemoteOperations(remote_async.catalog_band_assoc)


def catalog_tag() -> CatalogTagSyncRemoteOperations:
    """Create sync remote operations for catalog_tag table."""
    return CatalogTagSyncRemoteOperations(remote_async.catalog_tag)


def dataset() -> DatasetSyncRemoteOperations:
    """Create sync remote operations for dataset table."""
    return DatasetSyncRemoteOperations(remote_async.dataset)


def dataset_assoc() -> DatasetAssocSyncRemoteOperations:
    """Create sync remote operations for dataset_assoc table."""
    return DatasetAssocSyncRemoteOperations(remote_async.dataset_assoc)


def estimates() -> EstimatesSyncRemoteOperations:
    """Create sync remote operations for estimates table."""
    return EstimatesSyncRemoteOperations(remote_async.estimates)


def estimator() -> EstimatorSyncRemoteOperations:
    """Create sync remote operations for estimator table."""
    return EstimatorSyncRemoteOperations(remote_async.estimator)


def model() -> ModelSyncRemoteOperations:
    """Create sync remote operations for model table."""
    return ModelSyncRemoteOperations(remote_async.model)


def funcs() -> SyncRemoteFuncs:
    """Create sync remote operations for fuctions."""
    return SyncRemoteFuncs(remote_async.funcs)


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
    "funcs",
]
