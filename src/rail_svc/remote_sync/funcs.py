"""Sync wrapper for remote funcs operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from .. import models
from ..remote_async.funcs import AsyncRemoteFuncs
from .base import sync_wrapper

F = TypeVar("F", bound=Callable[..., Any])


class SyncRemoteFuncs:
    """Synchronous wrapper for AsyncRemoteFuncs.

    Provides blocking synchronous methods that wrap async remote funcs operations
    using asyncio.run(). Each method call creates a new event loop.

    Warning
    -------
    This wrapper is convenient but less efficient than using AsyncRemoteFuncs
    directly. For multiple operations, prefer the async API with context manager.

    Cannot be used from async code (will raise RuntimeError).

    Examples
    --------
    >>> funcs = SyncRemoteFuncs(async_funcs)
    >>> result = funcs.estimate_pdf(estimator_id=1, dataset_id=2, row=0)
    >>> estimators = funcs.get_estimators_for_dataset(dataset_id=2)
    """

    def __init__(self, async_ops: AsyncRemoteFuncs) -> None:
        self.async_ops = async_ops

    # Funcs operations

    @sync_wrapper(AsyncRemoteFuncs.estimate_pdf)
    def estimate_pdf(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.async_ops.estimate_pdf(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.estimate_ensemble)
    def estimate_ensemble(self, *args: Any, **kwargs: Any) -> models.EstimateEnsembleResponse:
        return self.async_ops.estimate_ensemble(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.get_estimators_for_dataset)
    def get_estimators_for_dataset(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self.async_ops.get_estimators_for_dataset(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.load_catalog_yaml)
    def load_catalog_yaml(self, *args: Any, **kwargs: Any) -> models.LoadCatalogYamlResponse:
        return self.async_ops.load_catalog_yaml(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.get_dataset_and_estimates)
    def get_dataset_and_estimates(self, *args: Any, **kwargs: Any) -> models.GetDatasetAndEstimatesResponse:
        return self.async_ops.get_dataset_and_estimates(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.get_data_and_estimates_data)
    def get_data_and_estimates_data(
        self, *args: Any, **kwargs: Any
    ) -> models.GetDataAndEstimatesDataResponse:
        return self.async_ops.get_data_and_estimates_data(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.create_matched_dataset)
    def create_matched_dataset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.async_ops.create_matched_dataset(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.estimate_pdf_for_slice)
    def estimate_pdf_for_slice(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.async_ops.estimate_pdf_for_slice(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteFuncs.estimate_dataset)
    def estimate_dataset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self.async_ops.estimate_dataset(*args, **kwargs)  # type: ignore
