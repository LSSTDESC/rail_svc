"""Sync wrapper for remote table operations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any

from pydantic import BaseModel

from .. import models
from ..remote_async.base import AsyncRemoteOperations


def run_async[F: Callable[..., Any]](func: F) -> F:
    """Decorator that wraps async method calls with asyncio.run().

    Automatically runs the coroutine returned by the decorated method
    using asyncio.run(), with error handling for async contexts.

    Parameters
    ----------
    func : Callable
        Method that returns a coroutine (calls async_ops methods)

    Returns
    -------
    Callable
        Wrapped method that runs the coroutine synchronously

    Raises
    ------
    RuntimeError
        If called from within an async context (event loop already running)

    Examples
    --------
    >>> @run_async
    >>> def get_row(self, row_id: int) -> ResponseT:
    ...     return self.async_ops.get_row(row_id)
    """

    @wraps(func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        coro = func(self, *args, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            return asyncio.run(coro)
        # Already in async context
        raise RuntimeError(
            f"{self.__class__.__name__} cannot be used from async code. "
            f"Use AsyncRemoteOperations directly with 'async with' instead."
        )

    return wrapper  # type: ignore


class SyncRemoteOperations[ResponseT: BaseModel, CreateT: BaseModel]:
    """Synchronous wrapper for AsyncRemoteOperations.

    Provides blocking synchronous methods that wrap async remote operations
    using asyncio.run(). Each method call creates a new event loop.

    Warning
    -------
    This wrapper is convenient but less efficient than using AsyncRemoteOperations
    directly. For multiple operations, prefer the async API with context manager.

    Cannot be used from async code (will raise RuntimeError).

    Examples
    --------
    >>> ops = SyncRemoteOperations(async_ops)
    >>> result = ops.get_row(1)
    >>> rows = ops.get_rows(limit=10)
    """

    def __init__(self, async_ops: AsyncRemoteOperations[ResponseT, CreateT]) -> None:
        self.async_ops = async_ops

    # CREATE operations

    @run_async
    def create_row(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.create_row(*args, **kwargs)  # type: ignore

    @run_async
    def create_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.create_rows(*args, **kwargs)  # type: ignore

    @run_async
    def create_rows_batched(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.create_rows_batched(*args, **kwargs)  # type: ignore

    @run_async
    def bulk_insert_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.bulk_insert_rows(*args, **kwargs)  # type: ignore

    # READ operations

    @run_async
    def get_row(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.get_row(*args, **kwargs)  # type: ignore

    @run_async
    def get_row_by_name(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.get_row_by_name(*args, **kwargs)  # type: ignore

    @run_async
    def get_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.get_rows(*args, **kwargs)  # type: ignore

    @run_async
    def get_row_or_none(self, *args: Any, **kwargs: Any) -> ResponseT | None:
        return self.async_ops.get_row_or_none(*args, **kwargs)  # type: ignore

    @run_async
    def count_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.count_rows(*args, **kwargs)  # type: ignore

    @run_async
    def lookup_by_id_or_name(self, *args: Any, **kwargs: Any) -> tuple[int, ResponseT]:
        return self.async_ops.lookup_by_id_or_name(*args, **kwargs)  # type: ignore

    # UPDATE operations

    @run_async
    def update_row(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.update_row(*args, **kwargs)  # type: ignore

    @run_async
    def update_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.update_rows(*args, **kwargs)  # type: ignore

    # DELETE operations

    @run_async
    def delete_row(self, *args: Any, **kwargs: Any) -> ResponseT | None:
        return self.async_ops.delete_row(*args, **kwargs)  # type: ignore

    @run_async
    def delete_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT] | int:
        return self.async_ops.delete_rows(*args, **kwargs)  # type: ignore

    @run_async
    def bulk_delete_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.bulk_delete_rows(*args, **kwargs)  # type: ignore

    # FILTER/QUERY operations

    @run_async
    def filter_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.filter_rows(*args, **kwargs)  # type: ignore

    @run_async
    def count_filtered_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.count_filtered_rows(*args, **kwargs)  # type: ignore

    @run_async
    def filter_one(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.filter_one(*args, **kwargs)  # type: ignore

    @run_async
    def filter_one_or_none(self, *args: Any, **kwargs: Any) -> ResponseT | None:
        return self.async_ops.filter_one_or_none(*args, **kwargs)  # type: ignore

    @run_async
    def find_by(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.find_by(*args, **kwargs)  # type: ignore

    @run_async
    def find_one_by(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.find_one_by(*args, **kwargs)  # type: ignore


# Subclasses


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


class DatasetAssocSyncRemoteOperations(SyncRemoteOperations[models.DatasetAssoc, models.DatasetAssocCreate]):
    """Sync wrapper for remote operations on DatasetAssoc table."""


class EstimatesSyncRemoteOperations(SyncRemoteOperations[models.Estimates, models.EstimatesCreate]):
    """Sync wrapper for remote operations on Estimates table."""


class EstimatorSyncRemoteOperations(SyncRemoteOperations[models.Estimator, models.EstimatorCreate]):
    """Sync wrapper for remote operations on Estimator table."""


class ModelSyncRemoteOperations(SyncRemoteOperations[models.Model, models.ModelCreate]):
    """Sync wrapper for remote operations on Model table."""
