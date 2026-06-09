"""Sync wrapper for remote table operations."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from pydantic import BaseModel

from .. import models
from ..remote_async.base import AsyncRemoteOperations

F = TypeVar("F", bound=Callable[..., Any])


def sync_wrapper(async_method: Callable[..., Any]) -> Callable[[F], F]:
    """Decorator that wraps an async method call with asyncio.run and copies its docstring.

    This decorator is designed for creating synchronous wrappers around async methods.
    It automatically calls asyncio.run() on the async method and copies the docstring
    from the async method to the sync wrapper.

    Parameters
    ----------
    async_method : Callable
        The async method to wrap (unbound method reference)

    Returns
    -------
    Callable
        Decorator function that creates a sync wrapper

    Examples
    --------
    >>> class AsyncOps:
    ...     async def get_data(self, x: int) -> int:
    ...         '''Fetch data asynchronously.'''
    ...         return x * 2
    >>>
    >>> class SyncOps:
    ...     def __init__(self, async_ops: AsyncOps):
    ...         self.async_ops = async_ops
    ...
    ...     @sync_wrapper(AsyncOps.get_data)
    ...     def get_data(self, *args, **kwargs):
    ...         return self.async_ops.get_data(*args, **kwargs)
    >>>
    >>> sync_ops = SyncOps(AsyncOps())
    >>> sync_ops.get_data(5)  # Automatically runs in asyncio.run()
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            # Call the original function to get the coroutine
            coro = func(self, *args, **kwargs)

            async def doit() -> Any:
                async with self.async_ops:
                    return await coro

            # Run it with asyncio.run
            return asyncio.run(doit())

        wrapped.__doc__ = async_method.__doc__
        return wrapped  # type: ignore

    return decorator


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

    @sync_wrapper(AsyncRemoteOperations.create_row)
    def create_row(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.create_row(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.create_rows)
    def create_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.create_rows(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.create_rows_batched)
    def create_rows_batched(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.create_rows_batched(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.bulk_insert_rows)
    def bulk_insert_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.bulk_insert_rows(*args, **kwargs)  # type: ignore

    # READ operations

    @sync_wrapper(AsyncRemoteOperations.get_row)
    def get_row(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.get_row(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.get_row_by_name)
    def get_row_by_name(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.get_row_by_name(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.get_rows)
    def get_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.get_rows(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.get_row_or_none)
    def get_row_or_none(self, *args: Any, **kwargs: Any) -> ResponseT | None:
        return self.async_ops.get_row_or_none(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.count_rows)
    def count_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.count_rows(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.lookup_by_id_or_name)
    def lookup_by_id_or_name(self, *args: Any, **kwargs: Any) -> tuple[int, ResponseT]:
        return self.async_ops.lookup_by_id_or_name(*args, **kwargs)  # type: ignore

    # UPDATE operations

    @sync_wrapper(AsyncRemoteOperations.update_row)
    def update_row(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.update_row(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.update_rows)
    def update_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.update_rows(*args, **kwargs)  # type: ignore

    # DELETE operations

    @sync_wrapper(AsyncRemoteOperations.delete_row)
    def delete_row(self, *args: Any, **kwargs: Any) -> ResponseT | None:
        return self.async_ops.delete_row(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.delete_rows)
    def delete_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT] | int:
        return self.async_ops.delete_rows(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.bulk_delete_rows)
    def bulk_delete_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.bulk_delete_rows(*args, **kwargs)  # type: ignore

    # FILTER/QUERY operations

    @sync_wrapper(AsyncRemoteOperations.filter_rows)
    def filter_rows(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.filter_rows(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.count_filtered_rows)
    def count_filtered_rows(self, *args: Any, **kwargs: Any) -> int:
        return self.async_ops.count_filtered_rows(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.filter_one)
    def filter_one(self, *args: Any, **kwargs: Any) -> ResponseT:
        return self.async_ops.filter_one(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.filter_one_or_none)
    def filter_one_or_none(self, *args: Any, **kwargs: Any) -> ResponseT | None:
        return self.async_ops.filter_one_or_none(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.find_by)
    def find_by(self, *args: Any, **kwargs: Any) -> list[ResponseT]:
        return self.async_ops.find_by(*args, **kwargs)  # type: ignore

    @sync_wrapper(AsyncRemoteOperations.find_one_by)
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
