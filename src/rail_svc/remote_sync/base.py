"""Sync wrapper for remote table operations."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine

from typing import Any

from pydantic import BaseModel

from .. import models
from ..remote_async.base import AsyncRemoteOperations
from ..models import Filter, OrderBy


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

    def _run_async(self, coro: Coroutine) -> Any:
        """Run async coroutine, with error handling for async contexts."""
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

    # CREATE operations

    def create_row(
        self,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> ResponseT:
        return self._run_async(self.async_ops.create_row(validate=validate, **kwargs))

    def create_rows(
        self,
        rows_data: list[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> list[ResponseT]:
        return self._run_async(self.async_ops.create_rows(rows_data, validate=validate))

    def create_rows_batched(
        self,
        rows_data: list[dict[str, Any]],
        *,
        validate: bool = True,
        batch_size: int = 1000,
    ) -> list[ResponseT]:
        return self._run_async(
            self.async_ops.create_rows_batched(rows_data, validate=validate, batch_size=batch_size)
        )

    def bulk_insert_rows(
        self,
        rows_data: list[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> int:
        return self._run_async(self.async_ops.bulk_insert_rows(rows_data, validate=validate))

    # READ operations

    def get_row(
        self,
        row_id: int,
    ) -> ResponseT:
        return self._run_async(self.async_ops.get_row(row_id))

    def get_row_by_name(
        self,
        name: str,
    ) -> ResponseT:
        return self._run_async(self.async_ops.get_row_by_name(name))

    def get_rows(
        self,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        return self._run_async(self.async_ops.get_rows(skip, limit))

    def get_row_or_none(
        self,
        row_id: int,
    ) -> ResponseT | None:
        return self._run_async(self.async_ops.get_row_or_none(row_id))

    def count_rows(self) -> int:
        return self._run_async(self.async_ops.count_rows())

    def lookup_by_id_or_name(
        self,
        row_id: int | None = None,
        name: str | None = None,
    ) -> tuple[int, ResponseT]:
        return self._run_async(self.async_ops.lookup_by_id_or_name(row_id, name))

    # UPDATE operations

    def update_row(
        self,
        row_id: int,
        **kwargs: Any,
    ) -> ResponseT:
        return self._run_async(self.async_ops.update_row(row_id, **kwargs))

    def update_rows(
        self,
        updates: list[dict[str, Any]],
    ) -> list[ResponseT]:
        return self._run_async(self.async_ops.update_rows(updates))

    # DELETE operations

    def delete_row(
        self,
        row_id: int,
        *,
        capture_data: bool = True,
    ) -> ResponseT | None:
        return self._run_async(self.async_ops.delete_row(row_id, capture_data=capture_data))

    def delete_rows(
        self,
        row_ids: list[int],
        *,
        capture_data: bool = False,
    ) -> list[ResponseT] | int:
        return self._run_async(self.async_ops.delete_rows(row_ids, capture_data=capture_data))

    def bulk_delete_rows(
        self,
        row_ids: list[int],
    ) -> int:
        return self._run_async(self.async_ops.bulk_delete_rows(row_ids))

    # FILTER/QUERY operations

    def filter_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        return self._run_async(self.async_ops.filter_rows(filters, logical_op, order_by, skip, limit))

    def count_filtered_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
    ) -> int:
        return self._run_async(self.async_ops.count_filtered_rows(filters, logical_op))

    def filter_one(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT:
        return self._run_async(self.async_ops.filter_one(filters, logical_op))

    def filter_one_or_none(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT | None:
        return self._run_async(self.async_ops.filter_one_or_none(filters, logical_op))

    def find_by(
        self,
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[ResponseT]:
        return self._run_async(self.async_ops.find_by(order_by, skip, limit, **kwargs))

    def find_one_by(
        self,
        **kwargs: Any,
    ) -> ResponseT:
        return self._run_async(self.async_ops.find_one_by(**kwargs))


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
