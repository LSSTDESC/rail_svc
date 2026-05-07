"""Base class for table-specific local operations."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .. import db, models
from ..db.base import Base
from ..db_funcs.filter import Filter, OrderBy
from ..local_async.base import LocalOperations


class SyncOperations[T: Base, ResponseT: BaseModel, CreateT: BaseModel]:
    """Synchronous wrapper for local operations.

    Wraps async LocalOperations methods to provide synchronous versions
    for use in non-async contexts like CLI commands or scripts.

    WARNING: These methods use asyncio.run() internally and cannot be
    called from within an already-running event loop. Use the async
    LocalOperations directly in async contexts.

    Parameters
    ----------
    async_ops : LocalOperations[T, ResponseT, CreateT]
        The async local operations instance to wrap

    Examples
    --------
    >>> from rail_svc.local import algorithm  # Async version
    >>> from rail_svc.local.base import SyncLocalOperations
    >>>
    >>> # Create sync wrapper for CLI
    >>> algo_sync = SyncLocalOperations(algorithm)
    >>>
    >>> # Use without await (in sync context only)
    >>> result = algo_sync.get_row(row_id=1)  # No await needed
    """

    def __init__(self, async_ops: LocalOperations[T, ResponseT, CreateT]) -> None:
        self.async_ops = async_ops

    def create_row(
        self,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> ResponseT:
        return asyncio.run(self.async_ops.create_row(validate=validate, **kwargs))

    def create_rows(
        self,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> list[ResponseT]:
        return asyncio.run(self.async_ops.create_rows(rows_data, validate=validate))

    def create_rows_batched(
        self,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
        batch_size: int = 1000,
    ) -> list[ResponseT]:
        return asyncio.run(
            self.async_ops.create_rows_batched(rows_data, validate=validate, batch_size=batch_size)
        )

    def bulk_insert_rows(
        self,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> int:
        return asyncio.run(self.async_ops.bulk_insert_rows(rows_data, validate=validate))

    def get_row(
        self,
        row_id: int,
    ) -> ResponseT:
        return asyncio.run(self.async_ops.get_row(row_id))

    def get_row_by_name(
        self,
        name: str,
    ) -> ResponseT:
        return asyncio.run(self.async_ops.get_row_by_name(name))

    def get_rows(
        self,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        return asyncio.run(self.async_ops.get_rows(skip, limit))

    def get_row_or_none(
        self,
        row_id: int,
    ) -> ResponseT | None:
        return asyncio.run(self.async_ops.get_row_or_none(row_id))

    def count_rows(
        self,
    ) -> int:
        return asyncio.run(self.async_ops.count_rows())

    def lookup_by_id_or_name(
        self,
        row_id: int | None,
        name: str | None,
        *,
        need_object: bool = False,
    ) -> tuple[int, ResponseT | None]:
        return asyncio.run(self.async_ops.lookup_by_id_or_name(row_id, name, need_object=need_object))

    def update_row(
        self,
        row_id: int,
        **kwargs: Any,
    ) -> ResponseT:
        return asyncio.run(self.async_ops.update_row(row_id, **kwargs))

    def update_rows(
        self,
        updates: Sequence[dict[str, Any]],
    ) -> list[ResponseT]:
        return asyncio.run(self.async_ops.update_rows(updates))

    def delete_row(
        self,
        row_id: int,
        *,
        capture_data: bool = True,
    ) -> dict[str, Any] | None:
        return asyncio.run(self.async_ops.delete_row(row_id, capture_data=capture_data))

    def delete_rows(
        self,
        row_ids: list[int],
        *,
        capture_data: bool = False,
    ) -> list[dict[str, Any]] | None:
        return asyncio.run(self.async_ops.delete_rows(row_ids, capture_data=capture_data))

    def bulk_delete_rows(
        self,
        row_ids: list[int],
    ) -> int:
        return asyncio.run(self.async_ops.bulk_delete_rows(row_ids))

    def filter_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        return asyncio.run(self.async_ops.filter_rows(filters, logical_op, order_by, skip, limit))

    def count_filtered_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
    ) -> int:
        return asyncio.run(self.async_ops.count_filtered_rows(filters, logical_op))

    def filter_one(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT:
        return asyncio.run(self.async_ops.filter_one(filters, logical_op))

    def filter_one_or_none(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT | None:
        return asyncio.run(self.async_ops.filter_one_or_none(filters, logical_op))

    def find_by(
        self,
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[ResponseT]:
        return asyncio.run(self.async_ops.find_by(order_by, skip, limit, **kwargs))

    def find_one_by(
        self,
        **kwargs: Any,
    ) -> ResponseT:
        return asyncio.run(self.async_ops.find_one_by(**kwargs))


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


class EstimatesSyncOperations(SyncOperations[db.Estimates, models.Estimates, models.EstimatesCreate]):
    """Operations on local DB for Estimates table."""


class EstimatorSyncOperations(SyncOperations[db.Estimator, models.Estimator, models.EstimatorCreate]):
    """Operations on local DB for Estimator table."""


class ModelSyncOperations(SyncOperations[db.Model, models.Model, models.ModelCreate]):
    """Operations on local DB for Model table."""
