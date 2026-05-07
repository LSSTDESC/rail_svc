"""Base class for table-specific local operations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from pydantic import BaseModel

from .. import db, models
from ..db.base import Base
from ..db.session import get_session
from ..db_funcs.filter import Filter, OrderBy
from ..db_oper.base import TableOperations


class LocalOperations[T: Base, ResponseT: BaseModel, CreateT: BaseModel]:
    """Base class for table-specific local operations.

    Dynamically binds API functions as methods on this instance,
    pre-bound with the table operations. All methods are async.

    Examples
    --------
    >>> from rail_svc.local import algorithm
    >>>
    >>> # In async context
    >>> algo = await algorithm.get_row(row_id=1)
    >>> algos = await algorithm.get_rows(limit=10)
    """

    def __init__(self, table_operations: TableOperations[T, ResponseT, CreateT]) -> None:
        """Initialize with table operations.

        Parameters
        ----------
        table_operations : TableOperations[T, ResponseT, CreateT]
            The table operations instance to wrap
        """
        self._table_ops = table_operations

    async def create_row(
        self,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> ResponseT:
        async with get_session() as session:
            async with session.begin():
                row = await self._table_ops.create_row(session, validate=validate, **kwargs)
                return self._table_ops.to_pydantic(row)

    async def create_rows(
        self,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> list[ResponseT]:
        async with get_session() as session:
            async with session.begin():
                rows = await self._table_ops.create_rows(session, rows_data, validate=validate)
                return self._table_ops.to_pydantic_list(rows)

    async def create_rows_batched(
        self,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
        batch_size: int = 1000,
    ) -> list[ResponseT]:
        async with get_session() as session:
            rows = await self._table_ops.create_rows_batched(
                session, rows_data, validate=validate, batch_size=batch_size
            )
            return self._table_ops.to_pydantic_list(rows)

    async def bulk_insert_rows(
        self,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> int:
        async with get_session() as session:
            return await self._table_ops.bulk_insert_rows(session, rows_data, validate=validate)

    async def get_row(
        self,
        row_id: int,
    ) -> ResponseT:
        async with get_session() as session:
            row = await self._table_ops.get_row(session, row_id)
            return self._table_ops.to_pydantic(row)

    async def get_row_by_name(
        self,
        name: str,
    ) -> ResponseT:
        async with get_session() as session:
            row = await self._table_ops.get_row_by_name(session, name)
            return self._table_ops.to_pydantic(row)

    async def get_rows(
        self,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        async with get_session() as session:
            rows = await self._table_ops.get_rows(session, skip=skip, limit=limit)
            return self._table_ops.to_pydantic_list(list(rows))

    async def get_rows_streaming(
        self,
        skip: int = 0,
        limit: int | None = None,
    ) -> AsyncIterator[ResponseT]:
        async with get_session() as session:
            async for row in self._table_ops.get_rows_streaming(session, skip=skip, limit=limit):
                yield self._table_ops.to_pydantic(row)

    async def get_row_or_none(
        self,
        row_id: int,
    ) -> ResponseT | None:
        async with get_session() as session:
            row = await self._table_ops.get_row_or_none(session, row_id)
            return self._table_ops.to_pydantic(row) if row is not None else None

    async def count_rows(
        self,
    ) -> int:
        async with get_session() as session:
            return await self._table_ops.count_rows(session)

    async def lookup_by_id_or_name(
        self,
        row_id: int | None,
        name: str | None,
        *,
        need_object: bool = False,  # pylint: disable=unused-argument
    ) -> tuple[int, ResponseT | None]:
        async with get_session() as session:
            row_id_resolved, row = await self._table_ops.lookup_by_id_or_name(
                session,
                row_id,
                name,
                need_object=True,
            )
            assert row
            return row_id_resolved, self._table_ops.to_pydantic(row)

    async def update_row(
        self,
        row_id: int,
        **kwargs: Any,
    ) -> ResponseT:
        async with get_session() as session:
            async with session.begin():
                row = await self._table_ops.update_row(session, row_id, **kwargs)
                return self._table_ops.to_pydantic(row)

    async def update_rows(
        self,
        updates: Sequence[dict[str, Any]],
    ) -> list[ResponseT]:
        async with get_session() as session:
            async with session.begin():
                rows = await self._table_ops.update_rows(session, updates)
                return self._table_ops.to_pydantic_list(rows)

    async def delete_row(
        self,
        row_id: int,
        *,
        capture_data: bool = True,
    ) -> dict[str, Any] | None:
        async with get_session() as session:
            async with session.begin():
                return await self._table_ops.delete_row(session, row_id, capture_data=capture_data)

    async def delete_rows(
        self,
        row_ids: list[int],
        *,
        capture_data: bool = False,
    ) -> list[dict[str, Any]] | None:
        async with get_session() as session:
            async with session.begin():
                return await self._table_ops.delete_rows(session, row_ids, capture_data=capture_data)

    async def bulk_delete_rows(
        self,
        row_ids: list[int],
    ) -> int:
        async with get_session() as session:
            async with session.begin():
                return await self._table_ops.bulk_delete_rows(session, row_ids)

    async def filter_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        async with get_session() as session:
            rows = await self._table_ops.filter_rows(
                session,
                filters=filters,
                logical_op=logical_op,
                order_by=order_by,
                skip=skip,
                limit=limit,
            )
            return self._table_ops.to_pydantic_list(list(rows))

    async def filter_rows_streaming(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
    ) -> AsyncIterator[ResponseT]:
        async with get_session() as session:
            async for row in self._table_ops.filter_rows_streaming(
                session,
                filters=filters,
                logical_op=logical_op,
                order_by=order_by,
                skip=skip,
                limit=limit,
            ):
                yield self._table_ops.to_pydantic(row)

    async def count_filtered_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
    ) -> int:
        async with get_session() as session:
            return await self._table_ops.count_filtered_rows(
                session,
                filters=filters,
                logical_op=logical_op,
            )

    async def filter_one(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT:
        async with get_session() as session:
            row = await self._table_ops.filter_one(
                session,
                filters=filters,
                logical_op=logical_op,
            )
            return self._table_ops.to_pydantic(row)

    async def filter_one_or_none(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT | None:
        async with get_session() as session:
            row = await self._table_ops.filter_one_or_none(
                session,
                filters=filters,
                logical_op=logical_op,
            )
            return self._table_ops.to_pydantic(row) if row is not None else None

    async def find_by(
        self,
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[ResponseT]:
        async with get_session() as session:
            rows = await self._table_ops.find_by(
                session,
                order_by=order_by,
                skip=skip,
                limit=limit,
                **kwargs,
            )
            return self._table_ops.to_pydantic_list(list(rows))

    async def find_one_by(
        self,
        **kwargs: Any,
    ) -> ResponseT:
        async with get_session() as session:
            row = await self._table_ops.find_one_by(session, **kwargs)
            return self._table_ops.to_pydantic(row)


class AlgorithmLocalOperations(LocalOperations[db.Algorithm, models.Algorithm, models.AlgorithmCreate]):
    """Operations on local DB for Algorithm table."""


class BandLocalOperations(LocalOperations[db.Band, models.Band, models.BandCreate]):
    """Operations on local DB for Band table."""


class CatalogBandAssocOperations(
    LocalOperations[db.CatalogBandAssoc, models.CatalogBandAssoc, models.CatalogBandAssocCreate]
):
    """Operations on local DB for CatalogBandAssoc table."""


class CatalogTagLocalOperations(LocalOperations[db.CatalogTag, models.CatalogTag, models.CatalogTagCreate]):
    """Operations on local DB for CatalogTag table."""


class DatasetLocalOperations(LocalOperations[db.Dataset, models.Dataset, models.DatasetCreate]):
    """Operations on local DB for Dataset table."""


class EstimatesLocalOperations(LocalOperations[db.Estimates, models.Estimates, models.EstimatesCreate]):
    """Operations on local DB for Estimates table."""


class EstimatorLocalOperations(LocalOperations[db.Estimator, models.Estimator, models.EstimatorCreate]):
    """Operations on local DB for Estimator table."""


class ModelLocalOperations(LocalOperations[db.Model, models.Model, models.ModelCreate]):
    """Operations on local DB for Model table."""
