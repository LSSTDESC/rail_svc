"""Base class for table-specific local operations."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from functools import wraps
from typing import Any

from pydantic import BaseModel

from .. import db, models
from ..db.base import Base
from ..db.session import get_session
from ..db_oper.base import TableOperations


def with_session[F: Callable[..., Any]](func: F) -> F:
    """Decorator that wraps a method with session management.

    Opens a session context and passes it as the first argument to the
    wrapped table operations method.

    Parameters
    ----------
    func : Callable
        Method that calls a table operations method

    Returns
    -------
    Callable
        Wrapped method with session management
    """

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        async with get_session() as session:
            return await func(self, session, *args, **kwargs)

    return wrapper  # type: ignore


def with_session_transaction[F: Callable[..., Any]](func: F) -> F:
    """Decorator that wraps a method with session and transaction management.

    Opens a session context with a transaction and passes the session as
    the first argument to the wrapped table operations method.

    Parameters
    ----------
    func : Callable
        Method that calls a table operations method

    Returns
    -------
    Callable
        Wrapped method with session and transaction management
    """

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        async with get_session() as session:
            async with session.begin():
                return await func(self, session, *args, **kwargs)

    return wrapper  # type: ignore


def to_pydantic[F: Callable[..., Any]](func: F) -> F:
    """Decorator that converts ORM result to Pydantic model.

    Wraps the result of a table operations call with to_pydantic conversion.

    Parameters
    ----------
    func : Callable
        Method that returns an ORM object

    Returns
    -------
    Callable
        Wrapped method that returns a Pydantic model
    """

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await func(self, *args, **kwargs)
        return self._table_ops.to_pydantic(result)

    return wrapper  # type: ignore


def to_pydantic_list[F: Callable[..., Any]](func: F) -> F:
    """Decorator that converts ORM result list to Pydantic models.

    Wraps the result of a table operations call with to_pydantic_list conversion.

    Parameters
    ----------
    func : Callable
        Method that returns a list/sequence of ORM objects

    Returns
    -------
    Callable
        Wrapped method that returns a list of Pydantic models
    """

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await func(self, *args, **kwargs)
        return self._table_ops.to_pydantic_list(list(result))

    return wrapper  # type: ignore


def to_pydantic_or_none[F: Callable[..., Any]](func: F) -> F:
    """Decorator that converts ORM result to Pydantic model or None.

    Wraps the result of a table operations call with to_pydantic conversion,
    handling None results.

    Parameters
    ----------
    func : Callable
        Method that returns an ORM object or None

    Returns
    -------
    Callable
        Wrapped method that returns a Pydantic model or None
    """

    @wraps(func)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = await func(self, *args, **kwargs)
        return self._table_ops.to_pydantic(result) if result is not None else None

    return wrapper  # type: ignore


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

    @with_session_transaction
    @to_pydantic
    async def create_row(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.create_row(session, *args, **kwargs)

    @with_session_transaction
    @to_pydantic_list
    async def create_rows(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.create_rows(session, *args, **kwargs)

    @with_session
    @to_pydantic_list
    async def create_rows_batched(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.create_rows_batched(session, *args, **kwargs)

    @with_session
    async def bulk_insert_rows(self, session: Any, *args: Any, **kwargs: Any) -> int:
        return await self._table_ops.bulk_insert_rows(session, *args, **kwargs)

    @with_session
    @to_pydantic
    async def get_row(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.get_row(session, *args, **kwargs)

    @with_session
    @to_pydantic
    async def get_row_by_name(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.get_row_by_name(session, *args, **kwargs)

    @with_session
    @to_pydantic_list
    async def get_rows(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.get_rows(session, *args, **kwargs)

    async def get_rows_streaming(self, *args: Any, **kwargs: Any) -> AsyncIterator[ResponseT]:
        async with get_session() as session:
            async for row in self._table_ops.get_rows_streaming(session, *args, **kwargs):
                yield self._table_ops.to_pydantic(row)

    @with_session
    @to_pydantic_or_none
    async def get_row_or_none(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.get_row_or_none(session, *args, **kwargs)

    @with_session
    async def count_rows(self, session: Any, *args: Any, **kwargs: Any) -> int:
        return await self._table_ops.count_rows(session, *args, **kwargs)

    @with_session
    async def lookup_by_id_or_name(
        self,
        session: Any,
        row_id: int | None,
        name: str | None,
        *,
        need_object: bool = False,  # pylint: disable=unused-argument
    ) -> tuple[int, ResponseT | None]:
        row_id_resolved, row = await self._table_ops.lookup_by_id_or_name(
            session,
            row_id,
            name,
            need_object=True,
        )
        assert row
        return row_id_resolved, self._table_ops.to_pydantic(row)

    @with_session_transaction
    @to_pydantic
    async def update_row(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.update_row(session, *args, **kwargs)

    @with_session_transaction
    @to_pydantic_list
    async def update_rows(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.update_rows(session, *args, **kwargs)

    @with_session_transaction
    async def delete_row(self, session: Any, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        return await self._table_ops.delete_row(session, *args, **kwargs)

    @with_session_transaction
    async def delete_rows(self, session: Any, *args: Any, **kwargs: Any) -> list[dict[str, Any]] | None:
        return await self._table_ops.delete_rows(session, *args, **kwargs)

    @with_session_transaction
    async def bulk_delete_rows(self, session: Any, *args: Any, **kwargs: Any) -> int:
        return await self._table_ops.bulk_delete_rows(session, *args, **kwargs)

    @with_session
    @to_pydantic_list
    async def filter_rows(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.filter_rows(session, *args, **kwargs)

    async def filter_rows_streaming(self, *args: Any, **kwargs: Any) -> AsyncIterator[ResponseT]:
        async with get_session() as session:
            async for row in self._table_ops.filter_rows_streaming(session, *args, **kwargs):
                yield self._table_ops.to_pydantic(row)

    @with_session
    async def count_filtered_rows(self, session: Any, *args: Any, **kwargs: Any) -> int:
        return await self._table_ops.count_filtered_rows(session, *args, **kwargs)

    @with_session
    @to_pydantic
    async def filter_one(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.filter_one(session, *args, **kwargs)

    @with_session
    @to_pydantic_or_none
    async def filter_one_or_none(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.filter_one_or_none(session, *args, **kwargs)

    @with_session
    @to_pydantic_list
    async def find_by(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.find_by(session, *args, **kwargs)

    @with_session
    @to_pydantic
    async def find_one_by(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await self._table_ops.find_one_by(session, *args, **kwargs)


class AlgorithmLocalOperations(LocalOperations[db.Algorithm, models.Algorithm, models.AlgorithmCreate]):
    """Operations on local DB for Algorithm table."""


class BandLocalOperations(LocalOperations[db.Band, models.Band, models.BandCreate]):
    """Operations on local DB for Band table."""


class CatalogBandAssocLocalOperations(
    LocalOperations[db.CatalogBandAssoc, models.CatalogBandAssoc, models.CatalogBandAssocCreate]
):
    """Operations on local DB for CatalogBandAssoc table."""


class CatalogTagLocalOperations(LocalOperations[db.CatalogTag, models.CatalogTag, models.CatalogTagCreate]):
    """Operations on local DB for CatalogTag table."""


class DatasetLocalOperations(LocalOperations[db.Dataset, models.Dataset, models.DatasetCreate]):
    """Operations on local DB for Dataset table."""


class DatasetAssocLocalOperations(
    LocalOperations[db.DatasetAssoc, models.DatasetAssoc, models.DatasetAssocCreate]
):
    """Operations on local DB for DatasetAssoc table."""


class EstimatesLocalOperations(LocalOperations[db.Estimates, models.Estimates, models.EstimatesCreate]):
    """Operations on local DB for Estimates table."""


class EstimatorLocalOperations(LocalOperations[db.Estimator, models.Estimator, models.EstimatorCreate]):
    """Operations on local DB for Estimator table."""


class ModelLocalOperations(LocalOperations[db.Model, models.Model, models.ModelCreate]):
    """Operations on local DB for Model table."""
