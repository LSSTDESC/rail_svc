"""Base class for table-specific local operations."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Callable, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..db.base import Base
from ..db_funcs.filter import Filter, OrderBy
from ..db_oper.base import TableOperations
# Import the API modules that contain the functions
from . import create, delete
from . import filter as filter_ops
from . import read, update


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

    if TYPE_CHECKING:
        # pylint: disable=unused-argument

        async def create_row(
            self,
            *,
            validate: bool = True,
            **kwargs: Any,
        ) -> ResponseT: ...

        async def create_rows(
            self,
            rows_data: Sequence[dict[str, Any]],
            *,
            validate: bool = True,
        ) -> list[ResponseT]: ...

        async def create_rows_batched(
            self,
            rows_data: Sequence[dict[str, Any]],
            *,
            validate: bool = True,
            batch_size: int = 1000,
        ) -> list[ResponseT]: ...

        async def bulk_insert_rows(
            self,
            rows_data: Sequence[dict[str, Any]],
            *,
            validate: bool = True,
        ) -> int: ...

        async def get_row(
            self,
            row_id: int,
        ) -> ResponseT: ...

        async def get_row_by_name(
            self,
            name: str,
        ) -> ResponseT: ...

        async def get_rows(
            self,
            skip: int = 0,
            limit: int | None = None,
        ) -> list[ResponseT]: ...

        async def get_rows_streaming(
            self,
            skip: int = 0,
            limit: int | None = None,
        ) -> AsyncIterator[ResponseT]: ...

        async def get_row_or_none(
            self,
            row_id: int,
        ) -> ResponseT | None: ...

        async def count_rows(
            self,
        ) -> int: ...

        async def lookup_by_id_or_name(
            self,
            row_id: int | None,
            name: str | None,
            *,
            need_object: bool = False,
        ) -> tuple[int, ResponseT | None]: ...

        async def update_row(
            self,
            row_id: int,
            **kwargs: Any,
        ) -> ResponseT: ...

        async def update_rows(
            self,
            updates: Sequence[dict[str, Any]],
        ) -> list[ResponseT]: ...

        async def delete_row(
            self,
            row_id: int,
            *,
            capture_data: bool = True,
        ) -> dict[str, Any] | None: ...

        async def delete_rows(
            self,
            row_ids: list[int],
            *,
            capture_data: bool = False,
        ) -> list[dict[str, Any]] | None: ...

        async def bulk_delete_rows(
            self,
            row_ids: list[int],
        ) -> int: ...

        async def filter_rows(
            self,
            filters: list[Filter] | None = None,
            logical_op: str = "and",
            order_by: OrderBy | list[OrderBy] | None = None,
            skip: int = 0,
            limit: int | None = None,
        ) -> list[ResponseT]: ...

        async def filter_rows_streaming(
            self,
            filters: list[Filter] | None = None,
            logical_op: str = "and",
            order_by: OrderBy | list[OrderBy] | None = None,
            skip: int = 0,
            limit: int | None = None,
        ) -> AsyncIterator[ResponseT]: ...

        async def count_filtered_rows(
            self,
            filters: list[Filter] | None = None,
            logical_op: str = "and",
        ) -> int: ...

        async def filter_one(
            self,
            filters: list[Filter],
            logical_op: str = "and",
        ) -> ResponseT: ...

        async def filter_one_or_none(
            self,
            filters: list[Filter],
            logical_op: str = "and",
        ) -> ResponseT | None: ...

        async def find_by(
            self,
            order_by: OrderBy | list[OrderBy] | None = None,
            skip: int = 0,
            limit: int | None = None,
            **kwargs: Any,
        ) -> list[ResponseT]: ...

        async def find_one_by(
            self,
            **kwargs: Any,
        ) -> ResponseT: ...

    DELEGATED_METHODS = {
        # Module name -> list of function names
        "read": [
            "get_row",
            "get_row_by_name",
            "get_rows",
            "get_rows_streaming",
            "get_row_or_none",
            "count_rows",
            "lookup_by_id_or_name",
        ],
        "create": [
            "create_row",
            "create_rows",
            "create_rows_batched",
            "bulk_insert_rows",
        ],
        "update": [
            "update_row",
            "update_rows",
        ],
        "delete": [
            "delete_row",
            "delete_rows",
            "bulk_delete_rows",
        ],
        "filter": [
            "filter_rows",
            "filter_rows_streaming",
            "count_filtered_rows",
            "filter_one",
            "filter_one_or_none",
            "find_by",
            "find_one_by",
        ],
    }

    def __init__(self, table_operations: TableOperations[T, ResponseT, CreateT]) -> None:
        """Initialize with table operations.

        Parameters
        ----------
        table_operations : TableOperations[T, ResponseT, CreateT]
            The table operations instance to wrap
        """
        self._table_ops = table_operations
        self._bind_delegated_methods()

    def _bind_delegated_methods(self) -> None:
        """Bind all delegated methods to this instance.

        This method dynamically attaches API functions from read, create,
        update, delete, and filter modules to this instance, pre-binding
        them with the table operations.
        """
        # Map module names to actual module objects
        module_map = {
            "read": read,
            "create": create,
            "update": update,
            "delete": delete,
            "filter": filter_ops,
        }

        for module_name, func_list in self.DELEGATED_METHODS.items():
            module = module_map[module_name]
            for func_name in func_list:
                func = getattr(module, func_name)
                # Pre-bind the table_ops as first argument
                bound_func = functools.partial(func, self._table_ops)
                functools.update_wrapper(bound_func, func)
                setattr(self, func_name, bound_func)


class SyncLocalOperations[T: Base, ResponseT: BaseModel, CreateT: BaseModel]:
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

    if TYPE_CHECKING:
        # pylint: disable=unused-argument

        def create_row(
            self,
            *,
            validate: bool = True,
            **kwargs: Any,
        ) -> ResponseT: ...

        def create_rows(
            self,
            rows_data: Sequence[dict[str, Any]],
            *,
            validate: bool = True,
        ) -> list[ResponseT]: ...

        def create_rows_batched(
            self,
            rows_data: Sequence[dict[str, Any]],
            *,
            validate: bool = True,
            batch_size: int = 1000,
        ) -> list[ResponseT]: ...

        def bulk_insert_rows(
            self,
            rows_data: Sequence[dict[str, Any]],
            *,
            validate: bool = True,
        ) -> int: ...

        def get_row(
            self,
            row_id: int,
        ) -> ResponseT: ...

        def get_row_by_name(
            self,
            name: str,
        ) -> ResponseT: ...

        def get_rows(
            self,
            skip: int = 0,
            limit: int | None = None,
        ) -> list[ResponseT]: ...

        def get_rows_streaming(
            self,
            skip: int = 0,
            limit: int | None = None,
        ) -> AsyncIterator[ResponseT]: ...

        def get_row_or_none(
            self,
            row_id: int,
        ) -> ResponseT | None: ...

        def count_rows(
            self,
        ) -> int: ...

        def lookup_by_id_or_name(
            self,
            row_id: int | None,
            name: str | None,
            *,
            need_object: bool = False,
        ) -> tuple[int, ResponseT | None]: ...

        def update_row(
            self,
            row_id: int,
            **kwargs: Any,
        ) -> ResponseT: ...

        def update_rows(
            self,
            updates: Sequence[dict[str, Any]],
        ) -> list[ResponseT]: ...

        def delete_row(
            self,
            row_id: int,
            *,
            capture_data: bool = True,
        ) -> dict[str, Any] | None: ...

        def delete_rows(
            self,
            row_ids: list[int],
            *,
            capture_data: bool = False,
        ) -> list[dict[str, Any]] | None: ...

        def bulk_delete_rows(
            self,
            row_ids: list[int],
        ) -> int: ...

        def filter_rows(
            self,
            filters: list[Filter] | None = None,
            logical_op: str = "and",
            order_by: OrderBy | list[OrderBy] | None = None,
            skip: int = 0,
            limit: int | None = None,
        ) -> list[ResponseT]: ...

        def filter_rows_streaming(
            self,
            filters: list[Filter] | None = None,
            logical_op: str = "and",
            order_by: OrderBy | list[OrderBy] | None = None,
            skip: int = 0,
            limit: int | None = None,
        ) -> AsyncIterator[ResponseT]: ...

        def count_filtered_rows(
            self,
            filters: list[Filter] | None = None,
            logical_op: str = "and",
        ) -> int: ...

        def filter_one(
            self,
            filters: list[Filter],
            logical_op: str = "and",
        ) -> ResponseT: ...

        def filter_one_or_none(
            self,
            filters: list[Filter],
            logical_op: str = "and",
        ) -> ResponseT | None: ...

        def find_by(
            self,
            order_by: OrderBy | list[OrderBy] | None = None,
            skip: int = 0,
            limit: int | None = None,
            **kwargs: Any,
        ) -> list[ResponseT]: ...

        def find_one_by(
            self,
            **kwargs: Any,
        ) -> ResponseT: ...

    def __init__(self, async_ops: LocalOperations[T, ResponseT, CreateT]) -> None:
        self._async_ops = async_ops
        self._bind_sync_methods()

    def _bind_sync_methods(self) -> None:
        """Bind synchronous wrapper methods to this instance."""
        for _module_name, func_list in LocalOperations.DELEGATED_METHODS.items():
            for func_name in func_list:
                async_func = getattr(self._async_ops, func_name)

                def make_sync(afunc: Callable) -> Callable:
                    """Make a sync wrapper for a function"""

                    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                        """Wrap a function with asyncio.run"""
                        return asyncio.run(afunc(*args, **kwargs))

                    functools.update_wrapper(sync_wrapper, afunc)
                    return sync_wrapper

                setattr(self, func_name, make_sync(async_func))


def create_local_operations[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    table_operations: TableOperations[T, ResponseT, CreateT],
) -> LocalOperations[T, ResponseT, CreateT]:
    """Create an async LocalOperations instance.

    Parameters
    ----------
    table_operations : TableOperations[T, ResponseT, CreateT]
        The table operations to wrap

    Returns
    -------
    LocalOperations[T, ResponseT, CreateT]
        Async local operations with all methods bound
    """
    return LocalOperations(table_operations)
