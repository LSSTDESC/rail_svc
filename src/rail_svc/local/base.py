"""Base class for table-specific local operations."""

from __future__ import annotations

import asyncio
import functools
from typing import TypeVar

from pydantic import BaseModel

from ..db.base import Base
from ..db_oper import TableOperations

# Import the API modules that contain the functions
from . import read, create, update, delete, filter as filter_ops

# Type variables
T = TypeVar("T", bound=Base)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
CreateT = TypeVar("CreateT", bound=BaseModel)


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

    _DELEGATED_METHODS = {
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

    @overload
    def __getattr__(self, name: str) -> Callable[..., Any]: ...
    
    def __getattr__(self, name: str) -> Any:
        # This will be called for dynamically added methods
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
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

        for module_name, func_list in self._DELEGATED_METHODS.items():
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

    def __init__(self, async_ops: LocalOperations[T, ResponseT, CreateT]) -> None:
        self._async_ops = async_ops
        self._bind_sync_methods()

    @overload
    def __getattr__(self, name: str) -> Callable[..., Any]: ...
    
    def __getattr__(self, name: str) -> Any:
        # This will be called for dynamically added methods
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
    def _bind_sync_methods(self) -> None:
        """Bind synchronous wrapper methods to this instance."""
        for module_name, func_list in LocalOperations._DELEGATED_METHODS.items():
            for func_name in func_list:
                async_func = getattr(self._async_ops, func_name)

                def make_sync(afunc):
                    def sync_wrapper(*args, **kwargs):
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
