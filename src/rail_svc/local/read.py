"""API layer functions for reading database records.

Provides high-level convenience functions that manage sessions automatically
and return Pydantic models. These functions are ideal for simple read operations
where you don't need explicit transaction control.

For operations requiring transactions or multiple related queries, use the
TableOperations methods directly with explicit session management.
"""

from collections.abc import AsyncIterator
from typing import TypeVar, Any

from pydantic import BaseModel

from ..db.base import Base
from ..db_oper.base import TableOperations
from ..db.session import get_session

T = TypeVar("T", bound=Base)
ResponseT = TypeVar("ResponseT", bound=BaseModel)


async def get_row(
    table_ops: TableOperations[T, ResponseT, Any],
    row_id: int,
) -> ResponseT:
    """Get a single row by ID with automatic session management.

    Creates and manages its own database session. For operations requiring
    explicit transaction control, use TableOperations.get_row() directly.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.dp_oper)
    row_id
        Primary key of the row to retrieve

    Returns
    -------
    ResponseT
        Pydantic representation of the row

    Raises
    ------
    NoResultFound
        If no row exists with the given ID

    See Also
    --------
    get_row_or_none : Returns None instead of raising if not found
    get_row_by_name : Look up by name field instead of ID
    """
    async with get_session() as session:
        row = await table_ops.get_row(session, row_id)
        return table_ops.to_pydantic(row)


async def get_row_by_name(
    table_ops: TableOperations[T, ResponseT, Any],
    name: str,
) -> ResponseT:
    """Get a single row by name with automatic session management.

    Creates and manages its own database session. For operations requiring
    explicit transaction control, use TableOperations.get_row_by_name() directly.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.dp_oper)
    name
        Value of the name field to search for

    Returns
    -------
    ResponseT
        Pydantic representation of the row

    Raises
    ------
    NoResultFound
        If no row exists with the given name

    See Also
    --------
    get_row : Look up by ID instead of name
    """
    async with get_session() as session:
        row = await table_ops.get_row_by_name(session, name)
        return table_ops.to_pydantic(row)


async def get_rows(
    table_ops: TableOperations[T, ResponseT, Any],
    skip: int = 0,
    limit: int | None = None,
) -> list[ResponseT]:
    """Get multiple rows with pagination and automatic session management.

    Creates and manages its own database session. For operations requiring
    explicit transaction control or streaming, use TableOperations methods directly.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.dp_oper)
    skip
        Number of rows to skip (offset for pagination)
    limit
        Maximum number of rows to return. If None, a default limit
        will be applied by the underlying query

    Returns
    -------
    list[ResponseT]
        List of Pydantic representations

    Notes
    -----
    - All rows are loaded into memory at once
    - For large result sets, consider using get_rows_streaming()
    - Default limits apply if limit is None (check TableOperations config)

    See Also
    --------
    get_rows_streaming : Stream results to reduce memory usage
    count_rows : Get total count without fetching data
    """
    async with get_session() as session:
        rows = await table_ops.get_rows(session, skip=skip, limit=limit)
        return table_ops.to_pydantic_list(rows)


async def get_rows_streaming(
    table_ops: TableOperations[T, ResponseT, Any],
    skip: int = 0,
    limit: int | None = None,
) -> AsyncIterator[ResponseT]:
    """Stream rows one at a time with automatic session management.

    More memory efficient than get_rows() for large result sets.
    The database session remains open during iteration, so process
    items promptly.

    IMPORTANT: The session context remains open during the entire iteration.
    Consuming code should process items promptly to avoid holding the
    database connection for extended periods.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.dp_oper)
    skip
        Number of rows to skip (offset for pagination)
    limit
        Maximum number of rows to return. If None, a default limit
        will be applied by the underlying query

    Yields
    ------
    ResponseT
        Pydantic representation of each row

    Notes
    -----
    - The database session remains open during the entire iteration
    - Process items promptly to avoid long-lived connections
    - For slow processing or multiple passes over data, use get_rows() instead
    - More memory efficient than get_rows() for large result sets

    See Also
    --------
    get_rows : Load all rows into memory at once (simpler for small sets)
    """
    async with get_session() as session:
        async for row in table_ops.get_rows_streaming(session, skip=skip, limit=limit):
            yield table_ops.to_pydantic(row)


async def get_row_or_none(
    table_ops: TableOperations[T, ResponseT, Any],
    row_id: int,
) -> ResponseT | None:
    """Get a single row by ID, returning None if not found.

    Creates and manages its own database session. For operations requiring
    explicit transaction control, use TableOperations.get_row_or_none() directly.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.dp_oper)
    row_id
        Primary key of the row to retrieve

    Returns
    -------
    ResponseT | None
        Pydantic representation of the row, or None if not found

    See Also
    --------
    get_row : Raises NoResultFound instead of returning None
    """
    async with get_session() as session:
        row = await table_ops.get_row_or_none(session, row_id)
        return table_ops.to_pydantic(row) if row is not None else None


async def count_rows(
    table_ops: TableOperations[T, ResponseT, Any],
) -> int:
    """Count total rows in a table with automatic session management.

    Creates and manages its own database session. For operations requiring
    explicit transaction control, use TableOperations.count_rows() directly.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.dp_oper)

    Returns
    -------
    int
        Total number of rows in the table

    See Also
    --------
    get_rows : Get paginated rows with skip/limit
    """
    async with get_session() as session:
        return await table_ops.count_rows(session)


async def lookup_by_id_or_name(
    table_ops: TableOperations[T, ResponseT, Any],
    row_id: int | None = None,
    name: str | None = None,
) -> ResponseT:
    """Look up a row by either ID or name with automatic session management.

    Convenience function that accepts either an ID or name and returns the
    corresponding row. Useful when user input could be either format.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.dp_oper)
    row_id
        Primary key of the row (provide this OR name)
    name
        Name field value (provide this OR row_id)

    Returns
    -------
    ResponseT
        Pydantic representation of the row

    Raises
    ------
    ValueError
        If neither row_id nor name provided, or if both provided
    NoResultFound
        If no row found with the given ID or name

    See Also
    --------
    get_row : Look up by ID only
    get_row_by_name : Look up by name only
    """
    async with get_session() as session:
        row_id_resolved, row = await table_ops.lookup_by_id_or_name(session, row_id, name, need_object=True)
        return table_ops.to_pydantic(row)
