"""API layer functions for filtering database records.

Provides high-level convenience functions for flexible filtering with automatic
session management. These functions support various comparison operators, logical
operators, and efficient streaming for large result sets.

For operations requiring explicit transaction control or complex multi-table queries,
use the TableOperations methods directly with explicit session management.
"""

from collections.abc import AsyncIterator, Sequence
from typing import TypeVar, Any

from pydantic import BaseModel

from ..db.base import Base
from ..db_oper.base import TableOperations
from ..db.session import get_session
from ..db_funcs.filter import Filter, FilterOp, OrderBy

T = TypeVar("T", bound=Base)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
CreateT = TypeVar("CreateT", bound=BaseModel)


async def filter_rows(
    table_ops: TableOperations[T, ResponseT, CreateT],
    filters: list[Filter] | None = None,
    logical_op: str = "and",
    order_by: OrderBy | list[OrderBy] | None = None,
    skip: int = 0,
    limit: int | None = None,
) -> list[ResponseT]:
    """Filter rows based on conditions with automatic session management.
    
    Loads all results into memory and returns as Pydantic models. For large
    result sets, consider using filter_rows_streaming() instead. Creates and
    manages its own database session.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    filters
        List of Filter objects to apply. If None, returns all rows.
    logical_op
        How to combine multiple filters: "and" (all must match) or
        "or" (any must match). Default is "and".
    order_by
        Single OrderBy or list of OrderBy directives for sorting results
    skip
        Number of rows to skip before returning results (offset)
    limit
        Maximum number of rows to return. If None, uses table's default
        pagination limit
        
    Returns
    -------
    list[ResponseT]
        List of Pydantic representations of matching rows
        
    Raises
    ------
    AttributeError
        If any filter references a non-existent field
    ValueError
        If logical_op is not "and" or "or", or filter values are invalid
        
    Notes
    -----
    - All results are loaded into memory
    - For large result sets, use filter_rows_streaming() instead
    - Filters can use various operators (EQ, GT, LIKE, IN, etc.)
    - Use Filter and FilterOp classes to construct filter conditions
    
    See Also
    --------
    filter_rows_streaming : Stream results for memory efficiency
    count_filtered_rows : Count matching rows without fetching data
    filter_one : Get exactly one matching row
    find_by : Convenience function for simple equality filters
    """
    async with get_session() as session:
        rows = await table_ops.filter_rows(
            session,
            filters=filters,
            logical_op=logical_op,
            order_by=order_by,
            skip=skip,
            limit=limit,
        )
        return table_ops.to_pydantic_list(rows)


async def filter_rows_streaming(
    table_ops: TableOperations[T, ResponseT, CreateT],
    filters: list[Filter] | None = None,
    logical_op: str = "and",
    order_by: OrderBy | list[OrderBy] | None = None,
    skip: int = 0,
    limit: int | None = None,
) -> AsyncIterator[ResponseT]:
    """Filter rows as an async iterator with automatic session management.
    
    Yields Pydantic models one at a time for memory-efficient processing of
    large result sets. Creates and manages its own database session.
    
    IMPORTANT: The session context remains open during the entire iteration.
    Consuming code should process items promptly to avoid holding the
    database connection for extended periods.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    filters
        List of Filter objects to apply. If None, returns all rows.
    logical_op
        How to combine multiple filters: "and" or "or". Default is "and".
    order_by
        Single OrderBy or list of OrderBy directives for sorting results
    skip
        Number of rows to skip before returning results (offset)
    limit
        Maximum number of rows to return. If None, uses table's default
        pagination limit
        
    Yields
    ------
    ResponseT
        Pydantic representation of each matching row
        
    Raises
    ------
    AttributeError
        If any filter references a non-existent field
    ValueError
        If logical_op is not "and" or "or", or filter values are invalid
        
    Notes
    -----
    - The database session remains open during the entire iteration
    - Process items promptly to avoid long-lived connections
    - More memory efficient than filter_rows() for large result sets
    - For slow processing, consider using filter_rows() instead
    
    See Also
    --------
    filter_rows : Load all results into memory at once
    """
    async with get_session() as session:
        async for row in table_ops.filter_rows_streaming(
            session,
            filters=filters,
            logical_op=logical_op,
            order_by=order_by,
            skip=skip,
            limit=limit,
        ):
            yield table_ops.to_pydantic(row)


async def count_filtered_rows(
    table_ops: TableOperations[T, ResponseT, CreateT],
    filters: list[Filter] | None = None,
    logical_op: str = "and",
) -> int:
    """Count rows matching filter criteria with automatic session management.
    
    Useful for pagination metadata (e.g., "showing 10 of 245 results").
    Creates and manages its own database session.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    filters
        List of Filter objects to apply. If None, counts all rows.
    logical_op
        How to combine multiple filters: "and" or "or". Default is "and".
        
    Returns
    -------
    int
        Number of rows matching the filter criteria
        
    Raises
    ------
    AttributeError
        If any filter references a non-existent field
    ValueError
        If logical_op is not "and" or "or", or filter values are invalid
        
    See Also
    --------
    filter_rows : Get the actual matching rows
    """
    async with get_session() as session:
        return await table_ops.count_filtered_rows(
            session,
            filters=filters,
            logical_op=logical_op,
        )


async def filter_one(
    table_ops: TableOperations[T, ResponseT, CreateT],
    filters: list[Filter],
    logical_op: str = "and",
) -> ResponseT:
    """Filter for exactly one row with automatic session management.
    
    Returns the single matching row as a Pydantic model. Raises an error
    if no rows or multiple rows match. Creates and manages its own database
    session.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    filters
        List of Filter objects to apply
    logical_op
        How to combine multiple filters: "and" or "or". Default is "and".
        
    Returns
    -------
    ResponseT
        Pydantic representation of the single matching row
        
    Raises
    ------
    AttributeError
        If any filter references a non-existent field
    ValueError
        If logical_op is not "and" or "or", or filter values are invalid
    KeyError
        If no rows match the criteria or multiple rows match
        
    See Also
    --------
    filter_one_or_none : Returns None instead of raising if not found
    find_one_by : Convenience function for simple equality filters
    """
    async with get_session() as session:
        row = await table_ops.filter_one(
            session,
            filters=filters,
            logical_op=logical_op,
        )
        return table_ops.to_pydantic(row)


async def filter_one_or_none(
    table_ops: TableOperations[T, ResponseT, CreateT],
    filters: list[Filter],
    logical_op: str = "and",
) -> ResponseT | None:
    """Filter for at most one row with automatic session management.
    
    Similar to filter_one() but returns None instead of raising KeyError
    when no rows are found. Creates and manages its own database session.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    filters
        List of Filter objects to apply
    logical_op
        How to combine multiple filters: "and" or "or". Default is "and".
        
    Returns
    -------
    ResponseT | None
        Pydantic representation of the single matching row, or None if no match
        
    Raises
    ------
    AttributeError
        If any filter references a non-existent field
    ValueError
        If logical_op is not "and" or "or", or filter values are invalid
    KeyError
        If multiple rows match the criteria
        
    See Also
    --------
    filter_one : Raises KeyError instead of returning None when not found
    """
    async with get_session() as session:
        row = await table_ops.filter_one_or_none(
            session,
            filters=filters,
            logical_op=logical_op,
        )
        return table_ops.to_pydantic(row) if row is not None else None


async def find_by(
    table_ops: TableOperations[T, ResponseT, CreateT],
    order_by: OrderBy | list[OrderBy] | None = None,
    skip: int = 0,
    limit: int | None = None,
    **kwargs: Any,
) -> list[ResponseT]:
    """Find rows by simple equality conditions with automatic session management.
    
    Convenience wrapper around filter_rows() for the common case of filtering
    by exact field values. Creates and manages its own database session.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    order_by
        Single OrderBy or list of OrderBy directives for sorting results
    skip
        Number of rows to skip before returning results (offset)
    limit
        Maximum number of rows to return
    **kwargs
        Field names and values to filter by (all must match - AND logic)
        
    Returns
    -------
    list[ResponseT]
        List of Pydantic representations of matching rows
        
    Raises
    ------
    AttributeError
        If any field doesn't exist on the model
        
    Notes
    -----
    - All kwargs are combined with AND logic (all must match)
    - Only supports equality (==) comparisons
    - For other operators, use filter_rows() with Filter objects
    
    See Also
    --------
    filter_rows : Full filtering with all operators
    find_one_by : Get exactly one row by equality conditions
    """
    async with get_session() as session:
        rows = await table_ops.find_by(
            session,
            order_by=order_by,
            skip=skip,
            limit=limit,
            **kwargs,
        )
        return table_ops.to_pydantic_list(rows)


async def find_one_by(
    table_ops: TableOperations[T, ResponseT, CreateT],
    **kwargs: Any,
) -> ResponseT:
    """Find exactly one row by simple equality conditions with automatic session management.
    
    Convenience wrapper around filter_one() for exact field matches. Creates
    and manages its own database session.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    **kwargs
        Field names and values to filter by (all must match)
        
    Returns
    -------
    ResponseT
        Pydantic representation of the single matching row
        
    Raises
    ------
    AttributeError
        If any field doesn't exist on the model
    KeyError
        If no rows match or multiple rows match
        
    Notes
    -----
    - All kwargs are combined with AND logic (all must match)
    - Only supports equality (==) comparisons
    - For other operators, use filter_one() with Filter objects
    
    See Also
    --------
    filter_one : Get one row with full filter support
    find_by : Get multiple rows by equality conditions
    """
    async with get_session() as session:
        row = await table_ops.find_one_by(session, **kwargs)
        return table_ops.to_pydantic(row)
