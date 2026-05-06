"""API layer functions for deleting database records.

Provides high-level convenience functions that manage sessions and transactions
automatically. These functions are ideal for simple delete operations where you
don't need explicit transaction control.

For operations requiring explicit transaction control, multiple related queries,
or custom error handling, use the TableOperations methods directly with explicit
session management.
"""

from typing import TypeVar, Any

from pydantic import BaseModel

from ..db.base import Base
from ..db_oper.base import TableOperations
from ..db.session import get_session

T = TypeVar("T", bound=Base)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
CreateT = TypeVar("CreateT", bound=BaseModel)


async def delete_row(
    table_ops: TableOperations[T, ResponseT, CreateT],
    row_id: int,
    *,
    capture_data: bool = True,
) -> dict[str, Any] | None:
    """Delete a single row by primary key with automatic session management.
    
    The deletion is automatically committed. Pre and post-delete hooks
    are called within the transaction - if either hook raises an exception,
    the deletion is rolled back. Creates and manages its own database
    session and transaction.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    row_id
        Primary key of the row to delete
    capture_data
        If True, capture row data before deletion and return it.
        Set to False for performance if you don't need the data.
        
    Returns
    -------
    dict[str, Any] | None
        Dictionary of deleted row data if capture_data=True, else None
        
    Raises
    ------
    KeyError
        If row with given ID does not exist
    IntegrityError
        If deletion violates database constraints (e.g., foreign key)
    Exception
        If pre or post-delete hooks raise an exception
        
    Notes
    -----
    - Pre-delete hook is called before deletion with access to the row object
    - Post-delete hook is called after deletion but before commit
    - If any hook raises an exception, the deletion is rolled back
    - The deletion is automatically committed
    
    See Also
    --------
    delete_rows : Delete multiple rows atomically
    bulk_delete_rows : Fast bulk deletion without hooks
    """
    async with get_session() as session:
        async with session.begin():
            return await table_ops.delete_row(session, row_id, capture_data=capture_data)


async def delete_rows(
    table_ops: TableOperations[T, ResponseT, CreateT],
    row_ids: list[int],
    *,
    capture_data: bool = False,
) -> list[dict[str, Any]] | None:
    """Delete multiple rows atomically with automatic session management.
    
    All rows are deleted in a single transaction - if any deletion fails,
    all deletions are rolled back. Creates and manages its own database
    session and transaction.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    row_ids
        List of primary keys to delete
    capture_data
        If True, capture data for all deleted rows and return it
        
    Returns
    -------
    list[dict[str, Any]] | None
        List of deleted row data dicts if capture_data=True, else None
        
    Raises
    ------
    ValueError
        If row_ids is empty
    KeyError
        If any row ID is not found
    IntegrityError
        If any deletion violates constraints
    Exception
        If any pre or post-delete hook raises an exception
        
    Notes
    -----
    - All deletions are performed atomically - partial success is not possible
    - Pre and post-delete hooks are called for each row
    - Deletions are committed to database before returning
    - If any hook raises an exception, all deletions are rolled back
    
    See Also
    --------
    delete_row : Delete a single row
    bulk_delete_rows : Fast bulk deletion without hooks or data capture
    """
    async with get_session() as session:
        async with session.begin():
            return await table_ops.delete_rows(session, row_ids, capture_data=capture_data)


async def bulk_delete_rows(
    table_ops: TableOperations[T, ResponseT, CreateT],
    row_ids: list[int],
) -> int:
    """Delete multiple rows using bulk SQL operation with automatic session management.
    
    This is much faster than delete_rows() but does NOT call hooks
    and does NOT return deleted row data. Creates and manages its own
    database session and transaction.
    
    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    row_ids
        List of primary keys to delete
        
    Returns
    -------
    int
        Number of rows actually deleted (may be less than len(row_ids)
        if some IDs didn't exist)
        
    Raises
    ------
    ValueError
        If row_ids is empty
    IntegrityError
        If deletion violates constraints
        
    Notes
    -----
    - Does NOT call pre/post-delete hooks
    - Does NOT verify rows exist before deleting
    - Does NOT capture deleted row data
    - Much faster for large deletions
    - Returns actual count of deleted rows (may differ from len(row_ids))
    
    See Also
    --------
    delete_rows : Delete with hooks and optional data capture
    delete_row : Delete a single row with hooks
    """
    async with get_session() as session:
        async with session.begin():
            return await table_ops.bulk_delete_rows(session, row_ids)
