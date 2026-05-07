"""API layer functions for updating database records.

Provides high-level convenience functions that manage sessions and transactions
automatically. These functions are ideal for simple update operations where you
don't need explicit transaction control.

For operations requiring explicit transaction control, multiple related queries,
or custom error handling, use the TableOperations methods directly with explicit
session management.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from ..db.base import Base
from ..db.session import get_session
from ..db_oper.base import TableOperations


async def update_row[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    table_ops: TableOperations[T, ResponseT, CreateT],
    row_id: int,
    **kwargs: Any,
) -> ResponseT:
    """Update a single row by primary key with automatic session management.

    The update is committed automatically. If the update fails, the
    transaction is rolled back. Creates and manages its own database
    session and transaction.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    row_id
        Primary key of the row to update
    **kwargs
        Column names and their new values

    Returns
    -------
    ResponseT
        Pydantic representation of the updated row

    Raises
    ------
    ValueError
        If attempting to change the row's ID
    KeyError
        If row with given ID does not exist
    StatementError
        If update statement is invalid (e.g., invalid column or type)

    Notes
    -----
    - The row's ID cannot be changed
    - The update is automatically committed
    - Returned Pydantic model includes any DB-generated values (e.g., updated_at)

    See Also
    --------
    update_rows : Update multiple rows atomically
    """
    async with get_session() as session:
        async with session.begin():
            row = await table_ops.update_row(session, row_id, **kwargs)
            return table_ops.to_pydantic(row)


async def update_rows[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    table_ops: TableOperations[T, ResponseT, CreateT],
    updates: Sequence[dict[str, Any]],
) -> list[ResponseT]:
    """Update multiple rows atomically with automatic session management.

    Each dict in updates must contain an 'id' key specifying which row to update.
    All updates are performed in a single transaction - if any update fails,
    all changes are rolled back. Creates and manages its own database session.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    updates
        Sequence of dicts, each containing 'id' and fields to update

    Returns
    -------
    list[ResponseT]
        List of Pydantic representations of updated rows

    Raises
    ------
    ValueError
        If updates is empty or any update dict is missing 'id' key
    KeyError
        If any row ID is not found
    StatementError
        If any update is invalid

    Notes
    -----
    - All updates are performed atomically - partial success is not possible
    - Updates are committed to database before returning
    - Each dict must contain 'id' key to identify the row
    - Returned Pydantic models include any DB-generated values

    See Also
    --------
    update_row : Update a single row
    """
    async with get_session() as session:
        async with session.begin():
            rows = await table_ops.update_rows(session, updates)
            return table_ops.to_pydantic_list(rows)
