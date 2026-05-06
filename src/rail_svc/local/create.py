"""API layer functions for creating database records.

Provides high-level convenience functions that manage sessions and transactions
automatically. These functions are ideal for simple create operations where you
don't need explicit transaction control.

For operations requiring explicit transaction control, multiple related queries,
or custom error handling, use the TableOperations methods directly with explicit
session management.
"""

from collections.abc import Sequence
from typing import TypeVar, Any

from pydantic import BaseModel

from ..db.base import Base
from ..db_oper.base import TableOperations
from ..db.session import get_session

T = TypeVar("T", bound=Base)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
CreateT = TypeVar("CreateT", bound=BaseModel)


async def create_row[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    table_ops: TableOperations[T, ResponseT, CreateT],
    *,
    validate: bool = True,
    **kwargs: Any,
) -> ResponseT:
    """Create a single row with automatic session and transaction management.

    Creates and manages its own database session and transaction. The row is
    committed to the database before returning. For operations requiring
    explicit transaction control, use TableOperations.create_row() directly.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    validate
        Whether to validate input against Pydantic model before creation
    **kwargs
        Column names and their values for the new row

    Returns
    -------
    ResponseT
        Pydantic representation of the newly created row

    Raises
    ------
    ValidationError
        Pydantic validation failed on the input data (if validate=True)
    IntegrityError
        Database integrity constraint violation (e.g., duplicate key)

    Notes
    -----
    - Row is committed to database before returning
    - When validate=True, validation may fail for cases where the database
      provides default values. Consider setting validate=False in such cases.
    - The returned Pydantic model includes all database-generated values

    See Also
    --------
    create_rows : Create multiple rows atomically
    create_rows_batched : Create many rows in batches
    """
    async with get_session() as session:
        async with session.begin():
            row = await table_ops.create_row(session, validate=validate, **kwargs)
            return table_ops.to_pydantic(row)


async def create_rows[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    table_ops: TableOperations[T, ResponseT, CreateT],
    rows_data: Sequence[dict[str, Any]],
    *,
    validate: bool = True,
) -> list[ResponseT]:
    """Create multiple rows atomically with automatic session management.

    All rows are created in a single transaction - if any row fails,
    all insertions are rolled back. Creates and manages its own database
    session and transaction.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    rows_data
        Sequence of dictionaries, each containing column names and values
        for a new row
    validate
        Whether to validate each row against Pydantic model before creation

    Returns
    -------
    list[ResponseT]
        List of Pydantic representations of newly created rows

    Raises
    ------
    ValidationError
        Pydantic validation failed on any row's input data (if validate=True)
    IntegrityError
        Database integrity constraint violation (e.g., duplicate key)
    ValueError
        If rows_data is empty

    Notes
    -----
    - All rows are created atomically - partial success is not possible
    - Rows are committed to database before returning
    - When validate=True, validation may fail for cases where the database
      provides default values. Consider setting validate=False in such cases.
    - For very large datasets, consider using create_rows_batched()

    See Also
    --------
    create_row : Create a single row
    create_rows_batched : Create many rows in batches with partial success possible
    bulk_insert_rows : High-performance bulk insert without object returns
    """
    async with get_session() as session:
        async with session.begin():
            rows = await table_ops.create_rows(session, rows_data, validate=validate)
            return table_ops.to_pydantic_list(rows)


async def create_rows_batched[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    table_ops: TableOperations[T, ResponseT, CreateT],
    rows_data: Sequence[dict[str, Any]],
    *,
    validate: bool = True,
    batch_size: int = 1000,
) -> list[ResponseT]:
    """Create multiple rows in batches with automatic session management.

    Unlike create_rows(), this commits after each batch, so partial
    success is possible if a later batch fails. Creates and manages
    its own database session.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    rows_data
        Sequence of dictionaries for new rows
    validate
        Whether to validate each row against Pydantic model
    batch_size
        Number of rows to insert per batch (default: 1000)

    Returns
    -------
    list[ResponseT]
        List of Pydantic representations of all newly created rows

    Raises
    ------
    ValidationError
        Pydantic validation failed on any row (if validate=True)
    IntegrityError
        Database integrity constraint violation in any batch
    ValueError
        If rows_data is empty or batch_size < 1

    Notes
    -----
    - This function commits after each batch
    - If a batch fails, previously committed batches remain in the database
    - Use create_rows() if you need atomic all-or-nothing behavior
    - For very large datasets, this is more memory efficient than create_rows()

    See Also
    --------
    create_rows : Atomic creation of multiple rows (all-or-nothing)
    bulk_insert_rows : Fastest bulk insert without object returns
    """
    async with get_session() as session:
        rows = await table_ops.create_rows_batched(
            session, rows_data, validate=validate, batch_size=batch_size
        )
        return table_ops.to_pydantic_list(rows)


async def bulk_insert_rows[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    table_ops: TableOperations[T, ResponseT, CreateT],
    rows_data: Sequence[dict[str, Any]],
    *,
    validate: bool = True,
) -> int:
    """Bulk insert rows with automatic session management.

    This is much faster than create_rows() but doesn't return the
    created objects or handle get_create_kwargs() preprocessing.
    Creates and manages its own database session and transaction.

    Parameters
    ----------
    table_ops
        Table operations instance (e.g., from rail_svc.tables)
    rows_data
        Sequence of dictionaries for new rows
    validate
        Whether to validate each row against Pydantic model

    Returns
    -------
    int
        Number of rows inserted

    Raises
    ------
    ValidationError
        Pydantic validation failed on any row (if validate=True)
    IntegrityError
        Database integrity constraint violation
    ValueError
        If rows_data is empty

    Notes
    -----
    - Much faster than create_rows() for large datasets
    - Does NOT call get_create_kwargs() - use for simple inserts only
    - Does NOT return created objects with DB-generated values
    - Does NOT trigger SQLAlchemy events (e.g., before_insert)
    - Returns only the count of inserted rows

    See Also
    --------
    create_rows : Returns created objects with DB-generated values
    create_rows_batched : Batched creation with partial success possible
    """
    async with get_session() as session:
        return await table_ops.bulk_insert_rows(session, rows_data, validate=validate)
