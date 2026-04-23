"""Database row creation functions for rail-svc.

This module provides functions for creating database rows with varying
levels of performance and atomicity guarantees:

- create_row: Single row creation with full validation
- create_rows: Atomic multi-row creation (all-or-nothing)
- create_rows_batched: Batched creation with partial success possible
- bulk_insert_rows: High-performance bulk insert without object returns
"""

from typing import Any, Sequence

from pydantic import ValidationError
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_scoped_session
import structlog

from rail_svc.db.base import Base, ensure_base_inheritance, T


logger = structlog.get_logger(__name__)


async def create_row(
    the_class: type[T],
    session: async_scoped_session,
    validate: bool = True,
    **kwargs: Any,
) -> T:
    """Create a single row in the database.
    
    Parameters
    ----------
    the_class
        The SQLAlchemy model class to instantiate
    session
        DB session manager
    validate
        Whether to validate input against Pydantic model before creation
    **kwargs
        Column names and their values for the new row

    Returns
    -------
    T
        Newly created row with database-generated values

    Raises
    ------
    TypeError
        If the_class does not inherit from rail_svc.db.base.Base
    IntegrityError
        Integrity constraint violation (e.g., duplicate key, null constraint)
    ValidationError
        Pydantic validation failed on the input data
        
    Notes
    -----
    When validate=True, validation may fail for cases where the database
    provides default values. Consider setting validate=False in such cases.
    
    Examples
    --------
    >>> from myapp.models import User
    >>> async with get_session() as session:
    ...     user = await create_row(
    ...         User, 
    ...         session, 
    ...         username="alice",
    ...         email="alice@example.com"
    ...     )
    """
    ensure_base_inheritance(the_class)
    
    logger.debug(
        "Creating row", 
        table=the_class.__name__, 
        fields=list(kwargs.keys())
    )

    if validate:
        try:
            # Validate against Pydantic model for early error detection
            pydantic_class = the_class.pydantic_model_class()
            pydantic_class.model_validate(kwargs)
        except ValidationError as e:
            logger.warning(
                "Validation failed in create_row",
                table=the_class.__name__,
                errors=e.errors(),
            )
            raise  # Re-raise with full error detail
    
    # Get any additional kwargs needed (preprocessing/transformation)
    create_kwargs = await the_class.get_create_kwargs(session, **kwargs)
    row = the_class(**create_kwargs)

    try:
        session.add(row)
        await session.flush()  # Flush to assign DB-generated values
        await session.commit()
    except IntegrityError as err:
        await session.rollback()
        logger.error(
            "Integrity error during create",
            table=the_class.__name__,
            error=str(err),
        )
        raise

    # Refresh to ensure all DB-generated values are loaded
    await session.refresh(row)
    logger.info("Row created successfully", table=the_class.__name__)
    return row


async def create_rows(
    the_class: type[T],
    session: async_scoped_session,
    rows_data: Sequence[dict[str, Any]],
    validate: bool = True,
    refresh: bool = True,
) -> list[T]:
    """Create multiple rows in the database atomically.
    
    All rows are created in a single transaction - if any row fails,
    all insertions are rolled back.
    
    Parameters
    ----------
    the_class
        The SQLAlchemy model class to instantiate
    session
        DB session manager
    rows_data
        Sequence of dictionaries, each containing column names and values
        for a new row
    validate
        Whether to validate each row against Pydantic model before creation
    refresh
        Whether to refresh rows to load server-side defaults (default: True)

    Returns
    -------
    list[T]
        List of newly created rows with database-generated values

    Raises
    ------
    TypeError
        If the_class does not inherit from rail_svc.db.base.Base
    IntegrityError
        Integrity constraint violation (e.g., duplicate key, null constraint)
    ValidationError
        Pydantic validation failed on any row's input data
    ValueError
        If rows_data is empty
        
    Notes
    -----
    - When validate=True, validation may fail for cases where the database
      provides default values. Consider setting validate=False in such cases.
    - All rows are created atomically - partial success is not possible.
    - For very large datasets, consider batching the calls to this function.
    - Set refresh=False for better performance if you don't need server-side
      defaults or computed columns.
    
    Examples
    --------
    >>> from myapp.models import User
    >>> async with get_session() as session:
    ...     users = await create_rows(
    ...         User,
    ...         session,
    ...         [
    ...             {"username": "alice", "email": "alice@example.com"},
    ...             {"username": "bob", "email": "bob@example.com"},
    ...             {"username": "charlie", "email": "charlie@example.com"},
    ...         ]
    ...     )
    ...     print(f"Created {len(users)} users")
    """
    ensure_base_inheritance(the_class)
    
    if not rows_data:
        raise ValueError("rows_data cannot be empty")
    
    logger.debug(
        "Creating multiple rows",
        table=the_class.__name__,
        count=len(rows_data)
    )

    # Validate all rows before any database interaction
    if validate:
        pydantic_class = the_class.pydantic_model_class()
        for idx, row_kwargs in enumerate(rows_data):
            try:
                pydantic_class.model_validate(row_kwargs)
            except ValidationError as e:
                logger.warning(
                    "Validation failed in create_rows",
                    table=the_class.__name__,
                    row_index=idx,
                    errors=e.errors(),
                )
                raise
    
    # Process all rows to get create kwargs
    rows = []
    for idx, row_kwargs in enumerate(rows_data):
        try:
            create_kwargs = await the_class.get_create_kwargs(
                session, 
                **row_kwargs
            )
            row = the_class(**create_kwargs)
            rows.append(row)
        except Exception as e:
            logger.error(
                "Failed to prepare row",
                table=the_class.__name__,
                row_index=idx,
                error=str(e)
            )
            raise

    try:
        # Add all rows to session
        session.add_all(rows)
        await session.flush()  # Flush to assign DB-generated values
        await session.commit()
    except IntegrityError as err:
        await session.rollback()
        logger.error(
            "Integrity error during bulk create",
            table=the_class.__name__,
            row_count=len(rows),
            error=str(err),
        )
        raise

    # Refresh all rows to ensure DB-generated values are loaded
    if refresh:
        for row in rows:
            await session.refresh(row)
    
    logger.info(
        "Rows created successfully",
        table=the_class.__name__,
        count=len(rows)
    )
    return rows


async def create_rows_batched(
    the_class: type[T],
    session: async_scoped_session,
    rows_data: Sequence[dict[str, Any]],
    validate: bool = True,
    batch_size: int = 1000,
) -> list[T]:
    """Create multiple rows in batches.
    
    Unlike create_rows(), this commits after each batch, so partial
    success is possible if a later batch fails.
    
    Parameters
    ----------
    the_class
        The SQLAlchemy model class to instantiate
    session
        DB session manager
    rows_data
        Sequence of dictionaries for new rows
    validate
        Whether to validate each row against Pydantic model
    batch_size
        Number of rows to insert per batch (default: 1000)

    Returns
    -------
    list[T]
        List of all newly created rows

    Raises
    ------
    TypeError
        If the_class does not inherit from rail_svc.db.base.Base
    IntegrityError
        Integrity constraint violation in any batch
    ValidationError
        Pydantic validation failed on any row
    ValueError
        If rows_data is empty or batch_size < 1
        
    Notes
    -----
    This function commits after each batch. If a batch fails, previously
    committed batches will remain in the database. Use create_rows() if
    you need atomic all-or-nothing behavior.
    
    Examples
    --------
    >>> # Create 10,000 users in batches of 500
    >>> users_data = [
    ...     {"username": f"user_{i}", "email": f"user_{i}@example.com"}
    ...     for i in range(10000)
    ... ]
    >>> users = await create_rows_batched(
    ...     User, session, users_data, batch_size=500
    ... )
    """
    ensure_base_inheritance(the_class)
    
    if not rows_data:
        raise ValueError("rows_data cannot be empty")
    
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    
    logger.info(
        "Creating rows in batches",
        table=the_class.__name__,
        total_rows=len(rows_data),
        batch_size=batch_size
    )

    all_rows = []
    
    # Process in batches
    for batch_start in range(0, len(rows_data), batch_size):
        batch_end = min(batch_start + batch_size, len(rows_data))
        batch_data = rows_data[batch_start:batch_end]
        
        logger.debug(
            "Processing batch",
            table=the_class.__name__,
            batch_start=batch_start,
            batch_end=batch_end,
            batch_size=len(batch_data)
        )
        
        try:
            batch_rows = await create_rows(
                the_class,
                session,
                batch_data,
                validate=validate
            )
            all_rows.extend(batch_rows)
            
        except Exception as e:
            logger.error(
                "Batch failed",
                table=the_class.__name__,
                batch_start=batch_start,
                batch_end=batch_end,
                rows_created_so_far=len(all_rows),
                error=str(e)
            )
            raise
    
    logger.info(
        "All batches completed",
        table=the_class.__name__,
        total_created=len(all_rows)
    )
    return all_rows


async def bulk_insert_rows(
    the_class: type[T],
    session: async_scoped_session,
    rows_data: Sequence[dict[str, Any]],
    validate: bool = True,
) -> int:
    """Bulk insert rows using SQLAlchemy's bulk operations.
    
    This is much faster than create_rows() but doesn't return the
    created objects or handle get_create_kwargs() preprocessing.
    
    Parameters
    ----------
    the_class
        The SQLAlchemy model class to instantiate
    session
        DB session manager
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
    TypeError
        If the_class does not inherit from rail_svc.db.base.Base
    IntegrityError
        Integrity constraint violation
    ValidationError
        Pydantic validation failed on any row
    ValueError
        If rows_data is empty
        
    Notes
    -----
    - Much faster than create_rows() for large datasets
    - Does NOT call get_create_kwargs() - use for simple inserts only
    - Does NOT return created objects with DB-generated values
    - Does NOT trigger SQLAlchemy events (e.g., before_insert)
    
    Examples
    --------
    >>> # Fast insert of 100,000 simple records
    >>> count = await bulk_insert_rows(
    ...     User,
    ...     session,
    ...     [{"username": f"user_{i}"} for i in range(100000)]
    ... )
    >>> print(f"Inserted {count} users")
    """
    ensure_base_inheritance(the_class)
    
    if not rows_data:
        raise ValueError("rows_data cannot be empty")
    
    logger.debug(
        "Bulk inserting rows",
        table=the_class.__name__,
        count=len(rows_data)
    )

    # Validate all rows
    if validate:
        pydantic_class = the_class.pydantic_model_class()
        for idx, row_kwargs in enumerate(rows_data):
            try:
                pydantic_class.model_validate(row_kwargs)
            except ValidationError as e:
                logger.warning(
                    "Validation failed in bulk_insert_rows",
                    table=the_class.__name__,
                    row_index=idx,
                    errors=e.errors(),
                )
                raise

    try:
        # Use insert statement for maximum performance
        stmt = insert(the_class).values(rows_data)
        await session.execute(stmt)
        await session.commit()
        
        logger.info(
            "Bulk insert completed",
            table=the_class.__name__,
            count=len(rows_data)
        )
        return len(rows_data)
        
    except IntegrityError as err:
        await session.rollback()
        logger.error(
            "Integrity error during bulk insert",
            table=the_class.__name__,
            row_count=len(rows_data),
            error=str(err),
        )
        raise
