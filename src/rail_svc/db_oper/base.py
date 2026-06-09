from __future__ import annotations

import asyncio
from abc import abstractmethod
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar, cast

import anyio
from pydantic import BaseModel, ValidationError
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from .. import db_funcs
from ..common import unexpected
from ..config import config as global_config
from ..db.base import Base, ensure_base_inheritance

logger = get_logger(__name__)


FORBID_TRAVERSAL = False

F = TypeVar("F", bound=Callable[..., Any])


def forward_to_db_funcs(module: Any, func_name: str) -> Callable[[F], F]:
    """Decorator that forwards method calls to db_funcs module functions.

    Extracts db_class from self.ctx and session from args, then calls
    the corresponding function in the db_funcs module with all arguments.

    Parameters
    ----------
    module : Any
        The db_funcs submodule (e.g., db_funcs.read, db_funcs.filter)
    func_name : str
        Name of the function to call in the module

    Returns
    -------
    Callable
        Decorator that forwards the call

    Examples
    --------
    >>> @forward_to_db_funcs(db_funcs.read, 'get_row')
    >>> async def get_row(self, session: AsyncSession, *args, **kwargs):
    ...     pass  # Implementation replaced by decorator
    """

    def decorator(func: F) -> F:
        db_func = getattr(module, func_name)

        @wraps(func)
        async def wrapper(self: Any, session: AsyncSession, *args: Any, **kwargs: Any) -> Any:
            return await db_func(self.ctx.db_class, session, *args, **kwargs)

        wrapper.__doc__ = db_func.__doc__
        return wrapper  # type: ignore

    return decorator


def forward_to_db_funcs_streaming(module: Any, func_name: str) -> Callable[[F], F]:
    """Decorator that forwards async generator calls to db_funcs module functions.

    Similar to forward_to_db_funcs but for async generators (streaming functions).

    Parameters
    ----------
    module : Any
        The db_funcs submodule
    func_name : str
        Name of the streaming function to call

    Returns
    -------
    Callable
        Decorator that forwards the streaming call
    """

    def decorator(func: F) -> F:
        db_func = getattr(module, func_name)

        @wraps(func)
        async def wrapper(self: Any, session: AsyncSession, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            async for row in db_func(self.ctx.db_class, session, *args, **kwargs):
                yield row

        wrapper.__doc__ = db_func.__doc__
        return wrapper  # type: ignore

    return decorator


@dataclass
class TableContext[T: Base, ResponseT: BaseModel, CreateT: BaseModel]:
    """
    Common context for database tables

    Encapsulates the common configuration needed for all tables with full
    type safety for database models and their Pydantic representations.

    Type Parameters
    ---------------
    T : TypeVar, bound=Base
        Database model class (SQLAlchemy)
    ResponseT : TypeVar, bound=BaseModel
        Pydantic model class for validating / streaming responses
    CreateT : TypeVar, bound=BaseModel
        Pydantic model class for validating creation data

    Parameters
    ----------
    db_class : type[T]
        Database model class (SQLAlchemy)
    response_class : type[ResponseT]
        Pydantic model class for validating / streaming
    create_class : type[CreateT]
        Pydantic model class for validating creation data
    class_string : str
        String identifier for the class

    Examples
    --------
    >>> from myapp.models import User, UserResponse, UserCreate
    >>> context = TableContext(
    ...     db_class=User,
    ...     response_class=UserResponse,
    ...     create_class=UserCreate,
    ...     class_string="user"
    ... )
    """

    db_class: type[T]
    response_class: type[ResponseT]
    create_class: type[CreateT]
    class_string: str

    @classmethod
    def from_db_class(cls, db_class: type[T]) -> TableContext[T, ResponseT, CreateT]:
        """
        Create context from database class using conventions.

        This method uses the db_class's methods to determine the Pydantic
        models. The return type uses BaseModel for response and create types
        since we can't know the specific types at compile time.

        Parameters
        ----------
        db_class : type[T]
            Database model class that implements pydantic_model_class(),
            pydantic_create_class(), and class_string() methods

        Returns
        -------
        TableContext[T, BaseModel, BaseModel]
            Configured TableContext with base Pydantic types

        Raises
        ------
        AttributeError
            If db_class doesn't implement required methods
        TypeError
            If methods don't return BaseModel subclasses

        Examples
        --------
        >>> from myapp.models import User
        >>> context = TableContext.from_db_class(User)
        >>> # context is TableContext[User, BaseModel, BaseModel]
        """
        # Validate required methods exist
        if not hasattr(db_class, "pydantic_model_class"):
            raise AttributeError(f"{db_class.__name__} must implement pydantic_model_class() method")
        if unexpected(not hasattr(db_class, "pydantic_create_class")):
            raise AttributeError(f"{db_class.__name__} must implement pydantic_create_class() method")
        if unexpected(not hasattr(db_class, "class_string")):
            raise AttributeError(f"{db_class.__name__} must implement class_string() method")

        # Get the classes
        response_class = cast(type[ResponseT], db_class.pydantic_model_class())
        create_class = cast(type[CreateT], db_class.pydantic_create_class())

        return cls(
            db_class=db_class,
            response_class=response_class,
            create_class=create_class,
            class_string=db_class.class_string(),
        )


class TableOperations[T: Base, ResponseT: BaseModel, CreateT: BaseModel]:
    """Base class for Table operations with full type safety.

    Provides common CRUD operations for database tables with validation,
    lifecycle hooks, and complete type safety across database models and
    their Pydantic representations.

    Type Parameters
    ---------------
    T : TypeVar, bound=Base
        Database model class (SQLAlchemy)
    ResponseT : TypeVar, bound=BaseModel
        Pydantic model class for responses
    CreateT : TypeVar, bound=BaseModel
        Pydantic model class for creation

    Important
    ---------
    All methods that modify the database (create_row, create_rows, etc.)
    DO NOT commit transactions. The caller MUST manage transactions using
    `async with session.begin()` or explicit commit/rollback.

    All create methods DO add objects to the session and flush them to
    get database-generated values (like auto-increment IDs), but the
    transaction must still be committed by the caller.

    Parameters
    ----------
    context : TableContext[T, ResponseT, CreateT]
        Shared configuration for this operation

    Examples
    --------
    >>> from myapp.models import User, UserResponse, UserCreate
    >>> context = TableContext(
    ...     db_class=User,
    ...     response_class=UserResponse,
    ...     create_class=UserCreate,
    ...     class_string="user"
    ... )
    >>> ops = TableOperations(context)
    >>>
    >>> async with get_session() as session:
    ...     async with session.begin():
    ...         user = await ops.create_row(
    ...             session,
    ...             username="alice",
    ...             email="alice@example.com"
    ...         )
    ...         # user is type User (T)
    ...         pydantic_user = ops.to_pydantic(user)
    ...         # pydantic_user is type UserResponse (ResponseT)
    """

    def __init__(self, context: TableContext[T, ResponseT, CreateT]) -> None:
        """
        Initialize operation with context.

        Parameters
        ----------
        context : TableContext[T, ResponseT, CreateT]
            Shared configuration for this operation, including database
            class and Pydantic model classes
        """
        self.ctx = context

    @forward_to_db_funcs(db_funcs.read, "get_row")
    async def get_row(self, session: AsyncSession, *args: Any, **kwargs: Any) -> T:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.read, "get_row_by_name")
    async def get_row_by_name(self, session: AsyncSession, *args: Any, **kwargs: Any) -> T:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.read, "get_rows")
    async def get_rows(self, session: AsyncSession, *args: Any, **kwargs: Any) -> Sequence[T]:  # type: ignore
        pass

    @forward_to_db_funcs_streaming(db_funcs.read, "get_rows_streaming")
    async def get_rows_streaming(  # pylint: disable=unused-argument
        self,
        session: AsyncSession,
        *args: Any,
        **kwargs: Any,
    ) -> AsyncIterator[T]:
        yield  # type: ignore

    @forward_to_db_funcs(db_funcs.read, "get_row_or_none")
    async def get_row_or_none(self, session: AsyncSession, *args: Any, **kwargs: Any) -> T | None:
        pass

    @forward_to_db_funcs(db_funcs.read, "count_rows")
    async def count_rows(self, session: AsyncSession, *args: Any, **kwargs: Any) -> int:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.read, "lookup_by_id_or_name")
    async def lookup_by_id_or_name(  # type: ignore
        self,
        session: AsyncSession,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[int, T | None]:
        pass

    @forward_to_db_funcs(db_funcs.update, "update_row")
    async def update_row(self, session: AsyncSession, *args: Any, **kwargs: Any) -> T:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.update, "update_rows")
    async def update_rows(self, session: AsyncSession, *args: Any, **kwargs: Any) -> list[T]:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.delete, "delete_row")
    async def delete_row(self, session: AsyncSession, *args: Any, **kwargs: Any) -> dict[str, Any] | None:
        pass

    @forward_to_db_funcs(db_funcs.delete, "delete_rows")
    async def delete_rows(
        self, session: AsyncSession, *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]] | None:
        pass

    @forward_to_db_funcs(db_funcs.delete, "bulk_delete_rows")
    async def bulk_delete_rows(self, session: AsyncSession, *args: Any, **kwargs: Any) -> int:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.filter, "filter_rows")
    async def filter_rows(self, session: AsyncSession, *args: Any, **kwargs: Any) -> Sequence[T]:  # type: ignore
        pass

    @forward_to_db_funcs_streaming(db_funcs.filter, "filter_rows_streaming")
    async def filter_rows_streaming(  # pylint: disable=unused-argument
        self, session: AsyncSession, *args: Any, **kwargs: Any
    ) -> AsyncIterator[T]:
        yield  # type: ignore

    @forward_to_db_funcs(db_funcs.filter, "count_filtered_rows")
    async def count_filtered_rows(self, session: AsyncSession, *args: Any, **kwargs: Any) -> int:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.filter, "filter_one")
    async def filter_one(self, session: AsyncSession, *args: Any, **kwargs: Any) -> T:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.filter, "filter_one_or_none")
    async def filter_one_or_none(self, session: AsyncSession, *args: Any, **kwargs: Any) -> T | None:
        pass

    @forward_to_db_funcs(db_funcs.filter, "find_by")
    async def find_by(self, session: AsyncSession, *args: Any, **kwargs: Any) -> Sequence[T]:  # type: ignore
        pass

    @forward_to_db_funcs(db_funcs.filter, "find_one_by")
    async def find_one_by(self, session: AsyncSession, *args: Any, **kwargs: Any) -> T:  # type: ignore
        pass

    async def get_create_kwargs(
        self,
        session: AsyncSession,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating an instance."""
        assert session
        return kwargs

    async def create_row(
        self,
        session: AsyncSession,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> T:
        """Create a single row in the database.

        The row is added to the session and flushed, but not committed.
        The caller is responsible for committing the transaction.

        Parameters
        ----------
        session : async_scoped_session
            DB session manager
        validate : bool, default=True
            Whether to validate input against Pydantic model (CreateT) before
            creation. If True and validation fails, raises ValidationError.
        **kwargs : Any
            Column names and their values for the new row

        Returns
        -------
        T
            Newly created row of the database model type with database-generated
            values (after flush)

        Raises
        ------
        TypeError
            If self.ctx.db_class does not inherit from rail_svc.db.base.Base
        ValidationError
            Pydantic validation failed on the input (if validate=True)
        IntegrityError
            Database integrity constraint violation

        Examples
        --------
        >>> from myapp.models import User, UserResponse, UserCreate
        >>> context = TableContext.from_db_class(User)
        >>> ops = TableOperations(context)
        >>>
        >>> async with get_session() as session:
        ...     async with session.begin():  # Caller manages transaction
        ...         user: User = await ops.create_row(
        ...             session,
        ...             username="alice",
        ...             email="alice@example.com"
        ...         )
        ...         # user is fully typed as User (T)
        ...         print(f"Created user {user.id}")
        ...         # Transaction commits automatically on context exit
        """
        ensure_base_inheritance(self.ctx.db_class)

        logger.debug("Creating row", table=self.ctx.db_class.__name__, fields=list(kwargs.keys()))

        # Validate input if requested
        if validate:
            try:
                self.ctx.create_class.model_validate(kwargs)
            except ValidationError as e:
                logger.warning(
                    "Validation failed in create_row",
                    table=self.ctx.db_class.__name__,
                    errors=e.errors(),
                )
                raise

        # update kwargs
        kwargs = await self.get_create_kwargs(session, **kwargs)

        # Pre-create hook
        kwargs = await self.ctx.db_class.pre_create_hook(session, kwargs)

        # Create the row
        row = self.ctx.db_class(**kwargs)

        # Add to session and flush to get DB-generated values
        session.add(row)
        await session.flush()

        logger.debug("Row created", table=self.ctx.db_class.__name__, row_id=getattr(row, "id", None))

        return row

    async def create_rows(
        self,
        session: AsyncSession,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> list[T]:
        """Create multiple rows in the database.

        All rows are added to the session and flushed together, but not
        committed. The caller is responsible for committing the transaction.

        Parameters
        ----------
        session : async_scoped_session
            DB session manager
        rows_data : Sequence[dict[str, Any]]
            Sequence of dictionaries, each containing column names and values
            for a new row
        validate : bool, default=True
            Whether to validate each row against Pydantic model (CreateT)
            before creation

        Returns
        -------
        list[T]
            List of newly created rows of the database model type with
            database-generated values

        Raises
        ------
        TypeError
            If self.ctx.db_class does not inherit from rail_svc.db.base.Base
        IntegrityError
            Integrity constraint violation (e.g., duplicate key, null constraint)
        ValidationError
            Pydantic validation failed on any row's input data (if validate=True)
        ValueError
            If rows_data is empty

        Notes
        -----
        - When validate=True, validation may fail for cases where the database
          provides default values. Consider setting validate=False in such cases.
        - All rows are flushed atomically within the session
        - The caller must commit the transaction for changes to persist
        - For very large datasets, consider batching the calls to this function.

        Examples
        --------
        >>> from myapp.models import User, UserResponse, UserCreate
        >>> context = TableContext.from_db_class(User)
        >>> ops = TableOperations(context)
        >>>
        >>> async with get_session() as session:
        ...     async with session.begin():  # Caller controls transaction
        ...         users: list[User] = await ops.create_rows(
        ...             session,
        ...             [
        ...                 {"username": "alice", "email": "alice@example.com"},
        ...                 {"username": "bob", "email": "bob@example.com"},
        ...                 {"username": "charlie", "email": "charlie@example.com"},
        ...             ]
        ...         )
        ...         # users is fully typed as list[User]
        ...         print(f"Created {len(users)} users")
        ...         # Transaction commits automatically on context exit
        """
        ensure_base_inheritance(self.ctx.db_class)

        if not rows_data:
            raise ValueError("rows_data cannot be empty")

        logger.debug("Creating multiple rows", table=self.ctx.db_class.__name__, count=len(rows_data))

        # Process all rows through hooks and validation
        processed_rows_data = []
        for idx, row_kwargs in enumerate(rows_data):
            try:
                # Validate if requested
                if validate:
                    try:
                        self.ctx.create_class.model_validate(row_kwargs)
                    except ValidationError as e:
                        logger.warning(
                            "Validation failed in create_rows",
                            table=self.ctx.db_class.__name__,
                            row_index=idx,
                            errors=e.errors(),
                        )
                        raise

                # update kwargs
                modified_kwargs = await self.get_create_kwargs(
                    session,
                    **row_kwargs.copy(),  # Copy to avoid modifying original
                )

                # Pre-create hook
                modified_kwargs = await self.ctx.db_class.pre_create_hook(
                    session,
                    modified_kwargs,
                )

                processed_rows_data.append(modified_kwargs)

            except Exception as e:
                logger.error(
                    "Failed to prepare row",
                    table=self.ctx.db_class.__name__,
                    row_index=idx,
                    error=str(e),
                )
                raise

        # Create row objects
        rows = [self.ctx.db_class(**data) for data in processed_rows_data]

        # Add all rows to session and flush to get DB-generated values
        session.add_all(rows)
        await session.flush()

        logger.debug("Rows created", table=self.ctx.db_class.__name__, count=len(rows))

        return rows

    async def create_rows_batched(
        self,
        session: AsyncSession,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
        batch_size: int = 1000,
    ) -> list[T]:
        """Create multiple rows in batches.

        Unlike create_rows(), this commits after each batch, so partial
        success is possible if a later batch fails.

        Parameters
        ----------
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
        if not rows_data:
            raise ValueError("rows_data cannot be empty")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        logger.info(
            f"Creating rows in batches: {len(rows_data)} {batch_size}",
        )

        all_rows = []

        # Process in batches
        for batch_start in range(0, len(rows_data), batch_size):
            batch_end = min(batch_start + batch_size, len(rows_data))
            batch_data = rows_data[batch_start:batch_end]

            logger.debug(
                f"Processing batch {batch_start} {batch_end}",
            )

            try:
                batch_rows: list = await self.create_rows(session, batch_data, validate=validate)
                all_rows.extend(batch_rows)

            except Exception as uexc:
                logger.error(
                    f"Batch failed {batch_start} {batch_end}",
                    error=uexc,
                )
                raise

        logger.info("All batches completed")
        return all_rows

    async def bulk_insert_rows(
        self,
        session: AsyncSession,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> int:
        """Bulk insert rows using SQLAlchemy's bulk operations.

        This is much faster than create_rows() but doesn't return the
        created objects or handle get_create_kwargs() preprocessing.

        Parameters
        ----------
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
        if not rows_data:
            raise ValueError("rows_data cannot be empty")

        logger.debug(f"Bulk inserting rows {len(rows_data)}")

        # Validate all rows
        if validate:
            for idx, row_kwargs in enumerate(rows_data):
                try:
                    self.ctx.create_class.model_validate(row_kwargs)
                except ValidationError:
                    logger.warning(
                        f"Validation failed in bulk_insert_rows {idx}",
                    )
                    raise

        try:
            # Use insert statement for maximum performance
            stmt = insert(self.ctx.db_class).values(rows_data)
            await session.execute(stmt)
            await session.commit()

            logger.info(f"Bulk insert completed {len(rows_data)}")
            return len(rows_data)

        except IntegrityError as uexc:
            await session.rollback()
            logger.error(
                f"Integrity error during bulk insert {len(rows_data)}",
                error=uexc,
            )
            raise

    def _validate_path_security(self, path: str) -> Path:
        """Validate path doesn't escape archive directory.

        Parameters
        ----------
        path : str
            Relative path to validate

        Returns
        -------
        Path
            Resolved, validated absolute path

        Raises
        ------
        ValueError
            If path would escape archive directory or contains
            invalid characters

        Notes
        -----
        This method requires `global_config.storage.archive` to be available.
        Consider moving to a specialized subclass if not all tables need
        file path validation.

        Examples
        --------
        >>> ops = TableOperations(context)
        >>> safe_path = ops._validate_path_security("data/users/alice.json")
        >>> # Returns: /var/archive/data/users/alice.json
        >>>
        >>> # This will raise ValueError
        >>> ops._validate_path_security("../../../etc/passwd")
        """

        # Check for obvious traversal attempts
        if ".." in path or path.startswith("/") or path.startswith("\\"):
            logger.error(
                "Invalid path detected",
                table=self.ctx.db_class.__name__,
                path=path,
            )
            raise ValueError(f"Invalid path: {path}")

        # Check path length
        if len(path) > 255:
            logger.error(
                "Path too long",
                table=self.ctx.db_class.__name__,
                path_length=len(path),
            )
            raise ValueError(f"Path too long: {len(path)} characters")

        # Check for null bytes (security issue)
        if "\x00" in path:
            logger.error(
                "Path contains null bytes",
                table=self.ctx.db_class.__name__,
            )
            raise ValueError("Path contains null bytes")

        # Resolve and check path is within archive
        archive_path = Path(global_config.storage.archive).resolve()
        fullpath = (archive_path / path).resolve()

        try:
            fullpath.relative_to(archive_path)
        except ValueError as uexc:
            if FORBID_TRAVERSAL:
                logger.error(
                    "Path traversal attempt detected",
                    table=self.ctx.db_class.__name__,
                    attempted_path=str(path),
                    archive_path=str(archive_path),
                    resolved_path=str(fullpath),
                    error=uexc,
                )
                raise ValueError(f"Path {path} would escape archive directory") from None

        return fullpath

    def to_pydantic(self, row: T) -> ResponseT:
        """Convert a database row to its Pydantic model representation.

        Transforms a SQLAlchemy model instance into the corresponding Pydantic
        response model (ResponseT) defined in the table context. This is
        useful for API responses, validation, and serialization.

        Parameters
        ----------
        row : T
            Database row instance (SQLAlchemy model)

        Returns
        -------
        ResponseT
            Pydantic model instance of the response type, containing the
            same data as the database row

        Raises
        ------
        ValidationError
            If the database row contains data that doesn't validate against
            the Pydantic model schema
        AttributeError
            If the db_class doesn't implement the to_pydantic method

        Examples
        --------
        >>> from myapp.models import User, UserResponse, UserCreate
        >>> context = TableContext(
        ...     db_class=User,
        ...     response_class=UserResponse,
        ...     create_class=UserCreate,
        ...     class_string="user"
        ... )
        >>> ops = TableOperations(context)
        >>>
        >>> async with get_session() as session:
        ...     user_row: User = await ops.get_row(session, user_id=1)
        ...     user_pydantic: UserResponse = ops.to_pydantic(user_row)
        ...     # Fully typed as UserResponse
        ...     print(user_pydantic.model_dump_json())
        {"id": 1, "username": "alice", "email": "alice@example.com"}

        See Also
        --------
        to_pydantic_list : Convert multiple rows at once
        to_pydantic_dict : Convert a row to a dictionary
        """
        return cast(ResponseT, self.ctx.db_class.to_pydantic(row))

    def to_pydantic_list(self, rows: list[T]) -> list[ResponseT]:
        """Convert a list of database rows to Pydantic model representations.

        Transforms multiple SQLAlchemy model instances into their corresponding
        Pydantic response models (ResponseT). This is a batch version of
        to_pydantic() that may be more efficient for converting multiple rows.

        Parameters
        ----------
        rows : list[T]
            List of database row instances (SQLAlchemy models)

        Returns
        -------
        list[ResponseT]
            List of Pydantic response model instances, in the same order
            as the input rows

        Raises
        ------
        ValidationError
            If any database row contains data that doesn't validate against
            the Pydantic model schema
        AttributeError
            If the db_class doesn't implement the to_pydantic_list method

        Notes
        -----
        - Empty input list returns empty output list
        - Order is preserved from input to output
        - Some implementations may optimize batch conversion vs. repeated
          single conversions

        Examples
        --------
        >>> from myapp.models import User, UserResponse, UserCreate
        >>> context = TableContext(
        ...     db_class=User,
        ...     response_class=UserResponse,
        ...     create_class=UserCreate,
        ...     class_string="user"
        ... )
        >>> ops = TableOperations(context)
        >>>
        >>> async with get_session() as session:
        ...     user_rows: list[User] = await ops.get_rows(session, limit=10)
        ...     user_models: list[UserResponse] = ops.to_pydantic_list(user_rows)
        ...     # Fully typed as list[UserResponse]
        ...     for user in user_models:
        ...         print(user.username)
        alice
        bob
        charlie

        >>> # Useful for API responses with full type safety
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>>
        >>> @app.get("/users", response_model=list[UserResponse])
        >>> async def get_users() -> list[UserResponse]:
        ...     async with get_session() as session:
        ...         rows = await ops.get_rows(session)
        ...         return ops.to_pydantic_list(rows)
        ...         # Return type is correctly typed

        See Also
        --------
        to_pydantic : Convert a single row
        to_pydantic_dict_list : Convert multiple rows to dictionaries
        """
        return cast(list[ResponseT], self.ctx.db_class.to_pydantic_list(rows))

    def to_pydantic_dict(self, row: T) -> dict[str, Any]:
        """Convert a database row to a dictionary via Pydantic validation.

        Transforms a SQLAlchemy model instance into a dictionary by first
        converting to a Pydantic response model (ResponseT, ensuring validation)
        and then serializing to a dict. The result is suitable for JSON
        serialization, logging, or other dictionary-based operations.

        Parameters
        ----------
        row : T
            Database row instance (SQLAlchemy model)

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the row, with keys corresponding to
            the Pydantic response model fields and values appropriately
            serialized

        Raises
        ------
        ValidationError
            If the database row contains data that doesn't validate against
            the Pydantic model schema
        AttributeError
            If the db_class doesn't implement the to_pydantic_dict method

        Notes
        -----
        - The output includes only fields defined in ResponseT,
          not all SQLAlchemy model attributes
        - Complex types (datetime, UUID, etc.) are serialized according to
          Pydantic's serialization rules
        - This method goes through Pydantic validation, unlike direct
          SQLAlchemy to dict conversion

        Examples
        --------
        >>> from myapp.models import User, UserResponse, UserCreate
        >>> context = TableContext(
        ...     db_class=User,
        ...     response_class=UserResponse,
        ...     create_class=UserCreate,
        ...     class_string="user"
        ... )
        >>> ops = TableOperations(context)
        >>>
        >>> async with get_session() as session:
        ...     user_row: User = await ops.get_row(session, user_id=1)
        ...     user_dict: dict[str, Any] = ops.to_pydantic_dict(user_row)
        ...     print(user_dict)
        {'id': 1, 'username': 'alice', 'email': 'alice@example.com',
         'created_at': '2025-01-15T10:30:00'}

        >>> # Useful for structured logging
        >>> logger.info("User details", **ops.to_pydantic_dict(user_row))

        >>> # Or JSON responses without FastAPI's automatic conversion
        >>> import json
        >>> json_str = json.dumps(ops.to_pydantic_dict(user_row))

        See Also
        --------
        to_pydantic : Get the Pydantic model instead of dict
        to_pydantic_dict_list : Convert multiple rows to dictionaries
        """
        return self.ctx.db_class.to_pydantic_dict(row)

    def to_pydantic_dict_list(self, rows: list[T]) -> list[dict[str, Any]]:
        """Convert a list of database rows to dictionaries via Pydantic validation.

        Transforms multiple SQLAlchemy model instances into dictionaries by first
        converting each to a Pydantic response model (ResponseT, ensuring
        validation) and then serializing to dicts. This is a batch version of
        to_pydantic_dict().

        Parameters
        ----------
        rows : list[T]
            List of database row instances (SQLAlchemy models)

        Returns
        -------
        list[dict[str, Any]]
            List of dictionary representations, each with keys corresponding to
            the Pydantic response model fields and values appropriately
            serialized, in the same order as the input rows

        Raises
        ------
        ValidationError
            If any database row contains data that doesn't validate against
            the Pydantic model schema
        AttributeError
            If the db_class doesn't implement the to_pydantic_dict_list method

        Notes
        -----
        - Empty input list returns empty output list
        - Order is preserved from input to output
        - Each dictionary includes only fields defined in ResponseT
        - Complex types are serialized according to Pydantic's rules
        - Some implementations may optimize batch conversion

        Examples
        --------
        >>> from myapp.models import User, UserResponse, UserCreate
        >>> context = TableContext(
        ...     db_class=User,
        ...     response_class=UserResponse,
        ...     create_class=UserCreate,
        ...     class_string="user"
        ... )
        >>> ops = TableOperations(context)
        >>>
        >>> async with get_session() as session:
        ...     user_rows: list[User] = await ops.get_rows(session, limit=3)
        ...     user_dicts: list[dict[str, Any]] = ops.to_pydantic_dict_list(user_rows)
        ...     for user_dict in user_dicts:
        ...         print(f"{user_dict['username']}: {user_dict['email']}")
        alice: alice@example.com
        bob: bob@example.com
        charlie: charlie@example.com

        >>> # Useful for bulk operations or exports
        >>> import csv
        >>> with open('users.csv', 'w') as f:
        ...     writer = csv.DictWriter(f, fieldnames=['id', 'username', 'email'])
        ...     writer.writeheader()
        ...     writer.writerows(ops.to_pydantic_dict_list(user_rows))

        >>> # Or for structured logging
        >>> logger.info("Batch user export", users=ops.to_pydantic_dict_list(user_rows))

        >>> # YAML/JSON export
        >>> import yaml
        >>> yaml.dump(ops.to_pydantic_dict_list(user_rows), open('users.yaml', 'w'))

        See Also
        --------
        to_pydantic_dict : Convert a single row to dictionary
        to_pydantic_list : Get Pydantic models instead of dicts
        """
        return self.ctx.db_class.to_pydantic_dict_list(rows)


class FileValidatedOperations[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    TableOperations[T, ResponseT, CreateT]
):
    """Base class for table operations with file-backed data validation.

    Provides common functionality for tables that store references to
    files requiring validation, including:

    - Path security validation (directory traversal protection)
    - File existence checking
    - Async file reading via executor to avoid blocking
    - Standardized error handling for I/O and format errors

    Subclasses must implement:
        - get_file_length(path): Extract object count from file
        - get_create_kwargs(): Handle foreign key resolution and
          call _process_path() with appropriate reference object

    Type Parameters
    ---------------
    T : TypeVar, bound=Base
        Database model class for file-backed data
    ResponseT : TypeVar, bound=BaseModel
        Pydantic response model class
    CreateT : TypeVar, bound=BaseModel
        Pydantic creation model class (must support path and
        n_objects fields)

    Examples
    --------
    >>> class DatasetOperations(FileValidatedOperations[...]):
    ...     def get_file_length(self, path: Path) -> int:
    ...         return tables_io.hdf5.get_input_data_length(str(path))
    ...
    ...     async def get_create_kwargs(self, session, **kwargs):
    ...         # Resolve foreign keys, then validate file
    ...         catalog_tag_id, catalog_tag = await lookup_by_id_or_name(...)
    ...         n_objects = await self._process_path(
    ...             path, catalog_tag, validate_file, extra_kwargs
    ...         )
    ...         return {"catalog_tag_id": catalog_tag_id, "n_objects": n_objects}

    See Also
    --------
    TableOperations : Base class for all table operations
    """

    @abstractmethod
    def get_file_length(self, path: Path) -> int:
        """Get number of objects in file. Implement in subclass."""

    async def _process_path(
        self,
        path: str | None,
        reference_obj: Base | None,
        *,
        validate_file: bool,
        extra_kwargs: dict[str, Any],
    ) -> int:
        """Process file path and determine n_objects.

        Parameters
        ----------
        path
            File path (relative to archive)
        reference_obj
            Reference object for validation (may be None). Typically used
            to validate file contents match expected schema.
        validate_file
            Whether to validate file
        extra_kwargs
            Additional kwargs that may contain n_objects

        Returns
        -------
        int
            Number of objects in file

        Raises
        ------
        ValueError
            If path invalid, validation fails, or n_objects not provided
            when needed
        FileNotFoundError
            If file doesn't exist when validation enabled
        """
        if path is None:
            # No path - must have n_objects in extra_kwargs
            n_objects = extra_kwargs.get("n_objects")
            if n_objects is None:
                logger.warning(
                    "No path or n_objects provided",
                    table=self.ctx.db_class.__name__,
                )
                raise ValueError("Either 'path' or 'n_objects' must be provided")
            return n_objects

        if not validate_file:
            # Path provided but validation disabled - use provided n_objects
            n_objects = extra_kwargs.get("n_objects")
            if n_objects is None:
                logger.warning(
                    "File validation disabled but n_objects not provided",
                    table=self.ctx.db_class.__name__,
                    path=path,
                )
                raise ValueError("When validate_file=False, 'n_objects' must be provided")
            return n_objects

        # Validate path and file
        fullpath = self._validate_path_security(path)
        n_objects = await self.validate_data_for_path(fullpath, reference_obj)

        # Check against user-provided value if present
        user_n_objects = extra_kwargs.get("n_objects")
        if user_n_objects is not None and user_n_objects != n_objects:
            logger.warning(
                "Provided n_objects doesn't match file content",
                table=self.ctx.db_class.__name__,
                path=path,
                provided=user_n_objects,
                actual=n_objects,
            )

        return n_objects

    async def validate_data_for_path(
        self,
        path: Path,
        reference_obj: Base | None = None,
    ) -> int:
        """Validate that data file exists and can be read.

        This method performs synchronous I/O in an executor to avoid
        blocking the event loop.

        Parameters
        ----------
        path
            Absolute path to the data file
        reference_obj
            Reference object for future validation (currently unused
            but reserved for validating data matches expected schema)

        Returns
        -------
        int
            Number of objects in the file

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist
        ValueError
            If the file cannot be read or has invalid format

        Notes
        -----
        Future enhancement: Use reference_obj to validate that the data
        format matches the expected schema for this object type.
        """
        # Reserved for future use: validate data matches reference_obj schema
        _ = reference_obj

        asnyc_path = anyio.Path(path)
        if not await asnyc_path.exists():
            logger.error(
                "Data file not found",
                table=self.ctx.db_class.__name__,
                path=str(path),
            )
            raise FileNotFoundError(f"File {path} not found")

        loop = asyncio.get_event_loop()
        try:
            n_objects = await loop.run_in_executor(None, self.get_file_length, path)
        except OSError as exc:
            # File system errors
            logger.error(
                "Failed to read data file",
                table=self.ctx.db_class.__name__,
                path=str(path),
                error=str(exc),
                error_type="io_error",
            )
            raise ValueError(f"Could not read data from {path}: {exc}") from exc
        except ValueError as exc:
            # Data format errors
            logger.error(
                "Invalid data format in file",
                table=self.ctx.db_class.__name__,
                path=str(path),
                error=str(exc),
                error_type="format_error",
            )
            raise ValueError(f"Invalid data format in {path}: {exc}") from exc
        except Exception as exc:
            # Unexpected errors - log with full traceback and re-raise
            logger.exception(
                "Unexpected error reading data file",
                table=self.ctx.db_class.__name__,
                path=str(path),
            )
            raise ValueError(f"Unexpected error reading {path}: {exc}") from exc

        logger.debug(
            "Data file validated",
            table=self.ctx.db_class.__name__,
            path=str(path),
            n_objects=n_objects,
        )

        return n_objects


def create_operations[T: Base, ResponseT: BaseModel, CreateT: BaseModel](
    db_class: type[T],
    response_class: type[ResponseT],
    create_class: type[CreateT],
) -> TableOperations[T, ResponseT, CreateT]:
    """Create fully-typed TableOperations from explicit classes.

    Parameters
    ----------
    db_class : type[T]
        SQLAlchemy database model class
    response_class : type[ResponseT]
        Pydantic response model class
    create_class : type[CreateT]
        Pydantic creation model class

    Returns
    -------
    TableOperations[T, ResponseT, CreateT]
        Fully typed operations instance
    """
    context = TableContext(
        db_class=db_class,
        response_class=response_class,
        create_class=create_class,
        class_string=db_class.class_string(),
    )
    return TableOperations(context)
