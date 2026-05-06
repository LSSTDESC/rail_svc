from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from structlog import get_logger

from ..db.base import ensure_base_inheritance, Base
from . import read, create, update, delete, filter as filter_ops

logger = get_logger(__name__)

# Type variables
T = TypeVar("T", bound=Base)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
CreateT = TypeVar("CreateT", bound=BaseModel)


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
    def from_db_class(cls, db_class: type[T]) -> TableContext[T, BaseModel, BaseModel]:
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
        if not hasattr(db_class, "pydantic_create_class"):
            raise AttributeError(f"{db_class.__name__} must implement pydantic_create_class() method")
        if not hasattr(db_class, "class_string"):
            raise AttributeError(f"{db_class.__name__} must implement class_string() method")

        # Get the classes
        response_class = db_class.pydantic_model_class()
        create_class = db_class.pydantic_create_class()

        # Validate return types
        if not (isinstance(response_class, type) and issubclass(response_class, BaseModel)):
            raise TypeError(
                f"{db_class.__name__}.pydantic_model_class() must return a "
                f"BaseModel subclass, got {type(response_class)}"
            )
        if not (isinstance(create_class, type) and issubclass(create_class, BaseModel)):
            raise TypeError(
                f"{db_class.__name__}.pydantic_create_class() must return a "
                f"BaseModel subclass, got {type(create_class)}"
            )

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

    _DELEGATED_METHODS = {
        # CREATE operations
        "create": ["create_rows_batched", "bulk_insert_rows"],
        # READ operations
        "read": [
            "get_row",
            "get_row_by_name",
            "get_rows",
            "get_rows_streaming",
            "get_row_or_none",
            "count_rows",
            "lookup_by_id_or_name",
        ],
        # UPDATE operations
        "update": [
            "update_row",
            "update_rows",
        ],
        # DELETE operations
        "delete": ["delete_row", "delete_rows", "bulk_delete_rows"],
    }

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
        self._bind_delegated_methods()

    def _bind_delegated_methods(self) -> None:
        """
        Bind all delegated methods to this instance.

        This method dynamically attaches CRUD operations from the db_funcs
        modules to this instance, pre-binding them with the database class
        from the context.

        Notes
        -----
        The delegated methods are defined in _DELEGATED_METHODS and organized
        by operation type (create, read, update, delete).
        """
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
                bound_func = functools.partial(func, self.ctx.db_class)
                functools.update_wrapper(bound_func, func)
                setattr(self, func_name, bound_func)

    async def get_create_kwargs(
        self,
        session: AsyncSession,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare kwargs for creating an instance."""
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
        kwargs = self.get_create_kwargs(session, **kwargs)

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

                # update kwargs
                modified_kwargs = await self.get_create_kwargs(
                    session, row_kwargs.copy()  # Copy to avoid modifying original
                )

                # Validate if requested
                if validate:
                    try:
                        self.ctx.create_class.model_validate(modified_kwargs)
                    except ValidationError as e:
                        logger.warning(
                            "Validation failed in create_rows",
                            table=self.ctx.db_class.__name__,
                            row_index=idx,
                            errors=e.errors(),
                        )
                        raise

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
        # Import here to avoid circular imports
        from ...config import global_config

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
        except ValueError:
            logger.error(
                "Path traversal attempt detected",
                table=self.ctx.db_class.__name__,
                attempted_path=str(path),
                archive_path=str(archive_path),
                resolved_path=str(fullpath),
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
        return self.ctx.db_class.to_pydantic(row)

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
        return self.ctx.db_class.to_pydantic_list(rows)

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
