from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Generic, TypeVar
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import async_scoped_session

from ..db.base import Base, ensure_base_inheritance
from ..config import config as global_config

logger = logging.getLogger(__name__)

# Type variable bound to SQLAlchemy Base
T = TypeVar("T", bound=Base)


@dataclass
class RowCreateContext(Generic[T]):
    """
    Shared context for creating rows.

    Encapsulates the common configuration needed for all tables.

    Parameters
    ----------
    db_class : type[T]
        Database model class (SQLAlchemy)
    create_class : type[BaseModel]
        Pydantic model class for validating creation data
    class_string : str
        String identifier for the class
    """

    db_class: type[T]
    create_class: type[BaseModel]
    class_string: str

    @classmethod
    def from_db_class(cls, db_class: type[T]) -> RowCreateContext[T]:
        """
        Create context from database class using conventions.

        Parameters
        ----------
        db_class : type[T]
            Database model class that implements pydantic_create_class()

        Returns
        -------
        RowCreateContext[T]
            Configured RowCreateContext

        Raises
        ------
        AttributeError
            If db_class doesn't implement required methods
        """
        # Validate required methods exist
        if not hasattr(db_class, "pydantic_create_class"):
            raise AttributeError(f"{db_class.__name__} must implement pydantic_create_class() method")
        if not hasattr(db_class, "class_string"):
            raise AttributeError(f"{db_class.__name__} must implement class_string() method")

        return cls(
            db_class=db_class,
            create_class=db_class.pydantic_create_class(),
            class_string=db_class.class_string(),
        )


class RowCreateBase(Generic[T]):
    """Base class for row creation operations.

    This class handles creating database rows with validation and hooks,
    but does not manage transactions. The caller is responsible for
    transaction boundaries (begin/commit/rollback).
    """

    def __init__(self, context: RowCreateContext[T]) -> None:
        """
        Initialize operation with context.

        Parameters
        ----------
        context : RowCreateContext[T]
            Shared configuration for this operation
        """
        self.ctx = context

    async def create_row(
        self,
        session: async_scoped_session,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> T:
        """Create a single row in the database.

        The row is added to the session and flushed, but not committed.
        The caller is responsible for committing the transaction.

        Parameters
        ----------
        session
            DB session manager
        validate
            Whether to validate input against Pydantic model before creation.
            If True and validation fails, raises ValidationError.
        **kwargs
            Column names and their values for the new row

        Returns
        -------
        T
            Newly created row with database-generated values (after flush)

        Raises
        ------
        TypeError
            If self.ctx.db_class does not inherit from rail_svc.db.base.Base
        ValidationError
            Pydantic validation failed on the input
        IntegrityError
            Database integrity constraint violation

        Examples
        --------
        >>> from myapp.models import User
        >>> creator = RowCreateBase(RowCreateContext.from_db_class(User))
        >>>
        >>> async with get_session() as session:
        ...     async with session.begin():  # Caller manages transaction
        ...         user = await creator.create_row(
        ...             session,
        ...             username="alice",
        ...             email="alice@example.com"
        ...         )
        ...         # Can do more work here
        ...         # Transaction commits automatically on context exit
        """
        ensure_base_inheritance(self.ctx.db_class)

        logger.debug("Creating row", table=self.ctx.db_class.__name__, fields=list(kwargs.keys()))

        # Pre-create hook (can modify kwargs)
        kwargs = await self.ctx.db_class.pre_create_hook(session, kwargs)

        # Validate input if requested (after hook modifications)
        if validate:
            try:
                pydantic_class = self.ctx.create_class
                pydantic_class.model_validate(kwargs)
            except ValidationError as e:
                logger.warning(
                    "Validation failed in create_row",
                    table=self.ctx.db_class.__name__,
                    errors=e.errors(),
                )
                raise

        # Create the row
        row = self.ctx.db_class(**kwargs)

        return row

    async def create_rows(
        self,
        session: async_scoped_session,
        rows_data: Sequence[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> list[T]:
        """Create multiple rows in the database.

        All rows are added to the session and flushed together, but not
        committed. The caller is responsible for committing the transaction.

        Parameters
        ----------
        session
            DB session manager
        rows_data
            Sequence of dictionaries, each containing column names and values
            for a new row
        validate
            Whether to validate each row against Pydantic model before creation

        Returns
        -------
        list[T]
            List of newly created rows with database-generated values

        Raises
        ------
        TypeError
            If self.ctx.db_class does not inherit from rail_svc.db.base.Base
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
        - All rows are flushed atomically within the session
        - The caller must commit the transaction for changes to persist
        - For very large datasets, consider batching the calls to this function.

        Examples
        --------
        >>> from myapp.models import User
        >>> creator = RowCreateBase(RowCreateContext.from_db_class(User))
        >>>
        >>> async with get_session() as session:
        ...     async with session.begin():  # Caller controls transaction
        ...         users = await creator.create_rows(
        ...             session,
        ...             [
        ...                 {"username": "alice", "email": "alice@example.com"},
        ...                 {"username": "bob", "email": "bob@example.com"},
        ...                 {"username": "charlie", "email": "charlie@example.com"},
        ...             ]
        ...         )
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
                # Pre-create hook
                modified_kwargs = await self.ctx.db_class.pre_create_hook(
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
        return rows

    def _validate_path_security(self, path: str) -> Path:
        """Validate path doesn't escape archive directory.

        Parameters
        ----------
        path
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
