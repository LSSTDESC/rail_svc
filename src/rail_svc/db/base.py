"""Database model base for rail-svc applications.

This module provides the declarative base class for all ORM models,
with consistent schema configuration and naming conventions.
"""

from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import DeclarativeBase

from ..config import config

# Define TypeVar for generic typing
T = TypeVar("T", bound="Base")

# Default pagination limit - can be overridden per-table if needed
DEFAULT_PAGINATION_LIMIT = 100

# Naming convention for constraints (helps with Alembic migrations)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for all database models.

    Provides:
        - Schema assignment from configuration
        - Consistent constraint naming for migrations
        - Shared metadata for all models
        - Required interface for Pydantic integration
    """

    metadata: ClassVar[MetaData] = MetaData(
        schema=config.db.table_schema if config.db.table_schema else None,
        naming_convention=NAMING_CONVENTION
    )

    @classmethod
    def class_string(cls) -> str:
        """Name to use for help functions and descriptions.

        Returns
        -------
        str
            The class name
        """
        return cls.__name__

    @classmethod
    def pydantic_create_class(cls) -> type[BaseModel]:
        """Pydantic model used to create rows in this table.

        Subclasses must implement this to specify their associated
        Pydantic model for creation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement pydantic_create_class() "
            f"to return a Pydantic model for row creation"
        )

    @classmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Pydantic model class for this table.

        Subclasses must implement this to specify their associated
        Pydantic model for serialization/validation.

        Returns
        -------
        type[BaseModel]
            The Pydantic model class
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement pydantic_model_class() "
            f"to return a Pydantic model for row serialization/validation"
        )

    @classmethod
    def get_pagination_limit(cls) -> int:
        """Get the default pagination limit for this table.

        Subclasses can override this to set table-specific limits.

        Returns
        -------
        int
            Maximum number of rows to return in a single query
        """
        return DEFAULT_PAGINATION_LIMIT

    @classmethod
    async def pre_delete_hook(
        cls: type[T],
        session: async_scoped_session,  # pylint: disable=unused-argument
        row: T,  # pylint: disable=unused-argument
    ) -> None:
        """Hook called during delete_row, BEFORE deletion

        Subclasses can override this to perform cleanup operations,
        cascade deletes, or other pre-deletion tasks. This hook receives
        the full row object, so it can access any field data needed.

        This is called within the transaction but before the delete is
        executed, so any errors raised here will prevent the deletion.

        Parameters
        ----------
        session
            DB session manager

        row
            The row object that will be deleted (with all fields accessible)
        """
        return None

    @classmethod
    async def after_delete_hook(
        cls: type[T],
        session: async_scoped_session,  # pylint: disable=unused-argument
        row_id: int,  # pylint: disable=unused-argument
        deleted_row_data: dict[str, Any] | None = None,  # pylint: disable=unused-argument
    ) -> None:
        """Hook called during delete_row, AFTER successful deletion

        Subclasses can override this to perform cleanup operations that
        should happen after the row is deleted, such as:
        - Cleaning up external resources (files, cache entries, etc.)
        - Logging/auditing
        - Triggering notifications
        - Cleanup that shouldn't prevent deletion even if it fails

        This hook is called AFTER the deletion has been flushed to the
        database but before the transaction commits. If this hook raises
        an exception, the entire transaction (including the delete) will
        be rolled back.

        If you need cleanup that should happen even if the delete fails,
        or cleanup that shouldn't prevent the delete, consider handling
        it outside the transaction or using try/except within this hook.

        Parameters
        ----------
        session
            DB session manager

        row_id
            The ID of the row that was deleted

        deleted_row_data
            Optional dictionary containing the row data before deletion.
            This is useful if you need to access field values after the
            row is deleted.

        Examples
        --------
        >>> class User(Base, RowMixin):
        ...     @classmethod
        ...     async def after_delete_hook(cls, session, row_id, deleted_row_data):
        ...         # Clean up user's uploaded files
        ...         if deleted_row_data and 'profile_image_path' in deleted_row_data:
        ...             try:
        ...                 await delete_file(deleted_row_data['profile_image_path'])
        ...             except FileNotFoundError:
        ...                 logger.warning(f"Profile image already deleted for user {row_id}")
        ...
        ...         # Clear cache
        ...         await cache.delete(f"user:{row_id}")
        ...
        ...         # Send audit event
        ...         await audit_log.record_deletion("User", row_id)

        Notes
        -----
        - This hook runs AFTER deletion, so the row is no longer in the database
        - Any exceptions raised will roll back the entire transaction
        - For operations that should succeed even if the hook fails, use try/except
        - **Hook implementations should be idempotent** where possible
        - This is called within the same transaction as the delete operation
        """
        return None

    @classmethod
    async def pre_create_hook(
        cls: type[T],
        session: async_scoped_session,  # pylint: disable=unused-argument
        data: dict[str, Any],  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        """Hook called during create_row, BEFORE row creation.

        Subclasses can override this to:
        - Validate or transform input data
        - Add computed/derived fields
        - Perform authorization checks
        - Look up foreign key values

        This is called within the transaction but before the row is
        instantiated, so any errors raised here will prevent creation.

        Parameters
        ----------
        session
            DB session manager for performing additional queries
        data
            Dictionary of field names and values for the new row.
            Can be modified before returning.

        Returns
        -------
        dict[str, Any]
            Modified or unchanged data dictionary used to create the row.
            The returned dict is what actually gets passed to the model constructor.

        Examples
        --------
        >>> class User(Base):
        ...     @classmethod
        ...     async def pre_create_hook(cls, session, data):
        ...         # Add computed field
        ...         if 'email' in data:
        ...             data['email_lower'] = data['email'].lower()
        ...
        ...         # Add timestamp
        ...         data['registered_at'] = datetime.utcnow()
        ...
        ...         # Validate business rules
        ...         if data.get('age', 0) < 18:
        ...             raise ValueError("Users must be 18 or older")
        ...
        ...         return data

        Notes
        -----
        - This hook runs BEFORE row creation
        - Must return the data dictionary (possibly modified)
        - Any exceptions raised will prevent row creation
        - This is called within the same transaction as the create operation
        """
        return data

    @classmethod
    async def after_create_hook(
        cls: type[T],
        session: async_scoped_session,  # pylint: disable=unused-argument
        row: T,  # pylint: disable=unused-argument
    ) -> None:
        """Hook called during create_row, AFTER successful creation.

        Subclasses can override this to perform operations after the row
        is created and flushed to the database, such as:
        - Creating related records
        - Updating caches
        - Sending notifications
        - Logging/auditing
        - Triggering background tasks

        This hook is called AFTER the row has been flushed to the database
        (so it has an ID and all database-generated values) but before the
        transaction commits. If this hook raises an exception, the entire
        transaction (including the create) will be rolled back.

        Parameters
        ----------
        session
            DB session manager for performing additional database operations
        row
            The newly created row object with all fields populated,
            including database-generated values like auto-increment IDs

        Examples
        --------
        >>> class User(Base):
        ...     @classmethod
        ...     async def after_create_hook(cls, session, row):
        ...         # Create default user preferences
        ...         prefs = UserPreferences(user_id=row.id)
        ...         async with session() as s:
        ...             s.add(prefs)
        ...             await s.flush()
        ...
        ...         # Warm cache
        ...         await cache.set(f"user:{row.id}", row.to_dict())
        ...
        ...         # Send welcome email (non-blocking)
        ...         await queue.enqueue('send_welcome_email', user_id=row.id)
        ...
        ...         # Audit log
        ...         await audit_log.record_creation("User", row.id, row.email)

        Notes
        -----
        - This hook runs AFTER creation and flush, so row.id is available
        - Any exceptions raised will roll back the entire transaction
        - For operations that shouldn't block creation, use try/except
        - **Hook implementations should be idempotent** where possible
        - This is called within the same transaction as the create operation
        """
        return None


def ensure_base_inheritance(cls: type[Any]) -> None:
    """Raise TypeError if a class does not inherit from Base.

    Parameters
    ----------
    cls
        The class to check for Base inheritance

    Raises
    ------
    TypeError
        If cls does not inherit from Base

    Examples
    --------
    >>> class MyModel(Base):
    ...     pass
    >>> ensure_base_inheritance(MyModel)  # No error
    >>>
    >>> class BadModel:
    ...     pass
    >>> ensure_base_inheritance(BadModel)  # Raises TypeError
    """
    if not issubclass(cls, Base):
        raise TypeError(f"Class {cls.__name__} must inherit from rail_svc.db.base.Base")
