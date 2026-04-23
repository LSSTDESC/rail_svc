"""Database model base for rail-svc applications.

This module provides the declarative base class for all ORM models,
with consistent schema configuration and naming conventions.
"""

from abc import abstractmethod
from typing import Any, ClassVar, TypeVar

from pydantic import BaseModel
from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import async_scoped_session
from sqlalchemy.orm import DeclarativeBase

from ..config import config


# Define TypeVar for generic typing
T = TypeVar('T', bound='Base')

# Default pagination limit - can be overridden per-table if needed
DEFAULT_PAGINATION_LIMIT = 100

# Naming convention for constraints (helps with Alembic migrations)
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s"
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
        schema=config.db.table_schema or None,
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
    @abstractmethod
    def pydantic_model_class(cls) -> type[BaseModel]:
        """Pydantic model class for this table.
        
        Subclasses must implement this to specify their associated
        Pydantic model for serialization/validation.
        
        Returns
        -------
        type[BaseModel]
            The Pydantic model class
        """
        ...

    @classmethod
    async def get_create_kwargs(
        cls: type[T],
        session: async_scoped_session,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Get additional keywords needed to create a row.

        This can be overridden by subclasses to add computed fields,
        foreign key lookups, or other preprocessing before row creation.

        The default implementation returns the input unchanged.

        Parameters
        ----------
        session
            DB session manager
        **kwargs
            Original column names and values for the new row

        Returns
        -------
        dict[str, Any]
            Keywords needed to create a new row (may include additional fields)

        Examples
        --------
        >>> class User(Base):
        ...     @classmethod
        ...     async def get_create_kwargs(cls, session, **kwargs):
        ...         # Add computed field
        ...         kwargs['full_name'] = f"{kwargs['first']} {kwargs['last']}"
        ...         return kwargs
        """
        return kwargs
        
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
        _session: async_scoped_session,
        _row: T,
    ) -> None:
        """Hook called during delete_row, BEFORE deletion

        Subclasses can override this to perform cleanup operations,
        cascade deletes, or other pre-deletion tasks. This hook receives
        the full row object, so it can access any field data needed.

        This is called within the transaction but before the delete is
        executed, so any errors raised here will prevent the deletion.

        Parameters
        ----------
        _session
            DB session manager

        _row
            The row object that will be deleted (with all fields accessible)
        """
        return

    @classmethod
    async def after_delete_hook(
        cls: type[T],
        _session: async_scoped_session,
        _row_id: int,
        _deleted_row_data: dict[str, Any] | None = None,
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
        _session
            DB session manager

        _row_id
            The ID of the row that was deleted

        _deleted_row_data
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
        - This is called within the same transaction as the delete operation
        """
        return

    

def ensure_base_inheritance(cls: type) -> None:
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
        raise TypeError(
            f"Class {cls.__name__} must inherit from rail_svc.db.base.Base"
        )
