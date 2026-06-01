from __future__ import annotations

import types
import warnings
from typing import Any, TypeVar

from pydantic import BaseModel

from ..client.base import RemoteAPI, RemoteTableOperations
from ..models import Filter, OrderBy

# Type variables
ResponseT = TypeVar("ResponseT", bound=BaseModel)
CreateT = TypeVar("CreateT", bound=BaseModel)


class AsyncRemoteOperations[ResponseT: BaseModel, CreateT: BaseModel]:
    """Async wrapper for remote table operations with connection management.

    Provides async methods for CRUD operations on remote database tables via HTTP API.
    Can be used as an async context manager for efficient connection reuse across
    multiple operations, or as a regular class for single operations.

    Type Parameters
    ---------------
    ResponseT : BaseModel
        Pydantic model type for API responses
    CreateT : BaseModel
        Pydantic model type for create/input operations

    Examples
    --------
    Using as async context manager (recommended for multiple operations):

    >>> async with AsyncRemoteOperations(
    ...     base_url="http://api.example.com",
    ...     table_name="users",
    ...     response_model=UserResponse,
    ...     create_model=UserCreate,
    ... ) as ops:
    ...     user = await ops.create_row(name="Alice", email="alice@example.com")
    ...     users = await ops.get_rows(limit=10)
    ...     await ops.delete_row(user.id)

    Using for single operations:

    >>> ops = AsyncRemoteOperations(...)
    >>> user = await ops.get_row(123)
    """

    def __init__(
        self,
        base_url: str,
        table_name: str,
        response_model: type[ResponseT],
        create_model: type[CreateT],
        api_prefix: str = "/api/v1",
        timeout: float = 30.0,
        auth_token: str | None = None,
    ) -> None:
        """Initialize the async remote table operations.

        Parameters
        ----------
        base_url : str
            The base URL of the remote API server
        table_name : str
            The name of the table to operate on
        response_model : type[ResponseT]
            Pydantic model for response data
        create_model : type[CreateT]
            Pydantic model for create/input data
        api_prefix : str, optional
            API version prefix, by default "/api/v1"
        timeout : float, optional
            Request timeout in seconds, by default 30.0
        auth_token : str | None, optional
            Authentication token for API requests, by default None
        """
        self.base_url = base_url
        self.table_name = table_name
        self.response_model = response_model
        self.create_model = create_model
        self.api_prefix = api_prefix
        self.timeout = timeout
        self.auth_token = auth_token
        self._api: RemoteAPI | None = None
        self._client: RemoteTableOperations[ResponseT, CreateT] | None = None
        self._owns_api = False
        self._has_warned = False

    async def __aenter__(self) -> AsyncRemoteOperations:
        """Enter async context manager.

        Creates and initializes the RemoteAPI client for reuse across operations.

        Returns
        -------
        AsyncRemoteTableOperations
            Self for context manager use
        """
        self._api = RemoteAPI(
            base_url=self.base_url,
            api_prefix=self.api_prefix,
            timeout=self.timeout,
            auth_token=self.auth_token,
        )
        await self._api.__aenter__()
        self._client = self._api.table(
            self.table_name,
            self.response_model,
            self.create_model,
        )
        self._owns_api = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """
        Properly closes the RemoteAPI client connection.
        """
        if self._api and self._owns_api:
            await self._api.__aexit__(exc_type, exc_val, exc_tb)
            self._api = None
            self._client = None
            self._owns_api = False

    async def _get_client(self) -> RemoteTableOperations[ResponseT, CreateT]:
        """Get or create the table operations client.

        If used within a context manager, returns the existing client.
        Otherwise, creates a temporary client for single-use operations.

        Returns
        -------
        RemoteTableOperations
            The table operations client

        Warnings
        --------
        If creating a temporary client (not using context manager), a warning
        is issued the first time. This is inefficient for multiple operations.

        Notes
        -----
        For best performance with multiple operations, use this class as an
        async context manager to reuse connections:

        >>> async with AsyncRemoteTableOperations(...) as ops:
        ...     await ops.create_row(...)
        ...     await ops.get_rows(...)
        """
        if self._client is not None:
            return self._client

        # Warn on first temporary client creation
        if not self._has_warned:
            warnings.warn(
                f"Creating temporary client for {self.__class__.__name__}. "
                "For better performance with multiple operations, use as async context manager: "
                "'async with AsyncRemoteTableOperations(...) as ops:'",
                ResourceWarning,
                stacklevel=3,
            )
            self._has_warned = True

        # Create temporary API and client for single operation
        api = RemoteAPI(
            base_url=self.base_url,
            api_prefix=self.api_prefix,
            timeout=self.timeout,
            auth_token=self.auth_token,
        )

        return api.table(
            self.table_name,
            self.response_model,
            self.create_model,
        )

    # CREATE operations

    async def create_row(
        self,
        *,
        validate: bool = True,
        **kwargs: Any,
    ) -> ResponseT:
        client = await self._get_client()
        return await client.create_row(validate=validate, **kwargs)

    async def create_rows(
        self,
        rows_data: list[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> list[ResponseT]:
        client = await self._get_client()
        return await client.create_rows(rows_data, validate=validate)

    async def create_rows_batched(
        self,
        rows_data: list[dict[str, Any]],
        *,
        validate: bool = True,
        batch_size: int = 1000,
    ) -> list[ResponseT]:
        client = await self._get_client()
        return await client.create_rows_batched(rows_data, validate=validate, batch_size=batch_size)

    async def bulk_insert_rows(
        self,
        rows_data: list[dict[str, Any]],
        *,
        validate: bool = True,
    ) -> int:
        client = await self._get_client()
        return await client.bulk_insert_rows(rows_data, validate=validate)

    # READ operations

    async def get_row(
        self,
        row_id: int,
    ) -> ResponseT:
        client = await self._get_client()
        return await client.get_row(row_id)

    async def get_row_or_none(
        self,
        row_id: int,
    ) -> ResponseT | None:
        client = await self._get_client()
        return await client.get_row_or_none(row_id)

    async def get_row_by_name(
        self,
        name: str,
    ) -> ResponseT:
        client = await self._get_client()
        return await client.get_row_by_name(name)

    async def get_rows(
        self,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        client = await self._get_client()
        return await client.get_rows(skip, limit)

    async def count_rows(self) -> int:
        client = await self._get_client()
        return await client.count_rows()

    async def lookup_by_id_or_name(
        self,
        row_id: int | None = None,
        name: str | None = None,
    ) -> tuple[int, ResponseT]:
        client = await self._get_client()
        return await client.lookup_by_id_or_name(row_id, name)

    # UPDATE operations

    async def update_row(
        self,
        row_id: int,
        **kwargs: Any,
    ) -> ResponseT:
        client = await self._get_client()
        return await client.update_row(row_id, **kwargs)

    async def update_rows(
        self,
        updates: list[dict[str, Any]],
    ) -> list[ResponseT]:
        client = await self._get_client()
        return await client.update_rows(updates)

    # DELETE operations

    async def delete_row(
        self,
        row_id: int,
        *,
        capture_data: bool = True,
    ) -> ResponseT | None:
        client = await self._get_client()
        return await client.delete_row(row_id, capture_data=capture_data)

    async def delete_rows(
        self,
        row_ids: list[int],
        *,
        capture_data: bool = False,
    ) -> list[ResponseT] | int:
        client = await self._get_client()
        return await client.delete_rows(row_ids, capture_data=capture_data)

    async def bulk_delete_rows(
        self,
        row_ids: list[int],
    ) -> int:
        client = await self._get_client()
        return await client.bulk_delete_rows(row_ids)

    # FILTER/QUERY operations

    async def filter_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
    ) -> list[ResponseT]:
        client = await self._get_client()
        return await client.filter_rows(filters, logical_op, order_by, skip, limit)

    async def count_filtered_rows(
        self,
        filters: list[Filter] | None = None,
        logical_op: str = "and",
    ) -> int:
        client = await self._get_client()
        return await client.count_filtered_rows(filters, logical_op)

    async def filter_one(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT:
        client = await self._get_client()
        return await client.filter_one(filters, logical_op)

    async def filter_one_or_none(
        self,
        filters: list[Filter],
        logical_op: str = "and",
    ) -> ResponseT | None:
        client = await self._get_client()
        return await client.filter_one_or_none(filters, logical_op)

    async def find_by(
        self,
        order_by: OrderBy | list[OrderBy] | None = None,
        skip: int = 0,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[ResponseT]:
        client = await self._get_client()
        return await client.find_by(order_by, skip, limit, **kwargs)

    async def find_one_by(
        self,
        **kwargs: Any,
    ) -> ResponseT:
        client = await self._get_client()
        return await client.find_one_by(**kwargs)
