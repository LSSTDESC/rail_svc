"""
Database operations with multi-layer abstraction.

This module provides CRUD operations that work across four layers:
1. Local CLI - Direct database access
2. API Server - FastAPI REST endpoints
3. HTTP Client - Client library methods
4. Remote CLI - CLI that calls the API

Usage Examples
--------------
Basic setup for a User model::

    from your_app.models import User

    # Create operation context
    user_ctx = OperationContext.from_db_class("users", User)

    # Create and register operations
    get_users = GetRowsOperation(user_ctx)
    get_users.create_local_command(local_cli_group)
    get_users.create_router_endpoint(api_router)
    get_users.create_remote_command(remote_cli_group)

Available Operations
-------------------
- GetRowsOperation: Paginated list retrieval
- GetRowByIdOperation: Single row by ID
- GetRowByNameOperation: Single row by name
- GetRowOrNoneOperation: Single row by ID (returns None if not found)
- CountRowsOperation: Count total rows
- StreamRowsOperation: Streaming for large datasets

Configuration
-------------
Set these in your config module::

    DEFAULT_PAGE_SIZE = 100
    MAX_PAGE_SIZE = 1000
    DEFAULT_BATCH_SIZE = 1000
    DEFAULT_TIMEOUT = 30.0
    STREAM_TIMEOUT = 60.0
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable, Sequence
from typing import TYPE_CHECKING
from urllib.parse import quote

import click
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from fastapi.responses import StreamingResponse
from httpx import HTTPError, TimeoutException
from pydantic import BaseModel, TypeAdapter
from pydantic_core import ValidationError as CoreValidationError
from safir.dependencies.db_session import db_session_dependency
from sqlalchemy.ext.asyncio import AsyncEngine, async_scoped_session
from structlog import get_logger
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from .. import db_funcs
from ..cli import common_options
from ..cli.utils import handle_cli_error
from ..config import config as global_config
from .base import (BaseOperation, OperationContext, build_url,
                   output_pydantic_list, output_pydantic_single)

if TYPE_CHECKING:
    from .client import ClientBase

logger = get_logger(__name__)


# Configuration defaults
DEFAULT_PAGE_SIZE = global_config.web_interface.default_page_size
MAX_PAGE_SIZE = global_config.web_interface.max_page_size
DEFAULT_BATCH_SIZE = global_config.web_interface.default_batch_size
MAX_BATCH_SIZE = global_config.web_interface.max_batch_size
DEFAULT_TIMEOUT = global_config.web_interface.default_timeout
STREAM_TIMEOUT = global_config.web_interface.stream_timeout


class GetRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Paginated row retrieval operation.

    Supports:
    - Configurable page sizes
    - Skip/limit pagination
    - Automatic multi-page fetching in client
    - Input validation

    Examples
    --------
    >>> from your_app.models import User
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = GetRowsOperation(ctx)
    >>> op.create_local_command(cli_group)
    >>> op.create_router_endpoint(router)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(name=ctx.name, help=f"List {ctx.router_string} rows")
        @common_options.db_engine()
        @common_options.output()
        @common_options.skip()
        @common_options.limit()
        @common_options.page_size()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            skip: int,
            limit: int | None,
            page_size: int,
        ) -> None:
            """List rows from the table with pagination."""
            # Validate parameters
            try:
                params = common_options.PaginationParams(skip=skip, limit=limit, page_size=page_size)
                params.validate()
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

            try:
                # Fetch data
                async with db_engine().begin() as session:
                    result = await db_funcs.read.get_rows(
                        ctx.db_class,
                        session,
                        skip=skip,
                        limit=limit or page_size,
                    )

                # Output results
                output_pydantic_list(result, output, ctx.col_names_optional)

            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Invalid parameters for get_rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                )
                click.echo(f"Error: Invalid parameters: {exc}", err=True)
                raise click.Abort()
            except Exception as exc:
                logger.error(
                    "Database error in get_rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                click.echo(f"Error: Database error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx
        ResponseModel = ctx.response_class

        @router.get(
            f"/{ctx.name}",
            response_model=list[ResponseModel],
            summary=f"List all {ctx.router_string}",
            description=f"Retrieve {ctx.router_string} rows with pagination support.",
        )
        async def endpoint(
            skip: int = Query(0, ge=0, description="Number of records to skip"),
            limit: int = Query(
                DEFAULT_PAGE_SIZE,
                ge=1,
                le=MAX_PAGE_SIZE,
                description="Maximum records to return"
            ),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> Sequence[BaseModel]:
            """Route handler to get rows from database with pagination."""
            try:
                async with session.begin():
                    return await db_funcs.read.get_rows(
                        ctx.db_class,
                        session,
                        skip=skip,
                        limit=limit,
                    )
            except (ValueError, KeyError) as exc:
                logger.warning(
                    "Invalid parameters for get_rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                )
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                logger.error(
                    "Database error in get_rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(list[ctx.response_class])

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            skip: int = 0,
            page_size: int = DEFAULT_PAGE_SIZE,
            max_results: int | None = None,
            timeout: float = DEFAULT_TIMEOUT,
        ) -> list[BaseModel]:
            """
            Fetch all rows with automatic pagination.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            skip
                Number of records to skip before starting
            page_size
                Number of records to fetch per request
            max_results
                Maximum total results to return (None for all)
            timeout
                Request timeout in seconds

            Returns
            -------
            list[BaseModel]
                All rows from the endpoint
            """
            results: list[BaseModel] = []
            query_url = build_url(ctx.router_string, ctx.name)
            current_skip = skip

            try:
                while True:
                    # Calculate effective limit
                    if max_results is not None:
                        remaining = max_results - len(results)
                        if remaining <= 0:
                            break
                        current_limit = min(page_size, remaining)
                    else:
                        current_limit = page_size

                    # Make request
                    logger.debug(
                        "Fetching page",
                        url=query_url,
                        skip=current_skip,
                        limit=current_limit,
                    )

                    response = client_object.client.get(
                        query_url,
                        params={"skip": current_skip, "limit": current_limit},
                        timeout=timeout,
                    )
                    response.raise_for_status()

                    # Parse and validate response
                    paged_results = response_adapter.validate_python(response.json())

                    # Break if no more results
                    if not paged_results:
                        logger.debug("No more results, pagination complete")
                        break

                    results.extend(paged_results)

                    # Last page detection
                    if len(paged_results) < current_limit:
                        logger.debug(
                            "Received partial page, pagination complete",
                            expected=current_limit,
                            received=len(paged_results),
                        )
                        break

                    current_skip += len(paged_results)

            except HTTPError as exc:
                logger.error(
                    "HTTP error fetching rows",
                    url=query_url,
                    skip=current_skip,
                    error=str(exc),
                )
                raise
            except CoreValidationError as exc:
                logger.error(
                    "Validation error parsing response",
                    url=query_url,
                    skip=current_skip,
                    error=str(exc),
                )
                raise

            logger.info(
                "Pagination complete",
                total_results=len(results),
                url=query_url,
            )
            return results

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(name=ctx.name, help=f"List {ctx.router_string} rows")
        @common_options.pz_client()
        @common_options.output()
        @common_options.output()
        @common_options.skip()
        @common_options.limit()
        @common_options.page_size()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            skip: int,
            limit: int | None,
            page_size: int,
            timeout: float,
        ) -> None:
            """List rows from the remote API with pagination."""
            # Validate parameters
            try:
                params = common_options.PaginationParams(skip=skip, limit=limit, page_size=page_size)
                params.validate()
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

            try:
                result = client_method(
                    client_object,
                    skip=skip,
                    page_size=page_size,
                    max_results=limit,
                    timeout=timeout,
                )

                output_pydantic_list(result, output, ctx.col_names_optional)

            except Exception as exc:
                handle_cli_error(exc, "fetch", ctx.router_string)

        return command


class GetRowByIdOperation[T: BaseModel](BaseOperation[T]):
    """
    Single row retrieval by ID.

    Features:
    - Returns 404 if not found
    - Validates ID > 0
    - Configurable ID field name

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = GetRowByIdOperation(ctx, id_field="user_id")
    >>> op.create_router_endpoint(router)
    """

    def __init__(self, context: OperationContext[T], id_field: str = "id_") -> None:
        """
        Initialize operation with ID field specification.

        Args:
            context: Operation context
            id_field: Name of the ID field in the database (default: "id_")
                     Note: Currently not used by db_funcs but reserved for future use
        """
        super().__init__(context)
        self.id_field = id_field

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(name=f"get-{ctx.name}", help=f"Get single {ctx.router_string} by ID")
        @common_options.db_engine()
        @common_options.output()
        @common_options.id_arg()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            id_: int,
        ) -> None:
            """Get a single row by ID."""
            if id_ <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.read.get_row(
                        ctx.db_class,
                        session,
                        id_,
                    )

                    if result is None:
                        click.echo(
                            f"Error: {ctx.router_string} with ID {id_} not found",
                            err=True
                        )
                        raise click.Abort()

                    output_pydantic_single(result, output, ctx.col_names_optional)

            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error fetching row",
                    db_class=ctx.db_class.__name__,
                    id_=id_,
                    error=str(exc),
                    exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx
        ResponseModel = ctx.response_class

        @router.get(
            f"/{ctx.name}/{{id_}}",
            response_model=ResponseModel,
            summary=f"Get {ctx.router_string} by ID",
            description=f"Retrieve a single {ctx.router_string} record by its ID.",
        )
        async def endpoint(
            id_: int = Path(..., description="Record ID", gt=0),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> BaseModel:
            """Get a single row by ID."""
            try:
                async with session.begin():
                    result = await db_funcs.read.get_row(
                        ctx.db_class,
                        session,
                        id_,
                    )

                    if result is None:
                        raise HTTPException(
                            status_code=404,
                            detail=f"{ctx.router_string} with ID {id_} not found"
                        )

                    return result

            except HTTPException:
                raise
            except Exception as exc:
                logger.error(
                    "Database error",
                    db_class=ctx.db_class.__name__,
                    id_=id_,
                    error=str(exc),
                    exc_info=True
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(ctx.response_class)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            id_: int,
            timeout: float = DEFAULT_TIMEOUT,
        ) -> BaseModel:
            """
            Get a single row by ID.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            id_
                Record ID to fetch
            timeout
                Request timeout in seconds

            Returns
            -------
            BaseModel
                The requested record

            Raises
            ------
            ValueError
                If the record is not found (404)
            HTTPError
                For other HTTP errors
            ValidationError
                If response validation fails
            """
            query_url = build_url(ctx.router_string, ctx.name, str(id_))

            try:
                logger.debug("Fetching row by ID", url=query_url, id_=id_)

                response = client_object.client.get(query_url, timeout=timeout)
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully fetched row", id_=id_)

                return result

            except HTTPError as exc:
                if hasattr(exc, 'response') and exc.response.status_code == 404:
                    error_msg = f"{ctx.router_string} with ID {id_} not found"
                    logger.warning("Record not found", id_=id_, url=query_url)
                    raise ValueError(error_msg) from exc

                logger.error(
                    "HTTP error fetching row",
                    url=query_url,
                    id_=id_,
                    error=str(exc)
                )
                raise
            except CoreValidationError as exc:
                logger.error(
                    "Validation error parsing response",
                    url=query_url,
                    id_=id_,
                    error=str(exc)
                )
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(name=f"get-{ctx.name}", help=f"Get {ctx.router_string} by ID")
        @common_options.pz_client()
        @common_options.output()
        @common_options.id_arg()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            id_: int,
            timeout: float,
        ) -> None:
            """Get a single row by ID from remote API."""
            if id_ <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            try:
                result = client_method(client_object, id_, timeout=timeout)
                output_pydantic_single(result, output, ctx.col_names_optional)

            except Exception as exc:
                handle_cli_error(exc, "fetch", ctx.router_string)

        return command


class GetRowByNameOperation[T: BaseModel](BaseOperation[T]):
    """
    Single row retrieval by name field.

    Features:
    - URL-safe name encoding
    - Returns 404 if not found
    - Validates model has name attribute
    - Empty name validation

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = GetRowByNameOperation(ctx)
    >>> op.create_router_endpoint(router)
    """

    def __init__(self, context: OperationContext[T], name_field: str = "name") -> None:
        """
        Initialize operation with name field specification.

        Args:
            context: Operation context
            name_field: Name of the name field in the database (default: "name")
                       Currently stored for future use
        """
        super().__init__(context)
        self.name_field = name_field

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"get-{ctx.name}-by-name",
            help=f"Get single {ctx.router_string} by name"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.name_arg()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            name: str,
        ) -> None:
            """Get a single row by name."""
            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.read.get_row_by_name(
                        ctx.db_class,
                        session,
                        name,
                    )

                    output_pydantic_single(result, output, ctx.col_names_optional)

            except KeyError as exc:
                # Row not found
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except AttributeError as exc:
                # Model doesn't have name field
                logger.error(
                    "Model configuration error",
                    db_class=ctx.db_class.__name__,
                    error=str(exc)
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error fetching row by name",
                    db_class=ctx.db_class.__name__,
                    name=name,
                    error=str(exc),
                    exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx
        ResponseModel = ctx.response_class

        @router.get(
            f"/{ctx.name}/by-name/{{name}}",
            response_model=ResponseModel,
            summary=f"Get {ctx.router_string} by name",
            description=f"Retrieve a single {ctx.router_string} record by its name.",
        )
        async def endpoint(
            name: str = Path(..., description="Record name", min_length=1),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> BaseModel:
            """Get a single row by name."""
            try:
                async with session.begin():
                    result = await db_funcs.read.get_row_by_name(
                        ctx.db_class,
                        session,
                        name,
                    )

                    return result

            except KeyError as exc:
                raise HTTPException(
                    status_code=404,
                    detail=str(exc)
                ) from exc
            except AttributeError as exc:
                # Model doesn't have name field - this is a configuration error
                logger.error(
                    "Model configuration error",
                    db_class=ctx.db_class.__name__,
                    error=str(exc)
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Server configuration error: {str(exc)}"
                ) from exc
            except Exception as exc:
                logger.error(
                    "Database error",
                    db_class=ctx.db_class.__name__,
                    name=name,
                    error=str(exc),
                    exc_info=True
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(ctx.response_class)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            name: str,
            timeout: float = DEFAULT_TIMEOUT,
        ) -> BaseModel:
            """
            Get a single row by name.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            name
                Record name to fetch
            timeout
                Request timeout in seconds

            Returns
            -------
            BaseModel
                The requested record

            Raises
            ------
            ValueError
                If the record is not found (404) or name is empty
            HTTPError
                For other HTTP errors
            ValidationError
                If response validation fails
            """
            if not name or not name.strip():
                raise ValueError("name cannot be empty")

            # URL encode the name parameter
            encoded_name = quote(name, safe='')
            query_url = build_url(ctx.router_string, ctx.name, "by-name", encoded_name)

            try:
                logger.debug("Fetching row by name", url=query_url, name=name)

                response = client_object.client.get(query_url, timeout=timeout)
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully fetched row", name=name)

                return result

            except HTTPError as exc:
                if hasattr(exc, 'response') and exc.response.status_code == 404:
                    error_msg = f"{ctx.router_string} with name '{name}' not found"
                    logger.warning("Record not found", name=name, url=query_url)
                    raise ValueError(error_msg) from exc
                logger.error(
                    "HTTP error fetching row",
                    url=query_url,
                    name=name,
                    error=str(exc)
                )
                raise
            except CoreValidationError as exc:
                logger.error(
                    "Validation error parsing response",
                    url=query_url,
                    name=name,
                    error=str(exc)
                )
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"get-{ctx.name}-by-name",
            help=f"Get {ctx.router_string} by name"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.name_arg()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            name: str,
            timeout: float,
        ) -> None:
            """Get a single row by name from remote API."""
            try:
                result = client_method(client_object, name, timeout=timeout)
                output_pydantic_single(result, output, ctx.col_names_optional)

            except Exception as exc:
                handle_cli_error(exc, "fetch", ctx.router_string)

        return command


class GetRowOrNoneOperation[T: BaseModel](BaseOperation[T]):
    """
    Single row retrieval by ID that returns None if not found.

    Unlike GetRowByIdOperation, this doesn't raise errors for missing records.
    Useful for optional lookups or conditional logic.

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = GetRowOrNoneOperation(ctx)
    >>> result = await op.get_row(session, 123)  # Returns None if not found
    """

    def __init__(self, context: OperationContext[T], id_field: str = "id_") -> None:
        """
        Initialize operation with ID field specification.

        Args:
            context: Operation context
            id_field: Name of the ID field in the database (default: "id_")
                     Reserved for future use when db_funcs supports it
        """
        super().__init__(context)
        self.id_field = id_field

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"get-{ctx.name}-or-none",
            help=f"Get single {ctx.router_string} by ID (returns nothing if not found)"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.id_arg()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            id_: int,
        ) -> None:
            """Get a single row by ID, or nothing if not found."""
            if id_ <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.read.get_row_or_none(
                        ctx.db_class,
                        session,
                        id_,
                    )

                    if result is None:
                        click.echo(
                            f"{ctx.router_string} with ID {id_} not found",
                            err=True
                        )
                        # Don't abort - this is expected behavior
                        return

                    output_pydantic_single(result, output, ctx.col_names_optional)

            except Exception as exc:
                logger.error(
                    "Error fetching row",
                    db_class=ctx.db_class.__name__,
                    id_=id_,
                    error=str(exc),
                    exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx
        ResponseModel = ctx.response_class

        @router.get(
            f"/{ctx.name}/or-none/{{id_}}",
            response_model=ResponseModel | None,
            summary=f"Get {ctx.router_string} by ID or null",
            description=f"Retrieve a single {ctx.router_string} record by its ID "
                        "returning null if not found.",
        )
        async def endpoint(
            id_: int = Path(..., description="Record ID", gt=0),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> BaseModel | None:
            """Get a single row by ID, or None if not found."""
            try:
                async with session.begin():
                    result = await db_funcs.read.get_row_or_none(
                        ctx.db_class,
                        session,
                        id_,
                    )

                    return result

            except Exception as exc:
                logger.error(
                    "Database error",
                    db_class=ctx.db_class.__name__,
                    id_=id_,
                    error=str(exc),
                    exc_info=True
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(ctx.response_class)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            id_: int,
            timeout: float = DEFAULT_TIMEOUT,
        ) -> BaseModel | None:
            """
            Get a single row by ID, or None if not found.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            id_
                Record ID to fetch
            timeout
                Request timeout in seconds

            Returns
            -------
            BaseModel | None
                The requested record or None if not found

            Raises
            ------
            HTTPError
                For HTTP errors other than 404
            ValidationError
                If response validation fails
            """
            query_url = build_url(ctx.router_string, ctx.name, "or-none", str(id_))

            try:
                logger.debug("Fetching row by ID (or None)", url=query_url, id_=id_)

                response = client_object.client.get(query_url, timeout=timeout)
                response.raise_for_status()

                # Handle null response
                json_data = response.json()
                if json_data is None:
                    logger.debug("Record not found", id_=id_)
                    return None

                result = response_adapter.validate_python(json_data)
                logger.debug("Successfully fetched row", id_=id_)

                return result

            except HTTPError as exc:
                logger.error(
                    "HTTP error fetching row",
                    url=query_url,
                    id_=id_,
                    error=str(exc)
                )
                raise
            except CoreValidationError as exc:
                logger.error(
                    "Validation error parsing response",
                    url=query_url,
                    id_=id_,
                    error=str(exc)
                )
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"get-{ctx.name}-or-none",
            help=f"Get {ctx.router_string} by ID (returns nothing if not found)"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.id_arg()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            id_: int,
            timeout: float,
        ) -> None:
            """Get a single row by ID from remote API, or nothing if not found."""
            if id_ <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            try:
                result = client_method(client_object, id_, timeout=timeout)

                if result is None:
                    click.echo(
                        f"{ctx.router_string} with ID {id_} not found",
                        err=True
                    )
                    # Don't abort - this is expected behavior
                    return

                output_pydantic_single(result, output, ctx.col_names_optional)

            except Exception as exc:
                handle_cli_error(exc, "fetch", ctx.router_string)

        return command


class CountRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Count total rows in a table.

    Useful for:
    - Pagination metadata
    - Dashboard statistics
    - Capacity planning

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = CountRowsOperation(ctx)
    >>> total = await op.count(session)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"count-{ctx.name}",
            help=f"Count total {ctx.router_string} rows"
        )
        @common_options.db_engine()
        async def command(
            db_engine: Callable[[], AsyncEngine],
        ) -> None:
            """Count total rows in the table."""
            try:
                async with db_engine().begin() as session:
                    count = await db_funcs.read.count_rows(
                        ctx.db_class,
                        session,
                    )

                    click.echo(f"Total {ctx.router_string}: {count}")

            except Exception as exc:
                logger.error(
                    "Error counting rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        # Define response model for count
        class CountResponse(BaseModel):
            """Response model for count endpoint."""
            count: int
            resource: str

            model_config = {"json_schema_extra": {
                "example": {
                    "count": 42,
                    "resource": "users"
                }
            }}

        @router.get(
            f"/{ctx.name}/count",
            response_model=CountResponse,
            summary=f"Count {ctx.router_string}",
            description=f"Get the total number of {ctx.router_string} records.",
        )
        async def endpoint(
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> CountResponse:
            """Count total rows in the table."""
            try:
                async with session.begin():
                    count = await db_funcs.read.count_rows(
                        ctx.db_class,
                        session,
                    )

                    return CountResponse(
                        count=count,
                        resource=ctx.router_string
                    )

            except Exception as exc:
                logger.error(
                    "Database error counting rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            timeout: float = DEFAULT_TIMEOUT,
        ) -> int:
            """
            Get the total count of rows.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            timeout
                Request timeout in seconds

            Returns
            -------
            int
                Total number of rows

            Raises
            ------
            HTTPError
                For HTTP errors
            TypeError
                If response count is not an integer
            KeyError
                If response doesn't contain 'count' field
            """
            query_url = build_url(ctx.router_string, ctx.name, "count")

            try:
                logger.debug("Fetching row count", url=query_url)

                response = client_object.client.get(query_url, timeout=timeout)
                response.raise_for_status()

                data = response.json()

                if "count" not in data:
                    raise KeyError("Response missing 'count' field")

                count = data["count"]

                if not isinstance(count, int):
                    raise TypeError(
                        f"Expected count to be int, got {type(count).__name__}"
                    )

                logger.debug("Successfully fetched row count", count=count)

                return count

            except HTTPError as exc:
                logger.error(
                    "HTTP error fetching count",
                    url=query_url,
                    error=str(exc)
                )
                raise
            except (KeyError, TypeError) as exc:
                logger.error(
                    "Validation error parsing count response",
                    url=query_url,
                    error=str(exc)
                )
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"count-{ctx.name}",
            help=f"Count total {ctx.router_string} rows"
        )
        @common_options.pz_client()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            timeout: float,
        ) -> None:
            """Count total rows from remote API."""
            try:
                count = client_method(client_object, timeout=timeout)
                click.echo(f"Total {ctx.router_string}: {count}")

            except Exception as exc:
                handle_cli_error(exc, "count", ctx.router_string)

        return command


class StreamRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Streaming row retrieval operation for large datasets.

    Uses NDJSON (newline-delimited JSON) format for efficient streaming.
    Batches database queries to avoid loading entire dataset into memory.

    Features:
    - Memory-efficient for large datasets
    - Configurable batch sizes
    - Progress indication in CLI
    - Proper error handling mid-stream

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("logs", Log)
    >>> op = StreamRowsOperation(ctx)
    >>> # Stream millions of records efficiently
    >>> for record in stream_records(session, batch_size=5000):
    ...     process(record)
    """
    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"stream-{ctx.name}",
            help=f"Stream {ctx.router_string} rows (for large datasets)"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.batch_size()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            batch_size: int,
        ) -> None:
            """Stream rows from the table in batches."""
            if batch_size <= 0:
                click.echo("Error: batch-size must be positive", err=True)
                raise click.Abort()
            if batch_size > MAX_BATCH_SIZE:
                click.echo(f"Error: batch-size cannot exceed {MAX_BATCH_SIZE}", err=True)
                raise click.Abort()

            try:
                total_count = 0
                async with db_engine().begin() as session:
                    offset = 0

                    while True:
                        batch = await db_funcs.read.get_rows(
                            ctx.db_class,
                            session,
                            skip=offset,
                            limit=batch_size,
                        )

                        if not batch:
                            break

                        # Output each batch
                        output_pydantic_list(batch, output, ctx.col_names_optional)

                        total_count += len(batch)
                        offset += batch_size

                        # Break if we got fewer results than requested (last batch)
                        if len(batch) < batch_size:
                            break

                logger.info("Streaming complete", total_records=total_count)
                click.echo(f"\nTotal records streamed: {total_count}", err=True)

            except Exception as exc:
                logger.error(
                    "Database error in stream_rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                click.echo(f"Error: Database error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        @router.get(
            f"/{ctx.name}/stream",
            response_class=StreamingResponse,
            summary=f"Stream {ctx.router_string}",
            description=f"Stream {ctx.router_string} rows as NDJSON for large datasets.",
        )
        async def endpoint(
            batch_size: int = Query(
                DEFAULT_BATCH_SIZE,
                ge=1,
                le=MAX_BATCH_SIZE,
                description="Records per batch"
            ),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> StreamingResponse:
            """Stream rows as newline-delimited JSON."""

            async def generate() -> AsyncGenerator[str]:
                """Generate NDJSON stream."""
                try:
                    async with session.begin():
                        offset = 0

                        while True:
                            batch = await db_funcs.read.get_rows(
                                ctx.db_class,
                                session,
                                skip=offset,
                                limit=batch_size,
                            )

                            if not batch:
                                break

                            # Yield each record as a JSON line
                            for record in batch:
                                # Ensure record is properly serialized
                                if isinstance(record, BaseModel):
                                    yield record.model_dump_json() + "\n"
                                else:
                                    yield json.dumps(record) + "\n"

                            offset += batch_size

                            if len(batch) < batch_size:
                                break

                except Exception as exc:
                    logger.error(
                        "Error during streaming",
                        db_class=ctx.db_class.__name__,
                        error=str(exc),
                        exc_info=True,
                    )
                    # In streaming, we can't raise HTTPException after starting
                    # So we yield an error line with a marker
                    error_obj = {
                        "__error__": True,
                        "error": str(exc),
                        "type": type(exc).__name__
                    }
                    yield json.dumps(error_obj) + "\n"

            return StreamingResponse(
                generate(),
                media_type="application/x-ndjson",
                headers={
                    "X-Content-Type-Options": "nosniff",
                    "Cache-Control": "no-cache",
                }
            )

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(ctx.response_class)

        def client_method(
            client_object: ClientBase,
            batch_size: int = DEFAULT_BATCH_SIZE,
            timeout: float = STREAM_TIMEOUT,
            *,
            yield_records: bool = False,
        ) -> list[BaseModel] | AsyncGenerator[BaseModel]:
            """
            Stream rows from the API.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            batch_size
                Records per batch
            timeout
                Request timeout in seconds
            yield_records
                If True, returns a generator that yields records as they arrive.
                If False, loads all records into memory and returns a list.

            Returns
            -------
            list[BaseModel] | Generator[BaseModel, None]
                All streamed rows (list) or generator of rows

            Notes
            -----
            When yield_records=True, you can process records incrementally:

                for record in client_method(client, yield_records=True):
                    process(record)  # Process without loading all into memory
            """
            query_url = build_url(ctx.router_string, ctx.name, "stream")

            def _stream_generator():
                """Internal generator for streaming records."""
                try:
                    logger.debug("Starting stream", url=query_url, batch_size=batch_size)

                    with client_object.client.stream(
                        "GET",
                        query_url,
                        params={"batch_size": batch_size},
                        timeout=timeout,
                    ) as response:
                        response.raise_for_status()

                        # Read line by line (NDJSON)
                        for line in response.iter_lines():
                            if not line.strip():
                                continue

                            # Check for error marker
                            try:
                                raw_data = json.loads(line)
                                if isinstance(raw_data, dict) and raw_data.get("__error__"):
                                    error_msg = raw_data.get("error", "Unknown stream error")
                                    logger.error("Stream error from server", error=error_msg)
                                    raise HTTPError(f"Stream error: {error_msg}")
                            except json.JSONDecodeError:
                                pass  # Not JSON, try validation

                            # Parse and validate each line
                            record = response_adapter.validate_json(line)
                            yield record

                    logger.info("Streaming complete")

                except HTTPError as exc:
                    logger.error(
                        "HTTP error during streaming",
                        url=query_url,
                        error=str(exc)
                    )
                    raise
                except CoreValidationError as exc:
                    logger.error(
                        "Validation error during streaming",
                        url=query_url,
                        error=str(exc)
                    )
                    raise

            if yield_records:
                return _stream_generator()
            # Load all into memory
            results = list(_stream_generator())
            logger.info("Stream loaded into memory", total_records=len(results))
            return results

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"stream-{ctx.name}",
            help=f"Stream {ctx.router_string} rows from remote API"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.batch_size()
        @common_options.timeout()
        @common_options.show_progress()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            batch_size: int,
            timeout: float,
            *
            show_progress: bool,
        ) -> None:
            """Stream rows from the remote API."""
            if batch_size <= 0:
                click.echo("Error: batch-size must be positive", err=True)
                raise click.Abort()
            if batch_size > MAX_BATCH_SIZE:
                click.echo(f"Error: batch-size cannot exceed {MAX_BATCH_SIZE}", err=True)
                raise click.Abort()

            try:
                if show_progress:
                    # Collect with progress bar
                    results = []
                    with click.progressbar(
                        length=None,
                        label=f'Streaming {ctx.router_string}',
                        show_eta=False,
                    ) as prog_bar:
                        for record in client_method(
                            client_object,
                            batch_size=batch_size,
                            timeout=timeout,
                            yield_records=True,
                        ):
                            results.append(record)
                            prog_bar.update(1)

                    # Output after streaming complete
                    output_pydantic_list(results, output, ctx.col_names_optional)
                    click.echo(f"\nTotal records: {len(results)}", err=True)
                else:
                    # Stream and output incrementally
                    total = 0
                    for record in client_method(
                        client_object,
                        batch_size=batch_size,
                        timeout=timeout,
                        yield_records=True,
                    ):
                        # Output each record as it arrives
                        output_pydantic_single(record, output, ctx.col_names_optional)
                        total += 1

                    click.echo(f"\nTotal records: {total}", err=True)

            except Exception as exc:
                handle_cli_error(exc, "stream", ctx.router_string)

        return command


# Export all operation classes
__all__ = [
    "GetRowsOperation",
    "GetRowByIdOperation",
    "GetRowByNameOperation",
    "GetRowOrNoneOperation",
    "CountRowsOperation",
    "StreamRowsOperation",
]
