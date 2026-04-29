"""
Create operations for database records.

Provides four creation strategies with different trade-offs:
- CreateRowOperation: Single row with full validation
- CreateRowsOperation: Atomic multi-row (all-or-nothing)
- CreateRowsBatchedOperation: Batched creation (partial success possible)
- BulkInsertRowsOperation: High-performance bulk insert

All operations support both local and remote execution.
"""

from __future__ import annotations

import json as json_lib
import click

from collections.abc import Callable
from typing import TYPE_CHECKING, Any
import aiofiles

from fastapi import APIRouter, Depends, HTTPException, status
from httpx import HTTPError, TimeoutException
from pydantic import BaseModel, Field, ValidationError, TypeAdapter
from pydantic_core import ValidationError as CoreValidationError
from safir.dependencies.db_session import db_session_dependency
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_scoped_session
from structlog import get_logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from .client import ClientBase

from ... import db_funcs
from ..cli import common_options
from ..cli.utils import handle_cli_error
from .base import BaseOperation, build_url, output_json

logger = get_logger(__name__)


# Response models
class CreateResponse(BaseModel):
    """Response model for single row creation."""

    success: bool = Field(..., description="Whether creation succeeded")
    resource: str = Field(..., description="Resource type")
    data: dict[str, Any] = Field(..., description="Created record data")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "resource": "users",
                "data": {"id": 123, "username": "alice", "email": "alice@example.com"},
            }
        }
    }


class CreateMultipleResponse(BaseModel):
    """Response model for multiple row creation."""

    success: bool = Field(..., description="Whether all creations succeeded")
    count: int = Field(..., description="Number of rows created")
    resource: str = Field(..., description="Resource type")
    data: list[dict[str, Any]] = Field(..., description="Created records data")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "count": 3,
                "resource": "users",
                "data": [
                    {"id": 1, "username": "alice"},
                    {"id": 2, "username": "bob"},
                    {"id": 3, "username": "charlie"},
                ],
            }
        }
    }


class BulkInsertResponse(BaseModel):
    """Response model for bulk insert."""

    success: bool = Field(..., description="Whether bulk insert succeeded")
    count: int = Field(..., description="Number of rows inserted")
    resource: str = Field(..., description="Resource type")

    model_config = {"json_schema_extra": {"example": {"success": True, "count": 10000, "resource": "users"}}}


class CreateRowOperation[T: BaseModel](BaseOperation[T]):
    """
    Single row creation operation with full validation.

    Features:
    - Pydantic validation before creation
    - get_create_kwargs preprocessing
    - Returns created object with DB-generated values
    - Full error handling

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = CreateRowOperation(ctx)
    >>> user = await op.create(session, username="alice", email="alice@example.com")
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(name=f"create-{ctx.name}", help=f"Create a single {ctx.router_string}")
        @common_options.db_engine()
        @common_options.output()
        @common_options.fields()
        @common_options.json_data()
        @common_options.no_validate()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            field_values: tuple[tuple[str, str], ...],
            json_data: str | None,
            *,
            no_validate: bool,
        ) -> None:
            """Create a single row."""
            # Parse data
            create_data: dict[str, Any] = {}

            if json_data:
                try:
                    create_data = json_lib.loads(json_data)
                except json_lib.JSONDecodeError as exc:
                    click.echo(f"Error: Invalid JSON: {exc}", err=True)
                    raise click.Abort()

            # Add --field options (override JSON)
            for field_name, field_value in field_values:
                try:
                    create_data[field_name] = json_lib.loads(field_value)
                except json_lib.JSONDecodeError:
                    create_data[field_name] = field_value

            if not create_data:
                click.echo("Error: No data provided. Use --field or --json-data", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.create.create_row(
                        ctx.db_class,
                        session,
                        validate=not no_validate,
                        **create_data,
                    )

                    # Convert to dict
                    result_dict = {
                        column.name: getattr(result, column.name) for column in ctx.db_class.__table__.columns
                    }

                    response = {
                        "success": True,
                        "resource": ctx.router_string,
                        "data": result_dict,
                    }

                    output_json(response, output)
                    click.echo(f"Successfully created {ctx.router_string}", err=True)

            except ValidationError as exc:
                click.echo(f"Error: Validation failed: {exc}", err=True)
                raise click.Abort()
            except IntegrityError as exc:
                logger.error("Integrity constraint violation", db_class=ctx.db_class.__name__, error=str(exc))
                click.echo(f"Error: Integrity constraint violation (duplicate key, etc.): {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error creating row", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class CreateRequest(BaseModel):
            """Request model for row creation."""

            data: dict[str, Any] = Field(..., description="Field values for new record", min_length=1)
            validate: bool = Field(default=True, description="Whether to validate with Pydantic")

        @router.post(
            f"/{ctx.name}",
            response_model=CreateResponse,
            status_code=status.HTTP_201_CREATED,
            summary=f"Create {ctx.router_string}",
            description=f"Create a single {ctx.router_string} record.",
        )
        async def endpoint(
            request: CreateRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> CreateResponse:
            """Create a single row."""
            try:
                async with session.begin():
                    result = await db_funcs.create.create_row(
                        ctx.db_class,
                        session,
                        validate=request.validate,
                        **request.data,
                    )

                    # Convert to dict
                    result_dict = {
                        column.name: getattr(result, column.name) for column in ctx.db_class.__table__.columns
                    }

                    return CreateResponse(
                        success=True,
                        resource=ctx.router_string,
                        data=result_dict,
                    )

            except ValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Validation failed: {str(exc)}"
                ) from exc
            except IntegrityError as exc:
                logger.error("Integrity constraint violation", db_class=ctx.db_class.__name__, error=str(exc))
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, detail=f"Integrity constraint violation: {str(exc)}"
                ) from exc
            except Exception as exc:
                logger.error("Database error", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(CreateResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            data: dict[str, Any],
            *,
            validate: bool = True,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> CreateResponse:
            """
            Create a single row.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            data
                Field values for new record
            validate
                Whether to validate with Pydantic
            timeout
                Request timeout in seconds

            Returns
            -------
            CreateResponse
                Creation result with created data

            Raises
            ------
            ValueError
                If validation fails or data is empty or integrity constraint violated
            HTTPError
                For other HTTP errors
            """
            if not data:
                raise ValueError("data cannot be empty")

            query_url = build_url(ctx.router_string, ctx.name)

            request_data = {
                "data": data,
                "validate": validate,
            }

            try:
                logger.debug("Creating row", url=query_url, fields=list(data.keys()))

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully created row")

                return result

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code == 422:
                        error_msg = "Validation failed"
                        logger.warning("Validation error", url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 409:
                        error_msg = "Integrity constraint violation (duplicate key, etc.)"
                        logger.warning("Integrity violation", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error creating row", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(name=f"create-{ctx.name}", help=f"Create a single {ctx.router_string}")
        @common_options.pz_client()
        @common_options.output()
        @common_options.fields()
        @common_options.json_data()
        @common_options.no_validate()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            field_values: tuple[tuple[str, str], ...],
            json_data: str | None,
            *,
            no_validate: bool,
            timeout: float,
        ) -> None:
            """Create a single row from remote API."""
            # Parse data
            create_data: dict[str, Any] = {}

            if json_data:
                try:
                    create_data = json_lib.loads(json_data)
                except json_lib.JSONDecodeError as exc:
                    click.echo(f"Error: Invalid JSON: {exc}", err=True)
                    raise click.Abort()

            for field_name, field_value in field_values:
                try:
                    create_data[field_name] = json_lib.loads(field_value)
                except json_lib.JSONDecodeError:
                    create_data[field_name] = field_value

            if not create_data:
                click.echo("Error: No data provided", err=True)
                raise click.Abort()

            try:
                result = client_method(
                    client_object,
                    data=create_data,
                    validate=not no_validate,
                    timeout=timeout,
                )

                output_json(result.model_dump(), output)
                click.echo(f"Successfully created {ctx.router_string}", err=True)

            except Exception as exc:
                handle_cli_error(exc, "create", ctx.router_string)

        return command


class CreateRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Atomic multi-row creation operation (all-or-nothing).

    Features:
    - All rows created in single transaction
    - Pydantic validation for all rows
    - get_create_kwargs preprocessing
    - Returns all created objects
    - Rollback on any error

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = CreateRowsOperation(ctx)
    >>> users_data = [
    ...     {"username": "alice", "email": "alice@example.com"},
    ...     {"username": "bob", "email": "bob@example.com"},
    ... ]
    >>> users = await op.create_multiple(session, users_data)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"create-{ctx.name}-multiple", help=f"Create multiple {ctx.router_string} atomically"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.json_file()
        @common_options.no_validate()
        @common_options.no_refresh()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            json_file: str,
            *,
            no_validate: bool,
            no_refresh: bool,
        ) -> None:
            """Create multiple rows atomically from JSON file."""
            # Load data from file
            try:
                async with aiofiles.open(json_file) as f:
                    rows_data = json_lib.load(f)
            except json_lib.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(rows_data, list):
                click.echo("Error: JSON file must contain an array", err=True)
                raise click.Abort()

            if not rows_data:
                click.echo("Error: Array is empty", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    results = await db_funcs.create.create_rows(
                        ctx.db_class,
                        session,
                        rows_data,
                        validate=not no_validate,
                        refresh=not no_refresh,
                    )

                    # Convert to dicts
                    results_data = [
                        {
                            column.name: getattr(result, column.name)
                            for column in ctx.db_class.__table__.columns
                        }
                        for result in results
                    ]

                    response = {
                        "success": True,
                        "count": len(results),
                        "resource": ctx.router_string,
                        "data": results_data,
                    }

                    output_json(response, output)
                    click.echo(f"Successfully created {len(results)} {ctx.router_string}", err=True)

            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except ValidationError as exc:
                click.echo(f"Error: Validation failed: {exc}", err=True)
                raise click.Abort()
            except IntegrityError as exc:
                logger.error("Integrity constraint violation", db_class=ctx.db_class.__name__, error=str(exc))
                click.echo(f"Error: Integrity constraint violation: {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error creating rows", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class CreateMultipleRequest(BaseModel):
            """Request model for multiple row creation."""

            rows_data: list[dict[str, Any]] = Field(
                ..., description="Array of objects to create", min_length=1
            )
            validate: bool = Field(default=True, description="Whether to validate with Pydantic")
            refresh: bool = Field(default=True, description="Whether to refresh rows after creation")

        @router.post(
            f"/{ctx.name}/multiple",
            response_model=CreateMultipleResponse,
            status_code=status.HTTP_201_CREATED,
            summary=f"Create multiple {ctx.router_string}",
            description=f"Create multiple {ctx.router_string} records atomically (all-or-nothing).",
        )
        async def endpoint(
            request: CreateMultipleRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> CreateMultipleResponse:
            """Create multiple rows atomically."""
            try:
                async with session.begin():
                    results = await db_funcs.create.create_rows(
                        ctx.db_class,
                        session,
                        request.rows_data,
                        validate=request.validate,
                        refresh=request.refresh,
                    )

                    # Convert to dicts
                    results_data = [
                        {
                            column.name: getattr(result, column.name)
                            for column in ctx.db_class.__table__.columns
                        }
                        for result in results
                    ]

                    return CreateMultipleResponse(
                        success=True,
                        count=len(results),
                        resource=ctx.router_string,
                        data=results_data,
                    )

            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except ValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Validation failed: {str(exc)}"
                ) from exc
            except IntegrityError as exc:
                logger.error("Integrity constraint violation", db_class=ctx.db_class.__name__, error=str(exc))
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, detail=f"Integrity constraint violation: {str(exc)}"
                ) from exc
            except Exception as exc:
                logger.error("Database error", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(CreateMultipleResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            rows_data: list[dict[str, Any]],
            *,
            validate: bool = True,
            refresh: bool = True,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> CreateMultipleResponse:
            """
            Create multiple rows atomically.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            rows_data
                List of dicts with field values for new records
            validate
                Whether to validate with Pydantic
            refresh
                Whether to refresh rows after creation
            timeout
                Request timeout in seconds

            Returns
            -------
            CreateMultipleResponse
                Creation result with all created data

            Raises
            ------
            ValueError
                If rows_data is empty, validation fails, or integrity constraint violated
            HTTPError
                For other HTTP errors
            """
            if not rows_data:
                raise ValueError("rows_data cannot be empty")

            query_url = build_url(ctx.router_string, ctx.name, "multiple")

            request_data = {
                "rows_data": rows_data,
                "validate": validate,
                "refresh": refresh,
            }

            try:
                logger.debug("Creating multiple rows", url=query_url, count=len(rows_data))

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully created rows", count=result.count)

                return result

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code == 422:
                        error_msg = "Validation failed"
                        logger.warning("Validation error", url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 409:
                        error_msg = "Integrity constraint violation"
                        logger.warning("Integrity violation", url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 400:
                        error_msg = "Invalid request"
                        logger.warning("Bad request", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error creating rows", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"create-{ctx.name}-multiple", help=f"Create multiple {ctx.router_string} atomically"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.json_file()
        @common_options.no_validate()
        @common_options.no_refresh()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            json_file: str,
            *,
            no_validate: bool,
            no_refresh: bool,
            timeout: float,
        ) -> None:
            """Create multiple rows atomically from remote API."""
            # Load data
            try:
                with open(json_file) as f:
                    rows_data = json_lib.load(f)
            except json_lib.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(rows_data, list):
                click.echo("Error: JSON file must contain an array", err=True)
                raise click.Abort()

            if not rows_data:
                click.echo("Error: Array is empty", err=True)
                raise click.Abort()

            try:
                result = client_method(
                    client_object,
                    rows_data=rows_data,
                    validate=not no_validate,
                    refresh=not no_refresh,
                    timeout=timeout,
                )

                output_json(result.model_dump(), output)
                click.echo(f"Successfully created {result.count} {ctx.router_string}", err=True)

            except Exception as exc:
                handle_cli_error(exc, "create", ctx.router_string)

        return command


class CreateRowsBatchedOperation[T: BaseModel](BaseOperation[T]):
    """
    Batched creation operation (partial success possible).

    Features:
    - Commits after each batch
    - Partial success if later batch fails
    - Configurable batch size
    - Progress tracking

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = CreateRowsBatchedOperation(ctx)
    >>> # Create 10,000 users in batches of 500
    >>> users = await op.create_batched(session, users_data, batch_size=500)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"create-{ctx.name}-batched", help=f"Create multiple {ctx.router_string} in batches"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.json_file()
        @common_options.batch_size()
        @common_options.no_validate()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            json_file: str,
            batch_size: int,
            *,
            no_validate: bool,
        ) -> None:
            """Create multiple rows in batches."""
            if batch_size < 1:
                click.echo("Error: batch-size must be at least 1", err=True)
                raise click.Abort()
            if batch_size > common_options.MAX_BATCH_SIZE:
                click.echo(f"Error: batch-size cannot exceed {common_options.MAX_BATCH_SIZE}", err=True)
                raise click.Abort()

            # Load data
            try:
                async with aiofiles.open(json_file) as f:
                    rows_data = json_lib.load(f)
            except json_lib.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(rows_data, list):
                click.echo("Error: JSON file must contain an array", err=True)
                raise click.Abort()

            if not rows_data:
                click.echo("Error: Array is empty", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    # Show progress
                    with click.progressbar(
                        length=len(rows_data),
                        label=f"Creating {ctx.router_string}",
                    ) as bar:
                        results = await db_funcs.create.create_rows_batched(
                            ctx.db_class,
                            session,
                            rows_data,
                            validate=not no_validate,
                            batch_size=batch_size,
                        )
                        bar.update(len(results))

                    # Convert to dicts
                    results_data = [
                        {
                            column.name: getattr(result, column.name)
                            for column in ctx.db_class.__table__.columns
                        }
                        for result in results
                    ]

                    response = {
                        "success": True,
                        "count": len(results),
                        "resource": ctx.router_string,
                        "data": results_data,
                    }

                    output_json(response, output)
                    click.echo(
                        f"\nSuccessfully created {len(results)} {ctx.router_string}"
                        "in batches of {batch_size}",
                        err=True,
                    )

            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except ValidationError as exc:
                click.echo(f"Error: Validation failed: {exc}", err=True)
                raise click.Abort()
            except IntegrityError as exc:
                logger.error("Integrity constraint violation", db_class=ctx.db_class.__name__, error=str(exc))
                click.echo(
                    f"Error: Integrity constraint violation (some batches may have succeeded): {exc}",
                    err=True,
                )
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error creating rows in batches",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class CreateBatchedRequest(BaseModel):
            """Request model for batched creation."""

            rows_data: list[dict[str, Any]] = Field(
                ..., description="Array of objects to create", min_length=1
            )
            validate: bool = Field(default=True, description="Whether to validate with Pydantic")
            batch_size: int = Field(
                common_options.DEFAULT_BATCH_SIZE,
                ge=1,
                le=common_options.common_options.MAX_BATCH_SIZE,
                description="Records per batch",
            )

        @router.post(
            f"/{ctx.name}/batched",
            response_model=CreateMultipleResponse,
            status_code=status.HTTP_201_CREATED,
            summary=f"Create multiple {ctx.router_string} in batches",
            description=f"Create multiple {ctx.router_string} in batches. "
                        "Partial success possible if later batch fails.",
        )
        async def endpoint(
            request: CreateBatchedRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> CreateMultipleResponse:
            """Create multiple rows in batches."""
            try:
                async with session.begin():
                    results = await db_funcs.create.create_rows_batched(
                        ctx.db_class,
                        session,
                        request.rows_data,
                        validate=request.validate,
                        batch_size=request.batch_size,
                    )

                    # Convert to dicts
                    results_data = [
                        {
                            column.name: getattr(result, column.name)
                            for column in ctx.db_class.__table__.columns
                        }
                        for result in results
                    ]

                    return CreateMultipleResponse(
                        success=True,
                        count=len(results),
                        resource=ctx.router_string,
                        data=results_data,
                    )

            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except ValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Validation failed: {str(exc)}"
                ) from exc
            except IntegrityError as exc:
                logger.error(
                    "Integrity constraint violation in batch", db_class=ctx.db_class.__name__, error=str(exc)
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Integrity constraint violation (some batches may have succeeded): {str(exc)}",
                ) from exc
            except Exception as exc:
                logger.error("Database error", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(CreateMultipleResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            rows_data: list[dict[str, Any]],
            *,
            validate: bool = True,
            batch_size: int = common_options.DEFAULT_BATCH_SIZE,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> CreateMultipleResponse:
            """
            Create multiple rows in batches.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            rows_data
                List of dicts with field values
            validate
                Whether to validate with Pydantic
            batch_size
                Records per batch
            timeout
                Request timeout in seconds

            Returns
            -------
            CreateMultipleResponse
                Creation result with all created data

            Raises
            ------
            ValueError
                If rows_data is empty or batch_size invalid
            HTTPError
                For HTTP errors

            Notes
            -----
            Partial success is possible - if a batch fails, previous
            batches will have been committed.
            """
            if not rows_data:
                raise ValueError("rows_data cannot be empty")
            if batch_size < 1 or batch_size > common_options.MAX_BATCH_SIZE:
                raise ValueError(f"batch_size must be between 1 and {common_options.MAX_BATCH_SIZE}")

            query_url = build_url(ctx.router_string, ctx.name, "batched")

            request_data = {
                "rows_data": rows_data,
                "validate": validate,
                "batch_size": batch_size,
            }

            try:
                logger.debug(
                    "Creating rows in batches", url=query_url, count=len(rows_data), batch_size=batch_size
                )

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully created rows in batches", count=result.count)

                return result

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code in (422, 409, 400):
                        error_msg = "Error creating batches (some may have succeeded)"
                        logger.warning("Batch creation error", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error creating batches", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"create-{ctx.name}-batched", help=f"Create multiple {ctx.router_string} in batches"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.json_file()
        @common_options.batch_size()
        @common_options.no_validate()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            json_file: str,
            batch_size: int,
            *,
            no_validate: bool,
            timeout: float,
        ) -> None:
            """Create multiple rows in batches from remote API."""
            if batch_size < 1 or batch_size > common_options.MAX_BATCH_SIZE:
                click.echo(
                    f"Error: batch-size must be between 1 and {common_options.MAX_BATCH_SIZE}", err=True
                )
                raise click.Abort()

            # Load data
            try:
                with open(json_file) as f:
                    rows_data = json_lib.load(f)
            except json_lib.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(rows_data, list) or not rows_data:
                click.echo("Error: JSON file must contain a non-empty array", err=True)
                raise click.Abort()

            try:
                result = client_method(
                    client_object,
                    rows_data=rows_data,
                    validate=not no_validate,
                    batch_size=batch_size,
                    timeout=timeout,
                )

                output_json(result.model_dump(), output)
                click.echo(f"Successfully created {result.count} {ctx.router_string} in batches", err=True)

            except Exception as exc:
                handle_cli_error(exc, "create batched", ctx.router_string)

        return command


class BulkInsertRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    High-performance bulk insert operation.

    Features:
    - SQL-based bulk insert (very fast)
    - Does NOT return created objects
    - Does NOT call get_create_kwargs
    - Does NOT trigger SQLAlchemy events
    - Returns count only

    Use Cases:
    - Large data imports
    - Performance-critical scenarios
    - Simple inserts without preprocessing

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("logs", Log)
    >>> op = BulkInsertRowsOperation(ctx)
    >>> # Insert 100,000 log entries fast
    >>> count = await op.bulk_insert(session, log_data)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"bulk-insert-{ctx.name}",
            help=f"Bulk insert {ctx.router_string} (fast, no objects returned)",
        )
        @common_options.db_engine()
        @common_options.json_file()
        @common_options.no_validate()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            json_file: str,
            *,
            no_validate: bool,
        ) -> None:
            """Bulk insert rows (maximum performance)."""
            # Load data
            try:
                async with aiofiles.open(json_file) as f:
                    rows_data = json_lib.load(f)
            except json_lib.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(rows_data, list):
                click.echo("Error: JSON file must contain an array", err=True)
                raise click.Abort()

            if not rows_data:
                click.echo("Error: Array is empty", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    count = await db_funcs.create.bulk_insert_rows(
                        ctx.db_class,
                        session,
                        rows_data,
                        validate=not no_validate,
                    )

                    click.echo(f"Successfully bulk inserted {count} {ctx.router_string}")

            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except ValidationError as exc:
                click.echo(f"Error: Validation failed: {exc}", err=True)
                raise click.Abort()
            except IntegrityError as exc:
                logger.error("Integrity constraint violation", db_class=ctx.db_class.__name__, error=str(exc))
                click.echo(f"Error: Integrity constraint violation: {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error bulk inserting rows", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class BulkInsertRequest(BaseModel):
            """Request model for bulk insert."""

            rows_data: list[dict[str, Any]] = Field(
                ..., description="Array of objects to insert", min_length=1
            )
            validate: bool = Field(default=True, description="Whether to validate with Pydantic")

        @router.post(
            f"/{ctx.name}/bulk",
            response_model=BulkInsertResponse,
            status_code=status.HTTP_201_CREATED,
            summary=f"Bulk insert {ctx.router_string}",
            description=f"High-performance bulk insert of {ctx.router_string}. "
                        "Does not return created objects.",
        )
        async def endpoint(
            request: BulkInsertRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> BulkInsertResponse:
            """Bulk insert rows without returning objects."""
            try:
                async with session.begin():
                    count = await db_funcs.create.bulk_insert_rows(
                        ctx.db_class,
                        session,
                        request.rows_data,
                        validate=request.validate,
                    )

                    return BulkInsertResponse(
                        success=True,
                        count=count,
                        resource=ctx.router_string,
                    )

            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except ValidationError as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Validation failed: {str(exc)}"
                ) from exc
            except IntegrityError as exc:
                logger.error("Integrity constraint violation", db_class=ctx.db_class.__name__, error=str(exc))
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT, detail=f"Integrity constraint violation: {str(exc)}"
                ) from exc
            except Exception as exc:
                logger.error("Database error", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(BulkInsertResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            rows_data: list[dict[str, Any]],
            *,
            validate: bool = True,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> int:
            """
            Bulk insert rows (maximum performance).

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            rows_data
                List of dicts with field values
            validate
                Whether to validate with Pydantic
            timeout
                Request timeout in seconds

            Returns
            -------
            int
                Number of rows inserted

            Raises
            ------
            ValueError
                If rows_data is empty or validation/integrity fails
            HTTPError
                For HTTP errors

            Notes
            -----
            - Does not return created objects
            - Does not call get_create_kwargs preprocessing
            - Maximum performance for large inserts
            """
            if not rows_data:
                raise ValueError("rows_data cannot be empty")

            query_url = build_url(ctx.router_string, ctx.name, "bulk")

            request_data = {
                "rows_data": rows_data,
                "validate": validate,
            }

            try:
                logger.debug("Bulk inserting rows", url=query_url, count=len(rows_data))

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully bulk inserted rows", count=result.count)

                return result.count

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code == 422:
                        error_msg = "Validation failed"
                        logger.warning("Validation error", url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 409:
                        error_msg = "Integrity constraint violation"
                        logger.warning("Integrity violation", url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 400:
                        error_msg = "Invalid request"
                        logger.warning("Bad request", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error bulk inserting", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"bulk-insert-{ctx.name}",
            help=f"Bulk insert {ctx.router_string} (fast, no objects returned)",
        )
        @common_options.pz_client()
        @common_options.json_file()
        @common_options.no_validate()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            json_file: str,
            *,
            no_validate: bool,
            timeout: float,
        ) -> None:
            """Bulk insert rows from remote API (maximum performance)."""
            # Load data
            try:
                with open(json_file) as f:
                    rows_data = json_lib.load(f)
            except json_lib.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(rows_data, list) or not rows_data:
                click.echo("Error: JSON file must contain a non-empty array", err=True)
                raise click.Abort()

            try:
                count = client_method(
                    client_object,
                    rows_data=rows_data,
                    validate=not no_validate,
                    timeout=timeout,
                )

                click.echo(f"Successfully bulk inserted {count} {ctx.router_string}")

            except Exception as exc:
                handle_cli_error(exc, "bulk insert", ctx.router_string)

        return command


# Export all create operation classes
__all__ = [
    "CreateRowOperation",
    "CreateRowsOperation",
    "CreateRowsBatchedOperation",
    "BulkInsertRowsOperation",
    "CreateResponse",
    "CreateMultipleResponse",
    "BulkInsertResponse",
]
