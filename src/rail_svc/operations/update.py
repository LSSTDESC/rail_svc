"""
Update operations for database records.

Provides two update operations:
- UpdateRowOperation: Single row update with validation
- UpdateRowsOperation: Multiple row atomic update

All operations support both local and remote execution with proper
validation, error handling, and transaction management.
"""

from __future__ import annotations

import click

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from httpx import HTTPError, TimeoutException
import json
import aiofiles

from pydantic import BaseModel, Field, TypeAdapter, field_validator
from pydantic_core import ValidationError as CoreValidationError
from safir.dependencies.db_session import db_session_dependency
from sqlalchemy.exc import StatementError
from sqlalchemy.ext.asyncio import AsyncEngine, async_scoped_session
from structlog import get_logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from .client import ClientBase

from ... import db_funcs
from ..cli import common_options
from ..cli.utils import handle_cli_error
from .base import BaseOperation, output_json, build_url

logger = get_logger(__name__)


# Response models for update operations
class UpdateResponse(BaseModel):
    """Response model for single row update."""

    success: bool = Field(..., description="Whether update succeeded")
    id: int = Field(..., description="ID of updated record")
    resource: str = Field(..., description="Resource type")
    updated_fields: list[str] = Field(..., description="List of fields that were updated")
    data: dict[str, Any] = Field(..., description="Complete updated record data")

    model_config = {"json_schema_extra": {
        "example": {
            "success": True,
            "id": 123,
            "resource": "users",
            "updated_fields": ["username", "email"],
            "data": {
                "id": 123,
                "username": "new_username",
                "email": "new@example.com",
                "updated_at": "2025-01-01T12:00:00Z"
            }
        }
    }}


class UpdateMultipleResponse(BaseModel):
    """Response model for multiple row update."""

    success: bool = Field(..., description="Whether all updates succeeded")
    count: int = Field(..., description="Number of rows updated")
    ids: list[int] = Field(..., description="IDs of updated records")
    resource: str = Field(..., description="Resource type")
    data: list[dict[str, Any]] = Field(..., description="Complete updated records data")

    model_config = {"json_schema_extra": {
        "example": {
            "success": True,
            "count": 3,
            "ids": [1, 2, 3],
            "resource": "users",
            "data": [
                {"id": 1, "status": "active"},
                {"id": 2, "status": "active"},
                {"id": 3, "status": "inactive"}
            ]
        }
    }}


class UpdateRowOperation[T: BaseModel](BaseOperation[T]):
    """
    Single row update operation.

    Features:
    - Validates row exists before updating
    - Prevents ID changes
    - Automatic commit with rollback on errors
    - Returns refreshed row data
    - Field-level validation

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = UpdateRowOperation(ctx)
    >>> updated = await op.update(session, 123, username="alice", email="alice@example.com")
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"update-{ctx.name}",
            help=f"Update single {ctx.router_string} by ID"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.id_arg()
        @common_options.field()
        @common_options.json_data()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            id: int,
            field: tuple[tuple[str, str], ...],
            json_data: str | None,
        ) -> None:
            """Update a single row by ID."""
            if id <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            # Parse update data
            update_data: dict[str, Any] = {}

            if json_data:
                try:
                    update_data = json.loads(json_data)
                except json.JSONDecodeError as exc:
                    click.echo(f"Error: Invalid JSON: {exc}", err=True)
                    raise click.Abort()

            # Add --field options (these override JSON if both provided)
            for field_name, field_value in field:
                # Try to parse as JSON for complex types
                try:
                    update_data[field_name] = json.loads(field_value)
                except (json.JSONDecodeError, ValueError):
                    # Just use as string if not valid JSON
                    update_data[field_name] = field_value

            if not update_data:
                click.echo("Error: No fields to update. Use --field or --json-data", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.update.update_row(
                        ctx.db_class,
                        session,
                        id,
                        **update_data,
                    )

                    # Convert result to dict for response
                    result_dict = {
                        column.name: getattr(result, column.name)
                        for column in ctx.db_class.__table__.columns
                    }

                    response = {
                        "success": True,
                        "id": id,
                        "resource": ctx.router_string,
                        "updated_fields": list(update_data.keys()),
                        "data": result_dict,
                    }

                    output_json(response, output)
                    click.echo(
                        f"Successfully updated {ctx.router_string} {id}",
                        err=True
                    )

            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except KeyError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except StatementError as exc:
                logger.error(
                    "Invalid update statement",
                    db_class=ctx.db_class.__name__,
                    id=id,
                    error=str(exc)
                )
                click.echo(
                    f"Error: Invalid field or value in update: {exc}",
                    err=True
                )
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error updating row",
                    db_class=ctx.db_class.__name__,
                    id=id,
                    error=str(exc),
                    exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class UpdateRequest(BaseModel):
            """Request model for row update."""
            updates: dict[str, Any] = Field(
                ...,
                description="Fields to update with their new values",
                min_length=1
            )

            @field_validator('updates')
            @classmethod
            def validate_no_id_change(cls, v: dict[str, Any]) -> dict[str, Any]:
                """Ensure ID is not being changed."""
                if 'id' in v:
                    raise ValueError("Cannot update 'id' field")
                return v

        @router.patch(
            f"/{ctx.name}/{{id}}",
            response_model=UpdateResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Update {ctx.router_string} by ID",
            description=f"Update a single {ctx.router_string} record. Returns refreshed data.",
        )
        async def endpoint(
            id: int = Path(..., description="Record ID to update", gt=0),
            request: UpdateRequest = Body(...),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> UpdateResponse:
            """Update a single row by ID."""
            try:
                async with session.begin():
                    result = await db_funcs.update.update_row(
                        ctx.db_class,
                        session,
                        id,
                        **request.updates,
                    )

                    # Convert result to dict
                    result_dict = {
                        column.name: getattr(result, column.name)
                        for column in ctx.db_class.__table__.columns
                    }

                    return UpdateResponse(
                        success=True,
                        id=id,
                        resource=ctx.router_string,
                        updated_fields=list(request.updates.keys()),
                        data=result_dict,
                    )

            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc)
                ) from exc
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(exc)
                ) from exc
            except StatementError as exc:
                logger.error(
                    "Invalid update statement",
                    db_class=ctx.db_class.__name__,
                    id=id,
                    error=str(exc)
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid field or value: {str(exc)}"
                ) from exc
            except Exception as exc:
                logger.error(
                    "Database error",
                    db_class=ctx.db_class.__name__,
                    id=id,
                    error=str(exc),
                    exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(UpdateResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            id: int,
            updates: dict[str, Any],
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> UpdateResponse:
            """
            Update a single row by ID.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            id
                Record ID to update
            updates
                Dictionary of fields to update with their new values
            timeout
                Request timeout in seconds

            Returns
            -------
            UpdateResponse
                Update result with refreshed data

            Raises
            ------
            ValueError
                If the record is not found, no fields provided, or trying to change ID
            HTTPError
                For other HTTP errors (422 for invalid fields/values)
            ValidationError
                If response validation fails
            """
            if not updates:
                raise ValueError("updates cannot be empty")

            if 'id' in updates:
                raise ValueError("Cannot update 'id' field")

            query_url = build_url(ctx.router_string, ctx.name, str(id))

            try:
                logger.debug("Updating row by ID", url=query_url, id=id, fields=list(updates.keys()))

                response = client_object.client.patch(
                    query_url,
                    json={"updates": updates},
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully updated row", id=id)

                return result

            except HTTPError as exc:
                if hasattr(exc, 'response'):
                    if exc.response.status_code == 404:
                        error_msg = f"{ctx.router_string} with ID {id} not found"
                        logger.warning("Record not found", id=id, url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 400:
                        error_msg = "Invalid update request"
                        logger.warning("Bad request", id=id, url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 422:
                        error_msg = "Invalid field or value in update"
                        logger.warning("Unprocessable entity", id=id, url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error(
                    "HTTP error updating row",
                    url=query_url,
                    id=id,
                    error=str(exc)
                )
                raise
            except CoreValidationError as exc:
                logger.error(
                    "Validation error parsing response",
                    url=query_url,
                    id=id,
                    error=str(exc)
                )
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"update-{ctx.name}",
            help=f"Update {ctx.router_string} by ID"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.id_arg()
        @common_options.field()
        @common_options.json_data()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            id: int,
            field: tuple[tuple[str, str], ...],
            json_data: str | None,
            timeout: float,
        ) -> None:
            """Update a single row by ID from remote API."""
            if id <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            # Parse update data
            update_data: dict[str, Any] = {}

            if json_data:
                try:
                    update_data = json.loads(json_data)
                except json.JSONDecodeError as exc:
                    click.echo(f"Error: Invalid JSON: {exc}", err=True)
                    raise click.Abort()

            # Add --field options (these override JSON if both provided)
            for field_name, field_value in field:
                # Try to parse as JSON for complex types
                try:
                    update_data[field_name] = json.loads(field_value)
                except (json.JSONDecodeError, ValueError):
                    # Just use as string if not valid JSON
                    update_data[field_name] = field_value

            if not update_data:
                click.echo("Error: No fields to update. Use --field or --json-data", err=True)
                raise click.Abort()

            try:
                result = client_method(
                    client_object,
                    id,
                    updates=update_data,
                    timeout=timeout
                )

                output_json(result.model_dump(), output)
                click.echo(
                    f"Successfully updated {ctx.router_string} {id}",
                    err=True
                )

            except Exception as exc:
                handle_cli_error(exc, "update", ctx.router_string)

        return command


class UpdateRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Multiple row atomic update operation.

    Features:
    - Atomic transaction (all or nothing)
    - Each update dict must contain 'id' key
    - Validates all rows exist before updating
    - Returns all refreshed row data
    - Field-level validation for each row

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = UpdateRowsOperation(ctx)
    >>> updates = [
    ...     {"id": 1, "status": "active"},
    ...     {"id": 2, "status": "active"},
    ...     {"id": 3, "status": "inactive"},
    ... ]
    >>> results = await op.update_multiple(session, updates)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"update-{ctx.name}-multiple",
            help=f"Update multiple {ctx.router_string} atomically"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.json_file()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            json_file: str,
        ) -> None:
            """Update multiple rows atomically from JSON file."""
            # Load updates from file
            try:
                async with aiofiles.open(json_file) as f:
                    updates = json.load(f)
            except json.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(updates, list):
                click.echo("Error: JSON file must contain an array of update objects", err=True)
                raise click.Abort()

            if not updates:
                click.echo("Error: Update array is empty", err=True)
                raise click.Abort()

            # Validate all have 'id'
            for i, update in enumerate(updates):
                if not isinstance(update, dict):
                    click.echo(f"Error: Update at index {i} is not an object", err=True)
                    raise click.Abort()
                if 'id' not in update:
                    click.echo(f"Error: Update at index {i} missing 'id' field", err=True)
                    raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    results = await db_funcs.update.update_rows(
                        ctx.db_class,
                        session,
                        updates,
                    )

                    # Convert results to dicts
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
                        "ids": [update['id'] for update in updates],
                        "resource": ctx.router_string,
                        "data": results_data,
                    }

                    output_json(response, output)
                    click.echo(
                        f"Successfully updated {len(results)} {ctx.router_string} records",
                        err=True
                    )

            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except KeyError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except StatementError as exc:
                logger.error(
                    "Invalid update statement",
                    db_class=ctx.db_class.__name__,
                    error=str(exc)
                )
                click.echo(
                    f"Error: Invalid field or value in one of the updates: {exc}",
                    err=True
                )
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error updating rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class UpdateMultipleRequest(BaseModel):
            """Request model for multiple row update."""
            updates: list[dict[str, Any]] = Field(
                ...,
                description="List of update objects, each must contain 'id'",
                min_length=1
            )

            @field_validator('updates')
            @classmethod
            def validate_all_have_id(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
                """Ensure all updates have 'id' field."""
                for i, update in enumerate(v):
                    if 'id' not in update:
                        raise ValueError(f"Update at index {i} missing 'id' field")
                return v

        @router.patch(
            f"/{ctx.name}/multiple",
            response_model=UpdateMultipleResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Update multiple {ctx.router_string}",
            description=f"Update multiple {ctx.router_string} records atomically. All or nothing.",
        )
        async def endpoint(
            request: UpdateMultipleRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> UpdateMultipleResponse:
            """Update multiple rows atomically."""
            try:
                async with session.begin():
                    results = await db_funcs.update.update_rows(
                        ctx.db_class,
                        session,
                        request.updates,
                    )

                    # Convert results to dicts
                    results_data = [
                        {
                            column.name: getattr(result, column.name)
                            for column in ctx.db_class.__table__.columns
                        }
                        for result in results
                    ]

                    return UpdateMultipleResponse(
                        success=True,
                        count=len(results),
                        ids=[update['id'] for update in request.updates],
                        resource=ctx.router_string,
                        data=results_data,
                    )

            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(exc)
                ) from exc
            except KeyError as exc:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=str(exc)
                ) from exc
            except StatementError as exc:
                logger.error(
                    "Invalid update statement",
                    db_class=ctx.db_class.__name__,
                    error=str(exc)
                )
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid field or value: {str(exc)}"
                ) from exc
            except Exception as exc:
                logger.error(
                    "Database error",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(UpdateMultipleResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            updates: list[dict[str, Any]],
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> UpdateMultipleResponse:
            """
            Update multiple rows atomically.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            updates
                List of update dicts, each must contain 'id' key
            timeout
                Request timeout in seconds

            Returns
            -------
            UpdateMultipleResponse
                Update result with refreshed data for all rows

            Raises
            ------
            ValueError
                If updates is empty, any update missing 'id', or any record not found
            HTTPError
                For other HTTP errors (422 for invalid fields/values)
            ValidationError
                If response validation fails
            """
            if not updates:
                raise ValueError("updates cannot be empty")

            # Validate all have 'id'
            for i, update in enumerate(updates):
                if 'id' not in update:
                    raise ValueError(f"Update at index {i} missing 'id' field")

            query_url = build_url(ctx.router_string, ctx.name, "multiple")

            try:
                logger.debug("Updating multiple rows", url=query_url, count=len(updates))

                response = client_object.client.patch(
                    query_url,
                    json={"updates": updates},
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully updated rows", count=len(updates))

                return result

            except HTTPError as exc:
                if hasattr(exc, 'response'):
                    if exc.response.status_code == 404:
                        error_msg = "One or more records not found"
                        logger.warning("Records not found", url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 400:
                        error_msg = "Invalid update request"
                        logger.warning("Bad request", url=query_url)
                        raise ValueError(error_msg) from exc
                    elif exc.response.status_code == 422:
                        error_msg = "Invalid field or value in one of the updates"
                        logger.warning("Unprocessable entity", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error(
                    "HTTP error updating rows",
                    url=query_url,
                    error=str(exc)
                )
                raise
            except CoreValidationError as exc:
                logger.error(
                    "Validation error parsing response",
                    url=query_url,
                    error=str(exc)
                )
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"update-{ctx.name}-multiple",
            help=f"Update multiple {ctx.router_string} atomically"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.json_file()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            json_file: str,
            timeout: float,
        ) -> None:
            """Update multiple rows atomically from remote API."""
            # Load updates from file
            try:
                with open(json_file) as f:
                    updates = json.load(f)
            except json.JSONDecodeError as exc:
                click.echo(f"Error: Invalid JSON file: {exc}", err=True)
                raise click.Abort()
            except OSError as exc:
                click.echo(f"Error: Cannot read file: {exc}", err=True)
                raise click.Abort()

            if not isinstance(updates, list):
                click.echo("Error: JSON file must contain an array of update objects", err=True)
                raise click.Abort()

            if not updates:
                click.echo("Error: Update array is empty", err=True)
                raise click.Abort()

            # Validate all have 'id'
            for i, update in enumerate(updates):
                if not isinstance(update, dict):
                    click.echo(f"Error: Update at index {i} is not an object", err=True)
                    raise click.Abort()
                if 'id' not in update:
                    click.echo(f"Error: Update at index {i} missing 'id' field", err=True)
                    raise click.Abort()

            try:
                result = client_method(
                    client_object,
                    updates=updates,
                    timeout=timeout
                )

                output_json(result.model_dump(), output)
                click.echo(
                    f"Successfully updated {result.count} {ctx.router_string} records",
                    err=True
                )

            except Exception as exc:
                handle_cli_error(exc, "update", ctx.router_string)

        return command


# Export all update operation classes
__all__ = [
    "UpdateRowOperation",
    "UpdateRowsOperation",
    "UpdateResponse",
    "UpdateMultipleResponse",
]
