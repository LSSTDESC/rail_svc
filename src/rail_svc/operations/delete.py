"""
Delete operations for database records.

Provides three delete operations with different trade-offs:
- DeleteRowOperation: Single row deletion with hooks and data capture
- DeleteRowsOperation: Multiple row atomic deletion with hooks
- BulkDeleteRowsOperation: High-performance bulk deletion without hooks

All operations support both local and remote execution.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import click
from fastapi import APIRouter, Body, Depends, HTTPException, Path, status
from httpx import HTTPError, TimeoutException
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_core import ValidationError as CoreValidationError
from safir.dependencies.db_session import db_session_dependency
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, async_scoped_session
from structlog import get_logger
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from .. import db_funcs
from ..cli import common_options
from ..cli.utils import handle_cli_error
from .base import BaseOperation, build_url, output_json

if TYPE_CHECKING:
    from .client import ClientBase

logger = get_logger(__name__)


# Response models for delete operations
class DeleteResponse(BaseModel):
    """Response model for single row deletion."""

    success: bool = Field(..., description="Whether deletion succeeded")
    id_: int = Field(..., description="ID of deleted record")
    resource: str = Field(..., description="Resource type")
    data: dict[str, Any] | None = Field(None, description="Deleted row data if captured")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "id_": 123,
                "resource": "users",
                "data": {"id_": 123, "username": "alice", "email": "alice@example.com"},
            }
        }
    }


class DeleteMultipleResponse(BaseModel):
    """Response model for multiple row deletion."""

    success: bool = Field(..., description="Whether all deletions succeeded")
    count: int = Field(..., description="Number of rows deleted")
    ids: list[int] = Field(..., description="IDs of deleted records")
    resource: str = Field(..., description="Resource type")
    data: list[dict[str, Any]] | None = Field(None, description="Deleted rows data if captured")

    model_config = {
        "json_schema_extra": {
            "example": {"success": True, "count": 3, "ids": [1, 2, 3], "resource": "users", "data": None}
        }
    }


class BulkDeleteResponse(BaseModel):
    """Response model for bulk deletion."""

    success: bool = Field(..., description="Whether deletion succeeded")
    requested: int = Field(..., description="Number of IDs requested for deletion")
    deleted: int = Field(..., description="Number of rows actually deleted")
    resource: str = Field(..., description="Resource type")

    model_config = {
        "json_schema_extra": {
            "example": {"success": True, "requested": 100, "deleted": 98, "resource": "users"}
        }
    }


class DeleteRowOperation[T: BaseModel](BaseOperation[T]):
    """
    Single row deletion operation with hooks and optional data capture.

    Features:
    - Pre and post-delete hook support
    - Optional data capture before deletion
    - Transaction rollback on errors
    - Integrity constraint validation

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = DeleteRowOperation(ctx)
    >>> deleted_data = await op.delete(session, 123, capture_data=True)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(name=f"delete-{ctx.name}", help=f"Delete single {ctx.router_string} by ID")
        @common_options.db_engine()
        @common_options.output()
        @common_options.id_arg()
        @common_options.capture_data()
        @common_options.confirm()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            id_: int,
            *,
            capture_data: bool,
            confirm: bool,
        ) -> None:
            """Delete a single row by ID."""
            if id_ <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            # Confirmation prompt unless --confirm flag
            if not confirm:
                if not click.confirm(f"Are you sure you want to delete {ctx.router_string} {id_}?"):
                    click.echo("Deletion cancelled", err=True)
                    raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.delete.delete_row(
                        ctx.db_class,
                        session,
                        id_,
                        capture_data=capture_data,
                    )

                    response = {
                        "success": True,
                        "id_": id_,
                        "resource": ctx.router_string,
                        "data": result,
                    }

                    output_json(response, output)
                    click.echo(f"Successfully deleted {ctx.router_string} {id_}", err=True)

            except KeyError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except IntegrityError as exc:
                logger.error(
                    "Integrity constraint violation", db_class=ctx.db_class.__name__, id_=id_, error=str(exc)
                )
                click.echo(
                    f"Error: Cannot delete {ctx.router_string} {id_} - " "it is referenced by other records",
                    err=True,
                )
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error deleting row", db_class=ctx.db_class.__name__, id_=id_, error=str(exc), exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        @router.delete(
            f"/{ctx.name}/{{id_}}",
            response_model=DeleteResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Delete {ctx.router_string} by ID",
            description=f"Delete a single {ctx.router_string} record. Supports pre/post-delete hooks.",
        )
        async def endpoint(
            id_: int = Path(..., description="Record ID to delete", gt=0),
            *,
            capture_data: bool = Body(
                default=False, description="Capture row data before deletion", embed=True
            ),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> DeleteResponse:
            """Delete a single row by ID."""
            try:
                async with session.begin():
                    result = await db_funcs.delete.delete_row(
                        ctx.db_class,
                        session,
                        id_,
                        capture_data=capture_data,
                    )

                    return DeleteResponse(
                        success=True,
                        id_=id_,
                        resource=ctx.router_string,
                        data=result,
                    )

            except KeyError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            except IntegrityError as exc:
                logger.error(
                    "Integrity constraint violation", db_class=ctx.db_class.__name__, id_=id_, error=str(exc)
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Cannot delete {ctx.router_string} {id_} - " "it is referenced by other records",
                ) from exc
            except Exception as exc:
                logger.error(
                    "Database error", db_class=ctx.db_class.__name__, id_=id_, error=str(exc), exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(DeleteResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            id_: int,
            *,
            capture_data: bool = False,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> DeleteResponse:
            """
            Delete a single row by ID.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            id_
                Record ID to delete
            capture_data
                Whether to capture row data before deletion
            timeout
                Request timeout in seconds

            Returns
            -------
            DeleteResponse
                Deletion result with optional data

            Raises
            ------
            ValueError
                If the record is not found (404)
            HTTPError
                For other HTTP errors (409 for integrity violations)
            ValidationError
                If response validation fails
            """
            query_url = build_url(ctx.router_string, ctx.name, str(id_))

            try:
                logger.debug("Deleting row by ID", url=query_url, id_=id_)

                response = client_object.client.delete(
                    query_url,
                    json={"capture_data": capture_data},
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully deleted row", id_=id_)

                return result

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code == 404:
                        error_msg = f"{ctx.router_string} with ID {id_} not found"
                        logger.warning("Record not found", id_=id_, url=query_url)
                        raise ValueError(error_msg) from exc
                    if exc.response.status_code == 409:
                        error_msg = f"Cannot delete {ctx.router_string} {id_} - integrity constraint"
                        logger.warning("Integrity violation", id_=id_, url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error deleting row", url=query_url, id_=id_, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, id_=id_, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(name=f"delete-{ctx.name}", help=f"Delete {ctx.router_string} by ID")
        @common_options.pz_client()
        @common_options.output()
        @common_options.id_arg()
        @common_options.capture_data()
        @common_options.confirm()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            id_: int,
            *,
            capture_data: bool,
            confirm: bool,
            timeout: float,
        ) -> None:
            """Delete a single row by ID from remote API."""
            if id_ <= 0:
                click.echo("Error: ID must be positive", err=True)
                raise click.Abort()

            # Confirmation prompt unless --confirm flag
            if not confirm:
                if not click.confirm(f"Are you sure you want to delete {ctx.router_string} {id_}?"):
                    click.echo("Deletion cancelled", err=True)
                    raise click.Abort()

            try:
                result = client_method(client_object, id_, capture_data=capture_data, timeout=timeout)

                output_json(result.model_dump(), output)
                click.echo(f"Successfully deleted {ctx.router_string} {id_}", err=True)

            except Exception as exc:
                handle_cli_error(exc, "delete", ctx.router_string)

        return command


class DeleteRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Multiple row atomic deletion operation with hooks.

    Features:
    - Atomic transaction (all or nothing)
    - Pre and post-delete hooks for each row
    - Optional data capture
    - Validates all IDs exist before deleting

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = DeleteRowsOperation(ctx)
    >>> result = await op.delete_multiple(session, [1, 2, 3])
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"delete-{ctx.name}-multiple", help=f"Delete multiple {ctx.router_string} by IDs (atomic)"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.id_args()
        @common_options.capture_data()
        @common_options.confirm()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            ids: tuple[int, ...],
            *,
            capture_data: bool,
            confirm: bool,
        ) -> None:
            """Delete multiple rows atomically."""
            id_list = list(ids)

            if not id_list:
                click.echo("Error: Must provide at least one ID", err=True)
                raise click.Abort()

            if any(id_ <= 0 for id_ in id_list):
                click.echo("Error: All IDs must be positive", err=True)
                raise click.Abort()

            # Confirmation prompt unless --confirm flag
            if not confirm:
                if not click.confirm(
                    f"Are you sure you want to delete {len(id_list)} {ctx.router_string} records?"
                ):
                    click.echo("Deletion cancelled", err=True)
                    raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.delete.delete_rows(
                        ctx.db_class,
                        session,
                        id_list,
                        capture_data=capture_data,
                    )

                    response = {
                        "success": True,
                        "count": len(id_list),
                        "ids": id_list,
                        "resource": ctx.router_string,
                        "data": result,
                    }

                    output_json(response, output)
                    click.echo(f"Successfully deleted {len(id_list)} {ctx.router_string} records", err=True)

            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except KeyError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except IntegrityError as exc:
                logger.error(
                    "Integrity constraint violation",
                    db_class=ctx.db_class.__name__,
                    ids=id_list,
                    error=str(exc),
                )
                click.echo(
                    f"Error: Cannot delete - one or more {ctx.router_string} "
                    "records are referenced by other records",
                    err=True,
                )
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error deleting rows",
                    db_class=ctx.db_class.__name__,
                    ids=id_list,
                    error=str(exc),
                    exc_info=True,
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class DeleteMultipleRequest(BaseModel):
            """Request model for multiple row deletion."""

            ids: list[int] = Field(..., description="List of IDs to delete", min_length=1)
            capture_data: bool = Field(default=False, description="Capture row data before deletion")

        @router.delete(
            f"/{ctx.name}/multiple",
            response_model=DeleteMultipleResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Delete multiple {ctx.router_string}",
            description=f"Delete multiple {ctx.router_string} records atomically. All or nothing.",
        )
        async def endpoint(
            request: DeleteMultipleRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> DeleteMultipleResponse:
            """Delete multiple rows atomically."""
            try:
                async with session.begin():
                    result = await db_funcs.delete.delete_rows(
                        ctx.db_class,
                        session,
                        request.ids,
                        capture_data=request.capture_data,
                    )

                    return DeleteMultipleResponse(
                        success=True,
                        count=len(request.ids),
                        ids=request.ids,
                        resource=ctx.router_string,
                        data=result,
                    )

            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except KeyError as exc:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            except IntegrityError as exc:
                logger.error(
                    "Integrity constraint violation",
                    db_class=ctx.db_class.__name__,
                    ids=request.ids,
                    error=str(exc),
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Cannot delete - one or more {ctx.router_string} "
                    "records are referenced by other records",
                ) from exc
            except Exception as exc:
                logger.error(
                    "Database error",
                    db_class=ctx.db_class.__name__,
                    ids=request.ids,
                    error=str(exc),
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(DeleteMultipleResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            ids: list[int],
            *,
            capture_data: bool = False,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> DeleteMultipleResponse:
            """
            Delete multiple rows atomically.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            ids
                List of record IDs to delete
            capture_data
                Whether to capture row data before deletion
            timeout
                Request timeout in seconds

            Returns
            -------
            DeleteMultipleResponse
                Deletion result with optional data

            Raises
            ------
            ValueError
                If any record is not found or ids is empty
            HTTPError
                For other HTTP errors (409 for integrity violations)
            ValidationError
                If response validation fails
            """
            if not ids:
                raise ValueError("ids cannot be empty")

            query_url = build_url(ctx.router_string, ctx.name, "multiple")

            try:
                logger.debug("Deleting multiple rows", url=query_url, count=len(ids))

                response = client_object.client.delete(
                    query_url,
                    json={"ids": ids, "capture_data": capture_data},
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully deleted rows", count=len(ids))

                return result

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code == 404:
                        error_msg = f"One or more {ctx.router_string} not found"
                        logger.warning("Records not found", url=query_url)
                        raise ValueError(error_msg) from exc
                    if exc.response.status_code == 409:
                        error_msg = "Cannot delete - integrity constraint violation"
                        logger.warning("Integrity violation", url=query_url)
                        raise ValueError(error_msg) from exc
                    if exc.response.status_code == 400:
                        error_msg = "Invalid request"
                        logger.warning("Bad request", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error deleting rows", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"delete-{ctx.name}-multiple", help=f"Delete multiple {ctx.router_string} by IDs (atomic)"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.id_args()
        @common_options.capture_data()
        @common_options.confirm()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            ids: tuple[int, ...],
            *,
            capture_data: bool,
            confirm: bool,
            timeout: float,
        ) -> None:
            """Delete multiple rows atomically from remote API."""
            id_list = list(ids)

            if not id_list:
                click.echo("Error: Must provide at least one ID", err=True)
                raise click.Abort()

            if any(id_ <= 0 for id_ in id_list):
                click.echo("Error: All IDs must be positive", err=True)
                raise click.Abort()

            # Confirmation prompt unless --confirm flag
            if not confirm:
                if not click.confirm(
                    f"Are you sure you want to delete {len(id_list)} {ctx.router_string} records?"
                ):
                    click.echo("Deletion cancelled", err=True)
                    raise click.Abort()

            try:
                result = client_method(client_object, id_list, capture_data=capture_data, timeout=timeout)

                output_json(result.model_dump(), output)
                click.echo(f"Successfully deleted {len(id_list)} {ctx.router_string} records", err=True)

            except Exception as exc:
                handle_cli_error(exc, "delete", ctx.router_string)

        return command


class BulkDeleteRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    High-performance bulk deletion operation without hooks.

    Features:
    - SQL-based bulk deletion (fast)
    - No hooks called
    - No data capture
    - No validation that rows exist
    - Returns count of actually deleted rows

    Use Cases:
    - Large batch deletions
    - Performance-critical scenarios
    - When hooks are not needed

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("logs", Log)
    >>> op = BulkDeleteRowsOperation(ctx)
    >>> # Delete 10,000 old log entries efficiently
    >>> deleted = await op.bulk_delete(session, old_log_ids)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"bulk-delete-{ctx.name}", help=f"Bulk delete {ctx.router_string} (fast, no hooks)"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.id_args()
        @common_options.confirm()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            ids: tuple[int, ...],
            *,
            confirm: bool,
        ) -> None:
            """Bulk delete rows (no hooks, maximum performance)."""
            id_list = list(ids)

            if not id_list:
                click.echo("Error: Must provide at least one ID", err=True)
                raise click.Abort()

            if any(id_ <= 0 for id_ in id_list):
                click.echo("Error: All IDs must be positive", err=True)
                raise click.Abort()

            # Confirmation prompt unless --confirm flag
            if not confirm:
                click.echo("WARNING: Bulk delete does not call hooks or validate rows exist.", err=True)
                if not click.confirm(
                    f"Are you sure you want to bulk delete {len(id_list)} {ctx.router_string} records?"
                ):
                    click.echo("Deletion cancelled", err=True)
                    raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    deleted_count = await db_funcs.delete.bulk_delete_rows(
                        ctx.db_class,
                        session,
                        id_list,
                    )

                    response = {
                        "success": True,
                        "requested": len(id_list),
                        "deleted": deleted_count,
                        "resource": ctx.router_string,
                    }

                    output_json(response, output)
                    click.echo(
                        f"Bulk deleted {deleted_count}/{len(id_list)} {ctx.router_string} records", err=True
                    )

                    if deleted_count < len(id_list):
                        click.echo(f"Note: {len(id_list) - deleted_count} IDs did not exist", err=True)

            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except IntegrityError as exc:
                logger.error(
                    "Integrity constraint violation",
                    db_class=ctx.db_class.__name__,
                    ids=id_list,
                    error=str(exc),
                )
                click.echo(
                    f"Error: Cannot delete - one or more {ctx.router_string} "
                    "records are referenced by other records",
                    err=True,
                )
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error bulk deleting rows",
                    db_class=ctx.db_class.__name__,
                    ids=id_list,
                    error=str(exc),
                    exc_info=True,
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class BulkDeleteRequest(BaseModel):
            """Request model for bulk deletion."""

            ids: list[int] = Field(..., description="List of IDs to delete", min_length=1)

        @router.delete(
            f"/{ctx.name}/bulk",
            response_model=BulkDeleteResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Bulk delete {ctx.router_string}",
            description=f"High-performance bulk deletion of {ctx.router_string}. "
            "Does not call hooks or validate rows exist.",
        )
        async def endpoint(
            request: BulkDeleteRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> BulkDeleteResponse:
            """Bulk delete rows without hooks."""
            try:
                async with session.begin():
                    deleted_count = await db_funcs.delete.bulk_delete_rows(
                        ctx.db_class,
                        session,
                        request.ids,
                    )

                    return BulkDeleteResponse(
                        success=True,
                        requested=len(request.ids),
                        deleted=deleted_count,
                        resource=ctx.router_string,
                    )

            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except IntegrityError as exc:
                logger.error(
                    "Integrity constraint violation",
                    db_class=ctx.db_class.__name__,
                    ids=request.ids,
                    error=str(exc),
                )
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Cannot delete - one or more {ctx.router_string} "
                    "records are referenced by other records",
                ) from exc
            except Exception as exc:
                logger.error(
                    "Database error",
                    db_class=ctx.db_class.__name__,
                    ids=request.ids,
                    error=str(exc),
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(BulkDeleteResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            ids: list[int],
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> BulkDeleteResponse:
            """
            Bulk delete rows without hooks (maximum performance).

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            ids
                List of record IDs to delete
            timeout
                Request timeout in seconds

            Returns
            -------
            BulkDeleteResponse
                Deletion result with requested and actual counts

            Raises
            ------
            ValueError
                If ids is empty
            HTTPError
                For HTTP errors (409 for integrity violations)
            ValidationError
                If response validation fails

            Notes
            -----
            - Does not validate rows exist before deleting
            - Does not call pre/post-delete hooks
            - Returns count may be less than requested if some IDs don't exist
            """
            if not ids:
                raise ValueError("ids cannot be empty")

            query_url = build_url(ctx.router_string, ctx.name, "bulk")

            try:
                logger.debug("Bulk deleting rows", url=query_url, count=len(ids))

                response = client_object.client.delete(
                    query_url,
                    json={"ids": ids},
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Bulk delete complete", requested=len(ids), deleted=result.deleted)

                return result

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code == 409:
                        error_msg = "Cannot delete - integrity constraint violation"
                        logger.warning("Integrity violation", url=query_url)
                        raise ValueError(error_msg) from exc
                    if exc.response.status_code == 400:
                        error_msg = "Invalid request"
                        logger.warning("Bad request", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error bulk deleting rows", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"bulk-delete-{ctx.name}", help=f"Bulk delete {ctx.router_string} (fast, no hooks)"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.id_args()
        @common_options.confirm()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            ids: tuple[int, ...],
            *,
            confirm: bool,
            timeout: float,
        ) -> None:
            """Bulk delete rows from remote API (no hooks)."""
            id_list = list(ids)

            if not id_list:
                click.echo("Error: Must provide at least one ID", err=True)
                raise click.Abort()

            if any(id_ <= 0 for id_ in id_list):
                click.echo("Error: All IDs must be positive", err=True)
                raise click.Abort()

            # Confirmation prompt unless --confirm flag
            if not confirm:
                click.echo("WARNING: Bulk delete does not call hooks or validate rows exist.", err=True)
                if not click.confirm(
                    f"Are you sure you want to bulk delete {len(id_list)} {ctx.router_string} records?"
                ):
                    click.echo("Deletion cancelled", err=True)
                    raise click.Abort()

            try:
                result = client_method(client_object, id_list, timeout=timeout)

                output_json(result.model_dump(), output)
                click.echo(
                    f"Bulk deleted {result.deleted}/{result.requested} {ctx.router_string} records", err=True
                )

                if result.deleted < result.requested:
                    click.echo(f"Note: {result.requested - result.deleted} IDs did not exist", err=True)

            except Exception as exc:
                handle_cli_error(exc, "bulk delete", ctx.router_string)

        return command


# Export all delete operation classes
__all__ = [
    "DeleteRowOperation",
    "DeleteRowsOperation",
    "BulkDeleteRowsOperation",
    "DeleteResponse",
    "DeleteMultipleResponse",
    "BulkDeleteResponse",
]
