"""
Filter operations for flexible database querying.

Provides advanced filtering capabilities with:
- Multiple comparison operators (eq, ne, lt, gt, in, like, etc.)
- Logical operators (AND/OR)
- Ordering and pagination
- Streaming for large result sets
- Count operations with filters

All operations support both local and remote execution.
"""

from __future__ import annotations

import json as json_lib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import click
from fastapi import APIRouter, Depends, HTTPException, Query, status
from httpx import HTTPError, TimeoutException
from pydantic import BaseModel, Field, TypeAdapter, field_validator
from pydantic_core import ValidationError as CoreValidationError
from safir.dependencies.db_session import db_session_dependency
from sqlalchemy.ext.asyncio import AsyncEngine, async_scoped_session
from structlog import get_logger
from tenacity import (retry, retry_if_exception_type, stop_after_attempt,
                      wait_exponential)

from .. import db_funcs
from ..cli import common_options
from ..cli.utils import handle_cli_error
from ..db_funcs.filter import Filter, FilterOp, OrderBy
from .base import (BaseOperation, build_url, output_json, output_pydantic_list,
                   output_pydantic_single)

if TYPE_CHECKING:
    from .client import ClientBase

logger = get_logger(__name__)


# Pydantic models for filter API
class FilterModel(BaseModel):
    """Pydantic model for a single filter condition."""

    field: str = Field(..., description="Field name to filter on")
    op: str = Field(..., description="Filter operator (eq, ne, lt, gt, in, like, etc.)")
    value: Any = Field(None, description="Value to compare against (not needed for is_null)")

    @field_validator("op")
    @classmethod
    def validate_op(cls, v: str) -> str:
        """Validate filter operator."""
        try:
            FilterOp(v)
        except ValueError:
            valid_ops = ", ".join([op.value for op in FilterOp])
            logger.error(f"Invalid operator '{v}'. Valid operators: {valid_ops}")
            raise
        return v

    def to_filter(self) -> Filter:
        """Convert to db_funcs Filter object."""
        return Filter(self.field, FilterOp(self.op), self.value)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"field": "status", "op": "eq", "value": "active"},
                {"field": "age", "op": "gt", "value": 18},
                {"field": "name", "op": "like", "value": "John%"},
                {"field": "deleted_at", "op": "is_null"},
            ]
        }
    }


class OrderByModel(BaseModel):
    """Pydantic model for ordering directive."""

    field: str = Field(..., description="Field name to order by")
    descending: bool = Field(default=False, description="True for descending, False for ascending")

    def to_order_by(self) -> OrderBy:
        """Convert to db_funcs OrderBy object."""
        return OrderBy(self.field, descending=self.descending)

    model_config = {
        "json_schema_extra": {
            "examples": [{"field": "created_at", "descending": True}, {"field": "name", "descending": False}]
        }
    }


class FilterRequest(BaseModel):
    """Request model for filter operations."""

    filters: list[FilterModel] | None = Field(None, description="List of filter conditions")
    logical_op: str = Field("and", description="How to combine filters: 'and' or 'or'")
    order_by: list[OrderByModel] | None = Field(None, description="Ordering directives")
    skip: int = Field(0, ge=0, description="Number of records to skip")
    limit: int = Field(
        common_options.DEFAULT_PAGE_SIZE,
        ge=1,
        le=common_options.MAX_PAGE_SIZE,
        description="Maximum records to return",
    )

    @field_validator("logical_op")
    @classmethod
    def validate_logical_op(cls, v: str) -> str:
        """Validate logical operator."""
        if v not in ("and", "or"):
            raise ValueError("logical_op must be 'and' or 'or'")
        return v


class FilterResponse(BaseModel):
    """Response model for filter operations."""

    success: bool = Field(..., description="Whether operation succeeded")
    count: int = Field(..., description="Number of results returned")
    total: int | None = Field(None, description="Total matching records (if counted)")
    skip: int = Field(..., description="Number of records skipped")
    limit: int = Field(..., description="Maximum records requested")
    resource: str = Field(..., description="Resource type")
    data: list[dict[str, Any]] = Field(..., description="Matching records")

    model_config = {
        "json_schema_extra": {
            "example": {
                "success": True,
                "count": 10,
                "total": 245,
                "skip": 0,
                "limit": 10,
                "resource": "users",
                "data": [{"id": 1, "username": "alice"}],
            }
        }
    }


class CountFilteredResponse(BaseModel):
    """Response model for count filtered operation."""

    success: bool = Field(..., description="Whether operation succeeded")
    count: int = Field(..., description="Number of matching records")
    resource: str = Field(..., description="Resource type")
    filters_applied: int = Field(..., description="Number of filters applied")

    model_config = {
        "json_schema_extra": {
            "example": {"success": True, "count": 245, "resource": "users", "filters_applied": 2}
        }
    }


def parse_filter_from_string(filter_str: str) -> Filter:
    """
    Parse filter from string format: field:op:value

    Examples:
        status:eq:active
        age:gt:18
        name:like:John%
        deleted_at:is_null
    """
    parts = filter_str.split(":", 2)

    if len(parts) < 2:
        raise ValueError(f"Invalid filter format: {filter_str}. Use field:op:value")

    field = parts[0]
    op_str = parts[1]

    # Validate operator
    try:
        op = FilterOp(op_str)
    except ValueError:
        valid_ops = ", ".join([o.value for o in FilterOp])
        logger.error(f"Invalid operator '{op_str}'. Valid: {valid_ops}")
        raise

    # Parse value
    if op in (FilterOp.IS_NULL, FilterOp.IS_NOT_NULL):
        value = None
    elif len(parts) < 3:
        raise ValueError(f"Filter {filter_str} requires a value")
    else:
        value_str = parts[2]

        # Try to parse as JSON for complex types
        try:
            value = json_lib.loads(value_str)
        except json_lib.JSONDecodeError:
            # Use as string
            value = value_str

    return Filter(field, op, value)


class FilterRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Advanced filtering operation with multiple operators and logic.

    Features:
    - Multiple comparison operators (eq, ne, lt, gt, in, like, ilike, etc.)
    - AND/OR logical combinations
    - Ordering (single or multiple fields)
    - Pagination support
    - Optional total count

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = FilterRowsOperation(ctx)
    >>> # Find active users over 18, ordered by creation date
    >>> filters = [
    ...     Filter("status", FilterOp.EQ, "active"),
    ...     Filter("age", FilterOp.GT, 18)
    ... ]
    >>> results = await op.filter(session, filters, order_by=OrderBy("created_at", descending=True))
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(name=f"filter-{ctx.name}", help=f"Filter {ctx.router_string} with advanced criteria")
        @common_options.db_engine()
        @common_options.output()
        @common_options.filters()
        @common_options.logical_op()
        @common_options.order_by()
        @common_options.skip()
        @common_options.limit()
        @common_options.with_count()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            filter_strs: tuple[str, ...],
            logical_op: str,
            order_by: tuple[str, ...],
            skip: int,
            limit: int,
            *,
            with_count: bool,
        ) -> None:
            """Filter rows with advanced criteria."""
            # Parse filters
            filters: list[Filter] | None = None
            if filter_strs:
                try:
                    filters = [parse_filter_from_string(f) for f in filter_strs]
                except ValueError as exc:
                    click.echo(f"Error: {exc}", err=True)
                    raise click.Abort()

            # Parse order_by
            order_by_list: list[OrderBy] | None = None
            if order_by:
                order_by_list = []
                for field_str in order_by:
                    if field_str.startswith("-"):
                        order_by_list.append(OrderBy(field_str[1:], descending=True))
                    else:
                        order_by_list.append(OrderBy(field_str, descending=False))

            try:
                async with db_engine().begin() as session:
                    # Get results
                    results = await db_funcs.filter.filter_rows(
                        ctx.db_class,
                        session,
                        filters=filters,
                        logical_op=logical_op,
                        order_by=order_by_list,
                        skip=skip,
                        limit=limit,
                    )

                    # Optionally get total count
                    total = None
                    if with_count and filters:
                        total = await db_funcs.filter.count_filtered_rows(
                            ctx.db_class,
                            session,
                            filters=filters,
                            logical_op=logical_op,
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
                        "total": total,
                        "skip": skip,
                        "limit": limit,
                        "resource": ctx.router_string,
                        "data": results_data,
                    }

                    output_json(response, output)

                    if total is not None:
                        click.echo(
                            f"Showing {len(results)} of {total} matching {ctx.router_string}", err=True
                        )
                    else:
                        click.echo(f"Found {len(results)} {ctx.router_string}", err=True)

            except AttributeError as exc:
                click.echo(f"Error: Invalid field: {exc}", err=True)
                raise click.Abort()
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error filtering rows", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        @router.post(
            f"/{ctx.name}/filter",
            response_model=FilterResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Filter {ctx.router_string}",
            description=f"Advanced filtering of {ctx.router_string} with multiple operators and logic.",
        )
        async def endpoint(
            request: FilterRequest,
            *,
            with_count: bool = Query(default=False, description="Include total count of matching records"),
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> FilterResponse:
            """Filter rows with advanced criteria."""
            try:
                # Convert Pydantic models to db_funcs objects
                filters = [f.to_filter() for f in request.filters] if request.filters else None
                order_by = [o.to_order_by() for o in request.order_by] if request.order_by else None

                async with session.begin():
                    # Get results
                    results = await db_funcs.filter.filter_rows(
                        ctx.db_class,
                        session,
                        filters=filters,
                        logical_op=request.logical_op,
                        order_by=order_by,
                        skip=request.skip,
                        limit=request.limit,
                    )

                    # Optionally get total count
                    total = None
                    if with_count and filters:
                        total = await db_funcs.filter.count_filtered_rows(
                            ctx.db_class,
                            session,
                            filters=filters,
                            logical_op=request.logical_op,
                        )

                    # Convert results to dicts
                    results_data = [
                        {
                            column.name: getattr(result, column.name)
                            for column in ctx.db_class.__table__.columns
                        }
                        for result in results
                    ]

                    return FilterResponse(
                        success=True,
                        count=len(results),
                        total=total,
                        skip=request.skip,
                        limit=request.limit,
                        resource=ctx.router_string,
                        data=results_data,
                    )

            except AttributeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid field: {str(exc)}"
                ) from exc
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except Exception as exc:
                logger.error(
                    "Database error filtering rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(FilterResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            filters: list[dict[str, Any]] | None = None,
            logical_op: str = "and",
            order_by: list[dict[str, Any]] | None = None,
            skip: int = 0,
            limit: int = common_options.DEFAULT_PAGE_SIZE,
            *,
            with_count: bool = False,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> FilterResponse:
            """
            Filter rows with advanced criteria.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            filters
                List of filter dicts with keys: field, op, value
                Example: [{"field": "status", "op": "eq", "value": "active"}]
            logical_op
                How to combine filters: "and" or "or"
            order_by
                List of order dicts with keys: field, descending
                Example: [{"field": "created_at", "descending": True}]
            skip
                Number of records to skip
            limit
                Maximum records to return
            with_count
                Include total count of matching records
            timeout
                Request timeout in seconds

            Returns
            -------
            FilterResponse
                Filter results with data and metadata

            Raises
            ------
            ValueError
                If filters/ordering are invalid
            HTTPError
                For HTTP errors
            ValidationError
                If response validation fails
            """
            query_url = build_url(ctx.router_string, ctx.name, "filter")

            request_data = {
                "filters": filters,
                "logical_op": logical_op,
                "order_by": order_by,
                "skip": skip,
                "limit": limit,
            }

            params = {"with_count": with_count}

            try:
                logger.debug(
                    "Filtering rows",
                    url=query_url,
                    filter_count=len(filters) if filters else 0,
                    logical_op=logical_op,
                )

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    params=params,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully filtered rows", count=result.count)

                return result

            except HTTPError as exc:
                if hasattr(exc, "response") and exc.response.status_code == 400:
                    error_msg = "Invalid filter request"
                    logger.warning("Bad request", url=query_url)
                    raise ValueError(error_msg) from exc

                logger.error("HTTP error filtering rows", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(name=f"filter-{ctx.name}", help=f"Filter {ctx.router_string} with advanced criteria")
        @common_options.pz_client()
        @common_options.output()
        @common_options.filters()
        @common_options.logical_op()
        @common_options.order_by()
        @common_options.skip()
        @common_options.limit()
        @common_options.with_count()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            filter_strs: tuple[str, ...],
            logical_op: str,
            order_by: tuple[str, ...],
            skip: int,
            limit: int,
            *,
            with_count: bool,
            timeout: float,
        ) -> None:
            """Filter rows from remote API with advanced criteria."""
            # Parse filters
            filters: list[dict[str, Any]] | None = None
            if filter_strs:
                try:
                    filters = []
                    for filter_str in filter_strs:
                        filter_obj = parse_filter_from_string(filter_str)
                        filters.append(
                            {
                                "field": filter_obj.field,
                                "op": filter_obj.op.value,
                                "value": filter_obj.value,
                            }
                        )
                except ValueError as exc:
                    click.echo(f"Error: {exc}", err=True)
                    raise click.Abort()

            # Parse order_by
            order_by_list: list[dict[str, Any]] | None = None
            if order_by:
                order_by_list = []
                for field_str in order_by:
                    if field_str.startswith("-"):
                        order_by_list.append({"field": field_str[1:], "descending": True})
                    else:
                        order_by_list.append({"field": field_str, "descending": False})

            try:
                result = client_method(
                    client_object,
                    filters=filters,
                    logical_op=logical_op,
                    order_by=order_by_list,
                    skip=skip,
                    limit=limit,
                    with_count=with_count,
                    timeout=timeout,
                )

                output_json(result.model_dump(), output)

                if result.total is not None:
                    click.echo(
                        f"Showing {result.count} of {result.total} matching {ctx.router_string}", err=True
                    )
                else:
                    click.echo(f"Found {result.count} {ctx.router_string}", err=True)

            except Exception as exc:
                handle_cli_error(exc, "filter", ctx.router_string)

        return command


class CountFilteredRowsOperation[T: BaseModel](BaseOperation[T]):
    """
    Count rows matching filter criteria.

    Useful for pagination metadata without fetching actual data.

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = CountFilteredRowsOperation(ctx)
    >>> filters = [Filter("status", FilterOp.EQ, "active")]
    >>> count = await op.count(session, filters)
    >>> print(f"Found {count} active users")
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"count-filtered-{ctx.name}", help=f"Count {ctx.router_string} matching filter criteria"
        )
        @common_options.db_engine()
        @common_options.filters()
        @common_options.logical_op()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            filter_strs: tuple[str, ...],
            logical_op: str,
        ) -> None:
            """Count rows matching filter criteria."""
            # Parse filters
            filters: list[Filter] | None = None
            if filter_strs:
                try:
                    filters = [parse_filter_from_string(f) for f in filter_strs]
                except ValueError as exc:
                    click.echo(f"Error: {exc}", err=True)
                    raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    count = await db_funcs.filter.count_filtered_rows(
                        ctx.db_class,
                        session,
                        filters=filters,
                        logical_op=logical_op,
                    )

                    if filters:
                        click.echo(f"Found {count} matching {ctx.router_string}")
                    else:
                        click.echo(f"Total {ctx.router_string}: {count}")

            except AttributeError as exc:
                click.echo(f"Error: Invalid field: {exc}", err=True)
                raise click.Abort()
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error counting filtered rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class CountFilterRequest(BaseModel):
            """Request model for count filtered operation."""

            filters: list[FilterModel] | None = Field(None, description="Filter conditions")
            logical_op: str = Field("and", description="How to combine filters: 'and' or 'or'")

            @field_validator("logical_op")
            @classmethod
            def validate_logical_op(cls, v: str) -> str:
                if v not in ("and", "or"):
                    raise ValueError("logical_op must be 'and' or 'or'")
                return v

        @router.post(
            f"/{ctx.name}/count-filtered",
            response_model=CountFilteredResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Count filtered {ctx.router_string}",
            description=f"Count {ctx.router_string} matching filter criteria.",
        )
        async def endpoint(
            request: CountFilterRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> CountFilteredResponse:
            """Count rows matching filter criteria."""
            try:
                filters = [f.to_filter() for f in request.filters] if request.filters else None

                async with session.begin():
                    count = await db_funcs.filter.count_filtered_rows(
                        ctx.db_class,
                        session,
                        filters=filters,
                        logical_op=request.logical_op,
                    )

                    return CountFilteredResponse(
                        success=True,
                        count=count,
                        resource=ctx.router_string,
                        filters_applied=len(request.filters) if request.filters else 0,
                    )

            except AttributeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid field: {str(exc)}"
                ) from exc
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except Exception as exc:
                logger.error(
                    "Database error counting filtered rows",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(CountFilteredResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            filters: list[dict[str, Any]] | None = None,
            logical_op: str = "and",
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> int:
            """
            Count rows matching filter criteria.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            filters
                List of filter dicts with keys: field, op, value
            logical_op
                How to combine filters: "and" or "or"
            timeout
                Request timeout in seconds

            Returns
            -------
            int
                Number of matching rows

            Raises
            ------
            ValueError
                If filters are invalid
            HTTPError
                For HTTP errors
            """
            query_url = build_url(ctx.router_string, ctx.name, "count-filtered")

            request_data = {
                "filters": filters,
                "logical_op": logical_op,
            }

            try:
                logger.debug(
                    "Counting filtered rows", url=query_url, filter_count=len(filters) if filters else 0
                )

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully counted rows", count=result.count)

                return result.count

            except HTTPError as exc:
                if hasattr(exc, "response") and exc.response.status_code == 400:
                    error_msg = "Invalid filter request"
                    logger.warning("Bad request", url=query_url)
                    raise ValueError(error_msg) from exc

                logger.error("HTTP error counting rows", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"count-filtered-{ctx.name}", help=f"Count {ctx.router_string} matching filter criteria"
        )
        @common_options.pz_client()
        @common_options.filters()
        @common_options.logical_op()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            filter_strs: tuple[str, ...],
            logical_op: str,
            timeout: float,
        ) -> None:
            """Count rows matching filter criteria from remote API."""
            # Parse filters
            filters: list[dict[str, Any]] | None = None
            if filter_strs:
                try:
                    filters = []
                    for filter_str in filter_strs:
                        filter_obj = parse_filter_from_string(filter_str)
                        filters.append(
                            {
                                "field": filter_obj.field,
                                "op": filter_obj.op.value,
                                "value": filter_obj.value,
                            }
                        )
                except ValueError as exc:
                    click.echo(f"Error: {exc}", err=True)
                    raise click.Abort()

            try:
                count = client_method(
                    client_object,
                    filters=filters,
                    logical_op=logical_op,
                    timeout=timeout,
                )

                if filters:
                    click.echo(f"Found {count} matching {ctx.router_string}")
                else:
                    click.echo(f"Total {ctx.router_string}: {count}")

            except Exception as exc:
                handle_cli_error(exc, "count", ctx.router_string)

        return command


class FindByOperation[T: BaseModel](BaseOperation[T]):
    """
    Convenience operation for simple equality filters.

    Simpler interface than FilterRowsOperation for the common case
    of filtering by exact field values with AND logic.

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = FindByOperation(ctx)
    >>> # Find all active admin users
    >>> users = await op.find_by(session, status="active", role="admin")
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"find-{ctx.name}-by", help=f"Find {ctx.router_string} by exact field values (AND logic)"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.fields()
        @common_options.order_by()
        @common_options.skip()
        @common_options.limit()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            field_values: tuple[tuple[str, str], ...],
            order_by: tuple[str, ...],
            skip: int,
            limit: int,
        ) -> None:
            """Find rows by exact field values."""
            if not field_values:
                click.echo("Error: Must provide at least one --field option", err=True)
                raise click.Abort()

            # Parse field values
            kwargs = {}
            for field_name, field_value in field_values:
                # Try to parse as JSON for complex types
                try:
                    kwargs[field_name] = json_lib.loads(field_value)
                except json_lib.JSONDecodeError:
                    kwargs[field_name] = field_value

            # Parse order_by
            order_by_list: list[OrderBy] | None = None
            if order_by:
                order_by_list = []
                for field_str in order_by:
                    if field_str.startswith("-"):
                        order_by_list.append(OrderBy(field_str[1:], descending=True))
                    else:
                        order_by_list.append(OrderBy(field_str, descending=False))

            try:
                async with db_engine().begin() as session:
                    results = await db_funcs.filter.find_by(
                        ctx.db_class,
                        session,
                        order_by=order_by_list,
                        skip=skip,
                        limit=limit,
                        **kwargs,
                    )

                    output_pydantic_list(results, output, ctx.col_names_optional)
                    click.echo(f"Found {len(results)} {ctx.router_string}", err=True)

            except AttributeError as exc:
                click.echo(f"Error: Invalid field: {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error finding rows", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx

        class FindByRequest(BaseModel):
            """Request model for find by operation."""

            criteria: dict[str, Any] = Field(
                ..., description="Field-value pairs to match (all must match)", min_length=1
            )
            order_by: list[OrderByModel] | None = Field(None, description="Ordering directives")
            skip: int = Field(0, ge=0, description="Number of records to skip")
            limit: int = Field(
                common_options.DEFAULT_PAGE_SIZE,
                ge=1,
                le=common_options.MAX_PAGE_SIZE,
                description="Max records",
            )

        @router.post(
            f"/{ctx.name}/find-by",
            response_model=FilterResponse,
            status_code=status.HTTP_200_OK,
            summary=f"Find {ctx.router_string} by exact values",
            description=f"Find {ctx.router_string} by exact field values (AND logic).",
        )
        async def endpoint(
            request: FindByRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> FilterResponse:
            """Find rows by exact field values."""
            try:
                order_by = [o.to_order_by() for o in request.order_by] if request.order_by else None

                async with session.begin():
                    results = await db_funcs.filter.find_by(
                        ctx.db_class,
                        session,
                        order_by=order_by,
                        skip=request.skip,
                        limit=request.limit,
                        **request.criteria,
                    )

                    # Convert results to dicts
                    results_data = [
                        {
                            column.name: getattr(result, column.name)
                            for column in ctx.db_class.__table__.columns
                        }
                        for result in results
                    ]

                    return FilterResponse(
                        success=True,
                        count=len(results),
                        total=None,
                        skip=request.skip,
                        limit=request.limit,
                        resource=ctx.router_string,
                        data=results_data,
                    )

            except AttributeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid field: {str(exc)}"
                ) from exc
            except Exception as exc:
                logger.error(
                    "Database error in find_by", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
                ) from exc

        return endpoint

    def create_client_method(self) -> Callable:
        ctx = self.ctx
        response_adapter = TypeAdapter(FilterResponse)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(TimeoutException),
        )
        def client_method(
            client_object: ClientBase,
            criteria: dict[str, Any],
            order_by: list[dict[str, Any]] | None = None,
            skip: int = 0,
            limit: int = common_options.DEFAULT_PAGE_SIZE,
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> FilterResponse:
            """
            Find rows by exact field values.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            criteria
                Dict of field-value pairs to match (all must match)
            order_by
                List of order dicts with keys: field, descending
            skip
                Number of records to skip
            limit
                Maximum records to return
            timeout
                Request timeout in seconds

            Returns
            -------
            FilterResponse
                Matching records with metadata

            Raises
            ------
            ValueError
                If criteria is empty or fields are invalid
            HTTPError
                For HTTP errors
            """
            if not criteria:
                raise ValueError("criteria cannot be empty")

            query_url = build_url(ctx.router_string, ctx.name, "find-by")

            request_data = {
                "criteria": criteria,
                "order_by": order_by,
                "skip": skip,
                "limit": limit,
            }

            try:
                logger.debug("Finding rows by criteria", url=query_url, criteria=criteria)

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully found rows", count=result.count)

                return result

            except HTTPError as exc:
                if hasattr(exc, "response") and exc.response.status_code == 400:
                    error_msg = "Invalid field in criteria"
                    logger.warning("Bad request", url=query_url)
                    raise ValueError(error_msg) from exc

                logger.error("HTTP error finding rows", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(name=f"find-{ctx.name}-by", help=f"Find {ctx.router_string} by exact field values")
        @common_options.pz_client()
        @common_options.output()
        @common_options.fields()
        @common_options.order_by()
        @common_options.skip()
        @common_options.limit()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            field_values: tuple[tuple[str, str], ...],
            order_by: tuple[str, ...],
            skip: int,
            limit: int,
            timeout: float,
        ) -> None:
            """Find rows by exact field values from remote API."""
            if not field_values:
                click.echo("Error: Must provide at least one --field option", err=True)
                raise click.Abort()

            # Parse field values
            criteria = {}
            for field_name, field_value in field_values:
                try:
                    criteria[field_name] = json_lib.loads(field_value)
                except json_lib.JSONDecodeError:
                    criteria[field_name] = field_value

            # Parse order_by
            order_by_list: list[dict[str, Any]] | None = None
            if order_by:
                order_by_list = []
                for field_str in order_by:
                    if field_str.startswith("-"):
                        order_by_list.append({"field": field_str[1:], "descending": True})
                    else:
                        order_by_list.append({"field": field_str, "descending": False})

            try:
                result = client_method(
                    client_object,
                    criteria=criteria,
                    order_by=order_by_list,
                    skip=skip,
                    limit=limit,
                    timeout=timeout,
                )

                output_json(result.model_dump(), output)
                click.echo(f"Found {result.count} {ctx.router_string}", err=True)

            except Exception as exc:
                handle_cli_error(exc, "find", ctx.router_string)

        return command


class FilterOneOperation[T: BaseModel](BaseOperation[T]):
    """
    Find exactly one row matching filter criteria.

    Raises error if zero or multiple rows match.

    Examples
    --------
    >>> ctx = OperationContext.from_db_class("users", User)
    >>> op = FilterOneOperation(ctx)
    >>> # Find user by unique email
    >>> filters = [Filter("email", FilterOp.EQ, "alice@example.com")]
    >>> user = await op.filter_one(session, filters)
    """

    def create_local_command(self, group: click.Group) -> Callable:
        ctx = self.ctx

        @group.command(
            name=f"filter-one-{ctx.name}", help=f"Find exactly one {ctx.router_string} matching criteria"
        )
        @common_options.db_engine()
        @common_options.output()
        @common_options.filters()
        @common_options.logical_op()
        async def command(
            db_engine: Callable[[], AsyncEngine],
            output: common_options.OutputEnum | None,
            filter_strs: tuple[str, ...],
            logical_op: str,
        ) -> None:
            """Find exactly one row matching criteria."""
            # Parse filters
            try:
                filters = [parse_filter_from_string(f) for f in filter_strs]
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

            try:
                async with db_engine().begin() as session:
                    result = await db_funcs.filter.filter_one(
                        ctx.db_class,
                        session,
                        filters=filters,
                        logical_op=logical_op,
                    )

                    output_pydantic_single(result, output, ctx.col_names_optional)
                    click.echo(f"Found matching {ctx.router_string}", err=True)

            except KeyError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except AttributeError as exc:
                click.echo(f"Error: Invalid field: {exc}", err=True)
                raise click.Abort()
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()
            except click.Abort:
                raise
            except Exception as exc:
                logger.error(
                    "Error finding single row", db_class=ctx.db_class.__name__, error=str(exc), exc_info=True
                )
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

        return command

    def create_router_endpoint(self, router: APIRouter) -> Callable:
        ctx = self.ctx
        ResponseModel = ctx.response_class

        class FilterOneRequest(BaseModel):
            """Request model for filter one operation."""

            filters: list[FilterModel] = Field(
                ..., description="Filter conditions (must match exactly one row)", min_length=1
            )
            logical_op: str = Field("and", description="How to combine filters: 'and' or 'or'")

            @field_validator("logical_op")
            @classmethod
            def validate_logical_op(cls, v: str) -> str:
                if v not in ("and", "or"):
                    raise ValueError("logical_op must be 'and' or 'or'")
                return v

        @router.post(
            f"/{ctx.name}/filter-one",
            response_model=ResponseModel,
            status_code=status.HTTP_200_OK,
            summary=f"Find one {ctx.router_string}",
            description=f"Find exactly one {ctx.router_string} matching criteria. "
                        "Returns 404 if none found, 409 if multiple found.",
        )
        async def endpoint(
            request: FilterOneRequest,
            session: async_scoped_session = Depends(db_session_dependency),
        ) -> BaseModel:
            """Find exactly one row matching criteria."""
            try:
                filters = [f.to_filter() for f in request.filters]

                async with session.begin():
                    result = await db_funcs.filter.filter_one(
                        ctx.db_class,
                        session,
                        filters=filters,
                        logical_op=request.logical_op,
                    )

                    return result

            except KeyError as exc:
                # No rows found
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
            except AttributeError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid field: {str(exc)}"
                ) from exc
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            except Exception as exc:
                # Check if it's a "multiple rows" error
                if "Multiple" in str(exc):
                    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

                logger.error(
                    "Database error in filter_one",
                    db_class=ctx.db_class.__name__,
                    error=str(exc),
                    exc_info=True,
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(exc)}"
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
            filters: list[dict[str, Any]],
            logical_op: str = "and",
            timeout: float = common_options.DEFAULT_TIMEOUT,
        ) -> BaseModel:
            """
            Find exactly one row matching criteria.

            Parameters
            ----------
            client_object
                HTTP client object with a .client attribute (httpx.Client)
            filters
                List of filter dicts with keys: field, op, value
            logical_op
                How to combine filters: "and" or "or"
            timeout
                Request timeout in seconds

            Returns
            -------
            BaseModel
                The single matching record

            Raises
            ------
            ValueError
                If no rows found, multiple rows found, or invalid filters
            HTTPError
                For other HTTP errors
            """
            if not filters:
                raise ValueError("filters cannot be empty")

            query_url = build_url(ctx.router_string, ctx.name, "filter-one")

            request_data = {
                "filters": filters,
                "logical_op": logical_op,
            }

            try:
                logger.debug("Finding single row by filters", url=query_url, filter_count=len(filters))

                response = client_object.client.post(
                    query_url,
                    json=request_data,
                    timeout=timeout,
                )
                response.raise_for_status()

                result = response_adapter.validate_python(response.json())
                logger.debug("Successfully found single row")

                return result

            except HTTPError as exc:
                if hasattr(exc, "response"):
                    if exc.response.status_code == 404:
                        error_msg = f"No {ctx.router_string} found matching filters"
                        logger.warning("No rows found", url=query_url)
                        raise ValueError(error_msg) from exc
                    if exc.response.status_code == 409:
                        error_msg = f"Multiple {ctx.router_string} found matching filters"
                        logger.warning("Multiple rows found", url=query_url)
                        raise ValueError(error_msg) from exc
                    if exc.response.status_code == 400:
                        error_msg = "Invalid filter request"
                        logger.warning("Bad request", url=query_url)
                        raise ValueError(error_msg) from exc

                logger.error("HTTP error finding single row", url=query_url, error=str(exc))
                raise
            except CoreValidationError as exc:
                logger.error("Validation error parsing response", url=query_url, error=str(exc))
                raise

        return client_method

    def create_remote_command(self, group: click.Group) -> Callable:
        ctx = self.ctx
        client_method = self.create_client_method()

        @group.command(
            name=f"filter-one-{ctx.name}", help=f"Find exactly one {ctx.router_string} matching criteria"
        )
        @common_options.pz_client()
        @common_options.output()
        @common_options.filters()
        @common_options.logical_op()
        @common_options.timeout()
        def command(
            client_object: ClientBase,
            output: common_options.OutputEnum | None,
            filter_strs: tuple[str, ...],
            logical_op: str,
            timeout: float,
        ) -> None:
            """Find exactly one row from remote API."""
            # Parse filters
            try:
                filters = []
                for filter_str in filter_strs:
                    filter_obj = parse_filter_from_string(filter_str)
                    filters.append(
                        {
                            "field": filter_obj.field,
                            "op": filter_obj.op.value,
                            "value": filter_obj.value,
                        }
                    )
            except ValueError as exc:
                click.echo(f"Error: {exc}", err=True)
                raise click.Abort()

            try:
                result = client_method(
                    client_object,
                    filters=filters,
                    logical_op=logical_op,
                    timeout=timeout,
                )

                output_pydantic_single(result, output, ctx.col_names_optional)
                click.echo(f"Found matching {ctx.router_string}", err=True)

            except Exception as exc:
                handle_cli_error(exc, "find", ctx.router_string)

        return command


# Export all filter operation classes
__all__ = [
    "FilterRowsOperation",
    "CountFilteredRowsOperation",
    "FindByOperation",
    "FilterOneOperation",
    "FilterModel",
    "OrderByModel",
    "FilterRequest",
    "FilterResponse",
    "CountFilteredResponse",
    "parse_filter_from_string",
]
