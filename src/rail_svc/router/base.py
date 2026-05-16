from typing import Any, AsyncGenerator, TypeVar, Generic
import logging
from functools import wraps

from fastapi import APIRouter, HTTPException, Query, Body, Depends, Header, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ValidationError

from ..local_asnyc import (
    LocalOperations,
)
from ..db_funcs.filter import Filter, OrderBy

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar("T")  # Database model type
ResponseT = TypeVar("ResponseT", bound=BaseModel)  # Response schema type
CreateT = TypeVar("CreateT", bound=BaseModel)  # Create schema type


class AsyncRouteError(Exception):
    """Custom exception for async route handling errors."""

    pass


class CountResponse(BaseModel):
    """Response model for count operations."""

    count: int


class LookupResponse(BaseModel, Generic[ResponseT]):
    """Response model for lookup operations."""

    id: int
    data: ResponseT


class DeleteResponse(BaseModel):
    """Response model for delete operations."""

    deleted: bool = True


class FilterRequest(BaseModel):
    """Request model for filter operations."""

    filters: list[Filter] = []
    logical_op: str = "and"
    order_by: OrderBy | list[OrderBy] | None = None
    skip: int = 0
    limit: int | None = None


class FindRequest(BaseModel):
    """Request model for find operations."""

    order_by: OrderBy | list[OrderBy] | None = None
    skip: int = 0
    limit: int | None = None

    class Config:
        extra = "allow"  # Allow additional fields for query params


def require_auth(authorization: str = Header(None)):
    """Dependency to require authentication.

    Parameters
    ----------
    authorization : str
        Authorization header value

    Raises
    ------
    HTTPException
        If authorization is invalid

    Returns
    -------
    str
        The validated token

    Example
    -------
    >>> @router.get("/protected")
    >>> async def protected_route(token: str = Depends(require_auth)):
    ...     return {"message": "authenticated"}
    """
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authorization header missing")

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid authorization header format"
        )

    token = authorization[7:]  # Remove 'Bearer ' prefix

    # TODO: Implement proper token validation
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    return token


def validate_pagination_params(skip: int, limit: int | None) -> tuple[int, int | None]:
    """Validate pagination parameters.

    Parameters
    ----------
    skip : int
        Number of rows to skip
    limit : int | None
        Maximum rows to return

    Returns
    -------
    tuple[int, int | None]
        Validated params

    Raises
    ------
    HTTPException
        If validation fails
    """
    if skip < 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="skip must be non-negative")

    if limit is not None:
        if limit < 1:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="limit must be positive")
        if limit > 10000:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="limit cannot exceed 10000")

    return skip, limit


def validate_batch_size(batch_size: int) -> int:
    """Validate batch size parameter.

    Parameters
    ----------
    batch_size : int
        Batch size to validate

    Returns
    -------
    int
        Validated batch size

    Raises
    ------
    HTTPException
        If validation fails
    """
    if not 1 <= batch_size <= 10000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="batch_size must be between 1 and 10000"
        )
    return batch_size


def create_table_router[T, ResponseT: BaseModel, CreateT: BaseModel](
    name: str,
    operations: LocalOperations[T, ResponseT, CreateT],
    id_param: str = "id",
) -> APIRouter:
    """Create a FastAPI router with CRUD endpoints for a table.

    Parameters
    ----------
    name : str
        Name of the table (used for router prefix and tags)
    operations : LocalOperations
        The local operations instance for this table
    id_param : str
        Name of the ID parameter in URLs (default: "id")

    Returns
    -------
    APIRouter
        FastAPI router with all CRUD endpoints
    """
    router = APIRouter(prefix=f"/{name}", tags=[name])

    # CREATE endpoints
    @router.post("/create_row", status_code=status.HTTP_201_CREATED, response_model=ResponseT)
    async def create_row(
        data: dict = Body(...), validate: bool = Query(True, description="Whether to validate data")
    ) -> ResponseT:
        """Create a single row.

        Request Body:
            JSON object with row data

        Query Parameters:
            validate (bool): Whether to validate data (default: true)

        Returns:
            201: Created row
            400: Validation error
            500: Internal server error
        """
        try:
            result = await operations.create_row(validate=validate, **data)
            return result
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Validation error", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error creating row")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/create_rows", status_code=status.HTTP_201_CREATED, response_model=list[ResponseT])
    async def create_rows(
        data: list[dict] = Body(...), validate: bool = Query(True, description="Whether to validate data")
    ) -> list[ResponseT]:
        """Create multiple rows.

        Request Body:
            JSON array of objects

        Query Parameters:
            validate (bool): Whether to validate data (default: true)

        Returns:
            201: Array of created rows
            400: Validation error or invalid input
            500: Internal server error
        """
        if len(data) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot create more than 10000 rows at once"
            )

        try:
            results = await operations.create_rows(data, validate=validate)
            return results
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Validation error", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error creating rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/create_rows_batched", status_code=status.HTTP_201_CREATED, response_model=list[ResponseT])
    async def create_rows_batched(
        data: list[dict] = Body(...),
        validate: bool = Query(True, description="Whether to validate data"),
        batch_size: int = Query(1000, ge=1, le=10000, description="Size of each batch"),
    ) -> list[ResponseT]:
        """Create multiple rows in batches.

        Request Body:
            JSON array of objects

        Query Parameters:
            validate (bool): Whether to validate data (default: true)
            batch_size (int): Size of each batch (default: 1000, max: 10000)

        Returns:
            201: Array of created rows
            400: Validation error or invalid input
            500: Internal server error
        """
        try:
            results = await operations.create_rows_batched(data, validate=validate, batch_size=batch_size)
            return results
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Validation error", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error creating rows batched")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/bulk_insert_rows", status_code=status.HTTP_201_CREATED, response_model=CountResponse)
    async def bulk_insert_rows(
        data: list[dict] = Body(...), validate: bool = Query(True, description="Whether to validate data")
    ) -> CountResponse:
        """Bulk insert rows (returns count only).

        Request Body:
            JSON array of objects

        Query Parameters:
            validate (bool): Whether to validate data (default: true)

        Returns:
            201: Object with count of inserted rows
            400: Validation error or invalid input
            500: Internal server error
        """
        if len(data) > 100000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot bulk insert more than 100000 rows at once",
            )

        try:
            count = await operations.bulk_insert_rows(data, validate=validate)
            return CountResponse(count=count)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Validation error", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error bulk inserting rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # READ endpoints
    @router.get(f"/get_row/{{{id_param}}}", response_model=ResponseT)
    async def get_row(**kwargs) -> ResponseT:
        """Get a single row by ID.

        Path Parameters:
            id (int): Row ID

        Returns:
            200: Row data
            404: Row not found
            500: Internal server error
        """
        try:
            row_id = kwargs[id_param]
            result = await operations.get_row(row_id)
            if result is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error getting row")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.get(f"/get_row_or_none/{{{id_param}}}", response_model=ResponseT | None)
    async def get_row_or_none(**kwargs) -> ResponseT | None:
        """Get a single row by ID or None if not found.

        Path Parameters:
            id (int): Row ID

        Returns:
            200: Row data or null
            500: Internal server error
        """
        try:
            row_id = kwargs[id_param]
            result = await operations.get_row_or_none(row_id)
            return result
        except Exception as e:
            logger.exception("Error getting row or none")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.get("/get_row_by_name/{name}", response_model=ResponseT)
    async def get_row_by_name(name: str) -> ResponseT:
        """Get a single row by name.

        Path Parameters:
            name (str): Row name

        Returns:
            200: Row data
            404: Row not found
            500: Internal server error
        """
        try:
            result = await operations.get_row_by_name(name)
            if result is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error getting row by name")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.get("/get_rows", response_model=list[ResponseT])
    async def get_rows(
        skip: int = Query(0, ge=0, description="Number of rows to skip"),
        limit: int | None = Query(None, ge=1, le=10000, description="Maximum rows to return"),
    ) -> list[ResponseT]:
        """Get multiple rows with pagination.

        Query Parameters:
            skip (int): Number of rows to skip (default: 0, min: 0)
            limit (int): Maximum rows to return (default: unlimited, max: 10000)

        Returns:
            200: Array of rows
            400: Invalid pagination parameters
            500: Internal server error
        """
        try:
            results = await operations.get_rows(skip=skip, limit=limit)
            return results
        except Exception as e:
            logger.exception("Error getting rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.get("/get_rows_streaming")
    async def get_rows_streaming(
        skip: int = Query(0, ge=0, description="Number of rows to skip"),
        limit: int | None = Query(None, ge=1, le=10000, description="Maximum rows to return"),
    ) -> StreamingResponse:
        """Get rows as a streaming response (NDJSON format).

        Query Parameters:
            skip (int): Number of rows to skip (default: 0, min: 0)
            limit (int): Maximum rows to return (default: unlimited, max: 10000)

        Returns:
            200: Stream of newline-delimited JSON objects
            400: Invalid pagination parameters
            500: Internal server error

        Note:
            Response format is NDJSON (newline-delimited JSON), not a JSON array.
            Each line is a complete JSON object representing one row.
        """

        async def generate():
            try:
                async for row in operations.get_rows_streaming(skip=skip, limit=limit):
                    yield row.model_dump_json() + "\n"
            except Exception as e:
                logger.exception("Error in streaming rows")
                # In streaming, we can't raise HTTP exceptions after starting
                yield f'{{"error": "{str(e)}"}}\n'

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @router.get("/count_rows", response_model=CountResponse)
    async def count_rows() -> CountResponse:
        """Get total count of rows.

        Returns:
            200: Object with count
            500: Internal server error
        """
        try:
            count = await operations.count_rows()
            return CountResponse(count=count)
        except Exception as e:
            logger.exception("Error counting rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.get("/lookup_by_id_or_name", response_model=LookupResponse[ResponseT])
    async def lookup_by_id_or_name(
        id: int | None = Query(None, description="Row ID"),
        name: str | None = Query(None, description="Row name"),
    ) -> LookupResponse[ResponseT]:
        """Lookup by ID or name.

        Query Parameters:
            id (int): Row ID (optional)
            name (str): Row name (optional)

        Note:
            At least one of id or name must be provided.

        Returns:
            200: Object with resolved ID and row data
            400: Neither id nor name provided
            404: Row not found
            500: Internal server error
        """
        if id is None and name is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Must provide either 'id' or 'name' parameter"
            )

        try:
            resolved_id, result = await operations.lookup_by_id_or_name(id, name)

            if result is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")

            return LookupResponse(id=resolved_id, data=result)
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error in lookup")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # UPDATE endpoints
    @router.put(f"/update_row/{{{id_param}}}", response_model=ResponseT)
    @router.patch(f"/update_row/{{{id_param}}}", response_model=ResponseT)
    async def update_row(data: dict = Body(...), **kwargs) -> ResponseT:
        """Update a single row.

        Path Parameters:
            id (int): Row ID

        Request Body:
            JSON object with fields to update

        Returns:
            200: Updated row
            400: Validation error
            404: Row not found
            500: Internal server error
        """
        try:
            row_id = kwargs[id_param]
            result = await operations.update_row(row_id, **data)
            if result is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
            return result
        except HTTPException:
            raise
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Validation error", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error updating row")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.put("/update_rows", response_model=list[ResponseT])
    @router.patch("/update_rows", response_model=list[ResponseT])
    async def update_rows(data: list[dict] = Body(...)) -> list[ResponseT]:
        """Update multiple rows.

        Request Body:
            JSON array of objects, each containing an 'id' field

        Returns:
            200: Array of updated rows
            400: Validation error or invalid input
            404: One or more rows not found
            500: Internal server error
        """
        if len(data) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot update more than 10000 rows at once"
            )

        # Validate that all items have an 'id' field
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Item at index {i} is not an object"
                )
            if "id" not in item:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Item at index {i} missing 'id' field"
                )

        try:
            results = await operations.update_rows(data)
            return results
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Validation error", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error updating rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # DELETE endpoints
    @router.delete(f"/delete_row/{{{id_param}}}", response_model=ResponseT | DeleteResponse)
    async def delete_row(
        capture_data: bool = Query(True, description="Whether to return deleted row data"), **kwargs
    ) -> ResponseT | DeleteResponse:
        """Delete a single row.

        Path Parameters:
            id (int): Row ID

        Query Parameters:
            capture_data (bool): Whether to return deleted row data (default: true)

        Returns:
            200: Deleted row data (if capture_data=true) or success message
            404: Row not found
            500: Internal server error
        """
        try:
            row_id = kwargs[id_param]
            result = await operations.delete_row(row_id, capture_data=capture_data)

            if result is None and capture_data:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")

            if result is None:
                return DeleteResponse()
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error deleting row")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.delete("/delete_rows", response_model=list[ResponseT] | CountResponse)
    async def delete_rows(
        data: list[int] = Body(...),
        capture_data: bool = Query(False, description="Whether to return deleted row data"),
    ) -> list[ResponseT] | CountResponse:
        """Delete multiple rows.

        Request Body:
            JSON array of row IDs

        Query Parameters:
            capture_data (bool): Whether to return deleted row data (default: false)

        Returns:
            200: Array of deleted rows (if capture_data=true) or count
            400: Invalid input
            500: Internal server error
        """
        if len(data) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Cannot delete more than 10000 rows at once"
            )

        # Validate all IDs are integers
        for i, item in enumerate(data):
            if not isinstance(item, int):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Item at index {i} is not an integer"
                )

        try:
            result = await operations.delete_rows(data, capture_data=capture_data)
            if capture_data:
                return result
            else:
                return CountResponse(count=len(data))
        except Exception as e:
            logger.exception("Error deleting rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.delete("/bulk_delete_rows", response_model=CountResponse)
    async def bulk_delete_rows(data: list[int] = Body(...)) -> CountResponse:
        """Bulk delete rows (returns count only).

        Request Body:
            JSON array of row IDs

        Returns:
            200: Object with count of deleted rows
            400: Invalid input
            500: Internal server error
        """
        if len(data) > 100000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot bulk delete more than 100000 rows at once",
            )

        # Validate all IDs are integers
        for i, item in enumerate(data):
            if not isinstance(item, int):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST, detail=f"Item at index {i} is not an integer"
                )

        try:
            count = await operations.bulk_delete_rows(data)
            return CountResponse(count=count)
        except Exception as e:
            logger.exception("Error bulk deleting rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    # FILTER/QUERY endpoints
    @router.post("/filter_rows", response_model=list[ResponseT])
    async def filter_rows(request: FilterRequest) -> list[ResponseT]:
        """Filter rows with complex criteria.

        Request Body:
            {
                "filters": [{"field": "name", "op": "eq", "value": "test"}],
                "logical_op": "and",  // "and" or "or"
                "order_by": {"field": "created_at", "direction": "desc"},  // or array
                "skip": 0,
                "limit": 100
            }

        Returns:
            200: Array of filtered rows
            400: Invalid filter syntax
            500: Internal server error
        """
        if request.logical_op not in ("and", "or"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="logical_op must be 'and' or 'or'"
            )

        # Validate pagination
        validate_pagination_params(request.skip, request.limit)

        try:
            results = await operations.filter_rows(
                filters=request.filters if request.filters else None,
                logical_op=request.logical_op,
                order_by=request.order_by,
                skip=request.skip,
                limit=request.limit,
            )
            return results
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid filter syntax", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error filtering rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/filter_rows_streaming")
    async def filter_rows_streaming(request: FilterRequest) -> StreamingResponse:
        """Filter rows with streaming response (NDJSON format).

        Request Body:
            Same as /filter endpoint

        Returns:
            200: Stream of newline-delimited JSON objects
            400: Invalid filter syntax
            500: Internal server error

        Note:
            Response format is NDJSON (newline-delimited JSON), not a JSON array.
        """
        if request.logical_op not in ("and", "or"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="logical_op must be 'and' or 'or'"
            )

        # Validate pagination
        validate_pagination_params(request.skip, request.limit)

        async def generate():
            try:
                async for row in operations.filter_rows_streaming(
                    filters=request.filters if request.filters else None,
                    logical_op=request.logical_op,
                    order_by=request.order_by,
                    skip=request.skip,
                    limit=request.limit,
                ):
                    yield row.model_dump_json() + "\n"
            except Exception as e:
                logger.exception("Error in streaming filtered rows")
                yield f'{{"error": "{str(e)}"}}\n'

        return StreamingResponse(generate(), media_type="application/x-ndjson")

    @router.post("/count_filtered_rows", response_model=CountResponse)
    async def count_filtered_rows(request: FilterRequest) -> CountResponse:
        """Count filtered rows.

        Request Body:
            {
                "filters": [{"field": "name", "op": "eq", "value": "test"}],
                "logical_op": "and"
            }

        Returns:
            200: Object with count
            400: Invalid filter syntax
            500: Internal server error
        """
        if request.logical_op not in ("and", "or"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="logical_op must be 'and' or 'or'"
            )

        try:
            count = await operations.count_filtered_rows(
                filters=request.filters if request.filters else None,
                logical_op=request.logical_op,
            )
            return CountResponse(count=count)
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid filter syntax", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error counting filtered rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/filter_one", response_model=ResponseT)
    async def filter_one(request: FilterRequest) -> ResponseT:
        """Filter to get exactly one row.

        Request Body:
            {
                "filters": [{"field": "name", "op": "eq", "value": "test"}],
                "logical_op": "and"
            }

        Returns:
            200: Single row
            400: Invalid filter syntax or filter returned != 1 row
            404: No rows matched filter
            500: Internal server error
        """
        if not request.filters:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'filters' array is required")

        if request.logical_op not in ("and", "or"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="logical_op must be 'and' or 'or'"
            )

        try:
            result = await operations.filter_one(filters=request.filters, logical_op=request.logical_op)
            if result is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
            return result
        except HTTPException:
            raise
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid filter syntax", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error filtering one row")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/filter_one_or_none", response_model=ResponseT | None)
    async def filter_one_or_none(request: FilterRequest) -> ResponseT | None:
        """Filter to get one row or None.

        Request Body:
            {
                "filters": [{"field": "name", "op": "eq", "value": "test"}],
                "logical_op": "and"
            }

        Returns:
            200: Single row or null
            400: Invalid filter syntax or filter returned > 1 row
            500: Internal server error
        """
        if not request.filters:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="'filters' array is required")

        if request.logical_op not in ("and", "or"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="logical_op must be 'and' or 'or'"
            )

        try:
            result = await operations.filter_one_or_none(
                filters=request.filters, logical_op=request.logical_op
            )
            return result
        except ValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid filter syntax", "details": e.errors()},
            )
        except Exception as e:
            logger.exception("Error filtering one or none row")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/find_by", response_model=list[ResponseT])
    async def find_by(request: dict = Body(...)) -> list[ResponseT]:
        """Find rows by field values.

        Request Body:
            {
                "field1": "value1",
                "field2": "value2",
                "order_by": {"field": "created_at", "direction": "desc"},
                "skip": 0,
                "limit": 100
            }

        Note:
            All fields except order_by, skip, and limit are treated as equality filters.

        Returns:
            200: Array of matching rows
            400: Invalid input
            500: Internal server error
        """
        # Extract special parameters
        order_by_data = request.pop("order_by", None)
        skip = request.pop("skip", 0)
        limit = request.pop("limit", None)

        # Remaining fields are query parameters
        query_params = request

        if not query_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="At least one query field is required"
            )

        # Validate pagination
        validate_pagination_params(skip, limit)

        # Parse order_by if provided
        order_by = None
        if order_by_data:
            try:
                if isinstance(order_by_data, list):
                    order_by = [OrderBy(**o) for o in order_by_data]
                else:
                    order_by = OrderBy(**order_by_data)
            except (TypeError, ValidationError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={"error": "Invalid order_by syntax", "details": str(e)},
                )

        try:
            results = await operations.find_by(
                order_by=order_by,
                skip=skip,
                limit=limit,
                **query_params,
            )
            return results
        except Exception as e:
            logger.exception("Error finding rows")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    @router.post("/find_one_by", response_model=ResponseT)
    async def find_one_by(data: dict = Body(...)) -> ResponseT:
        """Find exactly one row by field values.

        Request Body:
            {
                "field1": "value1",
                "field2": "value2"
            }

        Returns:
            200: Single matching row
            400: Invalid input or query returned != 1 row
            404: No rows matched
            500: Internal server error
        """
        if not data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="At least one query field is required"
            )

        try:
            result = await operations.find_one_by(**data)
            if result is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Resource not found")
            return result
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("Error finding one row")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return router
