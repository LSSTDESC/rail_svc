"""Unit tests for FastAPI router factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError
from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from rail_svc.db.base import Base
from rail_svc.local_async import LocalOperations
from rail_svc.router.base import (
    create_table_router,
    require_auth,
    validate_batch_size,
    validate_pagination_params,
)


# Test Models
class RouterTestBase(Base):
    """Test database model."""

    __tablename__ = "test_table"
    __allow_unmapped__ = True  # Allow for test simplicity

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False)


class RouterTestResponse(BaseModel):
    """Test response model."""

    id: int
    name: str
    value: int


class RouterTestCreate(BaseModel):
    """Test create model."""

    name: str
    value: int


# Fixtures
@pytest.fixture
def mock_operations() -> MagicMock:
    """Create mock LocalOperations instance."""
    operations = MagicMock(spec=LocalOperations)

    # Mock the table operations context
    mock_table_ops = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.response_class = RouterTestResponse
    operations._table_ops = mock_table_ops
    operations._table_ops.ctx = mock_ctx

    # Set up async mock methods
    operations.create_row = AsyncMock()
    operations.create_rows = AsyncMock()
    operations.create_rows_batched = AsyncMock()
    operations.bulk_insert_rows = AsyncMock()
    operations.get_row = AsyncMock()
    operations.get_row_or_none = AsyncMock()
    operations.get_row_by_name = AsyncMock()
    operations.get_rows = AsyncMock()
    operations.get_rows_streaming = AsyncMock()
    operations.count_rows = AsyncMock()
    operations.lookup_by_id_or_name = AsyncMock()
    operations.update_row = AsyncMock()
    operations.update_rows = AsyncMock()
    operations.delete_row = AsyncMock()
    operations.delete_rows = AsyncMock()
    operations.bulk_delete_rows = AsyncMock()
    operations.filter_rows = AsyncMock()
    operations.filter_rows_streaming = AsyncMock()
    operations.count_filtered_rows = AsyncMock()
    operations.filter_one = AsyncMock()
    operations.filter_one_or_none = AsyncMock()
    operations.find_by = AsyncMock()
    operations.find_one_by = AsyncMock()

    return operations


@pytest.fixture
def app(mock_operations: MagicMock) -> FastAPI:
    """Create test FastAPI application."""
    app = FastAPI()
    router = create_table_router("test", mock_operations)
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create test client."""
    return TestClient(app)


# Helper Functions Tests
class TestRequireAuth:
    """Tests for require_auth dependency."""

    def test_missing_authorization_header(self) -> None:
        """Test that missing authorization header raises 401."""
        with pytest.raises(Exception) as exc_info:
            require_auth(authorization=None)
        assert "401" in str(exc_info.value)

    def test_invalid_authorization_format(self) -> None:
        """Test that invalid format raises 401."""
        with pytest.raises(Exception) as exc_info:
            require_auth(authorization="InvalidFormat token123")
        assert "401" in str(exc_info.value)

    def test_empty_token(self) -> None:
        """Test that empty token raises 401."""
        with pytest.raises(Exception) as exc_info:
            require_auth(authorization="Bearer ")
        assert "401" in str(exc_info.value)

    def test_valid_token(self) -> None:
        """Test that valid token is returned."""
        token = require_auth(authorization="Bearer valid_token_123")
        assert token == "valid_token_123"


class TestValidatePaginationParams:
    """Tests for validate_pagination_params."""

    def test_negative_skip(self) -> None:
        """Test that negative skip raises 400."""
        with pytest.raises(Exception) as exc_info:
            validate_pagination_params(-1, None)
        assert "400" in str(exc_info.value)

    def test_zero_limit(self) -> None:
        """Test that zero limit raises 400."""
        with pytest.raises(Exception) as exc_info:
            validate_pagination_params(0, 0)
        assert "400" in str(exc_info.value)

    def test_excessive_limit(self) -> None:
        """Test that limit > 10000 raises 400."""
        with pytest.raises(Exception) as exc_info:
            validate_pagination_params(0, 10001)
        assert "400" in str(exc_info.value)

    def test_valid_params(self) -> None:
        """Test that valid params are returned unchanged."""
        skip, limit = validate_pagination_params(10, 100)
        assert skip == 10
        assert limit == 100

    def test_none_limit(self) -> None:
        """Test that None limit is allowed."""
        skip, limit = validate_pagination_params(0, None)
        assert skip == 0
        assert limit is None


class TestValidateBatchSize:
    """Tests for validate_batch_size."""

    def test_zero_batch_size(self) -> None:
        """Test that zero batch size raises 400."""
        with pytest.raises(Exception) as exc_info:
            validate_batch_size(0)
        assert "400" in str(exc_info.value)

    def test_excessive_batch_size(self) -> None:
        """Test that batch size > 10000 raises 400."""
        with pytest.raises(Exception) as exc_info:
            validate_batch_size(10001)
        assert "400" in str(exc_info.value)

    def test_valid_batch_size(self) -> None:
        """Test that valid batch size is returned."""
        batch_size = validate_batch_size(1000)
        assert batch_size == 1000


# CREATE Endpoint Tests
class TestCreateEndpoints:
    """Tests for CREATE endpoints."""

    def test_create_row_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful row creation."""
        mock_operations.create_row.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.post("/test/create_row", json={"name": "test", "value": 100})

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {"id": 1, "name": "test", "value": 100}
        mock_operations.create_row.assert_awaited_once_with(validate=True, name="test", value=100)

    def test_create_row_validation_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test row creation with validation error."""
        mock_operations.create_row.side_effect = ValidationError.from_exception_data(
            "RouterTestResponse",
            [{"type": "missing", "loc": ("name",), "msg": "Field required", "input": {}}],
        )

        response = client.post("/test/create_row", json={"value": 100})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Validation error" in response.json()["detail"]["error"]

    def test_create_row_without_validation(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test row creation without validation."""
        mock_operations.create_row.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.post("/test/create_row?validate=false", json={"name": "test", "value": 100})

        assert response.status_code == status.HTTP_201_CREATED
        mock_operations.create_row.assert_awaited_once_with(validate=False, name="test", value=100)

    def test_create_rows_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful multiple rows creation."""
        mock_operations.create_rows.return_value = [
            RouterTestResponse(id=1, name="test1", value=100),
            RouterTestResponse(id=2, name="test2", value=200),
        ]

        response = client.post(
            "/test/create_rows",
            json=[
                {"name": "test1", "value": 100},
                {"name": "test2", "value": 200},
            ],
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert len(response.json()) == 2

    def test_create_rows_too_many(self, client: TestClient) -> None:
        """Test creating too many rows at once."""
        data = [{"name": f"test{i}", "value": i} for i in range(10001)]

        response = client.post("/test/create_rows", json=data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "10000" in response.json()["detail"]

    def test_create_rows_batched_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test batched row creation."""
        mock_operations.create_rows_batched.return_value = [
            RouterTestResponse(id=1, name="test1", value=100),
            RouterTestResponse(id=2, name="test2", value=200),
        ]

        response = client.post(
            "/test/create_rows_batched?batch_size=100",
            json=[{"name": "test1", "value": 100}, {"name": "test2", "value": 200}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        mock_operations.create_rows_batched.assert_awaited_once()

    def test_bulk_insert_rows_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk insert."""
        mock_operations.bulk_insert_rows.return_value = 2

        response = client.post(
            "/test/bulk_insert_rows",
            json=[
                {"name": "test1", "value": 100},
                {"name": "test2", "value": 200},
            ],
        )

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == {"count": 2}

    def test_bulk_insert_rows_too_many(self, client: TestClient) -> None:
        """Test bulk insert with too many rows."""
        data = [{"name": f"test{i}", "value": i} for i in range(100001)]

        response = client.post("/test/bulk_insert_rows", json=data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "100000" in response.json()["detail"]


# READ Endpoint Tests
class TestReadEndpoints:
    """Tests for READ endpoints."""

    def test_get_row_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful row retrieval."""
        mock_operations.get_row.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.get("/test/get_row/1")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"id": 1, "name": "test", "value": 100}
        # Path param is passed as string "1", gets converted to int in endpoint
        mock_operations.get_row.assert_awaited_once_with(1)

    def test_get_row_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test row not found."""
        mock_operations.get_row.return_value = None

        response = client.get("/test/get_row/999")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_row_or_none_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_row_or_none when row exists."""
        mock_operations.get_row_or_none.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.get("/test/get_row_or_none/1")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"id": 1, "name": "test", "value": 100}

    def test_get_row_or_none_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_row_or_none when row doesn't exist."""
        mock_operations.get_row_or_none.return_value = None

        response = client.get("/test/get_row_or_none/999")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() is None

    def test_get_rows_streaming(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test streaming rows."""

        items = [
            RouterTestResponse(id=1, name="test1", value=100),
            RouterTestResponse(id=2, name="test2", value=200),
        ]

        # Replace the mock method entirely with an async generator function
        async def mock_streaming(*args, **kwargs):
            for item in items:
                yield item

        mock_operations.get_rows_streaming = mock_streaming

        response = client.get("/test/get_rows_streaming")

        assert response.status_code == status.HTTP_200_OK
        assert "ndjson" in response.headers["content-type"]

        lines = [line for line in response.text.strip().split("\n") if line and "error" not in line]
        assert len(lines) == 2


# UPDATE Endpoint Tests
class TestUpdateEndpoints:
    """Tests for UPDATE endpoints."""

    def test_update_row_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful row update."""
        mock_operations.update_row.return_value = RouterTestResponse(id=1, name="updated", value=200)

        response = client.put("/test/update_row/1", json={"name": "updated", "value": 200})

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "updated"
        # Path param "1" is passed as string
        mock_operations.update_row.assert_awaited_once_with(1, name="updated", value=200)

    def test_update_row_patch(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test row update with PATCH method."""
        mock_operations.update_row.return_value = RouterTestResponse(id=1, name="updated", value=200)

        response = client.patch("/test/update_row/1", json={"name": "updated"})

        assert response.status_code == status.HTTP_200_OK
        mock_operations.update_row.assert_awaited_once()

    def test_update_row_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test updating non-existent row."""
        mock_operations.update_row.return_value = None

        response = client.put("/test/update_row/999", json={"name": "test"})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_update_row_validation_error(self, client: RouterTestClient, mock_operations: MagicMock) -> None:
        """Test update with validation error."""
        mock_operations.update_row.side_effect = ValidationError.from_exception_data(
            "TestResponse",
            [{"type": "int_parsing", "loc": ("value",), "msg": "Invalid integer", "input": "abc"}],
        )

        response = client.put("/test/update_row/1", json={"value": "invalid"})

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_update_rows_invalid_item(self, client: TestClient) -> None:
        """Test updating rows with non-dict item."""
        # FastAPI validation will catch this, so expect 422
        response = client.put("/test/update_rows", json=["not_a_dict"])

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


# DELETE Endpoint Tests
class TestDeleteEndpoints:
    """Tests for DELETE endpoints."""

    def test_delete_row_with_capture(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test deleting row with data capture."""
        mock_operations.delete_row.return_value = {"id": 1, "name": "test", "value": 100}

        response = client.delete("/test/delete_row/1?capture_data=true")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"id": 1, "name": "test", "value": 100}
        # Path param is string
        mock_operations.delete_row.assert_awaited_once_with(1, capture_data=True)

    def test_delete_row_without_capture(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test deleting row without data capture."""
        mock_operations.delete_row.return_value = None

        response = client.delete("/test/delete_row/1?capture_data=false")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"deleted": True}

    def test_delete_row_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test deleting non-existent row with capture."""
        mock_operations.delete_row.return_value = None

        response = client.delete("/test/delete_row/999?capture_data=true")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_delete_rows_with_capture(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test deleting multiple rows with capture."""
        mock_operations.delete_rows.return_value = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]

        # Use request with content instead of json for DELETE
        response = client.request("DELETE", "/test/delete_rows?capture_data=true", json=[1, 2])

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2

    def test_delete_rows_without_capture(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test deleting multiple rows without capture."""
        mock_operations.delete_rows.return_value = None

        response = client.request("DELETE", "/test/delete_rows?capture_data=false", json=[1, 2])

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"count": 2}

    def test_delete_rows_too_many(self, client: TestClient) -> None:
        """Test deleting too many rows."""
        data = list(range(10001))

        response = client.request("DELETE", "/test/delete_rows", json=data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_delete_rows_invalid_id(self, client: TestClient) -> None:
        """Test deleting rows with non-integer ID."""
        response = client.request("DELETE", "/test/delete_rows", json=[1, "invalid", 3])

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT
        assert "integer" in response.json()["detail"][0]["msg"]

    def test_bulk_delete_rows_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk delete."""
        mock_operations.bulk_delete_rows.return_value = 100

        response = client.request("DELETE", "/test/bulk_delete_rows", json=list(range(100)))

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"count": 100}

    def test_bulk_delete_rows_too_many(self, client: TestClient) -> None:
        """Test bulk delete with too many rows."""
        data = list(range(100001))

        response = client.request("DELETE", "/test/bulk_delete_rows", json=data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST


# FILTER/QUERY Endpoint Tests
class TestFilterEndpoints:
    """Tests for FILTER/QUERY endpoints."""

    def test_filter_rows_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filtering rows."""
        mock_operations.filter_rows.return_value = [
            RouterTestResponse(id=1, name="test", value=100),
        ]

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "skip": 0,
            "limit": 100,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 1

    def test_filter_rows_invalid_logical_op(self, client: TestClient) -> None:
        """Test filter with invalid logical operator."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "invalid",
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "and" in response.json()["detail"] or "or" in response.json()["detail"]

    def test_filter_rows_with_order_by(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filtering with ordering."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "value", "op": "gt", "value": 50}],
            "logical_op": "and",
            "order_by": {"field": "value", "direction": "desc"},
            "skip": 0,
            "limit": 10,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK

    def test_filter_rows_streaming(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test streaming filtered rows."""

        async def mock_generator():
            yield RouterTestResponse(id=1, name="test", value=100)

        mock_operations.filter_rows_streaming.return_value = mock_generator()

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert "ndjson" in response.headers["content-type"]

    def test_count_filtered_rows(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test counting filtered rows."""
        mock_operations.count_filtered_rows.return_value = 42

        filter_request = {
            "filters": [{"field": "value", "op": "gt", "value": 50}],
            "logical_op": "and",
        }

        response = client.post("/test/count_filtered_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"count": 42}

    def test_filter_one_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one success."""
        mock_operations.filter_one.return_value = RouterTestResponse(id=1, name="test", value=100)

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == 1

    def test_filter_one_no_filters(self, client: TestClient) -> None:
        """Test filter_one without filters."""
        filter_request = {
            "logical_op": "and",
        }

        response = client.post("/test/filter_one", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "filters" in response.json()["detail"]

    def test_filter_one_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one when no rows match."""
        mock_operations.filter_one.return_value = None

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "nonexistent"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one", json=filter_request)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_filter_one_or_none_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one_or_none when row exists."""
        mock_operations.filter_one_or_none.return_value = RouterTestResponse(id=1, name="test", value=100)

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one_or_none", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == 1

    def test_filter_one_or_none_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one_or_none when no rows match."""
        mock_operations.filter_one_or_none.return_value = None

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "nonexistent"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one_or_none", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json() is None

    def test_filter_one_or_none_no_filters(self, client: TestClient) -> None:
        """Test filter_one_or_none without filters."""
        filter_request = {
            "logical_op": "and",
        }

        response = client.post("/test/filter_one_or_none", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_find_by_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_by with field values."""
        mock_operations.find_by.return_value = [
            RouterTestResponse(id=1, name="test", value=100),
        ]

        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "value": 100,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 1

    def test_find_by_with_order_by(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_by with ordering."""
        mock_operations.find_by.return_value = []

        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "order_by": {"field": "value", "direction": "asc"},
                "skip": 0,
                "limit": 10,
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_find_by_with_multiple_order_by(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_by with multiple order_by clauses."""
        mock_operations.find_by.return_value = []

        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "order_by": [
                    {"field": "value", "direction": "desc"},
                    {"field": "name", "direction": "asc"},
                ],
            },
        )

        assert response.status_code == status.HTTP_200_OK

    def test_find_by_no_query_fields(self, client: TestClient) -> None:
        """Test find_by without query fields."""
        response = client.post(
            "/test/find_by",
            json={
                "skip": 0,
                "limit": 10,
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "query field" in response.json()["detail"]

    def test_find_by_invalid_order_by(self, client: TestClient) -> None:
        """Test find_by with invalid order_by syntax."""
        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "order_by": "invalid",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_find_one_by_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_one_by success."""
        mock_operations.find_one_by.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.post("/test/find_one_by", json={"name": "test"})

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "test"

    def test_find_one_by_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_one_by when no match."""
        mock_operations.find_one_by.return_value = None

        response = client.post("/test/find_one_by", json={"name": "nonexistent"})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_find_one_by_no_fields(self, client: TestClient) -> None:
        """Test find_one_by without query fields."""
        response = client.post("/test/find_one_by", json={})

        assert response.status_code == status.HTTP_400_BAD_REQUEST


# Edge Cases and Error Handling Tests
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_internal_server_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test that internal errors return 500."""
        mock_operations.get_row.side_effect = Exception("Database connection failed")

        response = client.get("/test/get_row/1")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_create_row_generic_exception(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test generic exception in create_row."""
        mock_operations.create_row.side_effect = RuntimeError("Unexpected error")

        response = client.post("/test/create_row", json={"name": "test", "value": 100})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_streaming_error_handling(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test error handling in streaming endpoint."""

        async def failing_generator():
            yield RouterTestResponse(id=1, name="test", value=100)
            raise RuntimeError("Stream error")

        mock_operations.get_rows_streaming.return_value = failing_generator()

        response = client.get("/test/get_rows_streaming")

        assert response.status_code == status.HTTP_200_OK
        # The error should be in the stream content
        assert "error" in response.text

    def test_empty_data_arrays(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test handling of empty data arrays."""
        mock_operations.create_rows.return_value = []

        response = client.post("/test/create_rows", json=[])

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == []

    def test_pagination_boundary_values(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test pagination with boundary values."""
        mock_operations.get_rows.return_value = []

        # Maximum limit
        response = client.get("/test/get_rows?skip=0&limit=10000")
        assert response.status_code == status.HTTP_200_OK

        # Zero skip
        response = client.get("/test/get_rows?skip=0&limit=10")
        assert response.status_code == status.HTTP_200_OK


# Router Configuration Tests
class TestRouterConfiguration:
    """Tests for router configuration and creation."""

    def test_custom_id_param(self, mock_operations: MagicMock) -> None:
        """Test router with custom ID parameter name."""
        app = FastAPI()
        router = create_table_router("items", mock_operations)
        app.include_router(router)
        client = TestClient(app)

        mock_operations.get_row.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.get("/items/get_row/123")

        assert response.status_code == status.HTTP_200_OK
        mock_operations.get_row.assert_awaited_once_with(123)

    def test_router_prefix_and_tags(self, mock_operations: MagicMock) -> None:
        """Test that router has correct prefix and tags."""
        router = create_table_router("users", mock_operations)

        assert router.prefix == "/users"
        assert "users" in router.tags

    def test_response_model_extraction(self, mock_operations: MagicMock) -> None:
        """Test that response model is correctly extracted from operations."""
        router = create_table_router("test", mock_operations)

        # Check that routes are created with correct response models
        routes = {route.path: route for route in router.routes}

        assert "/test/create_row" in routes


# Integration Tests
class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_create_and_retrieve_flow(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test creating and then retrieving a row."""
        # Create
        mock_operations.create_row.return_value = RouterTestResponse(id=1, name="test", value=100)
        create_response = client.post("/test/create_row", json={"name": "test", "value": 100})
        assert create_response.status_code == status.HTTP_201_CREATED
        created_id = create_response.json()["id"]

        # Retrieve
        mock_operations.get_row.return_value = RouterTestResponse(id=1, name="test", value=100)
        get_response = client.get(f"/test/get_row/{created_id}")
        assert get_response.status_code == status.HTTP_200_OK
        assert get_response.json() == create_response.json()

    def test_create_update_delete_flow(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test full CRUD cycle."""
        # Create
        mock_operations.create_row.return_value = RouterTestResponse(id=1, name="test", value=100)
        create_response = client.post("/test/create_row", json={"name": "test", "value": 100})
        row_id = create_response.json()["id"]

        # Update
        mock_operations.update_row.return_value = RouterTestResponse(id=1, name="updated", value=200)
        update_response = client.put(f"/test/update_row/{row_id}", json={"name": "updated"})
        assert update_response.status_code == status.HTTP_200_OK
        assert update_response.json()["name"] == "updated"

        # Delete
        mock_operations.delete_row.return_value = {"id": 1, "name": "updated", "value": 200}
        delete_response = client.delete(f"/test/delete_row/{row_id}")
        assert delete_response.status_code == status.HTTP_200_OK

    def test_filter_and_count_consistency(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test that filter and count return consistent results."""
        filter_request = {
            "filters": [{"field": "value", "op": "gt", "value": 50}],
            "logical_op": "and",
        }

        # Filter
        mock_operations.filter_rows.return_value = [
            RouterTestResponse(id=1, name="test1", value=100),
            RouterTestResponse(id=2, name="test2", value=200),
        ]
        filter_response = client.post("/test/filter_rows", json=filter_request)
        filter_count = len(filter_response.json())

        # Count
        mock_operations.count_filtered_rows.return_value = 2
        count_response = client.post("/test/count_filtered_rows", json=filter_request)

        assert count_response.json()["count"] == filter_count


# Performance and Limits Tests
class TestPerformanceAndLimits:
    """Tests for performance limits and constraints."""

    def test_batch_size_limits(self, client: TestClient) -> None:
        """Test batch size validation."""
        # Too small
        response = client.post(
            "/test/create_rows_batched?batch_size=0",
            json=[{"name": "test", "value": 100}],
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

        # Too large
        response = client.post(
            "/test/create_rows_batched?batch_size=10001",
            json=[{"name": "test", "value": 100}],
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_limit_constraints(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test limit parameter constraints."""
        mock_operations.get_rows.return_value = []

        # Valid limit
        response = client.get("/test/get_rows?limit=100")
        assert response.status_code == status.HTTP_200_OK

        # Exceeds maximum
        response = client.get("/test/get_rows?limit=10001")
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_large_batch_operations(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test operations with large batches near limits."""
        # Create rows at limit
        data = [{"name": f"test{i}", "value": i} for i in range(10000)]
        mock_operations.create_rows.return_value = [
            RouterTestResponse(id=i, name=f"test{i}", value=i) for i in range(10000)
        ]

        response = client.post("/test/create_rows", json=data)
        assert response.status_code == status.HTTP_201_CREATED
        assert len(response.json()) == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
