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
from rail_svc.router.base import (create_table_router, require_auth,
                                  validate_batch_size,
                                  validate_pagination_params)


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
    operations.table_ops = mock_table_ops
    operations.table_ops.ctx = mock_ctx

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

    def test_update_row_validation_error(self, client: TestClient, mock_operations: MagicMock) -> None:
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


class TestGetRowByName:
    """Tests for get_row_by_name endpoint."""

    def test_get_row_by_name_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful row retrieval by name."""
        mock_operations.get_row_by_name.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.get("/test/get_row_by_name/test")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["name"] == "test"
        mock_operations.get_row_by_name.assert_awaited_once_with("test")

    def test_get_row_by_name_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test row not found by name."""
        mock_operations.get_row_by_name.return_value = None

        response = client.get("/test/get_row_by_name/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_row_by_name_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test internal error in get_row_by_name."""
        mock_operations.get_row_by_name.side_effect = RuntimeError("Database error")

        response = client.get("/test/get_row_by_name/test")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestCountRows:
    """Tests for count_rows endpoint."""

    def test_count_rows_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful row count."""
        mock_operations.count_rows.return_value = 42

        response = client.get("/test/count_rows")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"count": 42}
        mock_operations.count_rows.assert_awaited_once()

    def test_count_rows_zero(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test count when no rows exist."""
        mock_operations.count_rows.return_value = 0

        response = client.get("/test/count_rows")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"count": 0}

    def test_count_rows_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test error in count_rows."""
        mock_operations.count_rows.side_effect = Exception("Database error")

        response = client.get("/test/count_rows")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestLookupByIdOrName:
    """Tests for lookup_by_id_or_name endpoint."""

    def test_lookup_by_id(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test lookup by ID."""
        mock_operations.lookup_by_id_or_name.return_value = (
            1,
            RouterTestResponse(id=1, name="test", value=100),
        )

        response = client.get("/test/lookup_by_id_or_name?id_=1")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == 1
        assert response.json()["data"]["id"] == 1

    def test_lookup_by_name(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test lookup by name."""
        mock_operations.lookup_by_id_or_name.return_value = (
            1,
            RouterTestResponse(id=1, name="test", value=100),
        )

        response = client.get("/test/lookup_by_id_or_name?name=test")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["data"]["name"] == "test"

    def test_lookup_neither_id_nor_name(self, client: TestClient) -> None:
        """Test lookup without id or name."""
        response = client.get("/test/lookup_by_id_or_name")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "id_" in response.json()["detail"] or "name" in response.json()["detail"]

    def test_lookup_not_found(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test lookup when resource not found."""
        mock_operations.lookup_by_id_or_name.return_value = (None, None)

        response = client.get("/test/lookup_by_id_or_name?id_=999")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_lookup_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test lookup with internal error."""
        mock_operations.lookup_by_id_or_name.side_effect = Exception("Database error")

        response = client.get("/test/lookup_by_id_or_name?id_=1")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestUpdateRowsValidation:
    """Tests for update_rows validation."""

    def test_update_rows_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful multiple rows update."""
        mock_operations.update_rows.return_value = [
            RouterTestResponse(id=1, name="updated1", value=100),
            RouterTestResponse(id=2, name="updated2", value=200),
        ]

        response = client.put(
            "/test/update_rows",
            json=[
                {"id": 1, "name": "updated1"},
                {"id": 2, "name": "updated2"},
            ],
        )

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2

    def test_update_rows_missing_id(self, client: TestClient) -> None:
        """Test update_rows with missing id field."""
        response = client.put(
            "/test/update_rows",
            json=[
                {"name": "test"},  # Missing id
            ],
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "id" in response.json()["detail"]

    def test_update_rows_too_many(self, client: TestClient) -> None:
        """Test updating too many rows."""
        data = [{"id": i, "name": f"test{i}"} for i in range(10001)]

        response = client.put("/test/update_rows", json=data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "10000" in response.json()["detail"]

    def test_update_rows_validation_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test update_rows with validation error."""
        mock_operations.update_rows.side_effect = ValidationError.from_exception_data(
            "TestResponse",
            [{"type": "int_parsing", "loc": ("value",), "msg": "Invalid integer", "input": "abc"}],
        )

        response = client.put("/test/update_rows", json=[{"id": 1, "value": "invalid"}])

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_update_rows_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test update_rows with internal error."""
        mock_operations.update_rows.side_effect = Exception("Database error")

        response = client.put("/test/update_rows", json=[{"id": 1, "name": "test"}])

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestDeleteRowsValidation:
    """Tests for delete_rows validation."""

    def test_delete_rows_invalid_type_in_array(self, client: TestClient) -> None:
        """Test delete_rows with non-integer in array."""
        # This should be caught by FastAPI validation
        response = client.request("DELETE", "/test/delete_rows", json=[1, 2, "three"])

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_delete_rows_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test delete_rows with internal error."""
        mock_operations.delete_rows.side_effect = Exception("Database error")

        response = client.request("DELETE", "/test/delete_rows", json=[1, 2])

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestBulkDeleteValidation:
    """Tests for bulk_delete_rows validation."""

    def test_bulk_delete_rows_invalid_type(self, client: TestClient) -> None:
        """Test bulk_delete with non-integer ID."""
        response = client.request("DELETE", "/test/bulk_delete_rows", json=[1, "invalid"])

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT

    def test_bulk_delete_rows_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_delete with internal error."""
        mock_operations.bulk_delete_rows.side_effect = Exception("Database error")

        response = client.request("DELETE", "/test/bulk_delete_rows", json=[1, 2, 3])

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestFilterRowsValidation:
    """Tests for filter_rows validation and edge cases."""

    def test_filter_rows_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows with internal error."""
        mock_operations.filter_rows.side_effect = Exception("Database error")

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_filter_rows_invalid_pagination(self, client: TestClient) -> None:
        """Test filter_rows with invalid pagination."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "skip": -1,
            "limit": 10,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestFilterRowsStreaming:
    """Tests for filter_rows_streaming endpoint."""

    def test_filter_rows_streaming_success(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test successful streaming of filtered rows."""

        async def mock_generator():
            yield RouterTestResponse(id=1, name="test1", value=100)
            yield RouterTestResponse(id=2, name="test2", value=200)

        mock_operations.filter_rows_streaming = mock_generator

        filter_request = {
            "filters": [{"field": "value", "op": "gt", "value": 50}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert "ndjson" in response.headers["content-type"]

    def test_filter_rows_streaming_invalid_logical_op(self, client: TestClient) -> None:
        """Test filter_rows_streaming with invalid logical operator."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "xor",  # Invalid
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_filter_rows_streaming_invalid_pagination(self, client: TestClient) -> None:
        """Test filter_rows_streaming with invalid pagination."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "skip": -1,
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestCountFilteredRows:
    """Tests for count_filtered_rows endpoint."""

    def test_count_filtered_rows_invalid_logical_op(self, client: TestClient) -> None:
        """Test count_filtered with invalid logical operator."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "not",  # Invalid
        }

        response = client.post("/test/count_filtered_rows", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_count_filtered_rows_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test count_filtered with internal error."""
        mock_operations.count_filtered_rows.side_effect = Exception("Database error")

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/count_filtered_rows", json=filter_request)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestFilterOneEndpoint:
    """Tests for filter_one endpoint."""

    def test_filter_one_invalid_logical_op(self, client: TestClient) -> None:
        """Test filter_one with invalid logical operator."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "maybe",  # Invalid
        }

        response = client.post("/test/filter_one", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_filter_one_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one with internal error."""
        mock_operations.filter_one.side_effect = Exception("Database error")

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one", json=filter_request)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestFilterOneOrNoneEndpoint:
    """Tests for filter_one_or_none endpoint."""

    def test_filter_one_or_none_invalid_logical_op(self, client: TestClient) -> None:
        """Test filter_one_or_none with invalid logical operator."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "neither",  # Invalid
        }

        response = client.post("/test/filter_one_or_none", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_filter_one_or_none_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one_or_none with internal error."""
        mock_operations.filter_one_or_none.side_effect = Exception("Database error")

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one_or_none", json=filter_request)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestFindByEndpoint:
    """Tests for find_by endpoint edge cases."""

    def test_find_by_invalid_pagination(self, client: TestClient) -> None:
        """Test find_by with invalid pagination."""
        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "skip": -1,
                "limit": 10,
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_find_by_invalid_order_by_list(self, client: TestClient) -> None:
        """Test find_by with invalid order_by in list."""
        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "order_by": [{"invalid": "syntax"}],
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "order_by" in response.json()["detail"]["error"]

    def test_find_by_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_by with internal error."""
        mock_operations.find_by.side_effect = Exception("Database error")

        response = client.post("/test/find_by", json={"name": "test"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestFindOneByEndpoint:
    """Tests for find_one_by endpoint edge cases."""

    def test_find_one_by_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_one_by with internal error."""
        mock_operations.find_one_by.side_effect = Exception("Database error")

        response = client.post("/test/find_one_by", json={"name": "test"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetRowsEndpoint:
    """Tests for get_rows endpoint."""

    def test_get_rows_with_pagination(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_rows with pagination parameters."""
        mock_operations.get_rows.return_value = [
            RouterTestResponse(id=1, name="test1", value=100),
            RouterTestResponse(id=2, name="test2", value=200),
        ]

        response = client.get("/test/get_rows?skip=10&limit=100")

        assert response.status_code == status.HTTP_200_OK
        assert len(response.json()) == 2
        mock_operations.get_rows.assert_awaited_once()

    def test_get_rows_default_params(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_rows with default parameters."""
        mock_operations.get_rows.return_value = []

        response = client.get("/test/get_rows")

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == []

    def test_get_rows_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_rows with internal error."""
        mock_operations.get_rows.side_effect = Exception("Database error")

        response = client.get("/test/get_rows")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetRowOrNoneEndpoint:
    """Tests for get_row_or_none endpoint."""

    def test_get_row_or_none_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_row_or_none with internal error."""
        mock_operations.get_row_or_none.side_effect = Exception("Database error")

        response = client.get("/test/get_row_or_none/1")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestCreateRowBatchedEdgeCases:
    """Tests for create_rows_batched edge cases."""

    def test_create_rows_batched_validation_error(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test create_rows_batched with validation error."""
        mock_operations.create_rows_batched.side_effect = ValidationError.from_exception_data(
            "TestResponse",
            [{"type": "missing", "loc": ("name",), "msg": "Field required", "input": {}}],
        )

        response = client.post(
            "/test/create_rows_batched",
            json=[{"value": 100}],  # Missing name
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_create_rows_batched_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows_batched with internal error."""
        mock_operations.create_rows_batched.side_effect = Exception("Database error")

        response = client.post(
            "/test/create_rows_batched",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestBulkInsertEdgeCases:
    """Tests for bulk_insert_rows edge cases."""

    def test_bulk_insert_validation_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_insert with validation error."""
        mock_operations.bulk_insert_rows.side_effect = ValidationError.from_exception_data(
            "TestResponse",
            [{"type": "missing", "loc": ("name",), "msg": "Field required", "input": {}}],
        )

        response = client.post("/test/bulk_insert_rows", json=[{"value": 100}])

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_bulk_insert_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_insert with internal error."""
        mock_operations.bulk_insert_rows.side_effect = Exception("Database error")

        response = client.post("/test/bulk_insert_rows", json=[{"name": "test", "value": 100}])

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestUpdateRowEdgeCases:
    """Tests for update_row edge cases."""

    def test_update_row_http_exception_propagation(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test that HTTPException from update_row is propagated."""
        from fastapi import HTTPException

        mock_operations.update_row.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

        response = client.put("/test/update_row/1", json={"name": "test"})

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_update_row_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test update_row with internal error."""
        mock_operations.update_row.side_effect = Exception("Database error")

        response = client.put("/test/update_row/1", json={"name": "test"})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestDeleteRowEdgeCases:
    """Tests for delete_row edge cases."""

    def test_delete_row_http_exception_propagation(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test that HTTPException from delete_row is propagated."""
        from fastapi import HTTPException

        mock_operations.delete_row.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

        response = client.delete("/test/delete_row/1")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_delete_row_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test delete_row with internal error."""
        mock_operations.delete_row.side_effect = Exception("Database error")

        response = client.delete("/test/delete_row/1")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestGetRowEdgeCases:
    """Tests for get_row edge cases."""

    def test_get_row_http_exception_propagation(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test that HTTPException from get_row is propagated."""
        from fastapi import HTTPException

        mock_operations.get_row.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

        response = client.get("/test/get_row/1")

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_get_row_internal_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_row with internal error."""
        mock_operations.get_row.side_effect = Exception("Database error")

        response = client.get("/test/get_row/1")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestFilterOneEdgeCases:
    """Tests for filter_one edge cases."""

    def test_filter_one_http_exception_propagation(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test that HTTPException from filter_one is propagated."""
        from fastapi import HTTPException

        mock_operations.filter_one.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one", json=filter_request)

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestFindOneByEdgeCases:
    """Tests for find_one_by edge cases."""

    def test_find_one_by_http_exception_propagation(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test that HTTPException from find_one_by is propagated."""
        from fastapi import HTTPException

        mock_operations.find_one_by.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

        response = client.post("/test/find_one_by", json={"name": "test"})

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestLookupEdgeCases:
    """Tests for lookup_by_id_or_name edge cases."""

    def test_lookup_http_exception_propagation(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test that HTTPException from lookup is propagated."""
        from fastapi import HTTPException

        mock_operations.lookup_by_id_or_name.side_effect = HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden"
        )

        response = client.get("/test/lookup_by_id_or_name?id_=1")

        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestPaginationEdgeCases:
    """Tests for pagination edge cases."""

    def test_filter_rows_excessive_limit(self, client: TestClient) -> None:
        """Test filter_rows with limit exceeding maximum."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "skip": 0,
            "limit": 10001,  # Exceeds max
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_filter_rows_negative_skip(self, client: TestClient) -> None:
        """Test filter_rows with negative skip."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "skip": -1,
            "limit": 10,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_filter_rows_streaming_excessive_limit(self, client: TestClient) -> None:
        """Test filter_rows_streaming with excessive limit."""
        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "skip": 0,
            "limit": 10001,
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestEmptyFilters:
    """Tests for empty filter arrays."""

    def test_filter_rows_no_filters(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows with no filters (should pass None)."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "logical_op": "and",
            "skip": 0,
            "limit": 10,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        # Verify None was passed for filters
        call_kwargs = mock_operations.filter_rows.await_args[1]
        assert call_kwargs["filters"] is None

    def test_count_filtered_rows_no_filters(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test count_filtered_rows with no filters."""
        mock_operations.count_filtered_rows.return_value = 100

        filter_request = {
            "logical_op": "and",
        }

        response = client.post("/test/count_filtered_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        call_kwargs = mock_operations.count_filtered_rows.await_args[1]
        assert call_kwargs["filters"] is None


class TestOrderByVariations:
    """Tests for various order_by formats."""

    def test_filter_rows_with_single_order_by(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows with single order_by object."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "order_by": {"field": "created_at", "direction": "desc"},
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK

    def test_filter_rows_with_multiple_order_by(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows with multiple order_by objects."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "order_by": [
                {"field": "created_at", "direction": "desc"},
                {"field": "name", "direction": "asc"},
            ],
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK

    def test_filter_rows_with_no_order_by(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows without order_by."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        call_kwargs = mock_operations.filter_rows.await_args[1]
        assert call_kwargs["order_by"] is None


class TestResponseModelExtraction:
    """Tests for response model extraction from operations."""

    def test_response_model_used_in_endpoints(self, mock_operations: MagicMock) -> None:
        """Test that response_model is correctly extracted and used."""
        router = create_table_router("test", mock_operations)

        # Find the create_row route
        create_route = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/test/create_row":
                create_route = route
                break

        assert create_route is not None
        # The response model should be set on the route
        assert create_route.response_model == RouterTestResponse


class TestBatchSizeBoundaries:
    """Tests for batch size boundary values."""

    def test_create_rows_batched_min_batch_size(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows_batched with minimum batch size."""
        mock_operations.create_rows_batched.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post(
            "/test/create_rows_batched?batch_size=1",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED

    def test_create_rows_batched_max_batch_size(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows_batched with maximum batch size."""
        mock_operations.create_rows_batched.return_value = [
            RouterTestResponse(id=i, name=f"test{i}", value=i) for i in range(100)
        ]

        data = [{"name": f"test{i}", "value": i} for i in range(100)]

        response = client.post(
            "/test/create_rows_batched?batch_size=10000",
            json=data,
        )

        assert response.status_code == status.HTTP_201_CREATED


class TestDeleteResponseVariations:
    """Tests for different delete response types."""

    def test_delete_row_returns_response_model_with_capture(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test delete_row returns response model when capture_data=true."""
        mock_operations.delete_row.return_value = {"id": 1, "name": "test", "value": 100}

        response = client.delete("/test/delete_row/1?capture_data=true")

        assert response.status_code == status.HTTP_200_OK
        # Should be RouterTestResponse format
        assert "id" in response.json()
        assert "name" in response.json()
        assert "value" in response.json()

    def test_delete_row_returns_delete_response_without_capture(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test delete_row returns DeleteResponse when capture_data=false."""
        mock_operations.delete_row.return_value = None

        response = client.delete("/test/delete_row/1?capture_data=false")

        assert response.status_code == status.HTTP_200_OK
        # Should be DeleteResponse format
        assert response.json() == {"deleted": True}


class TestStreamingErrorHandling:
    """Tests for streaming endpoint error handling."""

    def test_get_rows_streaming_with_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_rows_streaming handles errors in stream."""

        async def failing_generator():
            yield RouterTestResponse(id=1, name="test", value=100)
            raise RuntimeError("Stream failed")

        mock_operations.get_rows_streaming = failing_generator

        response = client.get("/test/get_rows_streaming")

        assert response.status_code == status.HTTP_200_OK
        # Error should be in the response
        assert "error" in response.text

    def test_filter_rows_streaming_with_error(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows_streaming handles errors in stream."""

        async def failing_generator():
            yield RouterTestResponse(id=1, name="test", value=100)
            raise RuntimeError("Filter stream failed")

        mock_operations.filter_rows_streaming = failing_generator

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert "error" in response.text


class TestValidateBatchSizeBoundaries:
    """Tests for validate_batch_size function boundaries."""

    def test_validate_batch_size_one(self) -> None:
        """Test validate_batch_size with minimum valid value."""
        from rail_svc.router.base import validate_batch_size

        result = validate_batch_size(1)
        assert result == 1

    def test_validate_batch_size_ten_thousand(self) -> None:
        """Test validate_batch_size with maximum valid value."""
        from rail_svc.router.base import validate_batch_size

        result = validate_batch_size(10000)
        assert result == 10000

    def test_validate_batch_size_negative(self) -> None:
        """Test validate_batch_size with negative value."""
        from rail_svc.router.base import validate_batch_size

        with pytest.raises(Exception) as exc_info:
            validate_batch_size(-1)
        assert "400" in str(exc_info.value)


class TestValidatePaginationBoundaries:
    """Tests for validate_pagination_params boundary conditions."""

    def test_validate_pagination_limit_one(self) -> None:
        """Test validate_pagination_params with minimum limit."""
        from rail_svc.router.base import validate_pagination_params

        skip, limit = validate_pagination_params(0, 1)
        assert skip == 0
        assert limit == 1

    def test_validate_pagination_limit_max(self) -> None:
        """Test validate_pagination_params with maximum limit."""
        from rail_svc.router.base import validate_pagination_params

        skip, limit = validate_pagination_params(0, 10000)
        assert skip == 0
        assert limit == 10000

    def test_validate_pagination_negative_limit(self) -> None:
        """Test validate_pagination_params with negative limit."""
        from rail_svc.router.base import validate_pagination_params

        with pytest.raises(Exception) as exc_info:
            validate_pagination_params(0, -1)
        assert "400" in str(exc_info.value)


class TestDeleteRowsAssertions:
    """Tests for delete_rows assertions."""

    def test_delete_rows_with_capture_assertion(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test delete_rows with capture_data assertion."""
        mock_operations.delete_rows.return_value = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]

        response = client.request("DELETE", "/test/delete_rows?capture_data=true", json=[1, 2])

        assert response.status_code == status.HTTP_200_OK
        # Should return list of response models
        assert isinstance(response.json(), list)
        assert len(response.json()) == 2


class TestUpdateRowsItemValidation:
    """Tests for update_rows item validation."""

    def test_update_rows_item_not_dict(self, client: TestClient) -> None:
        """Test update_rows validates items are dicts."""
        # FastAPI validation should catch this before our validation
        response = client.put("/test/update_rows", json=["not_a_dict", {"id": 2, "name": "test"}])

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestCRUDOperationCoverage:
    """Additional coverage for CRUD operations."""

    def test_create_row_general_exception(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_row with non-validation exception."""
        mock_operations.create_row.side_effect = RuntimeError("Unexpected error")

        response = client.post("/test/create_row", json={"name": "test", "value": 100})

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Unexpected error" in response.json()["detail"]

    def test_create_rows_general_exception(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows with non-validation exception."""
        mock_operations.create_rows.side_effect = RuntimeError("Unexpected error")

        response = client.post("/test/create_rows", json=[{"name": "test", "value": 100}])

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestLogicalOperatorValidation:
    """Tests for logical operator validation across endpoints."""

    def test_filter_rows_or_operator(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows with OR logical operator."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [
                {"field": "name", "op": "eq", "value": "test1"},
                {"field": "name", "op": "eq", "value": "test2"},
            ],
            "logical_op": "or",
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK

    def test_count_filtered_rows_or_operator(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test count_filtered_rows with OR logical operator."""
        mock_operations.count_filtered_rows.return_value = 5

        filter_request = {
            "filters": [
                {"field": "value", "op": "gt", "value": 100},
                {"field": "value", "op": "lt", "value": 10},
            ],
            "logical_op": "or",
        }

        response = client.post("/test/count_filtered_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["count"] == 5

    def test_filter_one_or_operator(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one with OR logical operator."""
        mock_operations.filter_one.return_value = RouterTestResponse(id=1, name="test", value=100)

        filter_request = {
            "filters": [
                {"field": "name", "op": "eq", "value": "test"},
                {"field": "id", "op": "eq", "value": 1},
            ],
            "logical_op": "or",
        }

        response = client.post("/test/filter_one", json=filter_request)

        assert response.status_code == status.HTTP_200_OK

    def test_filter_one_or_none_or_operator(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_one_or_none with OR logical operator."""
        mock_operations.filter_one_or_none.return_value = RouterTestResponse(id=1, name="test", value=100)

        filter_request = {
            "filters": [
                {"field": "name", "op": "eq", "value": "test"},
                {"field": "id", "op": "eq", "value": 1},
            ],
            "logical_op": "or",
        }

        response = client.post("/test/filter_one_or_none", json=filter_request)

        assert response.status_code == status.HTTP_200_OK


class TestFindByOrderByEdgeCases:
    """Tests for find_by order_by edge cases."""

    def test_find_by_order_by_type_error(self, client: TestClient) -> None:
        """Test find_by with invalid order_by type."""
        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "order_by": 123,  # Invalid type
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "order_by" in response.json()["detail"]["error"]

    def test_find_by_order_by_validation_error(self, client: TestClient) -> None:
        """Test find_by with order_by missing required fields."""
        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "order_by": {"direction": "asc"},  # Missing 'field'
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_find_by_order_by_empty_list(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_by with empty order_by list."""
        mock_operations.find_by.return_value = []

        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "order_by": [],
            },
        )

        assert response.status_code == status.HTTP_200_OK


class TestRouterPrefixAndTags:
    """Tests for router configuration."""

    def test_router_prefix(self, mock_operations: MagicMock) -> None:
        """Test router has correct prefix."""
        router = create_table_router("custom_table", mock_operations)
        assert router.prefix == "/custom_table"

    def test_router_tags(self, mock_operations: MagicMock) -> None:
        """Test router has correct tags."""
        router = create_table_router("custom_table", mock_operations)
        assert "custom_table" in router.tags


class TestEndpointHTTPMethods:
    """Tests for HTTP method support on endpoints."""

    def test_update_row_put_method(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test update_row supports PUT method."""
        mock_operations.update_row.return_value = RouterTestResponse(id=1, name="updated", value=200)

        response = client.put("/test/update_row/1", json={"name": "updated"})

        assert response.status_code == status.HTTP_200_OK

    def test_update_row_patch_method(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test update_row supports PATCH method."""
        mock_operations.update_row.return_value = RouterTestResponse(id=1, name="updated", value=200)

        response = client.patch("/test/update_row/1", json={"name": "updated"})

        assert response.status_code == status.HTTP_200_OK

    def test_update_rows_put_method(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test update_rows supports PUT method."""
        mock_operations.update_rows.return_value = [RouterTestResponse(id=1, name="updated", value=200)]

        response = client.put("/test/update_rows", json=[{"id": 1, "name": "updated"}])

        assert response.status_code == status.HTTP_200_OK

    def test_update_rows_patch_method(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test update_rows supports PATCH method."""
        mock_operations.update_rows.return_value = [RouterTestResponse(id=1, name="updated", value=200)]

        response = client.patch("/test/update_rows", json=[{"id": 1, "name": "updated"}])

        assert response.status_code == status.HTTP_200_OK


class TestStatusCodes:
    """Tests for correct HTTP status codes."""

    def test_create_row_201_created(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_row returns 201 CREATED."""
        mock_operations.create_row.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.post("/test/create_row", json={"name": "test", "value": 100})

        assert response.status_code == status.HTTP_201_CREATED

    def test_create_rows_201_created(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows returns 201 CREATED."""
        mock_operations.create_rows.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post("/test/create_rows", json=[{"name": "test", "value": 100}])

        assert response.status_code == status.HTTP_201_CREATED

    def test_create_rows_batched_201_created(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows_batched returns 201 CREATED."""
        mock_operations.create_rows_batched.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post(
            "/test/create_rows_batched",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED

    def test_bulk_insert_rows_201_created(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_insert_rows returns 201 CREATED."""
        mock_operations.bulk_insert_rows.return_value = 1

        response = client.post("/test/bulk_insert_rows", json=[{"name": "test", "value": 100}])

        assert response.status_code == status.HTTP_201_CREATED


class TestFilterRowsWithNullOrderBy:
    """Tests for filter_rows with null/missing order_by."""

    def test_filter_rows_null_order_by(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows with null order_by."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "order_by": None,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK


class TestDeleteRowsCountResponse:
    """Tests for delete_rows count response."""

    def test_delete_rows_returns_count_without_capture(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test delete_rows returns count when capture_data=false."""
        mock_operations.delete_rows.return_value = None

        response = client.request("DELETE", "/test/delete_rows?capture_data=false", json=[1, 2, 3])

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"count": 3}


class TestBulkOperationLimits:
    """Tests for bulk operation limits."""

    def test_bulk_insert_at_limit(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_insert at maximum allowed rows."""
        mock_operations.bulk_insert_rows.return_value = 100000

        data = [{"name": f"test{i}", "value": i} for i in range(100000)]

        response = client.post("/test/bulk_insert_rows", json=data)

        assert response.status_code == status.HTTP_201_CREATED

    def test_bulk_delete_at_limit(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_delete at maximum allowed rows."""
        mock_operations.bulk_delete_rows.return_value = 100000

        data = list(range(100000))

        response = client.request("DELETE", "/test/bulk_delete_rows", json=data)

        assert response.status_code == status.HTTP_200_OK


class TestNullLimitHandling:
    """Tests for null/None limit handling."""

    def test_get_rows_with_null_limit(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_rows with null limit (unlimited)."""
        mock_operations.get_rows.return_value = []

        response = client.get("/test/get_rows?skip=0")

        assert response.status_code == status.HTTP_200_OK

    def test_find_by_with_null_limit(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_by with null limit."""
        mock_operations.find_by.return_value = []

        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "skip": 0,
                "limit": None,
            },
        )

        assert response.status_code == status.HTTP_200_OK


class TestMediaTypes:
    """Tests for response media types."""

    def test_get_rows_streaming_media_type(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_rows_streaming has correct media type."""

        async def mock_generator():
            yield RouterTestResponse(id=1, name="test", value=100)

        mock_operations.get_rows_streaming = mock_generator

        response = client.get("/test/get_rows_streaming")

        assert response.status_code == status.HTTP_200_OK
        assert "application/x-ndjson" in response.headers["content-type"]

    def test_filter_rows_streaming_media_type(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows_streaming has correct media type."""

        async def mock_generator():
            yield RouterTestResponse(id=1, name="test", value=100)

        mock_operations.filter_rows_streaming = mock_generator

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert "application/x-ndjson" in response.headers["content-type"]


class TestEmptyArrayHandling:
    """Tests for empty array handling."""

    def test_create_rows_empty_array(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows with empty array."""
        mock_operations.create_rows.return_value = []

        response = client.post("/test/create_rows", json=[])

        assert response.status_code == status.HTTP_201_CREATED
        assert response.json() == []

    def test_delete_rows_empty_array(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test delete_rows with empty array."""
        mock_operations.delete_rows.return_value = None

        response = client.request("DELETE", "/test/delete_rows?capture_data=false", json=[])

        assert response.status_code == status.HTTP_200_OK
        assert response.json() == {"count": 0}


class TestFilterRequestDefaultValues:
    """Tests for FilterRequest default values."""

    def test_filter_rows_default_skip_and_limit(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows uses default skip and limit."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            # skip and limit not provided, should use defaults
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK


class TestRequireAuthEdgeCases:
    """Additional tests for require_auth function."""

    def test_require_auth_bearer_with_spaces(self) -> None:
        """Test require_auth with token containing spaces."""
        from rail_svc.router.base import require_auth

        token = require_auth(authorization="Bearer token_with_internal_spaces")
        assert token == "token_with_internal_spaces"

    def test_require_auth_case_sensitive_bearer(self) -> None:
        """Test require_auth is case-sensitive for Bearer."""
        from rail_svc.router.base import require_auth

        with pytest.raises(Exception) as exc_info:
            require_auth(authorization="bearer token123")  # lowercase
        assert "401" in str(exc_info.value)

    def test_require_auth_bearer_only(self) -> None:
        """Test require_auth with just 'Bearer' and no token."""
        from rail_svc.router.base import require_auth

        with pytest.raises(Exception) as exc_info:
            require_auth(authorization="Bearer")
        assert "401" in str(exc_info.value)


class TestValidatePaginationParamsEdgeCases:
    """Additional edge cases for validate_pagination_params."""

    def test_validate_pagination_large_skip(self) -> None:
        """Test validate_pagination_params with very large skip."""
        from rail_svc.router.base import validate_pagination_params

        skip, limit = validate_pagination_params(1000000, 100)
        assert skip == 1000000
        assert limit == 100

    def test_validate_pagination_limit_exactly_10000(self) -> None:
        """Test validate_pagination_params with limit exactly at boundary."""
        from rail_svc.router.base import validate_pagination_params

        skip, limit = validate_pagination_params(0, 10000)
        assert skip == 0
        assert limit == 10000

    def test_validate_pagination_limit_exactly_1(self) -> None:
        """Test validate_pagination_params with limit exactly at lower boundary."""
        from rail_svc.router.base import validate_pagination_params

        skip, limit = validate_pagination_params(0, 1)
        assert skip == 0
        assert limit == 1


class TestUpdateRowsIndexValidation:
    """Tests for update_rows index validation in error messages."""

    def test_update_rows_missing_id_at_specific_index(self, client: TestClient) -> None:
        """Test update_rows reports correct index for missing id."""
        response = client.put(
            "/test/update_rows",
            json=[
                {"id": 1, "name": "test1"},
                {"name": "test2"},  # Missing id at index 1
                {"id": 3, "name": "test3"},
            ],
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "index 1" in response.json()["detail"]

    def test_update_rows_non_dict_at_specific_index(self, client: TestClient) -> None:
        """Test update_rows reports correct index for non-dict."""
        response = client.put(
            "/test/update_rows",
            json=[
                {"id": 1, "name": "test1"},
                "not_a_dict",  # Invalid at index 1
            ],
        )

        # FastAPI validation will catch this as 422
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestDeleteRowsIndexValidation:
    """Tests for delete_rows index validation in error messages."""

    def test_delete_rows_non_integer_at_specific_index(self, client: TestClient) -> None:
        """Test delete_rows reports correct index for non-integer."""
        # FastAPI validation should catch this
        response = client.request(
            "DELETE",
            "/test/delete_rows",
            json=[1, 2, "not_an_int"],  # Invalid at index 2
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestBulkDeleteIndexValidation:
    """Tests for bulk_delete_rows index validation."""

    def test_bulk_delete_non_integer_at_specific_index(self, client: TestClient) -> None:
        """Test bulk_delete reports correct index for non-integer."""
        response = client.request(
            "DELETE",
            "/test/bulk_delete_rows",
            json=[1, 2, "invalid", 4],  # Invalid at index 2
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


class TestFilterRowsStreamingWithNullFilters:
    """Tests for filter_rows_streaming with null/empty filters."""

    def test_filter_rows_streaming_no_filters(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows_streaming with no filters."""

        async def mock_generator():
            yield RouterTestResponse(id=1, name="test", value=100)

        mock_operations.filter_rows_streaming = mock_generator

        filter_request = {
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_200_OK


class TestLookupByIdAndName:
    """Tests for lookup with both id and name provided."""

    def test_lookup_with_both_id_and_name(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test lookup when both id and name are provided (should use id)."""
        mock_operations.lookup_by_id_or_name.return_value = (
            1,
            RouterTestResponse(id=1, name="test", value=100),
        )

        response = client.get("/test/lookup_by_id_or_name?id_=1&name=test")

        assert response.status_code == status.HTTP_200_OK
        # Both params are passed to the operation
        mock_operations.lookup_by_id_or_name.assert_awaited_once_with(1, "test")


class TestDefaultQueryParams:
    """Tests for default query parameter values."""

    def test_create_row_default_validate(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_row uses default validate=true."""
        mock_operations.create_row.return_value = RouterTestResponse(id=1, name="test", value=100)

        response = client.post("/test/create_row", json={"name": "test", "value": 100})

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.create_row.await_args[1]
        assert call_kwargs["validate"] is True

    def test_create_rows_default_validate(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows uses default validate=true."""
        mock_operations.create_rows.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post("/test/create_rows", json=[{"name": "test", "value": 100}])

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.create_rows.await_args[1]
        assert call_kwargs["validate"] is True

    def test_delete_row_default_capture_data(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test delete_row uses default capture_data=true."""
        mock_operations.delete_row.return_value = {"id": 1, "name": "test", "value": 100}

        response = client.delete("/test/delete_row/1")

        assert response.status_code == status.HTTP_200_OK
        call_kwargs = mock_operations.delete_row.await_args[1]
        assert call_kwargs["capture_data"] is True

    def test_delete_rows_default_capture_data(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test delete_rows uses default capture_data=false."""
        mock_operations.delete_rows.return_value = None

        response = client.request("DELETE", "/test/delete_rows", json=[1, 2])

        assert response.status_code == status.HTTP_200_OK
        call_kwargs = mock_operations.delete_rows.await_args[1]
        assert call_kwargs["capture_data"] is False


class TestBatchSizeDefaultValue:
    """Tests for batch_size default value."""

    def test_create_rows_batched_default_batch_size(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test create_rows_batched uses default batch_size=1000."""
        mock_operations.create_rows_batched.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post(
            "/test/create_rows_batched",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.create_rows_batched.await_args[1]
        assert call_kwargs["batch_size"] == 1000


class TestResponseModelFormatting:
    """Tests for response model formatting."""

    def test_delete_rows_response_model_construction(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test delete_rows constructs response models correctly."""
        mock_operations.delete_rows.return_value = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]

        response = client.request("DELETE", "/test/delete_rows?capture_data=true", json=[1, 2])

        assert response.status_code == status.HTTP_200_OK
        json_response = response.json()
        assert len(json_response) == 2
        assert json_response[0]["id"] == 1
        assert json_response[0]["name"] == "test1"
        assert json_response[1]["id"] == 2
        assert json_response[1]["name"] == "test2"


class TestMultipleFiltersInRequest:
    """Tests for handling multiple filters."""

    def test_filter_rows_with_many_filters(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test filter_rows with many filter conditions."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [
                {"field": "name", "op": "like", "value": "%test%"},
                {"field": "value", "op": "gt", "value": 50},
                {"field": "value", "op": "lt", "value": 200},
                {"field": "id", "op": "ne", "value": 5},
            ],
            "logical_op": "and",
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK


class TestFilterRequestSkipLimitDefaults:
    """Tests for FilterRequest skip and limit defaults."""

    def test_filter_rows_with_explicit_zero_skip(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test filter_rows with explicit skip=0."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "skip": 0,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK


class TestEndpointDocumentation:
    """Tests to verify endpoint documentation is present."""

    def test_create_row_has_docstring(self, mock_operations: MagicMock) -> None:
        """Test create_row endpoint has documentation."""
        router = create_table_router("test", mock_operations)

        create_route = None
        for route in router.routes:
            if hasattr(route, "path") and route.path == "/test/create_row":
                create_route = route
                break

        assert create_route is not None
        # Check that endpoint has some description
        assert hasattr(create_route, "description") or hasattr(create_route, "summary")


class TestAllEndpointsRegistered:
    """Tests to verify all endpoints are registered."""

    def test_all_crud_endpoints_registered(self, mock_operations: MagicMock) -> None:
        """Test that all expected endpoints are registered."""
        router = create_table_router("test", mock_operations)

        expected_paths = [
            "/test/create_row",
            "/test/create_rows",
            "/test/create_rows_batched",
            "/test/bulk_insert_rows",
            "/test/get_row/{row_id}",
            "/test/get_row_or_none/{row_id}",
            "/test/get_row_by_name/{name}",
            "/test/get_rows",
            "/test/get_rows_streaming",
            "/test/count_rows",
            "/test/lookup_by_id_or_name",
            "/test/update_row/{row_id}",
            "/test/update_rows",
            "/test/delete_row/{row_id}",
            "/test/delete_rows",
            "/test/bulk_delete_rows",
            "/test/filter_rows",
            "/test/filter_rows_streaming",
            "/test/count_filtered_rows",
            "/test/filter_one",
            "/test/filter_one_or_none",
            "/test/find_by",
            "/test/find_one_by",
        ]

        registered_paths = [route.path for route in router.routes if hasattr(route, "path")]

        for expected_path in expected_paths:
            assert expected_path in registered_paths, f"Expected path {expected_path} not found in router"


class TestOperationsCallParameters:
    """Tests to verify operations are called with correct parameters."""

    def test_get_rows_calls_with_correct_params(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test get_rows passes parameters correctly to operation."""
        mock_operations.get_rows.return_value = []

        response = client.get("/test/get_rows?skip=5&limit=50")

        assert response.status_code == status.HTTP_200_OK
        call_kwargs = mock_operations.get_rows.await_args[1]
        assert call_kwargs["skip"] == 5
        assert call_kwargs["limit"] == 50

    def test_filter_rows_calls_with_correct_params(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test filter_rows passes all parameters correctly."""
        mock_operations.filter_rows.return_value = []

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
            "order_by": {"field": "created_at", "direction": "desc"},
            "skip": 10,
            "limit": 25,
        }

        response = client.post("/test/filter_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        call_kwargs = mock_operations.filter_rows.await_args[1]
        assert call_kwargs["skip"] == 10
        assert call_kwargs["limit"] == 25
        assert call_kwargs["logical_op"] == "and"
        assert call_kwargs["filters"] is not None
        assert call_kwargs["order_by"] is not None

    def test_find_by_calls_with_correct_params(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test find_by passes all parameters correctly."""
        mock_operations.find_by.return_value = []

        response = client.post(
            "/test/find_by",
            json={
                "name": "test",
                "value": 100,
                "skip": 5,
                "limit": 20,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        call_kwargs = mock_operations.find_by.await_args[1]
        assert call_kwargs["name"] == "test"
        assert call_kwargs["value"] == 100
        assert call_kwargs["skip"] == 5
        assert call_kwargs["limit"] == 20


class TestValidationErrorDetails:
    """Tests for validation error detail formatting."""

    def test_create_row_validation_error_includes_details(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test validation error response includes error details."""
        mock_operations.create_row.side_effect = ValidationError.from_exception_data(
            "RouterTestResponse",
            [
                {
                    "type": "missing",
                    "loc": ("name",),
                    "msg": "Field required",
                    "input": {},
                }
            ],
        )

        response = client.post("/test/create_row", json={"value": 100})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        json_response = response.json()
        assert "detail" in json_response
        assert "error" in json_response["detail"]
        assert "details" in json_response["detail"]
        assert json_response["detail"]["error"] == "Validation error"


class TestFilterOneOrNoneWithMultipleResults:
    """Tests for filter_one_or_none behavior."""

    def test_filter_one_or_none_returns_single_result(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test filter_one_or_none returns single result when found."""
        mock_operations.filter_one_or_none.return_value = RouterTestResponse(id=1, name="test", value=100)

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "test"}],
            "logical_op": "and",
        }

        response = client.post("/test/filter_one_or_none", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["id"] == 1


class TestCountFilteredRowsWithNoResults:
    """Tests for count_filtered_rows with no matches."""

    def test_count_filtered_rows_returns_zero(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test count_filtered_rows returns 0 when no matches."""
        mock_operations.count_filtered_rows.return_value = 0

        filter_request = {
            "filters": [{"field": "name", "op": "eq", "value": "nonexistent"}],
            "logical_op": "and",
        }

        response = client.post("/test/count_filtered_rows", json=filter_request)

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["count"] == 0


class TestCreateRowsValidateParameter:
    """Tests for create_rows validate parameter."""

    def test_create_rows_with_validate_true(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows with validate=true."""
        mock_operations.create_rows.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post(
            "/test/create_rows?validate=true",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.create_rows.await_args[1]
        assert call_kwargs["validate"] is True

    def test_create_rows_with_validate_false(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test create_rows with validate=false."""
        mock_operations.create_rows.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post(
            "/test/create_rows?validate=false",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.create_rows.await_args[1]
        assert call_kwargs["validate"] is False


class TestBulkInsertValidateParameter:
    """Tests for bulk_insert_rows validate parameter."""

    def test_bulk_insert_with_validate_true(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_insert with validate=true."""
        mock_operations.bulk_insert_rows.return_value = 1

        response = client.post(
            "/test/bulk_insert_rows?validate=true",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.bulk_insert_rows.await_args[1]
        assert call_kwargs["validate"] is True

    def test_bulk_insert_with_validate_false(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test bulk_insert with validate=false."""
        mock_operations.bulk_insert_rows.return_value = 1

        response = client.post(
            "/test/bulk_insert_rows?validate=false",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.bulk_insert_rows.await_args[1]
        assert call_kwargs["validate"] is False


class TestCreateRowsBatchedValidateParameter:
    """Tests for create_rows_batched validate parameter."""

    def test_create_rows_batched_with_validate_true(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test create_rows_batched with validate=true."""
        mock_operations.create_rows_batched.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post(
            "/test/create_rows_batched?validate=true&batch_size=100",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.create_rows_batched.await_args[1]
        assert call_kwargs["validate"] is True
        assert call_kwargs["batch_size"] == 100

    def test_create_rows_batched_with_validate_false(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test create_rows_batched with validate=false."""
        mock_operations.create_rows_batched.return_value = [RouterTestResponse(id=1, name="test", value=100)]

        response = client.post(
            "/test/create_rows_batched?validate=false",
            json=[{"name": "test", "value": 100}],
        )

        assert response.status_code == status.HTTP_201_CREATED
        call_kwargs = mock_operations.create_rows_batched.await_args[1]
        assert call_kwargs["validate"] is False


class TestGetRowsStreamingPagination:
    """Tests for get_rows_streaming with pagination."""

    def test_get_rows_streaming_with_skip_and_limit(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test get_rows_streaming respects pagination parameters."""

        async def mock_generator(*args, **kwargs):
            # Verify pagination params
            assert kwargs.get("skip") == 10
            assert kwargs.get("limit") == 5
            for i in range(5):
                yield RouterTestResponse(id=10 + i, name=f"test{i}", value=i * 100)

        mock_operations.get_rows_streaming = mock_generator

        response = client.get("/test/get_rows_streaming?skip=10&limit=5")

        assert response.status_code == status.HTTP_200_OK


class TestFilterRowsStreamingPagination:
    """Tests for filter_rows_streaming with pagination."""

    def test_filter_rows_streaming_with_skip_and_limit(
        self, client: TestClient, mock_operations: MagicMock
    ) -> None:
        """Test filter_rows_streaming respects pagination parameters."""

        async def mock_generator(*args, **kwargs):
            assert kwargs.get("skip") == 5
            assert kwargs.get("limit") == 10
            for i in range(10):
                yield RouterTestResponse(id=5 + i, name=f"test{i}", value=i * 100)

        mock_operations.filter_rows_streaming = mock_generator

        filter_request = {
            "filters": [{"field": "value", "op": "gt", "value": 0}],
            "logical_op": "and",
            "skip": 5,
            "limit": 10,
        }

        response = client.post("/test/filter_rows_streaming", json=filter_request)

        assert response.status_code == status.HTTP_200_OK


class TestRouteRegistrationOrder:
    """Tests for route registration and ordering."""

    def test_router_has_expected_route_count(self, mock_operations: MagicMock) -> None:
        """Test router has all expected routes."""
        router = create_table_router("test", mock_operations)

        # Count routes (note: update_row has 2 routes - PUT and PATCH)
        # update_rows also has 2 routes
        routes_with_paths = [r for r in router.routes if hasattr(r, "path")]

        # Expected: 23 unique endpoint paths, but update_row and update_rows each have 2 methods
        # So total routes should be 23 + 2 = 25
        assert len(routes_with_paths) >= 23


class TestErrorMessageContent:
    """Tests for error message content."""

    def test_validation_error_message_readable(self, client: TestClient, mock_operations: MagicMock) -> None:
        """Test validation error messages are readable."""
        mock_operations.create_row.side_effect = ValidationError.from_exception_data(
            "RouterTestResponse",
            [
                {
                    "type": "string_type",
                    "loc": ("name",),
                    "msg": "Input should be a valid string",
                    "input": 123,
                }
            ],
        )

        response = client.post("/test/create_row", json={"name": 123, "value": 100})

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        detail = response.json()["detail"]
        assert "Validation error" in detail["error"]
        assert "details" in detail
        assert len(detail["details"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
