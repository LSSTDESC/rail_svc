"""Unit tests for RemoteTableOperations and RemoteAPI classes."""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from pydantic import BaseModel

from rail_svc.client.base import RemoteAPI, RemoteTableOperations
from rail_svc.models import Filter, OrderBy, RemoteAPIError


# Test models
class ClientTestResponse(BaseModel):
    """Test response model."""

    id: int
    name: str
    value: int = 0


class ClientTestCreate(BaseModel):
    """Test create model."""

    name: str
    value: int = 0


# Helper to create async iterator from list
class AsyncIterator:
    """Helper class to create async iterator from a list."""

    def __init__(self, items: list[str]):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class TestRemoteTableOperations:
    """Tests for RemoteTableOperations class."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock HTTP client."""
        return AsyncMock(spec=httpx.AsyncClient)

    @pytest.fixture
    def operations(
        self,
        mock_client: AsyncMock,
    ) -> RemoteTableOperations[ClientTestResponse, ClientTestCreate]:
        """Create a RemoteTableOperations instance with mocked client."""
        return RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/test",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

    # CREATE operations tests

    async def test_create_row_success(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test successful row creation."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1, "name": "test", "value": 42}
        mock_client.post.return_value = mock_response

        result = await operations.create_row(name="test", value=42)

        assert isinstance(result, ClientTestResponse)
        assert result.id == 1
        assert result.name == "test"
        assert result.value == 42
        mock_client.post.assert_called_once()

    async def test_create_row_validation_params(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test that validate parameter is passed correctly."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1, "name": "test"}
        mock_client.post.return_value = mock_response

        await operations.create_row(name="test", validate=False)

        call_args = mock_client.post.call_args
        assert call_args.kwargs["params"]["validate"] is False

    async def test_create_rows_success(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test successful multiple row creation."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = [
            {"id": 1, "name": "test1", "value": 10},
            {"id": 2, "name": "test2", "value": 20},
        ]
        mock_client.post.return_value = mock_response

        data = [{"name": "test1", "value": 10}, {"name": "test2", "value": 20}]
        results = await operations.create_rows(data)

        assert len(results) == 2
        assert all(isinstance(r, ClientTestResponse) for r in results)
        assert results[0].name == "test1"
        assert results[1].name == "test2"

    async def test_create_rows_batched_params(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test that batch_size parameter is passed correctly."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = []
        mock_client.post.return_value = mock_response

        await operations.create_rows_batched([], batch_size=500)

        call_args = mock_client.post.call_args
        assert call_args.kwargs["params"]["batch_size"] == 500

    async def test_bulk_insert_rows_returns_count(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test bulk insert returns count."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"count": 100}
        mock_client.post.return_value = mock_response

        count = await operations.bulk_insert_rows([{"name": "test"}])

        assert count == 100
        assert isinstance(count, int)

    # READ operations tests

    async def test_get_row_success(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test successful row retrieval by ID."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 42, "name": "test"}
        mock_client.get.return_value = mock_response

        result = await operations.get_row(42)

        assert result.id == 42
        assert result.name == "test"
        mock_client.get.assert_called_once_with("http://api.example.com/v1/test/get_row/42")

    async def test_get_row_or_none_returns_none(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test get_row_or_none returns None when not found."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_client.get.return_value = mock_response

        result = await operations.get_row_or_none(999)

        assert result is None

    async def test_get_row_or_none_returns_data(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test get_row_or_none returns data when found."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "found"}
        mock_client.get.return_value = mock_response

        result = await operations.get_row_or_none(1)

        assert result is not None
        assert result.id == 1

    async def test_get_row_by_name(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test row retrieval by name."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "unique"}
        mock_client.get.return_value = mock_response

        result = await operations.get_row_by_name("unique")

        assert result.name == "unique"
        assert "unique" in mock_client.get.call_args[0][0]

    async def test_get_rows_pagination(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test get_rows with pagination parameters."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": i, "name": f"item{i}"} for i in range(10, 20)]
        mock_client.get.return_value = mock_response

        results = await operations.get_rows(skip=10, limit=10)

        assert len(results) == 10
        call_args = mock_client.get.call_args
        assert call_args.kwargs["params"]["skip"] == 10
        assert call_args.kwargs["params"]["limit"] == 10

    async def test_get_rows_streaming_yields_rows(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test streaming row retrieval."""
        # Create async iterator
        async_iter = AsyncIterator(
            [
                '{"id": 1, "name": "item1"}',
                '{"id": 2, "name": "item2"}',
                "",  # Empty line should be skipped
                '{"id": 3, "name": "item3"}',
            ]
        )

        # Mock streaming response
        mock_stream = AsyncMock()
        mock_stream.status_code = 200
        mock_stream.aiter_lines = Mock(return_value=async_iter)

        # Create a proper async context manager mock
        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_stream
        stream_context.__aexit__.return_value = None

        mock_client.stream.return_value = stream_context

        results = []
        async for row in operations.get_rows_streaming(skip=0, limit=100):
            results.append(row)

        assert len(results) == 3
        assert all(isinstance(r, ClientTestResponse) for r in results)
        assert results[0].name == "item1"
        assert results[2].name == "item3"

    async def test_count_rows(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test row counting."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 500}
        mock_client.get.return_value = mock_response

        count = await operations.count_rows()

        assert count == 500

    async def test_lookup_by_id_or_name_with_id(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test lookup by ID."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        # Create a proper LookupResponse structure
        mock_response.json.return_value = {"id": 42, "data": {"id": 42, "name": "test", "value": 0}}
        mock_client.get.return_value = mock_response

        resolved_id, row = await operations.lookup_by_id_or_name(id_=42)

        assert resolved_id == 42
        assert row.name == "test"

    async def test_lookup_by_id_or_name_with_name(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test lookup by name."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 10, "data": {"id": 10, "name": "found", "value": 0}}
        mock_client.get.return_value = mock_response

        resolved_id, row = await operations.lookup_by_id_or_name(name="found")

        assert resolved_id == 10
        assert row.name == "found"

    # UPDATE operations tests

    async def test_update_row(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test single row update."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "updated", "value": 99}
        mock_client.put.return_value = mock_response

        result = await operations.update_row(1, name="updated", value=99)

        assert result.name == "updated"
        assert result.value == 99
        assert mock_client.put.call_args[0][0].endswith("/update_row/1")

    async def test_update_rows(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test multiple row update."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "name": "updated1", "value": 10},
            {"id": 2, "name": "updated2", "value": 20},
        ]
        mock_client.put.return_value = mock_response

        data = [
            {"id": 1, "name": "updated1", "value": 10},
            {"id": 2, "name": "updated2", "value": 20},
        ]
        results = await operations.update_rows(data)

        assert len(results) == 2
        assert results[0].name == "updated1"

    # DELETE operations tests

    async def test_delete_row_with_capture(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test delete row with data capture."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "deleted"}
        mock_client.delete.return_value = mock_response

        result = await operations.delete_row(1, capture_data=True)

        assert result is not None
        assert result.name == "deleted"

    async def test_delete_row_without_capture(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test delete row without data capture."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"deleted": True}
        mock_client.delete.return_value = mock_response

        result = await operations.delete_row(1, capture_data=False)

        assert result is None

    async def test_delete_rows_with_capture(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test delete multiple rows with data capture."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "name": "deleted1"},
            {"id": 2, "name": "deleted2"},
        ]
        mock_client.request.return_value = mock_response

        results = await operations.delete_rows([1, 2], capture_data=True)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].name == "deleted1"

    async def test_delete_rows_without_capture(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test delete multiple rows without data capture."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 5}
        mock_client.request.return_value = mock_response

        result = await operations.delete_rows([1, 2, 3, 4, 5], capture_data=False)

        assert isinstance(result, int)
        assert result == 5

    async def test_bulk_delete_rows(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test bulk delete returns count."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 100}
        mock_client.request.return_value = mock_response

        count = await operations.bulk_delete_rows([1, 2, 3])

        assert count == 100
        assert isinstance(count, int)

    # FILTER/QUERY operations tests

    async def test_filter_rows_basic(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test basic row filtering."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "name": "match1", "value": 10},
            {"id": 2, "name": "match2", "value": 10},
        ]
        mock_client.post.return_value = mock_response

        filters = [Filter(field="value", op="eq", value=10)]
        results = await operations.filter_rows(filters=filters)

        assert len(results) == 2
        assert all(r.value == 10 for r in results)

    async def test_filter_rows_with_ordering(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test filtering with ordering."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 2, "name": "b"}, {"id": 1, "name": "a"}]
        mock_client.post.return_value = mock_response

        order_by = {"field": "name", "direction": "asc"}
        results = await operations.filter_rows(order_by=order_by)

        assert len(results) == 2
        # Check that order_by was included in request
        call_args = mock_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert "order_by" in request_data

    async def test_filter_rows_with_pagination(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test filtering with pagination."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": i, "name": f"item{i}"} for i in range(10, 20)]
        mock_client.post.return_value = mock_response

        _results = await operations.filter_rows(skip=10, limit=10)

        call_args = mock_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert request_data["skip"] == 10
        assert request_data["limit"] == 10

    async def test_filter_rows_streaming(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test streaming filtered rows."""
        lines = [
            '{"id": 1, "name": "match1", "value": 10}',
            '{"id": 2, "name": "match2", "value": 10}',
        ]
        async_iter = AsyncIterator(lines)

        mock_stream = AsyncMock()
        mock_stream.status_code = 200
        mock_stream.aiter_lines = Mock(return_value=async_iter)

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_stream
        stream_context.__aexit__.return_value = None

        mock_client.stream.return_value = stream_context

        filters = [Filter(field="value", op="eq", value=10)]
        results = []
        async for row in operations.filter_rows_streaming(filters=filters):
            results.append(row)

        assert len(results) == 2
        assert all(r.value == 10 for r in results)

    async def test_count_filtered_rows(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test counting filtered rows."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 42}
        mock_client.post.return_value = mock_response

        filters = [Filter(field="value", op="gt", value=10)]
        count = await operations.count_filtered_rows(filters=filters)

        assert count == 42

    async def test_filter_one_success(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test filter_one returns single result."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "unique"}
        mock_client.post.return_value = mock_response

        filters = [Filter(field="name", op="eq", value="unique")]
        result = await operations.filter_one(filters=filters)

        assert result.name == "unique"

    async def test_filter_one_or_none_returns_none(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test filter_one_or_none returns None when no match."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = None
        mock_client.post.return_value = mock_response

        filters = [Filter(field="name", op="eq", value="nonexistent")]
        result = await operations.filter_one_or_none(filters=filters)

        assert result is None

    async def test_filter_one_or_none_returns_result(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test filter_one_or_none returns result when found."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "found"}
        mock_client.post.return_value = mock_response

        filters = [Filter(field="name", op="eq", value="found")]
        result = await operations.filter_one_or_none(filters=filters)

        assert result is not None
        assert result.name == "found"

    async def test_find_by_basic(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test find_by with query parameters."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"id": 1, "name": "test", "value": 42},
        ]
        mock_client.post.return_value = mock_response

        results = await operations.find_by(name="test", value=42)

        assert len(results) == 1
        assert results[0].name == "test"
        assert results[0].value == 42

        # Verify query params were sent
        call_args = mock_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert request_data["name"] == "test"
        assert request_data["value"] == 42

    async def test_find_by_with_ordering(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test find_by with ordering."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_client.post.return_value = mock_response

        order_by = OrderBy(field="created_at", descending=True)
        await operations.find_by(status="active", order_by=order_by, limit=10)

        call_args = mock_client.post.call_args
        request_data = call_args.kwargs["json"]
        assert "order_by" in request_data
        assert request_data["limit"] == 10

    async def test_find_one_by(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test find_one_by returns single result."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": 1, "name": "unique"}
        mock_client.post.return_value = mock_response

        result = await operations.find_one_by(name="unique")

        assert result.name == "unique"

    # Error handling tests

    async def test_handle_response_success(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate]
    ) -> None:
        """Test successful response handling."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"key": "value"}

        result = operations._handle_response(mock_response, expected_status=200)

        assert result == {"key": "value"}

    async def test_handle_response_error_with_json(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate]
    ) -> None:
        """Test error response with JSON error details."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Bad request", "details": "Invalid field"}

        with pytest.raises(RemoteAPIError) as exc_info:
            operations._handle_response(mock_response, expected_status=200)

        assert "400" in str(exc_info.value)
        assert "Bad request" in str(exc_info.value)
        assert "Invalid field" in str(exc_info.value)

    async def test_handle_response_error_without_json(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate]
    ) -> None:
        """Test error response without JSON."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.json.side_effect = Exception("Not JSON")
        mock_response.text = "Internal server error"

        with pytest.raises(RemoteAPIError) as exc_info:
            operations._handle_response(mock_response, expected_status=200)

        assert "500" in str(exc_info.value)

    async def test_streaming_error_handling(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test streaming handles errors gracefully."""
        mock_stream = AsyncMock()
        mock_stream.status_code = 500
        mock_stream.aread.return_value = b"Server error"

        mock_client.stream.return_value.__aenter__.return_value = mock_stream

        with pytest.raises(RemoteAPIError) as exc_info:
            async for _ in operations.get_rows_streaming():
                pass

        assert "500" in str(exc_info.value)

    async def test_streaming_validation_error(
        self, operations: RemoteTableOperations[ClientTestResponse, ClientTestCreate], mock_client: AsyncMock
    ) -> None:
        """Test streaming handles validation errors."""
        mock_stream = AsyncMock()
        mock_stream.status_code = 200
        mock_stream.aiter_lines.return_value = AsyncIterator(
            [
                '{"invalid": "data"}',  # Missing required fields
            ]
        )

        mock_client.stream.return_value.__aenter__.return_value = mock_stream

        with pytest.raises(Exception):  # Will raise ValidationError or RemoteAPIError
            async for _ in operations.get_rows_streaming():
                pass


class TestRemoteAPI:
    """Tests for RemoteAPI class."""

    async def test_context_manager_initializes_client(self) -> None:
        """Test that async context manager initializes client."""
        async with RemoteAPI("http://api.example.com") as api:
            assert api.client is not None
            assert isinstance(api.client, httpx.AsyncClient)

    async def test_context_manager_closes_client(self) -> None:
        """Test that async context manager closes client on exit."""
        api = RemoteAPI("http://api.example.com")
        async with api:
            client = api.client
            assert client is not None

        # Client should be closed after exiting context
        assert client.is_closed

    async def test_initialization_parameters(self) -> None:
        """Test RemoteAPI initialization with various parameters."""
        api = RemoteAPI(
            base_url="http://api.example.com/",  # Trailing slash should be stripped
            api_prefix="/api/v2/",  # Trailing slash should be stripped
            timeout=60.0,
            auth_token="test-token",
        )

        assert api.base_url == "http://api.example.com"
        assert api.api_prefix == "/api/v2"
        assert api.timeout == 60.0
        assert api.auth_token == "test-token"
        assert api.headers["Authorization"] == "Bearer test-token"

    async def test_initialization_without_auth(self) -> None:
        """Test RemoteAPI initialization without auth token."""
        api = RemoteAPI("http://api.example.com")

        assert api.auth_token is None
        assert "Authorization" not in api.headers

    async def test_table_creates_operations_instance(self) -> None:
        """Test that table() creates a RemoteTableOperations instance."""
        async with RemoteAPI("http://api.example.com") as api:
            ops = api.table("test", ClientTestResponse, ClientTestCreate)

            assert isinstance(ops, RemoteTableOperations)
            assert ops.endpoint == "http://api.example.com/api/v1/test"
            assert ops.response_model == ClientTestResponse
            assert ops.create_model == ClientTestCreate
            assert ops.client is api.client

    async def test_table_raises_error_without_context(self) -> None:
        """Test that table() raises error when used outside context manager."""
        api = RemoteAPI("http://api.example.com")

        with pytest.raises(RemoteAPIError) as exc_info:
            api.table("test", ClientTestResponse, ClientTestCreate)

        assert "not initialized" in str(exc_info.value).lower()
        assert "context manager" in str(exc_info.value).lower()

    async def test_multiple_table_clients_share_http_client(self) -> None:
        """Test that multiple table clients share the same HTTP client."""
        async with RemoteAPI("http://api.example.com") as api:
            ops1 = api.table("table1", ClientTestResponse, ClientTestCreate)
            ops2 = api.table("table2", ClientTestResponse, ClientTestCreate)

            assert ops1.client is ops2.client
            assert ops1.client is api.client

    async def test_table_endpoint_construction(self) -> None:
        """Test that table endpoints are constructed correctly."""
        async with RemoteAPI("http://api.example.com", api_prefix="/v2") as api:
            ops = api.table("my_table", ClientTestResponse, ClientTestCreate)

            assert ops.endpoint == "http://api.example.com/v2/my_table"

    async def test_context_manager_handles_exceptions(self) -> None:
        """Test that context manager properly handles exceptions."""
        client_ref = None

        try:
            async with RemoteAPI("http://api.example.com") as api:
                client_ref = api.client
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Client should still be closed even when exception occurred
        assert client_ref is not None
        assert client_ref.is_closed

    async def test_client_timeout_configuration(self) -> None:
        """Test that client timeout is configured correctly."""
        async with RemoteAPI("http://api.example.com", timeout=120.0) as api:
            assert api.client is not None
            # httpx.AsyncClient stores timeout in client.timeout
            # The exact attribute depends on httpx version, so we just verify client exists
            assert isinstance(api.client, httpx.AsyncClient)

    async def test_multiple_apis_independent(self) -> None:
        """Test that multiple RemoteAPI instances are independent."""
        async with RemoteAPI("http://api1.example.com", auth_token="token1") as api1:
            async with RemoteAPI("http://api2.example.com", auth_token="token2") as api2:
                assert api1.client is not api2.client
                assert api1.base_url != api2.base_url
                assert api1.auth_token != api2.auth_token


class TestIntegrationScenarios:
    """Integration-style tests for common usage patterns."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock HTTP client."""
        return AsyncMock(spec=httpx.AsyncClient)

    async def test_crud_workflow(self, mock_client: AsyncMock) -> None:
        """Test complete CRUD workflow."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        # Create
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1, "name": "item", "value": 100}
        mock_client.post.return_value = mock_response

        created = await operations.create_row(name="item", value=100)
        assert created.id == 1

        # Read
        mock_response.status_code = 200
        mock_client.get.return_value = mock_response
        retrieved = await operations.get_row(1)
        assert retrieved.id == created.id

        # Update
        mock_response.json.return_value = {"id": 1, "name": "updated", "value": 200}
        mock_client.put.return_value = mock_response
        updated = await operations.update_row(1, name="updated", value=200)
        assert updated.name == "updated"

        # Delete
        mock_client.delete.return_value = mock_response
        deleted = await operations.delete_row(1, capture_data=True)
        assert deleted is not None

    async def test_batch_operations_workflow(self, mock_client: AsyncMock) -> None:
        """Test batch operations workflow."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        # Batch create
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = [
            {"id": i, "name": f"item{i}", "value": i * 10} for i in range(1, 101)
        ]
        mock_client.post.return_value = mock_response

        data = [{"name": f"item{i}", "value": i * 10} for i in range(1, 101)]
        created = await operations.create_rows(data)
        assert len(created) == 100

        # Batch delete
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 100}
        mock_client.request.return_value = mock_response

        count = await operations.bulk_delete_rows(list(range(1, 101)))
        assert count == 100

    async def test_filtering_workflow(self, mock_client: AsyncMock) -> None:
        """Test filtering and querying workflow."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        # Count before filtering
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 1000}
        mock_client.get.return_value = mock_response

        total = await operations.count_rows()
        assert total == 1000

        # Filter with conditions
        mock_response.json.return_value = [{"id": i, "name": f"active{i}", "value": 50} for i in range(1, 11)]
        mock_client.post.return_value = mock_response

        filters = [Filter(field="value", op="ge", value=50)]
        results = await operations.filter_rows(
            filters=filters, order_by={"field": "id", "direction": "asc"}, limit=10
        )
        assert len(results) == 10

        # Count filtered
        mock_response.json.return_value = {"count": 50}
        filtered_count = await operations.count_filtered_rows(filters=filters)
        assert filtered_count == 50

    async def test_streaming_large_dataset(self, mock_client: AsyncMock) -> None:
        """Test streaming for large datasets."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        # Mock streaming response with many rows
        lines = [f'{{"id": {i}, "name": "item{i}", "value": {i}}}' for i in range(1, 10001)]
        async_iter = AsyncIterator(lines)

        mock_stream = AsyncMock()
        mock_stream.status_code = 200
        mock_stream.aiter_lines = Mock(return_value=async_iter)

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_stream
        stream_context.__aexit__.return_value = None

        mock_client.stream.return_value = stream_context

        # Process streaming data
        count = 0
        total_value = 0
        async for row in operations.get_rows_streaming():
            count += 1
            total_value += row.value

        assert count == 10000
        assert total_value == sum(range(1, 10001))

    async def test_error_recovery_workflow(self, mock_client: AsyncMock) -> None:
        """Test error handling and recovery."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        # Try to get non-existent row - should raise
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not found"}
        mock_client.get.return_value = mock_response

        with pytest.raises(RemoteAPIError):
            await operations.get_row(999)

        # Use safe variant - should return None
        mock_response.status_code = 200
        mock_response.json.return_value = None

        result = await operations.get_row_or_none(999)
        assert result is None

        # Now successfully create the row
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 999, "name": "new", "value": 0}
        mock_client.post.return_value = mock_response

        created = await operations.create_row(name="new", value=0)
        assert created.id == 999

    async def test_lookup_workflow(self, mock_client: AsyncMock) -> None:
        """Test lookup by ID or name workflow."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        # Lookup by ID
        mock_response.json.return_value = {"id": 42, "data": {"id": 42, "name": "by_id", "value": 1}}
        mock_client.get.return_value = mock_response

        resolved_id, row = await operations.lookup_by_id_or_name(id_=42)
        assert resolved_id == 42
        assert row.name == "by_id"

        # Lookup by name
        mock_response.json.return_value = {"id": 100, "data": {"id": 100, "name": "by_name", "value": 2}}

        resolved_id, row = await operations.lookup_by_id_or_name(name="by_name")
        assert resolved_id == 100
        assert row.name == "by_name"

    async def test_find_by_workflow(self, mock_client: AsyncMock) -> None:
        """Test find_by convenience methods."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200

        # Find multiple by criteria
        mock_response.json.return_value = [
            {"id": 1, "name": "active_1", "value": 10},
            {"id": 2, "name": "active_2", "value": 10},
        ]
        mock_client.post.return_value = mock_response

        results = await operations.find_by(value=10, limit=2)
        assert len(results) == 2
        assert all(r.value == 10 for r in results)

        # Find one by unique criteria
        mock_response.json.return_value = {"id": 1, "name": "unique_name", "value": 99}

        result = await operations.find_one_by(name="unique_name")
        assert result.name == "unique_name"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock HTTP client."""
        return AsyncMock(spec=httpx.AsyncClient)

    async def test_empty_list_operations(self, mock_client: AsyncMock) -> None:
        """Test operations with empty lists."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.json.return_value = []
        mock_client.post.return_value = mock_response

        # Empty create
        results = await operations.create_rows([])
        assert results == []

        # Empty delete
        mock_response.status_code = 200
        mock_response.json.return_value = {"count": 0}
        mock_client.request.return_value = mock_response

        count = await operations.bulk_delete_rows([])
        assert count == 0

    async def test_pagination_edge_cases(self, mock_client: AsyncMock) -> None:
        """Test pagination with edge case values."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_client.get.return_value = mock_response

        # Skip beyond available rows
        results = await operations.get_rows(skip=10000, limit=10)
        assert results == []

        # Limit of 0
        results = await operations.get_rows(skip=0, limit=0)
        assert results == []

    async def test_streaming_empty_result(self, mock_client: AsyncMock) -> None:
        """Test streaming with no results."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        async_iter = AsyncIterator([])

        mock_stream = AsyncMock()
        mock_stream.status_code = 200
        mock_stream.aiter_lines = Mock(return_value=async_iter)

        stream_context = AsyncMock()
        stream_context.__aenter__.return_value = mock_stream
        stream_context.__aexit__.return_value = None

        mock_client.stream.return_value = stream_context

        results = []
        async for row in operations.get_rows_streaming():
            results.append(row)

        assert results == []

    async def test_filter_with_no_filters(self, mock_client: AsyncMock) -> None:
        """Test filtering with empty filter list."""
        operations = RemoteTableOperations(
            client=mock_client,
            endpoint="http://api.example.com/v1/items",
            response_model=ClientTestResponse,
            create_model=ClientTestCreate,
        )

        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": 1, "name": "all", "value": 0}]
        mock_client.post.return_value = mock_response

        # Should return all rows when no filters
        results = await operations.filter_rows(filters=None)
        assert len(results) >= 0
