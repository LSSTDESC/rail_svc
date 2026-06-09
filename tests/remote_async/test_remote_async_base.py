"""Unit tests for AsyncRemoteOperations class."""

from __future__ import annotations

import warnings
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from rail_svc.client.base import RemoteAPI, RemoteTableOperations
from rail_svc.remote_async.base import AsyncRemoteOperations, with_client


# Test models
class RemoteAsyncTestResponse(BaseModel):
    """Test response model."""

    id: int
    name: str
    value: int = 0


class RemoteAsyncTestCreate(BaseModel):
    """Test create model."""

    name: str
    value: int = 0


class TestWithClientDecorator:
    """Tests for the with_client decorator."""

    async def test_decorator_injects_client(self) -> None:
        """Test that decorator properly injects client."""
        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_row.return_value = RemoteAsyncTestResponse(id=1, name="test", value=42)

        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )
        ops._client = mock_client

        @with_client
        async def test_method(
            self: AsyncRemoteOperations,
            client: RemoteTableOperations,
            row_id: int,
        ) -> RemoteAsyncTestResponse:
            return await client.get_row(row_id)

        result = await test_method(ops, 1)

        assert result.id == 1
        assert result.name == "test"
        mock_client.get_row.assert_called_once_with(1)

    async def test_decorator_passes_args_and_kwargs(self) -> None:
        """Test that decorator passes through args and kwargs."""
        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.filter_rows.return_value = [RemoteAsyncTestResponse(id=1, name="test")]

        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )
        ops._client = mock_client

        @with_client
        async def test_method(
            self: AsyncRemoteOperations,
            client: RemoteTableOperations,
            *args,
            **kwargs,
        ) -> list[RemoteAsyncTestResponse]:
            return await client.filter_rows(*args, **kwargs)

        result = await test_method(ops, filters=[{"field": "name", "operator": "eq", "value": "test"}])

        assert len(result) == 1
        mock_client.filter_rows.assert_called_once()


class TestAsyncRemoteOperationsInit:
    """Tests for AsyncRemoteOperations initialization."""

    def test_initialization_with_all_params(self) -> None:
        """Test initialization with all parameters."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test_table",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
            api_prefix="/v2",
            timeout=60.0,
            auth_token="test-token",
        )

        assert ops.base_url == "http://api.example.com"
        assert ops.table_name == "test_table"
        assert ops.response_model == RemoteAsyncTestResponse
        assert ops.create_model == RemoteAsyncTestCreate
        assert ops.api_prefix == "/v2"
        assert ops.timeout == 60.0
        assert ops.auth_token == "test-token"
        assert ops._api is None
        assert ops._client is None
        assert ops._owns_api is False
        assert ops._has_warned is False

    def test_initialization_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        assert ops.api_prefix == "/api/v1"
        assert ops.timeout == 30.0
        assert ops.auth_token is None


class TestAsyncRemoteOperationsContextManager:
    """Tests for AsyncRemoteOperations as async context manager."""

    async def test_context_manager_initializes_api(self) -> None:
        """Test that entering context manager initializes API."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        assert ops._api is None
        assert ops._client is None

        async with ops:
            assert ops._api is not None
            assert isinstance(ops._api, RemoteAPI)
            assert ops._client is not None
            assert isinstance(ops._client, RemoteTableOperations)
            assert ops._owns_api is True

    async def test_context_manager_cleans_up(self) -> None:
        """Test that exiting context manager cleans up resources."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        async with ops:
            api_ref = ops._api
            assert api_ref is not None

        assert ops._api is None
        assert ops._client is None
        assert ops._owns_api is False
        # API should be closed
        assert api_ref.client.is_closed

    async def test_context_manager_returns_self(self) -> None:
        """Test that context manager returns self."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        async with ops as returned:
            assert returned is ops

    async def test_context_manager_handles_exceptions(self) -> None:
        """Test that context manager handles exceptions properly."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        api_ref = None
        try:
            async with ops:
                api_ref = ops._api
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still clean up
        assert ops._api is None
        assert ops._client is None
        assert api_ref is not None
        assert api_ref.client.is_closed

    async def test_client_configuration(self) -> None:
        """Test that client is properly configured."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test_table",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
            api_prefix="/v2",
        )

        async with ops:
            assert ops._client.endpoint == "http://api.example.com/v2/test_table"
            assert ops._client.response_model == RemoteAsyncTestResponse
            assert ops._client.create_model == RemoteAsyncTestCreate


class TestGetClient:
    """Tests for get_client method."""

    async def test_get_client_with_context_manager(self) -> None:
        """Test get_client returns existing client when in context manager."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        async with ops:
            client1 = await ops.get_client()
            client2 = await ops.get_client()

            # Should return the same client
            assert client1 is client2
            assert client1 is ops._client

    async def test_get_client_without_context_manager_warns(self) -> None:
        """Test get_client warns when not using context manager."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        # Mock the actual operation to avoid needing real API initialization
        mock_client = AsyncMock(spec=RemoteTableOperations)

        with patch.object(RemoteAPI, "table", return_value=mock_client):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                _client = await ops.get_client()

                assert len(w) == 1
                assert issubclass(w[0].category, ResourceWarning)
                assert "temporary client" in str(w[0].message).lower()
                assert "context manager" in str(w[0].message).lower()

    async def test_get_client_warns_only_once(self) -> None:
        """Test that warning is only issued once."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)

        with patch.object(RemoteAPI, "table", return_value=mock_client):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await ops.get_client()
                await ops.get_client()
                await ops.get_client()

                # Should only warn once
                warning_count = sum(1 for warning in w if issubclass(warning.category, ResourceWarning))
                assert warning_count == 1

    async def test_get_client_creates_temporary_client(self) -> None:
        """Test that _get_client creates a temporary client when needed."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.endpoint = "http://api.example.com/api/v1/test"

        with patch.object(RemoteAPI, "table", return_value=mock_client):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                client = await ops.get_client()

            assert isinstance(client, AsyncMock)
            assert client.endpoint == "http://api.example.com/api/v1/test"


class TestCRUDOperations:
    """Tests for CRUD operation methods."""

    async def test_create_row(self) -> None:
        """Test create_row delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.create_row.return_value = RemoteAsyncTestResponse(id=1, name="created", value=100)
        ops._client = mock_client

        result = await ops.create_row(name="created", value=100)

        assert result.id == 1
        assert result.name == "created"
        mock_client.create_row.assert_called_once_with(name="created", value=100)

    async def test_create_rows(self) -> None:
        """Test create_rows delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.create_rows.return_value = [
            RemoteAsyncTestResponse(id=1, name="row1"),
            RemoteAsyncTestResponse(id=2, name="row2"),
        ]
        ops._client = mock_client

        data = [{"name": "row1"}, {"name": "row2"}]
        results = await ops.create_rows(data)

        assert len(results) == 2
        mock_client.create_rows.assert_called_once_with(data)

    async def test_get_row(self) -> None:
        """Test get_row delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_row.return_value = RemoteAsyncTestResponse(id=42, name="test")
        ops._client = mock_client

        result = await ops.get_row(42)

        assert result.id == 42
        mock_client.get_row.assert_called_once_with(42)

    async def test_get_row_or_none(self) -> None:
        """Test get_row_or_none delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_row_or_none.return_value = None
        ops._client = mock_client

        result = await ops.get_row_or_none(999)

        assert result is None
        mock_client.get_row_or_none.assert_called_once_with(999)

    async def test_get_rows(self) -> None:
        """Test get_rows delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_rows.return_value = [RemoteAsyncTestResponse(id=i, name=f"row{i}") for i in range(5)]
        ops._client = mock_client

        results = await ops.get_rows(skip=10, limit=5)

        assert len(results) == 5
        mock_client.get_rows.assert_called_once_with(skip=10, limit=5)

    async def test_count_rows(self) -> None:
        """Test count_rows delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.count_rows.return_value = 100
        ops._client = mock_client

        count = await ops.count_rows()

        assert count == 100
        mock_client.count_rows.assert_called_once()

    async def test_update_row(self) -> None:
        """Test update_row delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.update_row.return_value = RemoteAsyncTestResponse(id=1, name="updated", value=200)
        ops._client = mock_client

        result = await ops.update_row(1, name="updated", value=200)

        assert result.name == "updated"
        mock_client.update_row.assert_called_once_with(1, name="updated", value=200)

    async def test_delete_row(self) -> None:
        """Test delete_row delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.delete_row.return_value = RemoteAsyncTestResponse(id=1, name="deleted")
        ops._client = mock_client

        result = await ops.delete_row(1, capture_data=True)

        assert result is not None
        assert result.name == "deleted"
        mock_client.delete_row.assert_called_once_with(1, capture_data=True)

    async def test_bulk_delete_rows(self) -> None:
        """Test bulk_delete_rows delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.bulk_delete_rows.return_value = 50
        ops._client = mock_client

        count = await ops.bulk_delete_rows([1, 2, 3, 4, 5])

        assert count == 50
        mock_client.bulk_delete_rows.assert_called_once_with([1, 2, 3, 4, 5])


class TestFilterOperations:
    """Tests for filter and query operations."""

    async def test_filter_rows(self) -> None:
        """Test filter_rows delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.filter_rows.return_value = [
            RemoteAsyncTestResponse(id=1, name="match1", value=10),
            RemoteAsyncTestResponse(id=2, name="match2", value=10),
        ]
        ops._client = mock_client

        filters = [{"field": "value", "operator": "eq", "value": 10}]
        results = await ops.filter_rows(filters=filters)

        assert len(results) == 2
        mock_client.filter_rows.assert_called_once_with(filters=filters)

    async def test_count_filtered_rows(self) -> None:
        """Test count_filtered_rows delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.count_filtered_rows.return_value = 42
        ops._client = mock_client

        filters = [{"field": "value", "operator": "gt", "value": 5}]
        count = await ops.count_filtered_rows(filters=filters)

        assert count == 42
        mock_client.count_filtered_rows.assert_called_once_with(filters=filters)

    async def test_filter_one(self) -> None:
        """Test filter_one delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.filter_one.return_value = RemoteAsyncTestResponse(id=1, name="unique")
        ops._client = mock_client

        filters = [{"field": "name", "operator": "eq", "value": "unique"}]
        result = await ops.filter_one(filters=filters)

        assert result.name == "unique"
        mock_client.filter_one.assert_called_once_with(filters=filters)

    async def test_filter_one_or_none(self) -> None:
        """Test filter_one_or_none delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.filter_one_or_none.return_value = None
        ops._client = mock_client

        filters = [{"field": "name", "operator": "eq", "value": "nonexistent"}]
        result = await ops.filter_one_or_none(filters=filters)

        assert result is None
        mock_client.filter_one_or_none.assert_called_once_with(filters=filters)

    async def test_find_by(self) -> None:
        """Test find_by delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.find_by.return_value = [RemoteAsyncTestResponse(id=1, name="test", value=42)]
        ops._client = mock_client

        results = await ops.find_by(name="test", value=42)

        assert len(results) == 1
        mock_client.find_by.assert_called_once_with(name="test", value=42)

    async def test_find_one_by(self) -> None:
        """Test find_one_by delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.find_one_by.return_value = RemoteAsyncTestResponse(id=1, name="unique")
        ops._client = mock_client

        result = await ops.find_one_by(name="unique")

        assert result.name == "unique"
        mock_client.find_one_by.assert_called_once_with(name="unique")

    async def test_lookup_by_id_or_name(self) -> None:
        """Test lookup_by_id_or_name delegates to client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.lookup_by_id_or_name.return_value = (42, RemoteAsyncTestResponse(id=42, name="found"))
        ops._client = mock_client

        resolved_id, row = await ops.lookup_by_id_or_name(id_=42)

        assert resolved_id == 42
        assert row.name == "found"
        mock_client.lookup_by_id_or_name.assert_called_once_with(id_=42)


class TestIntegrationPatterns:
    """Integration-style tests for common usage patterns."""

    async def test_context_manager_multiple_operations(self) -> None:
        """Test using context manager for multiple operations."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.create_row.return_value = RemoteAsyncTestResponse(id=1, name="created")
        mock_client.get_row.return_value = RemoteAsyncTestResponse(id=1, name="created")
        mock_client.update_row.return_value = RemoteAsyncTestResponse(id=1, name="updated")
        mock_client.delete_row.return_value = RemoteAsyncTestResponse(id=1, name="updated")

        async with ops:
            ops._client = mock_client

            # Multiple operations
            _created = await ops.create_row(name="created")
            _retrieved = await ops.get_row(1)
            _updated = await ops.update_row(1, name="updated")
            _deleted = await ops.delete_row(1)

            # All operations should have used the same client
            assert mock_client.create_row.call_count == 1
            assert mock_client.get_row.call_count == 1
            assert mock_client.update_row.call_count == 1
            assert mock_client.delete_row.call_count == 1

    async def test_single_operation_without_context_manager(self) -> None:
        """Test using class for single operation without context manager."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        # Mock the _get_client to avoid actual API calls
        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_row.return_value = RemoteAsyncTestResponse(id=1, name="test")

        with patch.object(ops, "get_client", return_value=mock_client):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = await ops.get_row(1)

        assert result.id == 1
        assert result.name == "test"

    async def test_batch_operations(self) -> None:
        """Test batch operations."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.create_rows_batched.return_value = [
            RemoteAsyncTestResponse(id=i, name=f"row{i}") for i in range(1, 101)
        ]
        mock_client.bulk_delete_rows.return_value = 100

        async with ops:
            ops._client = mock_client

            # Create many rows
            data = [{"name": f"row{i}"} for i in range(1, 101)]
            created = await ops.create_rows_batched(data, batch_size=50)
            assert len(created) == 100

            # Delete many rows
            count = await ops.bulk_delete_rows(list(range(1, 101)))
            assert count == 100

    async def test_filter_workflow(self) -> None:
        """Test filtering workflow."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.count_rows.return_value = 1000
        mock_client.filter_rows.return_value = [
            RemoteAsyncTestResponse(id=i, name=f"active{i}", value=50) for i in range(1, 11)
        ]
        mock_client.count_filtered_rows.return_value = 50

        async with ops:
            ops._client = mock_client

            total = await ops.count_rows()
            assert total == 1000

            filters = [{"field": "value", "operator": "gte", "value": 50}]
            results = await ops.filter_rows(filters=filters, limit=10)
            assert len(results) == 10

            filtered_count = await ops.count_filtered_rows(filters=filters)
            assert filtered_count == 50


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_multiple_context_manager_entries(self) -> None:
        """Test that object can be used as context manager multiple times."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        # First use
        async with ops:
            assert ops._client is not None
        assert ops._client is None

        # Second use
        async with ops:
            assert ops._client is not None
        assert ops._client is None

    async def test_nested_operations_share_client(self) -> None:
        """Test that nested operation calls share the same client."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        call_count = 0
        original_get_client = ops.get_client

        async def counting_get_client():
            nonlocal call_count
            call_count += 1
            return await original_get_client()

        ops.get_client = counting_get_client

        async with ops:
            mock_client = AsyncMock(spec=RemoteTableOperations)
            mock_client.get_row.return_value = RemoteAsyncTestResponse(id=1, name="test")
            ops._client = mock_client

            await ops.get_row(1)
            await ops.get_row(2)
            await ops.get_row(3)

            # _get_client should be called for each operation
            assert call_count == 3
            # But they should all get the same client
            assert mock_client.get_row.call_count == 3

    async def test_operations_with_no_args(self) -> None:
        """Test operations that take no arguments."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.count_rows.return_value = 100

        async with ops:
            ops._client = mock_client
            count = await ops.count_rows()

        assert count == 100
        mock_client.count_rows.assert_called_once_with()

    async def test_operations_with_mixed_args_kwargs(self) -> None:
        """Test operations with both positional and keyword arguments."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.update_row.return_value = RemoteAsyncTestResponse(id=1, name="updated", value=99)

        async with ops:
            ops._client = mock_client
            result = await ops.update_row(1, name="updated", value=99)

        assert result.value == 99
        mock_client.update_row.assert_called_once_with(1, name="updated", value=99)

    async def test_client_configuration_propagation(self) -> None:
        """Test that all configuration is properly propagated."""
        ops = AsyncRemoteOperations(
            base_url="http://custom.api.com",
            table_name="custom_table",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
            api_prefix="/v3",
            timeout=120.0,
            auth_token="custom-token",
        )

        async with ops:
            # Check API configuration
            assert ops._api.base_url == "http://custom.api.com"
            assert ops._api.api_prefix == "/v3"
            assert ops._api.timeout == 120.0
            assert ops._api.auth_token == "custom-token"

            # Check client configuration
            assert ops._client.endpoint == "http://custom.api.com/v3/custom_table"
            assert ops._client.response_model == RemoteAsyncTestResponse
            assert ops._client.create_model == RemoteAsyncTestCreate


class TestWarningBehavior:
    """Tests for warning behavior when not using context manager."""

    async def test_warning_message_content(self) -> None:
        """Test that warning message contains helpful information."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)

        with patch.object(RemoteAPI, "table", return_value=mock_client):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                await ops.get_client()

                assert len(w) == 1
                message = str(w[0].message)
                assert "temporary client" in message.lower()
                assert "AsyncRemoteOperations" in message
                assert "context manager" in message.lower()
                assert "async with" in message.lower()

    async def test_no_warning_with_context_manager(self) -> None:
        """Test that no warning is issued when using context manager."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            async with ops:
                await ops.get_client()
                await ops.get_client()
                await ops.get_client()

            # Should have no ResourceWarnings
            resource_warnings = [warning for warning in w if issubclass(warning.category, ResourceWarning)]
            assert len(resource_warnings) == 0

    async def test_warning_state_persists(self) -> None:
        """Test that warning state persists across calls."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)

        with patch.object(RemoteAPI, "table", return_value=mock_client):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # First call - should warn
                await ops.get_client()
                first_count = len(w)

                # Subsequent calls - should not warn
                await ops.get_client()
                await ops.get_client()

                resource_warnings = [
                    warning for warning in w if issubclass(warning.category, ResourceWarning)
                ]
                assert len(resource_warnings) == 1
                assert first_count == 1


class TestTypeCorrectness:
    """Tests to verify type correctness and generics."""

    async def test_response_model_type(self) -> None:
        """Test that operations return correct response model type."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_row.return_value = RemoteAsyncTestResponse(id=1, name="test")

        async with ops:
            ops._client = mock_client
            result = await ops.get_row(1)

        # Should be the correct type
        assert isinstance(result, RemoteAsyncTestResponse)
        assert hasattr(result, "id")
        assert hasattr(result, "name")
        assert hasattr(result, "value")

    async def test_list_operations_return_list(self) -> None:
        """Test that list operations return lists of correct type."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_rows.return_value = [
            RemoteAsyncTestResponse(id=1, name="test1"),
            RemoteAsyncTestResponse(id=2, name="test2"),
        ]

        async with ops:
            ops._client = mock_client
            results = await ops.get_rows()

        assert isinstance(results, list)
        assert all(isinstance(r, RemoteAsyncTestResponse) for r in results)

    async def test_optional_return_types(self) -> None:
        """Test that optional return types work correctly."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)

        # Test None return
        mock_client.get_row_or_none.return_value = None
        async with ops:
            ops._client = mock_client
            result = await ops.get_row_or_none(999)
        assert result is None

        # Test value return
        mock_client.get_row_or_none.return_value = RemoteAsyncTestResponse(id=1, name="test")
        async with ops:
            ops._client = mock_client
            result = await ops.get_row_or_none(1)
        assert result is not None
        assert isinstance(result, RemoteAsyncTestResponse)


class TestBatchOperations:
    """Tests for batch operation methods."""

    async def test_create_rows_batched(self) -> None:
        """Test create_rows_batched with batch_size parameter."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.create_rows_batched.return_value = [
            RemoteAsyncTestResponse(id=i, name=f"row{i}") for i in range(1, 101)
        ]

        async with ops:
            ops._client = mock_client
            data = [{"name": f"row{i}"} for i in range(1, 101)]
            results = await ops.create_rows_batched(data, batch_size=50)

        assert len(results) == 100
        mock_client.create_rows_batched.assert_called_once_with(data, batch_size=50)

    async def test_bulk_insert_rows(self) -> None:
        """Test bulk_insert_rows returns count."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.bulk_insert_rows.return_value = 1000

        async with ops:
            ops._client = mock_client
            data = [{"name": f"row{i}"} for i in range(1000)]
            count = await ops.bulk_insert_rows(data)

        assert count == 1000
        assert isinstance(count, int)

    async def test_update_rows(self) -> None:
        """Test update_rows with multiple updates."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.update_rows.return_value = [
            RemoteAsyncTestResponse(id=1, name="updated1", value=10),
            RemoteAsyncTestResponse(id=2, name="updated2", value=20),
        ]

        async with ops:
            ops._client = mock_client
            updates = [
                {"id": 1, "name": "updated1", "value": 10},
                {"id": 2, "name": "updated2", "value": 20},
            ]
            results = await ops.update_rows(updates)

        assert len(results) == 2
        assert results[0].name == "updated1"
        assert results[1].name == "updated2"

    async def test_delete_rows_with_capture(self) -> None:
        """Test delete_rows with data capture."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.delete_rows.return_value = [
            RemoteAsyncTestResponse(id=1, name="deleted1"),
            RemoteAsyncTestResponse(id=2, name="deleted2"),
        ]

        async with ops:
            ops._client = mock_client
            results = await ops.delete_rows([1, 2], capture_data=True)

        assert isinstance(results, list)
        assert len(results) == 2

    async def test_delete_rows_without_capture(self) -> None:
        """Test delete_rows without data capture returns count."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.delete_rows.return_value = 5

        async with ops:
            ops._client = mock_client
            result = await ops.delete_rows([1, 2, 3, 4, 5], capture_data=False)

        assert isinstance(result, int)
        assert result == 5


class TestLookupOperations:
    """Tests for lookup operations."""

    async def test_get_row_by_name(self) -> None:
        """Test get_row_by_name lookup."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.get_row_by_name.return_value = RemoteAsyncTestResponse(id=1, name="unique_name")

        async with ops:
            ops._client = mock_client
            result = await ops.get_row_by_name("unique_name")

        assert result.name == "unique_name"
        mock_client.get_row_by_name.assert_called_once_with("unique_name")

    async def test_lookup_by_id_or_name_with_id(self) -> None:
        """Test lookup_by_id_or_name with ID parameter."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.lookup_by_id_or_name.return_value = (42, RemoteAsyncTestResponse(id=42, name="test"))

        async with ops:
            ops._client = mock_client
            resolved_id, row = await ops.lookup_by_id_or_name(id_=42)

        assert resolved_id == 42
        assert row.id == 42
        mock_client.lookup_by_id_or_name.assert_called_once_with(id_=42)

    async def test_lookup_by_id_or_name_with_name(self) -> None:
        """Test lookup_by_id_or_name with name parameter."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        mock_client = AsyncMock(spec=RemoteTableOperations)
        mock_client.lookup_by_id_or_name.return_value = (10, RemoteAsyncTestResponse(id=10, name="found"))

        async with ops:
            ops._client = mock_client
            resolved_id, row = await ops.lookup_by_id_or_name(name="found")

        assert resolved_id == 10
        assert row.name == "found"
        mock_client.lookup_by_id_or_name.assert_called_once_with(name="found")


class TestResourceManagement:
    """Tests for proper resource management."""

    async def test_api_cleanup_on_exception(self) -> None:
        """Test that API is cleaned up even when exception occurs."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        api_ref = None
        try:
            async with ops:
                api_ref = ops._api
                raise RuntimeError("Test error")
        except RuntimeError:
            pass

        # API should be closed
        assert api_ref is not None
        assert api_ref.client.is_closed
        # Internal state should be cleaned
        assert ops._api is None
        assert ops._client is None
        assert ops._owns_api is False

    async def test_only_owner_closes_api(self) -> None:
        """Test that only the context manager owner closes the API."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        # Manually set API without owning it
        external_api = RemoteAPI(
            base_url="http://api.example.com",
            api_prefix="/api/v1",
        )
        await external_api.__aenter__()

        ops._api = external_api
        ops._client = external_api.table("test", RemoteAsyncTestResponse, RemoteAsyncTestCreate)
        ops._owns_api = False  # Not owned by AsyncRemoteOperations

        # Exit should not close the API
        await ops.__aexit__(None, None, None)

        # API should still be open
        assert not external_api.client.is_closed

        # Clean up
        await external_api.__aexit__(None, None, None)

    async def test_reentrant_context_manager(self) -> None:
        """Test that context manager can be used multiple times sequentially."""
        ops = AsyncRemoteOperations(
            base_url="http://api.example.com",
            table_name="test",
            response_model=RemoteAsyncTestResponse,
            create_model=RemoteAsyncTestCreate,
        )

        # First entry/exit
        async with ops as ops1:
            api1 = ops._api
            assert ops1 is ops
            assert api1 is not None

        assert ops._api is None

        # Second entry/exit
        async with ops as ops2:
            api2 = ops._api
            assert ops2 is ops
            assert api2 is not None
            # Should be a new API instance
            assert api2 is not api1

        assert ops._api is None
