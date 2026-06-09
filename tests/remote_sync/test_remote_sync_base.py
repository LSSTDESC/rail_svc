"""Unit tests for sync_wrapper decorator and SyncRemoteOperations."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from rail_svc.remote_async.base import AsyncRemoteOperations
from rail_svc.remote_sync.base import SyncRemoteOperations


# Test models
class SyncTestResponse(BaseModel):
    """Test response model."""

    id: int
    name: str
    value: int = 0


class SyncTestCreate(BaseModel):
    """Test create model."""

    name: str
    value: int = 0


@pytest.fixture
def mock_async_ops():
    """Create a mock AsyncRemoteOperations with async context manager support."""
    mock = AsyncMock(spec=AsyncRemoteOperations)

    # Make it support async context manager
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=None)

    return mock


@pytest.fixture
def sync_ops(mock_async_ops):
    """Create SyncRemoteOperations with mocked async ops."""
    return SyncRemoteOperations(mock_async_ops)


class TestSyncWrapperDecorator:
    """Tests for the sync_wrapper decorator."""

    def test_decorator_runs_async_in_sync_context(self, sync_ops, mock_async_ops):
        """Test that decorator runs async method synchronously."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        result = sync_ops.get_row(1)

        assert result.id == 1
        assert result.name == "test"
        mock_async_ops.get_row.assert_called_once_with(1)

    def test_decorator_preserves_function_name(self, sync_ops):
        """Test that decorator preserves the function name."""
        assert sync_ops.get_row.__name__ == "get_row"
        assert sync_ops.create_row.__name__ == "create_row"
        assert sync_ops.update_row.__name__ == "update_row"

    def test_decorator_uses_context_manager(self, mock_async_ops):
        """Test that decorator enters/exits async context manager."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")
        sync_ops = SyncRemoteOperations(mock_async_ops)

        sync_ops.get_row(1)

        # Verify context manager was used
        mock_async_ops.__aenter__.assert_called_once()
        mock_async_ops.__aexit__.assert_called_once()

    def test_decorator_passes_args_and_kwargs(self, sync_ops, mock_async_ops):
        """Test that decorator passes through args and kwargs."""
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="updated", value=99)

        result = sync_ops.update_row(1, name="updated", value=99)

        assert result.value == 99
        mock_async_ops.update_row.assert_called_once_with(1, name="updated", value=99)

    def test_decorator_propagates_exceptions(self, sync_ops, mock_async_ops):
        """Test that decorator propagates exceptions from async method."""
        mock_async_ops.get_row.side_effect = ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            sync_ops.get_row(1)

        # Context manager should still have been called
        mock_async_ops.__aenter__.assert_called_once()
        mock_async_ops.__aexit__.assert_called_once()


class TestSyncCRUDOperations:
    """Tests for sync CRUD operations."""

    def test_create_row_calls_async_version(self, sync_ops, mock_async_ops):
        """Test create_row calls async version."""
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="created", value=100)

        result = sync_ops.create_row(name="created", value=100)

        assert result.id == 1
        mock_async_ops.create_row.assert_called_once_with(name="created", value=100)

    def test_get_row_calls_async_version(self, sync_ops, mock_async_ops):
        """Test get_row calls async version."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=42, name="test")

        result = sync_ops.get_row(42)

        assert result.id == 42
        mock_async_ops.get_row.assert_called_once_with(42)

    def test_update_row_calls_async_version(self, sync_ops, mock_async_ops):
        """Test update_row calls async version."""
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="updated", value=200)

        result = sync_ops.update_row(1, name="updated", value=200)

        assert result.name == "updated"
        mock_async_ops.update_row.assert_called_once_with(1, name="updated", value=200)

    def test_delete_row_calls_async_version(self, sync_ops, mock_async_ops):
        """Test delete_row calls async version."""
        mock_async_ops.delete_row.return_value = SyncTestResponse(id=1, name="deleted")

        result = sync_ops.delete_row(1, capture_data=True)

        assert result.name == "deleted"
        mock_async_ops.delete_row.assert_called_once_with(1, capture_data=True)

    def test_get_rows_returns_list(self, sync_ops, mock_async_ops):
        """Test get_rows returns list."""
        mock_async_ops.get_rows.return_value = [SyncTestResponse(id=i, name=f"row{i}") for i in range(5)]

        results = sync_ops.get_rows(skip=10, limit=5)

        assert len(results) == 5
        mock_async_ops.get_rows.assert_called_once_with(skip=10, limit=5)

    def test_count_rows_returns_int(self, sync_ops, mock_async_ops):
        """Test count_rows returns integer."""
        mock_async_ops.count_rows.return_value = 100

        count = sync_ops.count_rows()

        assert count == 100
        assert isinstance(count, int)


class TestSyncFilterOperations:
    """Tests for sync filter operations."""

    def test_filter_rows_calls_async_version(self, sync_ops, mock_async_ops):
        """Test filter_rows calls async version."""
        mock_async_ops.filter_rows.return_value = [
            SyncTestResponse(id=1, name="match1", value=10),
            SyncTestResponse(id=2, name="match2", value=10),
        ]

        filters = [{"field": "value", "operator": "eq", "value": 10}]
        results = sync_ops.filter_rows(filters=filters)

        assert len(results) == 2
        mock_async_ops.filter_rows.assert_called_once_with(filters=filters)

    def test_count_filtered_rows_calls_async_version(self, sync_ops, mock_async_ops):
        """Test count_filtered_rows calls async version."""
        mock_async_ops.count_filtered_rows.return_value = 42

        filters = [{"field": "value", "operator": "gt", "value": 5}]
        count = sync_ops.count_filtered_rows(filters=filters)

        assert count == 42
        mock_async_ops.count_filtered_rows.assert_called_once_with(filters=filters)

    def test_find_by_calls_async_version(self, sync_ops, mock_async_ops):
        """Test find_by calls async version."""
        mock_async_ops.find_by.return_value = [SyncTestResponse(id=1, name="test", value=42)]

        results = sync_ops.find_by(name="test", value=42)

        assert len(results) == 1
        mock_async_ops.find_by.assert_called_once_with(name="test", value=42)


class TestSyncBatchOperations:
    """Tests for sync batch operations."""

    def test_create_rows_batched_calls_async_version(self, sync_ops, mock_async_ops):
        """Test create_rows_batched calls async version."""
        mock_async_ops.create_rows_batched.return_value = [
            SyncTestResponse(id=i, name=f"row{i}") for i in range(1, 101)
        ]

        data = [{"name": f"row{i}"} for i in range(1, 101)]
        results = sync_ops.create_rows_batched(data, batch_size=50)

        assert len(results) == 100
        mock_async_ops.create_rows_batched.assert_called_once_with(data, batch_size=50)

    def test_bulk_insert_rows_returns_count(self, sync_ops, mock_async_ops):
        """Test bulk_insert_rows returns count."""
        mock_async_ops.bulk_insert_rows.return_value = 1000

        data = [{"name": f"row{i}"} for i in range(1000)]
        count = sync_ops.bulk_insert_rows(data)

        assert count == 1000
        assert isinstance(count, int)
        mock_async_ops.bulk_insert_rows.assert_called_once_with(data)


class TestOptionalReturnTypes:
    """Tests for operations with optional return types."""

    def test_get_row_or_none_returns_none(self, sync_ops, mock_async_ops):
        """Test get_row_or_none returns None when not found."""
        mock_async_ops.get_row_or_none.return_value = None

        result = sync_ops.get_row_or_none(999)

        assert result is None
        mock_async_ops.get_row_or_none.assert_called_once_with(999)

    def test_get_row_or_none_returns_value(self, sync_ops, mock_async_ops):
        """Test get_row_or_none returns value when found."""
        mock_async_ops.get_row_or_none.return_value = SyncTestResponse(id=1, name="test")

        result = sync_ops.get_row_or_none(1)

        assert result is not None
        assert isinstance(result, SyncTestResponse)
        mock_async_ops.get_row_or_none.assert_called_once_with(1)

    def test_delete_row_without_capture_returns_none(self, sync_ops, mock_async_ops):
        """Test delete_row returns None when capture_data=False."""
        mock_async_ops.delete_row.return_value = None

        result = sync_ops.delete_row(1, capture_data=False)

        assert result is None
        mock_async_ops.delete_row.assert_called_once_with(1, capture_data=False)


class TestErrorHandling:
    """Tests for error handling in sync operations."""

    def test_sync_operation_propagates_exceptions(self, sync_ops, mock_async_ops):
        """Test that exceptions from async operations are propagated."""
        mock_async_ops.get_row.side_effect = RuntimeError("Database error")

        with pytest.raises(RuntimeError, match="Database error"):
            sync_ops.get_row(1)

    def test_context_manager_exception_handling(self, mock_async_ops):
        """Test that context manager exceptions are handled properly."""
        mock_async_ops.__aenter__.side_effect = ConnectionError("Cannot connect")
        sync_ops = SyncRemoteOperations(mock_async_ops)

        with pytest.raises(ConnectionError, match="Cannot connect"):
            sync_ops.get_row(1)


class TestMultipleOperations:
    """Tests for multiple sequential operations."""

    def test_multiple_sync_calls_work(self, sync_ops, mock_async_ops):
        """Test that multiple sync calls work correctly."""
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="created")
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="created")
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="updated")

        # Multiple operations
        created = sync_ops.create_row(name="created")
        retrieved = sync_ops.get_row(1)
        updated = sync_ops.update_row(1, name="updated")

        assert created.id == 1
        assert retrieved.id == 1
        assert updated.name == "updated"

        # Each should have entered/exited context manager
        assert mock_async_ops.__aenter__.call_count == 3
        assert mock_async_ops.__aexit__.call_count == 3

    def test_sync_operations_are_independent(self, mock_async_ops):
        """Test that each sync operation is independent."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        sync_ops1 = SyncRemoteOperations(mock_async_ops)
        sync_ops2 = SyncRemoteOperations(mock_async_ops)

        sync_ops1.get_row(1)
        sync_ops2.get_row(1)

        # Both should work independently
        assert mock_async_ops.get_row.call_count == 2


class TestLookupOperations:
    """Tests for lookup operations."""

    def test_lookup_by_id_or_name_returns_tuple(self, sync_ops, mock_async_ops):
        """Test lookup_by_id_or_name returns tuple."""
        mock_async_ops.lookup_by_id_or_name.return_value = (42, SyncTestResponse(id=42, name="test"))

        resolved_id, row = sync_ops.lookup_by_id_or_name(id_=42)

        assert resolved_id == 42
        assert row.id == 42
        mock_async_ops.lookup_by_id_or_name.assert_called_once_with(id_=42)

    def test_get_row_by_name_calls_async_version(self, sync_ops, mock_async_ops):
        """Test get_row_by_name calls async version."""
        mock_async_ops.get_row_by_name.return_value = SyncTestResponse(id=1, name="unique_name")

        result = sync_ops.get_row_by_name("unique_name")

        assert result.name == "unique_name"
        mock_async_ops.get_row_by_name.assert_called_once_with("unique_name")


class TestUpdateAndDeleteOperations:
    """Tests for update and delete operations."""

    def test_update_rows_returns_list(self, sync_ops, mock_async_ops):
        """Test update_rows returns list of updated rows."""
        mock_async_ops.update_rows.return_value = [
            SyncTestResponse(id=1, name="updated1", value=10),
            SyncTestResponse(id=2, name="updated2", value=20),
        ]

        updates = [
            {"id": 1, "name": "updated1", "value": 10},
            {"id": 2, "name": "updated2", "value": 20},
        ]
        results = sync_ops.update_rows(updates)

        assert len(results) == 2
        assert results[0].name == "updated1"
        assert results[1].name == "updated2"
        mock_async_ops.update_rows.assert_called_once_with(updates)

    def test_delete_rows_with_capture_returns_list(self, sync_ops, mock_async_ops):
        """Test delete_rows with capture_data=True returns list."""
        mock_async_ops.delete_rows.return_value = [
            SyncTestResponse(id=1, name="deleted1"),
            SyncTestResponse(id=2, name="deleted2"),
        ]

        results = sync_ops.delete_rows([1, 2], capture_data=True)

        assert isinstance(results, list)
        assert len(results) == 2
        mock_async_ops.delete_rows.assert_called_once_with([1, 2], capture_data=True)

    def test_delete_rows_without_capture_returns_count(self, sync_ops, mock_async_ops):
        """Test delete_rows with capture_data=False returns count."""
        mock_async_ops.delete_rows.return_value = 5

        result = sync_ops.delete_rows([1, 2, 3, 4, 5], capture_data=False)

        assert isinstance(result, int)
        assert result == 5
        mock_async_ops.delete_rows.assert_called_once_with([1, 2, 3, 4, 5], capture_data=False)

    def test_bulk_delete_rows_returns_count(self, sync_ops, mock_async_ops):
        """Test bulk_delete_rows returns count."""
        mock_async_ops.bulk_delete_rows.return_value = 50

        count = sync_ops.bulk_delete_rows([1, 2, 3, 4, 5])

        assert count == 50
        mock_async_ops.bulk_delete_rows.assert_called_once_with([1, 2, 3, 4, 5])


class TestFilterOneOperations:
    """Tests for filter_one operations."""

    def test_filter_one_returns_single_result(self, sync_ops, mock_async_ops):
        """Test filter_one returns single result."""
        mock_async_ops.filter_one.return_value = SyncTestResponse(id=1, name="unique")

        filters = [{"field": "name", "operator": "eq", "value": "unique"}]
        result = sync_ops.filter_one(filters=filters)

        assert result.name == "unique"
        mock_async_ops.filter_one.assert_called_once_with(filters=filters)

    def test_filter_one_or_none_returns_none(self, sync_ops, mock_async_ops):
        """Test filter_one_or_none returns None when not found."""
        mock_async_ops.filter_one_or_none.return_value = None

        filters = [{"field": "name", "operator": "eq", "value": "nonexistent"}]
        result = sync_ops.filter_one_or_none(filters=filters)

        assert result is None
        mock_async_ops.filter_one_or_none.assert_called_once_with(filters=filters)

    def test_filter_one_or_none_returns_result(self, sync_ops, mock_async_ops):
        """Test filter_one_or_none returns result when found."""
        mock_async_ops.filter_one_or_none.return_value = SyncTestResponse(id=1, name="found")

        filters = [{"field": "name", "operator": "eq", "value": "found"}]
        result = sync_ops.filter_one_or_none(filters=filters)

        assert result is not None
        assert result.name == "found"
        mock_async_ops.filter_one_or_none.assert_called_once_with(filters=filters)

    def test_find_one_by_returns_single_result(self, sync_ops, mock_async_ops):
        """Test find_one_by returns single result."""
        mock_async_ops.find_one_by.return_value = SyncTestResponse(id=1, name="unique")

        result = sync_ops.find_one_by(name="unique")

        assert result.name == "unique"
        mock_async_ops.find_one_by.assert_called_once_with(name="unique")


class TestCreateOperations:
    """Tests for create operations."""

    def test_create_rows_returns_list(self, sync_ops, mock_async_ops):
        """Test create_rows returns list of created rows."""
        mock_async_ops.create_rows.return_value = [
            SyncTestResponse(id=1, name="row1"),
            SyncTestResponse(id=2, name="row2"),
        ]

        data = [{"name": "row1"}, {"name": "row2"}]
        results = sync_ops.create_rows(data)

        assert len(results) == 2
        mock_async_ops.create_rows.assert_called_once_with(data)


class TestIntegrationPatterns:
    """Integration-style tests for common usage patterns."""

    def test_typical_workflow(self, sync_ops, mock_async_ops):
        """Test a typical CRUD workflow."""
        # Setup mocks
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="test", value=0)
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test", value=0)
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="test", value=100)
        mock_async_ops.delete_row.return_value = SyncTestResponse(id=1, name="test", value=100)

        # Create
        created = sync_ops.create_row(name="test")
        assert created.id == 1

        # Read
        retrieved = sync_ops.get_row(1)
        assert retrieved.name == "test"

        # Update
        updated = sync_ops.update_row(1, value=100)
        assert updated.value == 100

        # Delete
        deleted = sync_ops.delete_row(1, capture_data=True)
        assert deleted is not None

        # Verify all operations were called
        mock_async_ops.create_row.assert_called_once()
        mock_async_ops.get_row.assert_called_once()
        mock_async_ops.update_row.assert_called_once()
        mock_async_ops.delete_row.assert_called_once()


class TestContextManagerBehavior:
    """Tests for context manager behavior of wrapped methods."""

    def test_context_manager_cleanup_on_success(self, sync_ops, mock_async_ops):
        """Test context manager properly cleans up on success."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        result = sync_ops.get_row(1)

        assert result is not None
        mock_async_ops.__aenter__.assert_called_once()
        mock_async_ops.__aexit__.assert_called_once()

        # Check __aexit__ was called with no exception
        call_args = mock_async_ops.__aexit__.call_args[0]
        assert call_args == (None, None, None)

    def test_context_manager_cleanup_on_exception(self, sync_ops, mock_async_ops):
        """Test context manager properly cleans up on exception."""
        mock_async_ops.get_row.side_effect = ValueError("Test error")

        with pytest.raises(ValueError):
            sync_ops.get_row(1)

        # Context manager should still have exited
        mock_async_ops.__aenter__.assert_called_once()
        mock_async_ops.__aexit__.assert_called_once()


class TestArgumentPassing:
    """Tests for argument passing through decorator."""

    def test_positional_args_only(self, sync_ops, mock_async_ops):
        """Test passing only positional arguments."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=42, name="test")

        result = sync_ops.get_row(42)

        assert result.id == 42
        mock_async_ops.get_row.assert_called_once_with(42)

    def test_keyword_args_only(self, sync_ops, mock_async_ops):
        """Test passing only keyword arguments."""
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="test", value=99)

        result = sync_ops.create_row(name="test", value=99)

        assert result.value == 99
        mock_async_ops.create_row.assert_called_once_with(name="test", value=99)

    def test_mixed_args_and_kwargs(self, sync_ops, mock_async_ops):
        """Test passing both positional and keyword arguments."""
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="updated", value=50)

        result = sync_ops.update_row(1, name="updated", value=50)

        assert result.name == "updated"
        mock_async_ops.update_row.assert_called_once_with(1, name="updated", value=50)

    def test_complex_argument_types(self, sync_ops, mock_async_ops):
        """Test passing complex argument types (lists, dicts)."""
        mock_async_ops.filter_rows.return_value = [SyncTestResponse(id=1, name="match")]

        filters = [
            {"field": "name", "operator": "eq", "value": "match"},
            {"field": "value", "operator": "gt", "value": 10},
        ]
        results = sync_ops.filter_rows(filters=filters)

        assert len(results) == 1
        mock_async_ops.filter_rows.assert_called_once_with(filters=filters)


class TestReturnTypes:
    """Tests for correct return types from sync methods."""

    def test_single_object_return(self, sync_ops, mock_async_ops):
        """Test methods returning single object."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        result = sync_ops.get_row(1)

        assert isinstance(result, SyncTestResponse)

    def test_list_return(self, sync_ops, mock_async_ops):
        """Test methods returning list."""
        mock_async_ops.get_rows.return_value = [
            SyncTestResponse(id=1, name="row1"),
            SyncTestResponse(id=2, name="row2"),
        ]

        results = sync_ops.get_rows()

        assert isinstance(results, list)
        assert all(isinstance(r, SyncTestResponse) for r in results)

    def test_integer_return(self, sync_ops, mock_async_ops):
        """Test methods returning integer."""
        mock_async_ops.count_rows.return_value = 42

        count = sync_ops.count_rows()

        assert isinstance(count, int)
        assert count == 42

    def test_tuple_return(self, sync_ops, mock_async_ops):
        """Test methods returning tuple."""
        mock_async_ops.lookup_by_id_or_name.return_value = (10, SyncTestResponse(id=10, name="found"))

        result = sync_ops.lookup_by_id_or_name(name="found")

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0] == 10
        assert isinstance(result[1], SyncTestResponse)

    def test_optional_return_none(self, sync_ops, mock_async_ops):
        """Test optional return returning None."""
        mock_async_ops.get_row_or_none.return_value = None

        result = sync_ops.get_row_or_none(999)

        assert result is None

    def test_optional_return_value(self, sync_ops, mock_async_ops):
        """Test optional return returning value."""
        mock_async_ops.get_row_or_none.return_value = SyncTestResponse(id=1, name="test")

        result = sync_ops.get_row_or_none(1)

        assert result is not None
        assert isinstance(result, SyncTestResponse)

    def test_union_return_list(self, sync_ops, mock_async_ops):
        """Test union return type returning list."""
        mock_async_ops.delete_rows.return_value = [SyncTestResponse(id=1, name="deleted")]

        result = sync_ops.delete_rows([1], capture_data=True)

        assert isinstance(result, list)

    def test_union_return_int(self, sync_ops, mock_async_ops):
        """Test union return type returning int."""
        mock_async_ops.delete_rows.return_value = 5

        result = sync_ops.delete_rows([1, 2, 3, 4, 5], capture_data=False)

        assert isinstance(result, int)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_list_argument(self, sync_ops, mock_async_ops):
        """Test passing empty list as argument."""
        mock_async_ops.create_rows.return_value = []

        results = sync_ops.create_rows([])

        assert results == []
        mock_async_ops.create_rows.assert_called_once_with([])

    def test_large_batch_operation(self, sync_ops, mock_async_ops):
        """Test operation with large batch of data."""
        large_batch = [SyncTestResponse(id=i, name=f"row{i}") for i in range(10000)]
        mock_async_ops.create_rows_batched.return_value = large_batch

        data = [{"name": f"row{i}"} for i in range(10000)]
        results = sync_ops.create_rows_batched(data, batch_size=1000)

        assert len(results) == 10000
        mock_async_ops.create_rows_batched.assert_called_once_with(data, batch_size=1000)

    def test_none_values_in_kwargs(self, sync_ops, mock_async_ops):
        """Test passing None values in keyword arguments."""
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="test", value=0)

        result = sync_ops.update_row(1, name=None, value=0)

        assert result is not None
        mock_async_ops.update_row.assert_called_once_with(1, name=None, value=0)

    def test_special_characters_in_strings(self, sync_ops, mock_async_ops):
        """Test handling special characters in string arguments."""
        special_name = 'test\'s "quoted" name with\nnewlines'
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name=special_name)

        result = sync_ops.create_row(name=special_name)

        assert result.name == special_name
        mock_async_ops.create_row.assert_called_once_with(name=special_name)

    def test_zero_and_negative_values(self, sync_ops, mock_async_ops):
        """Test handling zero and negative values."""
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="test", value=-100)

        result = sync_ops.create_row(name="test", value=-100)

        assert result.value == -100
        mock_async_ops.create_row.assert_called_once_with(name="test", value=-100)


class TestExceptionTypes:
    """Tests for different exception types."""

    def test_value_error_propagation(self, sync_ops, mock_async_ops):
        """Test ValueError is propagated correctly."""
        mock_async_ops.get_row.side_effect = ValueError("Invalid ID")

        with pytest.raises(ValueError, match="Invalid ID"):
            sync_ops.get_row(-1)

    def test_runtime_error_propagation(self, sync_ops, mock_async_ops):
        """Test RuntimeError is propagated correctly."""
        mock_async_ops.create_row.side_effect = RuntimeError("Database locked")

        with pytest.raises(RuntimeError, match="Database locked"):
            sync_ops.create_row(name="test")

    def test_type_error_propagation(self, sync_ops, mock_async_ops):
        """Test TypeError is propagated correctly."""
        mock_async_ops.update_row.side_effect = TypeError("Invalid type")

        with pytest.raises(TypeError, match="Invalid type"):
            sync_ops.update_row(1, name=123)  # Wrong type

    def test_key_error_propagation(self, sync_ops, mock_async_ops):
        """Test KeyError is propagated correctly."""
        mock_async_ops.get_row.side_effect = KeyError("not_found")

        with pytest.raises(KeyError):
            sync_ops.get_row(999)

    def test_attribute_error_propagation(self, sync_ops, mock_async_ops):
        """Test AttributeError is propagated correctly."""
        mock_async_ops.filter_rows.side_effect = AttributeError("Invalid attribute")

        with pytest.raises(AttributeError, match="Invalid attribute"):
            sync_ops.filter_rows(filters=[])


class TestAllMethods:
    """Comprehensive tests ensuring all methods are properly wrapped."""

    def test_all_create_methods_exist(self, sync_ops):
        """Test all CREATE methods exist and are callable."""
        assert callable(sync_ops.create_row)
        assert callable(sync_ops.create_rows)
        assert callable(sync_ops.create_rows_batched)
        assert callable(sync_ops.bulk_insert_rows)

    def test_all_read_methods_exist(self, sync_ops):
        """Test all READ methods exist and are callable."""
        assert callable(sync_ops.get_row)
        assert callable(sync_ops.get_row_by_name)
        assert callable(sync_ops.get_rows)
        assert callable(sync_ops.get_row_or_none)
        assert callable(sync_ops.count_rows)
        assert callable(sync_ops.lookup_by_id_or_name)

    def test_all_update_methods_exist(self, sync_ops):
        """Test all UPDATE methods exist and are callable."""
        assert callable(sync_ops.update_row)
        assert callable(sync_ops.update_rows)

    def test_all_delete_methods_exist(self, sync_ops):
        """Test all DELETE methods exist and are callable."""
        assert callable(sync_ops.delete_row)
        assert callable(sync_ops.delete_rows)
        assert callable(sync_ops.bulk_delete_rows)

    def test_all_filter_methods_exist(self, sync_ops):
        """Test all FILTER/QUERY methods exist and are callable."""
        assert callable(sync_ops.filter_rows)
        assert callable(sync_ops.count_filtered_rows)
        assert callable(sync_ops.filter_one)
        assert callable(sync_ops.filter_one_or_none)
        assert callable(sync_ops.find_by)
        assert callable(sync_ops.find_one_by)


class TestDecoratorMetadata:
    """Tests for decorator metadata preservation."""

    def test_wrapped_function_has_name(self, sync_ops):
        """Test wrapped functions preserve their names."""
        assert sync_ops.get_row.__name__ == "get_row"
        assert sync_ops.create_row.__name__ == "create_row"
        assert sync_ops.update_row.__name__ == "update_row"
        assert sync_ops.delete_row.__name__ == "delete_row"
        assert sync_ops.filter_rows.__name__ == "filter_rows"

    def test_wrapped_function_has_module(self, sync_ops):
        """Test wrapped functions preserve module information."""
        assert hasattr(sync_ops.get_row, "__module__")
        assert hasattr(sync_ops.create_row, "__module__")


class TestConcurrentCalls:
    """Tests for handling multiple concurrent-like calls."""

    def test_rapid_sequential_calls(self, sync_ops, mock_async_ops):
        """Test rapid sequential calls work correctly."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        # Make many rapid calls
        results = [sync_ops.get_row(1) for _ in range(100)]

        assert len(results) == 100
        assert all(r.id == 1 for r in results)
        assert mock_async_ops.get_row.call_count == 100
        # Each should have used context manager
        assert mock_async_ops.__aenter__.call_count == 100
        assert mock_async_ops.__aexit__.call_count == 100

    def test_alternating_operations(self, sync_ops, mock_async_ops):
        """Test alternating different operations."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")
        mock_async_ops.count_rows.return_value = 100

        # Alternate between different operations
        for i in range(10):
            if i % 2 == 0:
                sync_ops.get_row(1)
            else:
                sync_ops.count_rows()

        assert mock_async_ops.get_row.call_count == 5
        assert mock_async_ops.count_rows.call_count == 5


class TestInitialization:
    """Tests for SyncRemoteOperations initialization."""

    def test_initialization_with_async_ops(self, mock_async_ops):
        """Test initialization with async operations instance."""
        sync_ops = SyncRemoteOperations(mock_async_ops)

        assert sync_ops.async_ops is mock_async_ops

    def test_async_ops_attribute_accessible(self, sync_ops, mock_async_ops):
        """Test async_ops attribute is accessible."""
        assert hasattr(sync_ops, "async_ops")
        assert sync_ops.async_ops is mock_async_ops


class TestBatchSizeParameter:
    """Tests for batch_size parameter handling."""

    def test_batch_size_default(self, sync_ops, mock_async_ops):
        """Test batch operation with default batch size."""
        mock_async_ops.create_rows_batched.return_value = [
            SyncTestResponse(id=i, name=f"row{i}") for i in range(100)
        ]

        data = [{"name": f"row{i}"} for i in range(100)]
        _results = sync_ops.create_rows_batched(data)

        # Should be called with data and no explicit batch_size
        mock_async_ops.create_rows_batched.assert_called_once()
        call_args = mock_async_ops.create_rows_batched.call_args
        assert call_args[0][0] == data

    def test_batch_size_custom(self, sync_ops, mock_async_ops):
        """Test batch operation with custom batch size."""
        mock_async_ops.create_rows_batched.return_value = []

        data = [{"name": f"row{i}"} for i in range(1000)]
        sync_ops.create_rows_batched(data, batch_size=250)

        mock_async_ops.create_rows_batched.assert_called_once_with(data, batch_size=250)


class TestCaptureDataParameter:
    """Tests for capture_data parameter handling."""

    def test_delete_with_capture_true(self, sync_ops, mock_async_ops):
        """Test delete operations with capture_data=True."""
        mock_async_ops.delete_row.return_value = SyncTestResponse(id=1, name="deleted")

        result = sync_ops.delete_row(1, capture_data=True)

        assert result is not None
        assert isinstance(result, SyncTestResponse)
        mock_async_ops.delete_row.assert_called_once_with(1, capture_data=True)

    def test_delete_with_capture_false(self, sync_ops, mock_async_ops):
        """Test delete operations with capture_data=False."""
        mock_async_ops.delete_row.return_value = None

        result = sync_ops.delete_row(1, capture_data=False)

        assert result is None
        mock_async_ops.delete_row.assert_called_once_with(1, capture_data=False)

    def test_delete_rows_with_capture_true(self, sync_ops, mock_async_ops):
        """Test delete_rows with capture_data=True."""
        mock_async_ops.delete_rows.return_value = [
            SyncTestResponse(id=1, name="deleted1"),
            SyncTestResponse(id=2, name="deleted2"),
        ]

        result = sync_ops.delete_rows([1, 2], capture_data=True)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_delete_rows_with_capture_false(self, sync_ops, mock_async_ops):
        """Test delete_rows with capture_data=False."""
        mock_async_ops.delete_rows.return_value = 10

        result = sync_ops.delete_rows(list(range(1, 11)), capture_data=False)

        assert isinstance(result, int)
        assert result == 10


class TestFilterParameters:
    """Tests for filter parameter handling."""

    def test_filter_with_single_condition(self, sync_ops, mock_async_ops):
        """Test filtering with single condition."""
        mock_async_ops.filter_rows.return_value = [SyncTestResponse(id=1, name="match")]

        filters = [{"field": "name", "operator": "eq", "value": "match"}]
        results = sync_ops.filter_rows(filters=filters)

        assert len(results) == 1
        mock_async_ops.filter_rows.assert_called_once_with(filters=filters)

    def test_filter_with_multiple_conditions(self, sync_ops, mock_async_ops):
        """Test filtering with multiple conditions."""
        mock_async_ops.filter_rows.return_value = []

        filters = [
            {"field": "name", "operator": "eq", "value": "test"},
            {"field": "value", "operator": "gt", "value": 10},
            {"field": "value", "operator": "lt", "value": 100},
        ]
        sync_ops.filter_rows(filters=filters)

        mock_async_ops.filter_rows.assert_called_once_with(filters=filters)

    def test_filter_with_additional_params(self, sync_ops, mock_async_ops):
        """Test filtering with additional parameters like skip/limit."""
        mock_async_ops.filter_rows.return_value = []

        filters = [{"field": "value", "operator": "gte", "value": 0}]
        sync_ops.filter_rows(filters=filters, skip=10, limit=20)

        mock_async_ops.filter_rows.assert_called_once_with(filters=filters, skip=10, limit=20)


class TestSkipLimitParameters:
    """Tests for skip/limit parameter handling."""

    def test_get_rows_with_skip(self, sync_ops, mock_async_ops):
        """Test get_rows with skip parameter."""
        mock_async_ops.get_rows.return_value = []

        sync_ops.get_rows(skip=50)

        mock_async_ops.get_rows.assert_called_once_with(skip=50)

    def test_get_rows_with_limit(self, sync_ops, mock_async_ops):
        """Test get_rows with limit parameter."""
        mock_async_ops.get_rows.return_value = []

        sync_ops.get_rows(limit=25)
        mock_async_ops.get_rows.assert_called_once_with(limit=25)

    def test_get_rows_with_skip_and_limit(self, sync_ops, mock_async_ops):
        """Test get_rows with both skip and limit."""
        mock_async_ops.get_rows.return_value = []

        sync_ops.get_rows(skip=100, limit=50)

        mock_async_ops.get_rows.assert_called_once_with(skip=100, limit=50)

    def test_get_rows_with_zero_skip(self, sync_ops, mock_async_ops):
        """Test get_rows with skip=0."""
        mock_async_ops.get_rows.return_value = []

        sync_ops.get_rows(skip=0, limit=10)

        mock_async_ops.get_rows.assert_called_once_with(skip=0, limit=10)


class TestIdOrNameLookup:
    """Tests for ID or name lookup operations."""

    def test_lookup_by_id(self, sync_ops, mock_async_ops):
        """Test lookup_by_id_or_name with ID."""
        mock_async_ops.lookup_by_id_or_name.return_value = (42, SyncTestResponse(id=42, name="found"))

        resolved_id, row = sync_ops.lookup_by_id_or_name(id_=42)

        assert resolved_id == 42
        assert row.id == 42
        mock_async_ops.lookup_by_id_or_name.assert_called_once_with(id_=42)

    def test_lookup_by_name(self, sync_ops, mock_async_ops):
        """Test lookup_by_id_or_name with name."""
        mock_async_ops.lookup_by_id_or_name.return_value = (10, SyncTestResponse(id=10, name="by_name"))

        resolved_id, row = sync_ops.lookup_by_id_or_name(name="by_name")

        assert resolved_id == 10
        assert row.name == "by_name"
        mock_async_ops.lookup_by_id_or_name.assert_called_once_with(name="by_name")

    def test_get_row_by_name_unique(self, sync_ops, mock_async_ops):
        """Test get_row_by_name for unique name."""
        mock_async_ops.get_row_by_name.return_value = SyncTestResponse(id=5, name="unique")

        result = sync_ops.get_row_by_name("unique")

        assert result.id == 5
        assert result.name == "unique"
        mock_async_ops.get_row_by_name.assert_called_once_with("unique")


class TestComplexDataTypes:
    """Tests for handling complex data types."""

    def test_nested_dict_in_data(self, sync_ops, mock_async_ops):
        """Test handling nested dictionaries in data."""
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="test")

        # Pass complex nested data
        sync_ops.create_row(name="test", metadata={"nested": {"key": "value"}})

        mock_async_ops.create_row.assert_called_once()

    def test_list_of_dicts_in_batch(self, sync_ops, mock_async_ops):
        """Test handling list of dictionaries in batch operations."""
        mock_async_ops.create_rows.return_value = [SyncTestResponse(id=i, name=f"row{i}") for i in range(3)]

        data = [
            {"name": "row0", "value": 0},
            {"name": "row1", "value": 1},
            {"name": "row2", "value": 2},
        ]
        results = sync_ops.create_rows(data)

        assert len(results) == 3
        mock_async_ops.create_rows.assert_called_once_with(data)

    def test_filter_with_complex_values(self, sync_ops, mock_async_ops):
        """Test filters with complex value types."""
        mock_async_ops.filter_rows.return_value = []

        filters = [
            {"field": "tags", "operator": "contains", "value": ["tag1", "tag2"]},
            {"field": "metadata", "operator": "jsonb_contains", "value": {"key": "val"}},
        ]
        sync_ops.filter_rows(filters=filters)

        mock_async_ops.filter_rows.assert_called_once_with(filters=filters)


class TestBulkOperations:
    """Tests for bulk operations."""

    def test_bulk_insert_large_dataset(self, sync_ops, mock_async_ops):
        """Test bulk insert with large dataset."""
        mock_async_ops.bulk_insert_rows.return_value = 10000

        data = [{"name": f"row{i}", "value": i} for i in range(10000)]
        count = sync_ops.bulk_insert_rows(data)

        assert count == 10000
        mock_async_ops.bulk_insert_rows.assert_called_once_with(data)

    def test_bulk_delete_by_ids(self, sync_ops, mock_async_ops):
        """Test bulk delete with list of IDs."""
        mock_async_ops.bulk_delete_rows.return_value = 100

        ids = list(range(1, 101))
        count = sync_ops.bulk_delete_rows(ids)

        assert count == 100
        mock_async_ops.bulk_delete_rows.assert_called_once_with(ids)

    def test_bulk_delete_empty_list(self, sync_ops, mock_async_ops):
        """Test bulk delete with empty list."""
        mock_async_ops.bulk_delete_rows.return_value = 0

        count = sync_ops.bulk_delete_rows([])

        assert count == 0
        mock_async_ops.bulk_delete_rows.assert_called_once_with([])


class TestUpdateOperations:
    """Tests for various update operations."""

    def test_update_single_field(self, sync_ops, mock_async_ops):
        """Test updating single field."""
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="original", value=999)

        result = sync_ops.update_row(1, value=999)

        assert result.value == 999
        mock_async_ops.update_row.assert_called_once_with(1, value=999)

    def test_update_multiple_fields(self, sync_ops, mock_async_ops):
        """Test updating multiple fields."""
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="new_name", value=100)

        result = sync_ops.update_row(1, name="new_name", value=100)

        assert result.name == "new_name"
        assert result.value == 100
        mock_async_ops.update_row.assert_called_once_with(1, name="new_name", value=100)

    def test_update_multiple_rows(self, sync_ops, mock_async_ops):
        """Test updating multiple rows at once."""
        mock_async_ops.update_rows.return_value = [
            SyncTestResponse(id=1, name="updated1", value=10),
            SyncTestResponse(id=2, name="updated2", value=20),
            SyncTestResponse(id=3, name="updated3", value=30),
        ]

        updates = [
            {"id": 1, "name": "updated1", "value": 10},
            {"id": 2, "name": "updated2", "value": 20},
            {"id": 3, "name": "updated3", "value": 30},
        ]
        results = sync_ops.update_rows(updates)

        assert len(results) == 3
        mock_async_ops.update_rows.assert_called_once_with(updates)


class TestCountOperations:
    """Tests for count operations."""

    def test_count_all_rows(self, sync_ops, mock_async_ops):
        """Test counting all rows."""
        mock_async_ops.count_rows.return_value = 1000

        count = sync_ops.count_rows()

        assert count == 1000
        mock_async_ops.count_rows.assert_called_once_with()

    def test_count_filtered_rows(self, sync_ops, mock_async_ops):
        """Test counting filtered rows."""
        mock_async_ops.count_filtered_rows.return_value = 50

        filters = [{"field": "value", "operator": "gte", "value": 100}]
        count = sync_ops.count_filtered_rows(filters=filters)

        assert count == 50
        mock_async_ops.count_filtered_rows.assert_called_once_with(filters=filters)

    def test_count_zero_rows(self, sync_ops, mock_async_ops):
        """Test counting when result is zero."""
        mock_async_ops.count_rows.return_value = 0

        count = sync_ops.count_rows()

        assert count == 0
        assert isinstance(count, int)


class TestFindByOperations:
    """Tests for find_by operations."""

    def test_find_by_single_field(self, sync_ops, mock_async_ops):
        """Test find_by with single field."""
        mock_async_ops.find_by.return_value = [SyncTestResponse(id=1, name="match")]

        results = sync_ops.find_by(name="match")

        assert len(results) == 1
        mock_async_ops.find_by.assert_called_once_with(name="match")

    def test_find_by_multiple_fields(self, sync_ops, mock_async_ops):
        """Test find_by with multiple fields."""
        mock_async_ops.find_by.return_value = [SyncTestResponse(id=1, name="test", value=42)]

        results = sync_ops.find_by(name="test", value=42)

        assert len(results) == 1
        mock_async_ops.find_by.assert_called_once_with(name="test", value=42)

    def test_find_one_by_single_field(self, sync_ops, mock_async_ops):
        """Test find_one_by with single field."""
        mock_async_ops.find_one_by.return_value = SyncTestResponse(id=1, name="unique")

        result = sync_ops.find_one_by(name="unique")

        assert result.name == "unique"
        mock_async_ops.find_one_by.assert_called_once_with(name="unique")

    def test_find_one_by_multiple_fields(self, sync_ops, mock_async_ops):
        """Test find_one_by with multiple fields."""
        mock_async_ops.find_one_by.return_value = SyncTestResponse(id=1, name="test", value=99)

        result = sync_ops.find_one_by(name="test", value=99)

        assert result.name == "test"
        assert result.value == 99
        mock_async_ops.find_one_by.assert_called_once_with(name="test", value=99)


class TestContextManagerReuse:
    """Tests for context manager reuse behavior."""

    def test_each_call_creates_new_context(self, sync_ops, mock_async_ops):
        """Test that each sync call creates a new async context."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        # Make multiple calls
        sync_ops.get_row(1)
        sync_ops.get_row(1)
        sync_ops.get_row(1)

        # Each should have entered and exited context
        assert mock_async_ops.__aenter__.call_count == 3
        assert mock_async_ops.__aexit__.call_count == 3

    def test_context_properly_cleaned_between_calls(self, sync_ops, mock_async_ops):
        """Test context is properly cleaned between calls."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        # First call
        sync_ops.get_row(1)
        first_enter_count = mock_async_ops.__aenter__.call_count
        first_exit_count = mock_async_ops.__aexit__.call_count

        # Second call
        sync_ops.get_row(1)

        # Should have incremented both
        assert mock_async_ops.__aenter__.call_count == first_enter_count + 1
        assert mock_async_ops.__aexit__.call_count == first_exit_count + 1


class TestErrorRecovery:
    """Tests for error recovery behavior."""

    def test_error_in_first_call_doesnt_affect_second(self, sync_ops, mock_async_ops):
        """Test that error in first call doesn't affect subsequent calls."""
        # First call fails
        mock_async_ops.get_row.side_effect = [
            ValueError("First call error"),
            SyncTestResponse(id=1, name="success"),
        ]

        # First call should raise
        with pytest.raises(ValueError, match="First call error"):
            sync_ops.get_row(1)

        # Second call should succeed
        result = sync_ops.get_row(1)
        assert result.name == "success"

    def test_context_manager_cleaned_after_error(self, sync_ops, mock_async_ops):
        """Test context manager is cleaned up after error."""
        mock_async_ops.get_row.side_effect = ValueError("Error")

        with pytest.raises(ValueError):
            sync_ops.get_row(1)

        # Context manager should have been entered and exited
        assert mock_async_ops.__aenter__.call_count == 1
        assert mock_async_ops.__aexit__.call_count == 1


class TestTypeHints:
    """Tests for type hint preservation (runtime checks)."""

    def test_methods_are_typed(self, sync_ops):
        """Test that methods have type annotations."""
        assert hasattr(sync_ops.get_row, "__annotations__")
        assert hasattr(sync_ops.create_row, "__annotations__")
        assert hasattr(sync_ops.update_row, "__annotations__")
        assert hasattr(sync_ops.delete_row, "__annotations__")

    def test_return_types_match_expected(self, sync_ops, mock_async_ops):
        """Test that return types match expectations at runtime."""
        # Single object
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")
        result = sync_ops.get_row(1)
        assert isinstance(result, SyncTestResponse)

        # List
        mock_async_ops.get_rows.return_value = [SyncTestResponse(id=1, name="test")]
        result = sync_ops.get_rows()
        assert isinstance(result, list)

        # Integer
        mock_async_ops.count_rows.return_value = 42
        result = sync_ops.count_rows()
        assert isinstance(result, int)

        # Tuple
        mock_async_ops.lookup_by_id_or_name.return_value = (1, SyncTestResponse(id=1, name="test"))
        result = sync_ops.lookup_by_id_or_name(id_=1)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestAsyncOpsReference:
    """Tests for async_ops reference handling."""

    def test_async_ops_is_stored(self, sync_ops, mock_async_ops):
        """Test that async_ops reference is stored correctly."""
        assert sync_ops.async_ops is mock_async_ops

    def test_async_ops_not_modified(self, sync_ops, mock_async_ops):
        """Test that sync operations don't modify async_ops."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        original_async_ops = sync_ops.async_ops
        sync_ops.get_row(1)

        # Reference should still be the same
        assert sync_ops.async_ops is original_async_ops

    def test_multiple_sync_wrappers_same_async_ops(self, mock_async_ops):
        """Test multiple sync wrappers can share same async_ops."""
        sync_ops1 = SyncRemoteOperations(mock_async_ops)
        sync_ops2 = SyncRemoteOperations(mock_async_ops)

        assert sync_ops1.async_ops is mock_async_ops
        assert sync_ops2.async_ops is mock_async_ops
        assert sync_ops1.async_ops is sync_ops2.async_ops


class TestNoOperationWithoutCall:
    """Tests ensuring operations don't execute without explicit calls."""

    def test_creating_sync_ops_doesnt_call_async(self, mock_async_ops):
        """Test that just creating SyncRemoteOperations doesn't call async methods."""
        _sync_ops = SyncRemoteOperations(mock_async_ops)

        # No async methods should have been called
        mock_async_ops.get_row.assert_not_called()
        mock_async_ops.create_row.assert_not_called()
        mock_async_ops.update_row.assert_not_called()
        mock_async_ops.delete_row.assert_not_called()

    def test_accessing_method_doesnt_execute_it(self, sync_ops, mock_async_ops):
        """Test that accessing method without calling doesn't execute it."""
        # Just access the method
        _method = sync_ops.get_row

        # Should not have called async version
        mock_async_ops.get_row.assert_not_called()


class TestAllCRUDMethods:
    """Comprehensive test of all CRUD methods."""

    def test_create_row_method(self, sync_ops, mock_async_ops):
        """Test create_row method."""
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="created")
        result = sync_ops.create_row(name="created")
        assert result.id == 1

    def test_create_rows_method(self, sync_ops, mock_async_ops):
        """Test create_rows method."""
        mock_async_ops.create_rows.return_value = [
            SyncTestResponse(id=1, name="row1"),
            SyncTestResponse(id=2, name="row2"),
        ]
        results = sync_ops.create_rows([{"name": "row1"}, {"name": "row2"}])
        assert len(results) == 2

    def test_create_rows_batched_method(self, sync_ops, mock_async_ops):
        """Test create_rows_batched method."""
        mock_async_ops.create_rows_batched.return_value = [
            SyncTestResponse(id=i, name=f"row{i}") for i in range(10)
        ]
        data = [{"name": f"row{i}"} for i in range(10)]
        results = sync_ops.create_rows_batched(data, batch_size=5)
        assert len(results) == 10

    def test_bulk_insert_rows_method(self, sync_ops, mock_async_ops):
        """Test bulk_insert_rows method."""
        mock_async_ops.bulk_insert_rows.return_value = 100
        count = sync_ops.bulk_insert_rows([{"name": f"row{i}"} for i in range(100)])
        assert count == 100

    def test_get_row_method(self, sync_ops, mock_async_ops):
        """Test get_row method."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")
        result = sync_ops.get_row(1)
        assert result.id == 1

    def test_get_row_by_name_method(self, sync_ops, mock_async_ops):
        """Test get_row_by_name method."""
        mock_async_ops.get_row_by_name.return_value = SyncTestResponse(id=1, name="unique")
        result = sync_ops.get_row_by_name("unique")
        assert result.name == "unique"

    def test_get_rows_method(self, sync_ops, mock_async_ops):
        """Test get_rows method."""
        mock_async_ops.get_rows.return_value = [SyncTestResponse(id=i, name=f"row{i}") for i in range(5)]
        results = sync_ops.get_rows(limit=5)
        assert len(results) == 5

    def test_get_row_or_none_method(self, sync_ops, mock_async_ops):
        """Test get_row_or_none method."""
        mock_async_ops.get_row_or_none.return_value = None
        result = sync_ops.get_row_or_none(999)
        assert result is None

    def test_count_rows_method(self, sync_ops, mock_async_ops):
        """Test count_rows method."""
        mock_async_ops.count_rows.return_value = 42
        count = sync_ops.count_rows()
        assert count == 42

    def test_lookup_by_id_or_name_method(self, sync_ops, mock_async_ops):
        """Test lookup_by_id_or_name method."""
        mock_async_ops.lookup_by_id_or_name.return_value = (1, SyncTestResponse(id=1, name="test"))
        resolved_id, row = sync_ops.lookup_by_id_or_name(id_=1)
        assert resolved_id == 1

    def test_update_row_method(self, sync_ops, mock_async_ops):
        """Test update_row method."""
        mock_async_ops.update_row.return_value = SyncTestResponse(id=1, name="updated")
        result = sync_ops.update_row(1, name="updated")
        assert result.name == "updated"

    def test_update_rows_method(self, sync_ops, mock_async_ops):
        """Test update_rows method."""
        mock_async_ops.update_rows.return_value = [
            SyncTestResponse(id=1, name="updated1"),
            SyncTestResponse(id=2, name="updated2"),
        ]
        updates = [
            {"id": 1, "name": "updated1"},
            {"id": 2, "name": "updated2"},
        ]
        results = sync_ops.update_rows(updates)
        assert len(results) == 2

    def test_delete_row_method(self, sync_ops, mock_async_ops):
        """Test delete_row method."""
        mock_async_ops.delete_row.return_value = SyncTestResponse(id=1, name="deleted")
        result = sync_ops.delete_row(1, capture_data=True)
        assert result.name == "deleted"

    def test_delete_rows_method(self, sync_ops, mock_async_ops):
        """Test delete_rows method."""
        mock_async_ops.delete_rows.return_value = [
            SyncTestResponse(id=1, name="deleted1"),
            SyncTestResponse(id=2, name="deleted2"),
        ]
        results = sync_ops.delete_rows([1, 2], capture_data=True)
        assert len(results) == 2

    def test_bulk_delete_rows_method(self, sync_ops, mock_async_ops):
        """Test bulk_delete_rows method."""
        mock_async_ops.bulk_delete_rows.return_value = 10
        count = sync_ops.bulk_delete_rows(list(range(1, 11)))
        assert count == 10

    def test_filter_rows_method(self, sync_ops, mock_async_ops):
        """Test filter_rows method."""
        mock_async_ops.filter_rows.return_value = [SyncTestResponse(id=1, name="match")]
        filters = [{"field": "name", "operator": "eq", "value": "match"}]
        results = sync_ops.filter_rows(filters=filters)
        assert len(results) == 1

    def test_count_filtered_rows_method(self, sync_ops, mock_async_ops):
        """Test count_filtered_rows method."""
        mock_async_ops.count_filtered_rows.return_value = 5
        filters = [{"field": "value", "operator": "gt", "value": 10}]
        count = sync_ops.count_filtered_rows(filters=filters)
        assert count == 5

    def test_filter_one_method(self, sync_ops, mock_async_ops):
        """Test filter_one method."""
        mock_async_ops.filter_one.return_value = SyncTestResponse(id=1, name="unique")
        filters = [{"field": "name", "operator": "eq", "value": "unique"}]
        result = sync_ops.filter_one(filters=filters)
        assert result.name == "unique"

    def test_filter_one_or_none_method(self, sync_ops, mock_async_ops):
        """Test filter_one_or_none method."""
        mock_async_ops.filter_one_or_none.return_value = None
        filters = [{"field": "name", "operator": "eq", "value": "none"}]
        result = sync_ops.filter_one_or_none(filters=filters)
        assert result is None

    def test_find_by_method(self, sync_ops, mock_async_ops):
        """Test find_by method."""
        mock_async_ops.find_by.return_value = [SyncTestResponse(id=1, name="test")]
        results = sync_ops.find_by(name="test")
        assert len(results) == 1

    def test_find_one_by_method(self, sync_ops, mock_async_ops):
        """Test find_one_by method."""
        mock_async_ops.find_one_by.return_value = SyncTestResponse(id=1, name="unique")
        result = sync_ops.find_one_by(name="unique")
        assert result.name == "unique"


class TestDecoratorConsistency:
    """Tests for consistency of decorator behavior across all methods."""

    def test_all_methods_use_context_manager(self, sync_ops, mock_async_ops):
        """Test that all methods properly use async context manager."""
        methods_to_test = [
            ("create_row", {"name": "test"}),
            ("get_row", (1,)),
            ("update_row", (1,), {"name": "updated"}),
            ("delete_row", (1,)),
            ("count_rows", ()),
            ("filter_rows", (), {"filters": []}),
        ]

        for method_name, args, *kwargs_list in methods_to_test:
            # Reset mocks
            mock_async_ops.__aenter__.reset_mock()
            mock_async_ops.__aexit__.reset_mock()

            # Setup return value
            mock_method = getattr(mock_async_ops, method_name)
            if method_name == "count_rows":
                mock_method.return_value = 0
            elif method_name in ["filter_rows", "create_rows"]:
                mock_method.return_value = []
            elif method_name in ["delete_row", "get_row_or_none"]:
                mock_method.return_value = None
            else:
                mock_method.return_value = SyncTestResponse(id=1, name="test")

            # Call method
            kwargs = kwargs_list[0] if kwargs_list else {}
            sync_method = getattr(sync_ops, method_name)
            sync_method(*args, **kwargs)

            # Verify context manager was used
            mock_async_ops.__aenter__.assert_called_once()
            mock_async_ops.__aexit__.assert_called_once()


class TestMemoryAndResourceManagement:
    """Tests for memory and resource management."""

    def test_no_resource_leaks_on_repeated_calls(self, sync_ops, mock_async_ops):
        """Test that repeated calls don't leak resources."""
        mock_async_ops.get_row.return_value = SyncTestResponse(id=1, name="test")

        # Make many calls
        for _ in range(1000):
            sync_ops.get_row(1)

        # Context manager should have been entered and exited equal times
        assert mock_async_ops.__aenter__.call_count == mock_async_ops.__aexit__.call_count
        assert mock_async_ops.__aenter__.call_count == 1000

    def test_exception_doesnt_leak_resources(self, sync_ops, mock_async_ops):
        """Test that exceptions don't leak resources."""
        mock_async_ops.get_row.side_effect = ValueError("Error")

        # Try to make call that fails
        try:
            sync_ops.get_row(1)
        except ValueError:
            pass

        # Context manager should still be properly cleaned
        assert mock_async_ops.__aenter__.call_count == 1
        assert mock_async_ops.__aexit__.call_count == 1


class TestRealWorldScenarios:
    """Tests simulating real-world usage scenarios."""

    def test_pagination_workflow(self, sync_ops, mock_async_ops):
        """Test paginating through results."""
        # Setup mock to return different pages
        mock_async_ops.get_rows.side_effect = [
            [SyncTestResponse(id=i, name=f"row{i}") for i in range(0, 10)],
            [SyncTestResponse(id=i, name=f"row{i}") for i in range(10, 20)],
            [SyncTestResponse(id=i, name=f"row{i}") for i in range(20, 30)],
            [],  # No more results
        ]

        all_results = []
        skip = 0
        limit = 10

        while True:
            page = sync_ops.get_rows(skip=skip, limit=limit)
            if not page:
                break
            all_results.extend(page)
            skip += limit

        assert len(all_results) == 30
        assert mock_async_ops.get_rows.call_count == 4

    def test_search_and_update_workflow(self, sync_ops, mock_async_ops):
        """Test searching for records and updating them."""
        # Search
        mock_async_ops.filter_rows.return_value = [
            SyncTestResponse(id=1, name="old_name", value=0),
            SyncTestResponse(id=2, name="old_name", value=0),
        ]

        filters = [{"field": "name", "operator": "eq", "value": "old_name"}]
        results = sync_ops.filter_rows(filters=filters)

        # Update each
        mock_async_ops.update_row.side_effect = [
            SyncTestResponse(id=1, name="new_name", value=0),
            SyncTestResponse(id=2, name="new_name", value=0),
        ]

        for row in results:
            sync_ops.update_row(row.id, name="new_name")

        assert mock_async_ops.update_row.call_count == 2

    def test_conditional_delete_workflow(self, sync_ops, mock_async_ops):
        """Test filtering and conditionally deleting records."""
        # Find records to delete
        mock_async_ops.filter_rows.return_value = [
            SyncTestResponse(id=1, name="to_delete", value=0),
            SyncTestResponse(id=2, name="to_delete", value=0),
            SyncTestResponse(id=3, name="to_delete", value=0),
        ]

        filters = [{"field": "name", "operator": "eq", "value": "to_delete"}]
        to_delete = sync_ops.filter_rows(filters=filters)

        # Delete them
        ids_to_delete = [row.id for row in to_delete]
        mock_async_ops.bulk_delete_rows.return_value = len(ids_to_delete)

        deleted_count = sync_ops.bulk_delete_rows(ids_to_delete)

        assert deleted_count == 3
        mock_async_ops.bulk_delete_rows.assert_called_once_with([1, 2, 3])

    def test_batch_import_workflow(self, sync_ops, mock_async_ops):
        """Test importing large batch of records."""
        # Prepare data
        records = [{"name": f"import_{i}", "value": i} for i in range(1000)]

        # Import in batches
        mock_async_ops.create_rows_batched.return_value = [
            SyncTestResponse(id=i, name=f"import_{i}", value=i) for i in range(1000)
        ]

        imported = sync_ops.create_rows_batched(records, batch_size=100)

        assert len(imported) == 1000
        mock_async_ops.create_rows_batched.assert_called_once_with(records, batch_size=100)

    def test_data_validation_workflow(self, sync_ops, mock_async_ops):
        """Test validating data before creation."""
        # First check if exists
        mock_async_ops.find_one_by.side_effect = [
            None,  # Doesn't exist
            SyncTestResponse(id=1, name="exists", value=0),  # Exists
        ]

        # Try to create first one (allowed)
        mock_async_ops.create_row.return_value = SyncTestResponse(id=1, name="new_item", value=0)

        existing = sync_ops.find_one_by(name="new_item")
        if existing is None:
            created = sync_ops.create_row(name="new_item", value=0)
            assert created.id == 1

        # Try to create second one (already exists, skip)
        existing = sync_ops.find_one_by(name="exists")
        if existing is not None:
            # Skip creation
            pass

        assert mock_async_ops.create_row.call_count == 1


class TestComplexFilters:
    """Tests for complex filter scenarios."""

    def test_multiple_filter_conditions(self, sync_ops, mock_async_ops):
        """Test filtering with multiple conditions."""
        mock_async_ops.filter_rows.return_value = [SyncTestResponse(id=1, name="match", value=50)]

        filters = [
            {"field": "value", "operator": "gte", "value": 10},
            {"field": "value", "operator": "lte", "value": 100},
            {"field": "name", "operator": "like", "value": "mat%"},
        ]

        results = sync_ops.filter_rows(filters=filters)

        assert len(results) == 1
        mock_async_ops.filter_rows.assert_called_once_with(filters=filters)

    def test_filter_with_pagination(self, sync_ops, mock_async_ops):
        """Test filtering combined with pagination."""
        mock_async_ops.filter_rows.return_value = [
            SyncTestResponse(id=i, name=f"row{i}", value=i) for i in range(10, 20)
        ]

        filters = [{"field": "value", "operator": "gte", "value": 0}]
        results = sync_ops.filter_rows(filters=filters, skip=10, limit=10)

        assert len(results) == 10
        mock_async_ops.filter_rows.assert_called_once_with(filters=filters, skip=10, limit=10)

    def test_empty_filter_results(self, sync_ops, mock_async_ops):
        """Test filtering that returns no results."""
        mock_async_ops.filter_rows.return_value = []

        filters = [{"field": "name", "operator": "eq", "value": "nonexistent"}]
        results = sync_ops.filter_rows(filters=filters)

        assert results == []
        assert isinstance(results, list)


class TestBatchEdgeCases:
    """Tests for edge cases in batch operations."""

    def test_batch_with_single_item(self, sync_ops, mock_async_ops):
        """Test batch operation with single item."""
        mock_async_ops.create_rows_batched.return_value = [SyncTestResponse(id=1, name="single")]

        results = sync_ops.create_rows_batched([{"name": "single"}], batch_size=10)

        assert len(results) == 1

    def test_batch_with_exact_batch_size(self, sync_ops, mock_async_ops):
        """Test batch operation where data matches batch size exactly."""
        mock_async_ops.create_rows_batched.return_value = [
            SyncTestResponse(id=i, name=f"row{i}") for i in range(100)
        ]

        data = [{"name": f"row{i}"} for i in range(100)]
        results = sync_ops.create_rows_batched(data, batch_size=100)

        assert len(results) == 100

    def test_batch_larger_than_batch_size(self, sync_ops, mock_async_ops):
        """Test batch operation where data exceeds batch size."""
        mock_async_ops.create_rows_batched.return_value = [
            SyncTestResponse(id=i, name=f"row{i}") for i in range(250)
        ]

        data = [{"name": f"row{i}"} for i in range(250)]
        results = sync_ops.create_rows_batched(data, batch_size=100)

        assert len(results) == 250


class TestErrorMessages:
    """Tests for error message clarity."""

    def test_error_message_preserved(self, sync_ops, mock_async_ops):
        """Test that error messages from async operations are preserved."""
        error_msg = "Detailed error message about what went wrong"
        mock_async_ops.get_row.side_effect = ValueError(error_msg)

        with pytest.raises(ValueError) as exc_info:
            sync_ops.get_row(1)

        assert error_msg in str(exc_info.value)

    def test_error_type_preserved(self, sync_ops, mock_async_ops):
        """Test that error types from async operations are preserved."""
        mock_async_ops.create_row.side_effect = TypeError("Wrong type")

        with pytest.raises(TypeError):
            sync_ops.create_row(name=123)

    def test_nested_error_preserved(self, sync_ops, mock_async_ops):
        """Test that nested exceptions are preserved."""
        original_error = ValueError("Original error")
        wrapper_error = RuntimeError("Wrapper error")
        wrapper_error.__cause__ = original_error

        mock_async_ops.get_row.side_effect = wrapper_error

        with pytest.raises(RuntimeError) as exc_info:
            sync_ops.get_row(1)

        assert exc_info.value.__cause__ is original_error


class TestMethodSignatures:
    """Tests for method signature consistency."""

    def test_create_row_signature(self, sync_ops):
        """Test create_row has correct signature."""
        import inspect

        sig = inspect.signature(sync_ops.create_row)
        # Should accept *args and **kwargs
        assert "args" in str(sig) or "kwargs" in str(sig)

    def test_get_row_signature(self, sync_ops):
        """Test get_row has correct signature."""
        import inspect

        sig = inspect.signature(sync_ops.get_row)
        assert "args" in str(sig) or "kwargs" in str(sig)
