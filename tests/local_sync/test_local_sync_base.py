"""Unit tests for synchronous local operations wrapper."""

from typing import Any
from unittest.mock import AsyncMock, Mock

from pydantic import BaseModel

from rail_svc.local_sync.base import SyncOperations, sync_wrapper

# ============================================================================
# Mock classes for testing
# ============================================================================


class MockResponse(BaseModel):
    """Mock Pydantic response model."""

    id_: int
    name: str


class MockCreate(BaseModel):
    """Mock Pydantic create model."""

    name: str


# ============================================================================
# sync_wrapper decorator tests
# ============================================================================


def test_sync_wrapper_decorator():
    """Test sync_wrapper decorator wraps async method."""

    class AsyncClass:
        async def async_method(self, value: int) -> int:
            """Async method docstring."""
            return value * 2

    class SyncClass:
        def __init__(self):
            self.async_obj = AsyncClass()

        @sync_wrapper(AsyncClass.async_method)
        def sync_method(self, *args: Any, **kwargs: Any) -> Any:
            return self.async_obj.async_method(*args, **kwargs)

    sync_obj = SyncClass()
    result = sync_obj.sync_method(5)

    assert result == 10
    assert sync_obj.sync_method.__doc__ == "Async method docstring."


def test_sync_wrapper_preserves_docstring():
    """Test that sync_wrapper copies docstring from async method."""

    class AsyncClass:
        async def documented_method(self) -> str:
            """This is the documentation."""
            return "result"

    class SyncClass:
        def __init__(self):
            self.async_obj = AsyncClass()

        @sync_wrapper(AsyncClass.documented_method)
        def documented_method(self, *args: Any, **kwargs: Any) -> Any:
            return self.async_obj.documented_method(*args, **kwargs)

    sync_obj = SyncClass()
    assert sync_obj.documented_method.__doc__ == "This is the documentation."


def test_sync_wrapper_with_args_and_kwargs():
    """Test sync_wrapper passes args and kwargs correctly."""

    class AsyncClass:
        async def method_with_args(self, a: int, b: str, c: int = 10) -> tuple:
            """Method with args."""
            return (a, b, c)

    class SyncClass:
        def __init__(self):
            self.async_obj = AsyncClass()

        @sync_wrapper(AsyncClass.method_with_args)
        def method_with_args(self, *args: Any, **kwargs: Any) -> Any:
            return self.async_obj.method_with_args(*args, **kwargs)

    sync_obj = SyncClass()
    result = sync_obj.method_with_args(5, "test", c=20)

    assert result == (5, "test", 20)


# ============================================================================
# SyncOperations method tests
# ============================================================================


def test_sync_operations_create_row():
    """Test create_row synchronous wrapper."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.create_row = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.create_row(name="test")

    assert isinstance(result, MockResponse)
    assert result.id_ == 1
    assert result.name == "test"


def test_sync_operations_get_row():
    """Test get_row synchronous wrapper."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.get_row = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.get_row(row_id=1)

    assert isinstance(result, MockResponse)
    assert result.id_ == 1


def test_sync_operations_get_rows():
    """Test get_rows synchronous wrapper."""
    mock_responses = [MockResponse(id_=1, name="test1"), MockResponse(id_=2, name="test2")]
    mock_async_ops = Mock()
    mock_async_ops.get_rows = AsyncMock(return_value=mock_responses)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.get_rows(limit=10)

    assert len(result) == 2
    assert all(isinstance(r, MockResponse) for r in result)


def test_sync_operations_count_rows():
    """Test count_rows synchronous wrapper."""
    mock_async_ops = Mock()
    mock_async_ops.count_rows = AsyncMock(return_value=42)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.count_rows()

    assert result == 42


def test_sync_operations_update_row():
    """Test update_row synchronous wrapper."""
    mock_response = MockResponse(id_=1, name="updated")
    mock_async_ops = Mock()
    mock_async_ops.update_row = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.update_row(row_id=1, name="updated")

    assert isinstance(result, MockResponse)
    assert result.name == "updated"


def test_sync_operations_delete_row():
    """Test delete_row synchronous wrapper."""
    mock_async_ops = Mock()
    mock_async_ops.delete_row = AsyncMock(return_value={"id_": 1, "name": "deleted"})

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.delete_row(row_id=1)

    assert result == {"id_": 1, "name": "deleted"}


def test_sync_operations_filter_rows():
    """Test filter_rows synchronous wrapper."""
    mock_responses = [MockResponse(id_=1, name="filtered")]
    mock_async_ops = Mock()
    mock_async_ops.filter_rows = AsyncMock(return_value=mock_responses)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.filter_rows(filters=[])

    assert len(result) == 1
    assert result[0].name == "filtered"


def test_sync_operations_get_row_or_none_with_result():
    """Test get_row_or_none returns result when row exists."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.get_row_or_none = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.get_row_or_none(row_id=1)

    assert isinstance(result, MockResponse)


def test_sync_operations_get_row_or_none_with_none():
    """Test get_row_or_none returns None when row doesn't exist."""
    mock_async_ops = Mock()
    mock_async_ops.get_row_or_none = AsyncMock(return_value=None)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.get_row_or_none(row_id=99999)

    assert result is None


def test_sync_operations_lookup_by_id_or_name():
    """Test lookup_by_id_or_name synchronous wrapper."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.lookup_by_id_or_name = AsyncMock(return_value=(1, mock_response))

    sync_ops = SyncOperations(mock_async_ops)
    row_id, result = sync_ops.lookup_by_id_or_name(row_id=1, name=None)

    assert row_id == 1
    assert isinstance(result, MockResponse)


def test_sync_operations_filter_one():
    """Test filter_one synchronous wrapper."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.filter_one = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.filter_one(filters=[])

    assert isinstance(result, MockResponse)


def test_sync_operations_filter_one_or_none_found():
    """Test filter_one_or_none when row is found."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.filter_one_or_none = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.filter_one_or_none(filters=[])

    assert isinstance(result, MockResponse)


def test_sync_operations_filter_one_or_none_not_found():
    """Test filter_one_or_none when row is not found."""
    mock_async_ops = Mock()
    mock_async_ops.filter_one_or_none = AsyncMock(return_value=None)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.filter_one_or_none(filters=[])

    assert result is None


def test_sync_operations_find_by():
    """Test find_by synchronous wrapper."""
    mock_responses = [MockResponse(id_=1, name="test")]
    mock_async_ops = Mock()
    mock_async_ops.find_by = AsyncMock(return_value=mock_responses)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.find_by(name="test")

    assert len(result) == 1
    assert result[0].name == "test"


def test_sync_operations_find_one_by():
    """Test find_one_by synchronous wrapper."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.find_one_by = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.find_one_by(name="test")

    assert isinstance(result, MockResponse)


def test_sync_operations_bulk_insert_rows():
    """Test bulk_insert_rows synchronous wrapper."""
    mock_async_ops = Mock()
    mock_async_ops.bulk_insert_rows = AsyncMock(return_value=5)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.bulk_insert_rows(rows_data=[])

    assert result == 5


def test_sync_operations_bulk_delete_rows():
    """Test bulk_delete_rows synchronous wrapper."""
    mock_async_ops = Mock()
    mock_async_ops.bulk_delete_rows = AsyncMock(return_value=3)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.bulk_delete_rows(row_ids=[1, 2, 3])

    assert result == 3


def test_sync_operations_count_filtered_rows():
    """Test count_filtered_rows synchronous wrapper."""
    mock_async_ops = Mock()
    mock_async_ops.count_filtered_rows = AsyncMock(return_value=10)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.count_filtered_rows(filters=[])

    assert result == 10


def test_sync_operations_update_rows():
    """Test update_rows synchronous wrapper."""
    mock_responses = [MockResponse(id_=1, name="updated1"), MockResponse(id_=2, name="updated2")]
    mock_async_ops = Mock()
    mock_async_ops.update_rows = AsyncMock(return_value=mock_responses)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.update_rows(updates=[])

    assert len(result) == 2


def test_sync_operations_delete_rows():
    """Test delete_rows synchronous wrapper."""
    mock_async_ops = Mock()
    mock_async_ops.delete_rows = AsyncMock(
        return_value=[{"id_": 1, "name": "deleted1"}, {"id_": 2, "name": "deleted2"}]
    )

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.delete_rows(row_ids=[1, 2])

    assert len(result) == 2


def test_sync_operations_get_row_by_name():
    """Test get_row_by_name synchronous wrapper."""
    mock_response = MockResponse(id_=1, name="test")
    mock_async_ops = Mock()
    mock_async_ops.get_row_by_name = AsyncMock(return_value=mock_response)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.get_row_by_name(name="test")

    assert isinstance(result, MockResponse)
    assert result.name == "test"


def test_sync_operations_create_rows():
    """Test create_rows synchronous wrapper."""
    mock_responses = [MockResponse(id_=1, name="created1"), MockResponse(id_=2, name="created2")]
    mock_async_ops = Mock()
    mock_async_ops.create_rows = AsyncMock(return_value=mock_responses)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.create_rows(rows_data=[])

    assert len(result) == 2


def test_sync_operations_create_rows_batched():
    """Test create_rows_batched synchronous wrapper."""
    mock_responses = [MockResponse(id_=1, name="batch1"), MockResponse(id_=2, name="batch2")]
    mock_async_ops = Mock()
    mock_async_ops.create_rows_batched = AsyncMock(return_value=mock_responses)

    sync_ops = SyncOperations(mock_async_ops)
    result = sync_ops.create_rows_batched(rows_data=[], batch_size=10)

    assert len(result) == 2
