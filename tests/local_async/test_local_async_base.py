"""Unit tests for local operations base class."""

from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import BaseModel

from rail_svc.local_async.base import (
    LocalOperations,
    to_pydantic,
    to_pydantic_list,
    to_pydantic_or_none,
    with_session,
    with_session_transaction,
)

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
# Decorator tests
# ============================================================================


@pytest.mark.asyncio
async def test_with_session_decorator():
    """Test with_session decorator provides session."""
    session_used = None

    class TestClass:
        @with_session
        async def test_method(self, session: Any, value: int) -> int:
            nonlocal session_used
            session_used = session
            return value * 2

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        obj = TestClass()
        result = await obj.test_method(5)

        assert result == 10
        assert session_used is mock_session


@pytest.mark.asyncio
async def test_with_session_transaction_decorator():
    """Test with_session_transaction decorator provides session with transaction."""
    session_used = None

    class TestClass:
        @with_session_transaction
        async def test_method(self, session: Any, value: str) -> str:
            nonlocal session_used
            session_used = session
            return value.upper()

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        obj = TestClass()
        result = await obj.test_method("test")

        assert result == "TEST"
        assert session_used is mock_session
        mock_session.begin.assert_called_once()


@pytest.mark.asyncio
async def test_to_pydantic_decorator():
    """Test to_pydantic decorator converts result."""
    mock_orm = Mock()
    mock_orm.id_ = 1
    mock_orm.name = "test"

    class TestClass:
        def __init__(self):
            self._table_ops = Mock()
            self._table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

        @to_pydantic
        async def test_method(self) -> Any:
            return mock_orm

    obj = TestClass()
    result = await obj.test_method()

    assert isinstance(result, MockResponse)
    assert result.id_ == 1
    assert result.name == "test"
    obj._table_ops.to_pydantic.assert_called_once_with(mock_orm)


@pytest.mark.asyncio
async def test_to_pydantic_list_decorator():
    """Test to_pydantic_list decorator converts list of results."""
    mock_orm1 = Mock()
    mock_orm2 = Mock()

    class TestClass:
        def __init__(self):
            self._table_ops = Mock()
            self._table_ops.to_pydantic_list.return_value = [
                MockResponse(id_=1, name="test1"),
                MockResponse(id_=2, name="test2"),
            ]

        @to_pydantic_list
        async def test_method(self) -> Any:
            return [mock_orm1, mock_orm2]

    obj = TestClass()
    result = await obj.test_method()

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(r, MockResponse) for r in result)


@pytest.mark.asyncio
async def test_to_pydantic_or_none_decorator_with_result():
    """Test to_pydantic_or_none decorator with non-None result."""
    mock_orm = Mock()

    class TestClass:
        def __init__(self):
            self._table_ops = Mock()
            self._table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

        @to_pydantic_or_none
        async def test_method(self) -> Any:
            return mock_orm

    obj = TestClass()
    result = await obj.test_method()

    assert isinstance(result, MockResponse)
    obj._table_ops.to_pydantic.assert_called_once_with(mock_orm)


@pytest.mark.asyncio
async def test_to_pydantic_or_none_decorator_with_none():
    """Test to_pydantic_or_none decorator with None result."""

    class TestClass:
        def __init__(self):
            self._table_ops = Mock()

        @to_pydantic_or_none
        async def test_method(self) -> Any:
            return None

    obj = TestClass()
    result = await obj.test_method()

    assert result is None
    obj._table_ops.to_pydantic.assert_not_called()


# ============================================================================
# LocalOperations method tests
# ============================================================================


@pytest.mark.asyncio
async def test_local_operations_create_row():
    """Test create_row method."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.create_row = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.create_row(name="test")

        assert isinstance(result, MockResponse)
        mock_table_ops.create_row.assert_called_once()


@pytest.mark.asyncio
async def test_local_operations_get_row():
    """Test get_row method."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.get_row = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.get_row(row_id=1)

        assert isinstance(result, MockResponse)
        mock_table_ops.get_row.assert_called_once()


@pytest.mark.asyncio
async def test_local_operations_get_rows():
    """Test get_rows method."""
    mock_orm1 = Mock(id_=1)
    mock_orm2 = Mock(id_=2)
    mock_table_ops = Mock()
    mock_table_ops.get_rows = AsyncMock(return_value=[mock_orm1, mock_orm2])
    mock_table_ops.to_pydantic_list.return_value = [
        MockResponse(id_=1, name="test1"),
        MockResponse(id_=2, name="test2"),
    ]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.get_rows(limit=10)

        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_local_operations_get_rows_streaming():
    """Test get_rows_streaming method."""
    mock_orm1 = Mock(id_=1, name="test1")
    mock_orm2 = Mock(id_=2, name="test2")

    async def mock_streaming():
        yield mock_orm1
        yield mock_orm2

    mock_table_ops = Mock()
    mock_table_ops.get_rows_streaming.return_value = mock_streaming()
    mock_table_ops.to_pydantic.side_effect = [
        MockResponse(id_=1, name="test1"),
        MockResponse(id_=2, name="test2"),
    ]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        results = []
        async for row in ops.get_rows_streaming():
            results.append(row)

        assert len(results) == 2
        assert all(isinstance(r, MockResponse) for r in results)


@pytest.mark.asyncio
async def test_local_operations_count_rows():
    """Test count_rows method."""
    mock_table_ops = Mock()
    mock_table_ops.count_rows = AsyncMock(return_value=42)

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.count_rows()

        assert result == 42


@pytest.mark.asyncio
async def test_local_operations_update_row():
    """Test update_row method."""
    mock_orm = Mock(id_=1, name="updated")
    mock_table_ops = Mock()
    mock_table_ops.update_row = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="updated")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.update_row(row_id=1, name="updated")

        assert isinstance(result, MockResponse)
        assert result.name == "updated"


@pytest.mark.asyncio
async def test_local_operations_delete_row():
    """Test delete_row method."""
    mock_table_ops = Mock()
    mock_table_ops.delete_row = AsyncMock(return_value={"id_": 1, "name": "deleted"})

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.delete_row(row_id=1)

        assert result == {"id_": 1, "name": "deleted"}


@pytest.mark.asyncio
async def test_local_operations_filter_rows():
    """Test filter_rows method."""
    mock_orm = Mock(id_=1)
    mock_table_ops = Mock()
    mock_table_ops.filter_rows = AsyncMock(return_value=[mock_orm])
    mock_table_ops.to_pydantic_list.return_value = [MockResponse(id_=1, name="test")]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.filter_rows(filters=[])

        assert isinstance(result, list)
        assert len(result) == 1


@pytest.mark.asyncio
async def test_local_operations_lookup_by_id_or_name():
    """Test lookup_by_id_or_name method."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.lookup_by_id_or_name = AsyncMock(return_value=(1, mock_orm))
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        row_id, result = await ops.lookup_by_id_or_name(row_id=1, name=None)

        assert row_id == 1
        assert isinstance(result, MockResponse)


@pytest.mark.asyncio
async def test_local_operations_get_row_or_none_with_result():
    """Test get_row_or_none returns Pydantic model when row exists."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.get_row_or_none = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.get_row_or_none(row_id=1)

        assert isinstance(result, MockResponse)


@pytest.mark.asyncio
async def test_local_operations_get_row_or_none_with_none():
    """Test get_row_or_none returns None when row doesn't exist."""
    mock_table_ops = Mock()
    mock_table_ops.get_row_or_none = AsyncMock(return_value=None)

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.get_row_or_none(row_id=99999)

        assert result is None


@pytest.mark.asyncio
async def test_local_operations_filter_one():
    """Test filter_one method."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.filter_one = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.filter_one(filters=[])

        assert isinstance(result, MockResponse)


@pytest.mark.asyncio
async def test_local_operations_filter_one_or_none_found():
    """Test filter_one_or_none when row is found."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.filter_one_or_none = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.filter_one_or_none(filters=[])

        assert isinstance(result, MockResponse)


@pytest.mark.asyncio
async def test_local_operations_filter_one_or_none_not_found():
    """Test filter_one_or_none when row is not found."""
    mock_table_ops = Mock()
    mock_table_ops.filter_one_or_none = AsyncMock(return_value=None)

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.filter_one_or_none(filters=[])

        assert result is None


@pytest.mark.asyncio
async def test_local_operations_find_by():
    """Test find_by method."""
    mock_orm1 = Mock(id_=1)
    mock_orm2 = Mock(id_=2)
    mock_table_ops = Mock()
    mock_table_ops.find_by = AsyncMock(return_value=[mock_orm1, mock_orm2])
    mock_table_ops.to_pydantic_list.return_value = [
        MockResponse(id_=1, name="test1"),
        MockResponse(id_=2, name="test2"),
    ]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.find_by(name="test")

        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_local_operations_find_one_by():
    """Test find_one_by method."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.find_one_by = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.find_one_by(name="test")

        assert isinstance(result, MockResponse)


@pytest.mark.asyncio
async def test_local_operations_bulk_insert_rows():
    """Test bulk_insert_rows method."""
    mock_table_ops = Mock()
    mock_table_ops.bulk_insert_rows = AsyncMock(return_value=5)

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.bulk_insert_rows(rows_data=[])

        assert result == 5


@pytest.mark.asyncio
async def test_local_operations_bulk_delete_rows():
    """Test bulk_delete_rows method."""
    mock_table_ops = Mock()
    mock_table_ops.bulk_delete_rows = AsyncMock(return_value=3)

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.bulk_delete_rows(row_ids=[1, 2, 3])

        assert result == 3


@pytest.mark.asyncio
async def test_local_operations_count_filtered_rows():
    """Test count_filtered_rows method."""
    mock_table_ops = Mock()
    mock_table_ops.count_filtered_rows = AsyncMock(return_value=10)

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.count_filtered_rows(filters=[])

        assert result == 10


@pytest.mark.asyncio
async def test_local_operations_filter_rows_streaming():
    """Test filter_rows_streaming method."""
    mock_orm1 = Mock(id_=1, name="test1")
    mock_orm2 = Mock(id_=2, name="test2")

    async def mock_streaming():
        yield mock_orm1
        yield mock_orm2

    mock_table_ops = Mock()
    mock_table_ops.filter_rows_streaming.return_value = mock_streaming()
    mock_table_ops.to_pydantic.side_effect = [
        MockResponse(id_=1, name="test1"),
        MockResponse(id_=2, name="test2"),
    ]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        results = []
        async for row in ops.filter_rows_streaming(filters=[]):
            results.append(row)

        assert len(results) == 2
        assert all(isinstance(r, MockResponse) for r in results)


@pytest.mark.asyncio
async def test_local_operations_update_rows():
    """Test update_rows method."""
    mock_orm1 = Mock(id_=1)
    mock_orm2 = Mock(id_=2)
    mock_table_ops = Mock()
    mock_table_ops.update_rows = AsyncMock(return_value=[mock_orm1, mock_orm2])
    mock_table_ops.to_pydantic_list.return_value = [
        MockResponse(id_=1, name="updated1"),
        MockResponse(id_=2, name="updated2"),
    ]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.update_rows(updates=[])

        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_local_operations_delete_rows():
    """Test delete_rows method."""
    mock_table_ops = Mock()
    mock_table_ops.delete_rows = AsyncMock(
        return_value=[{"id_": 1, "name": "deleted1"}, {"id_": 2, "name": "deleted2"}]
    )

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.delete_rows(row_ids=[1, 2])

        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_local_operations_get_row_by_name():
    """Test get_row_by_name method."""
    mock_orm = Mock(id_=1, name="test")
    mock_table_ops = Mock()
    mock_table_ops.get_row_by_name = AsyncMock(return_value=mock_orm)
    mock_table_ops.to_pydantic.return_value = MockResponse(id_=1, name="test")

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.get_row_by_name(name="test")

        assert isinstance(result, MockResponse)
        assert result.name == "test"


@pytest.mark.asyncio
async def test_local_operations_create_rows():
    """Test create_rows method."""
    mock_orm1 = Mock(id_=1)
    mock_orm2 = Mock(id_=2)
    mock_table_ops = Mock()
    mock_table_ops.create_rows = AsyncMock(return_value=[mock_orm1, mock_orm2])
    mock_table_ops.to_pydantic_list.return_value = [
        MockResponse(id_=1, name="created1"),
        MockResponse(id_=2, name="created2"),
    ]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_transaction = AsyncMock()
        mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
        mock_transaction.__aexit__ = AsyncMock(return_value=None)
        mock_session.begin = Mock(return_value=mock_transaction)
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.create_rows(rows_data=[])

        assert isinstance(result, list)
        assert len(result) == 2


@pytest.mark.asyncio
async def test_local_operations_create_rows_batched():
    """Test create_rows_batched method."""
    mock_orm1 = Mock(id_=1)
    mock_orm2 = Mock(id_=2)
    mock_table_ops = Mock()
    mock_table_ops.create_rows_batched = AsyncMock(return_value=[mock_orm1, mock_orm2])
    mock_table_ops.to_pydantic_list.return_value = [
        MockResponse(id_=1, name="batch1"),
        MockResponse(id_=2, name="batch2"),
    ]

    ops = LocalOperations(mock_table_ops)

    with patch("rail_svc.local_async.base.get_session") as mock_get_session:
        mock_session = AsyncMock()
        mock_get_session.return_value.__aenter__.return_value = mock_session

        result = await ops.create_rows_batched(rows_data=[], batch_size=10)

        assert isinstance(result, list)
        assert len(result) == 2
