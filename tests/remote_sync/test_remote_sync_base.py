"""Unit tests for SyncRemoteOperations wrapper."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

from rail_svc.remote_async.base import AsyncRemoteOperations
from rail_svc.remote_sync.base import (AlgorithmSyncRemoteOperations,
                                       BandSyncRemoteOperations,
                                       CatalogBandAssocSyncRemoteOperations,
                                       CatalogTagSyncRemoteOperations,
                                       DatasetAssocSyncRemoteOperations,
                                       DatasetSyncRemoteOperations,
                                       EstimatesSyncRemoteOperations,
                                       EstimatorSyncRemoteOperations,
                                       ModelSyncRemoteOperations,
                                       SyncRemoteOperations, run_async)


# Test models
class RemoteSyncTestResponse(BaseModel):
    """Test response model."""

    id: int
    name: str
    value: int = 0


class RemoteSyncTestCreate(BaseModel):
    """Test create model."""

    name: str
    value: int = 0


class TestRunAsyncDecorator:
    """Tests for the run_async decorator."""

    def test_decorator_runs_coroutine_synchronously(self) -> None:
        """Test that decorator runs async function synchronously."""

        @run_async
        def sync_wrapper(self) -> str:
            async def async_func() -> str:
                return "test_result"

            return async_func()

        class TestClass:
            pass

        result = sync_wrapper(TestClass())
        assert result == "test_result"

    def test_decorator_raises_from_async_context(self) -> None:
        """Test that decorator raises error when called from async context."""

        @run_async
        def sync_wrapper(self) -> str:
            async def async_func() -> str:
                return "test_result"

            return async_func()

        class TestClass:
            pass

        async def test_in_async_context():
            with pytest.raises(RuntimeError) as exc_info:
                sync_wrapper(TestClass())
            assert "cannot be used from async code" in str(exc_info.value)
            assert "AsyncRemoteOperations" in str(exc_info.value)

        asyncio.run(test_in_async_context())

    def test_decorator_passes_args_and_kwargs(self) -> None:
        """Test that decorator passes through arguments."""

        @run_async
        def sync_wrapper(self, *args, **kwargs) -> dict:
            async def async_func(*args, **kwargs) -> dict:
                return {"args": args, "kwargs": kwargs}

            return async_func(*args, **kwargs)

        class TestClass:
            pass

        result = sync_wrapper(TestClass(), 1, 2, key="value")
        assert result["args"] == (1, 2)
        assert result["kwargs"] == {"key": "value"}


class TestSyncRemoteOperationsBasics:
    """Basic tests for SyncRemoteOperations."""

    def test_initialization(self) -> None:
        """Test SyncRemoteOperations initialization."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)
        sync_ops = SyncRemoteOperations(mock_async_ops)

        assert sync_ops.async_ops is mock_async_ops

    def test_sync_ops_wraps_async_ops(self) -> None:
        """Test that sync operations wrap async operations."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)
        sync_ops = SyncRemoteOperations(mock_async_ops)

        # All methods should exist
        assert hasattr(sync_ops, "get_row")
        assert hasattr(sync_ops, "create_row")
        assert hasattr(sync_ops, "update_row")
        assert hasattr(sync_ops, "delete_row")
        assert hasattr(sync_ops, "filter_rows")


class TestSyncCRUDOperations:
    """Tests for synchronous CRUD operations."""

    def test_get_row_calls_async_version(self) -> None:
        """Test that get_row calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_get_row(row_id: int) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=row_id, name="test")

        mock_async_ops.get_row = mock_get_row

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.get_row(42)

        assert isinstance(result, RemoteSyncTestResponse)
        assert result.id == 42
        assert result.name == "test"

    def test_create_row_calls_async_version(self) -> None:
        """Test that create_row calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_create_row(**kwargs) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=1, **kwargs)

        mock_async_ops.create_row = mock_create_row

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.create_row(name="created", value=100)

        assert isinstance(result, RemoteSyncTestResponse)
        assert result.name == "created"
        assert result.value == 100

    def test_update_row_calls_async_version(self) -> None:
        """Test that update_row calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_update_row(row_id: int, **kwargs) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=row_id, **kwargs)

        mock_async_ops.update_row = mock_update_row

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.update_row(1, name="updated", value=200)

        assert result.id == 1
        assert result.name == "updated"
        assert result.value == 200

    def test_delete_row_calls_async_version(self) -> None:
        """Test that delete_row calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_delete_row(row_id: int, **kwargs) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=row_id, name="deleted")

        mock_async_ops.delete_row = mock_delete_row

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.delete_row(1, capture_data=True)

        assert result is not None
        assert result.name == "deleted"

    def test_get_rows_returns_list(self) -> None:
        """Test that get_rows returns a list."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_get_rows(**kwargs) -> list[RemoteSyncTestResponse]:
            return [RemoteSyncTestResponse(id=i, name=f"row{i}") for i in range(5)]

        mock_async_ops.get_rows = mock_get_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        results = sync_ops.get_rows(limit=5)

        assert isinstance(results, list)
        assert len(results) == 5
        assert all(isinstance(r, RemoteSyncTestResponse) for r in results)

    def test_count_rows_returns_int(self) -> None:
        """Test that count_rows returns an integer."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_count_rows() -> int:
            return 100

        mock_async_ops.count_rows = mock_count_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        count = sync_ops.count_rows()

        assert isinstance(count, int)
        assert count == 100


class TestSyncFilterOperations:
    """Tests for synchronous filter operations."""

    def test_filter_rows_calls_async_version(self) -> None:
        """Test that filter_rows calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_filter_rows(**kwargs) -> list[RemoteSyncTestResponse]:
            return [RemoteSyncTestResponse(id=1, name="match", value=10)]

        mock_async_ops.filter_rows = mock_filter_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        results = sync_ops.filter_rows(filters=[{"field": "value", "operator": "eq", "value": 10}])

        assert len(results) == 1
        assert results[0].value == 10

    def test_count_filtered_rows_calls_async_version(self) -> None:
        """Test that count_filtered_rows calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_count_filtered_rows(**kwargs) -> int:
            return 42

        mock_async_ops.count_filtered_rows = mock_count_filtered_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        count = sync_ops.count_filtered_rows(filters=[])

        assert count == 42

    def test_find_by_calls_async_version(self) -> None:
        """Test that find_by calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_find_by(**kwargs) -> list[RemoteSyncTestResponse]:
            return [RemoteSyncTestResponse(id=1, name=kwargs.get("name", "test"))]

        mock_async_ops.find_by = mock_find_by

        sync_ops = SyncRemoteOperations(mock_async_ops)
        results = sync_ops.find_by(name="test")

        assert len(results) == 1
        assert results[0].name == "test"


class TestSyncBatchOperations:
    """Tests for synchronous batch operations."""

    def test_create_rows_batched_calls_async_version(self) -> None:
        """Test that create_rows_batched calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_create_rows_batched(data, **kwargs) -> list[RemoteSyncTestResponse]:
            return [RemoteSyncTestResponse(id=i, **row) for i, row in enumerate(data, 1)]

        mock_async_ops.create_rows_batched = mock_create_rows_batched

        sync_ops = SyncRemoteOperations(mock_async_ops)
        data = [{"name": f"row{i}"} for i in range(10)]
        results = sync_ops.create_rows_batched(data, batch_size=5)

        assert len(results) == 10

    def test_bulk_insert_rows_returns_count(self) -> None:
        """Test that bulk_insert_rows returns count."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_bulk_insert_rows(data, **kwargs) -> int:
            return len(data)

        mock_async_ops.bulk_insert_rows = mock_bulk_insert_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        count = sync_ops.bulk_insert_rows([{"name": "row"} for _ in range(100)])

        assert count == 100


class TestSpecializedSyncClasses:
    """Tests for specialized sync operation classes."""

    def test_algorithm_sync_operations_subclass(self) -> None:
        """Test AlgorithmSyncRemoteOperations is proper subclass."""
        assert issubclass(AlgorithmSyncRemoteOperations, SyncRemoteOperations)

    def test_dataset_sync_operations_subclass(self) -> None:
        """Test DatasetSyncRemoteOperations is proper subclass."""
        assert issubclass(DatasetSyncRemoteOperations, SyncRemoteOperations)

    def test_model_sync_operations_subclass(self) -> None:
        """Test ModelSyncRemoteOperations is proper subclass."""
        assert issubclass(ModelSyncRemoteOperations, SyncRemoteOperations)

    def test_all_specialized_classes_exist(self) -> None:
        """Test that all specialized sync classes are defined."""
        specialized_classes = [
            AlgorithmSyncRemoteOperations,
            BandSyncRemoteOperations,
            CatalogBandAssocSyncRemoteOperations,
            CatalogTagSyncRemoteOperations,
            DatasetSyncRemoteOperations,
            DatasetAssocSyncRemoteOperations,
            EstimatesSyncRemoteOperations,
            EstimatorSyncRemoteOperations,
            ModelSyncRemoteOperations,
        ]

        for cls in specialized_classes:
            assert issubclass(cls, SyncRemoteOperations)

    def test_specialized_class_can_be_instantiated(self) -> None:
        """Test that specialized classes can be instantiated."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)
        sync_ops = AlgorithmSyncRemoteOperations(mock_async_ops)

        assert isinstance(sync_ops, SyncRemoteOperations)
        assert sync_ops.async_ops is mock_async_ops


class TestOptionalReturnTypes:
    """Tests for operations with optional return types."""

    def test_get_row_or_none_returns_none(self) -> None:
        """Test that get_row_or_none can return None."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_get_row_or_none(row_id: int) -> RemoteSyncTestResponse | None:
            return None

        mock_async_ops.get_row_or_none = mock_get_row_or_none

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.get_row_or_none(999)

        assert result is None

    def test_get_row_or_none_returns_value(self) -> None:
        """Test that get_row_or_none can return a value."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_get_row_or_none(row_id: int) -> RemoteSyncTestResponse | None:
            return RemoteSyncTestResponse(id=row_id, name="found")

        mock_async_ops.get_row_or_none = mock_get_row_or_none

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.get_row_or_none(1)

        assert result is not None
        assert result.id == 1

    def test_delete_row_without_capture_returns_none(self) -> None:
        """Test that delete_row can return None."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_delete_row(row_id: int, **kwargs) -> RemoteSyncTestResponse | None:
            if kwargs.get("capture_data"):
                return RemoteSyncTestResponse(id=row_id, name="deleted")
            return None

        mock_async_ops.delete_row = mock_delete_row

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.delete_row(1, capture_data=False)

        assert result is None


class TestErrorHandling:
    """Tests for error handling in sync operations."""

    def test_sync_operation_cannot_be_called_from_async(self) -> None:
        """Test that sync operations raise error from async context."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_get_row(row_id: int) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=row_id, name="test")

        mock_async_ops.get_row = mock_get_row

        sync_ops = SyncRemoteOperations(mock_async_ops)

        async def test_from_async():
            with pytest.raises(RuntimeError) as exc_info:
                sync_ops.get_row(1)
            assert "cannot be used from async code" in str(exc_info.value)
            asyncio.run(test_from_async())

    def test_sync_operation_propagates_exceptions(self) -> None:
        """Test that exceptions from async operations are propagated."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_get_row(row_id: int) -> RemoteSyncTestResponse:
            raise ValueError("Test error")

        mock_async_ops.get_row = mock_get_row

        sync_ops = SyncRemoteOperations(mock_async_ops)

        with pytest.raises(ValueError) as exc_info:
            sync_ops.get_row(1)
        assert "Test error" in str(exc_info.value)


class TestMultipleOperations:
    """Tests for calling multiple operations."""

    def test_multiple_sync_calls_work(self) -> None:
        """Test that multiple sync operations can be called sequentially."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        call_count = {"get": 0, "create": 0, "update": 0}

        async def mock_get_row(row_id: int) -> RemoteSyncTestResponse:
            call_count["get"] += 1
            return RemoteSyncTestResponse(id=row_id, name="test")

        async def mock_create_row(**kwargs) -> RemoteSyncTestResponse:
            call_count["create"] += 1
            return RemoteSyncTestResponse(id=1, **kwargs)

        async def mock_update_row(row_id: int, **kwargs) -> RemoteSyncTestResponse:
            call_count["update"] += 1
            return RemoteSyncTestResponse(id=row_id, **kwargs)

        mock_async_ops.get_row = mock_get_row
        mock_async_ops.create_row = mock_create_row
        mock_async_ops.update_row = mock_update_row

        sync_ops = SyncRemoteOperations(mock_async_ops)

        # Multiple operations
        sync_ops.get_row(1)
        sync_ops.create_row(name="new")
        sync_ops.update_row(1, name="updated")

        assert call_count["get"] == 1
        assert call_count["create"] == 1
        assert call_count["update"] == 1

    def test_sync_operations_are_independent(self) -> None:
        """Test that each sync operation runs independently."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        state = {"counter": 0}

        async def mock_create_row(**kwargs) -> RemoteSyncTestResponse:
            state["counter"] += 1
            return RemoteSyncTestResponse(id=state["counter"], **kwargs)

        mock_async_ops.create_row = mock_create_row

        sync_ops = SyncRemoteOperations(mock_async_ops)

        # Each call should increment independently
        result1 = sync_ops.create_row(name="first")
        result2 = sync_ops.create_row(name="second")
        result3 = sync_ops.create_row(name="third")

        assert result1.id == 1
        assert result2.id == 2
        assert result3.id == 3


class TestLookupOperations:
    """Tests for lookup operations."""

    def test_lookup_by_id_or_name_returns_tuple(self) -> None:
        """Test that lookup_by_id_or_name returns tuple."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_lookup(id_=None, name=None) -> tuple[int, RemoteSyncTestResponse]:
            if id_ is not None:
                return (id_, RemoteSyncTestResponse(id=id_, name="by_id"))
            return (10, RemoteSyncTestResponse(id=10, name=name or "by_name"))

        mock_async_ops.lookup_by_id_or_name = mock_lookup

        sync_ops = SyncRemoteOperations(mock_async_ops)

        # Lookup by ID
        resolved_id, row = sync_ops.lookup_by_id_or_name(id_=42)
        assert resolved_id == 42
        assert row.name == "by_id"

        # Lookup by name
        resolved_id, row = sync_ops.lookup_by_id_or_name(name="test")
        assert resolved_id == 10
        assert row.name == "test"

    def test_get_row_by_name_calls_async_version(self) -> None:
        """Test that get_row_by_name calls the async version."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_get_row_by_name(name: str) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=1, name=name)

        mock_async_ops.get_row_by_name = mock_get_row_by_name

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.get_row_by_name("unique_name")

        assert result.name == "unique_name"


class TestUpdateAndDeleteOperations:
    """Tests for update and delete operations."""

    def test_update_rows_returns_list(self) -> None:
        """Test that update_rows returns a list."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_update_rows(data) -> list[RemoteSyncTestResponse]:
            return [RemoteSyncTestResponse(**row) for row in data]

        mock_async_ops.update_rows = mock_update_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        updates = [
            {"id": 1, "name": "updated1"},
            {"id": 2, "name": "updated2"},
        ]
        results = sync_ops.update_rows(updates)

        assert len(results) == 2
        assert results[0].name == "updated1"

    def test_delete_rows_with_capture_returns_list(self) -> None:
        """Test that delete_rows with capture returns list."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_delete_rows(ids, **kwargs) -> list[RemoteSyncTestResponse] | int:
            if kwargs.get("capture_data"):
                return [RemoteSyncTestResponse(id=i, name=f"deleted{i}") for i in ids]
            return len(ids)

        mock_async_ops.delete_rows = mock_delete_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        results = sync_ops.delete_rows([1, 2, 3], capture_data=True)

        assert isinstance(results, list)
        assert len(results) == 3

    def test_delete_rows_without_capture_returns_count(self) -> None:
        """Test that delete_rows without capture returns count."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_delete_rows(ids, **kwargs) -> list[RemoteSyncTestResponse] | int:
            if kwargs.get("capture_data"):
                return [RemoteSyncTestResponse(id=i, name=f"deleted{i}") for i in ids]
            return len(ids)

        mock_async_ops.delete_rows = mock_delete_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.delete_rows([1, 2, 3], capture_data=False)

        assert isinstance(result, int)
        assert result == 3

    def test_bulk_delete_rows_returns_count(self) -> None:
        """Test that bulk_delete_rows returns count."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_bulk_delete_rows(ids) -> int:
            return len(ids)

        mock_async_ops.bulk_delete_rows = mock_bulk_delete_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        count = sync_ops.bulk_delete_rows([1, 2, 3, 4, 5])

        assert count == 5


class TestFilterOneOperations:
    """Tests for filter_one operations."""

    def test_filter_one_returns_single_result(self) -> None:
        """Test that filter_one returns a single result."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_filter_one(**kwargs) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=1, name="unique")

        mock_async_ops.filter_one = mock_filter_one

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.filter_one(filters=[])

        assert isinstance(result, RemoteSyncTestResponse)
        assert result.name == "unique"

    def test_filter_one_or_none_returns_none(self) -> None:
        """Test that filter_one_or_none can return None."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_filter_one_or_none(**kwargs) -> RemoteSyncTestResponse | None:
            return None

        mock_async_ops.filter_one_or_none = mock_filter_one_or_none

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.filter_one_or_none(filters=[])

        assert result is None

    def test_filter_one_or_none_returns_result(self) -> None:
        """Test that filter_one_or_none can return a result."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_filter_one_or_none(**kwargs) -> RemoteSyncTestResponse | None:
            return RemoteSyncTestResponse(id=1, name="found")

        mock_async_ops.filter_one_or_none = mock_filter_one_or_none

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.filter_one_or_none(filters=[])

        assert result is not None
        assert result.name == "found"

    def test_find_one_by_returns_single_result(self) -> None:
        """Test that find_one_by returns a single result."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_find_one_by(**kwargs) -> RemoteSyncTestResponse:
            return RemoteSyncTestResponse(id=1, name=kwargs.get("name", "test"))

        mock_async_ops.find_one_by = mock_find_one_by

        sync_ops = SyncRemoteOperations(mock_async_ops)
        result = sync_ops.find_one_by(name="unique")

        assert result.name == "unique"


class TestCreateOperations:
    """Tests for create operations."""

    def test_create_rows_returns_list(self) -> None:
        """Test that create_rows returns a list."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        async def mock_create_rows(data, **kwargs) -> list[RemoteSyncTestResponse]:
            return [RemoteSyncTestResponse(id=i, **row) for i, row in enumerate(data, 1)]

        mock_async_ops.create_rows = mock_create_rows

        sync_ops = SyncRemoteOperations(mock_async_ops)
        data = [{"name": "row1"}, {"name": "row2"}]
        results = sync_ops.create_rows(data)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0].name == "row1"
        assert results[1].name == "row2"


class TestIntegrationPatterns:
    """Tests for common integration patterns."""

    def test_typical_workflow(self) -> None:
        """Test a typical CRUD workflow."""
        mock_async_ops = Mock(spec=AsyncRemoteOperations)

        storage = {}
        next_id = 1

        async def mock_create_row(**kwargs) -> RemoteSyncTestResponse:
            nonlocal next_id
            row = RemoteSyncTestResponse(id=next_id, **kwargs)
            storage[next_id] = row
            next_id += 1
            return row

        async def mock_get_row(row_id: int) -> RemoteSyncTestResponse:
            return storage[row_id]

        async def mock_update_row(row_id: int, **kwargs) -> RemoteSyncTestResponse:
            row = storage[row_id]
            updated = RemoteSyncTestResponse(
                id=row_id, name=kwargs.get("name", row.name), value=kwargs.get("value", row.value)
            )
            storage[row_id] = updated
            return updated

        async def mock_delete_row(row_id: int, **kwargs) -> RemoteSyncTestResponse | None:
            return storage.pop(row_id, None)

        mock_async_ops.create_row = mock_create_row
        mock_async_ops.get_row = mock_get_row
        mock_async_ops.update_row = mock_update_row
        mock_async_ops.delete_row = mock_delete_row

        sync_ops = SyncRemoteOperations(mock_async_ops)

        # Create
        created = sync_ops.create_row(name="test", value=100)
        assert created.id == 1

        # Read
        retrieved = sync_ops.get_row(1)
        assert retrieved.name == "test"

        # Update
        updated = sync_ops.update_row(1, name="updated", value=200)
        assert updated.name == "updated"

        # Delete
        deleted = sync_ops.delete_row(1)
        assert deleted is not None
