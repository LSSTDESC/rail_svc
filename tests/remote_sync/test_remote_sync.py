"""Unit tests for remote_sync factory functions."""

from __future__ import annotations

from rail_svc import remote_async
from rail_svc.remote_sync import (
    algorithm,
    dataset,
    model,
)
from rail_svc.remote_sync.rail_svc import SyncRemoteOperations


class TestFactoryWrapsAsyncOperations:
    """Test that factories wrap the correct async operations."""

    def test_algorithm_wraps_async_algorithm(self) -> None:
        """Test algorithm() wraps remote_async.algorithm."""
        ops = algorithm()
        assert ops.async_ops is remote_async.algorithm

    def test_dataset_wraps_async_dataset(self) -> None:
        """Test dataset() wraps remote_async.dataset."""
        ops = dataset()
        assert ops.async_ops is remote_async.dataset

    def test_model_wraps_async_model(self) -> None:
        """Test model() wraps remote_async.model."""
        ops = model()
        assert ops.async_ops is remote_async.model


class TestFactoryInstances:
    """Test that factory functions create new instances."""

    def test_creates_new_instances(self) -> None:
        """Test that calling a factory creates new instances wrapping shared async ops."""
        ops1 = algorithm()
        ops2 = algorithm()

        assert ops1 is not ops2
        assert ops1.async_ops is ops2.async_ops

    def test_returns_sync_remote_operations(self) -> None:
        """Test that factories return SyncRemoteOperations instances."""
        ops = algorithm()
        assert isinstance(ops, SyncRemoteOperations)


class TestSyncOperationsHaveMethods:
    """Test that sync operations have expected methods."""

    def test_has_crud_methods(self) -> None:
        """Test sync ops have CRUD methods."""
        ops = algorithm()

        assert callable(ops.get_row)
        assert callable(ops.create_row)
        assert callable(ops.update_row)
        assert callable(ops.delete_row)
        assert callable(ops.filter_rows)
        assert callable(ops.count_rows)
