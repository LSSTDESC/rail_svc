"""Unit tests for remote_async module initialization and pre-configured instances."""

from __future__ import annotations

from rail_svc import models
from rail_svc.remote_async import algorithm, band, dataset, model
from rail_svc.remote_async.base import AsyncRemoteOperations
from rail_svc.remote_async.funcs import AsyncRemoteFuncs


class TestModuleBasics:
    """Basic tests for module structure."""

    def test_all_exports_are_async_remote_operations(self) -> None:
        """Test that all exported instances are AsyncRemoteOperations."""
        import rail_svc.remote_async as remote_async
        from rail_svc.remote_async import __all__

        for export_name in __all__:
            obj = getattr(remote_async, export_name)
            assert isinstance(obj, (AsyncRemoteOperations, AsyncRemoteFuncs))

    def test_instances_are_usable(self) -> None:
        """Test that instances have the expected async methods."""
        # Just verify a few key methods exist and are callable
        assert callable(algorithm.get_row)
        assert callable(dataset.create_row)
        assert callable(model.filter_rows)


class TestInstanceConfiguration:
    """Test that instances are properly configured."""

    def test_algorithm_has_correct_models(self) -> None:
        """Test algorithm uses Algorithm models."""
        assert algorithm.response_model == models.Algorithm
        assert algorithm.create_model == models.AlgorithmCreate

    def test_dataset_has_correct_models(self) -> None:
        """Test dataset uses Dataset models."""
        assert dataset.response_model == models.Dataset
        assert dataset.create_model == models.DatasetCreate

    def test_model_has_correct_models(self) -> None:
        """Test model uses Model models."""
        assert model.response_model == models.Model
        assert model.create_model == models.ModelCreate

    def test_instances_have_different_table_names(self) -> None:
        """Test that instances target different tables."""
        table_names = {
            algorithm.table_name,
            band.table_name,
            dataset.table_name,
            model.table_name,
        }
        # Should all be different
        assert len(table_names) == 4


class TestUsagePatterns:
    """Test common usage patterns work correctly."""

    async def test_can_use_as_context_manager(self) -> None:
        """Test that instances work as async context managers."""
        async with algorithm as ops:
            assert ops is algorithm
            # Should be initialized
            assert ops._client is not None

        # Should be cleaned up
        assert algorithm._client is None

    async def test_instances_are_independent(self) -> None:
        """Test that different instances don't interfere with each other."""
        async with algorithm as algo_ops:
            async with dataset as data_ops:
                # Both should be initialized independently
                assert algo_ops._client is not None
                assert data_ops._client is not None
                # Should target different tables
                assert algo_ops.table_name != data_ops.table_name

    async def test_instance_can_be_reused(self) -> None:
        """Test that an instance can be used multiple times."""
        # First use
        async with algorithm:
            assert algorithm._client is not None

        assert algorithm._client is None

        # Second use - should work fine
        async with algorithm:
            assert algorithm._client is not None

        assert algorithm._client is None


class TestModelCoverage:
    """Test that all expected models have operations."""

    def test_all_major_tables_covered(self) -> None:
        """Test that operations exist for all major table types."""
        from rail_svc.remote_async import __all__

        # Just verify we have a reasonable set of operations
        assert len(__all__) >= 5  # Should have several tables
        assert "algorithm" in __all__
        assert "dataset" in __all__
        assert "model" in __all__
