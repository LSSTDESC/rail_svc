"""Unit tests for remote_sync factory functions."""

from __future__ import annotations

from rail_svc import remote_async
from rail_svc.remote_sync import (
    algorithm,
    band,
    catalog_band_assoc,
    catalog_tag,
    dataset,
    dataset_assoc,
    estimates,
    estimator,
    model,
)
from rail_svc.remote_sync.base import (
    AlgorithmSyncRemoteOperations,
    BandSyncRemoteOperations,
    CatalogBandAssocSyncRemoteOperations,
    CatalogTagSyncRemoteOperations,
    DatasetAssocSyncRemoteOperations,
    DatasetSyncRemoteOperations,
    EstimatesSyncRemoteOperations,
    EstimatorSyncRemoteOperations,
    ModelSyncRemoteOperations,
    SyncRemoteOperations,
)


class TestFactoryFunctions:
    """Tests for factory functions that create sync operations."""

    def test_algorithm_returns_correct_type(self) -> None:
        """Test algorithm() returns AlgorithmSyncRemoteOperations."""
        ops = algorithm()
        assert isinstance(ops, AlgorithmSyncRemoteOperations)
        assert isinstance(ops, SyncRemoteOperations)

    def test_band_returns_correct_type(self) -> None:
        """Test band() returns BandSyncRemoteOperations."""
        ops = band()
        assert isinstance(ops, BandSyncRemoteOperations)
        assert isinstance(ops, SyncRemoteOperations)

    def test_catalog_band_assoc_returns_correct_type(self) -> None:
        """Test catalog_band_assoc() returns correct type."""
        ops = catalog_band_assoc()
        assert isinstance(ops, CatalogBandAssocSyncRemoteOperations)

    def test_catalog_tag_returns_correct_type(self) -> None:
        """Test catalog_tag() returns correct type."""
        ops = catalog_tag()
        assert isinstance(ops, CatalogTagSyncRemoteOperations)

    def test_dataset_returns_correct_type(self) -> None:
        """Test dataset() returns DatasetSyncRemoteOperations."""
        ops = dataset()
        assert isinstance(ops, DatasetSyncRemoteOperations)

    def test_dataset_assoc_returns_correct_type(self) -> None:
        """Test dataset_assoc() returns correct type."""
        ops = dataset_assoc()
        assert isinstance(ops, DatasetAssocSyncRemoteOperations)

    def test_estimates_returns_correct_type(self) -> None:
        """Test estimates() returns correct type."""
        ops = estimates()
        assert isinstance(ops, EstimatesSyncRemoteOperations)

    def test_estimator_returns_correct_type(self) -> None:
        """Test estimator() returns correct type."""
        ops = estimator()
        assert isinstance(ops, EstimatorSyncRemoteOperations)

    def test_model_returns_correct_type(self) -> None:
        """Test model() returns ModelSyncRemoteOperations."""
        ops = model()
        assert isinstance(ops, ModelSyncRemoteOperations)


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

    def test_algorithm_creates_new_instances(self) -> None:
        """Test that calling algorithm() creates new instances."""
        ops1 = algorithm()
        ops2 = algorithm()

        # Should be different wrapper instances
        assert ops1 is not ops2
        # But should wrap the same async instance
        assert ops1.async_ops is ops2.async_ops

    def test_dataset_creates_new_instances(self) -> None:
        """Test that calling dataset() creates new instances."""
        ops1 = dataset()
        ops2 = dataset()

        assert ops1 is not ops2
        assert ops1.async_ops is ops2.async_ops


class TestSyncOperationsHaveMethods:
    """Test that sync operations have expected methods."""

    def test_algorithm_has_crud_methods(self) -> None:
        """Test algorithm sync ops have CRUD methods."""
        ops = algorithm()

        assert hasattr(ops, "get_row")
        assert hasattr(ops, "create_row")
        assert hasattr(ops, "update_row")
        assert hasattr(ops, "delete_row")
        assert callable(ops.get_row)

    def test_dataset_has_filter_methods(self) -> None:
        """Test dataset sync ops have filter methods."""
        ops = dataset()

        assert hasattr(ops, "filter_rows")
        assert hasattr(ops, "count_rows")
        assert callable(ops.filter_rows)


class TestModuleExports:
    """Tests for module-level exports."""

    def test_all_exports_defined(self) -> None:
        """Test that __all__ contains expected exports."""
        from rail_svc.remote_sync import __all__

        expected = [
            "algorithm",
            "band",
            "catalog_band_assoc",
            "catalog_tag",
            "dataset",
            "dataset_assoc",
            "estimates",
            "estimator",
            "model",
            "funcs",
        ]

        assert set(__all__) == set(expected)

    def test_all_exports_are_callable(self) -> None:
        """Test that all exports are callable factory functions."""
        import rail_svc.remote_sync as remote_sync
        from rail_svc.remote_sync import __all__

        for export_name in __all__:
            factory = getattr(remote_sync, export_name)
            assert callable(factory)


class TestFactoryCompleteness:
    """Test that all expected factories exist."""

    def test_all_factories_exist(self) -> None:
        """Test that factories exist for all major tables."""
        factories = [
            algorithm,
            band,
            catalog_band_assoc,
            catalog_tag,
            dataset,
            dataset_assoc,
            estimates,
            estimator,
            model,
        ]

        # All should be callable
        for factory in factories:
            assert callable(factory)
            # All should return SyncRemoteOperations instances
            ops = factory()
            assert isinstance(ops, SyncRemoteOperations)
