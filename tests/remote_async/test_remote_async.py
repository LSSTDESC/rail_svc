"""Unit tests for remote_async module initialization and pre-configured instances."""

from __future__ import annotations

from rail_svc.remote_async import algorithm, dataset


class TestUsagePatterns:
    """Test common usage patterns work correctly."""

    async def test_can_use_as_context_manager(self) -> None:
        """Test that instances work as async context managers."""
        async with algorithm as ops:
            assert ops is algorithm

    async def test_instances_are_independent(self) -> None:
        """Test that different instances don't interfere with each other."""
        async with algorithm as algo_ops:
            async with dataset as data_ops:
                assert algo_ops.table_name != data_ops.table_name

    async def test_instance_can_be_reused(self) -> None:
        """Test that an instance can be used multiple times."""
        async with algorithm:
            pass

        async with algorithm:
            pass
