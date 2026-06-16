"""Unit tests for Algorithm database model — entity-specific tests only.

Common tests (tablename, pydantic, CRUD, edge cases) are in test_db_shared.py.
"""

import pytest
from sqlalchemy import select

from rail_svc.db.algorithm import Algorithm


class TestAlgorithmValidation:
    """Tests for Algorithm field validation"""

    @pytest.mark.asyncio
    async def test_algorithm_requires_name(self, session):
        """Test that Algorithm requires a name"""
        with pytest.raises(Exception):
            algo = Algorithm(class_name="test.Class")
            session.add(algo)
            await session.commit()

    @pytest.mark.asyncio
    async def test_algorithm_requires_class_name(self, session):
        """Test that Algorithm requires a class_name"""
        with pytest.raises(Exception):
            algo = Algorithm(name="test")
            session.add(algo)
            await session.commit()

    @pytest.mark.asyncio
    async def test_algorithm_unique_name(self, session, sample_algorithm):
        """Test that algorithm name must be unique"""
        duplicate = Algorithm(name=sample_algorithm.name, class_name="different.Class")
        session.add(duplicate)
        with pytest.raises(Exception):
            await session.commit()

    @pytest.mark.asyncio
    async def test_algorithm_name_indexed(self, session):
        """Test that name field is indexed"""
        name_column = Algorithm.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True


class TestAlgorithmBatch:
    """Tests for batch operations with Algorithm-specific data"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session):
        """Test inserting multiple algorithms at once"""
        algorithms = [Algorithm(name=f"algo_{i}", class_name=f"class.Algo{i}") for i in range(10)]
        session.add_all(algorithms)
        await session.commit()

        for algo in algorithms:
            await session.refresh(algo)

        assert all(algo.id_ is not None for algo in algorithms)

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_algorithms):
        """Test querying multiple algorithms"""
        result = await session.execute(select(Algorithm))
        algos = result.scalars().all()

        assert len(algos) >= 3
        names = {algo.name for algo in algos}
        assert "knn" in names
        assert "random_forest" in names
        assert "xgboost" in names
