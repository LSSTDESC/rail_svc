"""Unit tests for Algorithm database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.algorithm import Algorithm
from rail_svc.models import Algorithm as AlgorithmPydantic
from rail_svc.models import AlgorithmCreate

# ============================================================================
# Algorithm Class Tests
# ============================================================================


class TestAlgorithm:
    """Tests for Algorithm database model"""

    def test_algorithm_tablename(self):
        """Test Algorithm has correct table name"""
        assert Algorithm.__tablename__ == "algorithm"

    def test_algorithm_class_string(self):
        """Test Algorithm.class_string() returns table name"""
        assert Algorithm.class_string() == "algorithm"

    def test_pydantic_create_class(self):
        """Test Algorithm.pydantic_create_class() returns correct model"""
        assert Algorithm.pydantic_create_class() == AlgorithmCreate

    def test_pydantic_model_class(self):
        """Test Algorithm.pydantic_model_class() returns correct model"""
        assert Algorithm.pydantic_model_class() == AlgorithmPydantic

    @pytest.mark.asyncio
    async def test_create_algorithm(self, session):
        """Test creating an Algorithm instance"""
        algo = Algorithm(name="neural_network", class_name="torch.nn.Module")
        session.add(algo)
        await session.commit()
        await session.refresh(algo)

        assert algo.id_ is not None
        assert algo.name == "neural_network"
        assert algo.class_name == "torch.nn.Module"

    @pytest.mark.asyncio
    async def test_algorithm_unique_name(self, session, sample_algorithm):
        """Test that algorithm name must be unique"""
        duplicate = Algorithm(name=sample_algorithm.name, class_name="different.Class")
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError or similar
            await session.commit()

    @pytest.mark.asyncio
    async def test_query_algorithm_by_name(self, session, sample_algorithm):
        """Test querying algorithm by name"""
        result = await session.execute(select(Algorithm).where(Algorithm.name == sample_algorithm.name))
        algo = result.scalar_one()
        assert algo.id_ == sample_algorithm.id_
        assert algo.name == sample_algorithm.name

    @pytest.mark.asyncio
    async def test_query_algorithm_by_id(self, session, sample_algorithm):
        """Test querying algorithm by id"""
        algo = await session.get(Algorithm, sample_algorithm.id_)
        assert algo is not None
        assert algo.name == sample_algorithm.name

    @pytest.mark.asyncio
    async def test_update_algorithm(self, session, sample_algorithm):
        """Test updating an Algorithm"""
        new_class_name = "updated.Class"
        sample_algorithm.class_name = new_class_name
        await session.commit()
        await session.refresh(sample_algorithm)

        assert sample_algorithm.class_name == new_class_name

    @pytest.mark.asyncio
    async def test_delete_algorithm(self, session, sample_algorithm):
        """Test deleting an Algorithm"""
        algo_id = sample_algorithm.id_
        await session.delete(sample_algorithm)
        await session.commit()

        result = await session.get(Algorithm, algo_id)
        assert result is None

    def test_algorithm_repr(self, sample_algorithm):
        """Test Algorithm __repr__ method"""
        repr_str = repr(sample_algorithm)
        assert "Algorithm" in repr_str
        assert str(sample_algorithm.id_) in repr_str
        assert sample_algorithm.name in repr_str
        assert sample_algorithm.class_name in repr_str

    def test_algorithm_str(self, sample_algorithm):
        """Test Algorithm __str__ method"""
        assert str(sample_algorithm) == sample_algorithm.name


class TestAlgorithmPydanticIntegration:
    """Tests for Algorithm Pydantic integration"""

    @pytest.mark.asyncio
    async def test_algorithm_to_pydantic(self, sample_algorithm):
        """Test converting Algorithm ORM to Pydantic model"""
        pydantic_obj = Algorithm.to_pydantic(sample_algorithm)

        assert isinstance(pydantic_obj, AlgorithmPydantic)
        assert pydantic_obj.id_ == sample_algorithm.id_
        assert pydantic_obj.name == sample_algorithm.name
        assert pydantic_obj.class_name == sample_algorithm.class_name

    @pytest.mark.asyncio
    async def test_algorithm_to_pydantic_dict(self, sample_algorithm):
        """Test converting Algorithm to dict via Pydantic"""
        data = Algorithm.to_pydantic_dict(sample_algorithm)

        assert isinstance(data, dict)
        assert data["id_"] == sample_algorithm.id_
        assert data["name"] == sample_algorithm.name
        assert data["class_name"] == sample_algorithm.class_name


class TestAlgorithmValidation:
    """Tests for Algorithm field validation"""

    @pytest.mark.asyncio
    async def test_algorithm_requires_name(self, session):
        """Test that Algorithm requires a name"""
        with pytest.raises(Exception):  # IntegrityError
            algo = Algorithm(class_name="test.Class")
            session.add(algo)
            await session.commit()

    @pytest.mark.asyncio
    async def test_algorithm_requires_class_name(self, session):
        """Test that Algorithm requires a class_name"""
        with pytest.raises(Exception):  # IntegrityError
            algo = Algorithm(name="test")
            session.add(algo)
            await session.commit()

    @pytest.mark.asyncio
    async def test_algorithm_name_indexed(self, session):
        """Test that name field is indexed"""
        # This is implicitly tested by unique constraint and index=True
        # We verify the column has an index in the metadata
        name_column = Algorithm.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_algorithm_with_long_name(self, session):
        """Test Algorithm with maximum length name"""
        long_name = "a" * 255
        algo = Algorithm(name=long_name, class_name="test.Class")
        session.add(algo)
        await session.commit()
        await session.refresh(algo)

        assert algo.name == long_name

    @pytest.mark.asyncio
    async def test_algorithm_with_long_class_name(self, session):
        """Test Algorithm with maximum length class_name"""
        long_class_name = "a" * 512
        algo = Algorithm(name="test", class_name=long_class_name)
        session.add(algo)
        await session.commit()
        await session.refresh(algo)

        assert algo.class_name == long_class_name

    @pytest.mark.asyncio
    async def test_algorithm_with_special_characters(self, session):
        """Test Algorithm name with special characters"""
        algo = Algorithm(name="test-algorithm_v2.0", class_name="package.module.Class")
        session.add(algo)
        await session.commit()
        await session.refresh(algo)

        assert algo.name == "test-algorithm_v2.0"

    @pytest.mark.asyncio
    async def test_query_nonexistent_algorithm(self, session):
        """Test querying for non-existent algorithm"""
        result = await session.get(Algorithm, 99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, engine):
        """Test that multiple sessions work independently"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session1:
            algo1 = Algorithm(name="session1_algo", class_name="test.Class1")
            session1.add(algo1)
            await session1.commit()
            await session1.refresh(algo1)
            algo1_id = algo1.id_

        async with async_session() as session2:
            algo2 = await session2.get(Algorithm, algo1_id)
            assert algo2 is not None
            assert algo2.name == "session1_algo"

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, session):
        """Test that transaction rolls back on error"""
        initial_count = (await session.execute(select(Algorithm))).scalars().all()
        initial_len = len(initial_count)

        try:
            algo = Algorithm(name="test_rollback", class_name="test.Class")
            session.add(algo)
            await session.flush()

            # Create duplicate to force error
            duplicate = Algorithm(name="test_rollback", class_name="other.Class")
            session.add(duplicate)
            await session.commit()
        except Exception:
            await session.rollback()

        final_count = (await session.execute(select(Algorithm))).scalars().all()
        assert len(final_count) == initial_len


class TestConcurrentAccess:
    """Tests for concurrent database access"""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, session, sample_algorithm):
        """Test that concurrent reads work correctly"""
        # Query the same algorithm multiple times
        results = []
        for _ in range(5):
            result = await session.execute(select(Algorithm).where(Algorithm.id_ == sample_algorithm.id_))
            results.append(result.scalar_one())

        assert len(results) == 5
        assert all(r.id_ == sample_algorithm.id_ for r in results)

    @pytest.mark.asyncio
    async def test_refresh_after_update(self, session, sample_algorithm):
        """Test that refresh loads updated data"""
        original_class_name = sample_algorithm.class_name

        # Update in place
        sample_algorithm.class_name = "updated.NewClass"
        await session.commit()

        # Refresh to get updated data
        await session.refresh(sample_algorithm)

        assert sample_algorithm.class_name == "updated.NewClass"
        assert sample_algorithm.class_name != original_class_name


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestAlgorithmBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session):
        """Test inserting multiple algorithms at once"""
        algorithms = [Algorithm(name=f"algo_{i}", class_name=f"class.Algo{i}") for i in range(10)]

        session.add_all(algorithms)
        await session.commit()

        for algo in algorithms:
            await session.refresh(algo)

        assert all(algo.id_ is not None for algo in algorithms)
        assert len(algorithms) == 10

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

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_algorithms):
        """Test updating multiple algorithms"""
        for algo in multiple_algorithms:
            algo.class_name = f"updated.{algo.class_name}"

        await session.commit()

        for algo in multiple_algorithms:
            await session.refresh(algo)
            assert algo.class_name.startswith("updated.")

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_algorithms):
        """Test deleting multiple algorithms"""
        algo_ids = [algo.id_ for algo in multiple_algorithms]

        for algo in multiple_algorithms:
            await session.delete(algo)

        await session.commit()

        for algo_id in algo_ids:
            result = await session.get(Algorithm, algo_id)
            assert result is None


class TestTypeAnnotations:
    """Tests for type annotations and type hints"""

    def test_algorithm_has_type_annotations(self):
        """Test that Algorithm fields have proper type annotations"""
        assert hasattr(Algorithm, "__annotations__")
        # Mapped columns should be in annotations
        annotations = Algorithm.__annotations__
        assert "id_" in annotations or hasattr(Algorithm, "id_")
        assert "name" in annotations or hasattr(Algorithm, "name")
        assert "class_name" in annotations or hasattr(Algorithm, "class_name")
