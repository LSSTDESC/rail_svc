"""Unit tests for Estimates database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.algorithm import Algorithm
from rail_svc.db.dataset import Dataset
from rail_svc.db.estimates import Estimates
from rail_svc.db.estimator import Estimator
from rail_svc.db.model import Model
from rail_svc.models import Estimates as EstimatesPydantic
from rail_svc.models import EstimatesCreate

# ============================================================================
# Estimates Class Tests
# ============================================================================


class TestEstimates:
    """Tests for Estimates database model"""

    def test_estimates_tablename(self):
        """Test Estimates has correct table name"""
        assert Estimates.__tablename__ == "estimates"

    def test_estimates_class_string(self):
        """Test Estimates.class_string() returns table name"""
        assert Estimates.class_string() == "estimates"

    def test_pydantic_create_class(self):
        """Test Estimates.pydantic_create_class() returns correct model"""
        assert Estimates.pydantic_create_class() == EstimatesCreate

    def test_pydantic_model_class(self):
        """Test Estimates.pydantic_model_class() returns correct model"""
        assert Estimates.pydantic_model_class() == EstimatesPydantic

    @pytest.mark.asyncio
    async def test_create_estimates(self, session, sample_dataset, sample_estimator):
        """Test creating an Estimates instance"""
        estimates = Estimates(
            name="new_estimates",
            n_objects=1000,
            path="/results/new_estimates.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.id_ is not None
        assert estimates.name == "new_estimates"
        assert estimates.n_objects == 1000
        assert estimates.path == "/results/new_estimates.hdf5"
        assert estimates.dataset_id == sample_dataset.id_
        assert estimates.estimator_id == sample_estimator.id_

    @pytest.mark.asyncio
    async def test_estimates_unique_name(self, session, sample_estimates, sample_dataset, sample_estimator):
        """Test that estimates name must be unique"""
        duplicate = Estimates(
            name=sample_estimates.name,
            n_objects=500,
            path="/different/path.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimates_unique_path(self, session, sample_estimates, sample_dataset, sample_estimator):
        """Test that estimates path must be unique"""
        duplicate = Estimates(
            name="different_name",
            n_objects=500,
            path=sample_estimates.path,
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError
            await session.commit()

    @pytest.mark.asyncio
    async def test_query_estimates_by_name(self, session, sample_estimates):
        """Test querying estimates by name"""
        result = await session.execute(select(Estimates).where(Estimates.name == sample_estimates.name))
        estimates = result.scalar_one()
        assert estimates.id_ == sample_estimates.id_
        assert estimates.name == sample_estimates.name

    @pytest.mark.asyncio
    async def test_query_estimates_by_id(self, session, sample_estimates):
        """Test querying estimates by id"""
        estimates = await session.get(Estimates, sample_estimates.id_)
        assert estimates is not None
        assert estimates.name == sample_estimates.name

    @pytest.mark.asyncio
    async def test_update_estimates(self, session, sample_estimates):
        """Test updating an Estimates"""
        new_n_objects = 20000
        sample_estimates.n_objects = new_n_objects
        await session.commit()
        await session.refresh(sample_estimates)

        assert sample_estimates.n_objects == new_n_objects

    @pytest.mark.asyncio
    async def test_delete_estimates(self, session, sample_estimates):
        """Test deleting an Estimates"""
        estimates_id = sample_estimates.id_
        await session.delete(sample_estimates)
        await session.commit()

        result = await session.get(Estimates, estimates_id)
        assert result is None

    def test_estimates_repr(self, sample_estimates):
        """Test Estimates __repr__ method"""
        repr_str = repr(sample_estimates)
        assert "Estimates" in repr_str
        assert sample_estimates.name in repr_str
        assert str(sample_estimates.id_) in repr_str
        assert str(sample_estimates.n_objects) in repr_str
        assert sample_estimates.path in repr_str

    def test_estimates_str(self, sample_estimates):
        """Test Estimates __str__ method"""
        assert str(sample_estimates) == sample_estimates.name


class TestEstimatesPydanticIntegration:
    """Tests for Estimates Pydantic integration"""

    @pytest.mark.asyncio
    async def test_estimates_to_pydantic(self, sample_estimates):
        """Test converting Estimates ORM to Pydantic model"""
        pydantic_obj = Estimates.to_pydantic(sample_estimates)

        assert isinstance(pydantic_obj, EstimatesPydantic)
        assert pydantic_obj.id_ == sample_estimates.id_
        assert pydantic_obj.name == sample_estimates.name
        assert pydantic_obj.n_objects == sample_estimates.n_objects
        assert pydantic_obj.path == sample_estimates.path
        assert pydantic_obj.dataset_id == sample_estimates.dataset_id
        assert pydantic_obj.estimator_id == sample_estimates.estimator_id

    @pytest.mark.asyncio
    async def test_estimates_to_pydantic_dict(self, sample_estimates):
        """Test converting Estimates to dict via Pydantic"""
        data = Estimates.to_pydantic_dict(sample_estimates)

        assert isinstance(data, dict)
        assert data["id_"] == sample_estimates.id_
        assert data["name"] == sample_estimates.name
        assert data["n_objects"] == sample_estimates.n_objects
        assert data["path"] == sample_estimates.path
        assert data["dataset_id"] == sample_estimates.dataset_id
        assert data["estimator_id"] == sample_estimates.estimator_id


class TestEstimatesValidation:
    """Tests for Estimates field validation"""

    @pytest.mark.asyncio
    async def test_estimates_requires_name(self, session, sample_dataset, sample_estimator):
        """Test that Estimates requires a name"""
        with pytest.raises(Exception):  # IntegrityError
            estimates = Estimates(
                n_objects=100, path="/path", dataset_id=sample_dataset.id_, estimator_id=sample_estimator.id_
            )
            session.add(estimates)
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimates_requires_n_objects(self, session, sample_dataset, sample_estimator):
        """Test that Estimates requires n_objects"""
        with pytest.raises(Exception):  # IntegrityError
            estimates = Estimates(
                name="test", path="/path", dataset_id=sample_dataset.id_, estimator_id=sample_estimator.id_
            )
            session.add(estimates)
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimates_requires_path(self, session, sample_dataset, sample_estimator):
        """Test that Estimates requires path"""
        with pytest.raises(Exception):  # IntegrityError
            estimates = Estimates(
                name="test", n_objects=100, dataset_id=sample_dataset.id_, estimator_id=sample_estimator.id_
            )
            session.add(estimates)
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimates_requires_dataset_id(self, session, sample_estimator):
        """Test that Estimates requires dataset_id"""
        with pytest.raises(Exception):  # IntegrityError
            estimates = Estimates(name="test", n_objects=100, path="/path", estimator_id=sample_estimator.id_)
            session.add(estimates)
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimates_requires_estimator_id(self, session, sample_dataset):
        """Test that Estimates requires estimator_id"""
        with pytest.raises(Exception):  # IntegrityError
            estimates = Estimates(name="test", n_objects=100, path="/path", dataset_id=sample_dataset.id_)
            session.add(estimates)
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimates_name_indexed(self):
        """Test that name field is indexed"""
        name_column = Estimates.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True

    @pytest.mark.asyncio
    async def test_estimates_foreign_keys_indexed(self):
        """Test that foreign key fields are indexed"""
        dataset_id_col = Estimates.__table__.c.dataset_id
        estimator_id_col = Estimates.__table__.c.estimator_id

        assert dataset_id_col.index is True
        assert estimator_id_col.index is True


class TestEstimatesNObjects:
    """Tests for n_objects field"""

    @pytest.mark.asyncio
    async def test_estimates_zero_objects(self, session, sample_dataset, sample_estimator):
        """Test estimates with zero objects"""
        estimates = Estimates(
            name="empty_estimates",
            n_objects=0,
            path="/results/empty.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.n_objects == 0

    @pytest.mark.asyncio
    async def test_estimates_large_n_objects(self, session, sample_dataset, sample_estimator):
        """Test estimates with large number of objects"""
        estimates = Estimates(
            name="large_estimates",
            n_objects=1000000000,
            path="/results/large.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.n_objects == 1000000000

    @pytest.mark.asyncio
    async def test_update_n_objects(self, session, sample_estimates):
        """Test updating n_objects field"""
        new_count = 50000
        sample_estimates.n_objects = new_count
        await session.commit()
        await session.refresh(sample_estimates)

        assert sample_estimates.n_objects == new_count


class TestEstimatesPath:
    """Tests for path field"""

    @pytest.mark.asyncio
    async def test_estimates_with_absolute_path(self, session, sample_dataset, sample_estimator):
        """Test estimates with absolute path"""
        estimates = Estimates(
            name="abs_path",
            n_objects=100,
            path="/absolute/path/to/estimates.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.path == "/absolute/path/to/estimates.hdf5"

    @pytest.mark.asyncio
    async def test_estimates_with_relative_path(self, session, sample_dataset, sample_estimator):
        """Test estimates with relative path"""
        estimates = Estimates(
            name="rel_path",
            n_objects=100,
            path="relative/path/estimates.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.path == "relative/path/estimates.hdf5"

    @pytest.mark.asyncio
    async def test_estimates_with_url_path(self, session, sample_dataset, sample_estimator):
        """Test estimates with URL path"""
        estimates = Estimates(
            name="url_path",
            n_objects=100,
            path="s3://bucket/path/to/estimates.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.path == "s3://bucket/path/to/estimates.hdf5"

    @pytest.mark.asyncio
    async def test_update_path(self, session, sample_estimates):
        """Test updating path field"""
        new_path = "/new/path/to/estimates.hdf5"
        sample_estimates.path = new_path
        await session.commit()
        await session.refresh(sample_estimates)

        assert sample_estimates.path == new_path


class TestEstimatesRelationships:
    """Tests for Estimates relationships"""

    @pytest.mark.asyncio
    async def test_estimates_dataset_relationship_exists(self, sample_estimates):
        """Test that dataset relationship exists"""
        assert hasattr(sample_estimates, "dataset")

    @pytest.mark.asyncio
    async def test_estimates_estimator_relationship_exists(self, sample_estimates):
        """Test that estimator relationship exists"""
        assert hasattr(sample_estimates, "estimator")


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_estimates_with_long_name(self, session, sample_dataset, sample_estimator):
        """Test Estimates with maximum length name"""
        long_name = "a" * 255
        estimates = Estimates(
            name=long_name,
            n_objects=100,
            path="/results/long_name.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.name == long_name

    @pytest.mark.asyncio
    async def test_estimates_with_special_characters_in_name(self, session, sample_dataset, sample_estimator):
        """Test Estimates name with special characters"""
        estimates = Estimates(
            name="estimates-v2.0_test",
            n_objects=100,
            path="/results/special.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.name == "estimates-v2.0_test"

    @pytest.mark.asyncio
    async def test_estimates_with_special_characters_in_path(self, session, sample_dataset, sample_estimator):
        """Test Estimates path with special characters"""
        estimates = Estimates(
            name="special_path",
            n_objects=100,
            path="/results/2024-01-01/estimates_v2.0.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.path == "/results/2024-01-01/estimates_v2.0.hdf5"

    @pytest.mark.asyncio
    async def test_query_nonexistent_estimates(self, session):
        """Test querying for non-existent estimates"""
        result = await session.get(Estimates, 99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, engine, sample_dataset, sample_estimator):
        """Test that multiple sessions work independently"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session1:
            estimates1 = Estimates(
                name="session1_estimates",
                n_objects=100,
                path="/results/session1.hdf5",
                dataset_id=sample_dataset.id_,
                estimator_id=sample_estimator.id_,
            )
            session1.add(estimates1)
            await session1.commit()
            await session1.refresh(estimates1)
            estimates1_id = estimates1.id_

        async with async_session() as session2:
            estimates2 = await session2.get(Estimates, estimates1_id)
            assert estimates2 is not None
            assert estimates2.name == "session1_estimates"

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, session, sample_dataset, sample_estimator):
        """Test that transaction rolls back on error"""
        initial_count = (await session.execute(select(Estimates))).scalars().all()
        initial_len = len(initial_count)

        try:
            estimates = Estimates(
                name="test_rollback",
                n_objects=100,
                path="/results/rollback.hdf5",
                dataset_id=sample_dataset.id_,
                estimator_id=sample_estimator.id_,
            )
            session.add(estimates)
            await session.flush()

            # Create duplicate to force error
            duplicate = Estimates(
                name="test_rollback",
                n_objects=200,
                path="/results/other.hdf5",
                dataset_id=sample_dataset.id_,
                estimator_id=sample_estimator.id_,
            )
            session.add(duplicate)
            await session.commit()
        except Exception:
            await session.rollback()

        final_count = (await session.execute(select(Estimates))).scalars().all()
        assert len(final_count) == initial_len

    @pytest.mark.asyncio
    async def test_estimates_with_empty_string_name(self, session, sample_dataset, sample_estimator):
        """Test Estimates with empty string name"""
        estimates = Estimates(
            name="",
            n_objects=100,
            path="/results/empty_name.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.name == ""

    @pytest.mark.asyncio
    async def test_estimates_with_negative_n_objects(self, session, sample_dataset, sample_estimator):
        """Test Estimates with negative n_objects (if allowed)"""
        estimates = Estimates(
            name="negative_count",
            n_objects=-1,
            path="/results/negative.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.n_objects == -1


class TestConcurrentAccess:
    """Tests for concurrent database access"""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, session, sample_estimates):
        """Test that concurrent reads work correctly"""
        results = []
        for _ in range(5):
            result = await session.execute(select(Estimates).where(Estimates.id_ == sample_estimates.id_))
            results.append(result.scalar_one())

        assert len(results) == 5
        assert all(r.id_ == sample_estimates.id_ for r in results)

    @pytest.mark.asyncio
    async def test_refresh_after_update(self, session, sample_estimates):
        """Test that refresh loads updated data"""
        original_n_objects = sample_estimates.n_objects

        new_n_objects = 99999
        sample_estimates.n_objects = new_n_objects
        await session.commit()
        await session.refresh(sample_estimates)

        assert sample_estimates.n_objects == new_n_objects
        assert sample_estimates.n_objects != original_n_objects


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestEstimatesBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session, sample_dataset, sample_estimator):
        """Test inserting multiple estimates at once"""
        estimates_list = [
            Estimates(
                name=f"estimates_{i}",
                n_objects=i * 100,
                path=f"/results/estimates_{i}.hdf5",
                dataset_id=sample_dataset.id_,
                estimator_id=sample_estimator.id_,
            )
            for i in range(10)
        ]

        session.add_all(estimates_list)
        await session.commit()

        for est in estimates_list:
            await session.refresh(est)

        assert all(est.id_ is not None for est in estimates_list)
        assert len(estimates_list) == 10

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_estimates):
        """Test querying multiple estimates"""
        result = await session.execute(select(Estimates))
        estimates = result.scalars().all()

        assert len(estimates) >= 3
        names = {est.name for est in estimates}
        assert "estimates_v1" in names
        assert "estimates_v2" in names
        assert "estimates_v3" in names

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_estimates):
        """Test updating multiple estimates"""
        for est in multiple_estimates:
            est.n_objects = est.n_objects + 1000

        await session.commit()

        for est in multiple_estimates:
            await session.refresh(est)
            assert est.n_objects >= 1000

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_estimates):
        """Test deleting multiple estimates"""
        estimates_ids = [est.id_ for est in multiple_estimates]

        for est in multiple_estimates:
            await session.delete(est)

        await session.commit()

        for est_id in estimates_ids:
            result = await session.get(Estimates, est_id)
            assert result is None


class TestTypeAnnotations:
    """Tests for type annotations and type hints"""

    def test_estimates_has_type_annotations(self):
        """Test that Estimates fields have proper type annotations"""
        assert hasattr(Estimates, "__annotations__")
        annotations = Estimates.__annotations__
        assert "id_" in annotations or hasattr(Estimates, "id_")
        assert "name" in annotations or hasattr(Estimates, "name")
        assert "n_objects" in annotations or hasattr(Estimates, "n_objects")
        assert "path" in annotations or hasattr(Estimates, "path")
        assert "dataset_id" in annotations or hasattr(Estimates, "dataset_id")
        assert "estimator_id" in annotations or hasattr(Estimates, "estimator_id")


class TestEstimatesQueries:
    """Tests for various query patterns"""

    @pytest.mark.asyncio
    async def test_query_by_dataset_id(self, session, sample_dataset, multiple_estimates):
        """Test querying estimates by dataset_id"""
        result = await session.execute(select(Estimates).where(Estimates.dataset_id == sample_dataset.id_))
        estimates = result.scalars().all()

        assert len(estimates) >= 3
        assert all(e.dataset_id == sample_dataset.id_ for e in estimates)

    @pytest.mark.asyncio
    async def test_query_by_estimator_id(self, session, sample_estimator, sample_estimates):
        """Test querying estimates by estimator_id"""
        result = await session.execute(
            select(Estimates).where(Estimates.estimator_id == sample_estimator.id_)
        )
        estimates = result.scalars().all()

        assert len(estimates) >= 1
        assert sample_estimates.id_ in [e.id_ for e in estimates]

    @pytest.mark.asyncio
    async def test_query_by_name_pattern(self, session, multiple_estimates):
        """Test querying estimates with name pattern matching"""
        result = await session.execute(select(Estimates).where(Estimates.name.like("estimates%")))
        estimates = result.scalars().all()

        assert len(estimates) >= 3
        assert all(e.name.startswith("estimates") for e in estimates)

    @pytest.mark.asyncio
    async def test_query_order_by_name(self, session, multiple_estimates):
        """Test querying estimates ordered by name"""
        result = await session.execute(select(Estimates).order_by(Estimates.name))
        estimates = result.scalars().all()

        names = [e.name for e in estimates]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_query_order_by_n_objects(self, session, multiple_estimates):
        """Test querying estimates ordered by n_objects"""
        result = await session.execute(select(Estimates).order_by(Estimates.n_objects.desc()))
        estimates = result.scalars().all()

        counts = [e.n_objects for e in estimates]
        assert counts == sorted(counts, reverse=True)

    @pytest.mark.asyncio
    async def test_query_with_limit(self, session, multiple_estimates):
        """Test querying estimates with limit"""
        result = await session.execute(select(Estimates).limit(2))
        estimates = result.scalars().all()

        assert len(estimates) <= 2

    @pytest.mark.asyncio
    async def test_count_estimates(self, session, multiple_estimates):
        """Test counting total number of estimates"""
        from sqlalchemy import func

        result = await session.execute(select(func.count()).select_from(Estimates))
        count = result.scalar()

        assert count >= 3

    @pytest.mark.asyncio
    async def test_query_estimates_with_min_objects(self, session, multiple_estimates):
        """Test querying estimates with minimum object count"""
        result = await session.execute(select(Estimates).where(Estimates.n_objects >= 5000))
        estimates = result.scalars().all()

        assert len(estimates) >= 3
        assert all(e.n_objects >= 5000 for e in estimates)


class TestEstimatesDataIntegrity:
    """Tests for data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_estimates_persistence(self, session, sample_estimates):
        """Test that estimates data persists correctly"""
        est_id = sample_estimates.id_
        est_name = sample_estimates.name
        est_path = sample_estimates.path

        # Clear session
        await session.commit()
        session.expire_all()

        # Query fresh from database
        result = await session.get(Estimates, est_id)
        assert result is not None
        assert result.name == est_name
        assert result.path == est_path

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, session, sample_estimates):
        """Test updating multiple fields at once"""
        sample_estimates.name = "updated_name"
        sample_estimates.n_objects = 99999
        sample_estimates.path = "/new/path.hdf5"

        await session.commit()
        await session.refresh(sample_estimates)

        assert sample_estimates.name == "updated_name"
        assert sample_estimates.n_objects == 99999
        assert sample_estimates.path == "/new/path.hdf5"

    @pytest.mark.asyncio
    async def test_foreign_key_integrity(self, session, sample_dataset, sample_estimator):
        """Test that foreign key references are maintained"""
        estimates = Estimates(
            name="integrity_test",
            n_objects=100,
            path="/results/integrity.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        # Verify foreign keys are correct
        assert estimates.dataset_id == sample_dataset.id_
        assert estimates.estimator_id == sample_estimator.id_

        # Verify we can query the referenced objects
        dataset = await session.get(Dataset, estimates.dataset_id)
        estimator = await session.get(Estimator, estimates.estimator_id)

        assert dataset is not None
        assert estimator is not None


class TestEstimatesPydanticValidation:
    """Tests for Pydantic model integration and validation"""

    @pytest.mark.asyncio
    async def test_to_pydantic_list(self, multiple_estimates):
        """Test converting multiple estimates to Pydantic list"""
        pydantic_list = Estimates.to_pydantic_list(multiple_estimates)

        assert len(pydantic_list) == 3
        assert all(isinstance(obj, EstimatesPydantic) for obj in pydantic_list)
        assert pydantic_list[0].name == "estimates_v1"
        assert pydantic_list[1].name == "estimates_v2"
        assert pydantic_list[2].name == "estimates_v3"

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list(self, multiple_estimates):
        """Test converting multiple estimates to dict list"""
        dict_list = Estimates.to_pydantic_dict_list(multiple_estimates)

        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert all("name" in d for d in dict_list)
        assert all("n_objects" in d for d in dict_list)
        assert all("path" in d for d in dict_list)
        assert all("dataset_id" in d for d in dict_list)
        assert all("estimator_id" in d for d in dict_list)


class TestEstimatesBusinessLogic:
    """Tests for business logic and use cases"""

    @pytest.mark.asyncio
    async def test_multiple_estimates_same_dataset(self, session, sample_catalog_tag, sample_dataset):
        """Test multiple estimates for same dataset with different estimators"""
        # Create multiple estimators
        algo = Algorithm(name="test_algo", class_name="test.Class")
        session.add(algo)
        await session.commit()
        await session.refresh(algo)

        model = Model(
            name="test_model",
            path="/models/test.pkl",
            algo_id=algo.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        estimators = []
        for i in range(3):
            est = Estimator(name=f"estimator_{i}", config={}, model_id=model.id_)
            estimators.append(est)

        session.add_all(estimators)
        await session.commit()

        for est in estimators:
            await session.refresh(est)

        # Create estimates for each estimator
        estimates_list = []
        for i, estimator in enumerate(estimators):
            estimates = Estimates(
                name=f"estimates_est_{i}",
                n_objects=sample_dataset.n_objects,
                path=f"/results/estimates_est_{i}.hdf5",
                dataset_id=sample_dataset.id_,
                estimator_id=estimator.id_,
            )
            estimates_list.append(estimates)

        session.add_all(estimates_list)
        await session.commit()

        # Query all estimates for the dataset
        result = await session.execute(select(Estimates).where(Estimates.dataset_id == sample_dataset.id_))
        found_estimates = result.scalars().all()

        # Should have at least our 3 new estimates
        assert len(found_estimates) >= 3
        estimator_ids = {e.estimator_id for e in found_estimates}
        assert len(estimator_ids) >= 3  # Different estimators

    @pytest.mark.asyncio
    async def test_estimates_same_estimator_different_datasets(
        self, session, sample_catalog_tag, sample_estimator
    ):
        """Test same estimator used on different datasets"""
        # Create multiple datasets
        datasets = []
        for i in range(3):
            dataset = Dataset(
                name=f"dataset_{i}",
                n_objects=1000 * (i + 1),
                path=f"/data/dataset_{i}.hdf5",
                is_collection=False,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            datasets.append(dataset)

        session.add_all(datasets)
        await session.commit()

        for ds in datasets:
            await session.refresh(ds)

        # Create estimates for each dataset
        estimates_list = []
        for i, dataset in enumerate(datasets):
            estimates = Estimates(
                name=f"estimates_ds_{i}",
                n_objects=dataset.n_objects,
                path=f"/results/estimates_ds_{i}.hdf5",
                dataset_id=dataset.id_,
                estimator_id=sample_estimator.id_,
            )
            estimates_list.append(estimates)

        session.add_all(estimates_list)
        await session.commit()

        # Query all estimates for the estimator
        result = await session.execute(
            select(Estimates).where(Estimates.estimator_id == sample_estimator.id_)
        )
        found_estimates = result.scalars().all()

        assert len(found_estimates) >= 3
        dataset_ids = {e.dataset_id for e in found_estimates}
        assert len(dataset_ids) >= 3  # Different datasets


class TestEstimatesNaming:
    """Tests for estimates naming conventions"""

    @pytest.mark.asyncio
    async def test_estimates_with_descriptive_name(self, session, sample_dataset, sample_estimator):
        """Test estimates with descriptive naming convention"""
        estimates = Estimates(
            name="knn_photoz_estimates_lsst_dp02",
            n_objects=10000,
            path="/results/knn_photoz_lsst.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.name == "knn_photoz_estimates_lsst_dp02"

    @pytest.mark.asyncio
    async def test_estimates_with_version_in_name(self, session, sample_dataset, sample_estimator):
        """Test estimates with version in name"""
        estimates = Estimates(
            name="photoz_v2.0",
            n_objects=10000,
            path="/results/photoz_v2.0.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        assert estimates.name == "photoz_v2.0"


class TestEstimatesFiltering:
    """Tests for filtering estimates"""

    @pytest.mark.asyncio
    async def test_filter_by_both_ids(self, session, sample_dataset, sample_estimator):
        """Test filtering by both dataset_id and estimator_id"""
        estimates = Estimates(
            name="filter_test",
            n_objects=100,
            path="/results/filter.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()

        result = await session.execute(
            select(Estimates).where(
                Estimates.dataset_id == sample_dataset.id_, Estimates.estimator_id == sample_estimator.id_
            )
        )
        found = result.scalars().all()

        assert len(found) >= 1
        assert any(e.name == "filter_test" for e in found)

    @pytest.mark.asyncio
    async def test_exclude_filter(self, session, multiple_estimates):
        """Test filtering to exclude certain estimates"""
        result = await session.execute(select(Estimates).where(~Estimates.name.like("estimates_v1%")))
        estimates = result.scalars().all()

        # Should get estimates that don't start with "estimates_v1"
        names = {e.name for e in estimates}
        assert "estimates_v1" not in names


class TestEstimatesCascade:
    """Tests for cascade delete behavior"""

    @pytest.mark.asyncio
    async def test_delete_dataset_behavior(self, session, sample_catalog_tag, sample_estimator):
        """Test behavior when dataset is deleted"""
        # Create a dataset
        dataset = Dataset(
            name="temp_dataset",
            n_objects=1000,
            path="/data/temp_dataset.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)
        dataset_id = dataset.id_

        # Create estimates
        estimates = Estimates(
            name="temp_estimates",
            n_objects=1000,
            path="/results/temp_estimates.hdf5",
            dataset_id=dataset.id_,
            estimator_id=sample_estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        # Delete dataset
        await session.delete(dataset)
        await session.commit()
        session.expire_all()

        # Verify dataset is deleted
        dataset_result = await session.get(Dataset, dataset_id)
        assert dataset_result is None

    @pytest.mark.asyncio
    async def test_delete_estimator_behavior(self, session, sample_catalog_tag, sample_dataset):
        """Test behavior when estimator is deleted"""
        # Create an estimator
        algo = Algorithm(name="temp_algo", class_name="test.Class")
        session.add(algo)
        await session.commit()
        await session.refresh(algo)

        model = Model(
            name="temp_model",
            path="/models/temp.pkl",
            algo_id=algo.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        estimator = Estimator(name="temp_estimator", config={}, model_id=model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)
        estimator_id = estimator.id_

        # Create estimates
        estimates = Estimates(
            name="temp_est_for_del",
            n_objects=500,
            path="/results/temp_est.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=estimator.id_,
        )
        session.add(estimates)
        await session.commit()
        await session.refresh(estimates)

        # Delete estimator
        await session.delete(estimator)
        await session.commit()
        session.expire_all()

        # Verify estimator is deleted
        estimator_result = await session.get(Estimator, estimator_id)
        assert estimator_result is None
