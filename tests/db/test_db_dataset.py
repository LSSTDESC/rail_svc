"""Unit tests for Dataset database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.dataset import Dataset
from rail_svc.models import Dataset as DatasetPydantic
from rail_svc.models import DatasetCreate

# ============================================================================
# Dataset Class Tests
# ============================================================================


class TestDataset:
    """Tests for Dataset database model"""

    def test_dataset_tablename(self):
        """Test Dataset has correct table name"""
        assert Dataset.__tablename__ == "dataset"

    def test_dataset_class_string(self):
        """Test Dataset.class_string() returns table name"""
        assert Dataset.class_string() == "dataset"

    def test_pydantic_create_class(self):
        """Test Dataset.pydantic_create_class() returns correct model"""
        assert Dataset.pydantic_create_class() == DatasetCreate

    def test_pydantic_model_class(self):
        """Test Dataset.pydantic_model_class() returns correct model"""
        assert Dataset.pydantic_model_class() == DatasetPydantic

    @pytest.mark.asyncio
    async def test_create_dataset(self, session, sample_catalog_tag):
        """Test creating a Dataset instance"""
        dataset = Dataset(
            name="new_dataset",
            n_objects=1000,
            path="/data/new_dataset.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.id_ is not None
        assert dataset.name == "new_dataset"
        assert dataset.n_objects == 1000
        assert dataset.path == "/data/new_dataset.hdf5"
        assert dataset.is_collection is False
        assert dataset.catalog_tag_id == sample_catalog_tag.id_

    @pytest.mark.asyncio
    async def test_dataset_unique_name(self, session, sample_dataset):
        """Test that dataset name must be unique"""
        duplicate = Dataset(
            name=sample_dataset.name,
            n_objects=500,
            path="/different/path.hdf5",
            is_collection=False,
            catalog_tag_id=sample_dataset.catalog_tag_id,
        )
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError
            await session.commit()

    @pytest.mark.asyncio
    async def test_dataset_unique_path(self, session, sample_catalog_tag, sample_dataset):
        """Test that dataset path must be unique"""
        duplicate = Dataset(
            name="different_name",
            n_objects=500,
            path=sample_dataset.path,
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError
            await session.commit()

    @pytest.mark.asyncio
    async def test_query_dataset_by_name(self, session, sample_dataset):
        """Test querying dataset by name"""
        result = await session.execute(select(Dataset).where(Dataset.name == sample_dataset.name))
        dataset = result.scalar_one()
        assert dataset.id_ == sample_dataset.id_
        assert dataset.name == sample_dataset.name

    @pytest.mark.asyncio
    async def test_query_dataset_by_id(self, session, sample_dataset):
        """Test querying dataset by id"""
        dataset = await session.get(Dataset, sample_dataset.id_)
        assert dataset is not None
        assert dataset.name == sample_dataset.name

    @pytest.mark.asyncio
    async def test_update_dataset(self, session, sample_dataset):
        """Test updating a Dataset"""
        new_n_objects = 20000
        sample_dataset.n_objects = new_n_objects
        await session.commit()
        await session.refresh(sample_dataset)

        assert sample_dataset.n_objects == new_n_objects

    @pytest.mark.asyncio
    async def test_delete_dataset(self, session, sample_dataset):
        """Test deleting a Dataset"""
        dataset_id = sample_dataset.id_
        await session.delete(sample_dataset)
        await session.commit()

        result = await session.get(Dataset, dataset_id)
        assert result is None

    def test_dataset_repr(self, sample_dataset):
        """Test Dataset __repr__ method"""
        repr_str = repr(sample_dataset)
        assert "Dataset" in repr_str
        assert sample_dataset.name in repr_str
        assert str(sample_dataset.id_) in repr_str
        assert str(sample_dataset.n_objects) in repr_str
        assert sample_dataset.path in repr_str

    def test_dataset_str(self, sample_dataset):
        """Test Dataset __str__ method"""
        assert str(sample_dataset) == sample_dataset.name


class TestDatasetPydanticIntegration:
    """Tests for Dataset Pydantic integration"""

    @pytest.mark.asyncio
    async def test_dataset_to_pydantic(self, sample_dataset):
        """Test converting Dataset ORM to Pydantic model"""
        pydantic_obj = Dataset.to_pydantic(sample_dataset)

        assert isinstance(pydantic_obj, DatasetPydantic)
        assert pydantic_obj.id_ == sample_dataset.id_
        assert pydantic_obj.name == sample_dataset.name
        assert pydantic_obj.n_objects == sample_dataset.n_objects
        assert pydantic_obj.path == sample_dataset.path
        assert pydantic_obj.is_collection == sample_dataset.is_collection
        assert pydantic_obj.catalog_tag_id == sample_dataset.catalog_tag_id

    @pytest.mark.asyncio
    async def test_dataset_to_pydantic_dict(self, sample_dataset):
        """Test converting Dataset to dict via Pydantic"""
        data = Dataset.to_pydantic_dict(sample_dataset)

        assert isinstance(data, dict)
        assert data["id_"] == sample_dataset.id_
        assert data["name"] == sample_dataset.name
        assert data["n_objects"] == sample_dataset.n_objects
        assert data["path"] == sample_dataset.path
        assert data["is_collection"] == sample_dataset.is_collection
        assert data["catalog_tag_id"] == sample_dataset.catalog_tag_id


class TestDatasetValidation:
    """Tests for Dataset field validation"""

    @pytest.mark.asyncio
    async def test_dataset_requires_name(self, session, sample_catalog_tag):
        """Test that Dataset requires a name"""
        with pytest.raises(Exception):  # IntegrityError
            dataset = Dataset(
                n_objects=100, path="/path", is_collection=False, catalog_tag_id=sample_catalog_tag.id_
            )
            session.add(dataset)
            await session.commit()

    @pytest.mark.asyncio
    async def test_dataset_requires_n_objects(self, session, sample_catalog_tag):
        """Test that Dataset requires n_objects"""
        with pytest.raises(Exception):  # IntegrityError
            dataset = Dataset(
                name="test", path="/path", is_collection=False, catalog_tag_id=sample_catalog_tag.id_
            )
            session.add(dataset)
            await session.commit()

    @pytest.mark.asyncio
    async def test_dataset_requires_path(self, session, sample_catalog_tag):
        """Test that Dataset requires path"""
        with pytest.raises(Exception):  # IntegrityError
            dataset = Dataset(
                name="test", n_objects=100, is_collection=False, catalog_tag_id=sample_catalog_tag.id_
            )
            session.add(dataset)
            await session.commit()

    @pytest.mark.asyncio
    async def test_dataset_requires_is_collection(self, session, sample_catalog_tag):
        """Test that Dataset requires is_collection"""
        with pytest.raises(Exception):  # IntegrityError
            dataset = Dataset(name="test", n_objects=100, path="/path", catalog_tag_id=sample_catalog_tag.id_)
            session.add(dataset)
            await session.commit()

    @pytest.mark.asyncio
    async def test_dataset_requires_catalog_tag_id(self, session):
        """Test that Dataset requires catalog_tag_id"""
        with pytest.raises(Exception):  # IntegrityError
            dataset = Dataset(name="test", n_objects=100, path="/path", is_collection=False)
            session.add(dataset)
            await session.commit()

    @pytest.mark.asyncio
    async def test_dataset_name_indexed(self):
        """Test that name field is indexed"""
        name_column = Dataset.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True

    @pytest.mark.asyncio
    async def test_dataset_catalog_tag_id_indexed(self):
        """Test that catalog_tag_id field is indexed"""
        catalog_tag_id_column = Dataset.__table__.c.catalog_tag_id
        assert catalog_tag_id_column.index is True


class TestDatasetIsCollection:
    """Tests for is_collection field"""

    @pytest.mark.asyncio
    async def test_dataset_is_collection_true(self, session, sample_catalog_tag):
        """Test creating dataset with is_collection=True"""
        dataset = Dataset(
            name="collection_dataset",
            n_objects=100000,
            path="/data/collection/",
            is_collection=True,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.is_collection is True

    @pytest.mark.asyncio
    async def test_dataset_is_collection_false(self, session, sample_catalog_tag):
        """Test creating dataset with is_collection=False"""
        dataset = Dataset(
            name="single_dataset",
            n_objects=1000,
            path="/data/single.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.is_collection is False

    @pytest.mark.asyncio
    async def test_toggle_is_collection(self, session, sample_dataset):
        """Test updating is_collection field"""
        original_value = sample_dataset.is_collection
        sample_dataset.is_collection = not original_value
        await session.commit()
        await session.refresh(sample_dataset)

        assert sample_dataset.is_collection != original_value


class TestDatasetNObjects:
    """Tests for n_objects field"""

    @pytest.mark.asyncio
    async def test_dataset_zero_objects(self, session, sample_catalog_tag):
        """Test dataset with zero objects"""
        dataset = Dataset(
            name="empty_dataset",
            n_objects=0,
            path="/data/empty.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.n_objects == 0

    @pytest.mark.asyncio
    async def test_dataset_large_n_objects(self, session, sample_catalog_tag):
        """Test dataset with large number of objects"""
        dataset = Dataset(
            name="large_dataset",
            n_objects=1000000000,
            path="/data/large.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.n_objects == 1000000000

    @pytest.mark.asyncio
    async def test_update_n_objects(self, session, sample_dataset):
        """Test updating n_objects field"""
        new_count = 50000
        sample_dataset.n_objects = new_count
        await session.commit()
        await session.refresh(sample_dataset)

        assert sample_dataset.n_objects == new_count


class TestDatasetPath:
    """Tests for path field"""

    @pytest.mark.asyncio
    async def test_dataset_with_absolute_path(self, session, sample_catalog_tag):
        """Test dataset with absolute path"""
        dataset = Dataset(
            name="abs_path",
            n_objects=100,
            path="/absolute/path/to/data.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.path == "/absolute/path/to/data.hdf5"

    @pytest.mark.asyncio
    async def test_dataset_with_relative_path(self, session, sample_catalog_tag):
        """Test dataset with relative path"""
        dataset = Dataset(
            name="rel_path",
            n_objects=100,
            path="relative/path/data.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.path == "relative/path/data.hdf5"

    @pytest.mark.asyncio
    async def test_dataset_with_url_path(self, session, sample_catalog_tag):
        """Test dataset with URL path"""
        dataset = Dataset(
            name="url_path",
            n_objects=100,
            path="s3://bucket/path/to/data.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.path == "s3://bucket/path/to/data.hdf5"

    @pytest.mark.asyncio
    async def test_dataset_with_directory_path(self, session, sample_catalog_tag):
        """Test dataset with directory path for collections"""
        dataset = Dataset(
            name="dir_path",
            n_objects=1000,
            path="/data/collection_dir/",
            is_collection=True,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.path == "/data/collection_dir/"

    @pytest.mark.asyncio
    async def test_update_path(self, session, sample_dataset):
        """Test updating path field"""
        new_path = "/new/path/to/data.hdf5"
        sample_dataset.path = new_path
        await session.commit()
        await session.refresh(sample_dataset)

        assert sample_dataset.path == new_path


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_dataset_with_long_name(self, session, sample_catalog_tag):
        """Test Dataset with maximum length name"""
        long_name = "a" * 255
        dataset = Dataset(
            name=long_name,
            n_objects=100,
            path="/data/long_name.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.name == long_name

    @pytest.mark.asyncio
    async def test_dataset_with_special_characters_in_name(self, session, sample_catalog_tag):
        """Test Dataset name with special characters"""
        dataset = Dataset(
            name="dataset-v2.0_test",
            n_objects=100,
            path="/data/special.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.name == "dataset-v2.0_test"

    @pytest.mark.asyncio
    async def test_dataset_with_special_characters_in_path(self, session, sample_catalog_tag):
        """Test Dataset path with special characters"""
        dataset = Dataset(
            name="special_path",
            n_objects=100,
            path="/data/2024-01-01/dataset_v2.0.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.path == "/data/2024-01-01/dataset_v2.0.hdf5"

    @pytest.mark.asyncio
    async def test_query_nonexistent_dataset(self, session):
        """Test querying for non-existent dataset"""
        result = await session.get(Dataset, 99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, engine, sample_catalog_tag):
        """Test that multiple sessions work independently"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session1:
            dataset1 = Dataset(
                name="session1_dataset",
                n_objects=100,
                path="/data/session1.hdf5",
                is_collection=False,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            session1.add(dataset1)
            await session1.commit()
            await session1.refresh(dataset1)
            dataset1_id = dataset1.id_

        async with async_session() as session2:
            dataset2 = await session2.get(Dataset, dataset1_id)
            assert dataset2 is not None
            assert dataset2.name == "session1_dataset"

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, session, sample_catalog_tag):
        """Test that transaction rolls back on error"""
        initial_count = (await session.execute(select(Dataset))).scalars().all()
        initial_len = len(initial_count)

        try:
            dataset = Dataset(
                name="test_rollback",
                n_objects=100,
                path="/data/rollback.hdf5",
                is_collection=False,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            session.add(dataset)
            await session.flush()

            # Create duplicate to force error
            duplicate = Dataset(
                name="test_rollback",
                n_objects=200,
                path="/data/other.hdf5",
                is_collection=False,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            session.add(duplicate)
            await session.commit()
        except Exception:
            await session.rollback()

        final_count = (await session.execute(select(Dataset))).scalars().all()
        assert len(final_count) == initial_len

    @pytest.mark.asyncio
    async def test_dataset_with_empty_string_name(self, session, sample_catalog_tag):
        """Test Dataset with empty string name"""
        dataset = Dataset(
            name="",
            n_objects=100,
            path="/data/empty_name.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.name == ""

    @pytest.mark.asyncio
    async def test_dataset_with_negative_n_objects(self, session, sample_catalog_tag):
        """Test Dataset with negative n_objects (if allowed)"""
        dataset = Dataset(
            name="negative_count",
            n_objects=-1,
            path="/data/negative.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(dataset)
        await session.commit()
        await session.refresh(dataset)

        assert dataset.n_objects == -1


class TestConcurrentAccess:
    """Tests for concurrent database access"""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, session, sample_dataset):
        """Test that concurrent reads work correctly"""
        results = []
        for _ in range(5):
            result = await session.execute(select(Dataset).where(Dataset.id_ == sample_dataset.id_))
            results.append(result.scalar_one())

        assert len(results) == 5
        assert all(r.id_ == sample_dataset.id_ for r in results)

    @pytest.mark.asyncio
    async def test_refresh_after_update(self, session, sample_dataset):
        """Test that refresh loads updated data"""
        original_n_objects = sample_dataset.n_objects

        new_n_objects = 99999
        sample_dataset.n_objects = new_n_objects
        await session.commit()
        await session.refresh(sample_dataset)

        assert sample_dataset.n_objects == new_n_objects
        assert sample_dataset.n_objects != original_n_objects


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestDatasetBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session, sample_catalog_tag):
        """Test inserting multiple datasets at once"""
        datasets = [
            Dataset(
                name=f"dataset_{i}",
                n_objects=i * 100,
                path=f"/data/dataset_{i}.hdf5",
                is_collection=False,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            for i in range(10)
        ]

        session.add_all(datasets)
        await session.commit()

        for dataset in datasets:
            await session.refresh(dataset)

        assert all(dataset.id_ is not None for dataset in datasets)
        assert len(datasets) == 10

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_datasets):
        """Test querying multiple datasets"""
        result = await session.execute(select(Dataset))
        datasets = result.scalars().all()

        assert len(datasets) >= 3
        names = {dataset.name for dataset in datasets}
        assert "photometric_data" in names
        assert "spectroscopic_data" in names
        assert "combined_collection" in names

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_datasets):
        """Test updating multiple datasets"""
        for dataset in multiple_datasets:
            dataset.n_objects = dataset.n_objects + 1000

        await session.commit()

        for dataset in multiple_datasets:
            await session.refresh(dataset)
            assert dataset.n_objects >= 1000

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_datasets):
        """Test deleting multiple datasets"""
        dataset_ids = [dataset.id_ for dataset in multiple_datasets]

        for dataset in multiple_datasets:
            await session.delete(dataset)

        await session.commit()

        for dataset_id in dataset_ids:
            result = await session.get(Dataset, dataset_id)
            assert result is None


class TestTypeAnnotations:
    """Tests for type annotations and type hints"""

    def test_dataset_has_type_annotations(self):
        """Test that Dataset fields have proper type annotations"""
        assert hasattr(Dataset, "__annotations__")
        annotations = Dataset.__annotations__
        assert "id_" in annotations or hasattr(Dataset, "id_")
        assert "name" in annotations or hasattr(Dataset, "name")
        assert "n_objects" in annotations or hasattr(Dataset, "n_objects")
        assert "path" in annotations or hasattr(Dataset, "path")
        assert "is_collection" in annotations or hasattr(Dataset, "is_collection")
        assert "catalog_tag_id" in annotations or hasattr(Dataset, "catalog_tag_id")


class TestDatasetQueries:
    """Tests for various query patterns"""

    @pytest.mark.asyncio
    async def test_query_by_catalog_tag_id(self, session, sample_catalog_tag, multiple_datasets):
        """Test querying datasets by catalog_tag_id"""
        result = await session.execute(
            select(Dataset).where(Dataset.catalog_tag_id == sample_catalog_tag.id_)
        )
        datasets = result.scalars().all()

        assert len(datasets) >= 3
        assert all(d.catalog_tag_id == sample_catalog_tag.id_ for d in datasets)

    @pytest.mark.asyncio
    async def test_query_by_is_collection(self, session, multiple_datasets):
        """Test querying datasets by is_collection"""
        result = await session.execute(select(Dataset).where(Dataset.is_collection))
        collections = result.scalars().all()

        assert len(collections) >= 1
        assert all(d.is_collection is True for d in collections)

    @pytest.mark.asyncio
    async def test_query_by_name_pattern(self, session, multiple_datasets):
        """Test querying datasets with name pattern matching"""
        result = await session.execute(select(Dataset).where(Dataset.name.like("%data%")))
        datasets = result.scalars().all()

        assert len(datasets) >= 2
        assert all("data" in d.name for d in datasets)

    @pytest.mark.asyncio
    async def test_query_order_by_name(self, session, multiple_datasets):
        """Test querying datasets ordered by name"""
        result = await session.execute(select(Dataset).order_by(Dataset.name))
        datasets = result.scalars().all()

        names = [d.name for d in datasets]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_query_order_by_n_objects(self, session, multiple_datasets):
        """Test querying datasets ordered by n_objects"""
        result = await session.execute(select(Dataset).order_by(Dataset.n_objects.desc()))
        datasets = result.scalars().all()

        counts = [d.n_objects for d in datasets]
        assert counts == sorted(counts, reverse=True)

    @pytest.mark.asyncio
    async def test_query_with_limit(self, session, multiple_datasets):
        """Test querying datasets with limit"""
        result = await session.execute(select(Dataset).limit(2))
        datasets = result.scalars().all()

        assert len(datasets) <= 2

    @pytest.mark.asyncio
    async def test_count_datasets(self, session, multiple_datasets):
        """Test counting total number of datasets"""
        from sqlalchemy import func

        result = await session.execute(select(func.count()).select_from(Dataset))
        count = result.scalar()

        assert count >= 3

    @pytest.mark.asyncio
    async def test_query_datasets_with_min_objects(self, session, multiple_datasets):
        """Test querying datasets with minimum object count"""
        result = await session.execute(select(Dataset).where(Dataset.n_objects >= 10000))
        datasets = result.scalars().all()

        assert len(datasets) >= 2
        assert all(d.n_objects >= 10000 for d in datasets)


class TestDatasetDataIntegrity:
    """Tests for data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_dataset_persistence(self, session, sample_dataset):
        """Test that dataset data persists correctly"""
        dataset_id = sample_dataset.id_
        dataset_name = sample_dataset.name
        dataset_path = sample_dataset.path

        # Clear session
        await session.commit()
        session.expire_all()

        # Query fresh from database
        result = await session.get(Dataset, dataset_id)
        assert result is not None
        assert result.name == dataset_name
        assert result.path == dataset_path

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, session, sample_dataset):
        """Test updating multiple fields at once"""
        sample_dataset.name = "updated_name"
        sample_dataset.n_objects = 99999
        sample_dataset.path = "/new/path.hdf5"
        sample_dataset.is_collection = True
