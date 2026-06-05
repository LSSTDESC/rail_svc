"""Unit tests for Model database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.algorithm import Algorithm
from rail_svc.db.catalog_tag import CatalogTag
from rail_svc.db.model import Model
from rail_svc.models import Model as ModelPydantic
from rail_svc.models import ModelCreate

# ============================================================================
# Model Class Tests
# ============================================================================


class TestModel:
    """Tests for Model database model"""

    def test_model_tablename(self):
        """Test Model has correct table name"""
        assert Model.__tablename__ == "model"

    def test_model_class_string(self):
        """Test Model.class_string() returns table name"""
        assert Model.class_string() == "model"

    def test_pydantic_create_class(self):
        """Test Model.pydantic_create_class() returns correct model"""
        assert Model.pydantic_create_class() == ModelCreate

    def test_pydantic_model_class(self):
        """Test Model.pydantic_model_class() returns correct model"""
        assert Model.pydantic_model_class() == ModelPydantic

    @pytest.mark.asyncio
    async def test_create_model(self, session, sample_algorithm, sample_catalog_tag):
        """Test creating a Model instance"""
        model = Model(
            name="new_model",
            path="/models/new_model.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.id_ is not None
        assert model.name == "new_model"
        assert model.path == "/models/new_model.pkl"
        assert model.algo_id == sample_algorithm.id_
        assert model.catalog_tag_id == sample_catalog_tag.id_

    @pytest.mark.asyncio
    async def test_model_unique_name(self, session, sample_model, sample_algorithm, sample_catalog_tag):
        """Test that model name must be unique"""
        duplicate = Model(
            name=sample_model.name,
            path="/different/path.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError
            await session.commit()

    @pytest.mark.asyncio
    async def test_query_model_by_name(self, session, sample_model):
        """Test querying model by name"""
        result = await session.execute(select(Model).where(Model.name == sample_model.name))
        model = result.scalar_one()
        assert model.id_ == sample_model.id_
        assert model.name == sample_model.name

    @pytest.mark.asyncio
    async def test_query_model_by_id(self, session, sample_model):
        """Test querying model by id"""
        model = await session.get(Model, sample_model.id_)
        assert model is not None
        assert model.name == sample_model.name

    @pytest.mark.asyncio
    async def test_update_model(self, session, sample_model):
        """Test updating a Model"""
        new_path = "/models/updated.pkl"
        sample_model.path = new_path
        await session.commit()
        await session.refresh(sample_model)

        assert sample_model.path == new_path

    @pytest.mark.asyncio
    async def test_delete_model(self, session, sample_model):
        """Test deleting a Model"""
        model_id = sample_model.id_
        await session.delete(sample_model)
        await session.commit()

        result = await session.get(Model, model_id)
        assert result is None

    def test_model_repr(self, sample_model):
        """Test Model __repr__ method"""
        repr_str = repr(sample_model)
        assert "Model" in repr_str
        assert sample_model.name in repr_str
        assert str(sample_model.id_) in repr_str
        assert str(sample_model.algo_id) in repr_str
        assert str(sample_model.catalog_tag_id) in repr_str
        assert sample_model.path in repr_str

    def test_model_str(self, sample_model):
        """Test Model __str__ method"""
        assert str(sample_model) == sample_model.name


class TestModelPydanticIntegration:
    """Tests for Model Pydantic integration"""

    @pytest.mark.asyncio
    async def test_model_to_pydantic(self, sample_model):
        """Test converting Model ORM to Pydantic model"""
        pydantic_obj = Model.to_pydantic(sample_model)

        assert isinstance(pydantic_obj, ModelPydantic)
        assert pydantic_obj.id_ == sample_model.id_
        assert pydantic_obj.name == sample_model.name
        assert pydantic_obj.path == sample_model.path
        assert pydantic_obj.algo_id == sample_model.algo_id
        assert pydantic_obj.catalog_tag_id == sample_model.catalog_tag_id

    @pytest.mark.asyncio
    async def test_model_to_pydantic_dict(self, sample_model):
        """Test converting Model to dict via Pydantic"""
        data = Model.to_pydantic_dict(sample_model)

        assert isinstance(data, dict)
        assert data["id_"] == sample_model.id_
        assert data["name"] == sample_model.name
        assert data["path"] == sample_model.path
        assert data["algo_id"] == sample_model.algo_id
        assert data["catalog_tag_id"] == sample_model.catalog_tag_id


class TestModelValidation:
    """Tests for Model field validation"""

    @pytest.mark.asyncio
    async def test_model_requires_name(self, session, sample_algorithm, sample_catalog_tag):
        """Test that Model requires a name"""
        with pytest.raises(Exception):  # IntegrityError
            model = Model(path="/path", algo_id=sample_algorithm.id_, catalog_tag_id=sample_catalog_tag.id_)
            session.add(model)
            await session.commit()

    @pytest.mark.asyncio
    async def test_model_requires_path(self, session, sample_algorithm, sample_catalog_tag):
        """Test that Model requires path"""
        with pytest.raises(Exception):  # IntegrityError
            model = Model(name="test", algo_id=sample_algorithm.id_, catalog_tag_id=sample_catalog_tag.id_)
            session.add(model)
            await session.commit()

    @pytest.mark.asyncio
    async def test_model_requires_algo_id(self, session, sample_catalog_tag):
        """Test that Model requires algo_id"""
        with pytest.raises(Exception):  # IntegrityError
            model = Model(name="test", path="/path", catalog_tag_id=sample_catalog_tag.id_)
            session.add(model)
            await session.commit()

    @pytest.mark.asyncio
    async def test_model_requires_catalog_tag_id(self, session, sample_algorithm):
        """Test that Model requires catalog_tag_id"""
        with pytest.raises(Exception):  # IntegrityError
            model = Model(name="test", path="/path", algo_id=sample_algorithm.id_)
            session.add(model)
            await session.commit()

    @pytest.mark.asyncio
    async def test_model_name_indexed(self):
        """Test that name field is indexed"""
        name_column = Model.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True

    @pytest.mark.asyncio
    async def test_model_foreign_keys_indexed(self):
        """Test that foreign key fields are indexed"""
        algo_id_col = Model.__table__.c.algo_id
        catalog_tag_id_col = Model.__table__.c.catalog_tag_id

        assert algo_id_col.index is True
        assert catalog_tag_id_col.index is True


class TestModelPath:
    """Tests for path field"""

    @pytest.mark.asyncio
    async def test_model_with_absolute_path(self, session, sample_algorithm, sample_catalog_tag):
        """Test model with absolute path"""
        model = Model(
            name="abs_path",
            path="/absolute/path/to/model.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.path == "/absolute/path/to/model.pkl"

    @pytest.mark.asyncio
    async def test_model_with_relative_path(self, session, sample_algorithm, sample_catalog_tag):
        """Test model with relative path"""
        model = Model(
            name="rel_path",
            path="relative/path/model.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.path == "relative/path/model.pkl"

    @pytest.mark.asyncio
    async def test_model_with_url_path(self, session, sample_algorithm, sample_catalog_tag):
        """Test model with URL path"""
        model = Model(
            name="url_path",
            path="s3://bucket/path/to/model.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.path == "s3://bucket/path/to/model.pkl"

    @pytest.mark.asyncio
    async def test_update_path(self, session, sample_model):
        """Test updating path field"""
        new_path = "/new/path/to/model.pkl"
        sample_model.path = new_path
        await session.commit()
        await session.refresh(sample_model)

        assert sample_model.path == new_path


class TestModelRelationships:
    """Tests for Model relationships"""

    @pytest.mark.asyncio
    async def test_model_algorithm_relationship_exists(self, sample_model):
        """Test that algorithm relationship exists"""
        assert hasattr(sample_model, "algorithm")

    @pytest.mark.asyncio
    async def test_model_catalog_tag_relationship_exists(self, sample_model):
        """Test that catalog_tag relationship exists"""
        assert hasattr(sample_model, "catalog_tag")


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_model_with_long_name(self, session, sample_algorithm, sample_catalog_tag):
        """Test Model with maximum length name"""
        long_name = "a" * 255
        model = Model(
            name=long_name,
            path="/models/long_name.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.name == long_name

    @pytest.mark.asyncio
    async def test_model_with_special_characters_in_name(self, session, sample_algorithm, sample_catalog_tag):
        """Test Model name with special characters"""
        model = Model(
            name="model-v2.0_test",
            path="/models/special.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.name == "model-v2.0_test"

    @pytest.mark.asyncio
    async def test_model_with_special_characters_in_path(self, session, sample_algorithm, sample_catalog_tag):
        """Test Model path with special characters"""
        model = Model(
            name="special_path",
            path="/models/2024-01-01/model_v2.0.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.path == "/models/2024-01-01/model_v2.0.pkl"

    @pytest.mark.asyncio
    async def test_query_nonexistent_model(self, session):
        """Test querying for non-existent model"""
        result = await session.get(Model, 99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, engine, sample_algorithm, sample_catalog_tag):
        """Test that multiple sessions work independently"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session1:
            model1 = Model(
                name="session1_model",
                path="/models/session1.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            session1.add(model1)
            await session1.commit()
            await session1.refresh(model1)
            model1_id = model1.id_

        async with async_session() as session2:
            model2 = await session2.get(Model, model1_id)
            assert model2 is not None
            assert model2.name == "session1_model"

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, session, sample_algorithm, sample_catalog_tag):
        """Test that transaction rolls back on error"""
        initial_count = (await session.execute(select(Model))).scalars().all()
        initial_len = len(initial_count)

        try:
            model = Model(
                name="test_rollback",
                path="/models/rollback.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            session.add(model)
            await session.flush()

            # Create duplicate to force error
            duplicate = Model(
                name="test_rollback",
                path="/models/other.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            session.add(duplicate)
            await session.commit()
        except Exception:
            await session.rollback()

        final_count = (await session.execute(select(Model))).scalars().all()
        assert len(final_count) == initial_len

    @pytest.mark.asyncio
    async def test_model_with_empty_string_name(self, session, sample_algorithm, sample_catalog_tag):
        """Test Model with empty string name"""
        model = Model(
            name="",
            path="/models/empty_name.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.name == ""


class TestConcurrentAccess:
    """Tests for concurrent database access"""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, session, sample_model):
        """Test that concurrent reads work correctly"""
        results = []
        for _ in range(5):
            result = await session.execute(select(Model).where(Model.id_ == sample_model.id_))
            results.append(result.scalar_one())

        assert len(results) == 5
        assert all(r.id_ == sample_model.id_ for r in results)

    @pytest.mark.asyncio
    async def test_refresh_after_update(self, session, sample_model):
        """Test that refresh loads updated data"""
        original_path = sample_model.path

        new_path = "/models/updated.pkl"
        sample_model.path = new_path
        await session.commit()
        await session.refresh(sample_model)

        assert sample_model.path == new_path
        assert sample_model.path != original_path


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestModelBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session, sample_algorithm, sample_catalog_tag):
        """Test inserting multiple models at once"""
        models = [
            Model(
                name=f"model_{i}",
                path=f"/models/model_{i}.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            for i in range(10)
        ]

        session.add_all(models)
        await session.commit()

        for model in models:
            await session.refresh(model)

        assert all(model.id_ is not None for model in models)
        assert len(models) == 10

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_models):
        """Test querying multiple models"""
        result = await session.execute(select(Model))
        models = result.scalars().all()

        assert len(models) >= 3
        names = {model.name for model in models}
        assert "model_v1" in names
        assert "model_v2" in names
        assert "model_v3" in names

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_models):
        """Test updating multiple models"""
        for model in multiple_models:
            model.path = f"/updated/{model.name}.pkl"

        await session.commit()

        for model in multiple_models:
            await session.refresh(model)
            assert model.path.startswith("/updated/")

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_models):
        """Test deleting multiple models"""
        model_ids = [model.id_ for model in multiple_models]

        for model in multiple_models:
            await session.delete(model)

        await session.commit()

        for model_id in model_ids:
            result = await session.get(Model, model_id)
            assert result is None


class TestTypeAnnotations:
    """Tests for type annotations and type hints"""

    def test_model_has_type_annotations(self):
        """Test that Model fields have proper type annotations"""
        assert hasattr(Model, "__annotations__")
        annotations = Model.__annotations__
        assert "id_" in annotations or hasattr(Model, "id_")
        assert "name" in annotations or hasattr(Model, "name")
        assert "path" in annotations or hasattr(Model, "path")
        assert "algo_id" in annotations or hasattr(Model, "algo_id")
        assert "catalog_tag_id" in annotations or hasattr(Model, "catalog_tag_id")


class TestModelQueries:
    """Tests for various query patterns"""

    @pytest.mark.asyncio
    async def test_query_by_algo_id(self, session, sample_algorithm, multiple_models):
        """Test querying models by algo_id"""
        result = await session.execute(select(Model).where(Model.algo_id == sample_algorithm.id_))
        models = result.scalars().all()

        assert len(models) >= 3
        assert all(m.algo_id == sample_algorithm.id_ for m in models)

    @pytest.mark.asyncio
    async def test_query_by_catalog_tag_id(self, session, sample_catalog_tag, multiple_models):
        """Test querying models by catalog_tag_id"""
        result = await session.execute(select(Model).where(Model.catalog_tag_id == sample_catalog_tag.id_))
        models = result.scalars().all()

        assert len(models) >= 3
        assert all(m.catalog_tag_id == sample_catalog_tag.id_ for m in models)

    @pytest.mark.asyncio
    async def test_query_by_name_pattern(self, session, multiple_models):
        """Test querying models with name pattern matching"""
        result = await session.execute(select(Model).where(Model.name.like("model%")))
        models = result.scalars().all()

        assert len(models) >= 3
        assert all(m.name.startswith("model") for m in models)

    @pytest.mark.asyncio
    async def test_query_order_by_name(self, session, multiple_models):
        """Test querying models ordered by name"""
        result = await session.execute(select(Model).order_by(Model.name))
        models = result.scalars().all()

        names = [m.name for m in models]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_query_with_limit(self, session, multiple_models):
        """Test querying models with limit"""
        result = await session.execute(select(Model).limit(2))
        models = result.scalars().all()

        assert len(models) <= 2

    @pytest.mark.asyncio
    async def test_count_models(self, session, multiple_models):
        """Test counting total number of models"""
        from sqlalchemy import func

        result = await session.execute(select(func.count()).select_from(Model))
        count = result.scalar()

        assert count >= 3


class TestModelDataIntegrity:
    """Tests for data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_model_persistence(self, session, sample_model):
        """Test that model data persists correctly"""
        model_id = sample_model.id_
        model_name = sample_model.name
        model_path = sample_model.path

        # Clear session
        await session.commit()
        session.expire_all()

        # Query fresh from database
        result = await session.get(Model, model_id)
        assert result is not None
        assert result.name == model_name
        assert result.path == model_path

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, session, sample_model):
        """Test updating multiple fields at once"""
        sample_model.name = "updated_name"
        sample_model.path = "/new/path.pkl"

        await session.commit()
        await session.refresh(sample_model)

        assert sample_model.name == "updated_name"
        assert sample_model.path == "/new/path.pkl"

    @pytest.mark.asyncio
    async def test_foreign_key_integrity(self, session, sample_algorithm, sample_catalog_tag):
        """Test that foreign key references are maintained"""
        model = Model(
            name="integrity_test",
            path="/models/integrity.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        # Verify foreign keys are correct
        assert model.algo_id == sample_algorithm.id_
        assert model.catalog_tag_id == sample_catalog_tag.id_

        # Verify we can query the referenced objects
        algorithm = await session.get(Algorithm, model.algo_id)
        catalog_tag = await session.get(CatalogTag, model.catalog_tag_id)

        assert algorithm is not None
        assert catalog_tag is not None


class TestModelPydanticValidation:
    """Tests for Pydantic model integration and validation"""

    @pytest.mark.asyncio
    async def test_to_pydantic_list(self, multiple_models):
        """Test converting multiple models to Pydantic list"""
        pydantic_list = Model.to_pydantic_list(multiple_models)

        assert len(pydantic_list) == 3
        assert all(isinstance(obj, ModelPydantic) for obj in pydantic_list)
        assert pydantic_list[0].name == "model_v1"
        assert pydantic_list[1].name == "model_v2"
        assert pydantic_list[2].name == "model_v3"

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list(self, multiple_models):
        """Test converting multiple models to dict list"""
        dict_list = Model.to_pydantic_dict_list(multiple_models)

        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert all("name" in d for d in dict_list)
        assert all("path" in d for d in dict_list)
        assert all("algo_id" in d for d in dict_list)
        assert all("catalog_tag_id" in d for d in dict_list)


class TestModelBusinessLogic:
    """Tests for business logic and use cases"""

    @pytest.mark.asyncio
    async def test_multiple_models_same_algorithm(self, session, sample_algorithm, sample_catalog_tag):
        """Test multiple models for same algorithm"""
        models = []
        for i in range(3):
            model = Model(
                name=f"algo_variant_{i}",
                path=f"/models/variant_{i}.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            models.append(model)

        session.add_all(models)
        await session.commit()

        # Query all models for the algorithm
        result = await session.execute(select(Model).where(Model.algo_id == sample_algorithm.id_))
        found_models = result.scalars().all()

        assert len(found_models) >= 3
        algo_ids = {m.algo_id for m in found_models}
        assert len(algo_ids) == 1  # All same algorithm

    @pytest.mark.asyncio
    async def test_multiple_models_same_catalog_tag(self, session, sample_algorithm, sample_catalog_tag):
        """Test multiple models for same catalog tag"""
        # Create another algorithm
        algo2 = Algorithm(name="rf", class_name="sklearn.ensemble.RandomForestClassifier")
        session.add(algo2)
        await session.commit()
        await session.refresh(algo2)

        models = [
            Model(
                name="model_knn",
                path="/models/knn.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            ),
            Model(
                name="model_rf",
                path="/models/rf.pkl",
                algo_id=algo2.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            ),
        ]

        session.add_all(models)
        await session.commit()

        # Query all models for the catalog tag
        result = await session.execute(select(Model).where(Model.catalog_tag_id == sample_catalog_tag.id_))
        found_models = result.scalars().all()

        assert len(found_models) >= 2
        catalog_tag_ids = {m.catalog_tag_id for m in found_models}
        assert len(catalog_tag_ids) == 1  # All same catalog tag


class TestModelNaming:
    """Tests for model naming conventions"""

    @pytest.mark.asyncio
    async def test_model_with_descriptive_name(self, session, sample_algorithm, sample_catalog_tag):
        """Test model with descriptive naming convention"""
        model = Model(
            name="knn_lsst_dp02_trained_2024",
            path="/models/knn_lsst_2024.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.name == "knn_lsst_dp02_trained_2024"

    @pytest.mark.asyncio
    async def test_model_with_version_in_name(self, session, sample_algorithm, sample_catalog_tag):
        """Test model with version in name"""
        model = Model(
            name="model_v2.0",
            path="/models/model_v2.0.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        assert model.name == "model_v2.0"


class TestModelFiltering:
    """Tests for filtering models"""

    @pytest.mark.asyncio
    async def test_filter_by_both_foreign_keys(self, session, sample_algorithm, sample_catalog_tag):
        """Test filtering by both algo_id and catalog_tag_id"""
        model = Model(
            name="filter_test",
            path="/models/filter.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()

        result = await session.execute(
            select(Model).where(
                Model.algo_id == sample_algorithm.id_, Model.catalog_tag_id == sample_catalog_tag.id_
            )
        )
        models = result.scalars().all()

        assert len(models) >= 1
        assert any(m.name == "filter_test" for m in models)

    @pytest.mark.asyncio
    async def test_exclude_filter(self, session, multiple_models):
        """Test filtering to exclude certain models"""
        result = await session.execute(select(Model).where(~Model.name.like("model_v1%")))
        models = result.scalars().all()

        # Should get models that don't start with "model_v1"
        names = {m.name for m in models}
        assert "model_v1" not in names


class TestModelPathPatterns:
    """Tests for various path patterns"""

    @pytest.mark.asyncio
    async def test_model_with_different_file_extensions(self, session, sample_algorithm, sample_catalog_tag):
        """Test models with various file extensions"""
        extensions = [".pkl", ".h5", ".joblib", ".pt", ".onnx", ".yaml"]

        for i, ext in enumerate(extensions):
            model = Model(
                name=f"model_ext_{i}",
                path=f"/models/model{ext}",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            session.add(model)

        await session.commit()

        result = await session.execute(select(Model))
        models = result.scalars().all()

        paths = {m.path for m in models}
        assert any(p.endswith(".pkl") for p in paths)
        assert any(p.endswith(".h5") for p in paths)


class TestModelCascade:
    """Tests for cascade delete behavior"""

    @pytest.mark.asyncio
    async def test_delete_algorithm_behavior(self, session, sample_catalog_tag):
        """Test behavior when algorithm is deleted"""
        # Create an algorithm
        algo = Algorithm(name="temp_algo", class_name="test.Class")
        session.add(algo)
        await session.commit()
        await session.refresh(algo)
        algo_id = algo.id_

        # Create model
        model = Model(
            name="temp_model",
            path="/models/temp.pkl",
            algo_id=algo.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        # Delete algorithm
        await session.delete(algo)
        await session.commit()
        session.expire_all()

        # Verify algorithm is deleted
        algo_result = await session.get(Algorithm, algo_id)
        assert algo_result is None

    @pytest.mark.asyncio
    async def test_delete_catalog_tag_behavior(self, session, sample_algorithm):
        """Test behavior when catalog_tag is deleted"""
        # Create a catalog tag
        tag = CatalogTag(name="temp_tag")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)
        tag_id = tag.id_

        # Create model
        model = Model(
            name="temp_model_2",
            path="/models/temp2.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        # Delete catalog tag
        await session.delete(tag)
        await session.commit()
        session.expire_all()

        # Verify catalog tag is deleted
        tag_result = await session.get(CatalogTag, tag_id)
        assert tag_result is None


class TestModelComplexQueries:
    """Tests for complex query scenarios"""

    @pytest.mark.asyncio
    async def test_join_with_algorithm(self, session, sample_model, sample_algorithm):
        """Test querying models with join to algorithm"""
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(Model).options(selectinload(Model.algorithm)).where(Model.id_ == sample_model.id_)
        )
        model = result.scalar_one()

        assert model.algorithm is not None
        assert model.algorithm.id_ == sample_algorithm.id_

    @pytest.mark.asyncio
    async def test_join_with_catalog_tag(self, session, sample_model, sample_catalog_tag):
        """Test querying models with join to catalog_tag"""
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(Model).options(selectinload(Model.catalog_tag)).where(Model.id_ == sample_model.id_)
        )
        model = result.scalar_one()

        assert model.catalog_tag is not None
        assert model.catalog_tag.id_ == sample_catalog_tag.id_

    @pytest.mark.asyncio
    async def test_filter_by_algorithm_name(self, session, sample_model, sample_algorithm):
        """Test filtering models by algorithm name"""
        result = await session.execute(
            select(Model).join(Model.algorithm).where(Algorithm.name == sample_algorithm.name)
        )
        models = result.scalars().all()

        assert len(models) >= 1
        assert sample_model.id_ in [m.id_ for m in models]

    @pytest.mark.asyncio
    async def test_filter_by_catalog_tag_name(self, session, sample_model, sample_catalog_tag):
        """Test filtering models by catalog tag name"""
        result = await session.execute(
            select(Model).join(Model.catalog_tag).where(CatalogTag.name == sample_catalog_tag.name)
        )
        models = result.scalars().all()

        assert len(models) >= 1
        assert sample_model.id_ in [m.id_ for m in models]


class TestModelSorting:
    """Tests for sorting models"""

    @pytest.mark.asyncio
    async def test_sort_ascending(self, session, multiple_models):
        """Test sorting models in ascending order"""
        result = await session.execute(select(Model).order_by(Model.name.asc()))
        models = result.scalars().all()

        names = [m.name for m in models]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_sort_descending(self, session, multiple_models):
        """Test sorting models in descending order"""
        result = await session.execute(select(Model).order_by(Model.name.desc()))
        models = result.scalars().all()

        names = [m.name for m in models]
        assert names == sorted(names, reverse=True)

    @pytest.mark.asyncio
    async def test_sort_by_id(self, session, multiple_models):
        """Test sorting models by id"""
        result = await session.execute(select(Model).order_by(Model.id_))
        models = result.scalars().all()

        ids = [m.id_ for m in models]
        assert ids == sorted(ids)


class TestModelPagination:
    """Tests for pagination"""

    @pytest.mark.asyncio
    async def test_pagination_first_page(self, session, sample_algorithm, sample_catalog_tag):
        """Test getting first page of results"""
        # Create many models
        models = [
            Model(
                name=f"page_test_{i:02d}",
                path=f"/models/page_{i}.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            for i in range(20)
        ]
        session.add_all(models)
        await session.commit()

        result = await session.execute(select(Model).order_by(Model.name).limit(10).offset(0))
        page1 = result.scalars().all()

        assert len(page1) == 10

    @pytest.mark.asyncio
    async def test_pagination_second_page(self, session, sample_algorithm, sample_catalog_tag):
        """Test getting second page of results"""
        # Create many models
        models = [
            Model(
                name=f"page_test2_{i:02d}",
                path=f"/models/page2_{i}.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            for i in range(20)
        ]
        session.add_all(models)
        await session.commit()

        result = await session.execute(select(Model).order_by(Model.name).limit(10).offset(10))
        page2 = result.scalars().all()

        assert len(page2) == 10

    @pytest.mark.asyncio
    async def test_pagination_boundary(self, session, sample_algorithm, sample_catalog_tag):
        """Test pagination at boundary"""
        # Create exact number of models
        models = [
            Model(
                name=f"boundary_{i}",
                path=f"/models/boundary_{i}.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            for i in range(15)
        ]
        session.add_all(models)
        await session.commit()

        result = await session.execute(select(Model).order_by(Model.name).limit(10).offset(10))
        page = result.scalars().all()

        assert len(page) == 5  # Only 5 remaining


class TestModelReprStr:
    """Tests for __repr__ and __str__ edge cases"""

    @pytest.mark.asyncio
    async def test_repr_with_special_characters(self, session, sample_algorithm, sample_catalog_tag):
        """Test __repr__ with special characters in name"""
        model = Model(
            name="test'model\"with\\special",
            path="/models/special.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)

        repr_str = repr(model)
        assert "Model" in repr_str
        assert str(model.id_) in repr_str

    @pytest.mark.asyncio
    async def test_str_returns_name_only(self, sample_model):
        """Test that __str__ returns only the name"""
        str_repr = str(sample_model)
        assert str_repr == sample_model.name
        assert "Model" not in str_repr
        assert str(sample_model.id_) not in str_repr


class TestModelCombinations:
    """Tests for various algorithm and catalog tag combinations"""

    @pytest.mark.asyncio
    async def test_same_algorithm_different_catalogs(self, session, sample_algorithm):
        """Test same algorithm with different catalog tags"""
        tags = [CatalogTag(name=f"tag_{i}") for i in range(3)]
        session.add_all(tags)
        await session.commit()

        for tag in tags:
            await session.refresh(tag)

        models = [
            Model(
                name=f"model_tag_{i}",
                path=f"/models/tag_{i}.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=tag.id_,
            )
            for i, tag in enumerate(tags)
        ]
        session.add_all(models)
        await session.commit()

        # All models use same algorithm
        result = await session.execute(select(Model).where(Model.algo_id == sample_algorithm.id_))
        found_models = result.scalars().all()

        assert len(found_models) >= 3
        catalog_tag_ids = {m.catalog_tag_id for m in found_models}
        assert len(catalog_tag_ids) >= 3  # Different catalog tags

    @pytest.mark.asyncio
    async def test_different_algorithms_same_catalog(self, session, sample_catalog_tag):
        """Test different algorithms with same catalog tag"""
        algos = [Algorithm(name=f"algo_{i}", class_name=f"test.Algo{i}") for i in range(3)]
        session.add_all(algos)
        await session.commit()

        for algo in algos:
            await session.refresh(algo)

        models = [
            Model(
                name=f"model_algo_{i}",
                path=f"/models/algo_{i}.pkl",
                algo_id=algo.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            for i, algo in enumerate(algos)
        ]
        session.add_all(models)
        await session.commit()

        # All models use same catalog tag
        result = await session.execute(select(Model).where(Model.catalog_tag_id == sample_catalog_tag.id_))
        found_models = result.scalars().all()

        assert len(found_models) >= 3
        algo_ids = {m.algo_id for m in found_models}
        assert len(algo_ids) >= 3  # Different algorithms
