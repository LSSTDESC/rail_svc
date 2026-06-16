"""Unit tests for Estimator database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.estimator import Estimator
from rail_svc.db.model import Model
from rail_svc.models import Estimator as EstimatorPydantic
from rail_svc.models import EstimatorCreate

# ============================================================================
# Estimator Class Tests
# ============================================================================


class TestEstimator:
    """Tests for Estimator database model"""

    def test_estimator_tablename(self):
        """Test Estimator has correct table name"""
        assert Estimator.__tablename__ == "estimator"

    def test_estimator_class_string(self):
        """Test Estimator.class_string() returns table name"""
        assert Estimator.class_string() == "estimator"

    def test_pydantic_create_class(self):
        """Test Estimator.pydantic_create_class() returns correct model"""
        assert Estimator.pydantic_create_class() == EstimatorCreate

    def test_pydantic_model_class(self):
        """Test Estimator.pydantic_model_class() returns correct model"""
        assert Estimator.pydantic_model_class() == EstimatorPydantic

    @pytest.mark.asyncio
    async def test_create_estimator(self, session, sample_model):
        """Test creating an Estimator instance"""
        estimator = Estimator(name="new_estimator", config={"learning_rate": 0.01}, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.id_ is not None
        assert estimator.name == "new_estimator"
        assert estimator.config == {"learning_rate": 0.01}
        assert estimator.model_id == sample_model.id_

    @pytest.mark.asyncio
    async def test_estimator_unique_name(self, session, sample_estimator, sample_model):
        """Test that estimator name must be unique"""
        duplicate = Estimator(name=sample_estimator.name, config={}, model_id=sample_model.id_)
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError
            await session.commit()

    @pytest.mark.asyncio
    async def test_query_estimator_by_name(self, session, sample_estimator):
        """Test querying estimator by name"""
        result = await session.execute(select(Estimator).where(Estimator.name == sample_estimator.name))
        estimator = result.scalar_one()
        assert estimator.id_ == sample_estimator.id_
        assert estimator.name == sample_estimator.name

    @pytest.mark.asyncio
    async def test_query_estimator_by_id(self, session, sample_estimator):
        """Test querying estimator by id"""
        estimator = await session.get(Estimator, sample_estimator.id_)
        assert estimator is not None
        assert estimator.name == sample_estimator.name

    @pytest.mark.asyncio
    async def test_update_estimator(self, session, sample_estimator):
        """Test updating an Estimator"""
        new_config = {"n_neighbors": 10}
        sample_estimator.config = new_config
        await session.commit()
        await session.refresh(sample_estimator)

        assert sample_estimator.config == new_config

    @pytest.mark.asyncio
    async def test_delete_estimator(self, session, sample_estimator):
        """Test deleting an Estimator"""
        estimator_id = sample_estimator.id_
        await session.delete(sample_estimator)
        await session.commit()

        result = await session.get(Estimator, estimator_id)
        assert result is None

    def test_estimator_repr(self, sample_estimator):
        """Test Estimator __repr__ method"""
        repr_str = repr(sample_estimator)
        assert "Estimator" in repr_str
        assert str(sample_estimator.id_) in repr_str
        assert sample_estimator.name in repr_str
        assert str(sample_estimator.model_id) in repr_str

    def test_estimator_str(self, sample_estimator):
        """Test Estimator __str__ method"""
        assert str(sample_estimator) == sample_estimator.name


class TestEstimatorValidation:
    """Tests for Estimator field validation"""

    @pytest.mark.asyncio
    async def test_estimator_requires_name(self, session, sample_model):
        """Test that Estimator requires a name"""
        with pytest.raises(Exception):  # IntegrityError
            estimator = Estimator(config={}, model_id=sample_model.id_)
            session.add(estimator)
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimator_requires_model_id(self, session):
        """Test that Estimator requires model_id"""
        with pytest.raises(Exception):  # IntegrityError
            estimator = Estimator(name="test", config={})
            session.add(estimator)
            await session.commit()

    @pytest.mark.asyncio
    async def test_estimator_name_indexed(self):
        """Test that name field is indexed"""
        name_column = Estimator.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True

    @pytest.mark.asyncio
    async def test_estimator_model_id_indexed(self):
        """Test that model_id field is indexed"""
        model_id_column = Estimator.__table__.c.model_id
        assert model_id_column.index is True


class TestEstimatorConfig:
    """Tests for config field"""

    @pytest.mark.asyncio
    async def test_estimator_with_none_config(self, session, sample_model):
        """Test estimator with None config"""
        estimator = Estimator(name="no_config", config=None, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.config is None

    @pytest.mark.asyncio
    async def test_estimator_with_empty_config(self, session, sample_model):
        """Test estimator with empty config dict"""
        estimator = Estimator(name="empty_config", config={}, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.config == {}

    @pytest.mark.asyncio
    async def test_estimator_with_complex_config(self, session, sample_model):
        """Test estimator with complex nested config"""
        config = {
            "optimizer": {"type": "adam", "lr": 0.001},
            "layers": [64, 128, 256],
            "dropout": 0.5,
            "metadata": {"version": "1.0", "author": "test"},
        }
        estimator = Estimator(name="complex_config", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.config == config
        assert estimator.config["optimizer"]["lr"] == 0.001
        assert estimator.config["layers"] == [64, 128, 256]

    @pytest.mark.asyncio
    async def test_update_config(self, session, sample_estimator):
        """Test updating config field"""
        new_config = {"updated": True, "value": 42}
        sample_estimator.config = new_config
        await session.commit()
        await session.refresh(sample_estimator)

        assert sample_estimator.config == new_config


class TestEstimatorRelationships:
    """Tests for Estimator relationships"""

    @pytest.mark.asyncio
    async def test_estimator_model_relationship_exists(self, sample_estimator):
        """Test that model relationship exists"""
        assert hasattr(sample_estimator, "model")


class TestEstimatorProperties:
    """Tests for Estimator convenience properties"""

    @pytest.mark.asyncio
    async def test_algo_id_property(self, session, sample_estimator, sample_algorithm):
        """Test algo_id property returns correct value"""
        # Need to load the relationship
        await session.refresh(sample_estimator, ["model"])
        assert sample_estimator.algo_id == sample_algorithm.id_

    @pytest.mark.asyncio
    async def test_catalog_tag_id_property(self, session, sample_estimator, sample_catalog_tag):
        """Test catalog_tag_id property returns correct value"""
        # Need to load the relationship
        await session.refresh(sample_estimator, ["model"])
        assert sample_estimator.catalog_tag_id == sample_catalog_tag.id_

    @pytest.mark.asyncio
    async def test_algo_property(self, session, sample_estimator, sample_algorithm):
        """Test algo property returns Algorithm instance"""
        # Need to load the relationships
        await session.refresh(sample_estimator, ["model"])
        await session.refresh(sample_estimator.model, ["algorithm"])
        algo = sample_estimator.algo
        assert algo.id_ == sample_algorithm.id_
        assert algo.name == sample_algorithm.name

    @pytest.mark.asyncio
    async def test_catalog_tag_property(self, session, sample_estimator, sample_catalog_tag):
        """Test catalog_tag property returns CatalogTag instance"""
        # Need to load the relationships
        await session.refresh(sample_estimator, ["model"])
        await session.refresh(sample_estimator.model, ["catalog_tag"])
        tag = sample_estimator.catalog_tag
        assert tag.id_ == sample_catalog_tag.id_
        assert tag.name == sample_catalog_tag.name

    @pytest.mark.asyncio
    async def test_algo_name_property(self, session, sample_estimator, sample_algorithm):
        """Test algo_name property returns algorithm name"""
        # Need to load the relationships
        await session.refresh(sample_estimator, ["model"])
        await session.refresh(sample_estimator.model, ["algorithm"])
        assert sample_estimator.algo_name == sample_algorithm.name

    @pytest.mark.asyncio
    async def test_catalog_tag_name_property(self, session, sample_estimator, sample_catalog_tag):
        """Test catalog_tag_name property returns catalog tag name"""
        # Need to load the relationships
        await session.refresh(sample_estimator, ["model"])
        await session.refresh(sample_estimator.model, ["catalog_tag"])
        assert sample_estimator.catalog_tag_name == sample_catalog_tag.name


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestEstimatorBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session, sample_model):
        """Test inserting multiple estimators at once"""
        estimators = [
            Estimator(name=f"estimator_{i}", config={"param": i}, model_id=sample_model.id_)
            for i in range(10)
        ]

        session.add_all(estimators)
        await session.commit()

        for est in estimators:
            await session.refresh(est)

        assert all(est.id_ is not None for est in estimators)
        assert len(estimators) == 10

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_estimators):
        """Test querying multiple estimators"""
        result = await session.execute(select(Estimator))
        estimators = result.scalars().all()

        assert len(estimators) >= 3
        names = {est.name for est in estimators}
        assert "estimator_v1" in names
        assert "estimator_v2" in names
        assert "estimator_v3" in names

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_estimators):
        """Test updating multiple estimators"""
        for est in multiple_estimators:
            est.config = {"updated": True}

        await session.commit()

        for est in multiple_estimators:
            await session.refresh(est)
            assert est.config == {"updated": True}

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_estimators):
        """Test deleting multiple estimators"""
        estimator_ids = [est.id_ for est in multiple_estimators]

        for est in multiple_estimators:
            await session.delete(est)

        await session.commit()

        for est_id in estimator_ids:
            result = await session.get(Estimator, est_id)
            assert result is None


class TestEstimatorQueries:
    """Tests for various query patterns"""

    @pytest.mark.asyncio
    async def test_query_by_model_id(self, session, sample_model, multiple_estimators):
        """Test querying estimators by model_id"""
        result = await session.execute(select(Estimator).where(Estimator.model_id == sample_model.id_))
        estimators = result.scalars().all()

        assert len(estimators) >= 3
        assert all(e.model_id == sample_model.id_ for e in estimators)

    @pytest.mark.asyncio
    async def test_query_by_name_pattern(self, session, multiple_estimators):
        """Test querying estimators with name pattern matching"""
        result = await session.execute(select(Estimator).where(Estimator.name.like("estimator%")))
        estimators = result.scalars().all()

        assert len(estimators) >= 3
        assert all(e.name.startswith("estimator") for e in estimators)

    @pytest.mark.asyncio
    async def test_query_order_by_name(self, session, multiple_estimators):
        """Test querying estimators ordered by name"""
        result = await session.execute(select(Estimator).order_by(Estimator.name))
        estimators = result.scalars().all()

        names = [e.name for e in estimators]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_query_with_limit(self, session, multiple_estimators):
        """Test querying estimators with limit"""
        result = await session.execute(select(Estimator).limit(2))
        estimators = result.scalars().all()

        assert len(estimators) <= 2

    @pytest.mark.asyncio
    async def test_count_estimators(self, session, multiple_estimators):
        """Test counting total number of estimators"""
        from sqlalchemy import func

        result = await session.execute(select(func.count()).select_from(Estimator))
        count = result.scalar()

        assert count >= 3


class TestEstimatorDataIntegrity:
    """Tests for data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_estimator_persistence(self, session, sample_estimator):
        """Test that estimator data persists correctly"""
        est_id = sample_estimator.id_
        est_name = sample_estimator.name
        est_config = sample_estimator.config

        # Clear session
        await session.commit()
        session.expire_all()

        # Query fresh from database
        result = await session.get(Estimator, est_id)
        assert result is not None
        assert result.name == est_name
        assert result.config == est_config

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, session, sample_estimator):
        """Test updating multiple fields at once"""
        sample_estimator.name = "updated_name"
        sample_estimator.config = {"new": "config"}

        await session.commit()
        await session.refresh(sample_estimator)

        assert sample_estimator.name == "updated_name"
        assert sample_estimator.config == {"new": "config"}

    @pytest.mark.asyncio
    async def test_foreign_key_integrity(self, session, sample_model):
        """Test that foreign key references are maintained"""
        estimator = Estimator(name="integrity_test", config={}, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        # Verify foreign key is correct
        assert estimator.model_id == sample_model.id_

        # Verify we can query the referenced model
        model = await session.get(Model, estimator.model_id)
        assert model is not None


class TestEstimatorPydanticValidation:
    """Tests for Pydantic model integration and validation"""

    @pytest.mark.asyncio
    async def test_to_pydantic_list(self, multiple_estimators):
        """Test converting multiple estimators to Pydantic list"""
        pydantic_list = Estimator.to_pydantic_list(multiple_estimators)

        assert len(pydantic_list) == 3
        assert all(isinstance(obj, EstimatorPydantic) for obj in pydantic_list)
        assert pydantic_list[0].name == "estimator_v1"
        assert pydantic_list[1].name == "estimator_v2"
        assert pydantic_list[2].name == "estimator_v3"

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list(self, multiple_estimators):
        """Test converting multiple estimators to dict list"""
        dict_list = Estimator.to_pydantic_dict_list(multiple_estimators)

        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert all("name" in d for d in dict_list)
        assert all("config" in d for d in dict_list)
        assert all("model_id" in d for d in dict_list)


class TestEstimatorBusinessLogic:
    """Tests for business logic and use cases"""

    @pytest.mark.asyncio
    async def test_multiple_estimators_same_model(self, session, sample_model):
        """Test multiple estimators for the same model with different configs"""
        estimators = []
        for i in range(3):
            est = Estimator(
                name=f"config_variant_{i}", config={"n_neighbors": (i + 1) * 3}, model_id=sample_model.id_
            )
            estimators.append(est)

        session.add_all(estimators)
        await session.commit()

        # Query all estimators for the model
        result = await session.execute(select(Estimator).where(Estimator.model_id == sample_model.id_))
        found_estimators = result.scalars().all()

        assert len(found_estimators) >= 3
        configs = {e.config["n_neighbors"] for e in found_estimators if e.config}
        assert 3 in configs
        assert 6 in configs
        assert 9 in configs


class TestEstimatorConfigSerialization:
    """Tests for config JSON serialization"""

    @pytest.mark.asyncio
    async def test_config_with_various_types(self, session, sample_model):
        """Test config with various Python types"""
        config = {
            "string": "value",
            "int": 42,
            "float": 3.14,
            "bool": True,
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "null": None,
        }
        estimator = Estimator(name="varied_types", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.config == config
        assert estimator.config["string"] == "value"
        assert estimator.config["int"] == 42
        assert estimator.config["float"] == 3.14
        assert estimator.config["bool"] is True
        assert estimator.config["list"] == [1, 2, 3]
        assert estimator.config["dict"]["nested"] == "value"
        assert estimator.config["null"] is None

    @pytest.mark.asyncio
    async def test_config_json_roundtrip(self, session, sample_model):
        """Test that config survives JSON serialization roundtrip"""
        config = {"hyperparams": {"lr": 0.001, "batch_size": 32}, "layers": [128, 256, 512]}
        estimator = Estimator(name="roundtrip_test", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        # Config should be preserved exactly
        assert estimator.config == config


class TestEstimatorCascade:
    """Tests for cascade delete behavior"""

    @pytest.mark.asyncio
    async def test_delete_model_behavior(self, session, sample_algorithm, sample_catalog_tag):
        """Test behavior when model is deleted"""
        # Create a model
        model = Model(
            name="temp_model",
            path="/models/temp.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        )
        session.add(model)
        await session.commit()
        await session.refresh(model)
        model_id = model.id_

        # Create estimator
        estimator = Estimator(name="temp_estimator", config={"test": "value"}, model_id=model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        # Delete model
        await session.delete(model)
        await session.commit()
        session.expire_all()

        # Verify model is deleted
        model_result = await session.get(Model, model_id)
        assert model_result is None


class TestEstimatorNaming:
    """Tests for estimator naming conventions"""

    @pytest.mark.asyncio
    async def test_estimator_with_descriptive_name(self, session, sample_model):
        """Test estimator with descriptive naming convention"""
        estimator = Estimator(
            name="knn_k5_weighted_distance",
            config={"n_neighbors": 5, "weights": "distance"},
            model_id=sample_model.id_,
        )
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.name == "knn_k5_weighted_distance"

    @pytest.mark.asyncio
    async def test_estimator_with_version_in_name(self, session, sample_model):
        """Test estimator with version in name"""
        estimator = Estimator(name="estimator_v2.0", config={}, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.name == "estimator_v2.0"


class TestEstimatorFiltering:
    """Tests for filtering estimators"""

    @pytest.mark.asyncio
    async def test_filter_by_model_id(self, session, sample_model):
        """Test filtering estimators by model_id"""
        estimator = Estimator(name="filter_test", config={}, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()

        result = await session.execute(select(Estimator).where(Estimator.model_id == sample_model.id_))
        estimators = result.scalars().all()

        assert len(estimators) >= 1
        assert any(e.name == "filter_test" for e in estimators)

    @pytest.mark.asyncio
    async def test_exclude_filter(self, session, multiple_estimators):
        """Test filtering to exclude certain estimators"""
        result = await session.execute(select(Estimator).where(~Estimator.name.like("estimator_v1%")))
        estimators = result.scalars().all()

        # Should get estimators that don't start with "estimator_v1"
        names = {e.name for e in estimators}
        assert "estimator_v1" not in names


class TestEstimatorConfigPatterns:
    """Tests for common config patterns"""

    @pytest.mark.asyncio
    async def test_config_with_hyperparameters(self, session, sample_model):
        """Test config with typical ML hyperparameters"""
        config = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100, "optimizer": "adam", "loss": "mse"}
        estimator = Estimator(name="hyperparams_test", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.config["learning_rate"] == 0.001
        assert estimator.config["batch_size"] == 32
        assert estimator.config["epochs"] == 100

    @pytest.mark.asyncio
    async def test_config_with_model_architecture(self, session, sample_model):
        """Test config with model architecture specification"""
        config = {
            "layers": [
                {"type": "dense", "units": 128, "activation": "relu"},
                {"type": "dropout", "rate": 0.5},
                {"type": "dense", "units": 64, "activation": "relu"},
                {"type": "dense", "units": 1, "activation": "linear"},
            ]
        }
        estimator = Estimator(name="architecture_test", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert len(estimator.config["layers"]) == 4
        assert estimator.config["layers"][0]["units"] == 128

    @pytest.mark.asyncio
    async def test_config_with_preprocessing_steps(self, session, sample_model):
        """Test config with preprocessing pipeline"""
        config = {
            "preprocessing": {
                "normalize": True,
                "scaler": "standard",
                "feature_selection": {"method": "variance", "threshold": 0.01},
            }
        }
        estimator = Estimator(name="preprocessing_test", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.config["preprocessing"]["normalize"] is True
        assert estimator.config["preprocessing"]["scaler"] == "standard"


class TestEstimatorComplexQueries:
    """Tests for complex query scenarios"""

    @pytest.mark.asyncio
    async def test_join_with_model(self, session, sample_estimator, sample_model):
        """Test querying estimators with join to model"""
        from sqlalchemy.orm import selectinload

        result = await session.execute(
            select(Estimator)
            .options(selectinload(Estimator.model))
            .where(Estimator.id_ == sample_estimator.id_)
        )
        estimator = result.scalar_one()

        assert estimator.model is not None
        assert estimator.model.id_ == sample_model.id_

    @pytest.mark.asyncio
    async def test_filter_by_related_model_name(self, session, sample_estimator, sample_model):
        """Test filtering estimators by related model name"""
        result = await session.execute(
            select(Estimator).join(Estimator.model).where(Model.name == sample_model.name)
        )
        estimators = result.scalars().all()

        assert len(estimators) >= 1
        assert sample_estimator.id_ in [e.id_ for e in estimators]


class TestEstimatorSorting:
    """Tests for sorting estimators"""

    @pytest.mark.asyncio
    async def test_sort_ascending(self, session, multiple_estimators):
        """Test sorting estimators in ascending order"""
        result = await session.execute(select(Estimator).order_by(Estimator.name.asc()))
        estimators = result.scalars().all()

        names = [e.name for e in estimators]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_sort_descending(self, session, multiple_estimators):
        """Test sorting estimators in descending order"""
        result = await session.execute(select(Estimator).order_by(Estimator.name.desc()))
        estimators = result.scalars().all()

        names = [e.name for e in estimators]
        assert names == sorted(names, reverse=True)

    @pytest.mark.asyncio
    async def test_sort_by_id(self, session, multiple_estimators):
        """Test sorting estimators by id"""
        result = await session.execute(select(Estimator).order_by(Estimator.id_))
        estimators = result.scalars().all()

        ids = [e.id_ for e in estimators]
        assert ids == sorted(ids)


class TestEstimatorPagination:
    """Tests for pagination"""

    @pytest.mark.asyncio
    async def test_pagination_first_page(self, session, sample_model):
        """Test getting first page of results"""
        # Create many estimators
        estimators = [
            Estimator(name=f"page_test_{i:02d}", config={}, model_id=sample_model.id_) for i in range(20)
        ]
        session.add_all(estimators)
        await session.commit()

        result = await session.execute(select(Estimator).order_by(Estimator.name).limit(10).offset(0))
        page1 = result.scalars().all()

        assert len(page1) == 10

    @pytest.mark.asyncio
    async def test_pagination_second_page(self, session, sample_model):
        """Test getting second page of results"""
        # Create many estimators
        estimators = [
            Estimator(name=f"page_test2_{i:02d}", config={}, model_id=sample_model.id_) for i in range(20)
        ]
        session.add_all(estimators)
        await session.commit()

        result = await session.execute(select(Estimator).order_by(Estimator.name).limit(10).offset(10))
        page2 = result.scalars().all()

        assert len(page2) == 10

    @pytest.mark.asyncio
    async def test_pagination_boundary(self, session, sample_model):
        """Test pagination at boundary"""
        # Create exact number of estimators
        estimators = [
            Estimator(name=f"boundary_{i}", config={}, model_id=sample_model.id_) for i in range(15)
        ]
        session.add_all(estimators)
        await session.commit()

        result = await session.execute(select(Estimator).order_by(Estimator.name).limit(10).offset(10))
        page = result.scalars().all()

        assert len(page) == 5  # Only 5 remaining


class TestEstimatorReprStr:
    """Tests for __repr__ and __str__ edge cases"""

    @pytest.mark.asyncio
    async def test_repr_with_special_characters(self, session, sample_model):
        """Test __repr__ with special characters in name"""
        estimator = Estimator(name="test'estimator\"with\\special", config={}, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        repr_str = repr(estimator)
        assert "Estimator" in repr_str
        assert str(estimator.id_) in repr_str

    @pytest.mark.asyncio
    async def test_str_returns_name_only(self, sample_estimator):
        """Test that __str__ returns only the name"""
        str_repr = str(sample_estimator)
        assert str_repr == sample_estimator.name
        assert "Estimator" not in str_repr
        assert str(sample_estimator.id_) not in str_repr


class TestEstimatorConfigEdgeCases:
    """Tests for config field edge cases"""

    @pytest.mark.asyncio
    async def test_config_with_unicode(self, session, sample_model):
        """Test config with unicode characters"""
        config = {"description": "测试 αβγ δεζ", "emoji": "🚀🎉", "special": "café"}
        estimator = Estimator(name="unicode_test", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert estimator.config["description"] == "测试 αβγ δεζ"
        assert estimator.config["emoji"] == "🚀🎉"

    @pytest.mark.asyncio
    async def test_config_with_large_data(self, session, sample_model):
        """Test config with large amount of data"""
        config = {"data": [i for i in range(1000)], "matrix": [[j for j in range(10)] for _ in range(10)]}
        estimator = Estimator(name="large_config", config=config, model_id=sample_model.id_)
        session.add(estimator)
        await session.commit()
        await session.refresh(estimator)

        assert len(estimator.config["data"]) == 1000
        assert len(estimator.config["matrix"]) == 10
