"""Unit tests for the Estimator Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.estimator import Estimator, EstimatorBase, EstimatorCreate


class TestEstimatorBase:
    """Tests for EstimatorBase model"""

    def test_valid_estimator_base(self):
        """Test creating a valid EstimatorBase"""
        estimator = EstimatorBase(
            name="test_estimator",
            config={"n_neighbors": 5, "weights": "uniform"},
        )
        assert estimator.name == "test_estimator"
        assert estimator.config == {"n_neighbors": 5, "weights": "uniform"}

    def test_estimator_base_with_none_config(self):
        """Test creating EstimatorBase with None config"""
        estimator = EstimatorBase(
            name="simple_estimator",
            config=None,
        )
        assert estimator.name == "simple_estimator"
        assert estimator.config is None

    def test_estimator_base_without_config(self):
        """Test creating EstimatorBase without specifying config (should default to None)"""
        estimator = EstimatorBase(name="default_estimator")
        assert estimator.name == "default_estimator"
        assert estimator.config is None

    def test_estimator_base_with_empty_config(self):
        """Test creating EstimatorBase with empty config dict"""
        estimator = EstimatorBase(
            name="empty_config",
            config={},
        )
        assert estimator.name == "empty_config"
        assert estimator.config == {}

    def test_estimator_base_with_nested_config(self):
        """Test creating EstimatorBase with nested config"""
        config = {
            "algorithm_params": {
                "n_estimators": 100,
                "max_depth": 10,
            },
            "preprocessing": {
                "normalize": True,
                "scale": "standard",
            },
        }
        estimator = EstimatorBase(name="nested_config", config=config)
        assert estimator.config == config

    def test_estimator_base_missing_name(self):
        """Test that name is required"""
        with pytest.raises(ValidationError) as exc_info:
            EstimatorBase(config={"param": "value"})
        assert "name" in str(exc_info.value)

    def test_estimator_base_config_must_be_dict_or_none(self):
        """Test that config must be a dict or None"""
        with pytest.raises(ValidationError) as exc_info:
            EstimatorBase(name="test", config="not a dict")
        
        with pytest.raises(ValidationError) as exc_info:
            EstimatorBase(name="test", config=123)
        
        with pytest.raises(ValidationError) as exc_info:
            EstimatorBase(name="test", config=["list", "not", "dict"])


class TestEstimatorCreate:
    """Tests for EstimatorCreate model"""

    def test_valid_estimator_create(self):
        """Test creating a valid EstimatorCreate"""
        estimator = EstimatorCreate(
            name="new_estimator",
            config={"learning_rate": 0.01},
            model_name="trained_model_v1",
        )
        assert estimator.name == "new_estimator"
        assert estimator.config == {"learning_rate": 0.01}
        assert estimator.model_name == "trained_model_v1"

    def test_estimator_create_with_none_config(self):
        """Test creating EstimatorCreate with None config"""
        estimator = EstimatorCreate(
            name="simple",
            model_name="model_v1",
        )
        assert estimator.config is None
        assert estimator.model_name == "model_v1"

    def test_estimator_create_missing_model_name(self):
        """Test that model_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            EstimatorCreate(
                name="test",
                config={"param": "value"},
            )
        assert "model_name" in str(exc_info.value)

    def test_estimator_create_inherits_validation(self):
        """Test that EstimatorCreate inherits validation from EstimatorBase"""
        with pytest.raises(ValidationError) as exc_info:
            EstimatorCreate(
                name="test",
                config="invalid",
                model_name="model",
            )


class TestEstimator:
    """Tests for Estimator model"""

    def test_valid_estimator(self):
        """Test creating a valid Estimator with all fields"""
        estimator = Estimator(
            id=1,
            name="production_estimator",
            config={"n_neighbors": 10, "metric": "euclidean"},
            algo_id=2,
            catalog_tag_id=3,
            model_id=4,
        )
        assert estimator.id == 1
        assert estimator.name == "production_estimator"
        assert estimator.config == {"n_neighbors": 10, "metric": "euclidean"}
        assert estimator.algo_id == 2
        assert estimator.catalog_tag_id == 3
        assert estimator.model_id == 4

    def test_estimator_with_none_config(self):
        """Test creating Estimator with None config"""
        estimator = Estimator(
            id=1,
            name="default_params",
            config=None,
            algo_id=2,
            catalog_tag_id=3,
            model_id=4,
        )
        assert estimator.config is None

    def test_estimator_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=0,
                name="test",
                algo_id=1,
                catalog_tag_id=1,
                model_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=-1,
                name="test",
                algo_id=1,
                catalog_tag_id=1,
                model_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_estimator_algo_id_must_be_positive(self):
        """Test that algo_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=0,
                catalog_tag_id=1,
                model_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=-5,
                catalog_tag_id=1,
                model_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_estimator_catalog_tag_id_must_be_positive(self):
        """Test that catalog_tag_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=1,
                catalog_tag_id=0,
                model_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=1,
                catalog_tag_id=-3,
                model_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_estimator_model_id_must_be_positive(self):
        """Test that model_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=1,
                catalog_tag_id=1,
                model_id=0,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=1,
                catalog_tag_id=1,
                model_id=-7,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_estimator_missing_id(self):
        """Test that id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                name="test",
                algo_id=1,
                catalog_tag_id=1,
                model_id=1,
            )
        assert "id" in str(exc_info.value)

    def test_estimator_missing_algo_id(self):
        """Test that algo_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                catalog_tag_id=1,
                model_id=1,
            )
        assert "algo_id" in str(exc_info.value)

    def test_estimator_missing_catalog_tag_id(self):
        """Test that catalog_tag_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=1,
                model_id=1,
            )
        assert "catalog_tag_id" in str(exc_info.value)

    def test_estimator_missing_model_id(self):
        """Test that model_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Estimator(
                id=1,
                name="test",
                algo_id=1,
                catalog_tag_id=1,
            )
        assert "model_id" in str(exc_info.value)

    def test_estimator_from_attributes(self):
        """Test that from_attributes config works"""
        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 5
            name = "orm_estimator"
            config = {"batch_size": 32}
            algo_id = 10
