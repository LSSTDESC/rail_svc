"""Shared parametrized tests for Pydantic model validation.

Tests common patterns (creation, from_attributes, col_names, id validation)
across all entity models.
"""

import pytest
from pydantic import ValidationError

from rail_svc import models

MODEL_CONFIGS = [
    {
        "model_class": models.Algorithm,
        "create_class": models.AlgorithmCreate,
        "valid_data": {"id_": 1, "name": "test_algo", "class_name": "rail.SOMEstimator"},
        "create_data": {"name": "test_algo", "class_name": "rail.SOMEstimator"},
        "name": "algorithm",
    },
    {
        "model_class": models.Band,
        "create_class": models.BandCreate,
        "valid_data": {
            "id_": 1,
            "name": "g_band",
            "band_wavelengths": [400.0, 500.0],
            "band_transmission": [0.1, 0.9],
        },
        "create_data": {
            "name": "g_band",
            "band_wavelengths": [400.0, 500.0],
            "band_transmission": [0.1, 0.9],
        },
        "name": "band",
    },
    {
        "model_class": models.CatalogTag,
        "create_class": models.CatalogTagCreate,
        "valid_data": {"id_": 1, "name": "lsst_dp02"},
        "create_data": {"name": "lsst_dp02"},
        "name": "catalog_tag",
    },
    {
        "model_class": models.Dataset,
        "create_class": models.DatasetCreate,
        "valid_data": {
            "id_": 1,
            "name": "ds",
            "path": "/d.hdf5",
            "n_objects": 100,
            "is_collection": False,
            "catalog_tag_id": 1,
        },
        "create_data": {
            "name": "ds",
            "path": "/d.hdf5",
            "n_objects": 100,
            "is_collection": False,
            "catalog_tag_name": "lsst",
        },
        "name": "dataset",
    },
    {
        "model_class": models.Estimates,
        "create_class": models.EstimatesCreate,
        "valid_data": {
            "id_": 1,
            "name": "est",
            "path": "/e.hdf5",
            "n_objects": 100,
            "dataset_id": 1,
            "estimator_id": 1,
        },
        "create_data": {
            "name": "est",
            "path": "/e.hdf5",
            "n_objects": 100,
            "dataset_name": "ds",
            "estimator_name": "bpz",
        },
        "name": "estimates",
    },
    {
        "model_class": models.Estimator,
        "create_class": models.EstimatorCreate,
        "valid_data": {"id_": 1, "name": "bpz", "config": {}, "model_id": 1},
        "create_data": {"name": "bpz", "config": {}, "model_name": "m1"},
        "name": "estimator",
    },
    {
        "model_class": models.Model,
        "create_class": models.ModelCreate,
        "valid_data": {"id_": 1, "name": "rf", "path": "/m.pkl", "algo_id": 1, "catalog_tag_id": 1},
        "create_data": {"name": "rf", "path": "/m.pkl", "algo_name": "RF", "catalog_tag_name": "lsst"},
        "name": "model",
    },
]


@pytest.fixture(params=MODEL_CONFIGS, ids=[c["name"] for c in MODEL_CONFIGS])
def model_config(request):
    return request.param


class TestModelCreation:
    """Test that models can be created with valid data."""

    def test_create_with_valid_data(self, model_config):
        obj = model_config["model_class"](**model_config["valid_data"])
        assert obj.id_ == model_config["valid_data"]["id_"]
        assert obj.name == model_config["valid_data"]["name"]

    def test_create_model_with_valid_data(self, model_config):
        obj = model_config["create_class"](**model_config["create_data"])
        assert obj.name == model_config["create_data"]["name"]


class TestModelAttributes:
    """Test common model attributes."""

    def test_has_col_names_for_table(self, model_config):
        assert hasattr(model_config["model_class"], "col_names_for_table")
        assert isinstance(model_config["model_class"].col_names_for_table, list)
        assert len(model_config["model_class"].col_names_for_table) > 0

    def test_from_attributes_enabled(self, model_config):
        config = model_config["model_class"].model_config
        assert config.get("from_attributes") is True


class TestModelValidation:
    """Test that models reject invalid data."""

    def test_missing_name_raises(self, model_config):
        data = {k: v for k, v in model_config["valid_data"].items() if k != "name"}
        with pytest.raises(ValidationError):
            model_config["model_class"](**data)
