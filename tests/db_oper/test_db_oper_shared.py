"""Shared parametrized tests for db_oper singletons and TableOperations.

Tests the common interface (instantiation, singleton identity, CRUD delegation)
across all entity operation modules.
"""

import pytest

from rail_svc import db
from rail_svc.db_oper import (
    algorithm,
    band,
    catalog_band_assoc,
    catalog_tag,
    dataset_assoc,
    estimator,
)
from rail_svc.db_oper.base import TableOperations

OPER_CONFIGS = [
    {
        "module_path": "rail_svc.db_oper.algorithm",
        "singleton": algorithm,
        "db_class": db.Algorithm,
        "name": "algorithm",
    },
    {"module_path": "rail_svc.db_oper.band", "singleton": band, "db_class": db.Band, "name": "band"},
    {
        "module_path": "rail_svc.db_oper.catalog_tag",
        "singleton": catalog_tag,
        "db_class": db.CatalogTag,
        "name": "catalog_tag",
    },
    {
        "module_path": "rail_svc.db_oper.catalog_band_assoc",
        "singleton": catalog_band_assoc,
        "db_class": db.CatalogBandAssoc,
        "name": "catalog_band_assoc",
    },
    {
        "module_path": "rail_svc.db_oper.dataset_assoc",
        "singleton": dataset_assoc,
        "db_class": db.DatasetAssoc,
        "name": "dataset_assoc",
    },
    {
        "module_path": "rail_svc.db_oper.estimator",
        "singleton": estimator,
        "db_class": db.Estimator,
        "name": "estimator",
    },
]


@pytest.fixture(params=OPER_CONFIGS, ids=[c["name"] for c in OPER_CONFIGS])
def oper_config(request):
    return request.param


class TestOperationsSingleton:
    """Test that module-level singletons are properly configured."""

    def test_singleton_exists(self, oper_config):
        assert oper_config["singleton"] is not None

    def test_singleton_is_table_operations(self, oper_config):
        assert isinstance(oper_config["singleton"], TableOperations)

    def test_singleton_has_correct_db_class(self, oper_config):
        assert oper_config["singleton"].ctx.db_class is oper_config["db_class"]

    def test_singleton_identity(self, oper_config):
        """Test that re-importing gives the same object."""
        import importlib

        module = importlib.import_module(oper_config["module_path"])
        reimported = getattr(module, oper_config["name"])
        assert reimported is oper_config["singleton"]

    def test_has_crud_methods(self, oper_config):
        ops = oper_config["singleton"]
        assert hasattr(ops, "get_row")
        assert hasattr(ops, "create_row")
        assert hasattr(ops, "delete_row")
        assert hasattr(ops, "find_by")
