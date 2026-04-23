"""Unit tests for the CatalogTag Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.catalog_tag import CatalogTag, CatalogTagBase, CatalogTagCreate


class TestCatalogTagBase:
    """Tests for CatalogTagBase model"""

    def test_valid_catalog_tag_base(self):
        """Test creating a valid CatalogTagBase"""
        tag = CatalogTagBase(
            name="lsst_dp02",
            class_name="rail.utils.catalog_utils.LsstDp02Catalog",
        )
        assert tag.name == "lsst_dp02"
        assert tag.class_name == "rail.utils.catalog_utils.LsstDp02Catalog"

    def test_catalog_tag_base_simple_class_name(self):
        """Test creating CatalogTagBase with simple class name"""
        tag = CatalogTagBase(
            name="simple_catalog",
            class_name="SimpleCatalog",
        )
        assert tag.name == "simple_catalog"
        assert tag.class_name == "SimpleCatalog"

    def test_catalog_tag_base_missing_name(self):
        """Test that name is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogTagBase(class_name="SomeClass")
        assert "name" in str(exc_info.value)

    def test_catalog_tag_base_missing_class_name(self):
        """Test that class_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogTagBase(name="some_catalog")
        assert "class_name" in str(exc_info.value)

    def test_catalog_tag_base_invalid_class_name(self):
        """Test that class_name must be a valid Python module path"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogTagBase(
                name="bad_catalog",
                class_name="invalid-class-name",
            )
        assert "valid Python module path" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogTagBase(
                name="bad_catalog",
                class_name="invalid.class.name!",
            )
        assert "valid Python module path" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogTagBase(
                name="bad_catalog",
                class_name="123invalid",
            )
        assert "valid Python module path" in str(exc_info.value)

    def test_catalog_tag_base_valid_dotted_paths(self):
        """Test various valid dotted Python paths"""
        valid_paths = [
            "module.Class",
            "package.module.Class",
            "a.b.c.d.e.Class",
            "_private.Module",
            "module._PrivateClass",
            "rail.utils.catalog_utils.DESCatalog",
        ]
        for path in valid_paths:
            tag = CatalogTagBase(name="test_catalog", class_name=path)
            assert tag.class_name == path


class TestCatalogTagCreate:
    """Tests for CatalogTagCreate model"""

    def test_valid_catalog_tag_create(self):
        """Test creating a valid CatalogTagCreate"""
        tag = CatalogTagCreate(
            name="hsc_pdr3",
            class_name="rail.utils.catalog_utils.HSCCatalog",
        )
        assert tag.name == "hsc_pdr3"
        assert tag.class_name == "rail.utils.catalog_utils.HSCCatalog"

    def test_catalog_tag_create_inherits_validation(self):
        """Test that CatalogTagCreate inherits validation from CatalogTagBase"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogTagCreate(
                name="bad_catalog",
                class_name="invalid-name",
            )
        assert "valid Python module path" in str(exc_info.value)


class TestCatalogTag:
    """Tests for CatalogTag model"""

    def test_valid_catalog_tag(self):
        """Test creating a valid CatalogTag with all fields"""
        tag = CatalogTag(
            id=1,
            name="lsst_dp02",
            class_name="rail.utils.catalog_utils.LsstDp02Catalog",
        )
        assert tag.id == 1
        assert tag.name == "lsst_dp02"
        assert tag.class_name == "rail.utils.catalog_utils.LsstDp02Catalog"

    def test_catalog_tag_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogTag(
                id=0,
                name="test",
                class_name="test.Class",
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            CatalogTag(
                id=-1,
                name="test",
                class_name="test.Class",
            )
        assert "greater than 0" in str(exc_info.value)

    def test_catalog_tag_missing_id(self):
        """Test that id is required"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogTag(
                name="test",
                class_name="test.Class",
            )
        assert "id" in str(exc_info.value)

    def test_catalog_tag_invalid_class_name(self):
        """Test that CatalogTag inherits class_name validation"""
        with pytest.raises(ValidationError) as exc_info:
            CatalogTag(
                id=1,
                name="test",
                class_name="bad-class",
            )
        assert "valid Python module path" in str(exc_info.value)

    def test_catalog_tag_from_attributes(self):
        """Test that from_attributes config works"""

        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 5
            name = "des_y3"
            class_name = "rail.utils.catalog_utils.DESCatalog"

        orm_obj = MockORMObject()
        tag = CatalogTag.model_validate(orm_obj)
        assert tag.id == 5
        assert tag.name == "des_y3"
        assert tag.class_name == "rail.utils.catalog_utils.DESCatalog"

    def test_catalog_tag_col_names_for_table(self):
        """Test that col_names_for_table class variable is set correctly"""
        expected_cols = ["id", "name", "class_name"]
        assert CatalogTag.col_names_for_table == expected_cols

    def test_catalog_tag_field_descriptions(self):
        """Test that field descriptions are set"""
        schema = CatalogTag.model_json_schema()
        assert "Unique name for this catalog tag" in schema["properties"]["name"]["description"]
        assert "Fully qualified Python class name" in schema["properties"]["class_name"]["description"]

    def test_catalog_tag_validator_class_method(self):
        """Test that the validator is properly defined as a classmethod"""
        # This ensures the validator can access cls parameter
        tag = CatalogTag(
            id=1,
            name="test",
            class_name="valid.Class",
        )
        assert tag.class_name == "valid.Class"

    def test_catalog_tag_edge_cases(self):
        """Test edge cases for class_name validation"""
        # Single identifier (no dots) should be valid
        tag = CatalogTag(
            id=1,
            name="simple",
            class_name="SimpleClass",
        )
        assert tag.class_name == "SimpleClass"

        # Empty string should fail
        with pytest.raises(ValidationError):
            CatalogTag(
                id=1,
                name="test",
                class_name="",
            )

        # Dots only should fail
        with pytest.raises(ValidationError):
            CatalogTag(
                id=1,
                name="test",
                class_name="...",
            )

        # Leading/trailing dots should fail
        with pytest.raises(ValidationError):
            CatalogTag(
                id=1,
                name="test",
                class_name=".module.Class",
            )

        with pytest.raises(ValidationError):
            CatalogTag(
                id=1,
                name="test",
                class_name="module.Class.",
            )

    def test_catalog_tag_realistic_examples(self):
        """Test with realistic catalog tag examples"""
        examples = [
            ("lsst_dp02", "rail.utils.catalog_utils.LsstDp02Catalog"),
            ("hsc_pdr3", "rail.utils.catalog_utils.HSCCatalog"),
            ("des_y3", "rail.utils.catalog_utils.DESCatalog"),
            ("cosmos", "rail.utils.catalog_utils.COSMOSCatalog"),
        ]

        for idx, (name, class_name) in enumerate(examples, start=1):
            tag = CatalogTag(
                id=idx,
                name=name,
                class_name=class_name,
            )
            assert tag.name == name
            assert tag.class_name == class_name
