"""Unit tests for the Algorithm Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.algorithm import Algorithm, AlgorithmBase, AlgorithmCreate


class TestAlgorithmBase:
    """Tests for AlgorithmBase model"""

    def test_valid_algorithm_base(self):
        """Test creating a valid AlgorithmBase"""
        algo = AlgorithmBase(
            name="KNearNeighEstimator",
            class_name="rail.estimation.algos.k_nearneigh.KNearNeighEstimator",
        )
        assert algo.name == "KNearNeighEstimator"
        assert algo.class_name == "rail.estimation.algos.k_nearneigh.KNearNeighEstimator"

    def test_algorithm_base_simple_class_name(self):
        """Test creating AlgorithmBase with simple class name"""
        algo = AlgorithmBase(
            name="SimpleEstimator",
            class_name="SimpleEstimator",
        )
        assert algo.name == "SimpleEstimator"
        assert algo.class_name == "SimpleEstimator"

    def test_algorithm_base_missing_name(self):
        """Test that name is required"""
        with pytest.raises(ValidationError) as exc_info:
            AlgorithmBase(class_name="SomeClass")
        assert "name" in str(exc_info.value)

    def test_algorithm_base_missing_class_name(self):
        """Test that class_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            AlgorithmBase(name="SomeAlgo")
        assert "class_name" in str(exc_info.value)

    def test_algorithm_base_invalid_class_name(self):
        """Test that class_name must be a valid Python module path"""
        with pytest.raises(ValidationError) as exc_info:
            AlgorithmBase(
                name="BadAlgo",
                class_name="invalid-class-name",
            )
        assert "valid Python module path" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            AlgorithmBase(
                name="BadAlgo",
                class_name="invalid.class.name!",
            )
        assert "valid Python module path" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            AlgorithmBase(
                name="BadAlgo",
                class_name="123invalid",
            )
        assert "valid Python module path" in str(exc_info.value)

    def test_algorithm_base_valid_dotted_paths(self):
        """Test various valid dotted Python paths"""
        valid_paths = [
            "module.Class",
            "package.module.Class",
            "a.b.c.d.e.Class",
            "_private.Module",
            "module._PrivateClass",
        ]
        for path in valid_paths:
            algo = AlgorithmBase(name="Test", class_name=path)
            assert algo.class_name == path


class TestAlgorithmCreate:
    """Tests for AlgorithmCreate model"""

    def test_valid_algorithm_create(self):
        """Test creating a valid AlgorithmCreate"""
        algo = AlgorithmCreate(
            name="SOMEstimator",
            class_name="rail.estimation.algos.som.SOMEstimator",
        )
        assert algo.name == "SOMEstimator"
        assert algo.class_name == "rail.estimation.algos.som.SOMEstimator"

    def test_algorithm_create_inherits_validation(self):
        """Test that AlgorithmCreate inherits validation from AlgorithmBase"""
        with pytest.raises(ValidationError) as exc_info:
            AlgorithmCreate(
                name="BadAlgo",
                class_name="invalid-name",
            )
        assert "valid Python module path" in str(exc_info.value)


class TestAlgorithm:
    """Tests for Algorithm model"""

    def test_valid_algorithm(self):
        """Test creating a valid Algorithm with all fields"""
        algo = Algorithm(
            id=1,
            name="BPZEstimator",
            class_name="rail.estimation.algos.bpz.BPZEstimator",
        )
        assert algo.id == 1
        assert algo.name == "BPZEstimator"
        assert algo.class_name == "rail.estimation.algos.bpz.BPZEstimator"

    def test_algorithm_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Algorithm(
                id=0,
                name="test",
                class_name="test.Class",
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Algorithm(
                id=-1,
                name="test",
                class_name="test.Class",
            )
        assert "greater than 0" in str(exc_info.value)

    def test_algorithm_missing_id(self):
        """Test that id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Algorithm(
                name="test",
                class_name="test.Class",
            )
        assert "id" in str(exc_info.value)

    def test_algorithm_invalid_class_name(self):
        """Test that Algorithm inherits class_name validation"""
        with pytest.raises(ValidationError) as exc_info:
            Algorithm(
                id=1,
                name="test",
                class_name="bad-class",
            )
        assert "valid Python module path" in str(exc_info.value)

    def test_algorithm_from_attributes(self):
        """Test that from_attributes config works"""

        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 5
            name = "FlexZBoostEstimator"
            class_name = "rail.estimation.algos.flexzboost.FlexZBoostEstimator"

        orm_obj = MockORMObject()
        algo = Algorithm.model_validate(orm_obj)
        assert algo.id == 5
        assert algo.name == "FlexZBoostEstimator"
        assert algo.class_name == "rail.estimation.algos.flexzboost.FlexZBoostEstimator"

    def test_algorithm_col_names_for_table(self):
        """Test that col_names_for_table class variable is set correctly"""
        expected_cols = ["id", "name", "class_name"]
        assert Algorithm.col_names_for_table == expected_cols

    def test_algorithm_field_descriptions(self):
        """Test that field descriptions are set"""
        schema = Algorithm.model_json_schema()
        assert "Unique name for this algorithm" in schema["properties"]["name"]["description"]
        assert "Fully qualified Python class name" in schema["properties"]["class_name"]["description"]

    def test_algorithm_validator_class_method(self):
        """Test that the validator is properly defined as a classmethod"""
        # This ensures the validator can access cls parameter
        algo = Algorithm(
            id=1,
            name="test",
            class_name="valid.Class",
        )
        assert algo.class_name == "valid.Class"

    def test_algorithm_edge_cases(self):
        """Test edge cases for class_name validation"""
        # Single identifier (no dots) should be valid
        algo = Algorithm(
            id=1,
            name="simple",
            class_name="SimpleClass",
        )
        assert algo.class_name == "SimpleClass"

        # Empty string should fail
        with pytest.raises(ValidationError):
            Algorithm(
                id=1,
                name="test",
                class_name="",
            )

        # Dots only should fail
        with pytest.raises(ValidationError):
            Algorithm(
                id=1,
                name="test",
                class_name="...",
            )

        # Leading/trailing dots should fail
        with pytest.raises(ValidationError):
            Algorithm(
                id=1,
                name="test",
                class_name=".module.Class",
            )

        with pytest.raises(ValidationError):
            Algorithm(
                id=1,
                name="test",
                class_name="module.Class.",
            )
