"""Unit tests for Algorithm Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.algorithm import Algorithm, AlgorithmBase, AlgorithmCreate


class TestAlgorithmBase:
    """Tests for AlgorithmBase model"""

    def test_valid_algorithm_base(self):
        """Test creating valid AlgorithmBase instance"""
        algo = AlgorithmBase(name="test_algorithm", class_name="rail.estimation.SOMEstimator")
        assert algo.name == "test_algorithm"
        assert algo.class_name == "rail.estimation.SOMEstimator"

    def test_invalid_class_name_with_spaces(self):
        """Test that class_name with spaces raises ValidationError"""
        with pytest.raises(ValidationError, match="class_name must be a valid Python module path"):
            AlgorithmBase(name="test", class_name="invalid class name")

    def test_invalid_class_name_with_special_chars(self):
        """Test that class_name with special characters raises ValidationError"""
        with pytest.raises(ValidationError, match="class_name must be a valid Python module path"):
            AlgorithmBase(name="test", class_name="rail.estimation.SOM-Estimator")

    def test_valid_single_part_class_name(self):
        """Test that single-part class name is valid"""
        algo = AlgorithmBase(name="test", class_name="Estimator")
        assert algo.class_name == "Estimator"

    def test_missing_required_fields(self):
        """Test that missing required fields raises ValidationError"""
        with pytest.raises(ValidationError):
            AlgorithmBase()


class TestAlgorithmCreate:
    """Tests for AlgorithmCreate model"""

    def test_algorithm_create_inherits_validation(self):
        """Test that AlgorithmCreate inherits validation from AlgorithmBase"""
        algo = AlgorithmCreate(name="creator", class_name="module.ClassName")
        assert algo.name == "creator"

        with pytest.raises(ValidationError):
            AlgorithmCreate(name="test", class_name="invalid-name")


class TestAlgorithm:
    """Tests for Algorithm model"""

    def test_valid_algorithm(self):
        """Test creating valid Algorithm instance"""
        algo = Algorithm(id_=1, name="test_algorithm", class_name="rail.estimation.SOMEstimator")
        assert algo.id_ == 1
        assert algo.name == "test_algorithm"

    def test_id_must_be_positive(self):
        """Test that id_ must be greater than 0"""
        with pytest.raises(ValidationError):
            Algorithm(id_=0, name="test", class_name="module.Class")

        with pytest.raises(ValidationError):
            Algorithm(id_=-1, name="test", class_name="module.Class")

    def test_from_attributes_config(self):
        """Test that from_attributes config works with ORM objects"""

        class MockORMObject:
            id_ = 5
            name = "orm_algo"
            class_name = "some.module.Class"

        algo = Algorithm.model_validate(MockORMObject())
        assert algo.id_ == 5
        assert algo.name == "orm_algo"

    def test_col_names_class_variable(self):
        """Test that col_names_for_table ClassVar is accessible"""
        assert Algorithm.col_names_for_table == ["id_", "name", "class_name"]
