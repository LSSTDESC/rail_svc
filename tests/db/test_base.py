"""Unit tests for Base database model class"""

import pytest
from pydantic import BaseModel
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from rail_svc.db.algorithm import Algorithm
from rail_svc.db.base import Base, ensure_base_inheritance

# ============================================================================
# Base Class Tests
# ============================================================================


class TestBase:
    """Tests for Base class"""

    def test_base_has_metadata(self):
        """Test that Base has metadata configured"""
        assert Base.metadata is not None
        assert hasattr(Base.metadata, "naming_convention")

    def test_base_default_pagination_limit(self):
        """Test that Base has default pagination limit"""
        assert Base.default_pagination_limit == 100

    def test_class_string(self):
        """Test Base.class_string() returns class name"""
        assert Base.class_string() == "Base"

    def test_get_pagination_limit(self):
        """Test Base.get_pagination_limit() returns default"""
        assert Base.get_pagination_limit() == 100

    def test_pydantic_create_class_not_implemented(self):
        """Test that pydantic_create_class raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match="must implement pydantic_create_class"):
            Base.pydantic_create_class()

    def test_pydantic_model_class_not_implemented(self):
        """Test that pydantic_model_class raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match="must implement pydantic_model_class"):
            Base.pydantic_model_class()

    def test_get_hooks_default(self):
        """Test get_hooks returns False for all hooks by default"""
        hooks = Base.get_hooks()
        assert hooks == {
            "pre_create": False,
            "after_create": False,
            "pre_update": False,
            "after_update": False,
            "pre_delete": False,
            "after_delete": False,
        }


class TestBaseHooks:
    """Tests for Base lifecycle hooks"""

    @pytest.mark.asyncio
    async def test_pre_create_hook_default(self, session):
        """Test default pre_create_hook returns data unchanged"""
        data = {"name": "test", "value": 42}
        result = await Base.pre_create_hook(session, data)
        assert result == data

    @pytest.mark.asyncio
    async def test_after_create_hook_default(self, session, sample_algorithm):
        """Test default after_create_hook does nothing"""
        result = await Base.after_create_hook(session, sample_algorithm)
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_update_hook_default(self, session, sample_algorithm):
        """Test default pre_update_hook returns data unchanged"""
        data = {"name": "updated"}
        result = await Base.pre_update_hook(session, sample_algorithm, data)
        assert result == data

    @pytest.mark.asyncio
    async def test_after_update_hook_default(self, session, sample_algorithm):
        """Test default after_update_hook does nothing"""
        updated_fields = {"name"}
        result = await Base.after_update_hook(session, sample_algorithm, updated_fields)
        assert result is None

    @pytest.mark.asyncio
    async def test_pre_delete_hook_default(self, session, sample_algorithm):
        """Test default pre_delete_hook does nothing"""
        result = await Base.pre_delete_hook(session, sample_algorithm)
        assert result is None

    @pytest.mark.asyncio
    async def test_after_delete_hook_default(self, session, sample_algorithm):
        """Test default after_delete_hook does nothing"""
        result = await Base.after_delete_hook(session, sample_algorithm)
        assert result is None


class TestBasePydanticIntegration:
    """Tests for Base Pydantic conversion methods"""

    @pytest.mark.asyncio
    async def test_to_pydantic(self, sample_algorithm):
        """Test to_pydantic converts ORM to Pydantic model"""
        from rail_svc.models import Algorithm as AlgorithmPydantic

        pydantic_obj = Algorithm.to_pydantic(sample_algorithm)
        assert isinstance(pydantic_obj, AlgorithmPydantic)
        assert pydantic_obj.id_ == sample_algorithm.id_
        assert pydantic_obj.name == sample_algorithm.name
        assert pydantic_obj.class_name == sample_algorithm.class_name

    @pytest.mark.asyncio
    async def test_to_pydantic_list(self, multiple_algorithms):
        """Test to_pydantic_list converts list of ORM to Pydantic models"""
        from rail_svc.models import Algorithm as AlgorithmPydantic

        pydantic_list = Algorithm.to_pydantic_list(multiple_algorithms)
        assert len(pydantic_list) == 3
        assert all(isinstance(obj, AlgorithmPydantic) for obj in pydantic_list)
        assert pydantic_list[0].name == "knn"
        assert pydantic_list[1].name == "random_forest"

    @pytest.mark.asyncio
    async def test_to_pydantic_list_empty(self):
        """Test to_pydantic_list with empty list"""
        pydantic_list = Algorithm.to_pydantic_list([])
        assert pydantic_list == []

    @pytest.mark.asyncio
    async def test_to_pydantic_dict(self, sample_algorithm):
        """Test to_pydantic_dict converts ORM to dict"""
        pydantic_dict = Algorithm.to_pydantic_dict(sample_algorithm)
        assert isinstance(pydantic_dict, dict)
        assert pydantic_dict["id_"] == sample_algorithm.id_
        assert pydantic_dict["name"] == sample_algorithm.name
        assert pydantic_dict["class_name"] == sample_algorithm.class_name

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list(self, multiple_algorithms):
        """Test to_pydantic_dict_list converts list of ORM to dicts"""
        dict_list = Algorithm.to_pydantic_dict_list(multiple_algorithms)
        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert dict_list[0]["name"] == "knn"
        assert dict_list[1]["name"] == "random_forest"

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list_empty(self):
        """Test to_pydantic_dict_list with empty list"""
        dict_list = Algorithm.to_pydantic_dict_list([])
        assert dict_list == []


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestEnsureBaseInheritance:
    """Tests for ensure_base_inheritance utility function"""

    def test_ensure_base_inheritance_valid(self):
        """Test ensure_base_inheritance with valid class"""
        ensure_base_inheritance(Algorithm)  # Should not raise

    def test_ensure_base_inheritance_invalid(self):
        """Test ensure_base_inheritance with invalid class"""

        class NotABaseClass:
            pass

        with pytest.raises(TypeError, match="must inherit from"):
            ensure_base_inheritance(NotABaseClass)

    def test_ensure_base_inheritance_with_base_itself(self):
        """Test ensure_base_inheritance with Base class itself"""
        ensure_base_inheritance(Base)  # Should not raise


# ============================================================================
# Custom Hook Tests
# ============================================================================


class TestCustomHooks:
    """Tests for custom hook implementations"""

    @pytest.mark.asyncio
    async def test_get_hooks_with_custom_implementation(self):
        """Test get_hooks detects custom hook implementations"""

        class CustomModelHooks(Base):
            __tablename__ = "custom_hooks"
            id_: Mapped[int] = mapped_column(primary_key=True)
            name: Mapped[str] = mapped_column(String(255))

            @classmethod
            def pydantic_create_class(cls):
                return BaseModel

            @classmethod
            def pydantic_model_class(cls):
                return BaseModel

            @classmethod
            async def pre_create_hook(cls, session, data):
                # Custom implementation
                data["custom_field"] = "added"
                return data

            @classmethod
            async def after_delete_hook(cls, session, row):
                # Custom implementation
                pass

        hooks = CustomModelHooks.get_hooks()
        assert hooks["pre_create"] is True
        assert hooks["after_delete"] is True
        assert hooks["after_create"] is True
        assert hooks["pre_update"] is True
        assert hooks["after_update"] is True
        assert hooks["pre_delete"] is True


class TestPaginationLimit:
    """Tests for pagination limit functionality"""

    def test_default_pagination_limit(self):
        """Test default pagination limit is accessible"""
        assert Algorithm.get_pagination_limit() == 100

    def test_custom_pagination_limit(self):
        """Test custom pagination limit can be set"""

        class CustomAlgorithm(Base):
            __tablename__ = "custom_algo"
            id_: Mapped[int] = mapped_column(primary_key=True)

            default_pagination_limit = 50

            @classmethod
            def pydantic_create_class(cls):
                return BaseModel

            @classmethod
            def pydantic_model_class(cls):
                return BaseModel

        assert CustomAlgorithm.get_pagination_limit() == 50


class TestMetadata:
    """Tests for SQLAlchemy metadata and naming conventions"""

    def test_metadata_naming_convention_exists(self):
        """Test that naming convention is configured"""
        assert Base.metadata.naming_convention is not None

    def test_metadata_naming_convention_has_keys(self):
        """Test that naming convention has expected keys"""
        nc = Base.metadata.naming_convention
        assert "ix" in nc
        assert "uq" in nc
        assert "ck" in nc
        assert "fk" in nc
        assert "pk" in nc

    def test_algorithm_table_in_metadata(self):
        """Test that Algorithm table is in metadata"""
        assert "algorithm" in Base.metadata.tables


# ============================================================================
# Custom Model Tests (verify Base works with subclasses)
# ============================================================================


class TestCustomModelInheritance:
    """Tests that Base works correctly with custom subclasses"""

    @pytest.mark.asyncio
    async def test_custom_model_with_overridden_methods(self, engine):
        """Test that custom models can override Base methods"""

        class CustomPydantic(BaseModel):
            id_: int
            value: str

        class CustomPydanticCreate(BaseModel):
            value: str

        class CustomModel(Base):
            __tablename__ = "custom_table"
            id_: Mapped[int] = mapped_column(primary_key=True)
            value: Mapped[str] = mapped_column(String(255))

            @classmethod
            def pydantic_create_class(cls):
                return CustomPydanticCreate

            @classmethod
            def pydantic_model_class(cls):
                return CustomPydantic

            @classmethod
            def class_string(cls):
                return "custom"

            @classmethod
            def get_pagination_limit(cls):
                return 25

        # Create table
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Test methods
        assert CustomModel.class_string() == "custom"
        assert CustomModel.get_pagination_limit() == 25
        assert CustomModel.pydantic_create_class() == CustomPydanticCreate
        assert CustomModel.pydantic_model_class() == CustomPydantic
