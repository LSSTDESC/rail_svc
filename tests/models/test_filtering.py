"""Unit tests for filtering models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.filtering import Filter, FilterOp, OrderBy


class TestFilterOp:
    """Tests for FilterOp enum"""

    def test_filter_op_string_values(self):
        """Test that FilterOp can be compared as strings"""
        assert FilterOp.EQ == "eq"
        assert FilterOp.GT == "gt"


class TestFilter:
    """Tests for Filter model"""

    def test_valid_filter_with_value(self):
        """Test creating valid Filter with value"""
        filter_obj = Filter(field="age", op=FilterOp.GT, value=18)
        assert filter_obj.field == "age"
        assert filter_obj.op == FilterOp.GT
        assert filter_obj.value == 18

    def test_filter_with_string_op(self):
        """Test creating Filter with string operator"""
        filter_obj = Filter(field="status", op="eq", value="active")
        assert filter_obj.op == FilterOp.EQ

    def test_filter_without_value(self):
        """Test creating Filter without value for IS_NULL"""
        filter_obj = Filter(field="deleted_at", op=FilterOp.IS_NULL)
        assert filter_obj.field == "deleted_at"
        assert filter_obj.op == FilterOp.IS_NULL
        assert filter_obj.value is None

    def test_filter_with_list_value(self):
        """Test creating Filter with list value for IN operator"""
        filter_obj = Filter(field="status", op=FilterOp.IN, value=["active", "pending"])
        assert filter_obj.value == ["active", "pending"]

    def test_filter_with_tuple_value_for_between(self):
        """Test creating Filter with tuple value for BETWEEN"""
        filter_obj = Filter(field="age", op=FilterOp.BETWEEN, value=(18, 65))
        assert filter_obj.value == (18, 65)

    def test_filter_with_like_pattern(self):
        """Test creating Filter with LIKE pattern"""
        filter_obj = Filter(field="name", op=FilterOp.LIKE, value="John%")
        assert filter_obj.value == "John%"

    def test_filter_with_none_value_explicitly(self):
        """Test creating Filter with explicitly None value"""
        filter_obj = Filter(field="test", op=FilterOp.EQ, value=None)
        assert filter_obj.value is None

    def test_missing_field_raises_error(self):
        """Test that missing field raises ValidationError"""
        with pytest.raises(ValidationError):
            Filter(op=FilterOp.EQ, value=10)

    def test_missing_op_raises_error(self):
        """Test that missing op raises ValidationError"""
        with pytest.raises(ValidationError):
            Filter(field="age", value=10)

    def test_invalid_op_raises_error(self):
        """Test that invalid operator raises ValidationError"""
        with pytest.raises(ValidationError):
            Filter(field="age", op="invalid_op", value=10)

    def test_filter_repr(self):
        """Test Filter __repr__ method"""
        filter_obj = Filter(field="age", op=FilterOp.GT, value=18)
        repr_str = repr(filter_obj)
        assert "Filter" in repr_str
        assert "age" in repr_str
        assert "gt" in repr_str
        assert "18" in repr_str

    def test_filter_with_various_value_types(self):
        """Test Filter with different value types"""
        # Integer
        f1 = Filter(field="count", op=FilterOp.EQ, value=42)
        assert f1.value == 42

        # Float
        f2 = Filter(field="price", op=FilterOp.GT, value=19.99)
        assert f2.value == 19.99

        # String
        f3 = Filter(field="name", op=FilterOp.EQ, value="test")
        assert f3.value == "test"

        # Boolean
        f4 = Filter(field="active", op=FilterOp.EQ, value=True)
        assert f4.value is True

        # Dict
        f5 = Filter(field="metadata", op=FilterOp.CONTAINS, value={"key": "value"})
        assert f5.value == {"key": "value"}


class TestOrderBy:
    """Tests for OrderBy model"""

    def test_valid_order_by_ascending(self):
        """Test creating valid OrderBy for ascending order"""
        order = OrderBy(field="name", descending=False)
        assert order.field == "name"
        assert order.descending is False

    def test_valid_order_by_descending(self):
        """Test creating valid OrderBy for descending order"""
        order = OrderBy(field="created_at", descending=True)
        assert order.field == "created_at"
        assert order.descending is True

    def test_descending_defaults_to_false(self):
        """Test that descending defaults to False"""
        order = OrderBy(field="name")
        assert order.descending is False

    def test_missing_field_raises_error(self):
        """Test that missing field raises ValidationError"""
        with pytest.raises(ValidationError):
            OrderBy(descending=True)

    def test_order_by_repr_ascending(self):
        """Test OrderBy __repr__ for ascending order"""
        order = OrderBy(field="name", descending=False)
        repr_str = repr(order)
        assert "OrderBy" in repr_str
        assert "name" in repr_str
        assert "ASC" in repr_str

    def test_order_by_repr_descending(self):
        """Test OrderBy __repr__ for descending order"""
        order = OrderBy(field="created_at", descending=True)
        repr_str = repr(order)
        assert "OrderBy" in repr_str
        assert "created_at" in repr_str
        assert "DESC" in repr_str


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_filter_to_dict(self):
        """Test converting Filter to dict"""
        filter_obj = Filter(field="age", op=FilterOp.GT, value=18)
        data = filter_obj.model_dump()
        assert data["field"] == "age"
        assert data["op"] == "gt"
        assert data["value"] == 18

    def test_filter_json_serialization(self):
        """Test JSON serialization round-trip for Filter"""
        original = Filter(field="status", op=FilterOp.IN, value=["active", "pending"])
        json_str = original.model_dump_json()
        restored = Filter.model_validate_json(json_str)
        assert restored.field == original.field
        assert restored.op == original.op
        assert restored.value == original.value

    def test_order_by_to_dict(self):
        """Test converting OrderBy to dict"""
        order = OrderBy(field="created_at", descending=True)
        data = order.model_dump()
        assert data["field"] == "created_at"
        assert data["descending"] is True

    def test_order_by_json_serialization(self):
        """Test JSON serialization round-trip for OrderBy"""
        original = OrderBy(field="name", descending=False)
        json_str = original.model_dump_json()
        restored = OrderBy.model_validate_json(json_str)
        assert restored.field == original.field
        assert restored.descending == original.descending

    def test_filter_from_dict(self):
        """Test creating Filter from dict"""
        data = {"field": "price", "op": "lt", "value": 100.0}
        filter_obj = Filter(**data)
        assert filter_obj.field == "price"
        assert filter_obj.op == FilterOp.LT
        assert filter_obj.value == 100.0


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_filter_with_zero_value(self):
        """Test Filter with zero as value"""
        filter_obj = Filter(field="count", op=FilterOp.EQ, value=0)
        assert filter_obj.value == 0

    def test_filter_with_empty_string_value(self):
        """Test Filter with empty string as value"""
        filter_obj = Filter(field="name", op=FilterOp.EQ, value="")
        assert filter_obj.value == ""

    def test_filter_with_empty_list_value(self):
        """Test Filter with empty list as value"""
        filter_obj = Filter(field="ids", op=FilterOp.IN, value=[])
        assert filter_obj.value == []

    def test_order_by_with_empty_string_field(self):
        """Test OrderBy with empty string field"""
        order = OrderBy(field="", descending=True)
        assert order.field == ""

    def test_multiple_filters_independence(self):
        """Test that multiple Filter instances are independent"""
        f1 = Filter(field="age", op=FilterOp.GT, value=18)
        f2 = Filter(field="status", op=FilterOp.EQ, value="active")

        assert f1.field != f2.field
        assert f1.op != f2.op
        assert f1.value != f2.value
