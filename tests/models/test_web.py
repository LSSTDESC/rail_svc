"""Unit tests for web models"""

import pytest
from pydantic import BaseModel, ValidationError

from rail_svc.models.filtering import Filter, FilterOp, OrderBy
from rail_svc.models.web import (AsyncRouteError, CountResponse,
                                 DeleteResponse, FilterRequest, FindRequest,
                                 LookupResponse, RemoteAPIError)


class TestCustomExceptions:
    """Tests for custom exception classes"""

    def test_async_route_error(self):
        """Test AsyncRouteError can be raised and caught"""
        with pytest.raises(AsyncRouteError):
            raise AsyncRouteError("Test error")

    def test_async_route_error_with_message(self):
        """Test AsyncRouteError preserves message"""
        error_msg = "Something went wrong"
        try:
            raise AsyncRouteError(error_msg)
        except AsyncRouteError as e:
            assert str(e) == error_msg

    def test_remote_api_error(self):
        """Test RemoteAPIError can be raised and caught"""
        with pytest.raises(RemoteAPIError):
            raise RemoteAPIError("API call failed")

    def test_remote_api_error_with_message(self):
        """Test RemoteAPIError preserves message"""
        error_msg = "Connection timeout"
        try:
            raise RemoteAPIError(error_msg)
        except RemoteAPIError as e:
            assert str(e) == error_msg


class TestCountResponse:
    """Tests for CountResponse model"""

    def test_valid_count_response(self):
        """Test creating valid CountResponse"""
        response = CountResponse(count=42)
        assert response.count == 42

    def test_zero_count(self):
        """Test CountResponse with zero count"""
        response = CountResponse(count=0)
        assert response.count == 0

    def test_large_count(self):
        """Test CountResponse with large count"""
        response = CountResponse(count=1000000)
        assert response.count == 1000000

    def test_missing_count_raises_error(self):
        """Test that missing count raises ValidationError"""
        with pytest.raises(ValidationError):
            CountResponse()

    def test_count_serialization(self):
        """Test JSON serialization of CountResponse"""
        response = CountResponse(count=100)
        data = response.model_dump()
        assert data == {"count": 100}


class TestLookupResponse:
    """Tests for LookupResponse model"""

    def test_valid_lookup_response(self):
        """Test creating valid LookupResponse"""

        class TestData(BaseModel):
            name: str
            value: int

        data = TestData(name="test", value=42)
        response = LookupResponse[TestData](id=1, data=data)
        assert response.id == 1
        assert response.data.name == "test"
        assert response.data.value == 42

    def test_lookup_response_with_simple_data(self):
        """Test LookupResponse with simple BaseModel data"""

        class SimpleModel(BaseModel):
            field: str

        response = LookupResponse[SimpleModel](id=99, data=SimpleModel(field="value"))
        assert response.id == 99
        assert response.data.field == "value"

    def test_missing_id_raises_error(self):
        """Test that missing id raises ValidationError"""

        class TestData(BaseModel):
            name: str

        with pytest.raises(ValidationError):
            LookupResponse[TestData](data=TestData(name="test"))

    def test_missing_data_raises_error(self):
        """Test that missing data raises ValidationError"""
        with pytest.raises(ValidationError):
            LookupResponse(id=1)

    def test_lookup_response_serialization(self):
        """Test serialization of LookupResponse"""

        class TestData(BaseModel):
            name: str

        response = LookupResponse[TestData](id=5, data=TestData(name="test"))
        data = response.model_dump()
        assert data["id"] == 5
        assert data["data"]["name"] == "test"


class TestDeleteResponse:
    """Tests for DeleteResponse model"""

    def test_valid_delete_response(self):
        """Test creating valid DeleteResponse"""
        response = DeleteResponse()
        assert response.deleted is True

    def test_delete_response_explicit_true(self):
        """Test DeleteResponse with explicit True"""
        response = DeleteResponse(deleted=True)
        assert response.deleted is True

    def test_delete_response_false(self):
        """Test DeleteResponse with False"""
        response = DeleteResponse(deleted=False)
        assert response.deleted is False

    def test_delete_response_serialization(self):
        """Test serialization of DeleteResponse"""
        response = DeleteResponse()
        data = response.model_dump()
        assert data == {"deleted": True}


class TestFilterRequest:
    """Tests for FilterRequest model"""

    def test_valid_filter_request(self):
        """Test creating valid FilterRequest"""
        filters = [
            Filter(field="age", op=FilterOp.GT, value=18),
            Filter(field="status", op=FilterOp.EQ, value="active"),
        ]
        request = FilterRequest(filters=filters, logical_op="and", skip=10, limit=50)
        assert len(request.filters) == 2
        assert request.logical_op == "and"
        assert request.skip == 10
        assert request.limit == 50

    def test_filter_request_defaults(self):
        """Test FilterRequest default values"""
        request = FilterRequest()
        assert request.filters == []
        assert request.logical_op == "and"
        assert request.order_by is None
        assert request.skip == 0
        assert request.limit is None

    def test_filter_request_with_single_order_by(self):
        """Test FilterRequest with single OrderBy"""
        order = OrderBy(field="created_at", descending=True)
        request = FilterRequest(order_by=order)
        assert request.order_by == order

    def test_filter_request_with_multiple_order_by(self):
        """Test FilterRequest with list of OrderBy"""
        orders = [OrderBy(field="priority", descending=True), OrderBy(field="name", descending=False)]
        request = FilterRequest(order_by=orders)
        assert request.order_by == orders
        assert len(request.order_by) == 2

    def test_filter_request_with_or_logical_op(self):
        """Test FilterRequest with 'or' logical operator"""
        request = FilterRequest(logical_op="or")
        assert request.logical_op == "or"

    def test_filter_request_zero_skip(self):
        """Test FilterRequest with zero skip"""
        request = FilterRequest(skip=0)
        assert request.skip == 0

    def test_filter_request_serialization(self):
        """Test serialization of FilterRequest"""
        filters = [Filter(field="test", op=FilterOp.EQ, value="value")]
        request = FilterRequest(filters=filters, limit=100)
        data = request.model_dump()
        assert len(data["filters"]) == 1
        assert data["limit"] == 100


class TestFindRequest:
    """Tests for FindRequest model"""

    def test_valid_find_request(self):
        """Test creating valid FindRequest"""
        order = OrderBy(field="name", descending=False)
        request = FindRequest(order_by=order, skip=20, limit=100)
        assert request.order_by == order
        assert request.skip == 20
        assert request.limit == 100

    def test_find_request_defaults(self):
        """Test FindRequest default values"""
        request = FindRequest()
        assert request.order_by is None
        assert request.skip == 0
        assert request.limit is None

    def test_find_request_with_multiple_order_by(self):
        """Test FindRequest with list of OrderBy"""
        orders = [OrderBy(field="date", descending=True), OrderBy(field="id", descending=False)]
        request = FindRequest(order_by=orders)
        assert request.order_by == orders

    def test_find_request_allows_extra_fields(self):
        """Test that FindRequest allows extra fields"""
        request = FindRequest(skip=0, limit=10, custom_field="custom_value", another_field=42)
        assert request.skip == 0
        assert request.limit == 10

    def test_find_request_serialization(self):
        """Test serialization of FindRequest"""
        order = OrderBy(field="created_at", descending=True)
        request = FindRequest(order_by=order, skip=5, limit=25)
        data = request.model_dump()
        assert data["skip"] == 5
        assert data["limit"] == 25
        assert "order_by" in data


class TestModelSerialization:
    """Tests for model serialization/deserialization"""

    def test_filter_request_json_serialization(self):
        """Test JSON serialization of FilterRequest"""
        filters = [Filter(field="age", op=FilterOp.GT, value=18)]
        original = FilterRequest(filters=filters, limit=50)
        json_str = original.model_dump_json()
        restored = FilterRequest.model_validate_json(json_str)
        assert len(restored.filters) == len(original.filters)
        assert restored.limit == original.limit

    def test_find_request_json_serialization(self):
        """Test JSON serialization of FindRequest"""
        original = FindRequest(skip=10, limit=20)
        json_str = original.model_dump_json()
        restored = FindRequest.model_validate_json(json_str)
        assert restored.skip == original.skip
        assert restored.limit == original.limit

    def test_count_response_json_serialization(self):
        """Test JSON serialization of CountResponse"""
        original = CountResponse(count=42)
        json_str = original.model_dump_json()
        restored = CountResponse.model_validate_json(json_str)
        assert restored.count == original.count

    def test_delete_response_json_serialization(self):
        """Test JSON serialization of DeleteResponse"""
        original = DeleteResponse(deleted=True)
        json_str = original.model_dump_json()
        restored = DeleteResponse.model_validate_json(json_str)
        assert restored.deleted == original.deleted


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_filter_request_with_empty_filters(self):
        """Test FilterRequest with empty filters list"""
        request = FilterRequest(filters=[])
        assert request.filters == []

    def test_filter_request_with_none_limit(self):
        """Test FilterRequest with None limit"""
        request = FilterRequest(limit=None)
        assert request.limit is None

    def test_find_request_with_zero_limit(self):
        """Test FindRequest with zero limit"""
        request = FindRequest(limit=0)
        assert request.limit == 0

    def test_negative_skip_allowed(self):
        """Test that negative skip values are allowed (no validation constraint)"""
        request = FilterRequest(skip=-1)
        assert request.skip == -1

    def test_negative_limit_allowed(self):
        """Test that negative limit values are allowed (no validation constraint)"""
        request = FilterRequest(limit=-1)
        assert request.limit == -1
