"""Unit tests for table operations."""

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rail_svc.db import Algorithm
from rail_svc.db_oper.base import (
    FileValidatedOperations,
    TableContext,
    TableOperations,
    create_operations,
    forward_to_db_funcs,
    forward_to_db_funcs_streaming,
)
from rail_svc.models.filtering import Filter, FilterOp


# ============================================================================
# Mock Models for Testing
# ============================================================================


class MockResponse(BaseModel):
    """Mock Pydantic response model."""

    id_: int
    name: str
    class_name: str


class MockCreate(BaseModel):
    """Mock Pydantic create model."""

    name: str
    class_name: str


class MockFileResponse(BaseModel):
    """Mock response model with file fields."""

    id_: int
    path: str
    n_objects: int


class MockFileCreate(BaseModel):
    """Mock create model with file fields."""

    path: str | None = None
    n_objects: int | None = None


# ============================================================================
# TableContext tests
# ============================================================================


def test_table_context_creation():
    """Test creating TableContext with explicit types."""
    context = TableContext(
        db_class=Algorithm, response_class=MockResponse, create_class=MockCreate, class_string="algorithm"
    )

    assert context.db_class == Algorithm
    assert context.response_class == MockResponse
    assert context.create_class == MockCreate
    assert context.class_string == "algorithm"


def test_table_context_from_db_class():
    """Test creating TableContext from database class."""
    context = TableContext.from_db_class(Algorithm)

    assert context.db_class == Algorithm
    assert issubclass(context.response_class, BaseModel)
    assert issubclass(context.create_class, BaseModel)
    assert context.class_string == "algorithm"


def test_table_context_from_db_class_missing_methods():
    """Test that from_db_class raises error for missing methods."""

    # Create a mock class that doesn't have required methods
    BadModel = Mock(spec=[])  # Empty spec means no methods
    BadModel.__name__ = "BadModel"

    with pytest.raises(AttributeError, match="must implement pydantic_model_class"):
        TableContext.from_db_class(BadModel)


# ============================================================================
# TableOperations basic tests
# ============================================================================


@pytest.fixture
def table_ops():
    """Create TableOperations fixture."""
    context = TableContext.from_db_class(Algorithm)
    return TableOperations(context)


@pytest.mark.asyncio
async def test_get_row_forwarding(session, sample_algorithm, table_ops):
    """Test that get_row forwards to db_funcs correctly."""
    result = await table_ops.get_row(session, sample_algorithm.id_)

    assert result.id_ == sample_algorithm.id_
    assert result.name == sample_algorithm.name


@pytest.mark.asyncio
async def test_get_row_by_name_forwarding(session, sample_algorithm, table_ops):
    """Test that get_row_by_name forwards correctly."""
    result = await table_ops.get_row_by_name(session, "test_algorithm")

    assert result.id_ == sample_algorithm.id_
    assert result.name == "test_algorithm"


@pytest.mark.asyncio
async def test_get_rows_forwarding(session, multiple_algorithms, table_ops):
    """Test that get_rows forwards correctly."""
    results = await table_ops.get_rows(session, skip=0, limit=10)

    assert len(results) == 3
    assert all(isinstance(r, Algorithm) for r in results)


@pytest.mark.asyncio
async def test_get_rows_streaming_forwarding(session, multiple_algorithms, table_ops):
    """Test that get_rows_streaming forwards correctly."""
    results = []
    async for row in table_ops.get_rows_streaming(session, skip=0, limit=10):
        results.append(row)

    assert len(results) == 3
    assert all(isinstance(r, Algorithm) for r in results)


@pytest.mark.asyncio
async def test_get_row_or_none_forwarding(session, table_ops):
    """Test that get_row_or_none forwards correctly."""
    result = await table_ops.get_row_or_none(session, 99999)
    assert result is None


@pytest.mark.asyncio
async def test_count_rows_forwarding(session, multiple_algorithms, table_ops):
    """Test that count_rows forwards correctly."""
    count = await table_ops.count_rows(session)
    assert count == 3


@pytest.mark.asyncio
async def test_lookup_by_id_or_name_forwarding(session, sample_algorithm, table_ops):
    """Test that lookup_by_id_or_name forwards correctly."""
    row_id, obj = await table_ops.lookup_by_id_or_name(
        session, row_id=sample_algorithm.id_, name=None, need_object=True
    )

    assert row_id == sample_algorithm.id_
    assert obj is not None
    assert obj.id_ == sample_algorithm.id_


@pytest.mark.asyncio
async def test_update_row_forwarding(session, sample_algorithm, table_ops):
    """Test that update_row forwards correctly."""
    updated = await table_ops.update_row(session, row_id=sample_algorithm.id_, name="updated_name")

    assert updated.name == "updated_name"


@pytest.mark.asyncio
async def test_update_rows_forwarding(session, multiple_algorithms, table_ops):
    """Test that update_rows forwards correctly."""
    updates = [
        {"id": multiple_algorithms[0].id_, "name": "updated_1"},
        {"id": multiple_algorithms[1].id_, "name": "updated_2"},
    ]

    updated = await table_ops.update_rows(session, updates)

    assert len(updated) == 2
    assert updated[0].name == "updated_1"
    assert updated[1].name == "updated_2"


@pytest.mark.asyncio
async def test_delete_row_forwarding(session, sample_algorithm, table_ops):
    """Test that delete_row forwards correctly."""
    result = await table_ops.delete_row(session, row_id=sample_algorithm.id_, capture_data=True)

    assert result is not None
    assert result["id_"] == sample_algorithm.id_


@pytest.mark.asyncio
async def test_delete_rows_forwarding(session, multiple_algorithms, table_ops):
    """Test that delete_rows forwards correctly."""
    row_ids = [algo.id_ for algo in multiple_algorithms]

    result = await table_ops.delete_rows(session, row_ids=row_ids, capture_data=True)

    assert result is not None
    assert len(result) == 3


@pytest.mark.asyncio
async def test_bulk_delete_rows_forwarding(session, multiple_algorithms, table_ops):
    """Test that bulk_delete_rows forwards correctly."""
    row_ids = [algo.id_ for algo in multiple_algorithms]

    count = await table_ops.bulk_delete_rows(session, row_ids=row_ids)

    assert count == 3


# ============================================================================
# Filter operations tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_forwarding(session, multiple_algorithms, table_ops):
    """Test that filter_rows forwards correctly."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    results = await table_ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "knn"


@pytest.mark.asyncio
async def test_filter_rows_streaming_forwarding(session, multiple_algorithms, table_ops):
    """Test that filter_rows_streaming forwards correctly."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    results = []

    async for row in table_ops.filter_rows_streaming(session, filters=filters):
        results.append(row)

    assert len(results) == 1
    assert results[0].name == "knn"


@pytest.mark.asyncio
async def test_count_filtered_rows_forwarding(session, multiple_algorithms, table_ops):
    """Test that count_filtered_rows forwards correctly."""
    filters = [Filter(field="name", op=FilterOp.IN, value=["knn", "xgboost"])]
    count = await table_ops.count_filtered_rows(session, filters=filters)

    assert count == 2


@pytest.mark.asyncio
async def test_filter_one_forwarding(session, sample_algorithm, table_ops):
    """Test that filter_one forwards correctly."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="test_algorithm")]
    result = await table_ops.filter_one(session, filters=filters)

    assert result.name == "test_algorithm"


@pytest.mark.asyncio
async def test_filter_one_or_none_forwarding(session, table_ops):
    """Test that filter_one_or_none forwards correctly."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="nonexistent")]
    result = await table_ops.filter_one_or_none(session, filters=filters)

    assert result is None


@pytest.mark.asyncio
async def test_find_by_forwarding(session, sample_algorithm, table_ops):
    """Test that find_by forwards correctly."""
    results = await table_ops.find_by(session, name="test_algorithm")

    assert len(results) == 1
    assert results[0].name == "test_algorithm"


@pytest.mark.asyncio
async def test_find_one_by_forwarding(session, sample_algorithm, table_ops):
    """Test that find_one_by forwards correctly."""
    result = await table_ops.find_one_by(session, name="test_algorithm")

    assert result.name == "test_algorithm"


# ============================================================================
# create_row tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_basic(session, table_ops):
    """Test creating a single row."""
    async with session.begin():
        result = await table_ops.create_row(session, name="new_algo", class_name="new.class.Name")

    assert result.name == "new_algo"
    assert result.class_name == "new.class.Name"
    assert result.id_ is not None


@pytest.mark.asyncio
async def test_create_row_with_validation(session, table_ops):
    """Test create_row with validation enabled."""
    async with session.begin():
        result = await table_ops.create_row(
            session, validate=True, name="validated_algo", class_name="validated.class"
        )

    assert result.name == "validated_algo"


@pytest.mark.asyncio
async def test_create_row_validation_failure(session, table_ops):
    """Test create_row fails validation with invalid data."""
    # Missing required field
    with pytest.raises(ValidationError):
        async with session.begin():
            await table_ops.create_row(
                session,
                validate=True,
                name="incomplete",
                # Missing class_name
            )


@pytest.mark.asyncio
async def test_create_row_without_validation(session, table_ops):
    """Test create_row with validation disabled."""
    async with session.begin():
        result = await table_ops.create_row(
            session, validate=False, name="no_validation", class_name="test.class"
        )

    assert result.name == "no_validation"


# ============================================================================
# create_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_rows_multiple(session, table_ops):
    """Test creating multiple rows."""
    rows_data = [
        {"name": "algo1", "class_name": "class1"},
        {"name": "algo2", "class_name": "class2"},
        {"name": "algo3", "class_name": "class3"},
    ]

    async with session.begin():
        results = await table_ops.create_rows(session, rows_data)

    assert len(results) == 3
    assert results[0].name == "algo1"
    assert results[1].name == "algo2"
    assert results[2].name == "algo3"


@pytest.mark.asyncio
async def test_create_rows_empty_list(session, table_ops):
    """Test create_rows with empty list raises ValueError."""
    with pytest.raises(ValueError, match="rows_data cannot be empty"):
        async with session.begin():
            await table_ops.create_rows(session, [])


@pytest.mark.asyncio
async def test_create_rows_validation_failure(session, table_ops):
    """Test create_rows fails if any row fails validation."""
    rows_data = [
        {"name": "valid", "class_name": "valid.class"},
        {"name": "invalid"},  # Missing class_name
    ]

    with pytest.raises(ValidationError):
        async with session.begin():
            await table_ops.create_rows(session, rows_data, validate=True)


@pytest.mark.asyncio
async def test_create_rows_without_validation(session, table_ops):
    """Test create_rows with validation disabled."""
    rows_data = [
        {"name": "no_val1", "class_name": "class1"},
        {"name": "no_val2", "class_name": "class2"},
    ]

    async with session.begin():
        results = await table_ops.create_rows(session, rows_data, validate=False)

    assert len(results) == 2


# ============================================================================
# create_rows_batched tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_rows_batched(session, table_ops):
    """Test creating rows in batches."""
    rows_data = [{"name": f"batch_{i}", "class_name": f"class_{i}"} for i in range(10)]

    results = await table_ops.create_rows_batched(session, rows_data, batch_size=3)

    assert len(results) == 10


@pytest.mark.asyncio
async def test_create_rows_batched_empty_list(session, table_ops):
    """Test batched create with empty list raises ValueError."""
    with pytest.raises(ValueError, match="rows_data cannot be empty"):
        await table_ops.create_rows_batched(session, [])


@pytest.mark.asyncio
async def test_create_rows_batched_invalid_batch_size(session, table_ops):
    """Test batched create with invalid batch size raises ValueError."""
    rows_data = [{"name": "test", "class_name": "test"}]

    with pytest.raises(ValueError, match="batch_size must be at least 1"):
        await table_ops.create_rows_batched(session, rows_data, batch_size=0)


# ============================================================================
# bulk_insert_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_bulk_insert_rows(session, table_ops):
    """Test bulk inserting rows."""
    rows_data = [{"name": f"bulk_{i}", "class_name": f"class_{i}"} for i in range(5)]

    count = await table_ops.bulk_insert_rows(session, rows_data)

    assert count == 5


@pytest.mark.asyncio
async def test_bulk_insert_rows_empty_list(session, table_ops):
    """Test bulk insert with empty list raises ValueError."""
    with pytest.raises(ValueError, match="rows_data cannot be empty"):
        await table_ops.bulk_insert_rows(session, [])


@pytest.mark.asyncio
async def test_bulk_insert_rows_validation(session, table_ops):
    """Test bulk insert with validation."""
    rows_data = [
        {"name": "valid", "class_name": "valid.class"},
        {"name": "invalid"},  # Missing class_name
    ]

    with pytest.raises(ValidationError):
        await table_ops.bulk_insert_rows(session, rows_data, validate=True)


@pytest.mark.asyncio
async def test_bulk_insert_rows_without_validation(session, table_ops):
    """Test bulk insert without validation."""
    rows_data = [
        {"name": "no_val1", "class_name": "class1"},
        {"name": "no_val2", "class_name": "class2"},
    ]

    count = await table_ops.bulk_insert_rows(session, rows_data, validate=False)

    assert count == 2


# ============================================================================
# Pydantic conversion tests
# ============================================================================


@pytest.mark.asyncio
async def test_to_pydantic(session, sample_algorithm, table_ops):
    """Test converting row to Pydantic model."""
    pydantic_obj = table_ops.to_pydantic(sample_algorithm)

    assert isinstance(pydantic_obj, BaseModel)
    assert hasattr(pydantic_obj, "name")
    assert pydantic_obj.name == sample_algorithm.name


@pytest.mark.asyncio
async def test_to_pydantic_list(session, multiple_algorithms, table_ops):
    """Test converting multiple rows to Pydantic models."""
    pydantic_list = table_ops.to_pydantic_list(multiple_algorithms)

    assert len(pydantic_list) == 3
    assert all(isinstance(obj, BaseModel) for obj in pydantic_list)
    assert pydantic_list[0].name == multiple_algorithms[0].name


@pytest.mark.asyncio
async def test_to_pydantic_dict(session, sample_algorithm, table_ops):
    """Test converting row to dictionary."""
    result_dict = table_ops.to_pydantic_dict(sample_algorithm)

    assert isinstance(result_dict, dict)
    assert "name" in result_dict
    assert result_dict["name"] == sample_algorithm.name


@pytest.mark.asyncio
async def test_to_pydantic_dict_list(session, multiple_algorithms, table_ops):
    """Test converting multiple rows to dictionaries."""
    dict_list = table_ops.to_pydantic_dict_list(multiple_algorithms)

    assert len(dict_list) == 3
    assert all(isinstance(d, dict) for d in dict_list)
    assert dict_list[0]["name"] == multiple_algorithms[0].name


@pytest.mark.asyncio
async def test_to_pydantic_list_empty(session, table_ops):
    """Test to_pydantic_list with empty list."""
    result = table_ops.to_pydantic_list([])

    assert result == []
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_to_pydantic_dict_list_empty(session, table_ops):
    """Test to_pydantic_dict_list with empty list."""
    result = table_ops.to_pydantic_dict_list([])

    assert result == []
    assert isinstance(result, list)


# ============================================================================
# Path validation tests
# ============================================================================


def test_validate_path_security_valid():
    """Test path validation with valid path."""
    context = TableContext.from_db_class(Algorithm)
    ops = TableOperations(context)

    with patch("rail_svc.db_oper.base.global_config") as mock_config:
        mock_config.storage.archive = "/var/archive"

        result = ops._validate_path_security("data/test.hdf5")

        assert result == Path("/var/archive/data/test.hdf5").resolve()


def test_validate_path_security_traversal_attempt():
    """Test path validation rejects directory traversal."""
    context = TableContext.from_db_class(Algorithm)
    ops = TableOperations(context)

    with patch("rail_svc.db_oper.base.global_config") as mock_config:
        mock_config.storage.archive = "/var/archive"

        with pytest.raises(ValueError, match="Invalid path"):
            ops._validate_path_security("../etc/passwd")


def test_validate_path_security_absolute_path():
    """Test path validation rejects absolute paths."""
    context = TableContext.from_db_class(Algorithm)
    ops = TableOperations(context)

    with patch("rail_svc.db_oper.base.global_config") as mock_config:
        mock_config.storage.archive = "/var/archive"

        with pytest.raises(ValueError, match="Invalid path"):
            ops._validate_path_security("/etc/passwd")


def test_validate_path_security_too_long():
    """Test path validation rejects overly long paths."""
    context = TableContext.from_db_class(Algorithm)
    ops = TableOperations(context)

    with patch("rail_svc.db_oper.base.global_config") as mock_config:
        mock_config.storage.archive = "/var/archive"
        long_path = "a" * 300

        with pytest.raises(ValueError, match="Path too long"):
            ops._validate_path_security(long_path)


def test_validate_path_security_null_bytes():
    """Test path validation rejects null bytes."""
    context = TableContext.from_db_class(Algorithm)
    ops = TableOperations(context)

    with patch("rail_svc.db_oper.base.global_config") as mock_config:
        mock_config.storage.archive = "/var/archive"

        with pytest.raises(ValueError, match="Path contains null bytes"):
            ops._validate_path_security("data\x00test.hdf5")


def test_validate_path_security_escape_with_forbid():
    """Test path validation with FORBID_TRAVERSAL enabled."""
    import rail_svc.db_oper.base as ops_module

    original_forbid = ops_module.FORBID_TRAVERSAL

    try:
        ops_module.FORBID_TRAVERSAL = True

        context = TableContext.from_db_class(Algorithm)
        ops = TableOperations(context)

        with patch("rail_svc.db_oper.base.global_config") as mock_config:
            mock_config.storage.archive = "/var/archive"

            # Path that would resolve outside archive
            with pytest.raises(ValueError, match="Invalid path"):
                ops._validate_path_security("data/../../etc/passwd")
    finally:
        ops_module.FORBID_TRAVERSAL = original_forbid


# ============================================================================
# FileValidatedOperations tests
# ============================================================================


class MockFileOps(FileValidatedOperations):
    """Mock FileValidatedOperations for testing."""

    def get_file_length(self, path: Path) -> int:
        """Mock implementation returning fixed length."""
        return 1000

    async def get_create_kwargs(self, session: AsyncSession, **kwargs: Any) -> dict[str, Any]:
        """Mock implementation."""
        return kwargs


@pytest.fixture
def file_ops():
    """Create FileValidatedOperations fixture."""
    context = TableContext(
        db_class=Algorithm,
        response_class=MockFileResponse,
        create_class=MockFileCreate,
        class_string="algorithm",
    )
    return MockFileOps(context)


@pytest.mark.asyncio
async def test_process_path_no_path_no_objects(session, file_ops):
    """Test _process_path raises error when neither path nor n_objects provided."""
    with pytest.raises(ValueError, match="Either 'path' or 'n_objects' must be provided"):
        await file_ops._process_path(None, None, validate_file=True, extra_kwargs={})


@pytest.mark.asyncio
async def test_process_path_no_path_with_objects(session, file_ops):
    """Test _process_path returns n_objects when no path provided."""
    result = await file_ops._process_path(None, None, validate_file=True, extra_kwargs={"n_objects": 500})

    assert result == 500


@pytest.mark.asyncio
async def test_process_path_no_validation_no_objects(session, file_ops):
    """Test _process_path requires n_objects when validation disabled."""
    with pytest.raises(ValueError, match="n_objects' must be provided"):
        await file_ops._process_path("data/test.hdf5", None, validate_file=False, extra_kwargs={})


@pytest.mark.asyncio
async def test_process_path_no_validation_with_objects(session, file_ops):
    """Test _process_path uses provided n_objects when validation disabled."""
    result = await file_ops._process_path(
        "data/test.hdf5", None, validate_file=False, extra_kwargs={"n_objects": 300}
    )

    assert result == 300


@pytest.mark.asyncio
async def test_validate_data_for_path_file_not_found(session, file_ops):
    """Test validate_data_for_path raises FileNotFoundError."""
    nonexistent_path = Path("/tmp/nonexistent_file_12345.hdf5")

    with pytest.raises(FileNotFoundError, match="not found"):
        await file_ops.validate_data_for_path(nonexistent_path, None)


@pytest.mark.asyncio
async def test_validate_data_for_path_success(session, file_ops, tmp_path):
    """Test validate_data_for_path successfully reads file."""
    test_file = tmp_path / "test.hdf5"
    test_file.write_text("test data")

    result = await file_ops.validate_data_for_path(test_file, None)

    assert result == 1000  # From mock get_file_length


@pytest.mark.asyncio
async def test_validate_data_for_path_read_error(session, file_ops, tmp_path):
    """Test validate_data_for_path handles read errors."""
    test_file = tmp_path / "test.hdf5"
    test_file.write_text("test data")

    # Mock get_file_length to raise OSError
    with patch.object(file_ops, "get_file_length", side_effect=OSError("Read error")):
        with pytest.raises(ValueError, match="Could not read data"):
            await file_ops.validate_data_for_path(test_file, None)


@pytest.mark.asyncio
async def test_validate_data_for_path_format_error(session, file_ops, tmp_path):
    """Test validate_data_for_path handles format errors."""
    test_file = tmp_path / "test.hdf5"
    test_file.write_text("invalid data")

    # Mock get_file_length to raise ValueError
    with patch.object(file_ops, "get_file_length", side_effect=ValueError("Bad format")):
        with pytest.raises(ValueError, match="Invalid data format"):
            await file_ops.validate_data_for_path(test_file, None)


# ============================================================================
# create_operations helper tests
# ============================================================================


def test_create_operations_helper():
    """Test create_operations helper function."""
    ops = create_operations(db_class=Algorithm, response_class=MockResponse, create_class=MockCreate)

    assert isinstance(ops, TableOperations)
    assert ops.ctx.db_class == Algorithm
    assert ops.ctx.response_class == MockResponse
    assert ops.ctx.create_class == MockCreate


# ============================================================================
# Integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_then_read(session, table_ops):
    """Test creating and then reading a row."""
    # Create
    async with session.begin():
        created = await table_ops.create_row(session, name="integration_test", class_name="integration.class")

    # Read
    read = await table_ops.get_row(session, created.id_)

    assert read.id_ == created.id_
    assert read.name == "integration_test"


@pytest.mark.asyncio
async def test_create_update_delete_workflow(session, table_ops):
    """Test complete CRUD workflow."""
    # Create
    async with session.begin():
        created = await table_ops.create_row(session, name="workflow_test", class_name="workflow.class")
    row_id = created.id_

    # Update
    updated = await table_ops.update_row(session, row_id=row_id, name="updated_workflow")
    assert updated.name == "updated_workflow"

    # Delete
    await table_ops.delete_row(session, row_id=row_id)

    # Verify deleted
    result = await table_ops.get_row_or_none(session, row_id)
    assert result is None


@pytest.mark.asyncio
async def test_filter_after_create(session, table_ops):
    """Test filtering works after creating rows."""
    # Create rows
    async with session.begin():
        await table_ops.create_rows(
            session,
            [
                {"name": "filter_test_1", "class_name": "class1"},
                {"name": "filter_test_2", "class_name": "class2"},
                {"name": "other_test", "class_name": "class3"},
            ],
        )

    # Filter
    filters = [Filter(field="name", op=FilterOp.STARTS_WITH, value="filter_test")]
    results = await table_ops.filter_rows(session, filters=filters)

    assert len(results) >= 2
    assert all(r.name.startswith("filter_test") for r in results)


@pytest.mark.asyncio
async def test_pydantic_conversion_roundtrip(session, table_ops):
    """Test converting to Pydantic and back."""
    # Create a row
    async with session.begin():
        created = await table_ops.create_row(session, name="pydantic_test", class_name="pydantic.class")

    # Convert to Pydantic
    pydantic_obj = table_ops.to_pydantic(created)

    # Convert to dict
    obj_dict = table_ops.to_pydantic_dict(created)

    # Verify data matches
    assert pydantic_obj.name == created.name
    assert obj_dict["name"] == created.name


@pytest.mark.asyncio
async def test_streaming_large_dataset(session, table_ops):
    """Test streaming works for larger datasets."""
    # Create many rows
    rows_data = [{"name": f"stream_{i}", "class_name": f"class_{i}"} for i in range(50)]
    async with session.begin():
        await table_ops.create_rows(session, rows_data, validate=False)

    # Stream them back
    count = 0
    async for row in table_ops.get_rows_streaming(session, limit=100):
        if row.name.startswith("stream_"):
            count += 1

    assert count >= 50


# ============================================================================
# Error handling tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_rows_partial_failure_rollback(session, table_ops):
    """Test that create_rows handles partial failures."""
    rows_data = [
        {"name": "valid_1", "class_name": "class_1"},
        {"name": "valid_2"},  # Invalid - missing class_name
    ]

    # Should fail validation
    with pytest.raises(ValidationError):
        async with session.begin():
            await table_ops.create_rows(session, rows_data, validate=True)


@pytest.mark.asyncio
async def test_get_create_kwargs_override(session):
    """Test that get_create_kwargs can be overridden."""

    class CustomOps(TableOperations):
        async def get_create_kwargs(self, session: AsyncSession, **kwargs: Any) -> dict[str, Any]:
            # Add custom processing
            kwargs["name"] = f"custom_{kwargs['name']}"
            return kwargs

    context = TableContext.from_db_class(Algorithm)
    ops = CustomOps(context)

    async with session.begin():
        result = await ops.create_row(session, name="test", class_name="test.class", validate=False)

    assert result.name == "custom_test"


# ============================================================================
# Edge cases
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_with_special_characters(session, table_ops):
    """Test creating row with special characters."""
    async with session.begin():
        result = await table_ops.create_row(
            session, name="test_алгоритм_算法_🚀", class_name="special.class", validate=False
        )

    assert result.name == "test_алгоритм_算法_🚀"


@pytest.mark.asyncio
async def test_create_rows_single_item(session, table_ops):
    """Test create_rows with single item works."""
    rows_data = [{"name": "single", "class_name": "single.class"}]

    async with session.begin():
        results = await table_ops.create_rows(session, rows_data, validate=False)

    assert len(results) == 1
    assert results[0].name == "single"


@pytest.mark.asyncio
async def test_bulk_insert_large_batch(session, table_ops):
    """Test bulk insert with large batch."""
    rows_data = [{"name": f"large_{i}", "class_name": f"class_{i}"} for i in range(100)]

    count = await table_ops.bulk_insert_rows(session, rows_data, validate=False)

    assert count == 100


# ============================================================================
# Transaction tests
# ============================================================================


@pytest.mark.asyncio
async def test_operations_do_not_auto_commit(session, table_ops):
    """Test that operations don't auto-commit."""
    # Create a row (should not auto-commit without explicit transaction)
    async with session.begin():
        created = await table_ops.create_row(
            session, name="no_commit_test", class_name="test.class", validate=False
        )
        row_id = created.id_

        # Rollback the transaction
        await session.rollback()

    # Verify it's not in database
    result = await table_ops.get_row_or_none(session, row_id)
    assert result is None


@pytest.mark.asyncio
async def test_multiple_operations_in_transaction(session, table_ops):
    """Test multiple operations in a single transaction."""
    # Don't use session.begin() - the session fixture already handles transactions
    # Create
    created1 = await table_ops.create_row(session, name="tx_test_1", class_name="test.class", validate=False)

    created2 = await table_ops.create_row(session, name="tx_test_2", class_name="test.class", validate=False)

    # Update
    await table_ops.update_row(session, row_id=created1.id_, name="tx_test_1_updated")

    # Commit the changes
    await session.commit()

    # Verify changes
    result1 = await table_ops.get_row(session, created1.id_)
    result2 = await table_ops.get_row(session, created2.id_)

    assert result1.name == "tx_test_1_updated"
    assert result2.name == "tx_test_2"


# ============================================================================
# FileValidatedOperations edge cases
# ============================================================================


@pytest.mark.asyncio
async def test_process_path_mismatch_warning(session, file_ops, tmp_path):
    """Test _process_path logs warning when provided n_objects doesn't match."""
    test_file = tmp_path / "test.hdf5"
    test_file.write_text("test")

    with patch("rail_svc.db_oper.base.global_config") as mock_config:
        mock_config.storage.archive = str(tmp_path)

        # Provide n_objects that doesn't match file length (1000 from mock)
        result = await file_ops._process_path(
            "test.hdf5", None, validate_file=True, extra_kwargs={"n_objects": 500}
        )

        # Should use file's actual length
        assert result == 1000


@pytest.mark.asyncio
async def test_validate_data_unexpected_error(session, file_ops, tmp_path):
    """Test validate_data_for_path handles unexpected errors."""
    test_file = tmp_path / "test.hdf5"
    test_file.write_text("test")

    # Mock get_file_length to raise unexpected error
    with patch.object(file_ops, "get_file_length", side_effect=RuntimeError("Unexpected")):
        with pytest.raises(ValueError, match="Unexpected error"):
            await file_ops.validate_data_for_path(test_file, None)


# ============================================================================
# Type safety tests
# ============================================================================


def test_table_context_type_parameters():
    """Test that TableContext preserves type parameters."""
    context = TableContext(
        db_class=Algorithm, response_class=MockResponse, create_class=MockCreate, class_string="test"
    )

    # Verify types are correct
    assert context.db_class == Algorithm
    assert context.response_class == MockResponse
    assert context.create_class == MockCreate


def test_table_operations_type_parameters():
    """Test that TableOperations preserves type parameters."""
    context = TableContext(
        db_class=Algorithm, response_class=MockResponse, create_class=MockCreate, class_string="test"
    )

    ops = TableOperations(context)

    # Verify context types are preserved
    assert ops.ctx.db_class == Algorithm
    assert ops.ctx.response_class == MockResponse
    assert ops.ctx.create_class == MockCreate


# ============================================================================
# Concurrency tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_creates(engine, table_ops):
    """Test that concurrent creates work correctly."""
    # Don't reuse the session fixture - create fresh sessions for concurrent operations
    async_session_maker = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def create_algo(name: str):
        # Each concurrent operation gets its own session
        async with async_session_maker() as sess:
            result = await table_ops.create_row(sess, name=name, class_name=f"{name}.class", validate=False)
            await sess.commit()
            return result

    # Create multiple algorithms concurrently
    tasks = [create_algo(f"concurrent_{i}") for i in range(5)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 5
    assert all(r.name.startswith("concurrent_") for r in results)

    # Verify all have unique IDs
    ids = {r.id_ for r in results}
    assert len(ids) == 5


@pytest.mark.asyncio
async def test_concurrent_reads(session, sample_algorithm, table_ops):
    """Test that concurrent reads work correctly."""

    async def read_algo():
        return await table_ops.get_row(session, sample_algorithm.id_)

    # Read same algorithm concurrently
    tasks = [read_algo() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    assert len(results) == 10
    assert all(r.id_ == sample_algorithm.id_ for r in results)


# ============================================================================
# Decorator tests
# ============================================================================


@pytest.mark.asyncio
async def test_forward_to_db_funcs_decorator(session, sample_algorithm):
    """Test forward_to_db_funcs decorator."""
    from rail_svc import db_funcs

    class TestOps:
        def __init__(self):
            self.ctx = TableContext.from_db_class(Algorithm)

        @forward_to_db_funcs(db_funcs.read, "get_row")
        async def get_row(self, session: AsyncSession, *args: Any, **kwargs: Any) -> Any:
            pass  # Will be replaced by decorator

    ops = TestOps()
    result = await ops.get_row(session, sample_algorithm.id_)

    assert result.id_ == sample_algorithm.id_


@pytest.mark.asyncio
async def test_forward_to_db_funcs_streaming_decorator(session, multiple_algorithms):
    """Test forward_to_db_funcs_streaming decorator."""
    from rail_svc import db_funcs

    class TestOps:
        def __init__(self):
            self.ctx = TableContext.from_db_class(Algorithm)

        @forward_to_db_funcs_streaming(db_funcs.read, "get_rows_streaming")
        async def get_rows_streaming(self, session: AsyncSession, *args: Any, **kwargs: Any) -> Any:
            yield  # Will be replaced by decorator

    ops = TestOps()
    results = []
    async for row in ops.get_rows_streaming(session):
        results.append(row)

    assert len(results) == 3


# ============================================================================
# Additional edge cases
# ============================================================================


@pytest.mark.asyncio
async def test_batch_operations_with_validation(session, table_ops):
    """Test batched operations with validation enabled."""
    rows_data = [{"name": f"batch_val_{i}", "class_name": f"class_{i}"} for i in range(25)]

    results = await table_ops.create_rows_batched(session, rows_data, validate=True, batch_size=10)

    assert len(results) == 25
    assert all(r.name.startswith("batch_val_") for r in results)


@pytest.mark.asyncio
async def test_bulk_operations_consistency(session, table_ops):
    """Test that bulk operations maintain consistency."""
    rows_data = [{"name": f"bulk_{i}", "class_name": f"class_{i}"} for i in range(10)]

    # Bulk insert
    count = await table_ops.bulk_insert_rows(session, rows_data, validate=False)
    assert count == 10

    # Verify all created
    total_count = await table_ops.count_rows(session)
    assert total_count >= 10
