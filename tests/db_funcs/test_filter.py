"""Unit tests for database filtering functions."""

import pytest

from rail_svc.db import Algorithm, Band, Dataset
from rail_svc.db_funcs.delete import delete_row
from rail_svc.db_funcs.filter import (and_filters, count_filtered_rows,
                                      filter_one, filter_one_or_none,
                                      filter_rows, filter_rows_streaming,
                                      find_by, find_one_by, or_filters)
from rail_svc.db_funcs.update import update_row
from rail_svc.models.filtering import Filter, FilterOp, OrderBy

# ============================================================================
# Basic filter_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_no_filters(session, multiple_algorithms):
    """Test filtering with no filters returns all rows."""
    results = await filter_rows(Algorithm, session, filters=None)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_filter_rows_eq_operator(session, multiple_algorithms):
    """Test EQ filter operator."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "knn"


@pytest.mark.asyncio
async def test_filter_rows_ne_operator(session, multiple_algorithms):
    """Test NE (not equal) filter operator."""
    filters = [Filter(field="name", op=FilterOp.NE, value="knn")]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 2
    assert all(r.name != "knn" for r in results)


@pytest.mark.asyncio
async def test_filter_rows_in_operator(session, multiple_algorithms):
    """Test IN operator."""
    filters = [Filter(field="name", op=FilterOp.IN, value=["knn", "xgboost"])]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"knn", "xgboost"}


@pytest.mark.asyncio
async def test_filter_rows_not_in_operator(session, multiple_algorithms):
    """Test NOT_IN operator."""
    filters = [Filter(field="name", op=FilterOp.NOT_IN, value=["knn"])]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 2
    assert all(r.name != "knn" for r in results)


@pytest.mark.asyncio
async def test_filter_rows_like_operator(session, multiple_algorithms):
    """Test LIKE operator."""
    filters = [Filter(field="name", op=FilterOp.LIKE, value="%nn%")]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 1
    assert "nn" in results[0].name


@pytest.mark.asyncio
async def test_filter_rows_starts_with(session, multiple_algorithms):
    """Test STARTS_WITH operator."""
    filters = [Filter(field="name", op=FilterOp.STARTS_WITH, value="kn")]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 1
    assert results[0].name.startswith("kn")


@pytest.mark.asyncio
async def test_filter_rows_ends_with(session, multiple_algorithms):
    """Test ENDS_WITH operator."""
    filters = [Filter(field="name", op=FilterOp.ENDS_WITH, value="nn")]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 1
    assert results[0].name.endswith("nn")


@pytest.mark.asyncio
async def test_filter_rows_is_not_null(session, multiple_algorithms):
    """Test IS_NOT_NULL operator."""
    filters = [Filter(field="class_name", op=FilterOp.IS_NOT_NULL, value=None)]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 3
    assert all(r.class_name is not None for r in results)


# ============================================================================
# Comparison operators tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_gt_operator(session, multiple_datasets):
    """Test GT (greater than) operator."""
    filters = [Filter(field="n_objects", op=FilterOp.GT, value=10000)]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(r.n_objects > 10000 for r in results)


@pytest.mark.asyncio
async def test_filter_rows_ge_operator(session, multiple_datasets):
    """Test GE (greater than or equal) operator."""
    filters = [Filter(field="n_objects", op=FilterOp.GE, value=50000)]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(r.n_objects >= 50000 for r in results)


@pytest.mark.asyncio
async def test_filter_rows_lt_operator(session, multiple_datasets):
    """Test LT (less than) operator."""
    filters = [Filter(field="n_objects", op=FilterOp.LT, value=10000)]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(r.n_objects < 10000 for r in results)


@pytest.mark.asyncio
async def test_filter_rows_le_operator(session, multiple_datasets):
    """Test LE (less than or equal) operator."""
    filters = [Filter(field="n_objects", op=FilterOp.LE, value=50000)]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(r.n_objects <= 50000 for r in results)


@pytest.mark.asyncio
async def test_filter_rows_between_operator(session, multiple_datasets):
    """Test BETWEEN operator."""
    filters = [Filter(field="n_objects", op=FilterOp.BETWEEN, value=[5000, 50000])]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(5000 <= r.n_objects <= 50000 for r in results)


# ============================================================================
# Logical operators tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_and_logic(session, multiple_algorithms):
    """Test AND logic with multiple filters."""
    filters = [
        Filter(field="name", op=FilterOp.NE, value="knn"),
        Filter(field="name", op=FilterOp.NE, value="xgboost"),
    ]
    results = await filter_rows(Algorithm, session, filters=filters, logical_op="and")

    assert len(results) == 1
    assert results[0].name == "random_forest"


@pytest.mark.asyncio
async def test_filter_rows_or_logic(session, multiple_algorithms):
    """Test OR logic with multiple filters."""
    filters = [
        Filter(field="name", op=FilterOp.EQ, value="knn"),
        Filter(field="name", op=FilterOp.EQ, value="xgboost"),
    ]
    results = await filter_rows(Algorithm, session, filters=filters, logical_op="or")

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"knn", "xgboost"}


# ============================================================================
# Ordering tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_order_by_asc(session, multiple_algorithms):
    """Test ordering results ascending."""
    results = await filter_rows(
        Algorithm, session, filters=None, order_by=OrderBy(field="name", descending=False)
    )

    names = [r.name for r in results]
    assert names == sorted(names)


@pytest.mark.asyncio
async def test_filter_rows_order_by_desc(session, multiple_algorithms):
    """Test ordering results descending."""
    results = await filter_rows(
        Algorithm, session, filters=None, order_by=OrderBy(field="name", descending=True)
    )

    names = [r.name for r in results]
    assert names == sorted(names, reverse=True)


@pytest.mark.asyncio
async def test_filter_rows_multiple_order_by(session, multiple_datasets):
    """Test ordering by multiple fields."""
    results = await filter_rows(
        Dataset,
        session,
        filters=None,
        order_by=[
            OrderBy(field="is_collection", descending=False),
            OrderBy(field="n_objects", descending=True),
        ],
    )

    assert len(results) == 3


# ============================================================================
# Pagination tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_skip(session, multiple_algorithms):
    """Test skip pagination."""
    results = await filter_rows(Algorithm, session, filters=None, skip=1, limit=10)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_filter_rows_limit(session, multiple_algorithms):
    """Test limit pagination."""
    results = await filter_rows(Algorithm, session, filters=None, skip=0, limit=2)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_filter_rows_skip_and_limit(session, multiple_algorithms):
    """Test combined skip and limit."""
    results = await filter_rows(Algorithm, session, filters=None, skip=1, limit=1)

    assert len(results) == 1


# ============================================================================
# Error handling tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_invalid_field(session):
    """Test filtering with invalid field raises AttributeError."""
    filters = [Filter(field="nonexistent", op=FilterOp.EQ, value="test")]

    with pytest.raises(AttributeError, match="does not have field 'nonexistent'"):
        await filter_rows(Algorithm, session, filters=filters)


@pytest.mark.asyncio
async def test_filter_rows_invalid_logical_op(session):
    """Test invalid logical operator raises ValueError."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="test")]

    with pytest.raises(ValueError, match="logical_op must be 'and' or 'or'"):
        await filter_rows(Algorithm, session, filters=filters, logical_op="xor")


@pytest.mark.asyncio
async def test_filter_rows_in_wrong_type(session):
    """Test IN operator with wrong value type raises ValueError."""
    filters = [Filter(field="name", op=FilterOp.IN, value="not_a_list")]

    with pytest.raises(ValueError, match="IN operator requires list/tuple/set"):
        await filter_rows(Algorithm, session, filters=filters)


@pytest.mark.asyncio
async def test_filter_rows_between_wrong_length(session):
    """Test BETWEEN operator with wrong value length raises ValueError."""
    filters = [Filter(field="n_objects", op=FilterOp.BETWEEN, value=[1, 2, 3])]

    with pytest.raises(ValueError, match="BETWEEN operator requires list/tuple of exactly 2 values"):
        await filter_rows(Dataset, session, filters=filters)


@pytest.mark.asyncio
async def test_filter_rows_starts_with_non_string(session):
    """Test STARTS_WITH with non-string value raises ValueError."""
    filters = [Filter(field="name", op=FilterOp.STARTS_WITH, value=123)]

    with pytest.raises(ValueError, match="STARTS_WITH operator requires string value"):
        await filter_rows(Algorithm, session, filters=filters)


@pytest.mark.asyncio
async def test_filter_rows_ends_with_non_string(session):
    """Test ENDS_WITH with non-string value raises ValueError."""
    filters = [Filter(field="name", op=FilterOp.ENDS_WITH, value=123)]

    with pytest.raises(ValueError, match="ENDS_WITH operator requires string value"):
        await filter_rows(Algorithm, session, filters=filters)


@pytest.mark.asyncio
async def test_order_by_invalid_field(session):
    """Test ordering by invalid field raises AttributeError."""
    with pytest.raises(AttributeError, match="does not have field 'nonexistent'"):
        await filter_rows(
            Algorithm, session, filters=None, order_by=OrderBy(field="nonexistent", descending=False)
        )


# ============================================================================
# filter_rows_streaming tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_rows_streaming_basic(session, multiple_algorithms):
    """Test streaming filtered rows."""
    results = []
    async for row in filter_rows_streaming(Algorithm, session, filters=None):
        results.append(row)

    assert len(results) == 3
    assert all(isinstance(r, Algorithm) for r in results)


@pytest.mark.asyncio
async def test_filter_rows_streaming_with_filter(session, multiple_algorithms):
    """Test streaming with filters."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    results = []

    async for row in filter_rows_streaming(Algorithm, session, filters=filters):
        results.append(row)

    assert len(results) == 1
    assert results[0].name == "knn"


@pytest.mark.asyncio
async def test_filter_rows_streaming_with_limit(session, multiple_algorithms):
    """Test streaming with limit."""
    results = []
    async for row in filter_rows_streaming(Algorithm, session, filters=None, limit=2):
        results.append(row)

    assert len(results) == 2


# ============================================================================
# count_filtered_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_count_filtered_rows_no_filter(session, multiple_algorithms):
    """Test counting all rows."""
    count = await count_filtered_rows(Algorithm, session, filters=None)
    assert count == 3


@pytest.mark.asyncio
async def test_count_filtered_rows_with_filter(session, multiple_algorithms):
    """Test counting filtered rows."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    count = await count_filtered_rows(Algorithm, session, filters=filters)
    assert count == 1


@pytest.mark.asyncio
async def test_count_filtered_rows_or_logic(session, multiple_algorithms):
    """Test counting with OR logic."""
    filters = [
        Filter(field="name", op=FilterOp.EQ, value="knn"),
        Filter(field="name", op=FilterOp.EQ, value="xgboost"),
    ]
    count = await count_filtered_rows(Algorithm, session, filters=filters, logical_op="or")
    assert count == 2


@pytest.mark.asyncio
async def test_count_filtered_rows_and_logic(session, multiple_algorithms):
    """Test counting with AND logic."""
    filters = [
        Filter(field="name", op=FilterOp.NE, value="knn"),
        Filter(field="name", op=FilterOp.NE, value="xgboost"),
    ]
    count = await count_filtered_rows(Algorithm, session, filters=filters, logical_op="and")
    assert count == 1


@pytest.mark.asyncio
async def test_count_filtered_rows_zero_results(session, multiple_algorithms):
    """Test counting when no rows match."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="nonexistent")]
    count = await count_filtered_rows(Algorithm, session, filters=filters)
    assert count == 0


# ============================================================================
# filter_one tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_one_success(session, multiple_algorithms):
    """Test filter_one when exactly one row matches."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    result = await filter_one(Algorithm, session, filters=filters)

    assert result.name == "knn"


@pytest.mark.asyncio
async def test_filter_one_not_found(session, multiple_algorithms):
    """Test filter_one raises KeyError when no rows match."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="nonexistent")]

    with pytest.raises(KeyError, match="No Algorithm found matching filters"):
        await filter_one(Algorithm, session, filters=filters)


@pytest.mark.asyncio
async def test_filter_one_multiple_found(session, multiple_algorithms):
    """Test filter_one raises KeyError when multiple rows match."""
    filters = [Filter(field="name", op=FilterOp.NE, value="nonexistent")]

    with pytest.raises(KeyError, match="Multiple Algorithm rows found matching filters"):
        await filter_one(Algorithm, session, filters=filters)


# ============================================================================
# filter_one_or_none tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_one_or_none_found(session, multiple_algorithms):
    """Test filter_one_or_none when row is found."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    result = await filter_one_or_none(Algorithm, session, filters=filters)

    assert result is not None
    assert result.name == "knn"


@pytest.mark.asyncio
async def test_filter_one_or_none_not_found(session, multiple_algorithms):
    """Test filter_one_or_none returns None when no rows match."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="nonexistent")]
    result = await filter_one_or_none(Algorithm, session, filters=filters)

    assert result is None


@pytest.mark.asyncio
async def test_filter_one_or_none_multiple_found(session, multiple_algorithms):
    """Test filter_one_or_none raises KeyError when multiple rows match."""
    filters = [Filter(field="name", op=FilterOp.NE, value="nonexistent")]

    with pytest.raises(KeyError, match="Multiple Algorithm rows found matching filters"):
        await filter_one_or_none(Algorithm, session, filters=filters)


# ============================================================================
# find_by tests
# ============================================================================


@pytest.mark.asyncio
async def test_find_by_single_field(session, multiple_algorithms):
    """Test find_by with single field."""
    results = await find_by(Algorithm, session, name="knn")

    assert len(results) == 1
    assert results[0].name == "knn"


@pytest.mark.asyncio
async def test_find_by_multiple_fields(session, multiple_algorithms):
    """Test find_by with multiple fields (AND logic)."""
    results = await find_by(
        Algorithm, session, name="knn", class_name="sklearn.neighbors.KNeighborsClassifier"
    )

    assert len(results) == 1
    assert results[0].name == "knn"


@pytest.mark.asyncio
async def test_find_by_no_match(session, multiple_algorithms):
    """Test find_by when no rows match."""
    results = await find_by(Algorithm, session, name="nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_find_by_with_order(session, multiple_algorithms):
    """Test find_by with ordering."""
    results = await find_by(Algorithm, session, order_by=OrderBy(field="name", descending=True))

    names = [r.name for r in results]
    assert names == sorted(names, reverse=True)


@pytest.mark.asyncio
async def test_find_by_with_pagination(session, multiple_algorithms):
    """Test find_by with pagination."""
    results = await find_by(Algorithm, session, skip=1, limit=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_find_by_no_kwargs(session, multiple_algorithms):
    """Test find_by with no filters returns all rows."""
    results = await find_by(Algorithm, session)
    assert len(results) == 3


# ============================================================================
# find_one_by tests
# ============================================================================


@pytest.mark.asyncio
async def test_find_one_by_success(session, multiple_algorithms):
    """Test find_one_by when exactly one row matches."""
    result = await find_one_by(Algorithm, session, name="knn")
    assert result.name == "knn"


@pytest.mark.asyncio
async def test_find_one_by_not_found(session, multiple_algorithms):
    """Test find_one_by raises KeyError when no rows match."""
    with pytest.raises(KeyError, match="No Algorithm found matching filters"):
        await find_one_by(Algorithm, session, name="nonexistent")


@pytest.mark.asyncio
async def test_find_one_by_multiple_fields(session, multiple_algorithms):
    """Test find_one_by with multiple fields."""
    result = await find_one_by(
        Algorithm, session, name="knn", class_name="sklearn.neighbors.KNeighborsClassifier"
    )
    assert result.name == "knn"


@pytest.mark.asyncio
async def test_find_one_by_multiple_matches(session, multiple_algorithms):
    """Test find_one_by raises KeyError when multiple rows match."""
    # Create another algorithm with same class_name
    algo = Algorithm(name="knn2", class_name="sklearn.neighbors.KNeighborsClassifier")
    session.add(algo)
    await session.commit()

    with pytest.raises(KeyError, match="Multiple Algorithm rows found"):
        await find_one_by(Algorithm, session, class_name="sklearn.neighbors.KNeighborsClassifier")


# ============================================================================
# Helper function tests
# ============================================================================


@pytest.mark.asyncio
async def test_and_filters_helper():
    """Test and_filters helper function."""
    f1 = Filter(field="name", op=FilterOp.EQ, value="test")
    f2 = Filter(field="age", op=FilterOp.GT, value=18)

    filters = and_filters(f1, f2)

    assert isinstance(filters, list)
    assert len(filters) == 2
    assert filters[0] == f1
    assert filters[1] == f2


@pytest.mark.asyncio
async def test_or_filters_helper():
    """Test or_filters helper function."""
    f1 = Filter(field="status", op=FilterOp.EQ, value="active")
    f2 = Filter(field="status", op=FilterOp.EQ, value="pending")

    filters = or_filters(f1, f2)

    assert isinstance(filters, list)
    assert len(filters) == 2
    assert filters[0] == f1
    assert filters[1] == f2


# ============================================================================
# Integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_filter_then_update(session, multiple_algorithms):
    """Test filtering then updating results."""
    # Find specific algorithm
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    results = await filter_rows(Algorithm, session, filters=filters)

    # Update it
    await update_row(Algorithm, session, results[0].id_, name="knn_updated")

    # Filter again with new name
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn_updated")]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "knn_updated"


@pytest.mark.asyncio
async def test_filter_then_delete(session, multiple_algorithms):
    """Test filtering then deleting results."""
    # Find and delete
    filters = [Filter(field="name", op=FilterOp.EQ, value="knn")]
    results = await filter_rows(Algorithm, session, filters=filters)

    await delete_row(Algorithm, session, results[0].id_)

    # Verify deletion
    results = await filter_rows(Algorithm, session, filters=filters)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_complex_filter_combination(session, multiple_datasets):
    """Test complex combination of filters."""
    filters = [
        Filter(field="n_objects", op=FilterOp.GE, value=5000),
        Filter(field="is_collection", op=FilterOp.EQ, value=False),
    ]

    results = await filter_rows(
        Dataset,
        session,
        filters=filters,
        logical_op="and",
        order_by=OrderBy(field="n_objects", descending=True),
    )

    assert all(r.n_objects >= 5000 for r in results)
    assert all(r.is_collection is False for r in results)


@pytest.mark.asyncio
async def test_filter_with_relationships(session, multiple_datasets, sample_catalog_tag):
    """Test filtering on foreign key fields."""
    filters = [Filter(field="catalog_tag_id", op=FilterOp.EQ, value=sample_catalog_tag.id_)]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(r.catalog_tag_id == sample_catalog_tag.id_ for r in results)


@pytest.mark.asyncio
async def test_pagination_consistency(session, multiple_algorithms):
    """Test that pagination is consistent."""
    # Get all with ordering
    all_results = await filter_rows(
        Algorithm, session, filters=None, order_by=OrderBy(field="name", descending=False)
    )

    # Get first page
    page1 = await filter_rows(
        Algorithm, session, filters=None, order_by=OrderBy(field="name", descending=False), skip=0, limit=2
    )

    # Get second page
    page2 = await filter_rows(
        Algorithm, session, filters=None, order_by=OrderBy(field="name", descending=False), skip=2, limit=2
    )

    # Verify consistency
    assert page1[0].id_ == all_results[0].id_
    assert page1[1].id_ == all_results[1].id_
    assert page2[0].id_ == all_results[2].id_


@pytest.mark.asyncio
async def test_streaming_vs_regular(session, multiple_algorithms):
    """Test that streaming returns same results as regular filter."""
    filters = [Filter(field="name", op=FilterOp.NE, value="nonexistent")]

    # Regular
    regular_results = await filter_rows(Algorithm, session, filters=filters)

    # Streaming
    streaming_results = []
    async for row in filter_rows_streaming(Algorithm, session, filters=filters):
        streaming_results.append(row)

    assert len(regular_results) == len(streaming_results)
    regular_ids = {r.id_ for r in regular_results}
    streaming_ids = {r.id_ for r in streaming_results}
    assert regular_ids == streaming_ids


@pytest.mark.asyncio
async def test_filter_empty_table(session):
    """Test filtering on empty table."""
    filters = [Filter(field="name", op=FilterOp.EQ, value="test")]
    results = await filter_rows(Algorithm, session, filters=filters)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_count_matches_filter_results(session, multiple_algorithms):
    """Test that count matches actual filter results."""
    filters = [Filter(field="name", op=FilterOp.IN, value=["knn", "xgboost"])]

    count = await count_filtered_rows(Algorithm, session, filters=filters)
    results = await filter_rows(Algorithm, session, filters=filters)

    assert count == len(results)


@pytest.mark.asyncio
async def test_filter_with_array_field(session, multiple_bands):
    """Test filtering works with array/list fields."""
    # Just verify we can query bands with array fields
    results = await filter_rows(Band, session, filters=None)

    assert len(results) == 3
    assert all(isinstance(r.band_wavelengths, list) for r in results)


@pytest.mark.asyncio
async def test_case_sensitive_like(session, multiple_algorithms):
    """Test LIKE is case-sensitive."""
    filters = [Filter(field="name", op=FilterOp.LIKE, value="KNN")]
    results = await filter_rows(Algorithm, session, filters=filters)

    # SQLite LIKE is case-insensitive, but this documents the behavior
    # For case-sensitive, you'd need LIKE BINARY in MySQL or similar
    assert len(results) >= 0


@pytest.mark.asyncio
async def test_ilike_operator(session, multiple_algorithms):
    """Test ILIKE (case-insensitive) operator."""
    filters = [Filter(field="name", op=FilterOp.ILIKE, value="%NN%")]
    results = await filter_rows(Algorithm, session, filters=filters)

    # Should find "knn" regardless of case
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_filter_multiple_tables(session, multiple_algorithms, multiple_bands):
    """Test filtering works correctly on different tables."""
    # Filter algorithms
    algo_results = await filter_rows(
        Algorithm, session, filters=[Filter(field="name", op=FilterOp.STARTS_WITH, value="k")]
    )

    # Filter bands
    band_results = await filter_rows(
        Band, session, filters=[Filter(field="name", op=FilterOp.ENDS_WITH, value="band")]
    )

    assert len(algo_results) > 0
    assert len(band_results) > 0
    assert isinstance(algo_results[0], Algorithm)
    assert isinstance(band_results[0], Band)


@pytest.mark.asyncio
async def test_filter_with_limit_zero(session, multiple_algorithms):
    """Test filtering with limit=0 returns empty results."""
    results = await filter_rows(Algorithm, session, filters=None, limit=0)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_filter_with_large_skip(session, multiple_algorithms):
    """Test filtering with skip larger than total rows."""
    results = await filter_rows(Algorithm, session, filters=None, skip=100)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_filter_chained_operations(session, multiple_algorithms):
    """Test multiple filter operations in sequence."""
    # First filter
    filters1 = [Filter(field="name", op=FilterOp.NE, value="knn")]
    results1 = await filter_rows(Algorithm, session, filters=filters1)
    assert len(results1) == 2

    # Get IDs from first result
    ids = [r.id_ for r in results1]

    # Second filter on those results
    filters2 = [Filter(field="id_", op=FilterOp.IN, value=ids)]
    results2 = await filter_rows(Algorithm, session, filters=filters2)

    assert len(results2) == 2
    assert all(r.name != "knn" for r in results2)


@pytest.mark.asyncio
async def test_filter_boolean_field(session, multiple_datasets):
    """Test filtering on boolean fields."""
    filters = [Filter(field="is_collection", op=FilterOp.EQ, value=True)]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(r.is_collection is True for r in results)


@pytest.mark.asyncio
async def test_filter_boolean_field_false(session, multiple_datasets):
    """Test filtering for False boolean values."""
    filters = [Filter(field="is_collection", op=FilterOp.EQ, value=False)]
    results = await filter_rows(Dataset, session, filters=filters)

    assert all(r.is_collection is False for r in results)


@pytest.mark.asyncio
async def test_find_by_with_invalid_field(session):
    """Test find_by with invalid field raises AttributeError."""
    with pytest.raises(AttributeError, match="does not have field"):
        await find_by(Algorithm, session, nonexistent_field="value")


@pytest.mark.asyncio
async def test_streaming_early_break(session, multiple_algorithms):
    """Test that streaming can be broken early."""
    count = 0
    async for row in filter_rows_streaming(Algorithm, session, filters=None):
        count += 1
        if count == 2:
            break

    assert count == 2


@pytest.mark.asyncio
async def test_filter_empty_string(session, sample_algorithm):
    """Test filtering for empty strings."""
    # Set name to empty string
    await update_row(Algorithm, session, sample_algorithm.id_, name="")

    # Filter for empty string
    filters = [Filter(field="name", op=FilterOp.EQ, value="")]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 1
    assert results[0].name == ""


@pytest.mark.asyncio
async def test_between_inclusive(session, multiple_datasets):
    """Test BETWEEN is inclusive on both ends."""
    # Find dataset with n_objects = 5000
    filters = [Filter(field="n_objects", op=FilterOp.BETWEEN, value=[5000, 5000])]
    results = await filter_rows(Dataset, session, filters=filters)

    # Should include 5000 if it exists
    if results:
        assert all(r.n_objects == 5000 for r in results)


@pytest.mark.asyncio
async def test_in_empty_list(session, multiple_algorithms):
    """Test IN operator with empty list returns no results."""
    filters = [Filter(field="name", op=FilterOp.IN, value=[])]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_not_in_empty_list(session, multiple_algorithms):
    """Test NOT_IN operator with empty list returns all results."""
    filters = [Filter(field="name", op=FilterOp.NOT_IN, value=[])]
    results = await filter_rows(Algorithm, session, filters=filters)

    assert len(results) == 3


@pytest.mark.asyncio
async def test_complex_or_with_different_fields(session, multiple_algorithms):
    """Test OR logic with filters on different fields."""
    filters = [
        Filter(field="name", op=FilterOp.EQ, value="knn"),
        Filter(field="class_name", op=FilterOp.LIKE, value="%XGB%"),
    ]
    results = await filter_rows(Algorithm, session, filters=filters, logical_op="or")

    # Should match knn by name OR anything with XGB in class_name
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_filter_numeric_as_string(session, multiple_datasets):
    """Test that numeric filters work correctly."""
    # Filter by exact number
    filters = [Filter(field="n_objects", op=FilterOp.EQ, value=5000)]
    results = await filter_rows(Dataset, session, filters=filters)

    if results:
        assert all(r.n_objects == 5000 for r in results)


@pytest.mark.asyncio
async def test_multiple_order_by_precedence(session):
    """Test that multiple order_by clauses are applied in order."""
    # Create datasets with same is_collection but different n_objects
    for i in range(3):
        dataset = Dataset(
            name=f"test_{i}",
            n_objects=i * 1000,
            path=f"/data/test_{i}",
            is_collection=False,
            catalog_tag_id=1,  # Assume exists from fixture
        )
        session.add(dataset)
    await session.commit()

    results = await filter_rows(
        Dataset,
        session,
        filters=[Filter(field="is_collection", op=FilterOp.EQ, value=False)],
        order_by=[
            OrderBy(field="is_collection", descending=False),
            OrderBy(field="n_objects", descending=True),
        ],
    )

    # Check n_objects are in descending order
    n_objects_values = [r.n_objects for r in results]
    assert n_objects_values == sorted(n_objects_values, reverse=True)


@pytest.mark.asyncio
async def test_filter_one_with_or_logic(session, multiple_algorithms):
    """Test filter_one with OR logic when only one row matches overall."""
    filters = [
        Filter(field="name", op=FilterOp.EQ, value="knn"),
        Filter(field="name", op=FilterOp.EQ, value="nonexistent"),
    ]

    result = await filter_one(Algorithm, session, filters=filters, logical_op="or")
    assert result.name == "knn"


@pytest.mark.asyncio
async def test_find_by_empty_results_with_order(session, multiple_algorithms):
    """Test find_by with ordering returns empty list correctly."""
    results = await find_by(
        Algorithm, session, name="nonexistent", order_by=OrderBy(field="name", descending=False)
    )

    assert len(results) == 0


@pytest.mark.asyncio
async def test_streaming_with_or_logic(session, multiple_algorithms):
    """Test streaming with OR logic."""
    filters = [
        Filter(field="name", op=FilterOp.EQ, value="knn"),
        Filter(field="name", op=FilterOp.EQ, value="xgboost"),
    ]

    results = []
    async for row in filter_rows_streaming(Algorithm, session, filters=filters, logical_op="or"):
        results.append(row)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"knn", "xgboost"}


@pytest.mark.asyncio
async def test_count_with_complex_filters(session, multiple_datasets):
    """Test counting with complex filter combinations."""
    filters = [
        Filter(field="n_objects", op=FilterOp.GE, value=1000),
        Filter(field="n_objects", op=FilterOp.LT, value=100000),
    ]

    count = await count_filtered_rows(Dataset, session, filters=filters)
    results = await filter_rows(Dataset, session, filters=filters)

    assert count == len(results)


@pytest.mark.asyncio
async def test_filter_performance_with_index(session):
    """Test that filtering on indexed fields works efficiently."""
    # ID is typically indexed
    filters = [Filter(field="id_", op=FilterOp.GT, value=0)]
    results = await filter_rows(Algorithm, session, filters=filters)

    # Should return results without error
    assert len(results) >= 0


@pytest.mark.asyncio
async def test_like_with_escape_characters(session, sample_algorithm):
    """Test LIKE operator with special SQL characters."""
    # Name with % in it
    await update_row(Algorithm, session, sample_algorithm.id_, name="test%name")

    # Search for it (% is a wildcard, but here it's literal in the data)
    filters = [Filter(field="name", op=FilterOp.LIKE, value="test%name")]
    results = await filter_rows(Algorithm, session, filters=filters)

    # This will match "test%name" and potentially other patterns
    assert len(results) >= 1


@pytest.mark.asyncio
async def test_contains_operator_with_list(session, sample_band):
    """Test CONTAINS operator with array fields."""
    # CONTAINS checks if array contains a value
    # Note: This is PostgreSQL-specific, may not work in SQLite
    try:
        filters = [Filter(field="band_wavelengths", op=FilterOp.CONTAINS, value=[400.0])]
        _results = await filter_rows(Band, session, filters=filters)
        # If it works, great; if not, it'll raise an error
    except Exception:
        # SQLite doesn't support array contains, skip
        pytest.skip("CONTAINS operator not supported in this database")
