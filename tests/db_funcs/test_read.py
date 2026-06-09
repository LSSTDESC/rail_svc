"""Unit tests for database reading functions."""

import pytest

from rail_svc.db import Algorithm, Band, CatalogTag
from rail_svc.db_funcs.read import (
    count_rows,
    get_row,
    get_row_by_name,
    get_row_or_none,
    get_rows,
    get_rows_streaming,
    lookup_by_id_or_name,
)

# ============================================================================
# get_row tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_row_success(session, sample_algorithm):
    """Test getting a row by ID successfully."""
    result = await get_row(Algorithm, session, sample_algorithm.id_)
    assert result.id_ == sample_algorithm.id_
    assert result.name == sample_algorithm.name
    assert result.class_name == sample_algorithm.class_name


@pytest.mark.asyncio
async def test_get_row_not_found(session):
    """Test getting a row that doesn't exist raises KeyError."""
    with pytest.raises(KeyError, match="Algorithm 99999 not found"):
        await get_row(Algorithm, session, 99999)


@pytest.mark.asyncio
async def test_get_row_multiple_tables(session, sample_band, sample_catalog_tag):
    """Test getting rows from different tables."""
    band_result = await get_row(Band, session, sample_band.id_)
    assert band_result.name == sample_band.name

    tag_result = await get_row(CatalogTag, session, sample_catalog_tag.id_)
    assert tag_result.name == sample_catalog_tag.name


# ============================================================================
# get_row_by_name tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_row_by_name_success(session, sample_algorithm):
    """Test getting a row by name successfully."""
    result = await get_row_by_name(Algorithm, session, "test_algorithm")
    assert result.id_ == sample_algorithm.id_
    assert result.name == "test_algorithm"
    assert result.class_name == sample_algorithm.class_name


@pytest.mark.asyncio
async def test_get_row_by_name_not_found(session):
    """Test getting a row by name that doesn't exist raises KeyError."""
    with pytest.raises(KeyError, match="Algorithm 'nonexistent' not found"):
        await get_row_by_name(Algorithm, session, "nonexistent")


@pytest.mark.asyncio
async def test_get_row_by_name_multiple_records(session, multiple_algorithms):
    """Test getting specific row by name when multiple exist."""
    result = await get_row_by_name(Algorithm, session, "knn")
    assert result.name == "knn"
    assert result.class_name == "sklearn.neighbors.KNeighborsClassifier"

    result = await get_row_by_name(Algorithm, session, "xgboost")
    assert result.name == "xgboost"
    assert result.class_name == "xgboost.XGBClassifier"


# ============================================================================
# get_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_rows_all(session, multiple_algorithms):
    """Test getting all rows without pagination."""
    results = await get_rows(Algorithm, session, skip=0, limit=100)
    assert len(results) == 3
    names = {r.name for r in results}
    assert names == {"knn", "random_forest", "xgboost"}


@pytest.mark.asyncio
async def test_get_rows_with_skip(session, multiple_algorithms):
    """Test getting rows with skip offset."""
    results = await get_rows(Algorithm, session, skip=1, limit=100)
    assert len(results) == 2

    results = await get_rows(Algorithm, session, skip=2, limit=100)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_get_rows_with_limit(session, multiple_algorithms):
    """Test getting rows with limit."""
    results = await get_rows(Algorithm, session, skip=0, limit=2)
    assert len(results) == 2

    results = await get_rows(Algorithm, session, skip=0, limit=1)
    assert len(results) == 1


@pytest.mark.asyncio
async def test_get_rows_pagination(session, multiple_algorithms):
    """Test pagination with skip and limit."""
    # First page
    page1 = await get_rows(Algorithm, session, skip=0, limit=1)
    assert len(page1) == 1

    # Second page
    page2 = await get_rows(Algorithm, session, skip=1, limit=1)
    assert len(page2) == 1

    # Third page
    page3 = await get_rows(Algorithm, session, skip=2, limit=1)
    assert len(page3) == 1

    # Ensure different records
    ids = {page1[0].id_, page2[0].id_, page3[0].id_}
    assert len(ids) == 3


@pytest.mark.asyncio
async def test_get_rows_empty_table(session):
    """Test getting rows from empty table."""
    results = await get_rows(Algorithm, session, skip=0, limit=10)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_rows_default_limit(session, multiple_bands):
    """Test getting rows with default limit (None)."""
    results = await get_rows(Band, session, skip=0, limit=None)
    assert len(results) == 3


# ============================================================================
# get_rows_streaming tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_rows_streaming_all(session, multiple_algorithms):
    """Test streaming all rows."""
    results = []
    async for row in get_rows_streaming(Algorithm, session, skip=0, limit=100):
        results.append(row)

    assert len(results) == 3
    names = {r.name for r in results}
    assert names == {"knn", "random_forest", "xgboost"}


@pytest.mark.asyncio
async def test_get_rows_streaming_with_skip(session, multiple_algorithms):
    """Test streaming rows with skip offset."""
    results = []
    async for row in get_rows_streaming(Algorithm, session, skip=1, limit=100):
        results.append(row)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_get_rows_streaming_with_limit(session, multiple_algorithms):
    """Test streaming rows with limit."""
    results = []
    async for row in get_rows_streaming(Algorithm, session, skip=0, limit=2):
        results.append(row)

    assert len(results) == 2


@pytest.mark.asyncio
async def test_get_rows_streaming_empty(session):
    """Test streaming from empty table."""
    results = []
    async for row in get_rows_streaming(Algorithm, session, skip=0, limit=10):
        results.append(row)

    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_rows_streaming_iteration(session, multiple_bands):
    """Test that streaming yields rows one at a time."""
    count = 0
    async for row in get_rows_streaming(Band, session, skip=0, limit=100):
        assert isinstance(row, Band)
        assert hasattr(row, "name")
        count += 1

    assert count == 3


# ============================================================================
# get_row_or_none tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_row_or_none_found(session, sample_algorithm):
    """Test getting a row that exists returns the row."""
    result = await get_row_or_none(Algorithm, session, sample_algorithm.id_)
    assert result is not None
    assert result.id_ == sample_algorithm.id_
    assert result.name == sample_algorithm.name


@pytest.mark.asyncio
async def test_get_row_or_none_not_found(session):
    """Test getting a row that doesn't exist returns None."""
    result = await get_row_or_none(Algorithm, session, 99999)
    assert result is None


@pytest.mark.asyncio
async def test_get_row_or_none_multiple_tables(session, sample_band):
    """Test get_row_or_none works with different tables."""
    # Exists
    result = await get_row_or_none(Band, session, sample_band.id_)
    assert result is not None
    assert result.name == sample_band.name

    # Doesn't exist
    result = await get_row_or_none(Band, session, 99999)
    assert result is None


# ============================================================================
# count_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_count_rows_empty(session):
    """Test counting rows in empty table."""
    count = await count_rows(Algorithm, session)
    assert count == 0


@pytest.mark.asyncio
async def test_count_rows_single(session, sample_algorithm):
    """Test counting rows with one record."""
    count = await count_rows(Algorithm, session)
    assert count == 1


@pytest.mark.asyncio
async def test_count_rows_multiple(session, multiple_algorithms):
    """Test counting rows with multiple records."""
    count = await count_rows(Algorithm, session)
    assert count == 3


@pytest.mark.asyncio
async def test_count_rows_different_tables(session, multiple_algorithms, multiple_bands):
    """Test counting rows in different tables."""
    algo_count = await count_rows(Algorithm, session)
    assert algo_count == 3

    band_count = await count_rows(Band, session)
    assert band_count == 3


# ============================================================================
# lookup_by_id_or_name tests
# ============================================================================


@pytest.mark.asyncio
async def test_lookup_by_id_without_object(session, sample_algorithm):
    """Test lookup by ID without fetching object."""
    row_id, obj = await lookup_by_id_or_name(
        Algorithm, session, row_id=sample_algorithm.id_, name=None, need_object=False
    )
    assert row_id == sample_algorithm.id_
    assert obj is None


@pytest.mark.asyncio
async def test_lookup_by_id_with_object(session, sample_algorithm):
    """Test lookup by ID with object fetch."""
    row_id, obj = await lookup_by_id_or_name(
        Algorithm, session, row_id=sample_algorithm.id_, name=None, need_object=True
    )
    assert row_id == sample_algorithm.id_
    assert obj is not None
    assert obj.id_ == sample_algorithm.id_
    assert obj.name == sample_algorithm.name


@pytest.mark.asyncio
async def test_lookup_by_name(session, sample_algorithm):
    """Test lookup by name always returns object."""
    row_id, obj = await lookup_by_id_or_name(
        Algorithm, session, row_id=None, name="test_algorithm", need_object=False
    )
    assert row_id == sample_algorithm.id_
    assert obj is not None
    assert obj.id_ == sample_algorithm.id_
    assert obj.name == "test_algorithm"


@pytest.mark.asyncio
async def test_lookup_by_name_with_object(session, sample_algorithm):
    """Test lookup by name with need_object=True."""
    row_id, obj = await lookup_by_id_or_name(
        Algorithm, session, row_id=None, name="test_algorithm", need_object=True
    )
    assert row_id == sample_algorithm.id_
    assert obj is not None
    assert obj.id_ == sample_algorithm.id_
    assert obj.name == "test_algorithm"


@pytest.mark.asyncio
async def test_lookup_neither_id_nor_name(session):
    """Test lookup with neither ID nor name raises ValueError."""
    with pytest.raises(ValueError, match="Either 'id_' or 'name' must be provided"):
        await lookup_by_id_or_name(Algorithm, session, row_id=None, name=None, need_object=False)


@pytest.mark.asyncio
async def test_lookup_by_name_not_found(session):
    """Test lookup by name that doesn't exist raises ValueError."""
    with pytest.raises(KeyError, match="Algorithm 'nonexistent' not found"):
        await lookup_by_id_or_name(Algorithm, session, row_id=None, name="nonexistent", need_object=False)


@pytest.mark.asyncio
async def test_lookup_by_id_not_found(session):
    """Test lookup by ID that doesn't exist raises ValueError."""
    with pytest.raises(KeyError, match="Algorithm 99999 not found"):
        await lookup_by_id_or_name(Algorithm, session, row_id=99999, name=None, need_object=True)


@pytest.mark.asyncio
async def test_lookup_different_tables(session, sample_band, sample_catalog_tag):
    """Test lookup works with different table types."""
    # Band by ID
    band_id, band_obj = await lookup_by_id_or_name(
        Band, session, row_id=sample_band.id_, name=None, need_object=True
    )
    assert band_id == sample_band.id_
    assert band_obj.name == sample_band.name

    # CatalogTag by name
    tag_id, tag_obj = await lookup_by_id_or_name(
        CatalogTag, session, row_id=None, name="lsst_dp02", need_object=False
    )
    assert tag_id == sample_catalog_tag.id_
    assert tag_obj.name == "lsst_dp02"


@pytest.mark.asyncio
async def test_lookup_priority_when_both_provided(session, sample_algorithm):
    """Test that ID takes priority when both ID and name are provided."""
    # Provide both - should use ID
    row_id, obj = await lookup_by_id_or_name(
        Algorithm,
        session,
        row_id=sample_algorithm.id_,
        name="ignored_name",  # This should be ignored
        need_object=False,
    )
    assert row_id == sample_algorithm.id_
    assert obj is None  # Didn't fetch because need_object=False
