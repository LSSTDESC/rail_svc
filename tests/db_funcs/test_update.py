"""Unit tests for database update functions."""

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from rail_svc.db import Algorithm, Band, CatalogTag, Dataset
from rail_svc.db_funcs.read import get_row, get_row_by_name
from rail_svc.db_funcs.update import update_row, update_rows

# ============================================================================
# update_row tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_row_single_field(session, sample_algorithm):
    """Test updating a single field."""
    original_name = sample_algorithm.name

    updated = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="updated_algorithm")

    assert updated.id_ == sample_algorithm.id_
    assert updated.name == "updated_algorithm"
    assert updated.name != original_name
    assert updated.class_name == sample_algorithm.class_name


@pytest.mark.asyncio
async def test_update_row_multiple_fields(session, sample_algorithm):
    """Test updating multiple fields at once."""
    updated = await update_row(
        Algorithm, session, row_id=sample_algorithm.id_, name="new_name", class_name="new.class.Name"
    )

    assert updated.id_ == sample_algorithm.id_
    assert updated.name == "new_name"
    assert updated.class_name == "new.class.Name"


@pytest.mark.asyncio
async def test_update_row_not_found(session):
    """Test updating a row that doesn't exist raises KeyError."""
    with pytest.raises(KeyError, match="Algorithm 99999 not found"):
        await update_row(Algorithm, session, row_id=99999, name="test")


@pytest.mark.asyncio
async def test_update_row_no_changes(session, sample_algorithm):
    """Test updating with no fields still returns the row."""
    updated = await update_row(Algorithm, session, row_id=sample_algorithm.id_)

    assert updated.id_ == sample_algorithm.id_
    assert updated.name == sample_algorithm.name
    assert updated.class_name == sample_algorithm.class_name


@pytest.mark.asyncio
async def test_update_row_prevent_id_change(session, sample_algorithm):
    """Test that attempting to change ID raises ValueError."""
    with pytest.raises(ValueError, match="Cannot change row ID"):
        await update_row(Algorithm, session, row_id=sample_algorithm.id_, id=99999, name="test")


@pytest.mark.asyncio
async def test_update_row_same_id_allowed(session, sample_algorithm):
    """Test that providing same ID in kwargs is allowed."""
    updated = await update_row(
        Algorithm,
        session,
        row_id=sample_algorithm.id_,
        id=sample_algorithm.id_,  # Same as row_id - should be filtered out
        name="updated_name",
    )

    assert updated.id_ == sample_algorithm.id_
    assert updated.name == "updated_name"


@pytest.mark.asyncio
async def test_update_row_different_tables(session, sample_band, sample_catalog_tag):
    """Test updating rows in different tables."""
    # Update Band
    updated_band = await update_row(Band, session, row_id=sample_band.id_, name="updated_band")
    assert updated_band.name == "updated_band"

    # Update CatalogTag
    updated_tag = await update_row(CatalogTag, session, row_id=sample_catalog_tag.id_, name="updated_tag")
    assert updated_tag.name == "updated_tag"


@pytest.mark.asyncio
async def test_update_row_persists(session, sample_algorithm):
    """Test that updates are persisted in the database."""
    await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="persisted_name")

    # Fetch fresh from database
    fresh = await session.get(Algorithm, sample_algorithm.id_)
    assert fresh.name == "persisted_name"


@pytest.mark.asyncio
async def test_update_row_complex_data(session, sample_band):
    """Test updating complex data types like lists."""
    new_wavelengths = [100.0, 200.0, 300.0, 400.0]
    new_transmission = [0.1, 0.5, 0.9, 0.3]

    updated = await update_row(
        Band,
        session,
        row_id=sample_band.id_,
        band_wavelengths=new_wavelengths,
        band_transmission=new_transmission,
    )

    assert updated.band_wavelengths == new_wavelengths
    assert updated.band_transmission == new_transmission


# ============================================================================
# update_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_rows_multiple(session, multiple_algorithms):
    """Test updating multiple rows at once."""
    updates = [
        {"id": multiple_algorithms[0].id_, "name": "updated_knn"},
        {"id": multiple_algorithms[1].id_, "name": "updated_rf"},
        {"id": multiple_algorithms[2].id_, "name": "updated_xgb"},
    ]

    updated = await update_rows(Algorithm, session, updates)

    assert len(updated) == 3
    assert updated[0].name == "updated_knn"
    assert updated[1].name == "updated_rf"
    assert updated[2].name == "updated_xgb"


@pytest.mark.asyncio
async def test_update_rows_different_fields(session, multiple_algorithms):
    """Test updating different fields for different rows."""
    updates = [
        {"id": multiple_algorithms[0].id_, "name": "new_name_1"},
        {"id": multiple_algorithms[1].id_, "class_name": "new.class.2"},
        {"id": multiple_algorithms[2].id_, "name": "new_name_3", "class_name": "new.class.3"},
    ]

    updated = await update_rows(Algorithm, session, updates)

    assert len(updated) == 3
    assert updated[0].name == "new_name_1"
    assert updated[1].class_name == "new.class.2"
    assert updated[2].name == "new_name_3"
    assert updated[2].class_name == "new.class.3"


@pytest.mark.asyncio
async def test_update_rows_empty_list(session):
    """Test that empty updates list raises ValueError."""
    with pytest.raises(ValueError, match="updates cannot be empty"):
        await update_rows(Algorithm, session, [])


@pytest.mark.asyncio
async def test_update_rows_missing_id(session, sample_algorithm):
    """Test that update dict without 'id' raises ValueError."""
    updates = [{"name": "test"}]  # Missing 'id'

    with pytest.raises(ValueError, match="Each update must contain 'id' key"):
        await update_rows(Algorithm, session, updates)


@pytest.mark.asyncio
async def test_update_rows_not_found(session, sample_algorithm):
    """Test that non-existent row ID raises KeyError."""
    updates = [
        {"id": sample_algorithm.id_, "name": "valid"},
        {"id": 99999, "name": "invalid"},  # Doesn't exist
    ]

    with pytest.raises(KeyError, match="Algorithm 99999 not found"):
        await update_rows(Algorithm, session, updates)


@pytest.mark.asyncio
async def test_update_rows_atomic(session, multiple_algorithms):
    """Test that updates are atomic - all succeed or all fail."""
    # Store original names before any updates
    original_names = {
        multiple_algorithms[0].id_: multiple_algorithms[0].name,
        multiple_algorithms[1].id_: multiple_algorithms[1].name,
        multiple_algorithms[2].id_: multiple_algorithms[2].name,
    }

    updates = [
        {"id": multiple_algorithms[0].id_, "name": "updated_1"},
        {"id": multiple_algorithms[1].id_, "name": "updated_2"},
        {"id": 99999, "name": "invalid"},  # This will fail
    ]

    try:
        await update_rows(Algorithm, session, updates)
        pytest.fail("Should have raised KeyError")
    except KeyError:
        pass

    # After rollback, we need to expunge objects and re-query
    # to avoid greenlet issues
    session.expunge_all()

    # Re-fetch the algorithms with fresh queries
    for algo_id, original_name in original_names.items():
        result = await session.execute(select(Algorithm).where(Algorithm.id_ == algo_id))
        fresh_algo = result.scalar_one()
        assert (
            fresh_algo.name == original_name
        ), f"Expected {original_name}, got {fresh_algo.name} for id {algo_id}"


@pytest.mark.asyncio
async def test_update_rows_persists(session, multiple_algorithms):
    """Test that bulk updates persist in the database."""
    updates = [
        {"id": multiple_algorithms[0].id_, "name": "persisted_1"},
        {"id": multiple_algorithms[1].id_, "name": "persisted_2"},
    ]

    await update_rows(Algorithm, session, updates)

    # Fetch fresh from database
    fresh_0 = await session.get(Algorithm, multiple_algorithms[0].id_)
    fresh_1 = await session.get(Algorithm, multiple_algorithms[1].id_)

    assert fresh_0.name == "persisted_1"
    assert fresh_1.name == "persisted_2"


@pytest.mark.asyncio
async def test_update_rows_single(session, sample_algorithm):
    """Test updating single row using update_rows."""
    updates = [{"id": sample_algorithm.id_, "name": "single_update"}]

    updated = await update_rows(Algorithm, session, updates)

    assert len(updated) == 1
    assert updated[0].name == "single_update"


@pytest.mark.asyncio
async def test_update_rows_different_tables(session, sample_band, sample_catalog_tag):
    """Test that update_rows works with different table types."""
    # Update Bands
    band_updates = [{"id": sample_band.id_, "name": "updated_band"}]
    updated_bands = await update_rows(Band, session, band_updates)
    assert updated_bands[0].name == "updated_band"

    # Update CatalogTags
    tag_updates = [{"id": sample_catalog_tag.id_, "name": "updated_tag"}]
    updated_tags = await update_rows(CatalogTag, session, tag_updates)
    assert updated_tags[0].name == "updated_tag"


@pytest.mark.asyncio
async def test_update_rows_complex_data(session, multiple_bands):
    """Test bulk updating complex data types."""
    updates = [
        {"id": multiple_bands[0].id_, "band_wavelengths": [100.0, 200.0], "band_transmission": [0.2, 0.8]},
        {"id": multiple_bands[1].id_, "band_wavelengths": [300.0, 400.0], "band_transmission": [0.3, 0.7]},
    ]

    updated = await update_rows(Band, session, updates)

    assert updated[0].band_wavelengths == [100.0, 200.0]
    assert updated[0].band_transmission == [0.2, 0.8]
    assert updated[1].band_wavelengths == [300.0, 400.0]
    assert updated[1].band_transmission == [0.3, 0.7]


@pytest.mark.asyncio
async def test_update_rows_preserves_other_fields(session, multiple_algorithms):
    """Test that updating one field doesn't affect others."""
    original_class_names = [algo.class_name for algo in multiple_algorithms]

    updates = [
        {"id": multiple_algorithms[0].id_, "name": "new_name_0"},
        {"id": multiple_algorithms[1].id_, "name": "new_name_1"},
    ]

    updated = await update_rows(Algorithm, session, updates)

    # Names changed
    assert updated[0].name == "new_name_0"
    assert updated[1].name == "new_name_1"

    # Class names unchanged
    assert updated[0].class_name == original_class_names[0]
    assert updated[1].class_name == original_class_names[1]


@pytest.mark.asyncio
async def test_update_rows_order_preserved(session, multiple_algorithms):
    """Test that returned rows are in same order as input."""
    updates = [
        {"id": multiple_algorithms[2].id_, "name": "third"},
        {"id": multiple_algorithms[0].id_, "name": "first"},
        {"id": multiple_algorithms[1].id_, "name": "second"},
    ]

    updated = await update_rows(Algorithm, session, updates)

    assert len(updated) == 3
    assert updated[0].id_ == multiple_algorithms[2].id_
    assert updated[1].id_ == multiple_algorithms[0].id_
    assert updated[2].id_ == multiple_algorithms[1].id_


# ============================================================================
# Additional edge case tests for update_row
# ============================================================================


@pytest.mark.asyncio
async def test_update_row_not_null_constraint(session, sample_algorithm):
    """Test updating a NOT NULL field to None raises IntegrityError."""
    # class_name has NOT NULL constraint
    with pytest.raises(IntegrityError):
        await update_row(Algorithm, session, row_id=sample_algorithm.id_, class_name=None)


@pytest.mark.asyncio
async def test_update_row_empty_string(session, sample_algorithm):
    """Test updating a field to empty string."""
    updated = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="")

    assert updated.name == ""


@pytest.mark.asyncio
async def test_update_row_very_long_string(session, sample_algorithm):
    """Test updating with very long string."""
    long_name = "a" * 1000

    updated = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name=long_name)

    assert updated.name == long_name
    assert len(updated.name) == 1000


@pytest.mark.asyncio
async def test_update_row_empty_list(session, sample_band):
    """Test updating list field to empty list."""
    updated = await update_row(
        Band, session, row_id=sample_band.id_, band_wavelengths=[], band_transmission=[]
    )

    assert updated.band_wavelengths == []
    assert updated.band_transmission == []


@pytest.mark.asyncio
async def test_update_row_boolean_fields(session, sample_dataset):
    """Test updating boolean fields."""
    original_collection = sample_dataset.is_collection

    updated = await update_row(
        Dataset, session, row_id=sample_dataset.id_, is_collection=not original_collection
    )

    assert updated.is_collection != original_collection


@pytest.mark.asyncio
async def test_update_row_numeric_fields(session, sample_dataset):
    """Test updating numeric fields."""
    updated = await update_row(Dataset, session, row_id=sample_dataset.id_, n_objects=99999)

    assert updated.n_objects == 99999


@pytest.mark.asyncio
async def test_update_row_zero_values(session, sample_dataset):
    """Test updating numeric field to zero."""
    updated = await update_row(Dataset, session, row_id=sample_dataset.id_, n_objects=0)

    assert updated.n_objects == 0


@pytest.mark.asyncio
async def test_update_row_negative_values(session, sample_dataset):
    """Test updating with negative numbers (if allowed)."""
    updated = await update_row(Dataset, session, row_id=sample_dataset.id_, n_objects=-1)

    assert updated.n_objects == -1


@pytest.mark.asyncio
async def test_update_row_float_values(session, sample_band):
    """Test updating with float values."""
    new_wavelengths = [123.456, 789.012, 345.678]

    updated = await update_row(Band, session, row_id=sample_band.id_, band_wavelengths=new_wavelengths)

    assert updated.band_wavelengths == new_wavelengths


# ============================================================================
# Additional edge case tests for update_rows
# ============================================================================


@pytest.mark.asyncio
async def test_update_rows_duplicate_ids(session, sample_algorithm):
    """Test updating same row multiple times in one call."""
    updates = [
        {"id": sample_algorithm.id_, "name": "first_update"},
        {"id": sample_algorithm.id_, "name": "second_update"},
    ]

    updated = await update_rows(Algorithm, session, updates)

    # Last update should win
    assert len(updated) == 2
    assert updated[1].name == "second_update"

    # Verify in database
    fresh = await session.get(Algorithm, sample_algorithm.id_)
    assert fresh.name == "second_update"


@pytest.mark.asyncio
async def test_update_rows_mixed_operations(session, multiple_algorithms):
    """Test mixing different update operations in one call."""
    updates = [
        {"id": multiple_algorithms[0].id_, "name": "only_name"},
        {"id": multiple_algorithms[1].id_, "class_name": "only_class"},
        {"id": multiple_algorithms[2].id_, "name": "both_name", "class_name": "both_class"},
    ]

    updated = await update_rows(Algorithm, session, updates)

    assert updated[0].name == "only_name"
    assert updated[1].class_name == "only_class"
    assert updated[2].name == "both_name"
    assert updated[2].class_name == "both_class"


@pytest.mark.asyncio
async def test_update_rows_large_batch(session):
    """Test updating a large batch of rows."""
    # Create test data
    algos = []
    for i in range(50):
        algo = Algorithm(name=f"batch_algo_{i}", class_name=f"batch.class.{i}")
        session.add(algo)
        algos.append(algo)
    await session.commit()

    # Refresh to get IDs
    for algo in algos:
        await session.refresh(algo)

    # Update all of them
    updates = [{"id": algo.id_, "name": f"updated_{algo.name}"} for algo in algos]

    updated = await update_rows(Algorithm, session, updates)

    assert len(updated) == 50
    for row in updated:
        assert row.name.startswith("updated_")


@pytest.mark.asyncio
async def test_update_rows_with_null_values_fails(session, multiple_algorithms):
    """Test bulk updating NOT NULL fields to NULL raises IntegrityError."""
    updates = [
        {"id": multiple_algorithms[0].id_, "class_name": None},
        {"id": multiple_algorithms[1].id_, "class_name": None},
    ]

    with pytest.raises(IntegrityError):
        await update_rows(Algorithm, session, updates)


# ============================================================================
# Integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_update_then_read(session, sample_algorithm):
    """Test that updates are immediately visible in subsequent reads."""

    # Update
    await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="integration_test")

    # Read back
    result = await get_row(Algorithm, session, sample_algorithm.id_)
    assert result.name == "integration_test"


@pytest.mark.asyncio
async def test_update_then_read_by_name(session, sample_algorithm):
    """Test reading by name after update."""

    new_name = "unique_integration_name"

    await update_row(Algorithm, session, row_id=sample_algorithm.id_, name=new_name)

    result = await get_row_by_name(Algorithm, session, new_name)
    assert result.id_ == sample_algorithm.id_


@pytest.mark.asyncio
async def test_multiple_sequential_updates(session, sample_algorithm):
    """Test multiple sequential updates to same row."""
    # First update
    updated1 = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="first")
    assert updated1.name == "first"

    # Second update
    updated2 = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="second")
    assert updated2.name == "second"

    # Third update
    updated3 = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="third")
    assert updated3.name == "third"

    # Verify final state
    fresh = await session.get(Algorithm, sample_algorithm.id_)
    assert fresh.name == "third"


@pytest.mark.asyncio
async def test_update_rows_then_update_row(session, multiple_algorithms):
    """Test bulk update followed by single update."""
    # Bulk update
    bulk_updates = [{"id": algo.id_, "name": f"bulk_{i}"} for i, algo in enumerate(multiple_algorithms)]
    await update_rows(Algorithm, session, bulk_updates)

    # Single update
    await update_row(Algorithm, session, row_id=multiple_algorithms[0].id_, name="single_override")

    # Verify
    fresh = await session.get(Algorithm, multiple_algorithms[0].id_)
    assert fresh.name == "single_override"


@pytest.mark.asyncio
async def test_update_foreign_key_relationship(
    session, sample_dataset, sample_catalog_tag, multiple_catalog_tags
):
    """Test updating foreign key relationships."""
    original_tag_id = sample_dataset.catalog_tag_id
    new_tag_id = multiple_catalog_tags[0].id_

    updated = await update_row(Dataset, session, row_id=sample_dataset.id_, catalog_tag_id=new_tag_id)

    assert updated.catalog_tag_id == new_tag_id
    assert updated.catalog_tag_id != original_tag_id


@pytest.mark.asyncio
async def test_update_preserves_relationships(session, sample_dataset):
    """Test that updating doesn't break existing relationships."""
    original_catalog_tag_id = sample_dataset.catalog_tag_id

    # Update unrelated field
    await update_row(Dataset, session, row_id=sample_dataset.id_, name="updated_name")

    # Verify relationship preserved
    fresh = await session.get(Dataset, sample_dataset.id_)
    assert fresh.catalog_tag_id == original_catalog_tag_id


@pytest.mark.asyncio
async def test_concurrent_safe_update(session, sample_algorithm):
    """Test that updates handle concurrent access safely."""
    # Simulate reading before update completes

    original = await get_row(Algorithm, session, sample_algorithm.id_)
    original_name = original.name

    # Update
    await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="concurrent_update")

    # Read again
    updated = await get_row(Algorithm, session, sample_algorithm.id_)

    assert original_name != updated.name
    assert updated.name == "concurrent_update"


@pytest.mark.asyncio
async def test_update_row_idempotent(session, sample_algorithm):
    """Test that updating to same value is idempotent."""
    # Update to new value
    first_update = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="idempotent_test")

    # Update to same value again
    second_update = await update_row(Algorithm, session, row_id=sample_algorithm.id_, name="idempotent_test")

    assert first_update.name == second_update.name
    assert first_update.id_ == second_update.id_


@pytest.mark.asyncio
async def test_update_rows_returns_fresh_data(session, multiple_algorithms):
    """Test that update_rows returns refreshed data from database."""
    updates = [{"id": algo.id_, "name": f"fresh_{i}"} for i, algo in enumerate(multiple_algorithms)]

    updated = await update_rows(Algorithm, session, updates)

    # Returned objects should have fresh data
    for i, row in enumerate(updated):
        assert row.name == f"fresh_{i}"

        # Verify by reading directly from DB
        fresh = await session.get(Algorithm, row.id_)
        assert fresh.name == row.name


@pytest.mark.asyncio
async def test_update_row_with_same_value(session, sample_algorithm):
    """Test updating a field to its current value."""
    original_name = sample_algorithm.name

    updated = await update_row(
        Algorithm, session, row_id=sample_algorithm.id_, name=original_name  # Same value
    )

    assert updated.name == original_name


@pytest.mark.asyncio
async def test_update_rows_partial_updates(session, multiple_algorithms):
    """Test that we can update only some rows in a table."""
    # Only update first two
    updates = [
        {"id": multiple_algorithms[0].id_, "name": "updated_0"},
        {"id": multiple_algorithms[1].id_, "name": "updated_1"},
    ]

    updated = await update_rows(Algorithm, session, updates)

    # Verify updated rows
    assert updated[0].name == "updated_0"
    assert updated[1].name == "updated_1"

    # Verify third row unchanged
    fresh = await session.get(Algorithm, multiple_algorithms[2].id_)
    assert fresh.name == multiple_algorithms[2].name
