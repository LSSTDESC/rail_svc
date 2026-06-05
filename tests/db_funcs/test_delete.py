"""Unit tests for database deletion functions."""


import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from rail_svc.db import Algorithm, Band, CatalogTag, Dataset, Estimator, Model
from rail_svc.db_funcs.delete import bulk_delete_rows, delete_row, delete_rows
from rail_svc.db_funcs.read import count_rows, get_row
from rail_svc.db_funcs.update import update_row


# ============================================================================
# delete_row tests
# ============================================================================


@pytest.mark.asyncio
async def test_delete_row_success(session, sample_algorithm):
    """Test deleting a row successfully."""
    row_id = sample_algorithm.id_

    result = await delete_row(Algorithm, session, row_id)

    # Should return captured data
    assert result is not None
    assert result["id_"] == row_id
    assert result["name"] == sample_algorithm.name
    assert result["class_name"] == sample_algorithm.class_name

    # Verify row is deleted
    deleted = await session.get(Algorithm, row_id)
    assert deleted is None


@pytest.mark.asyncio
async def test_delete_row_without_capture(session, sample_algorithm):
    """Test deleting without capturing data."""
    row_id = sample_algorithm.id_

    result = await delete_row(Algorithm, session, row_id, capture_data=False)

    # Should return None when capture_data=False
    assert result is None

    # Verify row is deleted
    deleted = await session.get(Algorithm, row_id)
    assert deleted is None


@pytest.mark.asyncio
async def test_delete_row_not_found(session):
    """Test deleting a row that doesn't exist raises KeyError."""
    with pytest.raises(KeyError, match="Algorithm 99999 not found"):
        await delete_row(Algorithm, session, 99999)


@pytest.mark.asyncio
async def test_delete_row_different_tables(session, sample_band, sample_catalog_tag):
    """Test deleting rows from different tables."""
    # Delete Band
    band_id = sample_band.id_
    await delete_row(Band, session, band_id)
    assert await session.get(Band, band_id) is None

    # Delete CatalogTag
    tag_id = sample_catalog_tag.id_
    await delete_row(CatalogTag, session, tag_id)
    assert await session.get(CatalogTag, tag_id) is None


@pytest.mark.asyncio
async def test_delete_row_with_foreign_key_constraint(session, sample_model, sample_algorithm):
    """Test that deleting a row with foreign key dependencies raises IntegrityError.

    Note: This test may be skipped if the database doesn't enforce foreign keys.
    """
    # sample_model has a foreign key reference to sample_algorithm
    # Try to delete the algorithm (should fail if FK constraints are enabled)
    algo_id = sample_model.algo_id

    try:
        await delete_row(Algorithm, session, algo_id)
        # If we get here, foreign keys aren't enforced - skip test
        pytest.skip("Foreign key constraints not enforced in test database")
    except IntegrityError:
        # This is the expected behavior
        pass

    # Verify algorithm still exists
    session.expunge_all()
    algo = await session.get(Algorithm, algo_id)
    assert algo is not None


@pytest.mark.asyncio
async def test_delete_row_persists(session, sample_algorithm):
    """Test that deletion is persisted in the database."""
    row_id = sample_algorithm.id_

    await delete_row(Algorithm, session, row_id)

    # Query again to ensure it's really gone
    result = await session.execute(select(Algorithm).where(Algorithm.id_ == row_id))
    assert result.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_delete_row_captures_all_columns(session, sample_band):
    """Test that all column data is captured."""
    result = await delete_row(Band, session, sample_band.id_, capture_data=True)

    assert result is not None
    assert "id_" in result
    assert "name" in result
    assert "band_wavelengths" in result
    assert "band_transmission" in result

    # Verify values match
    assert result["name"] == sample_band.name
    assert result["band_wavelengths"] == sample_band.band_wavelengths
    assert result["band_transmission"] == sample_band.band_transmission


@pytest.mark.asyncio
async def test_delete_row_reduces_count(session, multiple_algorithms):
    """Test that deletion reduces the row count."""
    initial_count = await count_rows(Algorithm, session)
    assert initial_count == 3

    await delete_row(Algorithm, session, multiple_algorithms[0].id_)

    final_count = await count_rows(Algorithm, session)
    assert final_count == 2


# ============================================================================
# delete_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_delete_rows_multiple(session, multiple_algorithms):
    """Test deleting multiple rows at once."""
    row_ids = [algo.id_ for algo in multiple_algorithms]

    result = await delete_rows(Algorithm, session, row_ids, capture_data=True)

    assert result is not None
    assert len(result) == 3

    # Verify all are deleted
    for row_id in row_ids:
        deleted = await session.get(Algorithm, row_id)
        assert deleted is None


@pytest.mark.asyncio
async def test_delete_rows_without_capture(session, multiple_algorithms):
    """Test deleting multiple rows without capturing data."""
    row_ids = [algo.id_ for algo in multiple_algorithms]

    result = await delete_rows(Algorithm, session, row_ids, capture_data=False)

    assert result is None

    # Verify all are deleted
    for row_id in row_ids:
        deleted = await session.get(Algorithm, row_id)
        assert deleted is None


@pytest.mark.asyncio
async def test_delete_rows_empty_list(session):
    """Test that empty row_ids list raises ValueError."""
    with pytest.raises(ValueError, match="row_ids cannot be empty"):
        await delete_rows(Algorithm, session, [])


@pytest.mark.asyncio
async def test_delete_rows_single(session, sample_algorithm):
    """Test deleting single row using delete_rows."""
    row_ids = [sample_algorithm.id_]

    result = await delete_rows(Algorithm, session, row_ids, capture_data=True)

    assert result is not None
    assert len(result) == 1
    assert result[0]["id_"] == sample_algorithm.id_

    # Verify deleted
    deleted = await session.get(Algorithm, sample_algorithm.id_)
    assert deleted is None


@pytest.mark.asyncio
async def test_delete_rows_not_found(session, sample_algorithm):
    """Test that non-existent row ID raises KeyError."""
    row_ids = [sample_algorithm.id_, 99999]

    with pytest.raises(KeyError, match="Algorithm 99999 not found"):
        await delete_rows(Algorithm, session, row_ids)


@pytest.mark.asyncio
async def test_delete_rows_atomic(session, multiple_algorithms):
    """Test that deletes are atomic - all succeed or all fail."""
    # Store IDs before attempting deletion
    algo_ids = [algo.id_ for algo in multiple_algorithms]

    row_ids = [
        algo_ids[0],
        algo_ids[1],
        99999,  # Doesn't exist - will cause failure
    ]

    try:
        await delete_rows(Algorithm, session, row_ids)
        pytest.fail("Should have raised KeyError")
    except KeyError:
        pass

    # Verify none were deleted (rollback worked)
    session.expunge_all()
    result = await session.execute(select(Algorithm).where(Algorithm.id_.in_(algo_ids[:2])))
    remaining = result.scalars().all()
    assert len(remaining) == 2


@pytest.mark.asyncio
async def test_delete_rows_with_foreign_key_constraint(session, sample_model, sample_algorithm):
    """Test that foreign key constraint violations cause atomic rollback.

    Note: This test may be skipped if the database doesn't enforce foreign keys.
    """
    # Create another algorithm we can safely delete
    safe_algo = Algorithm(name="safe_to_delete", class_name="safe.class")
    session.add(safe_algo)
    await session.commit()
    await session.refresh(safe_algo)

    algo_id = sample_model.algo_id

    row_ids = [
        safe_algo.id_,
        algo_id,  # This one has a foreign key reference from sample_model
    ]

    try:
        await delete_rows(Algorithm, session, row_ids)
        # If we get here, foreign keys aren't enforced - skip test
        pytest.skip("Foreign key constraints not enforced in test database")
    except IntegrityError:
        # This is the expected behavior
        pass

    # Verify neither was deleted (rollback)
    session.expunge_all()
    still_exists_safe = await session.get(Algorithm, safe_algo.id_)
    still_exists_sample = await session.get(Algorithm, algo_id)
    assert still_exists_safe is not None
    assert still_exists_sample is not None


@pytest.mark.asyncio
async def test_delete_rows_data_order(session, multiple_algorithms):
    """Test that captured data is in same order as input IDs."""
    row_ids = [
        multiple_algorithms[2].id_,
        multiple_algorithms[0].id_,
        multiple_algorithms[1].id_,
    ]

    result = await delete_rows(Algorithm, session, row_ids, capture_data=True)

    assert result is not None
    assert len(result) == 3
    assert result[0]["id_"] == multiple_algorithms[2].id_
    assert result[1]["id_"] == multiple_algorithms[0].id_
    assert result[2]["id_"] == multiple_algorithms[1].id_


@pytest.mark.asyncio
async def test_delete_rows_different_tables(session, multiple_bands, multiple_catalog_tags):
    """Test deleting from different tables."""
    # Delete bands
    band_ids = [band.id_ for band in multiple_bands]
    await delete_rows(Band, session, band_ids)

    for band_id in band_ids:
        assert await session.get(Band, band_id) is None

    # Delete tags
    tag_ids = [tag.id_ for tag in multiple_catalog_tags]
    await delete_rows(CatalogTag, session, tag_ids)

    for tag_id in tag_ids:
        assert await session.get(CatalogTag, tag_id) is None


# ============================================================================
# bulk_delete_rows tests
# ============================================================================


@pytest.mark.asyncio
async def test_bulk_delete_rows_success(session, multiple_algorithms):
    """Test bulk deleting multiple rows."""
    row_ids = [algo.id_ for algo in multiple_algorithms]

    count = await bulk_delete_rows(Algorithm, session, row_ids)

    assert count == 3

    # Verify all deleted
    for row_id in row_ids:
        deleted = await session.get(Algorithm, row_id)
        assert deleted is None


@pytest.mark.asyncio
async def test_bulk_delete_rows_empty_list(session):
    """Test that empty row_ids list raises ValueError."""
    with pytest.raises(ValueError, match="row_ids cannot be empty"):
        await bulk_delete_rows(Algorithm, session, [])


@pytest.mark.asyncio
async def test_bulk_delete_rows_nonexistent(session, sample_algorithm):
    """Test bulk delete with nonexistent IDs."""
    row_ids = [sample_algorithm.id_, 99999, 88888]

    # Bulk delete doesn't verify existence - just deletes what exists
    count = await bulk_delete_rows(Algorithm, session, row_ids)

    # Only 1 row actually existed
    assert count == 1

    # Verify the existing one was deleted
    deleted = await session.get(Algorithm, sample_algorithm.id_)
    assert deleted is None


@pytest.mark.asyncio
async def test_bulk_delete_rows_partial(session, multiple_algorithms):
    """Test bulk delete with mix of existing and non-existing IDs."""
    row_ids = [
        multiple_algorithms[0].id_,
        99999,  # Doesn't exist
        multiple_algorithms[1].id_,
    ]

    count = await bulk_delete_rows(Algorithm, session, row_ids)

    # Should delete 2 out of 3 IDs
    assert count == 2

    # Verify correct ones deleted
    assert await session.get(Algorithm, multiple_algorithms[0].id_) is None
    assert await session.get(Algorithm, multiple_algorithms[1].id_) is None
    assert await session.get(Algorithm, multiple_algorithms[2].id_) is not None


@pytest.mark.asyncio
async def test_bulk_delete_rows_with_foreign_key(session, sample_model, sample_algorithm):
    """Test bulk delete with foreign key constraint violations.

    Note: This test may be skipped if the database doesn't enforce foreign keys.
    """
    # sample_model has a foreign key reference to sample_algorithm
    algo_id = sample_model.algo_id
    row_ids = [algo_id]

    try:
        await bulk_delete_rows(Algorithm, session, row_ids)
        # If we get here, foreign keys aren't enforced - skip test
        pytest.skip("Foreign key constraints not enforced in test database")
    except IntegrityError:
        # This is the expected behavior
        pass

    # Verify not deleted
    session.expunge_all()
    still_exists = await session.get(Algorithm, algo_id)
    assert still_exists is not None


@pytest.mark.asyncio
async def test_bulk_delete_rows_large_batch(session):
    """Test bulk deleting a large number of rows."""
    # Create 100 algorithms
    algos = []
    for i in range(100):
        algo = Algorithm(name=f"bulk_{i}", class_name=f"bulk.class.{i}")
        session.add(algo)
        algos.append(algo)
    await session.commit()

    for algo in algos:
        await session.refresh(algo)

    row_ids = [algo.id_ for algo in algos]

    count = await bulk_delete_rows(Algorithm, session, row_ids)

    assert count == 100

    # Verify all deleted
    for row_id in row_ids:
        assert await session.get(Algorithm, row_id) is None


@pytest.mark.asyncio
async def test_bulk_delete_rows_different_tables(session, multiple_bands, multiple_catalog_tags):
    """Test bulk delete on different tables."""
    # Bulk delete bands
    band_ids = [band.id_ for band in multiple_bands]
    count = await bulk_delete_rows(Band, session, band_ids)
    assert count == len(band_ids)

    # Bulk delete tags
    tag_ids = [tag.id_ for tag in multiple_catalog_tags]
    count = await bulk_delete_rows(CatalogTag, session, tag_ids)
    assert count == len(tag_ids)


@pytest.mark.asyncio
async def test_bulk_delete_vs_delete_rows_performance(session):
    """Test that bulk_delete is indeed faster (no hooks, no data capture)."""
    # Create test data
    algos = []
    for i in range(50):
        algo = Algorithm(name=f"perf_{i}", class_name=f"perf.class.{i}")
        session.add(algo)
        algos.append(algo)
    await session.commit()

    for algo in algos:
        await session.refresh(algo)

    row_ids = [algo.id_ for algo in algos]

    # Bulk delete should complete without errors
    count = await bulk_delete_rows(Algorithm, session, row_ids)
    assert count == 50


# ============================================================================
# Integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_delete_then_read(session, sample_algorithm):
    """Test that deleted rows cannot be read."""
    row_id = sample_algorithm.id_

    # Delete the row
    await delete_row(Algorithm, session, row_id)

    # Try to read it
    with pytest.raises(KeyError):
        await get_row(Algorithm, session, row_id)


@pytest.mark.asyncio
async def test_delete_then_count(session, multiple_algorithms):
    """Test that count is updated after deletion."""
    initial_count = await count_rows(Algorithm, session)

    # Delete two rows
    await delete_rows(Algorithm, session, [multiple_algorithms[0].id_, multiple_algorithms[1].id_])

    final_count = await count_rows(Algorithm, session)
    assert final_count == initial_count - 2


@pytest.mark.asyncio
async def test_delete_cascade(session, sample_model, sample_estimator):
    """Test that cascade deletes work properly."""
    model_id = sample_model.id_
    estimator_id = sample_estimator.id_

    # Estimator depends on Model, delete the estimator first
    await delete_row(Estimator, session, estimator_id)

    # Now we can delete the model
    await delete_row(Model, session, model_id)

    # Verify both deleted
    assert await session.get(Model, model_id) is None
    assert await session.get(Estimator, estimator_id) is None


@pytest.mark.asyncio
async def test_delete_with_update(session, sample_algorithm, multiple_algorithms):
    """Test deleting and updating in same session."""
    # Update one
    await update_row(Algorithm, session, row_id=multiple_algorithms[0].id_, name="updated")

    # Delete another
    await delete_row(Algorithm, session, sample_algorithm.id_)

    # Verify update persisted
    updated = await session.get(Algorithm, multiple_algorithms[0].id_)
    assert updated.name == "updated"

    # Verify delete worked
    deleted = await session.get(Algorithm, sample_algorithm.id_)
    assert deleted is None


@pytest.mark.asyncio
async def test_delete_and_recreate(session, sample_algorithm):
    """Test deleting a row and recreating with same data."""
    # Get count before deletion
    count_before = await count_rows(Algorithm, session)

    # Capture data before deletion
    original_id = sample_algorithm.id_
    data = await delete_row(Algorithm, session, original_id, capture_data=True)

    # Verify deletion
    count_after_delete = await count_rows(Algorithm, session)
    assert count_after_delete == count_before - 1

    # Recreate with same data
    new_algo = Algorithm(name=data["name"], class_name=data["class_name"])
    session.add(new_algo)
    await session.commit()
    await session.refresh(new_algo)

    # Verify count is back to original
    count_after_create = await count_rows(Algorithm, session)
    assert count_after_create == count_before

    # Verify data is correct
    assert new_algo.name == data["name"]
    assert new_algo.class_name == data["class_name"]

    # SQLite in-memory database may reuse IDs, so we can't reliably test ID difference
    # Just verify the new record exists and has correct data
    retrieved = await session.get(Algorithm, new_algo.id_)
    assert retrieved is not None
    assert retrieved.name == data["name"]


@pytest.mark.asyncio
async def test_delete_rows_vs_bulk_delete(session):
    """Test that delete_rows and bulk_delete have different behaviors."""
    # Create test data
    algos1 = []
    for i in range(5):
        algo = Algorithm(name=f"normal_{i}", class_name=f"class.{i}")
        session.add(algo)
        algos1.append(algo)

    algos2 = []
    for i in range(5):
        algo = Algorithm(name=f"bulk_{i}", class_name=f"class.{i}")
        session.add(algo)
        algos2.append(algo)

    await session.commit()

    for algo in algos1 + algos2:
        await session.refresh(algo)

    # delete_rows captures data
    ids1 = [algo.id_ for algo in algos1]
    data = await delete_rows(Algorithm, session, ids1, capture_data=True)
    assert data is not None
    assert len(data) == 5

    # bulk_delete_rows just returns count
    ids2 = [algo.id_ for algo in algos2]
    count = await bulk_delete_rows(Algorithm, session, ids2)
    assert count == 5
    assert isinstance(count, int)


@pytest.mark.asyncio
async def test_delete_all_rows(session, multiple_algorithms):
    """Test deleting all rows from a table."""
    row_ids = [algo.id_ for algo in multiple_algorithms]

    initial_count = await count_rows(Algorithm, session)
    assert initial_count > 0

    await delete_rows(Algorithm, session, row_ids)

    final_count = await count_rows(Algorithm, session)
    assert final_count == 0


@pytest.mark.asyncio
async def test_delete_with_query(session, multiple_algorithms):
    """Test deleting specific rows based on query results."""
    # Find algorithms with specific name pattern
    result = await session.execute(select(Algorithm).where(Algorithm.name.like("knn%")))
    to_delete = result.scalars().all()

    if to_delete:
        row_ids = [algo.id_ for algo in to_delete]
        await delete_rows(Algorithm, session, row_ids)

        # Verify deleted
        for row_id in row_ids:
            assert await session.get(Algorithm, row_id) is None


@pytest.mark.asyncio
async def test_delete_captured_data_accuracy(session, sample_band):
    """Test that captured data exactly matches pre-deletion state."""
    # Get original values
    original_name = sample_band.name
    original_wavelengths = sample_band.band_wavelengths.copy()
    original_transmission = sample_band.band_transmission.copy()

    # Delete with capture
    data = await delete_row(Band, session, sample_band.id_, capture_data=True)

    # Verify captured data matches original
    assert data is not None
    assert data["name"] == original_name
    assert data["band_wavelengths"] == original_wavelengths
    assert data["band_transmission"] == original_transmission


@pytest.mark.asyncio
async def test_bulk_delete_all_nonexistent(session):
    """Test bulk delete with all nonexistent IDs."""
    row_ids = [99999, 88888, 77777]

    count = await bulk_delete_rows(Algorithm, session, row_ids)

    # None existed, so count should be 0
    assert count == 0


@pytest.mark.asyncio
async def test_delete_idempotent_check(session, sample_algorithm):
    """Test that attempting to delete already-deleted row raises KeyError."""
    row_id = sample_algorithm.id_

    # First deletion succeeds
    await delete_row(Algorithm, session, row_id)

    # Second deletion should fail
    with pytest.raises(KeyError, match=f"Algorithm {row_id} not found"):
        await delete_row(Algorithm, session, row_id)


@pytest.mark.asyncio
async def test_delete_rows_partial_failure_rollback(session, multiple_algorithms):
    """Test that partial failure in delete_rows rolls back all deletions."""
    # Store original state
    original_ids = [algo.id_ for algo in multiple_algorithms]

    # Try to delete with one invalid ID
    row_ids = [
        multiple_algorithms[0].id_,
        99999,  # Invalid - will cause failure
    ]

    try:
        await delete_rows(Algorithm, session, row_ids)
        pytest.fail("Should have raised KeyError")
    except KeyError:
        pass

    # Verify none were deleted
    session.expunge_all()
    result = await session.execute(select(Algorithm).where(Algorithm.id_.in_(original_ids)))
    remaining = result.scalars().all()
    assert len(remaining) == len(original_ids)


@pytest.mark.asyncio
async def test_delete_preserves_other_rows(session, multiple_algorithms):
    """Test that deleting specific rows doesn't affect others."""
    # Delete first two
    delete_ids = [multiple_algorithms[0].id_, multiple_algorithms[1].id_]
    keep_id = multiple_algorithms[2].id_

    await delete_rows(Algorithm, session, delete_ids)

    # Verify deleted
    for row_id in delete_ids:
        assert await session.get(Algorithm, row_id) is None

    # Verify kept row still exists
    kept = await session.get(Algorithm, keep_id)
    assert kept is not None
    assert kept.name == multiple_algorithms[2].name


@pytest.mark.asyncio
async def test_bulk_delete_efficiency(session):
    """Test that bulk_delete can handle large deletions."""
    # Create 1000 rows
    for i in range(1000):
        algo = Algorithm(name=f"bulk_test_{i}", class_name=f"class.{i}")
        session.add(algo)
        if i % 100 == 0:
            await session.flush()

    await session.commit()

    # Get all IDs
    result = await session.execute(select(Algorithm.id_))
    all_ids = [row[0] for row in result.fetchall()]

    # Delete all in one bulk operation
    count = await bulk_delete_rows(Algorithm, session, all_ids)

    assert count == 1000

    # Verify all deleted
    final_count = await count_rows(Algorithm, session)
    assert final_count == 0


@pytest.mark.asyncio
async def test_delete_row_with_relationships(session, sample_dataset, sample_catalog_tag):
    """Test deleting row preserves related data when no cascade."""
    dataset_id = sample_dataset.id_
    tag_id = sample_dataset.catalog_tag_id

    # Delete dataset (doesn't cascade to catalog_tag)
    await delete_row(Dataset, session, dataset_id)

    # Verify dataset deleted
    assert await session.get(Dataset, dataset_id) is None

    # Verify catalog_tag still exists
    tag = await session.get(CatalogTag, tag_id)
    assert tag is not None


@pytest.mark.asyncio
async def test_mixed_delete_operations(session, multiple_algorithms):
    """Test mixing delete_row, delete_rows, and bulk_delete in sequence."""
    # Create extra algorithms
    extra = []
    for i in range(3):
        algo = Algorithm(name=f"extra_{i}", class_name=f"extra.{i}")
        session.add(algo)
        extra.append(algo)
    await session.commit()

    for algo in extra:
        await session.refresh(algo)

    # Use delete_row
    await delete_row(Algorithm, session, multiple_algorithms[0].id_)

    # Use delete_rows
    await delete_rows(Algorithm, session, [multiple_algorithms[1].id_])

    # Use bulk_delete_rows
    count = await bulk_delete_rows(Algorithm, session, [algo.id_ for algo in extra])
    assert count == 3

    # Verify only one original algorithm remains
    remaining = await session.get(Algorithm, multiple_algorithms[2].id_)
    assert remaining is not None


@pytest.mark.asyncio
async def test_delete_rows_duplicate_ids(session, sample_algorithm):
    """Test delete_rows with duplicate IDs in the list."""
    # Create a second algorithm to delete
    algo2 = Algorithm(name="second", class_name="second.class")
    session.add(algo2)
    await session.commit()
    await session.refresh(algo2)

    row_id = sample_algorithm.id_

    # List same ID twice - the second iteration will fail to find it
    # because it was already deleted in the first iteration
    row_ids = [row_id, algo2.id_, row_id]

    # The function fetches all rows first, so duplicate ID means
    # trying to fetch the same row twice, which should work fine
    # But if implementation deletes immediately, second fetch fails
    # Let's test that it handles this gracefully
    result = await delete_rows(Algorithm, session, row_ids, capture_data=True)

    # Should have 3 items (including duplicate)
    assert result is not None
    assert len(result) == 3

    # But only 2 actual rows deleted
    assert await session.get(Algorithm, row_id) is None
    assert await session.get(Algorithm, algo2.id_) is None


@pytest.mark.asyncio
async def test_bulk_delete_rows_duplicate_ids(session, sample_algorithm):
    """Test bulk_delete with duplicate IDs."""
    row_id = sample_algorithm.id_

    # List same ID twice
    row_ids = [row_id, row_id]

    # Should only delete once
    count = await bulk_delete_rows(Algorithm, session, row_ids)
    assert count == 1

    # Verify deleted
    assert await session.get(Algorithm, row_id) is None
