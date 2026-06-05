"""Unit tests for DatasetAssoc table operations."""

import pytest

from rail_svc.db import DatasetAssoc
from rail_svc.db_oper.base import TableContext
from rail_svc.db_oper.dataset_assoc import DatasetAssocOperations, dataset_assoc
from rail_svc.models import DatasetAssoc as DatasetAssocModel


# ============================================================================
# DatasetAssocOperations class tests
# ============================================================================


def test_dataset_assoc_operations_can_be_instantiated():
    """Test that DatasetAssocOperations can be instantiated."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    assert isinstance(ops, DatasetAssocOperations)


# ============================================================================
# Module-level singleton tests
# ============================================================================


def test_module_singleton_exists():
    """Test that module-level dataset_assoc singleton exists."""
    assert dataset_assoc is not None


def test_module_singleton_is_dataset_assoc_operations():
    """Test that module singleton is a DatasetAssocOperations instance."""
    assert isinstance(dataset_assoc, DatasetAssocOperations)


def test_module_singleton_is_singleton():
    """Test that module exports the same instance."""
    from rail_svc.db_oper.dataset_assoc import dataset_assoc as dataset_assoc2

    # Should be the exact same object
    assert dataset_assoc is dataset_assoc2


# ============================================================================
# Integration with base class tests
# ============================================================================


@pytest.mark.asyncio
async def test_dataset_assoc_operations_inherits_crud_methods(session, sample_dataset_assoc):
    """Test that DatasetAssocOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_dataset_assoc.id_)
    assert result.id_ == sample_dataset_assoc.id_


@pytest.mark.asyncio
async def test_dataset_assoc_operations_inherits_filter_methods(session, multiple_dataset_assocs):
    """Test that DatasetAssocOperations inherits filter methods from base."""
    from rail_svc.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="gaia_to_match")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "gaia_to_match"


@pytest.mark.asyncio
async def test_dataset_assoc_operations_inherits_pydantic_conversion(session, sample_dataset_assoc):
    """Test that DatasetAssocOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_dataset_assoc)

    assert isinstance(pydantic_obj, DatasetAssocModel)
    assert pydantic_obj.name == sample_dataset_assoc.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_dataset_assoc):
    """Test that module singleton can perform get_row operation."""
    result = await dataset_assoc.get_row(session, sample_dataset_assoc.id_)

    assert result.id_ == sample_dataset_assoc.id_
    assert result.name == sample_dataset_assoc.name


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_dataset_assocs):
    """Test that module singleton can filter rows."""
    from rail_svc.models.filtering import Filter, FilterOp

    filters = [Filter(field="name", op=FilterOp.IN, value=["gaia_to_match", "sdss_to_match"])]
    results = await dataset_assoc.filter_rows(session, filters=filters)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"gaia_to_match", "sdss_to_match"}


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_dataset_assoc):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = dataset_assoc.to_pydantic(sample_dataset_assoc)

    assert isinstance(pydantic_obj, DatasetAssocModel)
    assert pydantic_obj.name == sample_dataset_assoc.name
    assert pydantic_obj.matched_dataset_id == sample_dataset_assoc.matched_dataset_id


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_dataset_assocs):
    """Test that module singleton can count rows."""
    count = await dataset_assoc.count_rows(session)

    assert count == 2


# ============================================================================
# get_create_kwargs tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_create_kwargs_by_ids(session, matched_dataset, component_dataset_1):
    """Test get_create_kwargs with IDs provided."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test_assoc",
        matched_dataset_id=matched_dataset.id_,
        component_dataset_id=component_dataset_1.id_,
    )

    assert kwargs["name"] == "test_assoc"
    assert kwargs["matched_dataset_id"] == matched_dataset.id_
    assert kwargs["component_dataset_id"] == component_dataset_1.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_by_names(session, matched_dataset, component_dataset_1):
    """Test get_create_kwargs with names provided."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test_assoc_by_name",
        matched_dataset_name=matched_dataset.name,
        component_dataset_name=component_dataset_1.name,
    )

    assert kwargs["name"] == "test_assoc_by_name"
    assert kwargs["matched_dataset_id"] == matched_dataset.id_
    assert kwargs["component_dataset_id"] == component_dataset_1.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_mixed_id_and_name(session, matched_dataset, component_dataset_1):
    """Test get_create_kwargs with mixed IDs and names."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test_assoc_mixed",
        matched_dataset_id=matched_dataset.id_,
        component_dataset_name=component_dataset_1.name,
    )

    assert kwargs["matched_dataset_id"] == matched_dataset.id_
    assert kwargs["component_dataset_id"] == component_dataset_1.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_self_reference_raises(session, matched_dataset):
    """Test get_create_kwargs raises error for self-reference."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    with pytest.raises(ValueError, match="A dataset cannot be associated with itself"):
        await ops.get_create_kwargs(
            session,
            name="self_ref",
            matched_dataset_id=matched_dataset.id_,
            component_dataset_id=matched_dataset.id_,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_matched_dataset(session, component_dataset_1):
    """Test get_create_kwargs raises error for nonexistent matched dataset."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    with pytest.raises((ValueError, KeyError)):
        await ops.get_create_kwargs(
            session,
            name="test",
            matched_dataset_name="nonexistent",
            component_dataset_id=component_dataset_1.id_,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_component_dataset(session, matched_dataset):
    """Test get_create_kwargs raises error for nonexistent component dataset."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    with pytest.raises((ValueError, KeyError)):
        await ops.get_create_kwargs(
            session, name="test", matched_dataset_id=matched_dataset.id_, component_dataset_name="nonexistent"
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_ignores_extra_kwargs(session, matched_dataset, component_dataset_1, caplog):
    """Test that extra kwargs are logged and ignored."""
    context = TableContext.from_db_class(DatasetAssoc)
    ops = DatasetAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        name="test",
        matched_dataset_id=matched_dataset.id_,
        component_dataset_id=component_dataset_1.id_,
        extra_field="should_be_ignored",
    )

    # Extra field should not be in result
    assert "extra_field" not in kwargs
    assert kwargs["name"] == "test"


# ============================================================================
# create_row integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_with_ids(session, matched_dataset, component_dataset_1):
    """Test creating association using IDs."""
    result = await dataset_assoc.create_row(
        session,
        name="test_assoc_create",
        matched_dataset_id=matched_dataset.id_,
        component_dataset_id=component_dataset_1.id_,
        validate=False,
    )
    await session.commit()

    assert result.name == "test_assoc_create"
    assert result.matched_dataset_id == matched_dataset.id_
    assert result.component_dataset_id == component_dataset_1.id_


@pytest.mark.asyncio
async def test_create_row_with_names(session, matched_dataset, component_dataset_2):
    """Test creating association using names."""
    result = await dataset_assoc.create_row(
        session,
        name="test_assoc_by_names",
        matched_dataset_name=matched_dataset.name,
        component_dataset_name=component_dataset_2.name,
        validate=False,
    )
    await session.commit()

    assert result.matched_dataset_id == matched_dataset.id_
    assert result.component_dataset_id == component_dataset_2.id_


@pytest.mark.asyncio
async def test_create_row_prevents_self_reference(session, matched_dataset):
    """Test that self-reference is prevented during creation."""
    with pytest.raises(ValueError, match="A dataset cannot be associated with itself"):
        await dataset_assoc.create_row(
            session,
            name="self_ref",
            matched_dataset_id=matched_dataset.id_,
            component_dataset_id=matched_dataset.id_,
            validate=False,
        )
