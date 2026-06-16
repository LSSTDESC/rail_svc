"""Unit tests for Band table operations."""

import pytest

from rail_svc.db import Band
from rail_svc.db_oper.band import BandOperations, band
from rail_svc.db_oper.base import TableContext
from rail_svc.models import Band as BandModel

# ============================================================================
# BandOperations class tests
# ============================================================================


async def test_band_operations_inherits_crud_methods(session, sample_band):
    """Test that BandOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(Band)
    ops = BandOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_band.id_)
    assert result.id_ == sample_band.id_


@pytest.mark.asyncio
async def test_band_operations_inherits_filter_methods(session, multiple_bands):
    """Test that BandOperations inherits filter methods from base."""
    from rail_svc.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(Band)
    ops = BandOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="name", op=FilterOp.EQ, value="u_band")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].name == "u_band"


@pytest.mark.asyncio
async def test_band_operations_inherits_pydantic_conversion(session, sample_band):
    """Test that BandOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(Band)
    ops = BandOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_band)

    assert isinstance(pydantic_obj, BandModel)
    assert pydantic_obj.name == sample_band.name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_band):
    """Test that module singleton can perform get_row operation."""
    result = await band.get_row(session, sample_band.id_)

    assert result.id_ == sample_band.id_
    assert result.name == sample_band.name


@pytest.mark.asyncio
async def test_module_singleton_create_row(session):
    """Test that module singleton can create rows."""
    result = await band.create_row(
        session,
        name="singleton_band",
        band_wavelengths=[100.0, 200.0],
        band_transmission=[0.5, 0.8],
        validate=False,
    )
    await session.commit()

    assert result.name == "singleton_band"
    assert result.band_wavelengths == [100.0, 200.0]
    assert result.band_transmission == [0.5, 0.8]


@pytest.mark.asyncio
async def test_module_singleton_filter_rows(session, multiple_bands):
    """Test that module singleton can filter rows."""
    from rail_svc.models.filtering import Filter, FilterOp

    filters = [Filter(field="name", op=FilterOp.IN, value=["u_band", "r_band"])]
    results = await band.filter_rows(session, filters=filters)

    assert len(results) == 2
    names = {r.name for r in results}
    assert names == {"u_band", "r_band"}


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_band):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = band.to_pydantic(sample_band)

    assert isinstance(pydantic_obj, BandModel)
    assert pydantic_obj.name == sample_band.name
    assert pydantic_obj.band_wavelengths == sample_band.band_wavelengths
    assert pydantic_obj.band_transmission == sample_band.band_transmission


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_bands):
    """Test that module singleton can count rows."""
    count = await band.count_rows(session)

    assert count == 3
