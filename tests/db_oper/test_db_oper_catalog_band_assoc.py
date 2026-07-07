"""Unit tests for CatalogBandAssoc table operations."""

import pytest

from rail_svc.db import CatalogBandAssoc
from macon.db_oper.base import TableContext
from rail_svc.db_oper.catalog_band_assoc import CatalogBandAssocOperations, catalog_band_assoc
from rail_svc.models import CatalogBandAssoc as CatalogBandAssocModel

# ============================================================================
# CatalogBandAssocOperations class tests
# ============================================================================


async def test_catalog_band_assoc_operations_inherits_crud_methods(session, sample_catalog_band_assoc):
    """Test that CatalogBandAssocOperations inherits CRUD methods from base."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    # Should have get_row method from base class
    result = await ops.get_row(session, sample_catalog_band_assoc.id_)
    assert result.id_ == sample_catalog_band_assoc.id_


@pytest.mark.asyncio
async def test_catalog_band_assoc_operations_inherits_filter_methods(session, multiple_catalog_band_assocs):
    """Test that CatalogBandAssocOperations inherits filter methods from base."""
    from macon.models.filtering import Filter, FilterOp

    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    # Should have filter_rows method from base class
    filters = [Filter(field="mag_column_name", op=FilterOp.EQ, value="r_mag")]
    results = await ops.filter_rows(session, filters=filters)

    assert len(results) == 1
    assert results[0].mag_column_name == "r_mag"


@pytest.mark.asyncio
async def test_catalog_band_assoc_operations_inherits_pydantic_conversion(session, sample_catalog_band_assoc):
    """Test that CatalogBandAssocOperations inherits Pydantic conversion methods."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    # Should have to_pydantic method from base class
    pydantic_obj = ops.to_pydantic(sample_catalog_band_assoc)

    assert isinstance(pydantic_obj, CatalogBandAssocModel)
    assert pydantic_obj.mag_column_name == sample_catalog_band_assoc.mag_column_name


# ============================================================================
# Module singleton functionality tests
# ============================================================================


@pytest.mark.asyncio
async def test_module_singleton_get_row(session, sample_catalog_band_assoc):
    """Test that module singleton can perform get_row operation."""
    result = await catalog_band_assoc.get_row(session, sample_catalog_band_assoc.id_)

    assert result.id_ == sample_catalog_band_assoc.id_
    assert result.mag_column_name == sample_catalog_band_assoc.mag_column_name


@pytest.mark.asyncio
async def test_module_singleton_to_pydantic(session, sample_catalog_band_assoc):
    """Test that module singleton can convert to Pydantic."""
    pydantic_obj = catalog_band_assoc.to_pydantic(sample_catalog_band_assoc)

    assert isinstance(pydantic_obj, CatalogBandAssocModel)
    assert pydantic_obj.mag_column_name == sample_catalog_band_assoc.mag_column_name
    assert pydantic_obj.mag_err_column_name == sample_catalog_band_assoc.mag_err_column_name


@pytest.mark.asyncio
async def test_module_singleton_count_rows(session, multiple_catalog_band_assocs):
    """Test that module singleton can count rows."""
    count = await catalog_band_assoc.count_rows(session)

    assert count == 3


# ============================================================================
# get_create_kwargs tests
# ============================================================================


@pytest.mark.asyncio
async def test_get_create_kwargs_by_ids(session, sample_catalog_tag, sample_band):
    """Test get_create_kwargs with IDs provided."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        mag_column_name="g_mag",
        mag_err_column_name="g_mag_err",
        catalog_tag_id=sample_catalog_tag.id_,
        band_id=sample_band.id_,
    )

    assert kwargs["mag_column_name"] == "g_mag"
    assert kwargs["mag_err_column_name"] == "g_mag_err"
    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_
    assert kwargs["band_id"] == sample_band.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_by_names(session, sample_catalog_tag, sample_band):
    """Test get_create_kwargs with names provided."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        mag_column_name="r_mag",
        mag_err_column_name="r_mag_err",
        catalog_tag_name=sample_catalog_tag.name,
        band_name=sample_band.name,
    )

    assert kwargs["mag_column_name"] == "r_mag"
    assert kwargs["mag_err_column_name"] == "r_mag_err"
    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_
    assert kwargs["band_id"] == sample_band.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_mixed_id_and_name(session, sample_catalog_tag, sample_band):
    """Test get_create_kwargs with mixed IDs and names."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        mag_column_name="i_mag",
        mag_err_column_name="i_mag_err",
        catalog_tag_id=sample_catalog_tag.id_,
        band_name=sample_band.name,
    )

    assert kwargs["mag_column_name"] == "i_mag"
    assert kwargs["mag_err_column_name"] == "i_mag_err"
    assert kwargs["catalog_tag_id"] == sample_catalog_tag.id_
    assert kwargs["band_id"] == sample_band.id_


@pytest.mark.asyncio
async def test_get_create_kwargs_invalid_mag_column_name(session, sample_catalog_tag, sample_band):
    """Test get_create_kwargs raises error for invalid mag_column_name."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    with pytest.raises(ValueError, match="mag_column_name must be a non-empty string"):
        await ops.get_create_kwargs(
            session,
            mag_column_name="",
            mag_err_column_name="g_mag_err",
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_invalid_mag_err_column_name(session, sample_catalog_tag, sample_band):
    """Test get_create_kwargs raises error for invalid mag_err_column_name."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    with pytest.raises(ValueError, match="mag_err_column_name must be a non-empty string"):
        await ops.get_create_kwargs(
            session,
            mag_column_name="g_mag",
            mag_err_column_name=None,
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_catalog_tag_name(session, sample_band):
    """Test get_create_kwargs raises error for nonexistent catalog tag name."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    with pytest.raises(ValueError, match="CatalogTag with name 'nonexistent' not found"):
        await ops.get_create_kwargs(
            session,
            mag_column_name="g_mag",
            mag_err_column_name="g_mag_err",
            catalog_tag_name="nonexistent",
            band_id=sample_band.id_,
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_nonexistent_band_name(session, sample_catalog_tag):
    """Test get_create_kwargs raises error for nonexistent band name."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    with pytest.raises(ValueError, match="Band with name 'nonexistent' not found"):
        await ops.get_create_kwargs(
            session,
            mag_column_name="g_mag",
            mag_err_column_name="g_mag_err",
            catalog_tag_id=sample_catalog_tag.id_,
            band_name="nonexistent",
        )


@pytest.mark.asyncio
async def test_get_create_kwargs_with_extra_kwargs(session, sample_catalog_tag, sample_band):
    """Test get_create_kwargs passes through extra kwargs."""
    context = TableContext.from_db_class(CatalogBandAssoc)
    ops = CatalogBandAssocOperations(context)

    kwargs = await ops.get_create_kwargs(
        session,
        mag_column_name="z_mag",
        mag_err_column_name="z_mag_err",
        catalog_tag_id=sample_catalog_tag.id_,
        band_id=sample_band.id_,
        extra_field="extra_value",
    )

    assert kwargs["extra_field"] == "extra_value"


# ============================================================================
# create_row integration tests
# ============================================================================


@pytest.mark.asyncio
async def test_create_row_with_ids(session, sample_catalog_tag, sample_band):
    """Test creating row using IDs."""
    result = await catalog_band_assoc.create_row(
        session,
        mag_column_name="test_mag",
        mag_err_column_name="test_mag_err",
        catalog_tag_id=sample_catalog_tag.id_,
        band_id=sample_band.id_,
        validate=False,
    )
    await session.commit()

    assert result.mag_column_name == "test_mag"
    assert result.mag_err_column_name == "test_mag_err"
    assert result.catalog_tag_id == sample_catalog_tag.id_
    assert result.band_id == sample_band.id_


@pytest.mark.asyncio
async def test_create_row_with_names(session, sample_catalog_tag, sample_band):
    """Test creating row using names."""
    result = await catalog_band_assoc.create_row(
        session,
        mag_column_name="test_mag_2",
        mag_err_column_name="test_mag_err_2",
        catalog_tag_name=sample_catalog_tag.name,
        band_name=sample_band.name,
        validate=False,
    )
    await session.commit()

    assert result.mag_column_name == "test_mag_2"
    assert result.catalog_tag_id == sample_catalog_tag.id_
    assert result.band_id == sample_band.id_


@pytest.mark.asyncio
async def test_create_row_mixed(session, sample_catalog_tag, sample_band):
    """Test creating row with mixed IDs and names."""
    result = await catalog_band_assoc.create_row(
        session,
        mag_column_name="test_mag_3",
        mag_err_column_name="test_mag_err_3",
        catalog_tag_id=sample_catalog_tag.id_,
        band_name=sample_band.name,
        validate=False,
    )
    await session.commit()

    assert result.catalog_tag_id == sample_catalog_tag.id_
    assert result.band_id == sample_band.id_
