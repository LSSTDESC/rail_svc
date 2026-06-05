"""Unit tests for CatalogBandAssoc database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.band import Band
from rail_svc.db.catalog_band_assoc import CatalogBandAssoc
from rail_svc.db.catalog_tag import CatalogTag
from rail_svc.models import CatalogBandAssoc as CatalogBandAssocPydantic
from rail_svc.models import CatalogBandAssocCreate

# ============================================================================
# CatalogBandAssoc Class Tests
# ============================================================================


class TestCatalogBandAssoc:
    """Tests for CatalogBandAssoc database model"""

    def test_catalog_band_assoc_tablename(self):
        """Test CatalogBandAssoc has correct table name"""
        assert CatalogBandAssoc.__tablename__ == "catalog_band_assoc"

    def test_catalog_band_assoc_class_string(self):
        """Test CatalogBandAssoc.class_string() returns table name"""
        assert CatalogBandAssoc.class_string() == "catalog_band_assoc"

    def test_pydantic_create_class(self):
        """Test CatalogBandAssoc.pydantic_create_class() returns correct model"""
        assert CatalogBandAssoc.pydantic_create_class() == CatalogBandAssocCreate

    def test_pydantic_model_class(self):
        """Test CatalogBandAssoc.pydantic_model_class() returns correct model"""
        assert CatalogBandAssoc.pydantic_model_class() == CatalogBandAssocPydantic

    @pytest.mark.asyncio
    async def test_create_catalog_band_assoc(self, session, sample_catalog_tag, sample_band):
        """Test creating a CatalogBandAssoc instance"""
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="test_mag",
            mag_err_column_name="test_mag_err",
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)

        assert assoc.id_ is not None
        assert assoc.catalog_tag_id == sample_catalog_tag.id_
        assert assoc.band_id == sample_band.id_
        assert assoc.mag_column_name == "test_mag"
        assert assoc.mag_err_column_name == "test_mag_err"

    @pytest.mark.asyncio
    async def test_query_catalog_band_assoc_by_id(self, session, sample_catalog_band_assoc):
        """Test querying catalog band assoc by id"""
        assoc = await session.get(CatalogBandAssoc, sample_catalog_band_assoc.id_)
        assert assoc is not None
        assert assoc.mag_column_name == sample_catalog_band_assoc.mag_column_name

    @pytest.mark.asyncio
    async def test_query_catalog_band_assoc_by_catalog_tag(
        self, session, sample_catalog_tag, sample_catalog_band_assoc
    ):
        """Test querying catalog band assoc by catalog_tag_id"""
        result = await session.execute(
            select(CatalogBandAssoc).where(CatalogBandAssoc.catalog_tag_id == sample_catalog_tag.id_)
        )
        assocs = result.scalars().all()
        assert len(assocs) >= 1
        assert sample_catalog_band_assoc.id_ in [a.id_ for a in assocs]

    @pytest.mark.asyncio
    async def test_query_catalog_band_assoc_by_band(self, session, sample_band, sample_catalog_band_assoc):
        """Test querying catalog band assoc by band_id"""
        result = await session.execute(
            select(CatalogBandAssoc).where(CatalogBandAssoc.band_id == sample_band.id_)
        )
        assocs = result.scalars().all()
        assert len(assocs) >= 1
        assert sample_catalog_band_assoc.id_ in [a.id_ for a in assocs]

    @pytest.mark.asyncio
    async def test_update_catalog_band_assoc(self, session, sample_catalog_band_assoc):
        """Test updating a CatalogBandAssoc"""
        new_mag_name = "updated_mag"
        sample_catalog_band_assoc.mag_column_name = new_mag_name
        await session.commit()
        await session.refresh(sample_catalog_band_assoc)

        assert sample_catalog_band_assoc.mag_column_name == new_mag_name

    @pytest.mark.asyncio
    async def test_delete_catalog_band_assoc(self, session, sample_catalog_band_assoc):
        """Test deleting a CatalogBandAssoc"""
        assoc_id = sample_catalog_band_assoc.id_
        await session.delete(sample_catalog_band_assoc)
        await session.commit()

        result = await session.get(CatalogBandAssoc, assoc_id)
        assert result is None

    def test_catalog_band_assoc_repr(self, sample_catalog_band_assoc):
        """Test CatalogBandAssoc __repr__ method"""
        repr_str = repr(sample_catalog_band_assoc)
        assert "CatalogBandAssoc" in repr_str
        assert str(sample_catalog_band_assoc.id_) in repr_str
        assert str(sample_catalog_band_assoc.catalog_tag_id) in repr_str
        assert str(sample_catalog_band_assoc.band_id) in repr_str
        assert sample_catalog_band_assoc.mag_column_name in repr_str

    def test_catalog_band_assoc_str(self, sample_catalog_band_assoc):
        """Test CatalogBandAssoc __str__ method"""
        assert str(sample_catalog_band_assoc) == sample_catalog_band_assoc.mag_column_name


class TestCatalogBandAssocPydanticIntegration:
    """Tests for CatalogBandAssoc Pydantic integration"""

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_to_pydantic(self, sample_catalog_band_assoc):
        """Test converting CatalogBandAssoc ORM to Pydantic model"""
        pydantic_obj = CatalogBandAssoc.to_pydantic(sample_catalog_band_assoc)

        assert isinstance(pydantic_obj, CatalogBandAssocPydantic)
        assert pydantic_obj.id_ == sample_catalog_band_assoc.id_
        assert pydantic_obj.catalog_tag_id == sample_catalog_band_assoc.catalog_tag_id
        assert pydantic_obj.band_id == sample_catalog_band_assoc.band_id
        assert pydantic_obj.mag_column_name == sample_catalog_band_assoc.mag_column_name
        assert pydantic_obj.mag_err_column_name == sample_catalog_band_assoc.mag_err_column_name

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_to_pydantic_dict(self, sample_catalog_band_assoc):
        """Test converting CatalogBandAssoc to dict via Pydantic"""
        data = CatalogBandAssoc.to_pydantic_dict(sample_catalog_band_assoc)

        assert isinstance(data, dict)
        assert data["id_"] == sample_catalog_band_assoc.id_
        assert data["catalog_tag_id"] == sample_catalog_band_assoc.catalog_tag_id
        assert data["band_id"] == sample_catalog_band_assoc.band_id
        assert data["mag_column_name"] == sample_catalog_band_assoc.mag_column_name
        assert data["mag_err_column_name"] == sample_catalog_band_assoc.mag_err_column_name


class TestCatalogBandAssocValidation:
    """Tests for CatalogBandAssoc field validation"""

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_requires_catalog_tag_id(self, session, sample_band):
        """Test that CatalogBandAssoc requires catalog_tag_id"""
        with pytest.raises(Exception):  # IntegrityError
            assoc = CatalogBandAssoc(
                band_id=sample_band.id_, mag_column_name="test_mag", mag_err_column_name="test_mag_err"
            )
            session.add(assoc)
            await session.commit()

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_requires_band_id(self, session, sample_catalog_tag):
        """Test that CatalogBandAssoc requires band_id"""
        with pytest.raises(Exception):  # IntegrityError
            assoc = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                mag_column_name="test_mag",
                mag_err_column_name="test_mag_err",
            )
            session.add(assoc)
            await session.commit()

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_requires_mag_column_name(
        self, session, sample_catalog_tag, sample_band
    ):
        """Test that CatalogBandAssoc requires mag_column_name"""
        with pytest.raises(Exception):  # IntegrityError
            assoc = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=sample_band.id_,
                mag_err_column_name="test_mag_err",
            )
            session.add(assoc)
            await session.commit()

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_requires_mag_err_column_name(
        self, session, sample_catalog_tag, sample_band
    ):
        """Test that CatalogBandAssoc requires mag_err_column_name"""
        with pytest.raises(Exception):  # IntegrityError
            assoc = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_, band_id=sample_band.id_, mag_column_name="test_mag"
            )
            session.add(assoc)
            await session.commit()

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_foreign_key_indexed(self):
        """Test that foreign key fields are indexed"""
        catalog_tag_id_col = CatalogBandAssoc.__table__.c.catalog_tag_id
        band_id_col = CatalogBandAssoc.__table__.c.band_id

        assert catalog_tag_id_col.index is True
        assert band_id_col.index is True


class TestCatalogBandAssocUniqueConstraints:
    """Tests for CatalogBandAssoc unique constraints"""

    @pytest.mark.asyncio
    async def test_unique_catalog_band_combination(self, session, sample_catalog_tag, sample_band):
        """Test that catalog_tag_id and band_id combination must be unique"""
        assoc1 = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="mag1",
            mag_err_column_name="mag_err1",
        )
        session.add(assoc1)
        await session.commit()

        # Try to create duplicate with different column names
        with pytest.raises(Exception):  # IntegrityError
            assoc2 = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=sample_band.id_,
                mag_column_name="mag2",
                mag_err_column_name="mag_err2",
            )
            session.add(assoc2)
            await session.commit()

    @pytest.mark.asyncio
    async def test_unique_mag_column_name_per_catalog(self, session, sample_catalog_tag, multiple_bands):
        """Test that mag_column_name must be unique per catalog_tag"""
        assoc1 = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=multiple_bands[0].id_,
            mag_column_name="duplicate_mag",
            mag_err_column_name="mag_err1",
        )
        session.add(assoc1)
        await session.commit()

        # Try to use same mag_column_name with different band
        with pytest.raises(Exception):  # IntegrityError
            assoc2 = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=multiple_bands[1].id_,
                mag_column_name="duplicate_mag",
                mag_err_column_name="mag_err2",
            )
            session.add(assoc2)
            await session.commit()

    @pytest.mark.asyncio
    async def test_unique_mag_err_column_name_per_catalog(self, session, sample_catalog_tag, multiple_bands):
        """Test that mag_err_column_name must be unique per catalog_tag"""
        assoc1 = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=multiple_bands[0].id_,
            mag_column_name="mag1",
            mag_err_column_name="duplicate_err",
        )
        session.add(assoc1)
        await session.commit()

        # Try to use same mag_err_column_name with different band
        with pytest.raises(Exception):  # IntegrityError
            assoc2 = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=multiple_bands[1].id_,
                mag_column_name="mag2",
                mag_err_column_name="duplicate_err",
            )
            session.add(assoc2)
            await session.commit()

    @pytest.mark.asyncio
    async def test_same_mag_column_name_different_catalogs(self, session, sample_band):
        """Test that same mag_column_name can be used in different catalogs"""
        tag1 = CatalogTag(name="catalog1")
        tag2 = CatalogTag(name="catalog2")
        session.add_all([tag1, tag2])
        await session.commit()
        await session.refresh(tag1)
        await session.refresh(tag2)

        assoc1 = CatalogBandAssoc(
            catalog_tag_id=tag1.id_,
            band_id=sample_band.id_,
            mag_column_name="g_mag",
            mag_err_column_name="g_mag_err1",
        )
        assoc2 = CatalogBandAssoc(
            catalog_tag_id=tag2.id_,
            band_id=sample_band.id_,
            mag_column_name="g_mag",
            mag_err_column_name="g_mag_err2",
        )
        session.add_all([assoc1, assoc2])
        await session.commit()

        # Should succeed - different catalogs
        await session.refresh(assoc1)
        await session.refresh(assoc2)
        assert assoc1.mag_column_name == assoc2.mag_column_name


class TestCatalogBandAssocCascade:
    """Tests for cascade delete behavior"""

    @pytest.mark.skip(reason="asycn sqlite doesn't handle cascade")
    @pytest.mark.asyncio
    async def test_cascade_delete_on_catalog_tag(self, session, sample_catalog_tag, sample_band):
        """Test that deleting catalog_tag cascades to associations"""
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="test_mag",
            mag_err_column_name="test_mag_err",
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)
        assoc_id = assoc.id_

        # Delete catalog tag
        await session.delete(sample_catalog_tag)
        await session.commit()

        # Association should be deleted
        result = await session.get(CatalogBandAssoc, assoc_id)
        assert result is None

    @pytest.mark.skip(reason="asycn sqlite doesn't handle cascade")
    @pytest.mark.asyncio
    async def test_cascade_delete_on_band(self, session, sample_catalog_tag, sample_band):
        """Test that deleting band cascades to associations"""
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="test_mag",
            mag_err_column_name="test_mag_err",
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)
        assoc_id = assoc.id_

        # Delete band
        await session.delete(sample_band)
        await session.commit()

        # Association should be deleted
        result = await session.get(CatalogBandAssoc, assoc_id)
        assert result is None


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_with_long_mag_column_name(
        self, session, sample_catalog_tag, sample_band
    ):
        """Test CatalogBandAssoc with maximum length mag_column_name"""
        long_name = "a" * 255
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name=long_name,
            mag_err_column_name="err",
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)

        assert assoc.mag_column_name == long_name

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_with_long_mag_err_column_name(
        self, session, sample_catalog_tag, sample_band
    ):
        """Test CatalogBandAssoc with maximum length mag_err_column_name"""
        long_name = "b" * 255
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="mag",
            mag_err_column_name=long_name,
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)

        assert assoc.mag_err_column_name == long_name

    @pytest.mark.asyncio
    async def test_catalog_band_assoc_with_special_characters(self, session, sample_catalog_tag, sample_band):
        """Test CatalogBandAssoc with special characters in column names"""
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="mag_g-band_v2.0",
            mag_err_column_name="err_g-band_v2.0",
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)

        assert assoc.mag_column_name == "mag_g-band_v2.0"
        assert assoc.mag_err_column_name == "err_g-band_v2.0"

    @pytest.mark.asyncio
    async def test_query_nonexistent_catalog_band_assoc(self, session):
        """Test querying for non-existent catalog band assoc"""
        result = await session.get(CatalogBandAssoc, 99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, engine, sample_catalog_tag, sample_band):
        """Test that multiple sessions work independently"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session1:
            assoc1 = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=sample_band.id_,
                mag_column_name="session1_mag",
                mag_err_column_name="session1_err",
            )
            session1.add(assoc1)
            await session1.commit()
            await session1.refresh(assoc1)
            assoc1_id = assoc1.id_

        async with async_session() as session2:
            assoc2 = await session2.get(CatalogBandAssoc, assoc1_id)
            assert assoc2 is not None
            assert assoc2.mag_column_name == "session1_mag"

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, session, sample_catalog_tag, sample_band):
        """Test that transaction rolls back on error"""
        initial_count = (await session.execute(select(CatalogBandAssoc))).scalars().all()
        initial_len = len(initial_count)

        try:
            assoc = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=sample_band.id_,
                mag_column_name="test_rollback",
                mag_err_column_name="test_rollback_err",
            )
            session.add(assoc)
            await session.flush()

            # Create duplicate to force error
            duplicate = CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=sample_band.id_,
                mag_column_name="other_mag",
                mag_err_column_name="other_err",
            )
            session.add(duplicate)
            await session.commit()
        except Exception:
            await session.rollback()

        final_count = (await session.execute(select(CatalogBandAssoc))).scalars().all()
        assert len(final_count) == initial_len

    @pytest.mark.asyncio
    async def test_empty_string_column_names(self, session, sample_catalog_tag, sample_band):
        """Test CatalogBandAssoc with empty string column names"""
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="",
            mag_err_column_name="",
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)

        assert assoc.mag_column_name == ""
        assert assoc.mag_err_column_name == ""


class TestConcurrentAccess:
    """Tests for concurrent database access"""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, session, sample_catalog_band_assoc):
        """Test that concurrent reads work correctly"""
        results = []
        for _ in range(5):
            result = await session.execute(
                select(CatalogBandAssoc).where(CatalogBandAssoc.id_ == sample_catalog_band_assoc.id_)
            )
            results.append(result.scalar_one())

        assert len(results) == 5
        assert all(r.id_ == sample_catalog_band_assoc.id_ for r in results)

    @pytest.mark.asyncio
    async def test_refresh_after_update(self, session, sample_catalog_band_assoc):
        """Test that refresh loads updated data"""
        original_mag_name = sample_catalog_band_assoc.mag_column_name

        new_mag_name = "updated_mag_name"
        sample_catalog_band_assoc.mag_column_name = new_mag_name
        await session.commit()
        await session.refresh(sample_catalog_band_assoc)

        assert sample_catalog_band_assoc.mag_column_name == new_mag_name
        assert sample_catalog_band_assoc.mag_column_name != original_mag_name


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestCatalogBandAssocBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session, sample_catalog_tag, multiple_bands):
        """Test inserting multiple catalog band assocs at once"""
        assocs = [
            CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=band.id_,
                mag_column_name=f"{band.name}_mag",
                mag_err_column_name=f"{band.name}_err",
            )
            for band in multiple_bands
        ]

        session.add_all(assocs)
        await session.commit()

        for assoc in assocs:
            await session.refresh(assoc)

        assert all(assoc.id_ is not None for assoc in assocs)
        assert len(assocs) == 3

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_catalog_band_assocs):
        """Test querying multiple catalog band assocs"""
        result = await session.execute(select(CatalogBandAssoc))
        assocs = result.scalars().all()

        assert len(assocs) >= 3
        mag_names = {assoc.mag_column_name for assoc in assocs}
        assert "r_mag" in mag_names
        assert "i_mag" in mag_names
        assert "z_mag" in mag_names

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_catalog_band_assocs):
        """Test updating multiple catalog band assocs"""
        for assoc in multiple_catalog_band_assocs:
            assoc.mag_column_name = f"updated_{assoc.mag_column_name}"

        await session.commit()

        for assoc in multiple_catalog_band_assocs:
            await session.refresh(assoc)
            assert assoc.mag_column_name.startswith("updated_")

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_catalog_band_assocs):
        """Test deleting multiple catalog band assocs"""
        assoc_ids = [assoc.id_ for assoc in multiple_catalog_band_assocs]

        for assoc in multiple_catalog_band_assocs:
            await session.delete(assoc)

        await session.commit()

        for assoc_id in assoc_ids:
            result = await session.get(CatalogBandAssoc, assoc_id)
            assert result is None


class TestTypeAnnotations:
    """Tests for type annotations and type hints"""

    def test_catalog_band_assoc_has_type_annotations(self):
        """Test that CatalogBandAssoc fields have proper type annotations"""
        assert hasattr(CatalogBandAssoc, "__annotations__")
        annotations = CatalogBandAssoc.__annotations__
        assert "id_" in annotations or hasattr(CatalogBandAssoc, "id_")
        assert "catalog_tag_id" in annotations or hasattr(CatalogBandAssoc, "catalog_tag_id")
        assert "band_id" in annotations or hasattr(CatalogBandAssoc, "band_id")
        assert "mag_column_name" in annotations or hasattr(CatalogBandAssoc, "mag_column_name")
        assert "mag_err_column_name" in annotations or hasattr(CatalogBandAssoc, "mag_err_column_name")


class TestCatalogBandAssocQueries:
    """Tests for various query patterns"""

    @pytest.mark.asyncio
    async def test_query_by_mag_column_name_pattern(self, session, multiple_catalog_band_assocs):
        """Test querying assocs with mag_column_name pattern matching"""
        result = await session.execute(
            select(CatalogBandAssoc).where(CatalogBandAssoc.mag_column_name.like("%_mag"))
        )
        assocs = result.scalars().all()

        assert len(assocs) >= 3
        assert all(assoc.mag_column_name.endswith("_mag") for assoc in assocs)

    @pytest.mark.asyncio
    async def test_query_order_by_mag_column_name(self, session, multiple_catalog_band_assocs):
        """Test querying assocs ordered by mag_column_name"""
        result = await session.execute(select(CatalogBandAssoc).order_by(CatalogBandAssoc.mag_column_name))
        assocs = result.scalars().all()

        # Check that results are ordered
        mag_names = [assoc.mag_column_name for assoc in assocs]
        assert mag_names == sorted(mag_names)

    @pytest.mark.asyncio
    async def test_query_with_limit(self, session, multiple_catalog_band_assocs):
        """Test querying assocs with limit"""
        result = await session.execute(select(CatalogBandAssoc).limit(2))
        assocs = result.scalars().all()

        assert len(assocs) <= 2

    @pytest.mark.asyncio
    async def test_count_catalog_band_assocs(self, session, multiple_catalog_band_assocs):
        """Test counting total number of catalog band assocs"""
        from sqlalchemy import func

        result = await session.execute(select(func.count()).select_from(CatalogBandAssoc))
        count = result.scalar()

        assert count >= 3

    @pytest.mark.asyncio
    async def test_query_by_multiple_criteria(self, session, sample_catalog_tag, sample_catalog_band_assoc):
        """Test querying with multiple WHERE conditions"""
        result = await session.execute(
            select(CatalogBandAssoc).where(
                CatalogBandAssoc.catalog_tag_id == sample_catalog_tag.id_,
                CatalogBandAssoc.mag_column_name == sample_catalog_band_assoc.mag_column_name,
            )
        )
        assoc = result.scalar_one()

        assert assoc.id_ == sample_catalog_band_assoc.id_


class TestCatalogBandAssocDataIntegrity:
    """Tests for data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_update_mag_column_name_independently(self, session, sample_catalog_band_assoc):
        """Test that mag_column_name can be updated without affecting mag_err_column_name"""
        original_err_name = sample_catalog_band_assoc.mag_err_column_name

        sample_catalog_band_assoc.mag_column_name = "new_mag_name"
        await session.commit()
        await session.refresh(sample_catalog_band_assoc)

        assert sample_catalog_band_assoc.mag_column_name == "new_mag_name"
        assert sample_catalog_band_assoc.mag_err_column_name == original_err_name

    @pytest.mark.asyncio
    async def test_update_mag_err_column_name_independently(self, session, sample_catalog_band_assoc):
        """Test that mag_err_column_name can be updated without affecting mag_column_name"""
        original_mag_name = sample_catalog_band_assoc.mag_column_name

        sample_catalog_band_assoc.mag_err_column_name = "new_err_name"
        await session.commit()
        await session.refresh(sample_catalog_band_assoc)

        assert sample_catalog_band_assoc.mag_err_column_name == "new_err_name"
        assert sample_catalog_band_assoc.mag_column_name == original_mag_name

    @pytest.mark.asyncio
    async def test_foreign_key_integrity(self, session, sample_catalog_tag, sample_band):
        """Test that foreign key references are maintained"""
        assoc = CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="integrity_mag",
            mag_err_column_name="integrity_err",
        )
        session.add(assoc)
        await session.commit()
        await session.refresh(assoc)

        # Verify foreign keys are correct
        assert assoc.catalog_tag_id == sample_catalog_tag.id_
        assert assoc.band_id == sample_band.id_

        # Verify we can query the referenced objects
        tag = await session.get(CatalogTag, assoc.catalog_tag_id)
        band = await session.get(Band, assoc.band_id)

        assert tag is not None
        assert band is not None


class TestCatalogBandAssocPydanticValidation:
    """Tests for Pydantic model integration and validation"""

    @pytest.mark.asyncio
    async def test_to_pydantic_list(self, multiple_catalog_band_assocs):
        """Test converting multiple assocs to Pydantic list"""
        pydantic_list = CatalogBandAssoc.to_pydantic_list(multiple_catalog_band_assocs)

        assert len(pydantic_list) == 3
        assert all(isinstance(obj, CatalogBandAssocPydantic) for obj in pydantic_list)
        assert pydantic_list[0].mag_column_name == "r_mag"
        assert pydantic_list[1].mag_column_name == "i_mag"
        assert pydantic_list[2].mag_column_name == "z_mag"

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list(self, multiple_catalog_band_assocs):
        """Test converting multiple assocs to dict list"""
        dict_list = CatalogBandAssoc.to_pydantic_dict_list(multiple_catalog_band_assocs)

        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert all("mag_column_name" in d for d in dict_list)
        assert all("mag_err_column_name" in d for d in dict_list)
        assert all("catalog_tag_id" in d for d in dict_list)
        assert all("band_id" in d for d in dict_list)


class TestTableConstraints:
    """Tests for table-level constraints"""

    def test_unique_constraints_exist(self):
        """Test that unique constraints are properly defined"""
        table = CatalogBandAssoc.__table__
        constraint_names = {c.name for c in table.constraints if hasattr(c, "name")}

        # Check for expected unique constraints
        assert "uq_catalog_band" in constraint_names
        assert "uq_catalog_mag_column_name" in constraint_names
        assert "uq_catalog_mag_err_column_name" in constraint_names

    def test_table_has_primary_key(self):
        """Test that table has primary key defined"""
        table = CatalogBandAssoc.__table__
        pk_columns = [col.name for col in table.primary_key.columns]

        assert "id_" in pk_columns

    def test_foreign_key_constraints_exist(self):
        """Test that foreign key constraints are properly defined"""
        table = CatalogBandAssoc.__table__

        catalog_tag_fk = table.c.catalog_tag_id.foreign_keys
        band_fk = table.c.band_id.foreign_keys

        assert len(catalog_tag_fk) > 0
        assert len(band_fk) > 0


class TestCatalogBandAssocMultipleCatalogs:
    """Tests for associations across multiple catalogs"""

    @pytest.mark.asyncio
    async def test_same_band_different_catalogs(self, session, sample_band):
        """Test that same band can have different aliases in different catalogs"""
        tag1 = CatalogTag(name="catalog_a")
        tag2 = CatalogTag(name="catalog_b")
        session.add_all([tag1, tag2])
        await session.commit()
        await session.refresh(tag1)
        await session.refresh(tag2)

        assoc1 = CatalogBandAssoc(
            catalog_tag_id=tag1.id_,
            band_id=sample_band.id_,
            mag_column_name="g_psf",
            mag_err_column_name="g_psf_err",
        )
        assoc2 = CatalogBandAssoc(
            catalog_tag_id=tag2.id_,
            band_id=sample_band.id_,
            mag_column_name="g_cmodel",
            mag_err_column_name="g_cmodel_err",
        )
        session.add_all([assoc1, assoc2])
        await session.commit()

        await session.refresh(assoc1)
        await session.refresh(assoc2)

        assert assoc1.band_id == assoc2.band_id
        assert assoc1.mag_column_name != assoc2.mag_column_name

    @pytest.mark.asyncio
    async def test_query_all_bands_for_catalog(
        self, session, sample_catalog_tag, multiple_catalog_band_assocs
    ):
        """Test querying all bands associated with a catalog"""
        result = await session.execute(
            select(CatalogBandAssoc).where(CatalogBandAssoc.catalog_tag_id == sample_catalog_tag.id_)
        )
        assocs = result.scalars().all()

        assert len(assocs) >= 3
        band_ids = {assoc.band_id for assoc in assocs}
        assert len(band_ids) >= 3  # At least 3 different bands

    @pytest.mark.asyncio
    async def test_query_all_catalogs_for_band(self, session, sample_band):
        """Test querying all catalogs that use a specific band"""
        # Create multiple catalogs using the same band
        tags = [CatalogTag(name=f"catalog_{i}") for i in range(3)]
        session.add_all(tags)
        await session.commit()

        for tag in tags:
            await session.refresh(tag)
            assoc = CatalogBandAssoc(
                catalog_tag_id=tag.id_,
                band_id=sample_band.id_,
                mag_column_name=f"mag_{tag.name}",
                mag_err_column_name=f"err_{tag.name}",
            )
            session.add(assoc)

        await session.commit()

        result = await session.execute(
            select(CatalogBandAssoc).where(CatalogBandAssoc.band_id == sample_band.id_)
        )
        assocs = result.scalars().all()

        assert len(assocs) >= 3
        catalog_ids = {assoc.catalog_tag_id for assoc in assocs}
        assert len(catalog_ids) >= 3  # At least 3 different catalogs
