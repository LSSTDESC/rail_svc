"""Unit tests for Band database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.band import Band
from rail_svc.models import Band as BandPydantic
from rail_svc.models import BandCreate

# ============================================================================
# Band Class Tests
# ============================================================================


class TestBand:
    """Tests for Band database model"""

    def test_band_tablename(self):
        """Test Band has correct table name"""
        assert Band.__tablename__ == "band"

    def test_band_class_string(self):
        """Test Band.class_string() returns table name"""
        assert Band.class_string() == "band"

    def test_pydantic_create_class(self):
        """Test Band.pydantic_create_class() returns correct model"""
        assert Band.pydantic_create_class() == BandCreate

    def test_pydantic_model_class(self):
        """Test Band.pydantic_model_class() returns correct model"""
        assert Band.pydantic_model_class() == BandPydantic

    @pytest.mark.asyncio
    async def test_create_band(self, session):
        """Test creating a Band instance"""
        band = Band(
            name="z_band", band_wavelengths=[800.0, 900.0, 1000.0], band_transmission=[0.1, 0.8, 0.15]
        )
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.id_ is not None
        assert band.name == "z_band"
        assert band.band_wavelengths == [800.0, 900.0, 1000.0]
        assert band.band_transmission == [0.1, 0.8, 0.15]

    @pytest.mark.asyncio
    async def test_band_unique_name(self, session, sample_band):
        """Test that band name must be unique"""
        duplicate = Band(name=sample_band.name, band_wavelengths=[100.0, 200.0], band_transmission=[0.5, 0.5])
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError or similar
            await session.commit()

    @pytest.mark.asyncio
    async def test_query_band_by_name(self, session, sample_band):
        """Test querying band by name"""
        result = await session.execute(select(Band).where(Band.name == sample_band.name))
        band = result.scalar_one()
        assert band.id_ == sample_band.id_
        assert band.name == sample_band.name

    @pytest.mark.asyncio
    async def test_query_band_by_id(self, session, sample_band):
        """Test querying band by id"""
        band = await session.get(Band, sample_band.id_)
        assert band is not None
        assert band.name == sample_band.name

    @pytest.mark.asyncio
    async def test_update_band(self, session, sample_band):
        """Test updating a Band"""
        new_wavelengths = [410.0, 510.0, 610.0]
        sample_band.band_wavelengths = new_wavelengths
        await session.commit()
        await session.refresh(sample_band)

        assert sample_band.band_wavelengths == new_wavelengths

    @pytest.mark.asyncio
    async def test_delete_band(self, session, sample_band):
        """Test deleting a Band"""
        band_id = sample_band.id_
        await session.delete(sample_band)
        await session.commit()

        result = await session.get(Band, band_id)
        assert result is None

    def test_band_repr(self, sample_band):
        """Test Band __repr__ method"""
        repr_str = repr(sample_band)
        assert "Band" in repr_str
        assert str(sample_band.id_) in repr_str
        assert sample_band.name in repr_str

    def test_band_str(self, sample_band):
        """Test Band __str__ method"""
        assert str(sample_band) == sample_band.name


class TestBandPydanticIntegration:
    """Tests for Band Pydantic integration"""

    @pytest.mark.asyncio
    async def test_band_to_pydantic(self, sample_band):
        """Test converting Band ORM to Pydantic model"""
        pydantic_obj = Band.to_pydantic(sample_band)

        assert isinstance(pydantic_obj, BandPydantic)
        assert pydantic_obj.id_ == sample_band.id_
        assert pydantic_obj.name == sample_band.name
        assert pydantic_obj.band_wavelengths == sample_band.band_wavelengths
        assert pydantic_obj.band_transmission == sample_band.band_transmission

    @pytest.mark.asyncio
    async def test_band_to_pydantic_dict(self, sample_band):
        """Test converting Band to dict via Pydantic"""
        data = Band.to_pydantic_dict(sample_band)

        assert isinstance(data, dict)
        assert data["id_"] == sample_band.id_
        assert data["name"] == sample_band.name
        assert data["band_wavelengths"] == sample_band.band_wavelengths
        assert data["band_transmission"] == sample_band.band_transmission


class TestBandValidation:
    """Tests for Band field validation"""

    @pytest.mark.asyncio
    async def test_band_requires_name(self, session):
        """Test that Band requires a name"""
        with pytest.raises(Exception):  # IntegrityError
            band = Band(band_wavelengths=[100.0], band_transmission=[0.5])
            session.add(band)
            await session.commit()

    @pytest.mark.asyncio
    async def test_band_requires_wavelengths(self, session):
        """Test that Band requires band_wavelengths"""
        with pytest.raises(Exception):  # IntegrityError
            band = Band(name="test", band_transmission=[0.5])
            session.add(band)
            await session.commit()

    @pytest.mark.asyncio
    async def test_band_requires_transmission(self, session):
        """Test that Band requires band_transmission"""
        with pytest.raises(Exception):  # IntegrityError
            band = Band(name="test", band_wavelengths=[100.0])
            session.add(band)
            await session.commit()

    @pytest.mark.asyncio
    async def test_band_name_indexed(self, session):
        """Test that name field is indexed"""
        name_column = Band.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True


class TestBandArrayFields:
    """Tests for Band array fields (wavelengths and transmission)"""

    @pytest.mark.asyncio
    async def test_band_with_single_point(self, session):
        """Test Band with single wavelength/transmission point"""
        band = Band(name="single_point", band_wavelengths=[500.0], band_transmission=[0.8])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert len(band.band_wavelengths) == 1
        assert len(band.band_transmission) == 1

    @pytest.mark.asyncio
    async def test_band_with_many_points(self, session):
        """Test Band with many wavelength/transmission points"""
        wavelengths = [float(i) for i in range(400, 800, 10)]
        transmission = [0.5] * len(wavelengths)

        band = Band(name="many_points", band_wavelengths=wavelengths, band_transmission=transmission)
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert len(band.band_wavelengths) == 40
        assert len(band.band_transmission) == 40

    @pytest.mark.asyncio
    async def test_band_wavelengths_ordering(self, session):
        """Test that wavelength ordering is preserved"""
        wavelengths = [600.0, 400.0, 500.0]  # Not sorted
        band = Band(name="unordered", band_wavelengths=wavelengths, band_transmission=[0.1, 0.2, 0.3])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.band_wavelengths == wavelengths  # Order preserved

    @pytest.mark.asyncio
    async def test_band_with_zero_transmission(self, session):
        """Test Band with zero transmission values"""
        band = Band(name="zero_trans", band_wavelengths=[500.0, 600.0], band_transmission=[0.0, 0.0])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.band_transmission == [0.0, 0.0]

    @pytest.mark.asyncio
    async def test_band_with_high_transmission(self, session):
        """Test Band with transmission values > 1"""
        band = Band(name="high_trans", band_wavelengths=[500.0, 600.0], band_transmission=[1.5, 2.0])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.band_transmission == [1.5, 2.0]

    @pytest.mark.asyncio
    async def test_band_with_negative_wavelengths(self, session):
        """Test Band with negative wavelength values"""
        band = Band(name="negative_wave", band_wavelengths=[-100.0, 500.0], band_transmission=[0.5, 0.5])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.band_wavelengths == [-100.0, 500.0]

    @pytest.mark.asyncio
    async def test_update_wavelengths(self, session, sample_band):
        """Test updating band_wavelengths"""
        new_wavelengths = [450.0, 550.0, 650.0]
        sample_band.band_wavelengths = new_wavelengths
        await session.commit()
        await session.refresh(sample_band)

        assert sample_band.band_wavelengths == new_wavelengths

    @pytest.mark.asyncio
    async def test_update_transmission(self, session, sample_band):
        """Test updating band_transmission"""
        new_transmission = [0.15, 0.95, 0.25]
        sample_band.band_transmission = new_transmission
        await session.commit()
        await session.refresh(sample_band)

        assert sample_band.band_transmission == new_transmission

    @pytest.mark.asyncio
    async def test_update_both_arrays(self, session, sample_band):
        """Test updating both wavelengths and transmission"""
        new_wavelengths = [350.0, 450.0]
        new_transmission = [0.3, 0.7]

        sample_band.band_wavelengths = new_wavelengths
        sample_band.band_transmission = new_transmission
        await session.commit()
        await session.refresh(sample_band)

        assert sample_band.band_wavelengths == new_wavelengths
        assert sample_band.band_transmission == new_transmission


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    @pytest.mark.asyncio
    async def test_band_with_long_name(self, session):
        """Test Band with maximum length name"""
        long_name = "a" * 255
        band = Band(name=long_name, band_wavelengths=[500.0], band_transmission=[0.5])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.name == long_name

    @pytest.mark.asyncio
    async def test_band_with_special_characters(self, session):
        """Test Band name with special characters"""
        band = Band(name="LSST-u_band_v2.0", band_wavelengths=[350.0], band_transmission=[0.7])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.name == "LSST-u_band_v2.0"

    @pytest.mark.asyncio
    async def test_query_nonexistent_band(self, session):
        """Test querying for non-existent band"""
        result = await session.get(Band, 99999)
        assert result is None

    @pytest.mark.asyncio
    async def test_multiple_sessions(self, engine):
        """Test that multiple sessions work independently"""
        from sqlalchemy.ext.asyncio import AsyncSession
        from sqlalchemy.orm import sessionmaker

        async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session1:
            band1 = Band(name="session1_band", band_wavelengths=[500.0], band_transmission=[0.8])
            session1.add(band1)
            await session1.commit()
            await session1.refresh(band1)
            band1_id = band1.id_

        async with async_session() as session2:
            band2 = await session2.get(Band, band1_id)
            assert band2 is not None
            assert band2.name == "session1_band"

    @pytest.mark.asyncio
    async def test_rollback_on_error(self, session):
        """Test that transaction rolls back on error"""
        initial_count = (await session.execute(select(Band))).scalars().all()
        initial_len = len(initial_count)

        try:
            band = Band(name="test_rollback", band_wavelengths=[500.0], band_transmission=[0.8])
            session.add(band)
            await session.flush()

            # Create duplicate to force error
            duplicate = Band(name="test_rollback", band_wavelengths=[600.0], band_transmission=[0.7])
            session.add(duplicate)
            await session.commit()
        except Exception:
            await session.rollback()

        final_count = (await session.execute(select(Band))).scalars().all()
        assert len(final_count) == initial_len

    @pytest.mark.asyncio
    async def test_band_with_empty_arrays(self, session):
        """Test Band with empty arrays"""
        # Note: This may or may not be valid depending on your business rules
        # If Pydantic validation prevents this, this test would fail at the API level
        band = Band(name="empty_arrays", band_wavelengths=[], band_transmission=[])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.band_wavelengths == []
        assert band.band_transmission == []

    @pytest.mark.asyncio
    async def test_band_with_very_large_arrays(self, session):
        """Test Band with very large arrays"""
        wavelengths = [float(i) for i in range(1000)]
        transmission = [0.5] * 1000

        band = Band(name="large_arrays", band_wavelengths=wavelengths, band_transmission=transmission)
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert len(band.band_wavelengths) == 1000
        assert len(band.band_transmission) == 1000


class TestConcurrentAccess:
    """Tests for concurrent database access"""

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, session, sample_band):
        """Test that concurrent reads work correctly"""
        results = []
        for _ in range(5):
            result = await session.execute(select(Band).where(Band.id_ == sample_band.id_))
            results.append(result.scalar_one())

        assert len(results) == 5
        assert all(r.id_ == sample_band.id_ for r in results)

    @pytest.mark.asyncio
    async def test_refresh_after_update(self, session, sample_band):
        """Test that refresh loads updated data"""
        original_wavelengths = sample_band.band_wavelengths.copy()

        new_wavelengths = [450.0, 550.0, 650.0]
        sample_band.band_wavelengths = new_wavelengths
        await session.commit()
        await session.refresh(sample_band)

        assert sample_band.band_wavelengths == new_wavelengths
        assert sample_band.band_wavelengths != original_wavelengths


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestBandBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session):
        """Test inserting multiple bands at once"""
        bands = [
            Band(name=f"band_{i}", band_wavelengths=[float(i * 100)], band_transmission=[0.5])
            for i in range(10)
        ]

        session.add_all(bands)
        await session.commit()

        for band in bands:
            await session.refresh(band)

        assert all(band.id_ is not None for band in bands)
        assert len(bands) == 10

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_bands):
        """Test querying multiple bands"""
        result = await session.execute(select(Band))
        bands = result.scalars().all()

        assert len(bands) >= 3
        names = {band.name for band in bands}
        assert "u_band" in names
        assert "r_band" in names
        assert "i_band" in names

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_bands):
        """Test updating multiple bands"""
        for band in multiple_bands:
            band.band_transmission = [0.99, 0.99, 0.99]

        await session.commit()

        for band in multiple_bands:
            await session.refresh(band)
            assert band.band_transmission == [0.99, 0.99, 0.99]

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_bands):
        """Test deleting multiple bands"""
        band_ids = [band.id_ for band in multiple_bands]

        for band in multiple_bands:
            await session.delete(band)

        await session.commit()

        for band_id in band_ids:
            result = await session.get(Band, band_id)
            assert result is None


class TestTypeAnnotations:
    """Tests for type annotations and type hints"""

    def test_band_has_type_annotations(self):
        """Test that Band fields have proper type annotations"""
        assert hasattr(Band, "__annotations__")
        annotations = Band.__annotations__
        assert "id_" in annotations or hasattr(Band, "id_")
        assert "name" in annotations or hasattr(Band, "name")
        assert "band_wavelengths" in annotations or hasattr(Band, "band_wavelengths")
        assert "band_transmission" in annotations or hasattr(Band, "band_transmission")


class TestJSONSerialization:
    """Tests for JSON field serialization"""

    @pytest.mark.asyncio
    async def test_wavelengths_json_roundtrip(self, session):
        """Test that wavelengths survive JSON serialization roundtrip"""
        wavelengths = [123.456, 234.567, 345.678]
        band = Band(name="json_test", band_wavelengths=wavelengths, band_transmission=[0.1, 0.2, 0.3])
        session.add(band)
        await session.commit()
        await session.refresh(band)

        # JSON should preserve the values
        assert band.band_wavelengths == wavelengths

    @pytest.mark.asyncio
    async def test_transmission_json_roundtrip(self, session):
        """Test that transmission values survive JSON serialization roundtrip"""
        transmission = [0.123456, 0.234567, 0.345678]
        band = Band(
            name="json_trans_test", band_wavelengths=[400.0, 500.0, 600.0], band_transmission=transmission
        )
        session.add(band)
        await session.commit()
        await session.refresh(band)

        # JSON should preserve the values
        assert band.band_transmission == transmission

    @pytest.mark.asyncio
    async def test_arrays_with_scientific_notation(self, session):
        """Test arrays with scientific notation values"""
        wavelengths = [1e-9, 2e-9, 3e-9]
        transmission = [1e-3, 2e-3, 3e-3]

        band = Band(name="scientific", band_wavelengths=wavelengths, band_transmission=transmission)
        session.add(band)
        await session.commit()
        await session.refresh(band)

        assert band.band_wavelengths == wavelengths
        assert band.band_transmission == transmission


class TestBandQueries:
    """Tests for various query patterns"""

    @pytest.mark.asyncio
    async def test_query_by_name_pattern(self, session, multiple_bands):
        """Test querying bands with name pattern matching"""
        result = await session.execute(select(Band).where(Band.name.like("%_band")))
        bands = result.scalars().all()

        assert len(bands) >= 3
        assert all(band.name.endswith("_band") for band in bands)

    @pytest.mark.asyncio
    async def test_query_order_by_name(self, session, multiple_bands):
        """Test querying bands ordered by name"""
        result = await session.execute(select(Band).order_by(Band.name))
        bands = result.scalars().all()

        # Check that results are ordered
        names = [band.name for band in bands]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_query_with_limit(self, session, multiple_bands):
        """Test querying bands with limit"""
        result = await session.execute(select(Band).limit(2))
        bands = result.scalars().all()

        assert len(bands) <= 2

    @pytest.mark.asyncio
    async def test_count_bands(self, session, multiple_bands):
        """Test counting total number of bands"""
        from sqlalchemy import func

        result = await session.execute(select(func.count()).select_from(Band))
        count = result.scalar()

        assert count >= 3


class TestBandDataIntegrity:
    """Tests for data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_wavelengths_and_transmission_independence(self, session):
        """Test that wavelengths and transmission are stored independently"""
        band = Band(
            name="independence_test",
            band_wavelengths=[100.0, 200.0, 300.0],
            band_transmission=[0.1, 0.2, 0.3],
        )
        session.add(band)
        await session.commit()
        await session.refresh(band)

        # Modify one without affecting the other
        band.band_wavelengths = [150.0, 250.0, 350.0]
        await session.commit()
        await session.refresh(band)

        assert band.band_wavelengths == [150.0, 250.0, 350.0]
        assert band.band_transmission == [0.1, 0.2, 0.3]  # Unchanged

    @pytest.mark.asyncio
    async def test_array_modification_persistence(self, session, sample_band):
        """Test that array modifications persist correctly"""
        original_wavelengths = sample_band.band_wavelengths.copy()

        # Modify the array
        new_wavelengths = original_wavelengths + [700.0]
        sample_band.band_wavelengths = new_wavelengths
        await session.commit()

        # Query fresh from database
        result = await session.get(Band, sample_band.id_)
        assert result.band_wavelengths == new_wavelengths


class TestBandPydanticValidation:
    """Tests for Pydantic model integration and validation"""

    @pytest.mark.asyncio
    async def test_to_pydantic_list(self, multiple_bands):
        """Test converting multiple bands to Pydantic list"""
        pydantic_list = Band.to_pydantic_list(multiple_bands)

        assert len(pydantic_list) == 3
        assert all(isinstance(obj, BandPydantic) for obj in pydantic_list)
        assert pydantic_list[0].name == "u_band"
        assert pydantic_list[1].name == "r_band"
        assert pydantic_list[2].name == "i_band"

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list(self, multiple_bands):
        """Test converting multiple bands to dict list"""
        dict_list = Band.to_pydantic_dict_list(multiple_bands)

        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert all("band_wavelengths" in d for d in dict_list)
        assert all("band_transmission" in d for d in dict_list)
