"""Unit tests for FilterAB table operations — get_create_kwargs FK resolution."""

import pytest

from rail_svc.db_oper.filter_ab import filter_ab


class TestFilterABGetCreateKwargs:
    """Test get_create_kwargs resolves band and sed foreign keys."""

    @pytest.mark.asyncio
    async def test_resolve_by_band_id_and_sed_id(self, session, sample_band, sample_sed):
        kwargs = await filter_ab.get_create_kwargs(
            session,
            name="test_fab",
            redshifts=[0.0, 0.5, 1.0],
            fluxes=[1.5, 2.3, 3.1],
            band_id=sample_band.id_,
            sed_id=sample_sed.id_,
        )

        assert kwargs["name"] == "test_fab"
        assert kwargs["band_id"] == sample_band.id_
        assert kwargs["sed_id"] == sample_sed.id_
        assert kwargs["redshifts"] == [0.0, 0.5, 1.0]
        assert kwargs["fluxes"] == [1.5, 2.3, 3.1]

    @pytest.mark.asyncio
    async def test_resolve_by_band_name_and_sed_name(self, session, sample_band, sample_sed):
        kwargs = await filter_ab.get_create_kwargs(
            session,
            name="test_fab",
            redshifts=[0.0, 0.5],
            fluxes=[1.5, 2.3],
            band_name=sample_band.name,
            sed_name=sample_sed.name,
        )

        assert kwargs["band_id"] == sample_band.id_
        assert kwargs["sed_id"] == sample_sed.id_

    @pytest.mark.asyncio
    async def test_resolve_mixed_id_and_name(self, session, sample_band, sample_sed):
        kwargs = await filter_ab.get_create_kwargs(
            session,
            name="test_fab",
            redshifts=[0.0],
            fluxes=[1.5],
            band_id=sample_band.id_,
            sed_name=sample_sed.name,
        )

        assert kwargs["band_id"] == sample_band.id_
        assert kwargs["sed_id"] == sample_sed.id_

    @pytest.mark.asyncio
    async def test_nonexistent_band_name_raises(self, session, sample_sed):
        with pytest.raises(Exception):
            await filter_ab.get_create_kwargs(
                session,
                name="test_fab",
                redshifts=[0.0],
                fluxes=[1.5],
                band_name="nonexistent_band",
                sed_id=sample_sed.id_,
            )

    @pytest.mark.asyncio
    async def test_nonexistent_sed_name_raises(self, session, sample_band):
        with pytest.raises(Exception):
            await filter_ab.get_create_kwargs(
                session,
                name="test_fab",
                redshifts=[0.0],
                fluxes=[1.5],
                band_id=sample_band.id_,
                sed_name="nonexistent_sed",
            )

    @pytest.mark.asyncio
    async def test_extra_kwargs_passed_through(self, session, sample_band, sample_sed):
        kwargs = await filter_ab.get_create_kwargs(
            session,
            name="test_fab",
            redshifts=[0.0],
            fluxes=[1.5],
            band_id=sample_band.id_,
            sed_id=sample_sed.id_,
            some_extra_field="extra_value",
        )

        assert kwargs["some_extra_field"] == "extra_value"
