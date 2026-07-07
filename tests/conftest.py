"""Shared test fixtures for database tests"""

import pytest
from macon.db.base import Base
from macon.db.session import close_db, init_db
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker

from rail_svc.db import (
    Algorithm,
    Band,
    CatalogBandAssoc,
    CatalogTag,
    Dataset,
    DatasetAssoc,
    Estimates,
    Estimator,
    FilterAB,
    Model,
    Sed,
)


async def create(session, obj):
    """Add, commit, and refresh a single ORM object."""
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return obj


async def create_all(session, objs):
    """Add, commit, and refresh multiple ORM objects."""
    session.add_all(objs)
    await session.commit()
    for obj in objs:
        await session.refresh(obj)
    return objs


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
async def engine():
    """Create an in-memory SQLite database engine for testing."""
    init_db("sqlite+aiosqlite:///:memory:", echo=False)

    from macon.db import session as macon_session

    engine = macon_session._engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await close_db()


@pytest.fixture
async def session(engine):
    """Create a database session for testing."""
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


# ============================================================================
# Algorithm Fixtures
# ============================================================================


@pytest.fixture
async def sample_algorithm(session):
    return await create(session, Algorithm(name="test_algorithm", class_name="rail.estimation.SOMEstimator"))


@pytest.fixture
async def multiple_algorithms(session):
    return await create_all(
        session,
        [
            Algorithm(name="knn", class_name="sklearn.neighbors.KNeighborsClassifier"),
            Algorithm(name="random_forest", class_name="sklearn.ensemble.RandomForestClassifier"),
            Algorithm(name="xgboost", class_name="xgboost.XGBClassifier"),
        ],
    )


# ============================================================================
# Band Fixtures
# ============================================================================


@pytest.fixture
async def sample_band(session):
    return await create(
        session,
        Band(name="g_band", band_wavelengths=[400.0, 500.0, 600.0], band_transmission=[0.1, 0.9, 0.2]),
    )


@pytest.fixture
async def multiple_bands(session):
    return await create_all(
        session,
        [
            Band(name="u_band", band_wavelengths=[300.0, 350.0, 400.0], band_transmission=[0.05, 0.85, 0.15]),
            Band(name="r_band", band_wavelengths=[550.0, 650.0, 750.0], band_transmission=[0.2, 0.95, 0.25]),
            Band(name="i_band", band_wavelengths=[650.0, 750.0, 850.0], band_transmission=[0.15, 0.9, 0.2]),
        ],
    )


# ============================================================================
# CatalogTag Fixtures
# ============================================================================


@pytest.fixture
async def sample_catalog_tag(session):
    return await create(session, CatalogTag(name="lsst_dp02"))


@pytest.fixture
async def multiple_catalog_tags(session):
    return await create_all(
        session,
        [
            CatalogTag(name="roman"),
            CatalogTag(name="rubin"),
            CatalogTag(name="hsc_pdr3"),
        ],
    )


# ============================================================================
# CatalogBandAssoc Fixtures
# ============================================================================


@pytest.fixture
async def sample_catalog_band_assoc(session, sample_catalog_tag, sample_band):
    return await create(
        session,
        CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=sample_band.id_,
            mag_column_name="g_mag",
            mag_err_column_name="g_mag_err",
        ),
    )


@pytest.fixture
async def multiple_catalog_band_assocs(session, sample_catalog_tag, multiple_bands):
    return await create_all(
        session,
        [
            CatalogBandAssoc(
                catalog_tag_id=sample_catalog_tag.id_,
                band_id=multiple_bands[i].id_,
                mag_column_name=f"{name}_mag",
                mag_err_column_name=f"{name}_mag_err",
            )
            for i, name in enumerate(["r", "i", "z"])
        ],
    )


# ============================================================================
# Dataset Fixtures
# ============================================================================


@pytest.fixture
async def sample_dataset(session, sample_catalog_tag):
    return await create(
        session,
        Dataset(
            name="test_dataset",
            n_objects=10000,
            path="/data/test_dataset.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
    )


@pytest.fixture
async def multiple_datasets(session, sample_catalog_tag):
    return await create_all(
        session,
        [
            Dataset(
                name="photometric_data",
                n_objects=50000,
                path="/data/photometric.hdf5",
                is_collection=False,
                catalog_tag_id=sample_catalog_tag.id_,
            ),
            Dataset(
                name="spectroscopic_data",
                n_objects=5000,
                path="/data/spectroscopic.hdf5",
                is_collection=False,
                catalog_tag_id=sample_catalog_tag.id_,
            ),
            Dataset(
                name="combined_collection",
                n_objects=55000,
                path="/data/combined/",
                is_collection=True,
                catalog_tag_id=sample_catalog_tag.id_,
            ),
        ],
    )


# ============================================================================
# DatasetAssoc Fixtures
# ============================================================================


@pytest.fixture
async def matched_dataset(session, sample_catalog_tag):
    return await create(
        session,
        Dataset(
            name="matched_catalog",
            n_objects=10000,
            path="/data/matched.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
    )


@pytest.fixture
async def component_dataset_1(session, sample_catalog_tag):
    return await create(
        session,
        Dataset(
            name="gaia_dr3",
            n_objects=5000,
            path="/data/gaia.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
    )


@pytest.fixture
async def component_dataset_2(session, sample_catalog_tag):
    return await create(
        session,
        Dataset(
            name="sdss_dr17",
            n_objects=6000,
            path="/data/sdss.hdf5",
            is_collection=False,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
    )


@pytest.fixture
async def sample_dataset_assoc(session, matched_dataset, component_dataset_1):
    return await create(
        session,
        DatasetAssoc(
            name="gaia_to_matched",
            matched_dataset_id=matched_dataset.id_,
            component_dataset_id=component_dataset_1.id_,
        ),
    )


@pytest.fixture
async def multiple_dataset_assocs(session, matched_dataset, component_dataset_1, component_dataset_2):
    return await create_all(
        session,
        [
            DatasetAssoc(
                name="gaia_to_match",
                matched_dataset_id=matched_dataset.id_,
                component_dataset_id=component_dataset_1.id_,
            ),
            DatasetAssoc(
                name="sdss_to_match",
                matched_dataset_id=matched_dataset.id_,
                component_dataset_id=component_dataset_2.id_,
            ),
        ],
    )


# ============================================================================
# Estimates Fixtures
# ============================================================================


@pytest.fixture
async def sample_estimates(session, sample_dataset, sample_estimator):
    return await create(
        session,
        Estimates(
            name="test_estimates",
            n_objects=10000,
            path="/results/test_estimates.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        ),
    )


@pytest.fixture
async def multiple_estimates(session, sample_dataset, sample_estimator):
    return await create_all(
        session,
        [
            Estimates(
                name=f"estimates_v{i}",
                n_objects=5000,
                path=f"/results/estimates_v{i}.hdf5",
                dataset_id=sample_dataset.id_,
                estimator_id=sample_estimator.id_,
            )
            for i in range(1, 4)
        ],
    )


# ============================================================================
# Estimator Fixtures
# ============================================================================


@pytest.fixture
async def sample_estimator(session, sample_model):
    return await create(
        session,
        Estimator(
            name="knn_estimator",
            config={"n_neighbors": 5, "weights": "distance"},
            model_id=sample_model.id_,
        ),
    )


@pytest.fixture
async def multiple_estimators(session, sample_model):
    return await create_all(
        session,
        [
            Estimator(name=f"estimator_v{i}", config={"n_neighbors": n}, model_id=sample_model.id_)
            for i, n in [(1, 3), (2, 5), (3, 10)]
        ],
    )


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
async def sample_model(session, sample_algorithm, sample_catalog_tag):
    return await create(
        session,
        Model(
            name="knn_model",
            path="/models/knn.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
    )


@pytest.fixture
async def multiple_models(session, sample_algorithm, sample_catalog_tag):
    return await create_all(
        session,
        [
            Model(
                name=f"model_v{i}",
                path=f"/models/model_v{i}.pkl",
                algo_id=sample_algorithm.id_,
                catalog_tag_id=sample_catalog_tag.id_,
            )
            for i in range(1, 4)
        ],
    )


# ============================================================================
# Sed Fixtures
# ============================================================================


@pytest.fixture
async def sample_sed(session):
    return await create(
        session,
        Sed(
            name="elliptical_01",
            sed_wavelengths=[300.0, 400.0, 500.0, 600.0, 700.0],
            sed_values=[0.1, 0.3, 0.5, 0.4, 0.2],
        ),
    )


@pytest.fixture
async def multiple_seds(session):
    return await create_all(
        session,
        [
            Sed(
                name="spiral_01",
                sed_wavelengths=[300.0, 400.0, 500.0, 600.0],
                sed_values=[0.2, 0.5, 0.8, 0.6],
            ),
            Sed(
                name="starburst_01",
                sed_wavelengths=[300.0, 400.0, 500.0, 600.0],
                sed_values=[0.8, 0.6, 0.3, 0.1],
            ),
            Sed(
                name="quiescent_01",
                sed_wavelengths=[300.0, 400.0, 500.0, 600.0],
                sed_values=[0.05, 0.1, 0.3, 0.5],
            ),
        ],
    )


# ============================================================================
# FilterAB Fixtures
# ============================================================================


@pytest.fixture
async def sample_filter_ab(session, sample_band, sample_sed):
    return await create(
        session,
        FilterAB(
            name="g_band_elliptical_01",
            band_id=sample_band.id_,
            sed_id=sample_sed.id_,
            redshifts=[0.0, 0.5, 1.0, 1.5, 2.0],
            fluxes=[1.0, 0.8, 0.5, 0.3, 0.1],
        ),
    )


@pytest.fixture
async def multiple_filter_abs(session, sample_band, multiple_seds):
    return await create_all(
        session,
        [
            FilterAB(
                name=f"g_band_{multiple_seds[i].name}",
                band_id=sample_band.id_,
                sed_id=multiple_seds[i].id_,
                redshifts=[0.0, 0.5, 1.0],
                fluxes=fluxes,
            )
            for i, fluxes in enumerate([[1.2, 0.9, 0.4], [2.0, 1.5, 0.8], [0.5, 0.3, 0.1]])
        ],
    )
