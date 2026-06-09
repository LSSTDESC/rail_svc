"""Shared test fixtures for database tests"""

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from rail_svc.db import (Algorithm, Band, CatalogBandAssoc, CatalogTag,
                         Dataset, DatasetAssoc, Estimates, Estimator, Model)
from rail_svc.db.base import Base

# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture
async def engine():
    """Create an in-memory SQLite database engine for testing."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


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
    """Create a sample algorithm for testing."""
    algo = Algorithm(name="test_algorithm", class_name="rail.estimation.SOMEstimator")
    session.add(algo)
    await session.commit()
    await session.refresh(algo)
    return algo


@pytest.fixture
async def multiple_algorithms(session):
    """Create multiple algorithms for testing."""
    algorithms = [
        Algorithm(name="knn", class_name="sklearn.neighbors.KNeighborsClassifier"),
        Algorithm(name="random_forest", class_name="sklearn.ensemble.RandomForestClassifier"),
        Algorithm(name="xgboost", class_name="xgboost.XGBClassifier"),
    ]
    for algo in algorithms:
        session.add(algo)
    await session.commit()

    for algo in algorithms:
        await session.refresh(algo)

    return algorithms


# ============================================================================
# Band Fixtures
# ============================================================================


@pytest.fixture
async def sample_band(session):
    """Create a sample band for testing."""
    band = Band(name="g_band", band_wavelengths=[400.0, 500.0, 600.0], band_transmission=[0.1, 0.9, 0.2])
    session.add(band)
    await session.commit()
    await session.refresh(band)
    return band


@pytest.fixture
async def multiple_bands(session):
    """Create multiple bands for testing."""
    bands = [
        Band(name="u_band", band_wavelengths=[300.0, 350.0, 400.0], band_transmission=[0.05, 0.85, 0.15]),
        Band(name="r_band", band_wavelengths=[550.0, 650.0, 750.0], band_transmission=[0.2, 0.95, 0.25]),
        Band(name="i_band", band_wavelengths=[650.0, 750.0, 850.0], band_transmission=[0.15, 0.9, 0.2]),
    ]
    for band in bands:
        session.add(band)
    await session.commit()

    for band in bands:
        await session.refresh(band)

    return bands


# ============================================================================
# CatalogTag Fixtures
# ============================================================================


@pytest.fixture
async def sample_catalog_tag(session):
    """Create a sample catalog tag for testing."""
    tag = CatalogTag(name="lsst_dp02")
    session.add(tag)
    await session.commit()
    await session.refresh(tag)
    return tag


@pytest.fixture
async def multiple_catalog_tags(session):
    """Create multiple catalog tags for testing."""
    tags = [
        CatalogTag(name="roman"),
        CatalogTag(name="rubin"),
        CatalogTag(name="hsc_pdr3"),
    ]
    for tag in tags:
        session.add(tag)
    await session.commit()

    for tag in tags:
        await session.refresh(tag)

    return tags


# ============================================================================
# CatalogBandAssoc Fixtures
# ============================================================================


@pytest.fixture
async def sample_catalog_band_assoc(session, sample_catalog_tag, sample_band):
    """Create a sample catalog band association for testing."""
    assoc = CatalogBandAssoc(
        catalog_tag_id=sample_catalog_tag.id_,
        band_id=sample_band.id_,
        mag_column_name="g_mag",
        mag_err_column_name="g_mag_err",
    )
    session.add(assoc)
    await session.commit()
    await session.refresh(assoc)
    return assoc


@pytest.fixture
async def multiple_catalog_band_assocs(session, sample_catalog_tag, multiple_bands):
    """Create multiple catalog band associations for testing."""
    assocs = [
        CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=multiple_bands[0].id_,
            mag_column_name="r_mag",
            mag_err_column_name="r_mag_err",
        ),
        CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=multiple_bands[1].id_,
            mag_column_name="i_mag",
            mag_err_column_name="i_mag_err",
        ),
        CatalogBandAssoc(
            catalog_tag_id=sample_catalog_tag.id_,
            band_id=multiple_bands[2].id_,
            mag_column_name="z_mag",
            mag_err_column_name="z_mag_err",
        ),
    ]
    for assoc in assocs:
        session.add(assoc)
    await session.commit()

    for assoc in assocs:
        await session.refresh(assoc)

    return assocs


# ============================================================================
# Dataset Fixtures
# ============================================================================


@pytest.fixture
async def sample_dataset(session, sample_catalog_tag):
    """Create a sample dataset for testing."""
    dataset = Dataset(
        name="test_dataset",
        n_objects=10000,
        path="/data/test_dataset.hdf5",
        is_collection=False,
        catalog_tag_id=sample_catalog_tag.id_,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


@pytest.fixture
async def multiple_datasets(session, sample_catalog_tag):
    """Create multiple datasets for testing."""
    datasets = [
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
    ]
    for dataset in datasets:
        session.add(dataset)
    await session.commit()

    for dataset in datasets:
        await session.refresh(dataset)

    return datasets


# ============================================================================
# DatasetAssoc Fixtures
# ============================================================================


@pytest.fixture
async def matched_dataset(session, sample_catalog_tag):
    """Create a matched dataset for testing."""
    dataset = Dataset(
        name="matched_catalog",
        n_objects=10000,
        path="/data/matched.hdf5",
        is_collection=False,
        catalog_tag_id=sample_catalog_tag.id_,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


@pytest.fixture
async def component_dataset_1(session, sample_catalog_tag):
    """Create first component dataset for testing."""
    dataset = Dataset(
        name="gaia_dr3",
        n_objects=5000,
        path="/data/gaia.hdf5",
        is_collection=False,
        catalog_tag_id=sample_catalog_tag.id_,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


@pytest.fixture
async def component_dataset_2(session, sample_catalog_tag):
    """Create second component dataset for testing."""
    dataset = Dataset(
        name="sdss_dr17",
        n_objects=6000,
        path="/data/sdss.hdf5",
        is_collection=False,
        catalog_tag_id=sample_catalog_tag.id_,
    )
    session.add(dataset)
    await session.commit()
    await session.refresh(dataset)
    return dataset


@pytest.fixture
async def sample_dataset_assoc(session, matched_dataset, component_dataset_1):
    """Create a sample dataset association for testing."""
    assoc = DatasetAssoc(
        name="gaia_to_matched",
        matched_dataset_id=matched_dataset.id_,
        component_dataset_id=component_dataset_1.id_,
    )
    session.add(assoc)
    await session.commit()
    await session.refresh(assoc)
    return assoc


@pytest.fixture
async def multiple_dataset_assocs(session, matched_dataset, component_dataset_1, component_dataset_2):
    """Create multiple dataset associations for testing."""
    assocs = [
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
    ]
    for assoc in assocs:
        session.add(assoc)
    await session.commit()

    for assoc in assocs:
        await session.refresh(assoc)

    return assocs


# ============================================================================
# Estimates Fixtures
# ============================================================================


@pytest.fixture
async def sample_estimates(session, sample_dataset, sample_estimator):
    """Create a sample estimates for testing."""
    estimates = Estimates(
        name="test_estimates",
        n_objects=10000,
        path="/results/test_estimates.hdf5",
        dataset_id=sample_dataset.id_,
        estimator_id=sample_estimator.id_,
    )
    session.add(estimates)
    await session.commit()
    await session.refresh(estimates)
    return estimates


@pytest.fixture
async def multiple_estimates(session, sample_dataset, sample_estimator):
    """Create multiple estimates for testing."""
    estimates_list = [
        Estimates(
            name="estimates_v1",
            n_objects=5000,
            path="/results/estimates_v1.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        ),
        Estimates(
            name="estimates_v2",
            n_objects=5000,
            path="/results/estimates_v2.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        ),
        Estimates(
            name="estimates_v3",
            n_objects=5000,
            path="/results/estimates_v3.hdf5",
            dataset_id=sample_dataset.id_,
            estimator_id=sample_estimator.id_,
        ),
    ]
    for est in estimates_list:
        session.add(est)
    await session.commit()

    for est in estimates_list:
        await session.refresh(est)

    return estimates_list


# ============================================================================
# Estimator Fixtures
# ============================================================================


@pytest.fixture
async def sample_estimator(session, sample_model):
    """Create a sample estimator for testing."""
    estimator = Estimator(
        name="knn_estimator", config={"n_neighbors": 5, "weights": "distance"}, model_id=sample_model.id_
    )
    session.add(estimator)
    await session.commit()
    await session.refresh(estimator)
    return estimator


@pytest.fixture
async def multiple_estimators(session, sample_model):
    """Create multiple estimators for testing."""
    estimators = [
        Estimator(name="estimator_v1", config={"n_neighbors": 3}, model_id=sample_model.id_),
        Estimator(name="estimator_v2", config={"n_neighbors": 5}, model_id=sample_model.id_),
        Estimator(name="estimator_v3", config={"n_neighbors": 10}, model_id=sample_model.id_),
    ]
    for est in estimators:
        session.add(est)
    await session.commit()

    for est in estimators:
        await session.refresh(est)

    return estimators


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
async def sample_model(session, sample_algorithm, sample_catalog_tag):
    """Create a sample model for testing."""
    model = Model(
        name="knn_model",
        path="/models/knn.pkl",
        algo_id=sample_algorithm.id_,
        catalog_tag_id=sample_catalog_tag.id_,
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


@pytest.fixture
async def multiple_models(session, sample_algorithm, sample_catalog_tag):
    """Create multiple models for testing."""
    models = [
        Model(
            name="model_v1",
            path="/models/model_v1.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
        Model(
            name="model_v2",
            path="/models/model_v2.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
        Model(
            name="model_v3",
            path="/models/model_v3.pkl",
            algo_id=sample_algorithm.id_,
            catalog_tag_id=sample_catalog_tag.id_,
        ),
    ]
    for model in models:
        session.add(model)
    await session.commit()

    for model in models:
        await session.refresh(model)

    return models
