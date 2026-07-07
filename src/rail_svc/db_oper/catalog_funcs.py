"""Catalog-level database operations.

Composite operations that span multiple tables to manage catalogs,
datasets, and their associated estimates. These functions coordinate
band, catalog_tag, catalog_band_assoc, dataset, dataset_assoc, estimates,
and estimator operations to implement higher-level workflows.
"""

from pathlib import Path

import anyio
import numpy as np
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from .. import db, rail_funcs
from macon.config import config as global_config
from .band import band
from .catalog_band_assoc import catalog_band_assoc
from .catalog_tag import catalog_tag
from .dataset import dataset
from .dataset_assoc import dataset_assoc
from .estimates import estimates
from .estimator import estimator
from .filter_ab import filter_ab
from .sed import sed


async def load_catalog_yaml(
    session: AsyncSession, catalog_yaml: Path, filter_dir: Path | None = None
) -> tuple[list[db.Band], list[db.CatalogTag], list[db.CatalogBandAssoc]]:
    """Load a catalog definition from a YAML file and persist to the database.

    Parses the YAML to extract band, catalog tag, and band-association
    definitions, then creates the corresponding rows in the database.

    Parameters
    ----------
    session : AsyncSession
        Active database session (must be within a transaction).
    catalog_yaml : Path
        Path to the catalog YAML definition file.
    filter_dir : Path | None
        Optional directory containing filter transmission curve files
        referenced by the YAML. If None, paths in the YAML are used as-is.

    Returns
    -------
    tuple[list[db.Band], list[db.CatalogTag], list[db.CatalogBandAssoc]]
        The created Band, CatalogTag, and CatalogBandAssoc ORM objects.
    """
    band_creates, catalog_tag_creates, catalog_band_assoc_creates = (
        rail_funcs.catalog_funcs.load_catalog_yaml(catalog_yaml, filter_dir)
    )

    bands = await band.create_rows(session, [band_.model_dump() for band_ in band_creates])
    catalog_tags = await catalog_tag.create_rows(
        session, [catalog_tag_.model_dump() for catalog_tag_ in catalog_tag_creates]
    )
    catalog_band_assocs = await catalog_band_assoc.create_rows(
        session, [catalog_band_assoc_.model_dump() for catalog_band_assoc_ in catalog_band_assoc_creates]
    )

    return (bands, catalog_tags, catalog_band_assocs)


async def get_catalog_row(
    session: AsyncSession,
    dataset_id: int,
    row: int,
) -> dict[str, np.ndarray]:
    """Read a single row from a dataset's backing file.

    Looks up the dataset path from the database, resolves it against the
    configured archive directory, and reads the specified row.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    dataset_id : int
        Primary key of the dataset to read from.
    row : int
        Zero-based row index to extract.

    Returns
    -------
    dict[str, np.ndarray]
        Column name to value mapping for the requested row.
    """
    the_dataset = await dataset.get_row(session, dataset_id)
    archive_dir = Path(await anyio.Path(global_config.storage.archive).absolute())
    return rail_funcs.catalog_funcs.get_catalog_row(archive_dir / the_dataset.path, row)


async def get_estimates_row(
    session: AsyncSession,
    estimates_id: int,
    row: int,
) -> dict[str, np.ndarray]:
    """Read a single row from an estimates file.

    Looks up the estimates path from the database, resolves it against the
    configured archive directory, and reads the specified row.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    estimates_id : int
        Primary key of the Estimates record.
    row : int
        Zero-based row index to extract.

    Returns
    -------
    dict[str, np.ndarray]
        Column name to value mapping for the requested estimates row.
    """
    the_estimates = await estimates.get_row(session, estimates_id)
    archive_dir = Path(await anyio.Path(global_config.storage.archive).absolute())
    return rail_funcs.catalog_funcs.get_estimates_row(archive_dir / the_estimates.path, row)


async def get_dataset_and_estimates(
    session: AsyncSession,
    dataset_id: int,
) -> tuple[db.Dataset, list[db.Estimates]]:
    """Retrieve a dataset and all its associated estimates.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    dataset_id : int
        Primary key of the dataset.

    Returns
    -------
    tuple[db.Dataset, list[db.Estimates]]
        The Dataset ORM object and a list of all Estimates rows
        linked to it via ``dataset_id``.
    """
    the_dataset = await dataset.get_row(session, dataset_id)
    the_estimates = await estimates.find_by(session, dataset_id=the_dataset.id_)
    return (the_dataset, list(the_estimates))


async def get_data_and_estimates_data(
    session: AsyncSession,
    dataset_id: int,
    row: int,
) -> tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]:
    """Read catalog data and all estimate results for a single object.

    Retrieves the photometric data row from the dataset file and the
    corresponding photo-z estimate from each associated estimates file,
    keyed by estimator name.

    Parameters
    ----------
    session : AsyncSession
        Active database session.
    dataset_id : int
        Primary key of the dataset.
    row : int
        Zero-based row index of the object to retrieve.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]
        A two-element tuple where the first element is the catalog data
        for the row, and the second is a dict mapping estimator name to
        the qp.Ensemble photo-z estimate for that row.
    """
    the_dataset, the_estimates = await get_dataset_and_estimates(session, dataset_id)
    archive_dir = Path(await anyio.Path(global_config.storage.archive).absolute())
    data = rail_funcs.catalog_funcs.get_catalog_row(archive_dir / the_dataset.path, row)
    the_estimates_dict: dict[str, qp.Ensemble] = {}
    for the_estimates_ in the_estimates:
        the_estimator = await estimator.get_row(session, the_estimates_.estimator_id)
        the_estimates_dict[the_estimator.name] = rail_funcs.catalog_funcs.get_estimates_row(
            archive_dir / the_estimates_.path, row
        )
    return (data, the_estimates_dict)


async def create_matched_dataset(
    session: AsyncSession,
    matched_dataset_name: str,
    catalog_tag_name: str,
    component_dataset_names: list[str],
    path: str | None,
    n_objects: int,
) -> tuple[db.Dataset, list[db.DatasetAssoc]]:
    """Create a matched (collection) dataset from component datasets.

    Creates a new Dataset marked as a collection, then creates
    DatasetAssoc rows linking it to each named component dataset.

    Parameters
    ----------
    session : AsyncSession
        Active database session (must be within a transaction).
    matched_dataset_name : str
        Name for the new matched dataset.
    catalog_tag_name : str
        Name of the catalog tag to associate with the matched dataset.
    component_dataset_names : list[str]
        Names of existing datasets to include as components.
    path : str | None
        Optional file path for the matched dataset. May be None if
        the collection is virtual.
    n_objects : int
        Total number of objects in the matched dataset.

    Returns
    -------
    tuple[db.Dataset, list[db.DatasetAssoc]]
        The created matched Dataset and the list of DatasetAssoc rows
        linking it to its components.
    """
    the_matched_dataset = await dataset.create_row(
        session,
        name=matched_dataset_name,
        catalog_tag_name=catalog_tag_name,
        path=path,
        n_objects=n_objects,
        is_collection=True,
        validate_file=False,
    )

    assoc_list: list[db.DatasetAssoc] = []
    for component_name in component_dataset_names:
        assoc_list.append(
            await dataset_assoc.create_row(
                session,
                name=f"{matched_dataset_name}_{component_name}",
                matched_dataset_name=matched_dataset_name,
                component_dataset_name=component_name,
            )
        )
    return the_matched_dataset, assoc_list


async def load_seds(
    session: AsyncSession,
    sed_dir: Path | str,
    names: list[str] | None = None,
    names_file: Path | str | None = None,
) -> list[db.Sed]:
    """Load SED files from a directory and persist to the database.

    Reads two-column text files (.sed) from a directory and creates
    corresponding Sed rows in the database.

    Parameters
    ----------
    session : AsyncSession
        Active database session (must be within a transaction).
    sed_dir : Path or str
        Directory containing .sed text files.
    names : list[str] | None
        Optional list of SED names (without extension) to load.
    names_file : Path or str | None
        Optional path to a text file listing SED filenames to load.

    Returns
    -------
    list[db.Sed]
        The created Sed ORM objects.
    """
    sed_creates = rail_funcs.catalog_funcs.make_sed_create_models(sed_dir, names=names, names_file=names_file)
    seds = await sed.create_rows(session, [s.model_dump() for s in sed_creates])
    return seds


async def load_filter_abs(
    session: AsyncSession,
    filter_ab_dir: Path | str,
    names: list[str] | None = None,
) -> list[db.FilterAB]:
    """Load FilterAB files from a directory and persist to the database.

    Reads two-column text files ({sed}.{band}.AB) from a directory and
    creates corresponding FilterAB rows in the database. Band and Sed
    names are extracted from the filename convention.

    Parameters
    ----------
    session : AsyncSession
        Active database session (must be within a transaction).
    filter_ab_dir : Path or str
        Directory containing .AB text files following the naming
        convention ``{sed}.{band}.AB``.
    names : list[str] | None
        Optional list of file stems (e.g. ``["elliptical.g"]``) to load.
        If None, all .AB files in the directory are loaded.

    Returns
    -------
    list[db.FilterAB]
        The created FilterAB ORM objects.
    """
    fab_creates = rail_funcs.catalog_funcs.make_filter_ab_create_models(filter_ab_dir, names=names)
    filter_abs_list = await filter_ab.create_rows(session, [f.model_dump() for f in fab_creates])
    return filter_abs_list
