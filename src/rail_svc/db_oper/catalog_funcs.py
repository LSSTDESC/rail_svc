from pathlib import Path

import anyio
import numpy as np
from pathlib import Path
import qp
import tables_io

from sqlalchemy.ext.asyncio import AsyncSession

from .. import db, rail_funcs
from ..config import config as global_config
from .band import band
from .catalog_band_assoc import catalog_band_assoc
from .catalog_tag import catalog_tag
from .dataset import dataset
from .dataset_assoc import dataset_assoc
from .estimates import estimates
from .estimator import estimator


async def load_catalog_yaml(
    session: AsyncSession, catalog_yaml: Path, filter_dir: Path | None = None
) -> tuple[list[db.Band], list[db.CatalogTag], list[db.CatalogBandAssoc]]:

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

    the_dataset = await dataset.get_row(session, dataset_id)
    archive_dir = Path(await anyio.Path(global_config.storage.archive).absolute())
    return rail_funcs.catalog_funcs.get_catalog_row(archive_dir / the_dataset.path, row)


async def get_estimates_row(
    session: AsyncSession,
    estimates_id: int,
    row: int,
) -> dict[str, np.ndarray]:

    the_estimates = await estimates.get_row(session, estimates_id)
    archive_dir = Path(await anyio.Path(global_config.storage.archive).absolute())
    return rail_funcs.catalog_funcs.get_estimates_row(archive_dir / the_estimates.path, row)


async def get_dataset_and_estimates(
    session: AsyncSession,
    dataset_id: int,
) -> tuple[db.Dataset, list[db.Estimates]]:

    the_dataset = await dataset.get_row(session, dataset_id)
    the_estimates = await estimates.find_by(session, dataset_id=the_dataset.id_)
    return (the_dataset, list(the_estimates))


async def get_data_and_estimates_data(
    session: AsyncSession,
    dataset_id: int,
    row: int,
) -> tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]:

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


def read_single_catalog_slice(
    dataset_path: str | Path,
    the_slice: slice | int | None = None,
) -> dict[str, np.ndarray]:
    """
    Read a slice of data from a single catalog dataset.
    
    This function reads data from a catalog file stored in the archive directory,
    optionally returning only a subset of rows specified by the_slice parameter.
    
    Parameters
    ----------
    dataset_path
        Path to the catalog file
    the_slice
        Slice specification for which rows to read:
        - slice object: read a range of rows (e.g., slice(0, 100))
        - int: read a single row
        - None: read all rows (default)
    
    Returns
    -------
        Dictionary mapping column names to numpy arrays containing the data
    
    Raises
    ------
    FileNotFoundError
        If the dataset file does not exist in the archive directory
    """
    archive_dir = Path(global_config.storage.archive)
    file_path = archive_dir / dataset_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    
    return tables_io.read(file_path, slice_dict=the_slice)


def read_multi_catalog_slice(
    matched_dataset_path: str | Path,
    component_dataset_paths: dict[str, str | Path],
    the_slice: slice | int | None = None,
) -> dict[str, np.ndarray]:
    """
    Read and merge data from multiple matched catalog datasets.
    
    This function reads a matched dataset containing cross-references to multiple
    component datasets, then retrieves and merges the corresponding rows from each
    component dataset into a single dictionary.
    
    Parameters
    ----------
    matched_dataset_path
        Path to file containing match indices/keys that reference rows in component datasets
    component_dataset_paths
        Dictionary mapping keys to paths to datasets. The keys must correspond to
        columns in the matched_dataset that contain indices into each component.
    the_slice
        Slice specification for which rows to read from the matched dataset:
        - slice object: read a range of matched rows
        - int: read a single matched row
        - None: read all matched rows (default)
    
    Returns
    -------
        Dictionary containing merged data from the matched dataset and all
        component datasets. Includes all columns from the match dataset plus
        the columns from component datasets for the matched rows.
    
    Raises
    ------
    FileNotFoundError
        If any dataset file does not exist in the archive directory
    KeyError
        If a component dataset key does not exist in the matched dataset
    
    Examples
    --------
    >>> matched = Dataset(path="matches.hdf5")
    >>> components = {
    ...     "spec_idx": Dataset(path="spectroscopy.hdf5"),
    ...     "phot_idx": Dataset(path="photometry.hdf5")
    ... }
    >>> data = read_multi_catalog_slice(matched, components, slice(0, 100))
    """
    archive_dir = Path(global_config.storage.archive)
    matched_path = archive_dir / matched_dataset_path
    
    if not matched_path.exists():
        raise FileNotFoundError(f"Matched dataset not found: {matched_path}")
    
    match_set = tables_io.read(matched_path, slice_dict=the_slice)
    full_set = match_set.copy()
    
    for key, ds_path in component_dataset_paths.items():
        component_path = archive_dir / ds_path
        
        if not component_path.exists():
            raise FileNotFoundError(f"Component dataset not found: {component_path}")
        
        if key not in match_set:
            raise KeyError(f"Match key '{key}' not found in matched dataset")
        
        component_data = tables_io.read(component_path)
        indexed_data = component_data[match_set[key]]
        full_set.update(indexed_data)
    
    return full_set


def read_estimates_slice(
    estimates_path: str | Path,
    the_slice: slice | int | None = None,
) -> qp.Ensemble:
    """
    Read a slice of probability distribution estimates.
    
    This function reads quantile-parameterized (qp) probability distribution
    ensembles from a file, optionally returning only a subset specified by
    the_slice parameter.
    
    Parameters
    ----------
    estimates
        Path to the qp ensemble file
    the_slice
        Slice specification for which distributions to read:
        - slice object: read a range of distributions (e.g., slice(0, 100))
        - int: read a single distribution
        - None: read all distributions (default)
    
    Returns
    -------
        Ensemble of probability distributions for the requested slice
    
    Raises
    ------
    FileNotFoundError
        If the estimates file does not exist in the archive directory
    
    Notes
    -----
    This function uses the qp library's native reading functionality, which
    may have different parameter naming conventions (read_slice vs slice_dict)
    compared to tables_io.
    """
    archive_dir = Path(global_config.storage.archive)
    file_path = archive_dir / estimates_path
    
    if not file_path.exists():
        raise FileNotFoundError(f"Estimates file not found: {file_path}")
    
    return qp.read(file_path, read_slice=the_slice)


