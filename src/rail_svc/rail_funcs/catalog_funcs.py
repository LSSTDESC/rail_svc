import logging
from pathlib import Path

import numpy as np
import qp
import tables_io
from rail.utils import catalog_utils

from macon.common import unexpected
from macon.config import config as global_config
from ..models import BandCreate, CatalogBandAssocCreate, CatalogTagCreate, FilterABCreate, SedCreate

logger = logging.getLogger(__name__)


def extract_padded_non_zeros(array: np.ndarray) -> np.ndarray:
    """
    Extract non-zero elements from a 2D array with one-element padding.

    This function identifies the range of rows where the second column contains
    non-zero values, then extracts that range with one additional row on each
    side for padding (if available).

    Parameters
    ----------
    array : np.ndarray
        A 2D numpy array with shape (n, 2). The second column (array[:, 1])
        is used to identify non-zero values.

    Returns
    -------
    np.ndarray
        A slice of the input array containing non-zero values with padding.
        Returns an empty (0, 2) array if input is invalid or all zeros.

    Notes
    -----
    The padding helps preserve context around the non-zero data range.
    If the non-zero range starts at index 0, no padding is added at the start.
    Similarly, no padding is added at the end if the range extends to the last element.

    Examples
    --------
    >>> arr = np.array([[1, 0], [2, 0], [3, 5], [4, 10], [5, 0], [6, 0]])
    >>> result = extract_padded_non_zeros(arr)
    >>> # Returns rows from index 1 to 5 (includes one zero row on each side)
    """
    # Input validation
    if array.ndim != 2:
        logger.error(f"Expected 2D array, got {array.ndim}D array")
        return np.empty((0, 2))

    if array.shape[1] != 2:
        logger.error(f"Expected array with 2 columns, got {array.shape[1]} columns")
        return np.empty((0, 2))

    if len(array) == 0:
        logger.warning("Received empty array")
        return np.empty((0, 2))

    # Extract values and identify non-zero entries
    vals = array[:, 1]
    non_zero = vals != 0

    # Handle case where all values are zero
    if not np.any(non_zero):
        logger.warning("All values in second column are zero")
        return np.empty((0, 2))

    # Find first and last non-zero indices
    first_nonzero = np.argmax(non_zero)
    last_nonzero = len(array) - np.argmax(non_zero[::-1]) - 1

    # Add padding (one element on each side, if available)
    padded_start = max(first_nonzero - 1, 0)
    padded_end = min(last_nonzero + 2, len(array))  # +2 because slice end is exclusive

    the_slice = slice(padded_start, padded_end)
    result = array[the_slice]

    logger.debug(
        f"Extracted {len(result)} rows from original {len(array)} rows "
        f"(non-zero range: {first_nonzero} to {last_nonzero})"
    )

    return result


def read_band_res_file(band_name: str, filter_dir: Path | str | None = None) -> np.ndarray:
    """
    Read a band response file and extract non-zero data with padding.

    Parameters
    ----------
    band_name : str
        Name of the band (e.g., 'u', 'g', 'r', 'i', 'z'). Used to construct
        the filename as '{band_name}.res'.
    filter_dir : Path or None, optional
        Directory containing filter response files. If None, uses the default
        RAIL examples data directory.

    Returns
    -------
    np.ndarray
        A 2D array with shape (n, 2) containing wavelengths and transmission values.
        Empty (0, 2) array if the file cannot be read or contains no valid data.

    Raises
    ------
    FileNotFoundError
        If the band response file does not exist.
    ValueError
        If the file contains invalid data format.

    Notes
    -----
    The .res files are expected to contain two columns: wavelength and transmission.
    Leading and trailing rows with zero transmission are removed, with one-row padding.
    """
    # Input validation
    if not isinstance(band_name, str) or not band_name.strip():
        logger.error(f"Invalid band_name: {band_name}")
        raise ValueError("band_name must be a non-empty string")

    # Determine filter directory
    if filter_dir is None:
        try:
            filter_dir = Path(catalog_utils.find_rail_file("examples_data/estimation_data/data/FILTER"))
        except Exception as uexc:
            logger.error(f"Failed to find default filter directory: {uexc}")
            raise

    if not isinstance(filter_dir, Path):  # pragma: no cover
        filter_dir = Path(filter_dir)

    if not filter_dir.exists():
        logger.error(f"Filter directory does not exist: {filter_dir}")
        raise FileNotFoundError(f"Filter directory not found: {filter_dir}")

    # Construct file path
    file_path = filter_dir / f"{band_name}.res"

    if unexpected(not file_path.exists()):
        logger.error(f"Band response file not found: {file_path}")
        raise FileNotFoundError(f"Band file not found: {file_path}")

    logger.info(f"Reading band response file: {file_path}")

    try:
        full_array = np.loadtxt(file_path)
    except Exception as e:
        logger.error(f"Failed to load band file {file_path}: {e}")
        raise ValueError(f"Invalid data format in {file_path}") from e

    # Validate loaded data
    if full_array.ndim != 2 or full_array.shape[1] != 2:
        logger.error(f"Expected 2-column data in {file_path}, got shape {full_array.shape}")
        raise ValueError(f"Invalid data shape in {file_path}")

    return extract_padded_non_zeros(full_array)


def make_band_create_model(band_name: str, filter_dir: Path | str | None) -> BandCreate:
    """
    Create a BandCreate model from a band response file.

    Parameters
    ----------
    band_name : str
        Name of the photometric band.
    filter_dir : Path or None
        Directory containing filter response files. If None, uses default location.

    Returns
    -------
    BandCreate
        A model containing the band name, wavelengths, and transmission values.
        If the file cannot be read, returns a model with empty wavelength and
        transmission lists.

    Notes
    -----
    This function handles errors gracefully by logging warnings and returning
    empty data rather than raising exceptions, allowing the system to continue
    operation even when some band files are missing.
    """
    # Input validation
    if not isinstance(band_name, str) or not band_name.strip():
        logger.error(f"Invalid band_name: {band_name}")
        raise ValueError("band_name must be a non-empty string")

    try:
        band_data = read_band_res_file(band_name, filter_dir)
    except FileNotFoundError:
        logger.warning(
            f"Filter response file not found for band '{band_name}'. Creating band with empty data."
        )
        band_data = np.empty((0, 2))
    except ValueError as e:
        logger.warning(
            f"Invalid data format in filter file for band '{band_name}': {e}. Creating band with empty data."
        )
        band_data = np.empty((0, 2))
    except Exception as uexc:
        logger.error(f"Unexpected error reading band '{band_name}': {uexc}")
        raise

    return BandCreate(
        name=band_name,
        band_wavelengths=band_data[:, 0].tolist(),
        band_transmission=band_data[:, 1].tolist(),
    )


def make_band_create_models(filter_dir: Path | str | None = None) -> list[BandCreate]:
    """
    Create BandCreate models for all registered photometric bands.

    Parameters
    ----------
    filter_dir :
        Directory containing filter response files. If None, uses default location.

    Returns
    -------
    list of BandCreate
        A list of BandCreate models for all bands registered in the BandFactory.

    Notes
    -----
    This function queries the BandFactory to get all registered band names,
    then creates a model for each one. Missing or invalid band files are
    handled gracefully with logging.
    """
    bands: list[BandCreate] = []

    try:
        band_names = catalog_utils.BandFactory.get_bands()
    except Exception as uexc:
        logger.error(f"Failed to retrieve band names from BandFactory: {uexc}")
        raise

    if not band_names:
        logger.warning("BandFactory returned no bands")
        return bands

    logger.info(f"Creating models for {len(band_names)} bands")

    for band_name in band_names:
        try:
            band = make_band_create_model(band_name, filter_dir)
            bands.append(band)
        except Exception as e:
            logger.error(f"Failed to create model for band '{band_name}': {e}")
            # Continue processing other bands
            continue

    logger.info(f"Successfully created {len(bands)} band models")
    return bands


def read_sed_file(file_path: Path | str) -> np.ndarray:
    """Read a SED file containing two columns: wavelength and SED values.

    Parameters
    ----------
    file_path : Path or str
        Path to the SED text file. The file should contain two
        whitespace-separated columns of numbers (wavelength, sed_value).

    Returns
    -------
    np.ndarray
        A 2D array with shape (n, 2) containing wavelengths and SED values.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains invalid data format or does not have 2 columns.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"SED file not found: {file_path}")
        raise FileNotFoundError(f"SED file not found: {file_path}")

    logger.info(f"Reading SED file: {file_path}")

    try:
        full_array = np.loadtxt(file_path)
    except Exception as e:
        logger.error(f"Failed to load SED file {file_path}: {e}")
        raise ValueError(f"Invalid data format in {file_path}") from e

    if full_array.ndim != 2 or full_array.shape[1] != 2:
        logger.error(f"Expected 2-column data in {file_path}, got shape {full_array.shape}")
        raise ValueError(f"Invalid data shape in {file_path}")

    return full_array


def make_sed_create_model(name: str, file_path: Path | str) -> SedCreate:
    """Create a SedCreate model from a SED text file.

    Parameters
    ----------
    name : str
        Name for this SED entry.
    file_path : Path or str
        Path to the SED text file containing two columns
        (wavelength, sed_value).

    Returns
    -------
    SedCreate
        A model containing the SED name, wavelengths, and values.

    Raises
    ------
    ValueError
        If name is empty or the file has invalid format.
    FileNotFoundError
        If the file does not exist.
    """
    if not isinstance(name, str) or not name.strip():
        logger.error(f"Invalid SED name: {name}")
        raise ValueError("name must be a non-empty string")

    sed_data = read_sed_file(file_path)

    return SedCreate(
        name=name,
        sed_wavelengths=sed_data[:, 0].tolist(),
        sed_values=sed_data[:, 1].tolist(),
    )


def make_sed_create_models(
    sed_dir: Path | str,
    names: list[str] | None = None,
    names_file: Path | str | None = None,
) -> list[SedCreate]:
    """Create SedCreate models from .sed files in a directory.

    Parameters
    ----------
    sed_dir : Path or str
        Directory containing SED text files. Files are expected to have
        a .sed extension and contain two columns (wavelength, sed_value).
    names : list[str] | None
        Optional list of SED names (without extension) to load.
        If None and names_file is also None, all .sed files in the
        directory are loaded.
    names_file : Path or str | None
        Optional path to a text file listing SED filenames (one per line)
        relative to sed_dir. Lines are stripped of whitespace; blank lines
        and lines starting with ``#`` are skipped. The .sed extension is
        stripped to derive the entry name. Takes precedence over globbing
        but is ignored if ``names`` is provided.

    Returns
    -------
    list[SedCreate]
        A list of SedCreate models, one per file.

    Raises
    ------
    FileNotFoundError
        If sed_dir or names_file does not exist.
    """
    if not isinstance(sed_dir, Path):
        sed_dir = Path(sed_dir)

    if not sed_dir.exists():
        logger.error(f"SED directory does not exist: {sed_dir}")
        raise FileNotFoundError(f"SED directory not found: {sed_dir}")

    if names is not None:
        files = [sed_dir / f"{n}.sed" for n in names]
    elif names_file is not None:
        if not isinstance(names_file, Path):
            names_file = Path(names_file)
        if not names_file.exists():
            logger.error(f"SED names file not found: {names_file}")
            raise FileNotFoundError(f"SED names file not found: {names_file}")
        lines = names_file.read_text().splitlines()
        filenames = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
        files = [sed_dir / fn for fn in filenames]
    else:
        files = sorted(sed_dir.glob("*.sed"))

    if not files:
        logger.warning(f"No .sed files found in {sed_dir}")
        return []

    logger.info(f"Creating models for {len(files)} SED files from {sed_dir}")

    seds: list[SedCreate] = []
    for file_path in files:
        try:
            name = file_path.stem
            sed = make_sed_create_model(name, file_path)
            seds.append(sed)
        except Exception as e:
            logger.error(f"Failed to create SED model from '{file_path}': {e}")
            continue

    logger.info(f"Successfully created {len(seds)} SED models")
    return seds


def read_filter_ab_file(file_path: Path | str) -> np.ndarray:
    """Read a FilterAB file containing two columns: redshift and flux.

    Parameters
    ----------
    file_path : Path or str
        Path to the FilterAB text file. The file should contain two
        whitespace-separated columns of numbers (redshift, flux).

    Returns
    -------
    np.ndarray
        A 2D array with shape (n, 2) containing redshifts and fluxes.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file contains invalid data format or does not have 2 columns.
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        logger.error(f"FilterAB file not found: {file_path}")
        raise FileNotFoundError(f"FilterAB file not found: {file_path}")

    logger.info(f"Reading FilterAB file: {file_path}")

    try:
        full_array = np.loadtxt(file_path)
    except Exception as e:
        logger.error(f"Failed to load FilterAB file {file_path}: {e}")
        raise ValueError(f"Invalid data format in {file_path}") from e

    if full_array.ndim != 2 or full_array.shape[1] != 2:
        logger.error(f"Expected 2-column data in {file_path}, got shape {full_array.shape}")
        raise ValueError(f"Invalid data shape in {file_path}")

    return full_array


def make_filter_ab_create_model(
    name: str, file_path: Path | str, band_name: str, sed_name: str
) -> FilterABCreate:
    """Create a FilterABCreate model from a FilterAB text file.

    Parameters
    ----------
    name : str
        Name for this FilterAB entry.
    file_path : Path or str
        Path to the FilterAB text file containing two columns
        (redshift, flux).
    band_name : str
        Name of the associated Band (resolved to band_id at DB level).
    sed_name : str
        Name of the associated Sed (resolved to sed_id at DB level).

    Returns
    -------
    FilterABCreate
        A model containing the name, redshifts, fluxes, and foreign
        key names for Band and Sed.

    Raises
    ------
    ValueError
        If name, band_name, or sed_name is empty, or the file has
        invalid format.
    FileNotFoundError
        If the file does not exist.
    """
    if not isinstance(name, str) or not name.strip():
        logger.error(f"Invalid FilterAB name: {name}")
        raise ValueError("name must be a non-empty string")

    if not isinstance(band_name, str) or not band_name.strip():
        raise ValueError("band_name must be a non-empty string")

    if not isinstance(sed_name, str) or not sed_name.strip():
        raise ValueError("sed_name must be a non-empty string")

    filter_ab_data = read_filter_ab_file(file_path)

    return FilterABCreate(
        name=name,
        redshifts=filter_ab_data[:, 0].tolist(),
        fluxes=filter_ab_data[:, 1].tolist(),
        band_name=band_name,
        sed_name=sed_name,
    )


def make_filter_ab_create_models(
    filter_ab_dir: Path | str,
    names: list[str] | None = None,
) -> list[FilterABCreate]:
    """Create FilterABCreate models from .AB files in a directory.

    Files follow the naming convention ``{sed}.{band}.AB``. The band and
    sed names are extracted from the filename. The entry name is set to
    the full stem (``{sed}.{band}``).

    Parameters
    ----------
    filter_ab_dir : Path or str
        Directory containing FilterAB text files with .AB extension,
        each containing two columns (redshift, flux).
    names : list[str] | None
        Optional list of file stems (e.g. ``["elliptical.g"]``) to load.
        If None, all .AB files in the directory are loaded.

    Returns
    -------
    list[FilterABCreate]
        A list of FilterABCreate models, one per file.

    Raises
    ------
    FileNotFoundError
        If filter_ab_dir does not exist.
    """
    if not isinstance(filter_ab_dir, Path):
        filter_ab_dir = Path(filter_ab_dir)

    if not filter_ab_dir.exists():
        logger.error(f"FilterAB directory does not exist: {filter_ab_dir}")
        raise FileNotFoundError(f"FilterAB directory not found: {filter_ab_dir}")

    if names is not None:
        files = [filter_ab_dir / f"{n}.AB" for n in names]
    else:
        files = sorted(filter_ab_dir.glob("*.AB"))

    if not files:
        logger.warning(f"No .AB files found in {filter_ab_dir}")
        return []

    logger.info(f"Creating models for {len(files)} FilterAB files from {filter_ab_dir}")

    filter_abs: list[FilterABCreate] = []
    for file_path in files:
        try:
            # Filename is {sed}.{band}.AB — strip .AB to get stem, then split
            stem = file_path.name.removesuffix(".AB")
            parts = stem.rsplit(".", 1)
            if len(parts) != 2:
                logger.error(
                    f"FilterAB filename '{file_path.name}' does not match "
                    f"expected pattern '{{sed}}.{{band}}.AB'"
                )
                continue
            sed_name, band_name = parts
            entry_name = stem
            fab = make_filter_ab_create_model(entry_name, file_path, band_name, sed_name)
            filter_abs.append(fab)
        except Exception as e:
            logger.error(f"Failed to create FilterAB model from '{file_path}': {e}")
            continue

    logger.info(f"Successfully created {len(filter_abs)} FilterAB models")
    return filter_abs


def make_catalog_tag_create_model(catalog_tag_name: str) -> CatalogTagCreate:
    """
    Create a CatalogTagCreate model from a catalog tag name.

    Parameters
    ----------
    catalog_tag_name : str
        Name of the catalog tag (e.g., 'lsst_dp0', 'des_y6').

    Returns
    -------
    CatalogTagCreate
        A model containing the catalog tag name.

    Raises
    ------
    ValueError
        If catalog_tag_name is not a valid non-empty string.
    """
    if not isinstance(catalog_tag_name, str) or not catalog_tag_name.strip():
        logger.error(f"Invalid catalog_tag_name: {catalog_tag_name}")
        raise ValueError("catalog_tag_name must be a non-empty string")

    return CatalogTagCreate(name=catalog_tag_name)


def make_catalog_tag_create_models() -> list[CatalogTagCreate]:
    """
    Create CatalogTagCreate models for all registered catalog tags.

    Returns
    -------
    list of CatalogTagCreate
        A list of CatalogTagCreate models for all tags registered in the
        CatalogTagFactory.

    Notes
    -----
    This function queries the CatalogTagFactory to get all registered catalog
    tag names, then creates a model for each one.
    """
    catalog_tags: list[CatalogTagCreate] = []

    try:
        tag_dict = catalog_utils.CatalogTagFactory.get_catalog_tags()
    except Exception as uexc:
        logger.error(f"Failed to retrieve catalog tags from CatalogTagFactory: {uexc}")
        raise

    if not tag_dict:
        logger.warning("CatalogTagFactory returned no catalog tags")
        return catalog_tags

    logger.info(f"Creating models for {len(tag_dict)} catalog tags")

    for tag_name in tag_dict:
        try:
            catalog_tag = make_catalog_tag_create_model(tag_name)
            catalog_tags.append(catalog_tag)
        except Exception as uexc:
            logger.error(f"Failed to create model for catalog tag '{tag_name}': {uexc}")
            continue

    logger.info(f"Successfully created {len(catalog_tags)} catalog tag models")
    return catalog_tags


def make_catalog_band_assoc_create_models(
    ct: catalog_utils.CatalogTag,
) -> list[CatalogBandAssocCreate]:
    """
    Create CatalogBandAssocCreate models for a catalog tag's bands.

    This function maps photometric bands to their corresponding magnitude
    and error column names in the catalog, creating association models.

    Parameters
    ----------
    ct : catalog_utils.CatalogTag
        A CatalogTag object containing band configuration information.

    Returns
    -------
    list of CatalogBandAssocCreate
        A list of association models linking catalog tags to bands with
        their corresponding column names.

    Raises
    ------
    ValueError
        If ct is not a valid CatalogTag object.
    AttributeError
        If ct is missing required configuration attributes.

    Notes
    -----
    Column names are determined using templates from the catalog configuration.
    Individual band configurations can override the templates.
    """
    # Input validation
    if not isinstance(ct, catalog_utils.CatalogTag):
        logger.error(f"Expected CatalogTag object, got {type(ct)}")
        raise ValueError("ct must be a CatalogTag object")

    if not hasattr(ct, "config"):
        logger.error("CatalogTag object missing 'config' attribute")
        raise AttributeError("CatalogTag must have a 'config' attribute")

    associations: list[CatalogBandAssocCreate] = []

    try:
        band_list = ct.config.band_list
        bands = ct.config.bands
        catalog_name = ct.config.name
    except AttributeError as uexc:
        logger.error(f"CatalogTag config missing required attributes: {uexc}")
        raise

    if not band_list:
        logger.warning(f"Catalog tag '{catalog_name}' has no bands in band_list")
        return associations

    logger.info(f"Creating band associations for catalog '{catalog_name}' ({len(band_list)} bands)")

    for band_key in band_list:
        try:
            # Get band-specific config or use defaults
            band_config = bands.get(band_key, {})
            band_info = ct.config.bands[band_key]

            # Determine filter name
            band_name = band_config.get("filter", ct.config.filter_template.format(band=band_key))

            # Determine magnitude column name
            mag_column_name = band_info.get(
                "mag_column_name",
                ct.config.mag_column_template.format(band=band_key),
            )

            # Determine magnitude error column name
            mag_err_column_name = band_info.get(
                "mag_err_column_name",
                ct.config.mag_err_column_template.format(band=band_key),
            )

            associations.append(
                CatalogBandAssocCreate(
                    mag_column_name=mag_column_name,
                    mag_err_column_name=mag_err_column_name,
                    band_name=band_name,
                    catalog_tag_name=catalog_name,
                )
            )

            logger.debug(
                f"Created association: {catalog_name}.{band_key} -> "
                f"band={band_name}, mag={mag_column_name}, err={mag_err_column_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to create association for band '{band_key}' in catalog '{catalog_name}': {e}"
            )
            continue
        logger.info(f"Successfully created {len(associations)} band associations for '{catalog_name}'")
    return associations


def make_all_catalog_band_assoc_create_models() -> list[CatalogBandAssocCreate]:
    """
    Create CatalogBandAssocCreate models for all registered catalog tags.

    This function iterates through all catalog tags registered in the
    CatalogTagFactory and creates band associations for each one.

    Returns
    -------
    list of CatalogBandAssocCreate
        A list of all band association models across all catalog tags.

    Notes
    -----
    This function aggregates associations from all catalogs. If a catalog
    has issues, it logs the error and continues with other catalogs.
    """
    all_associations: list[CatalogBandAssocCreate] = []

    try:
        catalog_tags = catalog_utils.CatalogTagFactory.get_catalog_tags()
    except Exception as uexc:
        logger.error(f"Failed to retrieve catalog tags from CatalogTagFactory: {uexc}")
        raise

    if not catalog_tags:
        logger.warning("CatalogTagFactory returned no catalog tags")
        return all_associations

    logger.info(f"Creating band associations for {len(catalog_tags)} catalog tags")

    for tag_name, catalog_tag in catalog_tags.items():
        try:
            associations = make_catalog_band_assoc_create_models(catalog_tag)
            all_associations.extend(associations)
        except Exception as e:
            logger.error(f"Failed to create band associations for catalog tag '{tag_name}': {e}")
            continue

    logger.info(f"Successfully created {len(all_associations)} total band associations")
    return all_associations


def load_catalog_yaml(
    catalog_yaml: Path | str, filter_dir: Path | str | None = None
) -> tuple[list[BandCreate], list[CatalogTagCreate], list[CatalogBandAssocCreate]]:
    """
    Load catalog configuration from a YAML file and create all necessary models.

    This function is the main entry point for loading catalog metadata. It reads
    a YAML configuration file, registers the catalogs and bands, and creates
    model objects for bands, catalog tags, and their associations.

    Parameters
    ----------
    catalog_yaml :
        Path to the YAML file containing catalog configuration.
    filter_dir :
        Directory containing filter response files. If None, uses default location.

    Returns
    -------
    bands : list of BandCreate
        Models for all photometric bands.
    catalog_tags : list of CatalogTagCreate
        Models for all catalog tags.
    catalog_band_assocs : list of CatalogBandAssocCreate
        Association models linking catalog tags to bands.

    Raises
    ------
    FileNotFoundError
        If catalog_yaml does not exist.
    ValueError
        If catalog_yaml is not a valid YAML file or has invalid content.

    Notes
    -----
    This function has side effects: it registers catalogs and bands in the
    CatalogTagFactory and BandFactory, respectively. These registrations
    persist for the lifetime of the Python session.

    Examples
    --------
    >>> bands, tags, assocs = load_catalog_yaml(Path("config/catalogs.yaml"))
    >>> print(f"Loaded {len(bands)} bands, {len(tags)} catalogs")
    """
    # Input validation
    if not isinstance(catalog_yaml, Path):
        try:
            catalog_yaml = Path(catalog_yaml)
        except Exception as uexc:
            logger.error(f"Failed to convert catalog_yaml to Path: {uexc}")
            raise ValueError(f"Invalid catalog_yaml path: {catalog_yaml}") from uexc

    if not catalog_yaml.exists():
        logger.error(f"Catalog YAML file does not exist: {catalog_yaml}")
        raise FileNotFoundError(f"Catalog YAML file not found: {catalog_yaml}")

    if not catalog_yaml.is_file():
        logger.error(f"Catalog YAML path is not a file: {catalog_yaml}")
        raise ValueError(f"catalog_yaml must be a file, not a directory: {catalog_yaml}")

    if unexpected(catalog_yaml.suffix.lower() not in [".yaml", ".yml"]):
        logger.warning(f"Catalog file '{catalog_yaml}' does not have .yaml or .yml extension")

    logger.info(f"Loading catalog configuration from: {catalog_yaml}")

    # Load and register catalogs
    try:
        catalog_utils.load_yaml(catalog_yaml)
        logger.info("Successfully loaded and registered catalog configuration")
    except Exception as e:
        logger.error(f"Failed to load catalog YAML file: {e}")
        raise ValueError(f"Invalid catalog YAML file: {catalog_yaml}") from e

    # Create band models
    try:
        bands = make_band_create_models(filter_dir)
        logger.info(f"Created {len(bands)} band models")
    except Exception as e:
        logger.error(f"Failed to create band models: {e}")
        raise

    # Create catalog tag models
    try:
        catalog_tags = make_catalog_tag_create_models()
        logger.info(f"Created {len(catalog_tags)} catalog tag models")
    except Exception as uexc:
        logger.error(f"Failed to create catalog tag models: {uexc}")
        raise

    # Create catalog-band association models
    try:
        catalog_band_assocs = make_all_catalog_band_assoc_create_models()
        logger.info(f"Created {len(catalog_band_assocs)} catalog-band association models")
    except Exception as uexc:
        logger.error(f"Failed to create catalog-band association models: {uexc}")
        raise

    logger.info(
        f"Successfully loaded catalog configuration: "
        f"{len(bands)} bands, {len(catalog_tags)} catalogs, "
        f"{len(catalog_band_assocs)} associations"
    )

    return (bands, catalog_tags, catalog_band_assocs)


def get_catalog_row(catalog_path: Path | str, row: int) -> dict[str, np.ndarray]:
    """
    Read a single row from a catalog file.

    Parameters
    ----------
    catalog_path :
        Path to the catalog file (typically HDF5 or FITS format).
    row : int
        Zero-based index of the row to read.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary mapping column names to their values for the requested row.
        Each value is typically a single-element array.

    Raises
    ------
    FileNotFoundError
        If catalog_path does not exist.
    ValueError
        If row is negative or out of bounds.
    IOError
        If the file cannot be read or has an invalid format.

    Notes
    -----
    This function uses the tables_io library to read catalog files, which
    supports multiple astronomical file formats.

    Examples
    --------
    >>> row_data = get_catalog_row(Path("catalog.hdf5"), 42)
    >>> print(row_data['mag_g'])
    """
    # Input validation
    if not isinstance(catalog_path, Path):
        try:
            catalog_path = Path(catalog_path)
        except Exception as uexc:
            logger.error(f"Failed to convert catalog_path to Path: {uexc}")
            raise ValueError(f"Invalid catalog_path: {catalog_path}") from uexc

    if not catalog_path.exists():
        logger.error(f"Catalog file does not exist: {catalog_path}")
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    if not catalog_path.is_file():
        logger.error(f"Catalog path is not a file: {catalog_path}")
        raise ValueError(f"catalog_path must be a file: {catalog_path}")

    if row < 0:
        logger.error(f"Row index must be non-negative, got {row}")
        raise ValueError(f"row must be non-negative, got {row}")

    logger.debug(f"Reading row {row} from catalog: {catalog_path}")

    try:
        row_data = tables_io.read(str(catalog_path), slice_dict=row)
    except IndexError as e:
        logger.error(f"Row {row} is out of bounds in catalog {catalog_path}: {e}")
        raise ValueError(f"Row index {row} out of bounds") from e
    except Exception as e:
        logger.error(f"Failed to read row {row} from catalog {catalog_path}: {e}")
        raise OSError(f"Cannot read catalog file: {catalog_path}") from e

    if not isinstance(row_data, dict):
        logger.error(f"Unexpected return type from tables_io.read: {type(row_data)}")
        raise OSError(f"Invalid data format in catalog: {catalog_path}")

    logger.debug(f"Successfully read row {row} with {len(row_data)} columns")
    return row_data


def get_estimates_row(estimates_path: Path | str, row: int) -> dict[str, np.ndarray]:
    """
    Read a single row from a photo-z estimates file.

    This function reads probability distribution estimates (typically posterior
    redshift distributions) stored in qp format.

    Parameters
    ----------
    estimates_path : Path
        Path to the estimates file (qp format, typically HDF5).
    row : int
        Zero-based index of the row to read.

    Returns
    -------
    dict of str to np.ndarray
        Dictionary containing the probability distribution data for the
        requested row. The exact contents depend on the qp parameterization.

    Raises
    ------
    FileNotFoundError
        If estimates_path does not exist.
    ValueError
        If row is negative or out of bounds.
    IOError
        If the file cannot be read or has an invalid format.

    Notes
    -----
    This function uses the qp library to read quantile-parameterized
    probability distributions. The returned data structure depends on
    the specific parameterization used in the file.

    Examples
    --------
    >>> estimates = get_estimates_row(Path("photo_z_pdfs.hdf5"), 42)
    >>> # Access the distribution data
    """
    # Input validation
    if not isinstance(estimates_path, Path):
        try:
            estimates_path = Path(estimates_path)
        except Exception as uexc:
            logger.error(f"Failed to convert estimates_path to Path: {uexc}")
            raise ValueError(f"Invalid estimates_path: {estimates_path}") from uexc

    if not estimates_path.exists():
        logger.error(f"Estimates file does not exist: {estimates_path}")
        raise FileNotFoundError(f"Estimates file not found: {estimates_path}")

    if not estimates_path.is_file():
        logger.error(f"Estimates path is not a file: {estimates_path}")
        raise ValueError(f"estimates_path must be a file: {estimates_path}")

    if row < 0:
        logger.error(f"Row index must be non-negative, got {row}")
        raise ValueError(f"row must be non-negative, got {row}")

    logger.debug(f"Reading row {row} from estimates file: {estimates_path}")

    try:
        # Read single row as a slice
        row_slice = slice(row, row + 1)
        estimate_data = qp.read(str(estimates_path), read_slice=row_slice)
    except IndexError as e:
        logger.error(f"Row {row} is out of bounds in estimates file {estimates_path}: {e}")
        raise ValueError(f"Row index {row} out of bounds") from e
    except Exception as e:
        logger.error(f"Failed to read row {row} from estimates file {estimates_path}: {e}")
        raise OSError(f"Cannot read estimates file: {estimates_path}") from e

    if estimate_data is None:
        logger.error(f"qp.read returned None for row {row} in {estimates_path}")
        raise OSError(f"Invalid or empty data at row {row} in estimates file")

    logger.debug(f"Successfully read estimate for row {row}")
    return estimate_data


def get_multi_catalog_row(
    matched_dataset_path: Path | str,
    component_dataset_paths: dict[str, Path | str],
    row: int,
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
    row : int
        Zero-based index of the row to read.

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
    >>> data = get_multi_catalog_row(matched, components, 4)
    """
    archive_dir = Path(global_config.storage.archive)
    matched_path = archive_dir / matched_dataset_path

    if not matched_path.exists():
        raise FileNotFoundError(f"Matched dataset not found: {matched_path}")

    match_set = tables_io.read(matched_path, slice_dict=row)
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
