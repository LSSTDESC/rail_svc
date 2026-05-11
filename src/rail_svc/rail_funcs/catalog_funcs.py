from pathlib import Path

import numpy as np
import qp
import tables_io
from rail.utils import catalog_utils

from ..models import BandCreate, CatalogBandAssocCreate, CatalogTagCreate


def extract_padded_non_zeros(the_array: np.ndarray) -> np.ndarray:
    vals = the_array[:, 1]
    non_zero = vals != 0
    padded_start = max(int(np.argmax(non_zero)) - 1, 0)
    padded_end = min(int(len(the_array) - np.argmax(non_zero[::-1]) + 1), len(the_array))
    the_slice = slice(padded_start, padded_end)
    return the_array[the_slice]


def read_band_res_file(band_name: str, filter_dir: Path | None = None) -> np.ndarray:
    if filter_dir is None:
        filter_dir = Path(catalog_utils.find_rail_file("examples_data/estimation_data/data/FILTER"))

    full_array = np.loadtxt(filter_dir / f"{band_name}.res")
    return extract_padded_non_zeros(full_array)


def make_band_create_model(band_name: str, filter_dir: Path | None) -> BandCreate:
    try:
        band_data = read_band_res_file(band_name, filter_dir)
    except:
        band_data = np.zeros(shape=(2, 2))
    return BandCreate(
        name=band_name,
        band_wavelengths=band_data[:, 0].tolist(),
        band_transmission=band_data[:, 1].tolist(),
    )


def make_band_create_models(filter_dir: Path | None = None) -> list[BandCreate]:
    the_bands: list[BandCreate] = []
    for k in catalog_utils.BandFactory.get_bands():
        the_bands.append(make_band_create_model(k, filter_dir))
    return the_bands


def make_catalog_tag_create_model(catalog_tag_name: str) -> CatalogTagCreate:
    return CatalogTagCreate(name=catalog_tag_name)


def make_catalog_tag_create_models() -> list[CatalogTagCreate]:
    the_catalog_tags: list[CatalogTagCreate] = []
    for k in catalog_utils.CatalogTagFactory.get_catalog_tags():
        the_catalog_tags.append(make_catalog_tag_create_model(k))
    return the_catalog_tags


def make_catalog_band_assoc_create_models(ct: catalog_utils.CatalogTag) -> list[CatalogBandAssocCreate]:
    ret_list: list[CatalogTagCreate] = []
    for k in ct.config.band_list:
        band_name = ct.config.bands[k].get("filter", ct.config.filter_template.format(band=k))
        mag_column_name = ct.config.bands[k].get("mag_column", ct.config.mag_column_template.format(band=k))
        mag_err_column_name = ct.config.bands[k].get(
            "mag_err_column", ct.config.mag_err_column_template.format(band=k)
        )
        ret_list.append(
            CatalogBandAssocCreate(
                mag_column_name=mag_column_name,
                mag_err_column_name=mag_err_column_name,
                band_name=band_name,
                catalog_tag_name=ct.config.name,
            )
        )
    return ret_list


def make_all_catalog_band_assoc_create_models() -> list[CatalogBandAssocCreate]:
    ret_list: list[CatalogBandAssocCreate] = []
    for _k, v in catalog_utils.CatalogTagFactory.get_catalog_tags().items():
        ret_list += make_catalog_band_assoc_create_models(v)
    return ret_list


def load_catalog_yaml(
    catalog_yaml: Path, filter_dir: Path | None = None
) -> tuple[list[BandCreate], list[CatalogTagCreate], list[CatalogBandAssocCreate]]:
    catalog_utils.load_yaml(catalog_yaml)
    bands = make_band_create_models(filter_dir)
    catalog_tags = make_catalog_tag_create_models()
    catalog_band_assocs = make_all_catalog_band_assoc_create_models()

    return (bands, catalog_tags, catalog_band_assocs)


def get_catalog_row(catalog_path: Path, row: int) -> dict[str, np.ndarray]:
    return tables_io.read(str(catalog_path), slice_dict=row)


def get_estimates_row(estimates_path: Path, row: int) -> dict[str, np.ndarray]:
    return qp.read(str(estimates_path), read_slice=slice(row, row + 1))
