import os
from pathlib import Path

import numpy as np
import qp
from sqlalchemy.ext.asyncio import AsyncSession

from .. import models, rail_funcs
from ..config import config as global_config
from .band import band
from .catalog_band_assoc import catalog_band_assoc
from .catalog_tag import catalog_tag
from .dataset import dataset
from .estimates import estimates
from .estimator import estimator


async def load_catalog_yaml(
    session: AsyncSession, catalog_yaml: Path, filter_dir: Path | None = None
) -> tuple[list[models.Band], list[models.CatalogTag], list[models.CatalogBandAssoc]]:

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
    archive_dir = Path(os.path.abspath(global_config.storage.archive))
    return rail_funcs.catalog_funcs.get_catalog_row(archive_dir / the_dataset.path, row)


async def get_estimates_row(
    session: AsyncSession,
    estimates_id: int,
    row: int,
) -> dict[str, np.ndarray]:

    the_estimates = await estimates.get_row(session, estimates_id)
    archive_dir = Path(os.path.abspath(global_config.storage.archive))
    return rail_funcs.catalog_funcs.get_estimates_row(archive_dir / the_estimates.path, row)


async def get_dataset_and_estimates(
    session: AsyncSession,
    dataset_id: int,
) -> tuple[models.Dataset, list[models.Estimates]]:

    the_dataset = await dataset.get_row(session, dataset_id)
    the_estimates = await estimates.find_by(session, dataset_id=the_dataset.id_)
    return (the_dataset, list(the_estimates))


async def get_data_and_estimates_data(
    session: AsyncSession,
    dataset_id: int,
    row: int,
) -> tuple[dict[str, np.ndarray], dict[str, qp.Ensemble]]:

    the_dataset, the_estimates = await get_dataset_and_estimates(session, dataset_id)
    archive_dir = Path(os.path.abspath(global_config.storage.archive))
    data = rail_funcs.catalog_funcs.get_catalog_row(archive_dir / the_dataset.path, row)
    the_estimates_dict: dist[str, qp.Ensemble] = {}
    for the_estimates_ in the_estimates:
        the_estimator = await estimator.get_row(session, the_estimates_.estimator_id)
        the_estimates_dict[the_estimator.name] = rail_funcs.catalog_funcs.get_estimates_row(
            archive_dir / the_estimates_.path, row
        )
    return (data, the_estimates_dict)
