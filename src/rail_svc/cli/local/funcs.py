from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast
from collections.abc import Sequence

import click

from ... import local_sync
from macon.db.session import init_db
from ...models.utils import OutputEnum, output_pydantic
from ... import models
from .. import common_options
from .rail_svc import handle_database_error

logger = logging.getLogger(__name__)


@click.group(name="funcs")
def funcs_group() -> None:  # pragma: no cover
    """Commands to execute specific rail functionality"""


@funcs_group.command(name="estimate-pdf")
@common_options.estimator_id()
@common_options.dataset_id()
@common_options.row()
@common_options.output()
def estimate_pdf(
    estimator_id: int,
    dataset_id: int,
    row: int,
    output: OutputEnum,
) -> None:
    # Ensure database engine is initialized
    init_db()

    try:
        data = local_sync.funcs.estimate_pdf(
            estimator_id=estimator_id,
            dataset_id=dataset_id,
            row=row,
        )
        # Output the data
        if output == OutputEnum.json:
            click.echo(json.dumps(data, indent=2, default=str))
        else:  # pragma: no cover
            click.echo(data)

    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="estimate-ensemble")
@common_options.estimator_id()
@common_options.dataset_id()
@common_options.output_path()
def estimate_ensemble(
    estimator_id: int,
    dataset_id: int,
    output_path: str | Path,
) -> None:

    # Ensure database engine is initialized
    init_db()

    try:
        output_file = local_sync.funcs.estimate_ensemble(
            estimator_id=estimator_id,
            dataset_id=dataset_id,
            output_file_path=output_path,
        )
        print(f"Wrote data to {output_file}")

    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="get-estimators-for-dataest")
@common_options.dataset_id()
@common_options.output()
def get_estimators_for_dataest(
    dataset_id: int,
    output: OutputEnum,
) -> None:

    # Ensure database engine is initialized
    init_db()

    try:
        data = local_sync.funcs.get_estimators_for_dataest(
            dataset_id=dataset_id,
        )
        print(
            output_pydantic(
                cast(Sequence[models.Estimator], data), output, models.Estimator.col_names_for_table
            )
        )

    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="load-catalog-yaml")
@click.option(
    "--catalog-yaml",
    "catalog_yaml",
    type=click.Path(exists=True),
    help="Path to catalog definition yaml file",
)
@click.option(
    "--filter-dir", "filter_dir", type=click.Path(exists=True, dir_okay=True), help="Path to filters"
)
@common_options.output()
def load_catalog_yaml(
    catalog_yaml: Path,
    filter_dir: Path | None,
    output: OutputEnum,
) -> None:
    # Ensure database engine is initialized
    init_db()

    try:
        bands, catalog_tags, catalog_band_assocs = local_sync.funcs.load_catalog_yaml(
            catalog_yaml=catalog_yaml,
            filter_dir=filter_dir,
        )
        print(output_pydantic(bands, output, models.Band.col_names_for_table))
        print(output_pydantic(catalog_tags, output, models.CatalogTag.col_names_for_table))
        print(output_pydantic(catalog_band_assocs, output, models.CatalogBandAssoc.col_names_for_table))

    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="get-dataset-and-estimates")
@common_options.dataset_id()
@common_options.output()
def get_dataset_and_estimates(
    dataset_id: int,
    output: OutputEnum,
) -> None:
    # Ensure database engine is initialized
    init_db()

    try:
        the_dataset, the_estimates = local_sync.funcs.get_dataset_and_estimates(
            dataset_id=dataset_id,
        )
        print(output_pydantic(the_dataset, output, models.Dataset.col_names_for_table))
        print(output_pydantic(the_estimates, output, models.Estimates.col_names_for_table))

    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="get-data-and-estimates-data")
@common_options.dataset_id()
@common_options.row()
@common_options.output()
def get_data_and_estimates_data(
    dataset_id: int,
    row: int,
    output: OutputEnum,
) -> None:
    # Ensure database engine is initialized
    init_db()

    try:
        data, the_estimates_dict = local_sync.funcs.get_data_and_estimates_data(
            dataset_id=dataset_id,
            row=row,
        )
        if output == OutputEnum.json:
            click.echo(json.dumps(data, indent=2, default=str))
            click.echo(json.dumps(the_estimates_dict, indent=2, default=str))
        else:
            click.echo(data)
            click.echo(the_estimates_dict)

    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="create-matched-dataset")
@click.option("--matched-dataset-name", "matched_dataset_name", type=str, help="Name of dataset to create")
@click.option("--catalog-tag-name", "catalog_tag_name", type=str, help="Name of catalog tag to use")
@click.option(
    "--component-dataset-names",
    "component_dataset_names",
    multiple=True,
    type=str,
    help="Name of catalog tag to use",
)
@common_options.path()
@click.option("--n-objects", "n_objects", type=int, help="Number of objects in dataset")
@common_options.output()
def create_matched_dataset(
    matched_dataset_name: str,
    catalog_tag_name: str,
    component_dataset_names: list[str],
    path: str | None,
    n_objects: int,
    output: OutputEnum,
) -> None:
    # Ensure database engine is initialized
    init_db()

    try:
        dataset, _assocs = local_sync.funcs.create_matched_dataset(
            matched_dataset_name=matched_dataset_name,
            catalog_tag_name=catalog_tag_name,
            component_dataset_names=component_dataset_names,
            path=path,
            n_objects=n_objects,
        )
        print(output_pydantic(dataset, output, models.Dataset.col_names_for_table))
    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="estimate-pdf-for-slice")
@common_options.estimator_id()
@common_options.dataset_id()
@common_options.slice_option()
@click.option("--recompute-if-exists", is_flag=True, help="Recompute event if it already exists")
@common_options.output()
def estimate_pdf_for_slice(
    estimator_id: int,
    dataset_id: int,
    slice_option: slice | int | None,
    *,
    recompute_if_exists: bool = False,
    output: OutputEnum,
) -> None:
    # Ensure database engine is initialized
    init_db()

    try:
        data = local_sync.funcs.estimate_pdf_for_slice(
            estimator_id=estimator_id,
            dataset_id=dataset_id,
            the_slice=slice_option,
            recompute_if_exists=recompute_if_exists,
        )
        # Output the data
        if output == OutputEnum.json:
            click.echo(json.dumps(data, indent=2, default=str))
        else:  # pragma: no cover
            click.echo(data)

    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")


@funcs_group.command(name="estimate-dataset")
@common_options.estimator_id()
@common_options.dataset_id()
@click.option("--raise-if-exists", is_flag=True, help="Raise Error if it already exists")
@common_options.output()
def estimate_dataset(
    estimator_id: int,
    dataset_id: int,
    *,
    raise_if_exists: bool = False,
    output: OutputEnum,
) -> None:
    # Ensure database engine is initialized
    init_db()

    try:
        data = local_sync.funcs.estimate_dataset(
            estimator_id=estimator_id,
            dataset_id=dataset_id,
            raise_if_exists=raise_if_exists,
        )
        print(output_pydantic(data, output, models.Estimates.col_names_for_table))
    except Exception as uexc:
        handle_database_error(uexc, f"{uexc}")
