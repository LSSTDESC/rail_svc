"""CLI entry point for rail-svc-client."""

import asyncio
from typing import Any, TypeVar

import click
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.schema import CreateSchema

from ... import __version__, db, local_sync
from ...config import config
from ...db.base import Base
from .base import CliOperations

T = TypeVar("T", bound=Base)
ResponseT = TypeVar("ResponseT", bound=BaseModel)
CreateT = TypeVar("CreateT", bound=BaseModel)


@click.command(name="init")
@click.option("--reset", is_flag=True, help="Delete all existing database data.")
def init(*, reset: bool) -> None:
    """Initialize the DB"""

    def _init_db_sync() -> None:
        """Synchronous wrapper for async init."""

        async def _init_db() -> None:
            engine = create_async_engine(config.db.url)
            try:
                conn = engine.connect()
            except Exception as msg:
                await engine.dispose()
                raise RuntimeError(f"{msg}") from msg
            try:
                await conn.start()

                if db.Base.metadata.schema is not None:
                    await conn.execute(CreateSchema(db.Base.metadata.schema, if_not_exists=True))
                if reset:
                    await conn.run_sync(db.Base.metadata.drop_all)
                await conn.run_sync(db.Base.metadata.create_all)
            except Exception as msg:
                await conn.rollback()
                await conn.close()
                await engine.dispose()
                raise RuntimeError(f"{msg}") from msg

            await conn.close()
            await engine.dispose()

        asyncio.run(_init_db())

    _init_db_sync()


def make_table_group(name: str, ops: Any, desc: str) -> click.Group:
    """Create table CLI group with all commands."""

    @click.group(name=name, help=desc)
    def grp() -> None:  # pragma: no cover
        pass

    cli_ops = CliOperations(ops, grp)
    cli_ops.register_all_create_commands()
    cli_ops.register_all_read_commands()
    cli_ops.register_all_update_commands()
    cli_ops.register_all_delete_commands()
    cli_ops.register_all_filter_commands()
    return grp


# One-line per table
TABLES = [
    ("algorithm", local_sync.algorithm, "Manage Algorithm table"),
    ("band", local_sync.band, "Manage Band table"),
    ("catalog-band-assoc", local_sync.catalog_band_assoc, "Manage CatalogBandAssoc table"),
    ("catalog-tag", local_sync.catalog_tag, "Manage CatalogTag table"),
    ("dataset", local_sync.dataset, "Manage Dataset table"),
    ("dataset-assoc", local_sync.dataset_assoc, "Manage DatasetAssoc table"),
    ("estimates", local_sync.estimates, "Manage Estimates table"),
    ("estimator", local_sync.estimator, "Manage Estimator table"),
    ("model", local_sync.model, "Manage Model table"),
]


@click.group(
    name="rail-svc-client",
    commands=[init] + [make_table_group(t[0], t[1], t[2]) for t in TABLES],
)
@click.version_option(version=__version__)
def cli() -> None:
    """Administrative CLI for rail-svc."""


if __name__ == "__main__":
    cli()
