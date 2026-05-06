"""CLI entry point for rail-svc-client."""

import click
from ... import local, __version__
from .base import CliOperations


def make_table_group(name: str, ops, desc: str) -> click.Group:
    """Create table CLI group with all commands."""

    @click.group(name=name, help=desc)
    def grp():
        pass

    cli = CliOperations(ops, grp)
    cli.register_all_create_commands()
    cli.register_all_read_commands()
    cli.register_all_update_commands()
    cli.register_all_delete_commands()
    cli.register_all_filter_commands()
    return grp


# One-line per table
TABLES = [
    ("algorithm", local.algorithm, "Manage Algorithm table"),
    ("band", local.band, "Manage Band table"),
    ("catalog-band-assoc", local.catalog_band_assoc, "Manage CatalogBandAssoc table"),
    ("catalog-tag", local.catalog_tag, "Manage CatalogTag table"),
    ("dataset", local.dataset, "Manage Dataset table"),
    ("estimates", local.estimates, "Manage Estimates table"),
    ("estimator", local.estimator, "Manage Estimator table"),
    ("model", local.model, "Manage Model table"),
]


@click.group(
    name="rail-svc-client",
    commands=[make_table_group(*t) for t in TABLES],
)
@click.version_option(version=__version__)
def cli() -> None:
    """Administrative CLI for rail-svc."""
    pass


if __name__ == "__main__":
    cli()
