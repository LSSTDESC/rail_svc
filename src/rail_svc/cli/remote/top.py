"""CLI entry point for rail-svc-client remote operations."""

from typing import Any

import click

from ... import __version__, remote_sync
from .base import CliRemoteOperations


def make_table_group(name: str, ops_factory: Any, desc: str) -> click.Group:
    """Create table CLI group with all commands.

    Parameters
    ----------
    name : str
        Name of the CLI group
    ops_factory : callable
        Factory function that creates a SyncRemoteOperations instance
    desc : str
        Description for the CLI group

    Returns
    -------
    click.Group
        Configured Click group with all table commands
    """

    @click.group(name=name, help=desc)
    def grp() -> None:
        pass

    # Create the operations instance
    ops = ops_factory()

    cli_ops = CliRemoteOperations(ops, grp)
    cli_ops.register_all_create_commands()
    cli_ops.register_all_read_commands()
    cli_ops.register_all_update_commands()
    cli_ops.register_all_delete_commands()
    cli_ops.register_all_filter_commands()
    return grp


# One-line per table
TABLES = [
    ("algorithm", remote_sync.algorithm, "Manage Algorithm table"),
    ("band", remote_sync.band, "Manage Band table"),
    ("catalog-band-assoc", remote_sync.catalog_band_assoc, "Manage CatalogBandAssoc table"),
    ("catalog-tag", remote_sync.catalog_tag, "Manage CatalogTag table"),
    ("dataset", remote_sync.dataset, "Manage Dataset table"),
    ("dataset-assoc", remote_sync.dataset_assoc, "Manage DatasetAssoc table"),
    ("estimates", remote_sync.estimates, "Manage Estimates table"),
    ("estimator", remote_sync.estimator, "Manage Estimator table"),
    ("model", remote_sync.model, "Manage Model table"),
]


@click.group(
    name="rail-svc-client-remote",
    commands=[make_table_group(t[0], t[1], t[2]) for t in TABLES],
)
@click.version_option(version=__version__)
@click.option(
    "--base-url",
    envvar="RAIL_SVC_BASE_URL",
    help="Base URL of the rail-svc API server",
)
@click.option(
    "--timeout",
    type=float,
    default=30.0,
    help="Request timeout in seconds (default: 30.0)",
)
@click.option(
    "--auth-token",
    envvar="RAIL_SVC_AUTH_TOKEN",
    help="Authentication token for API requests",
)
@click.pass_context
def cli(ctx: click.Context, base_url: str | None, timeout: float, auth_token: str | None) -> None:
    """Administrative CLI for rail-svc remote operations.

    This CLI interacts with a remote rail-svc API server via HTTP.
    Configure the base URL with --base-url or RAIL_SVC_BASE_URL env var.

    Examples:
        rail-svc-client-remote --base-url http://localhost:8000 algorithm get-rows
        RAIL_SVC_BASE_URL=http://api.example.com rail-svc-client-remote band count
    """
    # Store configuration in context for potential use by subcommands
    ctx.ensure_object(dict)
    ctx.obj["base_url"] = base_url
    ctx.obj["timeout"] = timeout
    ctx.obj["auth_token"] = auth_token


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
