from __future__ import annotations

import click
from httpx import HTTPError, TimeoutException
from pydantic_core import ValidationError as CoreValidationError
from structlog import get_logger

logger = get_logger(__name__)


def handle_cli_error(exc: Exception, operation: str, resource: str) -> None:
    """
    Common CLI error handler.

    Parameters
    ----------
    exc : Exception
        The exception to handle
    operation : str
        The operation being performed (e.g., "fetch", "create")
    resource : str
        The resource type (e.g., "users", "items")

    Raises
    ------
    click.Abort
        Always raises after logging/displaying error
    """
    if isinstance(exc, HTTPError):
        logger.error("API error", operation=operation, resource=resource, error=str(exc))
        click.echo(f"Error: Failed to {operation} {resource}: {exc}", err=True)
    elif isinstance(exc, (CoreValidationError, ValueError, TypeError)):
        logger.error("Validation error", operation=operation, resource=resource, error=str(exc))
        click.echo(f"Error: Invalid data: {exc}", err=True)
    elif isinstance(exc, TimeoutException):
        logger.error("Timeout error", operation=operation, resource=resource, error=str(exc))
        click.echo(f"Error: Request timed out while trying to {operation} {resource}", err=True)
    elif isinstance(exc, KeyError):
        logger.warning("Not found", operation=operation, resource=resource, error=str(exc))
        click.echo(f"Error: {exc}", err=True)
    else:
        logger.error(
            "Unexpected error", operation=operation, resource=resource, error=str(exc), exc_info=True
        )
        click.echo(f"Error: {exc}", err=True)
    raise click.Abort()
