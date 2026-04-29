"""
Multi-layer database operation abstraction.

Architecture Layers:
- Local: Direct database access via CLI commands
- Server: FastAPI REST API endpoints
- Client: HTTP client methods for API consumption
- Remote: CLI commands that call the HTTP client

This design allows operations to be defined once and automatically
work across all interaction modes (local CLI, REST API, remote CLI).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
import urljoin
import tabulate

import json
import yaml

import click
from fastapi import APIRouter
from pydantic import BaseModel
from structlog import get_logger

logger = get_logger(__name__)


# Type Protocols for better type hints
class ClickCommand(Protocol):
    """Protocol for Click command callables."""
    def __call__(self, *args, **kwargs) -> None: ...


class FastAPIEndpoint(Protocol):
    """Protocol for FastAPI endpoint callables."""
    async def __call__(self, *args, **kwargs) -> BaseModel: ...


class ClientMethod(Protocol):
    """Protocol for HTTP client method callables."""
    def __call__(self, *args, **kwargs) -> BaseModel: ...


# Type aliases for clarity
type RouterDecorator = Callable[[FastAPIEndpoint], FastAPIEndpoint]
type CommandDecorator = Callable[[ClickCommand], ClickCommand]


@dataclass
class OperationContext[T: BaseModel]:
    """
    Shared context for all operation layers.

    Encapsulates the common configuration needed across all layers
    of an operation (CLI, API, client).

    Parameters
    ----------
    name : str
        Human-readable operation name (e.g., "list", "create")
    db_class : type[T]
        Database model class
    response_class : type[BaseModel]
        Pydantic response model class
    router_string : str
        URL path segment for API routes
    """
    name: str
    db_class: type[T]
    response_class: type[BaseModel]
    router_string: str

    @classmethod
    def from_db_class(cls, name: str, db_class: type[T]) -> OperationContext[T]:
        """
        Create context from database class using conventions.

        Parameters
        ----------
        name : str
            Operation name
        db_class : type[T]
            Database model class that implements pydantic_model_class()

        Returns
        -------
        OperationContext[T]
            Configured OperationContext

        Raises
        ------
        AttributeError
            If db_class doesn't implement required methods
        """
        # Validate required methods exist
        if not hasattr(db_class, 'pydantic_model_class'):
            raise AttributeError(
                f"{db_class.__name__} must implement pydantic_model_class() method"
            )
        if not hasattr(db_class, 'class_string'):
            raise AttributeError(
                f"{db_class.__name__} must implement class_string() method"
            )

        return cls(
            name=name,
            db_class=db_class,
            response_class=db_class.pydantic_model_class(),
            router_string=db_class.class_string(),
        )

    @property
    def col_names(self) -> list[str]:
        """
        Get column names for table output.

        Returns
        -------
        list[str]
            List of column names for CLI table display

        Raises
        ------
        NotImplementedError
            If response_class doesn't define col_names_for_table
        TypeError
            If col_names_for_table is not a list
        """
        if not hasattr(self.response_class, 'col_names_for_table'):
            raise NotImplementedError(
                f"{self.response_class.__name__} must define 'col_names_for_table' "
                "class attribute for table display"
            )
        col_names = self.response_class.col_names_for_table

        if not isinstance(col_names, list):
            raise TypeError(
                f"col_names_for_table must be a list, got {type(col_names).__name__}"
            )

        return col_names

    @property
    def col_names_optional(self) -> list[str] | None:
        """
        Get column names if available, None otherwise (non-raising version).

        Returns
        -------
        list[str] or None
            List of column names if available, None otherwise
        """
        return getattr(self.response_class, 'col_names_for_table', None)


class BaseOperation[T: BaseModel](ABC):
    """
    Base class for database operations across all layers.

    Subclasses implement four methods to define an operation:
    - create_local_command: CLI for direct database access
    - create_router_endpoint: FastAPI REST endpoint
    - create_client_method: HTTP client wrapper
    - create_remote_command: CLI that calls HTTP client

    Parameters
    ----------
    context : OperationContext[T]
        Shared configuration for this operation

    Examples
    --------
    >>> class ListOperation(BaseOperation[User]):
    ...     def create_local_command(self, group):
    ...         @group.command(name=f"list-{self.ctx.router_string}")
    ...         def list_cmd():
    ...             # Direct DB access
    ...             pass
    ...         return list_cmd
    ...     # ... implement other methods
    """

    def __init__(self, context: OperationContext[T]) -> None:
        """
        Initialize operation with context.

        Parameters
        ----------
        context : OperationContext[T]
            Shared configuration for this operation
        """
        self.ctx = context
        self._validate_context()

    def _validate_context(self) -> None:
        """
        Validate that context is properly configured.

        Raises
        ------
        ValueError
            If context is invalid
        """
        if not self.ctx.name:
            raise ValueError(f"Operation name cannot be empty: {self.ctx.db_class}")
        if not self.ctx.router_string:
            raise ValueError(f"Router string cannot be empty: {self.ctx.db_class}")

    @abstractmethod
    def create_local_command(self, group: click.Group) -> ClickCommand:
        """
        Create CLI command for local database access.

        This command should perform the operation directly against
        the database without going through the API.

        Parameters
        ----------
        group : click.Group
            Click command group to add command to

        Returns
        -------
        ClickCommand
            The created Click command function

        Examples
        --------
        >>> @group.command(name=f"list-{self.ctx.router_string}")
        ... @click.option('--limit', default=10)
        ... def list_local(limit: int):
        ...     # Direct database access
        ...     results = self.ctx.db_class.query().limit(limit).all()
        ...     display_table(results, self.ctx.col_names)
        """
        pass

    @abstractmethod
    def create_router_endpoint(self, router: APIRouter) -> FastAPIEndpoint:
        """
        Create FastAPI route handler.

        This creates a REST endpoint that will be served by the API.

        Parameters
        ----------
        router : APIRouter
            FastAPI router to add endpoint to

        Returns
        -------
        FastAPIEndpoint
            The created async endpoint function

        Examples
        --------
        >>> @router.get(
        ...     f"/{self.ctx.router_string}",
        ...     response_model=list[self.ctx.response_class]
        ... )
        ... async def list_endpoint(limit: int = 10):
        ...     results = await self.ctx.db_class.query().limit(limit).all()
        ...     return results
        """
        pass

    @abstractmethod
    def create_client_method(self) -> ClientMethod:
        """
        Create HTTP client method.

        This method wraps API calls with proper error handling
        and response parsing.

        Returns
        -------
        ClientMethod
            HTTP client method callable

        Examples
        --------
        >>> def list_items(limit: int = 10) -> list[ResponseModel]:
        ...     response = httpx.get(
        ...         f"{base_url}/{self.ctx.router_string}",
        ...         params={"limit": limit}
        ...     )
        ...     response.raise_for_status()
        ...     return [self.ctx.response_class(**item) for item in response.json()]
        """
        pass

    @abstractmethod
    def create_remote_command(self, group: click.Group) -> ClickCommand:
        """
        Create CLI command for remote API access.

        This command should call the client method to interact
        with the remote API.

        Parameters
        ----------
        group : click.Group
            Click command group to add command to

        Returns
        -------
        ClickCommand
            The created Click command function

        Examples
        --------
        >>> @group.command(name=f"list-{self.ctx.router_string}-remote")
        ... @click.option('--limit', default=10)
        ... def list_remote(limit: int):
        ...     client_method = self.create_client_method()
        ...     results = client_method(limit=limit)
        ...     display_table(results, self.ctx.col_names)
        """
        pass


def display_table(data: list[dict], col_names: list[str]) -> None:
    """
    Display data as a formatted table.

    Parameters
    ----------
    data : list[dict]
        List of dictionaries containing the data to display
    col_names : list[str]
        Column names to display and extract from each dict

    Notes
    -----
    Uses tabulate library to format the table. Missing keys in data
    dictionaries will be displayed as empty cells.
    """
    if not data:
        click.echo("No data to display")
        return

    # Extract rows using specified column names
    rows = []
    for item in data:
        row = [item.get(col, '') for col in col_names]
        rows.append(row)

    # Display using tabulate
    table = tabulate(rows, headers=col_names, tablefmt='simple')
    click.echo(table)

def output_json(
    response: dict | list | str,
    output: str
) -> None:
    """
    Output JSON data in the specified format.

    Parameters
    ----------
    response : dict, list, or str
        JSON data to output. Can be a dict, list, or JSON string
    output : str
        Output format: 'json' or 'yaml'

    Raises
    ------
    ValueError
        If output format is not recognized or not supported for JSON data
    """
    # If response is already a string, parse it first
    if isinstance(response, str):
        data = json.loads(response)
    else:
        data = response

    if output == 'json':
        # Output as formatted JSON
        click.echo(json.dumps(data, indent=2))
    elif output == 'yaml':
        # Output as YAML
        click.echo(yaml.dump(data, default_flow_style=False))
    else:
        raise ValueError(f"Unknown output format: {output}")


def output_pydantic_list(
    result: list[BaseModel],
    output: str,
    col_names: list[str] | None
) -> None:
    """
    Output a list of Pydantic models in the specified format.

    Parameters
    ----------
    result : list[BaseModel]
        List of Pydantic model instances to output
    output : str
        Output format: 'json', 'yaml', or 'table'
    col_names : list[str] or None
        Column names for table output. If None and output is 'table',
        will raise an error

    Raises
    ------
    click.BadParameter
        If output format is 'table' but col_names is None
    ValueError
        If output format is not recognized
    """
    if output == 'json':
        # Convert to JSON-serializable dicts and output
        data = [item.model_dump() for item in result]
        click.echo(json.dumps(data, indent=2))
    elif output == 'yaml':
        # Convert to dicts and output as YAML
        data = [item.model_dump() for item in result]
        click.echo(yaml.dump(data, default_flow_style=False))
    elif output == 'table':
        if col_names is None:
            raise click.BadParameter(
                "Table output requires column names to be defined"
            )
        # Convert to list of dicts for table display
        data = [item.model_dump() for item in result]
        display_table(data, col_names)
    else:
        raise ValueError(f"Unknown output format: {output}")


def output_pydantic_single(
    result: BaseModel,
    output: str,
    col_names: list[str] | None
) -> None:
    """
    Output a single Pydantic model in the specified format.

    Parameters
    ----------
    result : BaseModel
        Pydantic model instance to output
    output : str
        Output format: 'json', 'yaml', or 'table'
    col_names : list[str] or None
        Column names for table output. If None and output is 'table',
        will raise an error

    Raises
    ------
    click.BadParameter
        If output format is 'table' but col_names is None
    ValueError
        If output format is not recognized
    """
    if output == 'json':
        # Convert to JSON-serializable dict and output
        data = result.model_dump()
        click.echo(json.dumps(data, indent=2))
    elif output == 'yaml':
        # Convert to dict and output as YAML
        data = result.model_dump()
        click.echo(yaml.dump(data, default_flow_style=False))
    elif output == 'table':
        if col_names is None:
            raise click.BadParameter(
                "Table output requires column names to be defined"
            )
        # Convert to list with single dict for table display
        data = [result.model_dump()]
        display_table(data, col_names)
    else:
        raise ValueError(f"Unknown output format: {output}")


def normalize_url_base(base: str) -> str:
    """
    Normalize URL base by ensuring single trailing slash.

    Parameters
    ----------
    base : str
        Base URL string

    Returns
    -------
    str
        Normalized URL with single trailing slash
    """
    return base.rstrip('/') + '/'


def build_url(base: str, *parts: str) -> str:
    """
    Build URL from parts with proper joining.

    Parameters
    ----------
    base : str
        Base URL
    *parts : str
        URL path parts

    Returns
    -------
    str
        Complete URL
    """
    normalized_base = normalize_url_base(base)
    path = '/'.join(str(p).strip('/') for p in parts if p)
    return urljoin(normalized_base, path)
