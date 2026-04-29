"""Common configuration parameters for pz-rail-service related packages"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Configuration", "config"]

load_dotenv()


class WebInterfaceConfiguration(BaseModel):
    """Configuration for web client"""

    default_page_size: int = Field(
        description="Default page size for pagination",
        default=100,
    )

    max_page_size : int = Field(
        description="Max page size for pagination",
        default=1000,
    )

    default_batch_size: int = Field(
        description="Default size for batch operations",
        default=1000,
    )

    max_batch_size : int = Field(
        description="Max size for batch operations",
        default=10000,
    )

    default_timeout: float = Field(
        description="Default query timeout (seconds)",
        default=30.,
    )

    stream_timeout: float = Field(
        description="Default streaming timeout (seconds)",
        default=60.,
    )


class AsgiConfiguration(BaseModel):
    """Configuration for the application's ASGI web server."""

    title: str = Field(
        description="Title of the ASGI application",
        default="rail-svc",
    )

    host: str = Field(
        description=(
            "The host address to which the asgi server should bind. "
            "WARNING: 0.0.0.0 binds to all interfaces and may be insecure in production."
        ),
        default="0.0.0.0",
    )

    port: int = Field(
        description="Port number for the asgi server to listen on",
        default=8080,
        ge=1,
        le=65535,
    )

    prefix: str = Field(
        description="The URL prefix for the pz-rail-service API",
        default="/rail-svc",
    )

    frontend_prefix: str = Field(
        description="The URL prefix for the frontend web app",
        default="/rail-svc",
    )

    reload: bool = Field(
        description="Whether to support ASGI server reload on content change.",
        default=True,
    )


class LoggingConfiguration(BaseModel):
    """Configuration for the application's logging facility."""

    handle: str = Field(
        default="rail-svc",
        title="Handle or name of the root logger",
    )

    level: str = Field(
        default="INFO",
        title="Log level of the application's logger",
    )

    profile: str = Field(
        default="development",
        title="Application logging profile",
    )

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v_upper


class DaemonConfiguration(BaseModel):
    """Settings for the Daemon nested model.

    Set according to DAEMON__FIELD environment variables.
    """

    processing_interval: int = Field(
        default=30,
        ge=1,
        description=(
            "The maximum wait time (seconds) between daemon processing intervals "
            "and the minimum time between element processing attempts. This "
            "duration may be lengthened depending on the element type."
        ),
    )


class DatabaseConfiguration(BaseModel):
    """Database configuration nested model.

    Set according to DB__FIELD environment variables.
    """

    url: str = Field(
        default="sqlite:///rail_svc.db",
        description="The URL for the rail-svc database",
    )

    password: SecretStr | None = Field(
        default=None,
        description="The password for the rail-svc database",
    )

    table_schema: str | None = Field(
        default=None,
        description="Schema to use for rail-svc database",
    )

    echo: bool = Field(
        default=False,
        description="SQLAlchemy engine echo setting for the rail-svc database",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate database URL scheme"""
        if v and not v.startswith(("postgresql://", "sqlite://", "mysql://", "postgresql+psycopg2://")):
            raise ValueError(
                "Database URL must start with a valid scheme (postgresql://, sqlite://, mysql://)"
            )
        return v


class StorageConfiguration(BaseModel):
    """Database storage configuration nested model.

    Set according to STORAGE__FIELD environment variables.
    """

    archive: str = Field(
        default="archive",
        description="The path for the archived files for rail-svc database",
    )

    import_area: str = Field(
        default="import",
        description="The path for the import area for files for rail-svc database",
    )

    @field_validator("archive", "import_area")
    @classmethod
    def ensure_path_exists(cls, v: str) -> str:
        """Ensure storage path exists"""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Path {path} does not exist")
        if not path.is_dir():
            raise ValueError(f"Path {path} is not a directory")
        if not os.access(path, os.W_OK):
            raise ValueError(f"Path {path} is not writeable")
        return v


class Configuration(BaseSettings):
    """Configuration for pz-rail-service.

    Nested models may be consumed from environment variables named according to
    the pattern 'NESTED_MODEL__FIELD' or via any `validation_alias` applied to
    a field.
    """

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
        case_sensitive=False,
        extra="ignore",
    )

    # Nested Models
    web_interface: WebInterfaceConfiguration = WebInterfaceConfiguration()
    asgi: AsgiConfiguration = AsgiConfiguration()
    daemon: DaemonConfiguration = DaemonConfiguration()
    db: DatabaseConfiguration = DatabaseConfiguration()
    logging: LoggingConfiguration = LoggingConfiguration()
    storage: StorageConfiguration = StorageConfiguration()


config = Configuration()
"""Configuration for rail-svc."""
