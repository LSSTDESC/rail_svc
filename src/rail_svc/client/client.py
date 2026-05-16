from typing import Any, TypeVar
import logging
from contextlib import asynccontextmanager

import httpx
from pydantic import BaseModel, ValidationError

from ..db_funcs.filter import Filter, OrderBy
from .. import models

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic operations
T = TypeVar("T")  # Database model type
ResponseT = TypeVar("ResponseT", bound=BaseModel)  # Response schema type
CreateT = TypeVar("CreateT", bound=BaseModel)  # Create schema type


from typing import Final
from .base import RemoteTableOperations, RemoteAPI

# Define table configuration
TABLE_CONFIGS: Final[dict[str, tuple[type[BaseModel], type[BaseModel]]]] = {
    "algorithms": (models.Algorithm, models.AlgorithmCreate),
    "bands": (models.Band, models.BandCreate),
    "catalog_band_assocs": (models.CatalogBandAssoc, models.CatalogBandAssocCreate),
    "catalog_tags": (models.CatalogTag, models.CatalogTagCreate),
    "datasets": (models.Dataset, models.DatasetCreate),
    "estimates": (models.Estimate, models.EstimateCreate),
    "estimators": (models.Estimator, models.EstimatorCreate),
    "models": (models.Model, models.ModelCreate),
}


class RemoteDatabase:
    """High-level interface providing access to all table clients."""

    def __init__(
        self,
        base_url: str,
        api_prefix: str = "/api/v1",
        timeout: float = 30.0,
        auth_token: str | None = None,
    ):
        self.base_url = base_url
        self.api_prefix = api_prefix
        self.timeout = timeout
        self.auth_token = auth_token
        self._api: RemoteAPI | None = None

    async def __aenter__(self):
        self._api = RemoteAPI(
            base_url=self.base_url,
            api_prefix=self.api_prefix,
            timeout=self.timeout,
            auth_token=self.auth_token,
        )
        await self._api.__aenter__()
        self._setup_clients()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._api:
            await self._api.__aexit__(exc_type, exc_val, exc_tb)

    def _setup_clients(self):
        """Setup all table clients dynamically."""
        assert self._api is not None, "API not initialized"

        for table_name, (model, create_model) in TABLE_CONFIGS.items():
            client = self._api.table(table_name, model, create_model)
            setattr(self, table_name, client)

    def list_tables(self) -> list[str]:
        """List all available table names."""
        return list(TABLE_CONFIGS.keys())

    def get_client(self, table_name: str) -> RemoteTableOperations | None:
        """Get a client by table name."""
        return getattr(self, table_name, None)
