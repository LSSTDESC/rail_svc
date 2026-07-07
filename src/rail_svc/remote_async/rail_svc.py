"""Domain-specific async remote operations for rail_svc tables."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from macon.remote_async.base import AsyncRemoteOperations, with_client

__all__ = ["AsyncRemoteOperations", "with_client"]

from .. import models
from ..client.rail_svc import (
    RemoteDatasetOperations,
    RemoteEstimatesOperations,
    RemoteModelOperations,
)


class AsyncRemoteDatasetOperations(AsyncRemoteOperations[models.Dataset, models.DatasetCreate]):
    """Extended async remote operations for Dataset table."""

    @with_client
    async def load(self, client: RemoteDatasetOperations, *args: Any, **kwargs: Any) -> models.Dataset:
        return await client.load(*args, **kwargs)

    @with_client
    async def read_slice(self, client: RemoteDatasetOperations, *args: Any, **kwargs: Any) -> Any:
        return await client.read_slice(*args, **kwargs)

    @with_client
    async def download(self, client: RemoteDatasetOperations, *args: Any, **kwargs: Any) -> Path:
        return await client.download(*args, **kwargs)


class AsyncRemoteEstimatesOperations(AsyncRemoteOperations[models.Estimates, models.EstimatesCreate]):
    """Extended async remote operations for Estimates table."""

    @with_client
    async def load(self, client: RemoteEstimatesOperations, *args: Any, **kwargs: Any) -> models.Estimates:
        return await client.load(*args, **kwargs)

    @with_client
    async def read_slice(self, client: RemoteEstimatesOperations, *args: Any, **kwargs: Any) -> Any:
        return await client.read_slice(*args, **kwargs)

    @with_client
    async def download(self, client: RemoteEstimatesOperations, *args: Any, **kwargs: Any) -> Path:
        return await client.download(*args, **kwargs)


class AsyncRemoteModelOperations(AsyncRemoteOperations[models.Model, models.ModelCreate]):
    """Extended async remote operations for Model table."""

    @with_client
    async def load(self, client: RemoteModelOperations, *args: Any, **kwargs: Any) -> models.Model:
        return await client.load(*args, **kwargs)

    @with_client
    async def download(self, client: RemoteModelOperations, *args: Any, **kwargs: Any) -> Path:
        return await client.download(*args, **kwargs)
