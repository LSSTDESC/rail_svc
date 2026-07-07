"""Domain-specific remote client operations for rail_svc tables."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import qp

from macon.client.base import (
    RemoteAPI,
    RemoteFileOperations,
    RemoteTableOperations,
)

__all__ = [
    "RemoteAPI",
    "RemoteFileOperations",
    "RemoteTableOperations",
    "RemoteDatasetOperations",
    "RemoteEstimatesOperations",
    "RemoteModelOperations",
]

from .. import models
from macon.common import slice_to_str


class RemoteDatasetOperations(RemoteFileOperations[models.Dataset, models.DatasetCreate]):
    """Extended remote client for Dataset table with custom operations."""

    _default_filename_prefix = "dataset"

    async def read_slice(
        self,
        row_id: int,
        the_slice: slice | int | None = None,
    ) -> dict[str, np.ndarray]:
        """Read a slice of data from a dataset."""
        params = dict(read_slice=slice_to_str(the_slice))
        response = await self.client.get(
            f"{self.endpoint}/read_slice/{row_id}",
            params=params,
        )
        result = cast(dict[str, Any], self._handle_response(response, expected_status=200))
        out_data = json.loads(result["data"])
        return out_data


class RemoteEstimatesOperations(RemoteFileOperations[models.Estimates, models.EstimatesCreate]):
    """Extended remote client for Estimates table with custom operations."""

    _default_filename_prefix = "estimates"

    async def read_slice(
        self,
        row_id: int,
        the_slice: slice | int | None = None,
    ) -> dict[str, np.ndarray]:
        """Read a slice of data from estimates."""
        params = dict(read_slice=slice_to_str(the_slice))
        response = await self.client.get(
            f"{self.endpoint}/read_slice/{row_id}",
            params=params,
        )
        result = cast(dict[str, Any], self._handle_response(response))
        return qp.from_json(result)


class RemoteModelOperations(RemoteFileOperations[models.Model, models.ModelCreate]):
    """Extended remote client for Model table with custom operations."""

    _default_filename_prefix = "model"
