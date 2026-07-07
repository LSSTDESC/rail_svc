"""Domain-specific local operations for rail_svc tables."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import qp
from macon.local_async.base import (
    LocalOperations,
    to_pydantic,
    to_pydantic_list,
    to_pydantic_or_none,
    with_session,
    with_session_transaction,
)

__all__ = [
    "LocalOperations",
    "to_pydantic",
    "to_pydantic_list",
    "to_pydantic_or_none",
    "with_session",
    "with_session_transaction",
    "AlgorithmLocalOperations",
    "BandLocalOperations",
    "CatalogBandAssocLocalOperations",
    "CatalogTagLocalOperations",
    "DatasetLocalOperations",
    "DatasetAssocLocalOperations",
    "EstimatesLocalOperations",
    "EstimatorLocalOperations",
    "ModelLocalOperations",
    "FilterABLocalOperations",
    "SedLocalOperations",
]

from .. import db, models
from ..db_oper.dataset import DatasetOperations
from ..db_oper.estimates import EstimatesOperations
from ..db_oper.model import ModelOperations


class AlgorithmLocalOperations(LocalOperations[db.Algorithm, models.Algorithm, models.AlgorithmCreate]):
    """Operations on local DB for Algorithm table."""


class BandLocalOperations(LocalOperations[db.Band, models.Band, models.BandCreate]):
    """Operations on local DB for Band table."""


class CatalogBandAssocLocalOperations(
    LocalOperations[db.CatalogBandAssoc, models.CatalogBandAssoc, models.CatalogBandAssocCreate]
):
    """Operations on local DB for CatalogBandAssoc table."""


class CatalogTagLocalOperations(LocalOperations[db.CatalogTag, models.CatalogTag, models.CatalogTagCreate]):
    """Operations on local DB for CatalogTag table."""


class DatasetLocalOperations(LocalOperations[db.Dataset, models.Dataset, models.DatasetCreate]):
    """Operations on local DB for Dataset table."""

    @with_session_transaction
    @to_pydantic
    async def load(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await cast(DatasetOperations, self._table_ops).load(session, *args, **kwargs)

    @with_session
    async def read_slice(self, session: Any, *args: Any, **kwargs: Any) -> dict[str, np.ndarray]:
        return await cast(DatasetOperations, self._table_ops).read_slice(session, *args, **kwargs)


class DatasetAssocLocalOperations(
    LocalOperations[db.DatasetAssoc, models.DatasetAssoc, models.DatasetAssocCreate]
):
    """Operations on local DB for DatasetAssoc table."""


class EstimatesLocalOperations(LocalOperations[db.Estimates, models.Estimates, models.EstimatesCreate]):
    """Operations on local DB for Estimates table."""

    @with_session_transaction
    @to_pydantic
    async def load(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await cast(EstimatesOperations, self._table_ops).load(session, *args, **kwargs)

    @with_session
    async def read_slice(self, session: Any, *args: Any, **kwargs: Any) -> qp.Ensemble:
        return await cast(EstimatesOperations, self._table_ops).read_slice(session, *args, **kwargs)


class EstimatorLocalOperations(LocalOperations[db.Estimator, models.Estimator, models.EstimatorCreate]):
    """Operations on local DB for Estimator table."""


class ModelLocalOperations(LocalOperations[db.Model, models.Model, models.ModelCreate]):
    """Operations on local DB for Model table."""

    @with_session_transaction
    @to_pydantic
    async def load(self, session: Any, *args: Any, **kwargs: Any) -> Any:
        return await cast(ModelOperations, self._table_ops).load(session, *args, **kwargs)


class FilterABLocalOperations(LocalOperations[db.FilterAB, models.FilterAB, models.FilterABCreate]):
    """Operations on local DB for FilterAB table."""


class SedLocalOperations(LocalOperations[db.Sed, models.Sed, models.SedCreate]):
    """Operations on local DB for Sed table."""
