"""
Algorithm table operations.

Provides CRUD operations and Pydantic conversions for the Algorithm table.
"""

from .. import db, models
from .base import TableContext, TableOperations


class AlgorithmOperations(TableOperations[db.Algorithm, models.Algorithm, models.AlgorithmCreate]):
    """DB operations for Algorithm table."""


__all__ = ["AlgorithmOperations", "algorithm"]

# Module-level singleton
algorithm: AlgorithmOperations = AlgorithmOperations(TableContext.from_db_class(db.Algorithm))
