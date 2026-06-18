"""
Sed table operations.

Provides CRUD operations and Pydantic conversions for the Sed table.
"""

from .. import db, models
from .base import TableContext, TableOperations


class SedOperations(TableOperations[db.Sed, models.Sed, models.SedCreate]):
    """Create operations for Sed table."""


__all__ = ["SedOperations", "sed"]

# Module-level singleton
sed: SedOperations = SedOperations(TableContext.from_db_class(db.Sed))
