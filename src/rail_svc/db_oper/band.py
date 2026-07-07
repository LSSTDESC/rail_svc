"""
Band table operations.

Provides CRUD operations and Pydantic conversions for the Band table.
"""

from macon.db_oper.base import TableContext, TableOperations

from .. import db, models


class BandOperations(TableOperations[db.Band, models.Band, models.BandCreate]):
    """Create operations for Band table."""


__all__ = ["BandOperations", "band"]

# Module-level singleton
band: BandOperations = BandOperations(TableContext.from_db_class(db.Band))
