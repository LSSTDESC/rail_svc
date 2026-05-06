"""
Band table operations.

Provides CRUD operations and Pydantic conversions for the Band table.
"""

from .. import db, models
from .base import create_operations

__all__ = ["band"]

band = create_operations(
    db.Band,
    models.Band,
    models.BandCreate,
)
