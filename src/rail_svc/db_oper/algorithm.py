"""
Algorithm table operations.

Provides CRUD operations and Pydantic conversions for the Algorithm table.
"""

from .. import db, models
from .base import create_operations

__all__ = ["algorithm"]

algorithm = create_operations(
    db.Algorithm,
    models.Algorithm,
    models.AlgorithmCreate,
)
