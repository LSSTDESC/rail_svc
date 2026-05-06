"""
CatalogTag table operations.

Provides CRUD operations and Pydantic conversions for the CatalogTag table.
"""

from .. import db, models
from .base import create_operations

__all__ = ["catalog_tag"]

catalog_tag = create_operations(
    db.CatalogTag,
    models.CatalogTag,
    models.CatalogTagCreate,
)
