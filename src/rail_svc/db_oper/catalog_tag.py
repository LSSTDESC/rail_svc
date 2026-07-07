"""
CatalogTag table operations.

Provides CRUD operations and Pydantic conversions for the CatalogTag table.
"""

from macon.db_oper.base import TableContext, TableOperations

from .. import db, models


class CatalogTagOperations(TableOperations[db.CatalogTag, models.CatalogTag, models.CatalogTagCreate]):
    """Create operations for CatalogTag table."""


__all__ = ["CatalogTagOperations", "catalog_tag"]

# Module-level singleton
catalog_tag: CatalogTagOperations = CatalogTagOperations(TableContext.from_db_class(db.CatalogTag))
