"""CatalogTag creation operations."""

from ..db import CatalogTag
from .row_create import RowCreateBase, RowCreateContext

__all__ = ["catalog_tag_creator"]

catalog_tag_creator: RowCreateBase[CatalogTag] = RowCreateBase(RowCreateContext.from_db_class(CatalogTag))
