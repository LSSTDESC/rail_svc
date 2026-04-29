"""Band creation operations."""

from ..db import Band
from .row_create import RowCreateBase, RowCreateContext

__all__ = ["band_creator"]

band_creator: RowCreateBase[Band] = RowCreateBase(RowCreateContext.from_db_class(Band))
