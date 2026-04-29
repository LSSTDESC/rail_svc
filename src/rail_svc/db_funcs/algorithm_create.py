"""Algorithm creation operations."""

from ..db import Algorithm
from .row_create import RowCreateBase, RowCreateContext

__all__ = ["algorithm_creator"]

algorithm_creator: RowCreateBase[Algorithm] = RowCreateBase(RowCreateContext.from_db_class(Algorithm))
