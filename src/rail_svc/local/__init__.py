"""Local operations API with automatic session management.

This module provides simplified table operations that automatically
manage database sessions and return Pydantic models.

Examples
--------
>>> from rail_svc.local import algorithm, dataset
>>>
>>> # Get single row
>>> algo = await algorithm.get_row(row_id=1)
>>>
>>> # Create new row  
>>> new_ds = await dataset.create_row(
...     path="data/catalog.h5",
...     catalog_tag_name="SDSS_DR16",
...     validate_file=True
... )
>>>
>>> # Filter rows
>>> from rail_svc.db_funcs.filter import Filter, FilterOp
>>> active_algos = await algorithm.filter_rows(
...     filters=[Filter("active", FilterOp.EQ, True)]
... )
"""

from .. import db_oper
from .base import create_local_operations

# Create local operations - each has all methods via dynamic binding
algorithm = create_local_operations(db_oper.algorithm)
band = create_local_operations(db_oper.band)
catalog_band_assoc = create_local_operations(db_oper.catalog_band_assoc)
catalog_tag = create_local_operations(db_oper.catalog_tag)
dataset = create_local_operations(db_oper.dataset)
estimates = create_local_operations(db_oper.estimates)
estimator = create_local_operations(db_oper.estimator)
model = create_local_operations(db_oper.model)

__all__ = [
    'algorithm',
    'band',
    'catalog_band_assoc',
    'catalog_tag',
    'dataset',
    'estimates',
    'estimator',
    'model',
]
