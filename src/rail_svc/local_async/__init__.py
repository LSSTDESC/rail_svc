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
>>> from macon.models.filtering import Filter, FilterOp
>>> active_algos = await algorithm.filter_rows(
...     filters=[Filter("active", FilterOp.EQ, True)]
... )
"""

from .. import db_oper
from . import funcs
from .base import (
    AlgorithmLocalOperations,
    BandLocalOperations,
    CatalogBandAssocLocalOperations,
    CatalogTagLocalOperations,
    DatasetAssocLocalOperations,
    DatasetLocalOperations,
    EstimatesLocalOperations,
    EstimatorLocalOperations,
    FilterABLocalOperations,
    LocalOperations,
    ModelLocalOperations,
    SedLocalOperations,
)

# Create local operations - each has all methods via dynamic binding
algorithm = AlgorithmLocalOperations(db_oper.algorithm)
band = BandLocalOperations(db_oper.band)
catalog_band_assoc = CatalogBandAssocLocalOperations(db_oper.catalog_band_assoc)
catalog_tag = CatalogTagLocalOperations(db_oper.catalog_tag)
dataset = DatasetLocalOperations(db_oper.dataset)
dataset_assoc = DatasetAssocLocalOperations(db_oper.dataset_assoc)
estimates = EstimatesLocalOperations(db_oper.estimates)
estimator = EstimatorLocalOperations(db_oper.estimator)
model = ModelLocalOperations(db_oper.model)
filter_ab = FilterABLocalOperations(db_oper.filter_ab)
sed = SedLocalOperations(db_oper.sed)

__all__ = [
    "LocalOperations",
    "algorithm",
    "band",
    "catalog_band_assoc",
    "catalog_tag",
    "dataset",
    "dataset_assoc",
    "estimates",
    "estimator",
    "filter_ab",
    "funcs",
    "model",
    "sed",
]
