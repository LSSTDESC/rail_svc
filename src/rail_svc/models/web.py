"""Responsex model for load_catalog_yaml function."""

from typing import Any, TypeVar

from pydantic import BaseModel

from .filtering import Filter, OrderBy
from .dataset import Dataset
from .estimates import Estimates

ResponseT = TypeVar("ResponseT", bound=BaseModel)  # Response schema type


class AsyncRouteError(Exception):
    """Custom exception for async route handling errors."""


class RemoteAPIError(Exception):
    """Custom exception for remote API errors."""


class CountResponse(BaseModel):
    """Response model for count operations."""

    count: int


class LookupResponse[ResponseT](BaseModel):
    """Response model for lookup operations."""

    id: int
    data: ResponseT


class DeleteResponse(BaseModel):
    """Response model for delete operations."""

    deleted: bool = True


class FilterRequest(BaseModel):
    """Request model for filter operations."""

    filters: list[Filter] = []
    logical_op: str = "and"
    order_by: OrderBy | list[OrderBy] | None = None
    skip: int = 0
    limit: int | None = None


class FindRequest(BaseModel):
    """Request model for find operations."""

    order_by: OrderBy | list[OrderBy] | None = None
    skip: int = 0
    limit: int | None = None

    class ConfigDict:
        """pydantic config"""

        extra = "allow"  # Allow additional fields for query params


class EstimatePdfRequest(BaseModel):
    """Request model for estimate_pdf function."""

    estimator_id: int
    dataset_id: int
    row: int


class EstimateEnsembleRequest(BaseModel):
    """Request model for estimate_ensemble function."""

    estimator_id: int
    dataset_id: int
    output_file_path: str


class EstimateEnsembleResponse(BaseModel):
    """Response model for estimate_ensemble function."""

    output_file: str
    message: str


class LoadCatalogYamlRequest(BaseModel):
    """Request model for load_catalog_yaml function."""

    catalog_yaml: str
    filter_dir: str | None = None


class LoadCatalogYamlResponse(BaseModel):
    """Response model for load_catalog_yaml function."""

    bands: list[dict[str, Any]]
    catalog_tags: list[dict[str, Any]]
    catalog_band_assocs: list[dict[str, Any]]


class GetDatasetAndEstimatesResponse(BaseModel):
    """Response model for get_dataset_and_estimates function."""

    dataset: Dataset
    estimates: dict[str, Estimates]


class GetDataAndEstimatesDataResponse(BaseModel):
    """Response model for get_data_and_estimates_data function."""

    data: dict[str, Any]
    estimates_dict: dict[str, Any]


class CreateMatchedDatasetRequest(BaseModel):
    """Request model for create_matched_dataset function."""

    matched_dataset_name: str
    catalog_tag_name: str
    component_dataset_names: list[str]
    path: str | None = None
    n_objects: int


class EstimatePdfForSliceRequest(BaseModel):
    """Request model for estimate_pdf_for_slice function."""

    estimator_id: int
    dataset_id: int
    the_slice: str | None = None  # Will be parsed into slice
    recompute_if_exists: bool = False


class EstimateDatasetRequest(BaseModel):
    """Request model for estimate_dataset function."""

    estimator_id: int
    dataset_id: int
    raise_if_exists: bool = False
