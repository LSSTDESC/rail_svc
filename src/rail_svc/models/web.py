"""Request/response models for the rail_svc API."""

from typing import Any

from macon.models.web import (
    AsyncRouteError,
    CountResponse,
    DeleteResponse,
    FilterRequest,
    FindRequest,
    LookupResponse,
    RemoteAPIError,
)
from pydantic import BaseModel

from .dataset import Dataset
from .estimates import Estimates

__all__ = [
    "AsyncRouteError",
    "CountResponse",
    "DeleteResponse",
    "FilterRequest",
    "FindRequest",
    "LookupResponse",
    "RemoteAPIError",
    "EstimatePdfRequest",
    "EstimateEnsembleRequest",
    "EstimateEnsembleResponse",
    "LoadCatalogYamlRequest",
    "LoadCatalogYamlResponse",
    "GetDatasetAndEstimatesResponse",
    "GetDataAndEstimatesDataResponse",
    "CreateMatchedDatasetRequest",
    "EstimatePdfForSliceRequest",
    "EstimateDatasetRequest",
]


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
    the_slice: str | None = None
    recompute_if_exists: bool = False


class EstimateDatasetRequest(BaseModel):
    """Request model for estimate_dataset function."""

    estimator_id: int
    dataset_id: int
    raise_if_exists: bool = False
