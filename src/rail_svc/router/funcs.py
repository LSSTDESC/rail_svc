from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from macon.common import str_to_slice
from pydantic import ValidationError

from .. import local_async
from ..models import (
    CreateMatchedDatasetRequest,
    Dataset,
    DatasetAssoc,
    EstimateDatasetRequest,
    EstimateEnsembleRequest,
    EstimateEnsembleResponse,
    EstimatePdfForSliceRequest,
    EstimatePdfRequest,
    Estimator,
    GetDataAndEstimatesDataResponse,
    GetDatasetAndEstimatesResponse,
    LoadCatalogYamlRequest,
    LoadCatalogYamlResponse,
)

logger = logging.getLogger(__name__)

funcs_router = APIRouter(prefix="/funcs", tags=["funcs"])


@funcs_router.post("/estimate-pdf")
async def estimate_pdf(request: EstimatePdfRequest) -> dict[str, Any]:
    """Estimate PDF for a specific row in a dataset."""
    try:
        result = await local_async.funcs.estimate_pdf(  # pylint: disable=no-value-for-parameter
            estimator_id=request.estimator_id,
            dataset_id=request.dataset_id,
            row=request.row,
        )  # type: ignore
        return result
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception(
            "Failed to estimate PDF for estimator_id=%s, dataset_id=%s, row=%s",
            request.estimator_id,
            request.dataset_id,
            request.row,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@funcs_router.post("/estimate-ensemble", response_model=EstimateEnsembleResponse)
async def estimate_ensemble(request: EstimateEnsembleRequest) -> EstimateEnsembleResponse:
    """Estimate ensemble for a dataset."""
    try:
        result = await local_async.funcs.estimate_ensemble(  # pylint: disable=no-value-for-parameter
            estimator_id=request.estimator_id,
            dataset_id=request.dataset_id,
            output_file_path=request.output_file_path,
        )  # type: ignore
        return EstimateEnsembleResponse(output_file=str(result), message=f"Wrote data to {result}")
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception(
            "Failed to estimate ensemble for estimator_id=%s, dataset_id=%s",
            request.estimator_id,
            request.dataset_id,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@funcs_router.get("/get-estimators-for-dataset/{dataset_id}")
async def get_estimators_for_dataset(dataset_id: int) -> list[Estimator]:
    """Get all estimators for a given dataset."""
    try:
        result = await local_async.funcs.get_estimators_for_dataest(  # pylint: disable=no-value-for-parameter
            dataset_id=dataset_id,
        )  # type: ignore
        # Convert to dict for JSON response
        return result
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as exc:
        logger.exception("Failed to get estimators for dataset_id=%s", dataset_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@funcs_router.post("/load-catalog-yaml", response_model=LoadCatalogYamlResponse)
async def load_catalog_yaml(request: LoadCatalogYamlRequest) -> LoadCatalogYamlResponse:
    """Load catalog from YAML file."""
    try:
        (
            bands,
            catalog_tags,
            catalog_band_assocs,
        ) = await local_async.funcs.load_catalog_yaml(  # pylint: disable=no-value-for-parameter
            catalog_yaml=Path(request.catalog_yaml),
            filter_dir=Path(request.filter_dir) if request.filter_dir else None,
        )  # type: ignore
        return LoadCatalogYamlResponse(
            bands=[b.model_dump() if hasattr(b, "model_dump") else dict(b) for b in bands],
            catalog_tags=[ct.model_dump() if hasattr(ct, "model_dump") else dict(ct) for ct in catalog_tags],
            catalog_band_assocs=[
                cba.model_dump() if hasattr(cba, "model_dump") else dict(cba) for cba in catalog_band_assocs
            ],
        )
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as exc:
        logger.exception("Failed to load catalog YAML from %s", request.catalog_yaml)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@funcs_router.get("/get-dataset-and-estimates/{dataset_id}", response_model=GetDatasetAndEstimatesResponse)
async def get_dataset_and_estimates(dataset_id: int) -> GetDatasetAndEstimatesResponse:
    """Get dataset and its estimates."""
    try:
        (
            the_dataset,
            the_estimates,
        ) = await local_async.funcs.get_dataset_and_estimates(  # pylint: disable=no-value-for-parameter
            dataset_id=dataset_id,
        )  # type: ignore
        return GetDatasetAndEstimatesResponse(
            dataset=the_dataset,
            estimates=the_estimates,
        )
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception("Failed to get dataset and estimates for dataset_id=%s", dataset_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@funcs_router.get(
    "/get-data-and-estimates-data/{dataset_id}/{row}", response_model=GetDataAndEstimatesDataResponse
)
async def get_data_and_estimates_data(dataset_id: int, row: int) -> GetDataAndEstimatesDataResponse:
    """Get data and estimates data for a specific row."""
    try:
        (
            data,
            the_estimates_dict,
        ) = await local_async.funcs.get_data_and_estimates_data(  # pylint: disable=no-value-for-parameter
            dataset_id=dataset_id,
            row=row,
        )  # type: ignore
        return GetDataAndEstimatesDataResponse(
            data=data,
            estimates_dict=the_estimates_dict,
        )
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as exc:
        logger.exception("Failed to get data and estimates data for dataset_id=%s, row=%s", dataset_id, row)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@funcs_router.post("/create-matched-dataset")
async def create_matched_dataset(request: CreateMatchedDatasetRequest) -> tuple[Dataset, list[DatasetAssoc]]:
    """Create a matched dataset from component datasets."""
    try:
        result = await local_async.funcs.create_matched_dataset(  # pylint: disable=no-value-for-parameter
            matched_dataset_name=request.matched_dataset_name,
            catalog_tag_name=request.catalog_tag_name,
            component_dataset_names=request.component_dataset_names,
            path=request.path,
            n_objects=request.n_objects,
        )  # type: ignore
        return result
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception(
            "Failed to create matched dataset '%s' with catalog_tag '%s'",
            request.matched_dataset_name,
            request.catalog_tag_name,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@funcs_router.post("/estimate-pdf-for-slice")
async def estimate_pdf_for_slice(request: EstimatePdfForSliceRequest) -> dict[str, Any]:
    """Estimate PDF for a slice of the dataset."""
    try:
        the_slice = str_to_slice(request.the_slice)
        result = await local_async.funcs.estimate_pdf_for_slice(  # pylint: disable=no-value-for-parameter
            estimator_id=request.estimator_id,
            dataset_id=request.dataset_id,
            the_slice=the_slice,
            recompute_if_exists=request.recompute_if_exists,
        )  # type: ignore
        return result
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception(
            "Failed to estimate PDF for slice for estimator_id=%s, dataset_id=%s, slice=%s",
            request.estimator_id,
            request.dataset_id,
            request.the_slice,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@funcs_router.post("/estimate-dataset")
async def estimate_dataset(request: EstimateDatasetRequest) -> dict[str, Any]:
    """Estimate entire dataset."""
    try:
        result = await local_async.funcs.estimate_dataset(  # pylint: disable=no-value-for-parameter
            estimator_id=request.estimator_id,
            dataset_id=request.dataset_id,
            raise_if_exists=request.raise_if_exists,
        )  # type: ignore
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as exc:
        logger.exception(
            "Failed to estimate dataset for estimator_id=%s, dataset_id=%s",
            request.estimator_id,
            request.dataset_id,
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
