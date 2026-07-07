"""Domain routers for rail_svc tables."""

from __future__ import annotations

import logging
from pathlib import Path

import tables_io
from fastapi import Body, HTTPException, Query, status
from fastapi.responses import FileResponse
from pydantic import ValidationError

from macon.router.base import create_table_router

from .. import local_async, models
from macon.common import LoadType, str_to_slice, unexpected
from macon.config import config as global_config

logger = logging.getLogger(__name__)

algo_router = create_table_router("algorithm", local_async.algorithm)

band_router = create_table_router("band", local_async.band)

catalog_band_assoc_router = create_table_router("catalog_band_assoc", local_async.catalog_band_assoc)

catalog_tag_router = create_table_router("catalog_tag", local_async.catalog_tag)

dataset_router = create_table_router("dataset", local_async.dataset)

estimates_router = create_table_router("estimates", local_async.estimates)

estimator_router = create_table_router("estimator", local_async.estimator)

model_router = create_table_router("model", local_async.model)

filter_ab_router = create_table_router("filter_ab", local_async.filter_ab)

sed_router = create_table_router("sed", local_async.sed)


@dataset_router.post("/load", response_model=models.Dataset, status_code=status.HTTP_201_CREATED)
async def dataset_load(
    path: Path | str = Body(...),
    load_type: LoadType = Body(default=LoadType.in_place),
    data: dict = Body(default_factory=dict),
    *,
    validate: bool = Query(default=True, description="Whether to validate data"),
) -> models.Dataset:
    try:
        result = await local_async.dataset.load(
            path=path,
            load_type=load_type,
            validate=validate,
            **data,
        )
        return result
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception("Error loading dataset")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@dataset_router.get("/read_slice/{row_id}")
async def dataset_read_slice(
    row_id: int,
    read_slice: str | None = Query(default=None, description="Slice"),
) -> dict:
    try:
        slice_obj = str_to_slice(read_slice)
        data = await local_async.dataset.read_slice(row=row_id, the_slice=slice_obj)  # type: ignore[call-arg]
        json_table = tables_io.convert(data, tables_io.types.JSON_STRING)
        return {"data": json_table}
    except Exception as uexc:
        logger.exception(f"Error reading slice from dataset {row_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@dataset_router.get("/download/{row_id}")
async def dataset_download(
    row_id: int,
    output_path: str | None = Query(default=None, description="Optional output path"),
) -> FileResponse:
    try:
        archive_dir = Path(global_config.storage.archive)
        result = await local_async.dataset.get_row(row_id)
        if unexpected(result is None):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Dataset not found")
        source_path = archive_dir / result.path
        if unexpected(not source_path.exists()):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset file not found at {source_path}"
            )
        dest_path = output_path if output_path else result.path
        return FileResponse(path=source_path, filename=dest_path, media_type="application/octet-stream")
    except HTTPException:  # pragma: no cover
        raise
    except Exception as uexc:
        logger.exception(f"Error downloading dataset {row_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@estimates_router.post("/load", response_model=models.Estimates, status_code=status.HTTP_201_CREATED)
async def estimates_load(
    path: Path | str = Body(...),
    load_type: LoadType = Body(default=LoadType.in_place),
    data: dict = Body(default_factory=dict),
    *,
    validate: bool = Query(default=True, description="Whether to validate data"),
) -> models.Estimates:
    try:
        result = await local_async.estimates.load(
            path=path,
            load_type=load_type,
            validate=validate,
            **data,
        )
        return result
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception("Error loading estimates")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@estimates_router.get("/read_slice/{row_id}")
async def estimates_read_slice(
    row_id: int,
    read_slice: str | None = Query(default=None, description="Slice"),
) -> dict:
    try:
        slice_obj = str_to_slice(read_slice)
        data = await local_async.estimates.read_slice(row=row_id, the_slice=slice_obj)  # type: ignore[call-arg]
        json_tables = data.to_json()
        return json_tables
    except Exception as uexc:
        logger.exception(f"Error reading slice from estimates {row_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@estimates_router.get("/download/{row_id}")
async def estimates_download(
    row_id: int,
    output_path: str | None = Query(default=None, description="Optional output path"),
) -> FileResponse:
    try:
        archive_dir = Path(global_config.storage.archive)
        result = await local_async.estimates.get_row(row_id)
        if unexpected(result is None):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Estimates not found")
        source_path = archive_dir / result.path
        if unexpected(not source_path.exists()):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Estimates file not found at {source_path}"
            )
        dest_path = output_path if output_path else result.path
        return FileResponse(path=source_path, filename=dest_path, media_type="application/octet-stream")
    except HTTPException:  # pragma: no cover
        raise
    except Exception as uexc:
        logger.exception(f"Error downloading estimates {row_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@model_router.post("/load", response_model=models.Model, status_code=status.HTTP_201_CREATED)
async def model_load(
    path: Path | str = Body(...),
    load_type: LoadType = Body(default=LoadType.in_place),
    data: dict = Body(default_factory=dict),
    *,
    validate: bool = Query(default=True, description="Whether to validate data"),
) -> models.Model:
    try:
        result = await local_async.model.load(
            path=path,
            load_type=load_type,
            validate=validate,
            **data,
        )
        return result
    except ValidationError as uexc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Validation error", "details": uexc.errors()},
        ) from uexc
    except Exception as uexc:
        logger.exception("Error loading model")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


@model_router.get("/download/{row_id}")
async def model_download(
    row_id: int,
    output_path: str | None = Query(default=None, description="Optional output path"),
) -> FileResponse:
    try:
        archive_dir = Path(global_config.storage.archive)
        result = await local_async.model.get_row(row_id)
        if unexpected(result is None):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Model not found")
        source_path = archive_dir / result.path
        if unexpected(not source_path.exists()):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Model file not found at {source_path}"
            )
        dest_path = output_path if output_path else result.path
        return FileResponse(path=dest_path, filename=source_path.name, media_type="application/octet-stream")
    except HTTPException:  # pragma: no cover
        raise
    except Exception as uexc:
        logger.exception(f"Error downloading model {row_id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(uexc)) from uexc


all_routers = [
    algo_router,
    band_router,
    catalog_band_assoc_router,
    catalog_tag_router,
    dataset_router,
    estimates_router,
    estimator_router,
    filter_ab_router,
    model_router,
    sed_router,
]
