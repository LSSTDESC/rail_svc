from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import httpx

from ..models import (
    RemoteAPIError,
    EstimatePdfRequest,
    EstimateEnsembleRequest,
    EstimateEnsembleResponse,
    LoadCatalogYamlRequest,
    LoadCatalogYamlResponse,
    GetDatasetAndEstimatesResponse,
    GetDataAndEstimatesDataResponse,
    CreateMatchedDatasetRequest,
    EstimatePdfForSliceRequest,
    EstimateDatasetRequest,
)

logger = logging.getLogger(__name__)


class RemoteFuncsOperations:
    """Remote client for funcs operations via HTTP API.

    This class provides the same interface as the local funcs module but executes
    operations against a remote FastAPI server via HTTP requests.

    Parameters
    ----------
    client : httpx.AsyncClient
        Shared HTTP client instance
    endpoint : str
        Base endpoint URL for the funcs operations (e.g., "http://localhost:8000/funcs")

    Note
    ----
    This class expects to receive an already initialized httpx.AsyncClient.
    Use RemoteAPI context manager to manage the client lifecycle.
    """

    def __init__(self, client: httpx.AsyncClient, endpoint: str):
        """Initialize the remote funcs operations client."""
        self.client = client
        self.endpoint = endpoint.rstrip("/")

    def _handle_response(
        self, response: httpx.Response, expected_status: int = 200
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle HTTP response and raise appropriate errors.

        Parameters
        ----------
        response : httpx.Response
            The HTTP response object
        expected_status : int
            Expected status code (default: 200)

        Returns
        -------
        dict[str, Any] | list[dict[str, Any]]
            Parsed JSON response

        Raises
        ------
        RemoteAPIError
            If the response indicates an error
        """
        if response.status_code == expected_status:
            return response.json()

        # Try to parse error details
        try:
            error_data = response.json()
            error_msg = error_data.get("detail", "Unknown error")
            if isinstance(error_msg, dict):
                error_text = error_msg.get("error", "Unknown error")
                details = error_msg.get("details", "")
                if details:
                    error_text = f"{error_text}: {details}"
                error_msg = error_text
        except Exception:
            error_msg = response.text or f"HTTP {response.status_code}"

        raise RemoteAPIError(f"API request failed with status {response.status_code}: {error_msg}")

    async def estimate_pdf(self, estimator_id: int, dataset_id: int, row: int) -> dict[str, Any]:
        """Estimate PDF for a specific row in a dataset.

        Parameters
        ----------
        estimator_id : int
            ID of the estimator to use
        dataset_id : int
            ID of the dataset
        row : int
            Row index to estimate

        Returns
        -------
        dict[str, Any]
            Estimation result

        Raises
        ------
        RemoteAPIError
            If the API request fails
        """
        request = EstimatePdfRequest(estimator_id=estimator_id, dataset_id=dataset_id, row=row)
        response = await self.client.post(
            f"{self.endpoint}/estimate-pdf",
            json=request.model_dump(),
        )
        return cast(dict[str, Any], self._handle_response(response))

    async def estimate_ensemble(
        self, estimator_id: int, dataset_id: int, output_file_path: str
    ) -> EstimateEnsembleResponse:
        """Estimate ensemble for a dataset.

        Parameters
        ----------
        estimator_id : int
            ID of the estimator to use
        dataset_id : int
            ID of the dataset
        output_file_path : str
            Path where output file should be written

        Returns
        -------
        EstimateEnsembleResponse
            Response containing output file path and message

        Raises
        ------
        RemoteAPIError
            If the API request fails
        ValidationError
            If the response data is invalid
        """
        request = EstimateEnsembleRequest(
            estimator_id=estimator_id,
            dataset_id=dataset_id,
            output_file_path=output_file_path,
        )
        response = await self.client.post(
            f"{self.endpoint}/estimate-ensemble",
            json=request.model_dump(),
        )
        result = cast(dict[str, Any], self._handle_response(response))
        return EstimateEnsembleResponse(**result)

    async def get_estimators_for_dataset(self, dataset_id: int) -> list[dict[str, Any]]:
        """Get all estimators for a given dataset.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset

        Returns
        -------
        list[dict[str, Any]]
            List of estimator data

        Raises
        ------
        RemoteAPIError
            If the API request fails
        """
        response = await self.client.get(f"{self.endpoint}/get-estimators-for-dataset/{dataset_id}")
        return cast(list[dict[str, Any]], self._handle_response(response))

    async def load_catalog_yaml(
        self, catalog_yaml: Path, filter_dir: Path | None = None
    ) -> LoadCatalogYamlResponse:
        """Load catalog from YAML file.

        Parameters
        ----------
        catalog_yaml : Path
            Path to catalog YAML file
        filter_dir : Path | None
            Optional path to filter directory

        Returns
        -------
        LoadCatalogYamlResponse
            Response containing bands, catalog tags, and associations

        Raises
        ------
        RemoteAPIError
            If the API request fails
        ValidationError
            If the response data is invalid
        """
        request = LoadCatalogYamlRequest(
            catalog_yaml=str(catalog_yaml),
            filter_dir=str(filter_dir) if filter_dir else None,
        )
        response = await self.client.post(
            f"{self.endpoint}/load-catalog-yaml",
            json=request.model_dump(),
        )
        result = cast(dict[str, Any], self._handle_response(response))
        return LoadCatalogYamlResponse(**result)

    async def get_dataset_and_estimates(self, dataset_id: int) -> GetDatasetAndEstimatesResponse:
        """Get dataset and its estimates.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset

        Returns
        -------
        GetDatasetAndEstimatesResponse
            Response containing dataset and estimates

        Raises
        ------
        RemoteAPIError
            If the API request fails
        ValidationError
            If the response data is invalid
        """
        response = await self.client.get(f"{self.endpoint}/get-dataset-and-estimates/{dataset_id}")
        result = cast(dict[str, Any], self._handle_response(response))
        return GetDatasetAndEstimatesResponse(**result)

    async def get_data_and_estimates_data(self, dataset_id: int, row: int) -> GetDataAndEstimatesDataResponse:
        """Get data and estimates data for a specific row.

        Parameters
        ----------
        dataset_id : int
            ID of the dataset
        row : int
            Row index

        Returns
        -------
        GetDataAndEstimatesDataResponse
            Response containing data and estimates data

        Raises
        ------
        RemoteAPIError
            If the API request fails
        ValidationError
            If the response data is invalid
        """
        response = await self.client.get(f"{self.endpoint}/get-data-and-estimates-data/{dataset_id}/{row}")
        result = cast(dict[str, Any], self._handle_response(response))
        return GetDataAndEstimatesDataResponse(**result)

    async def create_matched_dataset(
        self,
        matched_dataset_name: str,
        catalog_tag_name: str,
        component_dataset_names: list[str],
        path: str,
        n_objects: int | None = None,
    ) -> dict[str, Any]:
        """Create a matched dataset from component datasets.

        Parameters
        ----------
        matched_dataset_name : str
            Name for the matched dataset
        catalog_tag_name : str
            Name of the catalog tag
        component_dataset_names : list[str]
            List of component dataset names
        path : str
            Path where matched dataset should be stored
        n_objects : int | None
            Optional number of objects to include

        Returns
        -------
        dict[str, Any]
            Created matched dataset data

        Raises
        ------
        RemoteAPIError
            If the API request fails
        """
        request = CreateMatchedDatasetRequest(
            matched_dataset_name=matched_dataset_name,
            catalog_tag_name=catalog_tag_name,
            component_dataset_names=component_dataset_names,
            path=path,
            n_objects=n_objects,
        )
        response = await self.client.post(
            f"{self.endpoint}/create-matched-dataset",
            json=request.model_dump(),
        )
        return cast(dict[str, Any], self._handle_response(response))

    async def estimate_pdf_for_slice(
        self,
        estimator_id: int,
        dataset_id: int,
        the_slice: str,
        *,
        recompute_if_exists: bool = False,
    ) -> dict[str, Any]:
        """Estimate PDF for a slice of the dataset.

        Parameters
        ----------
        estimator_id : int
            ID of the estimator to use
        dataset_id : int
            ID of the dataset
        the_slice : str
            String representation of the slice (e.g., "0:100")
        recompute_if_exists : bool
            Whether to recompute if results already exist

        Returns
        -------
        dict[str, Any]
            Estimation result

        Raises
        ------
        RemoteAPIError
            If the API request fails
        """
        request = EstimatePdfForSliceRequest(
            estimator_id=estimator_id,
            dataset_id=dataset_id,
            the_slice=the_slice,
            recompute_if_exists=recompute_if_exists,
        )
        response = await self.client.post(
            f"{self.endpoint}/estimate-pdf-for-slice",
            json=request.model_dump(),
        )
        return cast(dict[str, Any], self._handle_response(response))

    async def estimate_dataset(
        self, estimator_id: int, dataset_id: int, *, raise_if_exists: bool = True
    ) -> dict[str, Any]:
        """Estimate entire dataset.

        Parameters
        ----------
        estimator_id : int
            ID of the estimator to use
        dataset_id : int
            ID of the dataset
        raise_if_exists : bool
            Whether to raise error if estimate already exists

        Returns
        -------
        dict[str, Any]
            Estimation result

        Raises
        ------
        RemoteAPIError
            If the API request fails
        """
        request = EstimateDatasetRequest(
            estimator_id=estimator_id,
            dataset_id=dataset_id,
            raise_if_exists=raise_if_exists,
        )
        response = await self.client.post(
            f"{self.endpoint}/estimate-dataset",
            json=request.model_dump(),
        )
        return cast(dict[str, Any], self._handle_response(response))
