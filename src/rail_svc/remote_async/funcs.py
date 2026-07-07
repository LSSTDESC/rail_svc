from __future__ import annotations

import types
import warnings
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from .. import models
from ..client.funcs import RemoteFuncsOperations
from ..client.rail_svc import RemoteAPI

# Type variable
F = TypeVar("F", bound=Callable[..., Any])


def with_funcs_client[F: Callable[..., Any]](func: F) -> F:
    """Decorator that injects the remote funcs client into async methods.

    Gets or creates a RemoteFuncsOperations client and passes the call through.
    The decorated method should accept `client` as its first argument after `self`.

    Parameters
    ----------
    func : Callable
        Async method that operates on a RemoteFuncsOperations client

    Returns
    -------
    Callable
        Wrapped method that automatically gets the client

    Examples
    --------
    >>> @with_funcs_client
    >>> async def estimate_pdf(self, client: RemoteFuncsOperations, *args, **kwargs) -> dict:
    ...     return await client.estimate_pdf(*args, **kwargs)
    """

    @wraps(func)
    async def wrapper(self: AsyncRemoteFuncs, *args: Any, **kwargs: Any) -> Any:
        client = await self.get_client()
        return await func(self, client, *args, **kwargs)

    return wrapper  # type: ignore


class AsyncRemoteFuncs:
    """Async wrapper for remote funcs operations with connection management.

    Provides async methods for function operations on remote API via HTTP.
    Can be used as an async context manager for efficient connection reuse across
    multiple operations, or as a regular class for single operations.

    Examples
    --------
    Using as async context manager (recommended for multiple operations):

    >>> async with AsyncRemoteFuncs(
    ...     base_url="http://api.example.com",
    ... ) as funcs:
    ...     result = await funcs.estimate_pdf(estimator_id=1, dataset_id=2, row=0)
    ...     estimators = await funcs.get_estimators_for_dataset(dataset_id=2)

    Using for single operations:

    >>> funcs = AsyncRemoteFuncs(base_url="http://api.example.com")
    >>> result = await funcs.estimate_pdf(estimator_id=1, dataset_id=2, row=0)
    """

    def __init__(
        self,
        base_url: str,
        api_prefix: str = "/api/v1",
        timeout: float = 30.0,
        auth_token: str | None = None,
    ) -> None:
        """Initialize the async remote funcs operations.

        Parameters
        ----------
        base_url : str
            The base URL of the remote API server
        api_prefix : str, optional
            API version prefix, by default "/api/v1"
        timeout : float, optional
            Request timeout in seconds, by default 30.0
        auth_token : str | None, optional
            Authentication token for API requests, by default None
        """
        self.base_url = base_url
        self.api_prefix = api_prefix
        self.timeout = timeout
        self.auth_token = auth_token
        self._api: RemoteAPI | None = None
        self._client: RemoteFuncsOperations | None = None
        self._owns_api = False
        self._has_warned = False

    async def __aenter__(self) -> AsyncRemoteFuncs:
        """Enter async context manager.

        Creates and initializes the RemoteAPI client for reuse across operations.

        Returns
        -------
        AsyncRemoteFuncs
            Self for context manager use
        """
        self._api = RemoteAPI(
            base_url=self.base_url,
            api_prefix=self.api_prefix,
            timeout=self.timeout,
            auth_token=self.auth_token,
        )
        await self._api.__aenter__()

        funcs_endpoint = f"{self.base_url}{self.api_prefix}/funcs"
        assert self._api.client
        self._client = RemoteFuncsOperations(
            client=self._api.client,
            endpoint=funcs_endpoint,
        )
        self._owns_api = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Properly closes the RemoteAPI client connection."""
        if self._api and self._owns_api:
            await self._api.__aexit__(exc_type, exc_val, exc_tb)
            self._api = None
            self._client = None
            self._owns_api = False

    async def get_client(self) -> RemoteFuncsOperations:
        """Get or create the funcs operations client.

        If used within a context manager, returns the existing client.
        Otherwise, creates a temporary client for single-use operations.

        Returns
        -------
        RemoteFuncsOperations
            The funcs operations client

        Warnings
        --------
        If creating a temporary client (not using context manager), a warning
        is issued the first time. This is inefficient for multiple operations.

        Notes
        -----
        For best performance with multiple operations, use this class as an
        async context manager to reuse connections:

        >>> async with AsyncRemoteFuncs(...) as funcs:
        ...     await funcs.estimate_pdf(...)
        ...     await funcs.estimate_ensemble(...)
        """
        if self._client is not None:
            return self._client

        # Warn on first temporary client creation
        if not self._has_warned:
            warnings.warn(
                f"Creating temporary client for {self.__class__.__name__}. "
                "For better performance with multiple operations, use as async context manager: "
                "'async with AsyncRemoteFuncs(...) as funcs:'",
                ResourceWarning,
                stacklevel=3,
            )
            self._has_warned = True

        # Create temporary API and client for single operation
        api = RemoteAPI(
            base_url=self.base_url,
            api_prefix=self.api_prefix,
            timeout=self.timeout,
            auth_token=self.auth_token,
        )

        funcs_endpoint = f"{self.base_url}{self.api_prefix}/funcs"
        assert api.client

        return RemoteFuncsOperations(
            client=api.client,
            endpoint=funcs_endpoint,
        )

    # Funcs operations

    @with_funcs_client
    async def estimate_pdf(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Estimate PDF for a specific row in a dataset.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.estimate_pdf
        **kwargs : Any
            Keyword arguments passed to client.estimate_pdf

        Returns
        -------
        dict[str, Any]
            Estimation result
        """
        return await client.estimate_pdf(*args, **kwargs)

    @with_funcs_client
    async def estimate_ensemble(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> models.EstimateEnsembleResponse:
        """Estimate ensemble for a dataset.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.estimate_ensemble
        **kwargs : Any
            Keyword arguments passed to client.estimate_ensemble

        Returns
        -------
        EstimateEnsembleResponse
            Response containing output file path and message
        """
        return await client.estimate_ensemble(*args, **kwargs)

    @with_funcs_client
    async def get_estimators_for_dataset(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Get all estimators for a given dataset.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.get_estimators_for_dataset
        **kwargs : Any
            Keyword arguments passed to client.get_estimators_for_dataset

        Returns
        -------
        list[dict[str, Any]]
            List of estimator data
        """
        return await client.get_estimators_for_dataset(*args, **kwargs)

    @with_funcs_client
    async def load_catalog_yaml(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> models.LoadCatalogYamlResponse:
        """Load catalog from YAML file.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.load_catalog_yaml
        **kwargs : Any
            Keyword arguments passed to client.load_catalog_yaml

        Returns
        -------
        LoadCatalogYamlResponse
            Response containing bands, catalog tags, and associations
        """
        return await client.load_catalog_yaml(*args, **kwargs)

    @with_funcs_client
    async def get_dataset_and_estimates(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> models.GetDatasetAndEstimatesResponse:
        """Get dataset and its estimates.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.get_dataset_and_estimates
        **kwargs : Any
            Keyword arguments passed to client.get_dataset_and_estimates

        Returns
        -------
        GetDatasetAndEstimatesResponse
            Response containing dataset and estimates
        """
        return await client.get_dataset_and_estimates(*args, **kwargs)

    @with_funcs_client
    async def get_data_and_estimates_data(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> models.GetDataAndEstimatesDataResponse:
        """Get data and estimates data for a specific row.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.get_data_and_estimates_data
        **kwargs : Any
            Keyword arguments passed to client.get_data_and_estimates_data

        Returns
        -------
        GetDataAndEstimatesDataResponse
            Response containing data and estimates data
        """
        return await client.get_data_and_estimates_data(*args, **kwargs)

    @with_funcs_client
    async def create_matched_dataset(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Create a matched dataset from component datasets.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.create_matched_dataset
        **kwargs : Any
            Keyword arguments passed to client.create_matched_dataset

        Returns
        -------
        dict[str, Any]
            Created matched dataset data
        """
        return await client.create_matched_dataset(*args, **kwargs)

    @with_funcs_client
    async def estimate_pdf_for_slice(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Estimate PDF for a slice of the dataset.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.estimate_pdf_for_slice
        **kwargs : Any
            Keyword arguments passed to client.estimate_pdf_for_slice

        Returns
        -------
        dict[str, Any]
            Estimation result
        """
        return await client.estimate_pdf_for_slice(*args, **kwargs)

    @with_funcs_client
    async def estimate_dataset(
        self,
        client: RemoteFuncsOperations,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Estimate entire dataset.

        Parameters
        ----------
        *args : Any
            Positional arguments passed to client.estimate_dataset
        **kwargs : Any
            Keyword arguments passed to client.estimate_dataset

        Returns
        -------
        dict[str, Any]
            Estimation result
        """
        return await client.estimate_dataset(*args, **kwargs)
