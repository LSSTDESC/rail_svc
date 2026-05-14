"""
Wrapper classes for CatEstimator objects from the RAIL framework.

This module provides abstraction layers for working with CatEstimator objects,
supporting both PDF-based and ensemble-based estimation workflows.
"""

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import qp
from ceci.stage import PipelineStage
from rail.estimation.estimator import CatEstimator
from rail.utils import catalog_utils

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="CatEstimatorWrapperBase")


class CatEstimatorWrapperBase(ABC):
    """
    Abstract base class for wrapping CatEstimator objects.

    This class provides a common interface for different types of estimator
    wrappers, with a factory method for instantiation and abstract methods
    for implementation-specific behavior.
    """

    @classmethod
    @abstractmethod
    def _build_wrapper(
        cls: type[T],
        estim_class: type[CatEstimator],
        **kwargs: Any,
    ) -> T:
        """
        Build a wrapper instance for the given estimator class.

        This abstract method must be implemented by subclasses to handle
        their specific initialization requirements.

        Parameters
        ----------
        estim_class
            The CatEstimator class to wrap.
        **kwargs
            Additional keyword arguments for estimator configuration.

        Returns
        -------
        CatEstimatorWrapperBase
            An instance of the wrapper subclass.
        """

    @abstractmethod
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Process data through the wrapped estimator.

        This abstract method defines the interface for using the wrapper.
        Subclasses must implement their specific calling conventions.

        Parameters
        ----------
        *args
            Positional arguments for the estimator.
        **kwargs
            Keyword arguments for the estimator.

        Returns
        -------
        Any
            Results from the estimator processing.
        """

    @classmethod
    def build_wrapper(
        cls: type[T],
        estim_name: str,
        estim_class_name: str,
        model_path: Path,
        catalog_tag: str | None = None,
        **kwargs: Any,
    ) -> T:
        """
        Factory method to build wrapper instances.

        This method handles the common setup required for all wrapper types,
        including dynamic module import, class retrieval, and catalog
        configuration.

        Parameters
        ----------
        estim_name
            Name identifier for the estimator (for logging/tracking).
        estim_class_name
            Fully qualified class name (e.g., 'rail.estimation.algos.SomeEstimator').
        model_path
            Path to the model file for the estimator.
        catalog_tag
            Optional catalog tag to apply via catalog_utils.
        **kwargs
            Additional keyword arguments passed to the estimator.

        Returns
        -------
        CatEstimatorWrapperBase
            An instance of the appropriate wrapper subclass.

        Raises
        ------
        ImportError
            If the specified module cannot be imported.
        ValueError
            If the specified class cannot be found in the module.

        Examples
        --------
        >>> wrapper = CatEstimatorPdfWrapper.build_wrapper(
        ...     estim_name="my_estimator",
        ...     estim_class_name="rail.estimation.algos.BPZ",
        ...     model_path=Path("models/bpz_model.pkl"),
        ...     catalog_tag="lsst_dp0"
        ... )
        """
        tokens = estim_class_name.split(".")
        class_name = tokens[-1]
        module_name = ".".join(tokens[:-1])

        # Dynamically import the module if not already loaded
        try:
            if module_name not in sys.modules:
                logger.info(f"Importing module: {module_name} for {estim_name}")
                __import__(module_name)
        except ImportError as e:
            raise ImportError(f"Cannot import module '{module_name}' for class '{class_name}': {e}") from e

        # Retrieve the estimator class
        try:
            estim_class = PipelineStage.get_stage(class_name, module_name)
        except Exception as e:
            raise ValueError(f"Cannot find class '{class_name}' in module '{module_name}': {e}") from e

        # Apply catalog tag if provided
        if catalog_tag is not None:
            logger.info(f"Applying catalog tag: {catalog_tag}")
            catalog_utils.apply(catalog_tag)

        # Prepare keyword arguments with model path
        all_kwargs = kwargs.copy()
        all_kwargs["model"] = str(model_path)

        return cls._build_wrapper(estim_class, **all_kwargs)


class CatEstimatorPdfWrapper(CatEstimatorWrapperBase):
    """
    Wrapper for CatEstimator that processes individual objects and returns PDFs.

    This wrapper is designed for interactive use cases where you want to
    process single objects or small batches and receive probability
    distribution functions (PDFs) as qp.Ensemble objects.

    Attributes
    ----------
    _estimator : CatEstimator
        The wrapped estimator instance.
    _names : list of str
        List of column names expected in the input data.
    """

    def __init__(
        self,
        cat_estimator: CatEstimator,
        names: list[str],
    ):
        """
        Initialize the PDF wrapper.

        Parameters
        ----------
        cat_estimator
            CatEstimator instance to wrap.
        names
            List of column names for input data (e.g., magnitudes and errors).

        Notes
        -----
        The constructor performs a dummy run with a single object to
        initialize the estimator's output handle. This ensures the
        estimator is fully configured for subsequent calls.
        """
        self._estimator = cat_estimator

        # Set up the estimator
        self._estimator.open_model(**self._estimator.config)
        self._estimator.data_store.clear()
        self._estimator._input_length = 1
        self._estimator._initialize_run()
        self._estimator.config.output_mode = "return"
        self._names = names.copy()

        # Process a single dummy object to create the output handle
        dummy_data = {name: np.array([1.0]) for name in names}
        self._estimator._process_chunk(0, 1, dummy_data, first=True)

    def __call__(self, vals: dict[str, np.ndarray] | np.ndarray, start: int = 0, end: int = 1) -> qp.Ensemble:
        """
        Process input data and return PDF estimates.

        This method accepts input data in multiple formats and processes it
        through the wrapped estimator to produce probability distribution
        function estimates.

        Parameters
        ----------
        vals
            Input data, either as:
            - dict mapping column names to numpy arrays
            - 2D numpy array with shape (n_features, n_objects)
            - 1D numpy array with shape (n_features,) for a single object
        start
            Starting index for chunk processing.
        end
            Ending index for chunk processing.

        Returns
        -------
        qp.Ensemble
            Ensemble of probability distributions for the processed objects.

        Raises
        ------
        TypeError
            If vals is not a dict or numpy array, or if array has wrong dimensions.
        RuntimeError
            If chunk processing fails.

        Examples
        --------
        Process a single object with dict input:

        >>> data = {
        ...     'mag_g': np.array([22.5]),
        ...     'mag_r': np.array([21.8]),
        ...     'mag_g_err': np.array([0.1]),
        ...     'mag_r_err': np.array([0.09])
        ... }
        >>> pdfs = wrapper(data)

        Process with array input:

        >>> data = np.array([[22.5, 21.8, 0.1, 0.09]])  # Single object
        >>> pdfs = wrapper(data)
        """
        # Convert array input to dict format if needed
        if isinstance(vals, np.ndarray):
            if vals.ndim == 2:
                # 2D array: each row is a feature
                if vals.shape[0] != len(self._names):
                    raise ValueError(f"Expected {len(self._names)} features, got {vals.shape[0]}")
                vals = {self._names[i]: vals[i, :] for i in range(vals.shape[0])}
            elif vals.ndim == 1:
                # 1D array: single object
                if vals.shape[0] != len(self._names):
                    raise ValueError(f"Expected {len(self._names)} features, got {vals.shape[0]}")
                vals = {self._names[i]: np.array([vals[i]]) for i in range(vals.shape[0])}
            else:
                raise TypeError(f"CatEstimatorPdfWrapper expects a 1D or 2D array, got {vals.ndim}D")
        elif not isinstance(vals, dict):
            raise TypeError(f"CatEstimatorPdfWrapper expects np.ndarray or dict, got {type(vals)}")

        # Process the chunk
        try:
            self._estimator._process_chunk(start, end, vals, first=False)
        except Exception as e:
            raise RuntimeError(f"Failed to process chunk [{start}:{end}]: {e}") from e

        # Retrieve and return the estimates
        estimates = self._estimator._output_handle.data
        return estimates

    @classmethod
    def _build_wrapper(
        cls,
        estim_class: type[CatEstimator],
        **kwargs: Any,
    ) -> CatEstimatorPdfWrapper:
        """
        Build a PDF wrapper instance.

        This method constructs the estimator with appropriate configuration
        and wraps it in a CatEstimatorPdfWrapper.

        Parameters
        ----------
        estim_class
            The CatEstimator class to instantiate.
        **kwargs
            Keyword arguments for estimator configuration.

        Returns
        -------
        CatEstimatorPdfWrapper
            Configured wrapper instance.

        Notes
        -----
        This method automatically determines the required column names
        from the active catalog tag, including both magnitude columns
        and their associated error columns.
        """
        # Ensure output mode is set to 'return'
        all_kwargs = kwargs.copy()
        all_kwargs["output_mode"] = "return"

        # Get column names from catalog tag
        names = list(catalog_utils.get_active_tag().band_name_dict().values())
        var_names = []
        var_names += names
        var_names += [f"{name}_err" for name in names]

        # Create estimator and wrapper
        estimator = estim_class.make_stage(**all_kwargs)
        wrapper = cls(estimator, var_names)
        return wrapper


class CatEstimatorEnsembleWrapper(CatEstimatorWrapperBase):
    """
    Wrapper for CatEstimator that processes complete catalogs.

    This wrapper is designed for batch processing of entire catalogs,
    reading from input files and writing results to output files. It
    provides a simpler interface for large-scale processing workflows.

    Attributes
    ----------
    _estimator : CatEstimator
        The wrapped estimator instance.
    """

    def __init__(
        self,
        cat_estimator: CatEstimator,
    ):
        """
        Initialize the ensemble wrapper.

        Parameters
        ----------
        cat_estimator
            CatEstimator instance to wrap.
        """
        self._estimator = cat_estimator

    def __call__(self, input_file: Path | str, output_file: Path | str) -> Any:
        """
        Process an entire catalog from file.

        This method runs the estimator on a complete input catalog and
        writes the results to an output file.

        Parameters
        ----------
        input_file
            Path to the input catalog file.
        output_file
            Path where output results will be written.

        Returns
        -------
        Any
            Handle to the output data (typically a QPHandle).

        Notes
        -----
        This method clears the estimator's data store before processing
        to ensure a clean state for each run.

        Examples
        --------
        >>> wrapper = CatEstimatorEnsembleWrapper.build_wrapper(
        ...     estim_name="batch_estimator",
        ...     estim_class_name="rail.estimation.algos.FlexZBoost",
        ...     model_path=Path("models/flexz.pkl")
        ... )
        >>> output_handle = wrapper(
        ...     input_file="catalogs/input.hdf5",
        ...     output_file="results/output.hdf5"
        ... )
        """
        # Clear any previous data
        self._estimator.data_store.clear()

        # Configure input and output
        self._estimator.add_handle("input", path=str(input_file))
        self._estimator.config.output = str(output_file)

        # Run the estimator
        logger.info(f"Processing catalog: {input_file} -> {output_file}")
        self._estimator.run()

        return self._estimator._output_handle

    @classmethod
    def _build_wrapper(
        cls,
        estim_class: type[CatEstimator],
        **kwargs: Any,
    ) -> CatEstimatorEnsembleWrapper:
        """
        Build an ensemble wrapper instance.

        This method constructs the estimator and wraps it in a
        CatEstimatorEnsembleWrapper for batch processing.

        Parameters
        ----------
        estim_class
            The CatEstimator class to instantiate.
        **kwargs
            Keyword arguments for estimator configuration.

        Returns
        -------
        CatEstimatorEnsembleWrapper
            Configured wrapper instance.
        """
        estimator = estim_class.make_stage(**kwargs)
        wrapper = cls(estimator)
        return wrapper
