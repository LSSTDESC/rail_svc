"""Unit tests for rail_svc.rail_funcs.estimation_funcs"""

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import qp

from rail_svc.rail_funcs import wrappers


class TestCatEstimatorWrapperBase:
    """Tests for CatEstimatorWrapperBase abstract class"""

    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract base class cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            wrappers.CatEstimatorWrapperBase("test_name")

    def test_estim_name_property(self):
        """Test estim_name property"""

        # Create a concrete subclass for testing
        class ConcreteWrapper(wrappers.CatEstimatorWrapperBase):
            def _build_wrapper(self, estim_name, estim_class, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                pass

        wrapper = ConcreteWrapper.__new__(ConcreteWrapper)
        wrapper._estim_name = "test_estimator"

        assert wrapper.estim_name == "test_estimator"

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.apply")
    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_build_wrapper_successful(self, mock_get_stage, mock_catalog_apply):
        """Test successful wrapper building"""
        # Create mock estimator class
        mock_estim_class = Mock()
        mock_get_stage.return_value = mock_estim_class

        # Create concrete subclass with _build_wrapper implemented
        class ConcreteWrapper(wrappers.CatEstimatorWrapperBase):
            @classmethod
            def _build_wrapper(cls, estim_name, estim_class, **kwargs):
                wrapper = cls.__new__(cls)
                wrapper._estim_name = estim_name
                wrapper._estim_class = estim_class
                return wrapper

            def __call__(self, *args, **kwargs):
                pass

        # Mock module already imported
        sys.modules["rail.estimation.algos"] = Mock()

        result = ConcreteWrapper.build_wrapper(
            estim_name="test_estimator",
            estim_class_name="rail.estimation.algos.BPZ",
            model_path=Path("/path/to/model.pkl"),
            catalog_tag="lsst_dp0",
        )

        assert result.estim_name == "test_estimator"
        assert result._estim_class == mock_estim_class
        mock_catalog_apply.assert_called_once_with("lsst_dp0")

    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_build_wrapper_module_import(self, mock_get_stage):
        """Test that module is imported if not already loaded"""
        mock_estim_class = Mock()
        mock_get_stage.return_value = mock_estim_class

        class ConcreteWrapper(wrappers.CatEstimatorWrapperBase):
            @classmethod
            def _build_wrapper(cls, estim_name, estim_class, **kwargs):
                wrapper = cls.__new__(cls)
                wrapper._estim_name = estim_name
                return wrapper

            def __call__(self, *args, **kwargs):
                pass

        # Ensure module not in sys.modules
        module_name = "rail.estimation.test_module"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with patch("builtins.__import__") as mock_import:
            _result = ConcreteWrapper.build_wrapper(
                estim_name="test",
                estim_class_name=f"{module_name}.TestEstimator",
                model_path=Path("/path/to/model.pkl"),
            )

            mock_import.assert_called_once_with(module_name)

    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_build_wrapper_import_error(self, mock_get_stage):
        """Test handling of module import errors"""

        class ConcreteWrapper(wrappers.CatEstimatorWrapperBase):
            @classmethod
            def _build_wrapper(cls, estim_name, estim_class, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                pass

        module_name = "nonexistent.module"
        if module_name in sys.modules:
            del sys.modules[module_name]

        with patch("builtins.__import__", side_effect=ImportError("Module not found")):
            with pytest.raises(ImportError, match="Cannot import module"):
                ConcreteWrapper.build_wrapper(
                    estim_name="test",
                    estim_class_name=f"{module_name}.TestClass",
                    model_path=Path("/path/to/model.pkl"),
                )

    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_build_wrapper_class_not_found(self, mock_get_stage):
        """Test handling when class cannot be found"""
        mock_get_stage.side_effect = Exception("Class not found")

        class ConcreteWrapper(wrappers.CatEstimatorWrapperBase):
            @classmethod
            def _build_wrapper(cls, estim_name, estim_class, **kwargs):
                pass

            def __call__(self, *args, **kwargs):
                pass

        sys.modules["rail.estimation.algos"] = Mock()

        with pytest.raises(ValueError, match="Cannot find class"):
            ConcreteWrapper.build_wrapper(
                estim_name="test",
                estim_class_name="rail.estimation.algos.NonExistentClass",
                model_path=Path("/path/to/model.pkl"),
            )

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.apply")
    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_build_wrapper_without_catalog_tag(self, mock_get_stage, mock_catalog_apply):
        """Test building wrapper without catalog tag"""
        mock_estim_class = Mock()
        mock_get_stage.return_value = mock_estim_class

        class ConcreteWrapper(wrappers.CatEstimatorWrapperBase):
            @classmethod
            def _build_wrapper(cls, estim_name, estim_class, **kwargs):
                wrapper = cls.__new__(cls)
                wrapper._estim_name = estim_name
                return wrapper

            def __call__(self, *args, **kwargs):
                pass

        sys.modules["rail.estimation.algos"] = Mock()

        _result = ConcreteWrapper.build_wrapper(
            estim_name="test",
            estim_class_name="rail.estimation.algos.BPZ",
            model_path=Path("/path/to/model.pkl"),
        )

        mock_catalog_apply.assert_not_called()

    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_build_wrapper_kwargs_passed_correctly(self, mock_get_stage):
        """Test that kwargs are passed correctly including model path"""
        mock_estim_class = Mock()
        mock_get_stage.return_value = mock_estim_class

        captured_kwargs = {}

        class ConcreteWrapper(wrappers.CatEstimatorWrapperBase):
            @classmethod
            def _build_wrapper(cls, estim_name, estim_class, **kwargs):
                captured_kwargs.update(kwargs)
                wrapper = cls.__new__(cls)
                wrapper._estim_name = estim_name
                return wrapper

            def __call__(self, *args, **kwargs):
                pass

        sys.modules["rail.estimation.algos"] = Mock()

        ConcreteWrapper.build_wrapper(
            estim_name="test",
            estim_class_name="rail.estimation.algos.BPZ",
            model_path=Path("/path/to/model.pkl"),
            custom_param="value",
        )

        assert "model" in captured_kwargs
        assert captured_kwargs["model"] == "/path/to/model.pkl"
        assert captured_kwargs["custom_param"] == "value"


class TestCatEstimatorPdfWrapper:
    """Tests for CatEstimatorPdfWrapper"""

    def create_mock_estimator(self):
        """Helper to create a mock CatEstimator with proper dict-like config"""
        mock_estimator = Mock()
        # Make config a MagicMock that can be unpacked with **
        mock_estimator.config = MagicMock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_estimator._output_handle.data = Mock()
        return mock_estimator

    def test_init_successful(self):
        """Test successful initialization"""
        mock_estimator = self.create_mock_estimator()
        names = ["mag_g", "mag_r", "mag_g_err", "mag_r_err"]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test_pdf", cat_estimator=mock_estimator, names=names
        )

        assert wrapper.estim_name == "test_pdf"
        assert wrapper._estimator == mock_estimator
        assert wrapper._names == names

    def test_init_sets_output_mode(self):
        """Test that initialization sets output mode to 'return'"""
        mock_estimator = self.create_mock_estimator()
        names = ["mag_g"]

        _wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        assert mock_estimator.config.output_mode == "return"

    def test_call_with_dict_input(self):
        """Test calling wrapper with dict input"""
        mock_estimator = self.create_mock_estimator()
        mock_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator._output_handle.data = mock_ensemble

        names = ["mag_g", "mag_r"]
        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        input_data = {"mag_g": np.array([22.5, 23.1]), "mag_r": np.array([21.8, 22.3])}

        result = wrapper(input_data)

        assert result == mock_ensemble

    def test_call_with_2d_array_input(self):
        """Test calling wrapper with 2D numpy array"""
        mock_estimator = self.create_mock_estimator()
        mock_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator._output_handle.data = mock_ensemble

        names = ["mag_g", "mag_r"]
        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        # 2D array: shape (n_features, n_objects)
        input_data = np.array([[22.5, 23.1], [21.8, 22.3]])

        result = wrapper(input_data)

        assert result == mock_ensemble

    def test_call_with_1d_array_input(self):
        """Test calling wrapper with 1D numpy array (single object)"""
        mock_estimator = self.create_mock_estimator()
        mock_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator._output_handle.data = mock_ensemble

        names = ["mag_g", "mag_r"]
        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        # 1D array for single object
        input_data = np.array([22.5, 21.8])

        result = wrapper(input_data)

        assert result == mock_ensemble

    def test_call_with_wrong_shape_2d_array(self):
        """Test error when 2D array has wrong number of features"""
        mock_estimator = self.create_mock_estimator()
        names = ["mag_g", "mag_r"]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        # Wrong number of features (3 instead of 2)
        input_data = np.array([[22.5], [21.8], [20.1]])

        with pytest.raises(ValueError, match="Expected 2 features"):
            wrapper(input_data)

    def test_call_with_wrong_shape_1d_array(self):
        """Test error when 1D array has wrong number of features"""
        mock_estimator = self.create_mock_estimator()
        names = ["mag_g", "mag_r"]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        # Wrong number of features
        input_data = np.array([22.5])

        with pytest.raises(ValueError, match="Expected 2 features"):
            wrapper(input_data)

    def test_call_with_3d_array_raises_error(self):
        """Test error when array has wrong number of dimensions"""
        mock_estimator = self.create_mock_estimator()
        names = ["mag_g"]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        # 3D array
        input_data = np.array([[[1, 2], [3, 4]]])

        with pytest.raises(TypeError, match="expects a 1D or 2D array"):
            wrapper(input_data)

    def test_call_with_invalid_type(self):
        """Test error when input is not dict or array"""
        mock_estimator = self.create_mock_estimator()
        names = ["mag_g"]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        with pytest.raises(TypeError, match="expects np.ndarray or dict"):
            wrapper("invalid input")

    def test_call_with_start_end_parameters(self):
        """Test calling with custom start and end indices"""
        mock_estimator = self.create_mock_estimator()
        mock_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator._output_handle.data = mock_ensemble

        names = ["mag_g"]
        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        input_data = {"mag_g": np.array([22.5, 23.1, 24.0])}

        result = wrapper(input_data, start=5, end=8)

        # Verify result is returned
        assert result == mock_ensemble

    def test_call_processing_failure(self):
        """Test error handling when chunk processing fails"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator._process_chunk.side_effect = [
            None,  # First call (dummy) succeeds
            Exception("Processing failed"),  # Second call fails
        ]

        names = ["mag_g"]
        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        input_data = {"mag_g": np.array([22.5])}

        with pytest.raises(RuntimeError, match="Failed to process chunk"):
            wrapper(input_data)

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag")
    def test_build_wrapper_successful(self, mock_get_active_tag):
        """Test successful wrapper building"""
        # Mock catalog tag
        mock_tag = Mock()
        mock_tag.band_name_dict.return_value = {"g": "mag_g", "r": "mag_r"}
        mock_get_active_tag.return_value = mock_tag

        # Mock estimator class
        mock_estim_class = Mock()
        mock_estimator_instance = self.create_mock_estimator()
        mock_estim_class.make_stage.return_value = mock_estimator_instance

        result = wrappers.CatEstimatorPdfWrapper._build_wrapper(
            estim_name="test_wrapper", estim_class=mock_estim_class, model="/path/to/model.pkl"
        )

        assert isinstance(result, wrappers.CatEstimatorPdfWrapper)
        assert result.estim_name == "test_wrapper"

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag")
    def test_build_wrapper_includes_error_columns(self, mock_get_active_tag):
        """Test that wrapper includes both mag and error columns"""
        mock_tag = Mock()
        mock_tag.band_name_dict.return_value = {"g": "mag_g", "r": "mag_r"}
        mock_get_active_tag.return_value = mock_tag

        mock_estim_class = Mock()
        mock_estimator_instance = self.create_mock_estimator()
        mock_estim_class.make_stage.return_value = mock_estimator_instance

        result = wrappers.CatEstimatorPdfWrapper._build_wrapper(
            estim_name="test", estim_class=mock_estim_class
        )

        # Should include both mag and mag_err columns
        expected_names = ["mag_g", "mag_r", "mag_g_err", "mag_r_err"]
        assert result._names == expected_names

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag")
    def test_build_wrapper_preserves_kwargs(self, mock_get_active_tag):
        """Test that kwargs are preserved and passed to estimator"""
        mock_tag = Mock()
        mock_tag.band_name_dict.return_value = {"g": "mag_g"}
        mock_get_active_tag.return_value = mock_tag

        mock_estim_class = Mock()
        mock_estimator_instance = self.create_mock_estimator()
        mock_estim_class.make_stage.return_value = mock_estimator_instance

        wrappers.CatEstimatorPdfWrapper._build_wrapper(
            estim_name="test",
            estim_class=mock_estim_class,
            model="/path/to/model.pkl",
            custom_param="value",
            another_param=42,
        )

        call_kwargs = mock_estim_class.make_stage.call_args[1]
        assert call_kwargs["custom_param"] == "value"
        assert call_kwargs["another_param"] == 42


class TestCatEstimatorEnsembleWrapper:
    """Tests for CatEstimatorEnsembleWrapper"""

    def create_mock_estimator(self):
        """Helper to create a mock CatEstimator with proper dict-like config"""
        mock_estimator = Mock()
        # Make config a MagicMock that can be unpacked with **
        mock_estimator.config = MagicMock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_estimator._output_handle.data = Mock()
        return mock_estimator

    def test_init_successful(self):
        """Test successful initialization"""
        mock_estimator = self.create_mock_estimator()

        wrapper = wrappers.CatEstimatorEnsembleWrapper(
            estim_name="test_ensemble", cat_estimator=mock_estimator
        )

        assert wrapper.estim_name == "test_ensemble"
        assert wrapper._estimator == mock_estimator

    def test_call_successful(self):
        """Test successful catalog processing"""
        mock_estimator = self.create_mock_estimator()
        mock_output_handle = Mock()
        mock_output_handle.path = "output.hdf5"
        mock_estimator._output_handle = mock_output_handle

        wrapper = wrappers.CatEstimatorEnsembleWrapper(estim_name="test", cat_estimator=mock_estimator)

        input_file = Path("/path/to/input.hdf5")
        output_file = Path("/path/to/output.hdf5")

        result = wrapper(input_file, output_file)

        assert result == mock_output_handle
        mock_estimator.data_store.clear.assert_called_once()
        mock_estimator.add_handle.assert_called_once_with("input", path=str(input_file))
        mock_estimator.run.assert_called_once()
        mock_estimator.finalize.assert_called_once()

    def test_call_with_string_paths(self):
        """Test calling with string paths instead of Path objects"""
        mock_estimator = self.create_mock_estimator()
        mock_output_handle = Mock()
        mock_output_handle.path = "output.hdf5"
        mock_estimator._output_handle = mock_output_handle

        wrapper = wrappers.CatEstimatorEnsembleWrapper(estim_name="test", cat_estimator=mock_estimator)

        input_file = "/path/to/input.hdf5"
        output_file = "/path/to/output.hdf5"

        result = wrapper(input_file, output_file)

        assert result == mock_output_handle
        mock_estimator.add_handle.assert_called_once_with("input", path=input_file)

    def test_build_wrapper_successful(self):
        """Test successful wrapper building"""
        mock_estim_class = Mock()
        mock_estimator_instance = self.create_mock_estimator()
        mock_estim_class.make_stage.return_value = mock_estimator_instance

        result = wrappers.CatEstimatorEnsembleWrapper._build_wrapper(
            estim_name="test_wrapper", estim_class=mock_estim_class, model="/path/to/model.pkl"
        )

        assert isinstance(result, wrappers.CatEstimatorEnsembleWrapper)
        assert result.estim_name == "test_wrapper"
        assert result._estimator == mock_estimator_instance

    def test_build_wrapper_passes_kwargs(self):
        """Test that kwargs are passed to make_stage"""
        mock_estim_class = Mock()
        mock_estimator_instance = self.create_mock_estimator()
        mock_estim_class.make_stage.return_value = mock_estimator_instance

        wrappers.CatEstimatorEnsembleWrapper._build_wrapper(
            estim_name="test",
            estim_class=mock_estim_class,
            model="/path/to/model.pkl",
            chunk_size=1000,
            custom_param="value",
        )

        call_kwargs = mock_estim_class.make_stage.call_args[1]
        assert call_kwargs["model"] == "/path/to/model.pkl"
        assert call_kwargs["chunk_size"] == 1000
        assert call_kwargs["custom_param"] == "value"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def create_mock_estimator(self):
        """Helper to create a mock CatEstimator with proper dict-like config"""
        mock_estimator = Mock()
        # Make config a MagicMock that can be unpacked with **
        mock_estimator.config = MagicMock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_estimator._output_handle.data = Mock()
        return mock_estimator

    def test_pdf_wrapper_with_empty_names_list(self):
        """Test PDF wrapper with empty column names list"""
        mock_estimator = self.create_mock_estimator()

        wrapper = wrappers.CatEstimatorPdfWrapper(estim_name="test", cat_estimator=mock_estimator, names=[])

        assert wrapper._names == []

    def test_pdf_wrapper_with_single_column(self):
        """Test PDF wrapper with only one column"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator._output_handle.data = Mock(spec=qp.Ensemble)

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=["mag_g"]
        )

        input_data = {"mag_g": np.array([22.5])}
        result = wrapper(input_data)

        assert result is not None

    def test_pdf_wrapper_with_many_columns(self):
        """Test PDF wrapper with many columns"""
        mock_estimator = self.create_mock_estimator()

        # Create many column names
        names = [f"col_{i}" for i in range(100)]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        assert len(wrapper._names) == 100

    def test_ensemble_wrapper_with_path_objects(self):
        """Test ensemble wrapper explicitly with Path objects"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator._output_handle.path = "output.hdf5"

        wrapper = wrappers.CatEstimatorEnsembleWrapper(estim_name="test", cat_estimator=mock_estimator)

        input_path = Path("/data/catalog.hdf5")
        output_path = Path("/results/output.hdf5")

        wrapper(input_path, output_path)

        mock_estimator.add_handle.assert_called_with("input", path=str(input_path))
        mock_estimator.run.assert_called_once()
        mock_estimator.finalize.assert_called_once()

    def test_pdf_wrapper_array_conversion_preserves_data(self):
        """Test that array to dict conversion preserves data correctly"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator._output_handle.data = Mock(spec=qp.Ensemble)

        names = ["mag_g", "mag_r", "mag_i"]
        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        # Create input data with specific values
        input_array = np.array(
            [
                [22.5, 23.1, 24.0],  # mag_g values
                [21.8, 22.3, 23.5],  # mag_r values
                [21.0, 21.5, 22.8],  # mag_i values
            ]
        )

        wrapper(input_array)

        # Get the data that was passed to _process_chunk
        passed_data = mock_estimator._process_chunk.call_args[0][2]

        # Verify conversion preserved data
        np.testing.assert_array_equal(passed_data["mag_g"], input_array[0, :])
        np.testing.assert_array_equal(passed_data["mag_r"], input_array[1, :])
        np.testing.assert_array_equal(passed_data["mag_i"], input_array[2, :])

    def test_pdf_wrapper_with_zero_length_arrays(self):
        """Test PDF wrapper with zero-length arrays"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator._output_handle.data = Mock(spec=qp.Ensemble)

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=["mag_g", "mag_r"]
        )

        # Empty arrays
        input_data = {"mag_g": np.array([]), "mag_r": np.array([])}

        result = wrapper(input_data)
        assert result is not None

    def test_pdf_wrapper_array_with_nan_values(self):
        """Test PDF wrapper handles NaN values in arrays"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator._output_handle.data = Mock(spec=qp.Ensemble)

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=["mag_g", "mag_r"]
        )

        # Arrays with NaN
        input_data = np.array([[22.5, np.nan], [21.8, 22.3]])

        _result = wrapper(input_data)

        # Verify NaN was passed through
        passed_data = mock_estimator._process_chunk.call_args[0][2]
        assert np.isnan(passed_data["mag_g"][1])

    def test_pdf_wrapper_array_with_inf_values(self):
        """Test PDF wrapper handles infinity values in arrays"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator._output_handle.data = Mock(spec=qp.Ensemble)

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=["mag_g"]
        )

        # Array with infinity
        input_data = {"mag_g": np.array([22.5, np.inf, -np.inf])}

        _result = wrapper(input_data)

        # Verify inf was passed through
        passed_data = mock_estimator._process_chunk.call_args[0][2]
        assert np.isinf(passed_data["mag_g"][1])
        assert np.isinf(passed_data["mag_g"][2])


class TestIntegration:
    """Integration tests combining multiple components"""

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.apply")
    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag")
    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_complete_pdf_wrapper_workflow(self, mock_get_stage, mock_get_tag, mock_apply):
        """Test complete workflow for PDF wrapper"""
        # Setup mocks
        mock_tag = Mock()
        mock_tag.band_name_dict.return_value = {"g": "mag_g", "r": "mag_r"}
        mock_get_tag.return_value = mock_tag

        mock_estimator_instance = Mock()
        mock_estimator_instance.config = MagicMock()
        mock_estimator_instance.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator_instance.config.keys = Mock(return_value=[])
        mock_estimator_instance.config.items = Mock(return_value=[])
        mock_estimator_instance.data_store = Mock()
        mock_estimator_instance._output_handle = Mock()
        mock_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator_instance._output_handle.data = mock_ensemble

        mock_estim_class = Mock()
        mock_estim_class.make_stage.return_value = mock_estimator_instance
        mock_get_stage.return_value = mock_estim_class

        sys.modules["rail.estimation.algos"] = Mock()

        # Build wrapper
        wrapper = wrappers.CatEstimatorPdfWrapper.build_wrapper(
            estim_name="integration_test",
            estim_class_name="rail.estimation.algos.BPZ",
            model_path=Path("/models/bpz.pkl"),
            catalog_tag="lsst_dp0",
        )

        # Use wrapper
        input_data = {
            "mag_g": np.array([22.5, 23.1]),
            "mag_r": np.array([21.8, 22.3]),
            "mag_g_err": np.array([0.1, 0.15]),
            "mag_r_err": np.array([0.08, 0.12]),
        }

        result = wrapper(input_data)

        # Verify
        assert result == mock_ensemble
        mock_apply.assert_called_once_with("lsst_dp0")

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag")
    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_complete_ensemble_wrapper_workflow(self, mock_get_stage, mock_get_tag):
        """Test complete workflow for ensemble wrapper"""
        # Setup mocks
        mock_estimator_instance = Mock()
        mock_estimator_instance.config = Mock()
        mock_estimator_instance.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator_instance.config.keys = Mock(return_value=[])
        mock_estimator_instance.config.items = Mock(return_value=[])
        mock_estimator_instance.data_store = Mock()
        mock_output_handle = Mock()
        mock_estimator_instance._output_handle = mock_output_handle

        mock_estim_class = Mock()
        mock_estim_class.make_stage.return_value = mock_estimator_instance
        mock_get_stage.return_value = mock_estim_class

        sys.modules["rail.estimation.algos"] = Mock()

        # Build wrapper
        wrapper = wrappers.CatEstimatorEnsembleWrapper.build_wrapper(
            estim_name="batch_test",
            estim_class_name="rail.estimation.algos.FlexZBoost",
            model_path=Path("/models/flexz.pkl"),
        )

        # Use wrapper
        result = wrapper(input_file=Path("/data/input.hdf5"), output_file=Path("/results/output.hdf5"))

        # Verify
        assert result == mock_output_handle
        mock_estimator_instance.run.assert_called_once()

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag")
    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_multiple_wrapper_instances(self, mock_get_stage, mock_get_tag):
        """Test creating multiple wrapper instances"""
        mock_tag = Mock()
        mock_tag.band_name_dict.return_value = {"g": "mag_g"}
        mock_get_tag.return_value = mock_tag

        # Create different mock estimators for each instance
        def create_estimator():
            mock = Mock()
            mock.config = Mock()
            mock.config.__iter__ = Mock(return_value=iter([]))
            mock.config.keys = Mock(return_value=[])
            mock.config.items = Mock(return_value=[])
            mock.data_store = Mock()
            mock._output_handle = Mock()
            mock._output_handle.data = Mock(spec=qp.Ensemble)
            return mock

        mock_estim_class = Mock()
        mock_estim_class.make_stage.side_effect = [create_estimator(), create_estimator(), create_estimator()]
        mock_get_stage.return_value = mock_estim_class

        sys.modules["rail.estimation.algos"] = Mock()

        # Create multiple wrappers
        wrapper1 = wrappers.CatEstimatorPdfWrapper.build_wrapper(
            estim_name="wrapper1",
            estim_class_name="rail.estimation.algos.BPZ",
            model_path=Path("/model1.pkl"),
        )

        wrapper2 = wrappers.CatEstimatorPdfWrapper.build_wrapper(
            estim_name="wrapper2",
            estim_class_name="rail.estimation.algos.BPZ",
            model_path=Path("/model2.pkl"),
        )

        wrapper3 = wrappers.CatEstimatorEnsembleWrapper.build_wrapper(
            estim_name="wrapper3",
            estim_class_name="rail.estimation.algos.FlexZBoost",
            model_path=Path("/model3.pkl"),
        )

        # Verify they are distinct instances
        assert wrapper1.estim_name == "wrapper1"
        assert wrapper2.estim_name == "wrapper2"
        assert wrapper3.estim_name == "wrapper3"
        assert isinstance(wrapper1, wrappers.CatEstimatorPdfWrapper)
        assert isinstance(wrapper2, wrappers.CatEstimatorPdfWrapper)
        assert isinstance(wrapper3, wrappers.CatEstimatorEnsembleWrapper)


class TestErrorHandling:
    """Tests for error handling and validation"""

    def create_mock_estimator(self):
        """Helper to create a mock CatEstimator with proper dict-like config"""
        mock_estimator = Mock()
        # Make config a MagicMock that can be unpacked with **
        mock_estimator.config = MagicMock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_estimator._output_handle.data = Mock()
        return mock_estimator

    def test_pdf_wrapper_handles_estimator_initialization_error(self):
        """Test handling of errors during estimator initialization"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator.open_model.side_effect = Exception("Failed to open model")

        with pytest.raises(Exception, match="Failed to open model"):
            wrappers.CatEstimatorPdfWrapper(estim_name="test", cat_estimator=mock_estimator, names=["mag_g"])

    def test_ensemble_wrapper_handles_run_error(self):
        """Test handling of errors during catalog processing"""
        mock_estimator = self.create_mock_estimator()
        mock_estimator.run.side_effect = Exception("Processing failed")

        wrapper = wrappers.CatEstimatorEnsembleWrapper(estim_name="test", cat_estimator=mock_estimator)

        with pytest.raises(Exception, match="Processing failed"):
            wrapper(Path("/input.hdf5"), Path("/output.hdf5"))

    def test_pdf_wrapper_names_list_copied(self):
        """Test that names list is copied, not referenced"""
        mock_estimator = self.create_mock_estimator()

        original_names = ["mag_g", "mag_r"]
        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=original_names
        )

        # Modify original list
        original_names.append("mag_i")

        # Wrapper should have original list
        assert wrapper._names == ["mag_g", "mag_r"]
        assert len(wrapper._names) == 2


class TestAbstractMethods:
    """Tests for abstract method implementations"""

    def test_pdf_wrapper_implements_call(self):
        """Test that PDF wrapper implements __call__"""
        assert hasattr(wrappers.CatEstimatorPdfWrapper, "__call__")
        assert callable(getattr(wrappers.CatEstimatorPdfWrapper, "__call__"))

    def test_pdf_wrapper_implements_build_wrapper(self):
        """Test that PDF wrapper implements _build_wrapper"""
        assert hasattr(wrappers.CatEstimatorPdfWrapper, "_build_wrapper")

    def test_ensemble_wrapper_implements_call(self):
        """Test that ensemble wrapper implements __call__"""
        assert hasattr(wrappers.CatEstimatorEnsembleWrapper, "__call__")
        assert callable(getattr(wrappers.CatEstimatorEnsembleWrapper, "__call__"))

    def test_ensemble_wrapper_implements_build_wrapper(self):
        """Test that ensemble wrapper implements _build_wrapper"""
        assert hasattr(wrappers.CatEstimatorEnsembleWrapper, "_build_wrapper")


class TestDataFlow:
    """Tests for data flow through wrappers"""

    def test_pdf_wrapper_data_passes_through_estimator(self):
        """Test that data correctly flows through the estimator"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()

        # Create a mock ensemble that we can verify
        expected_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator._output_handle.data = expected_ensemble

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=["mag_g", "mag_r"]
        )

        input_data = {"mag_g": np.array([22.5, 23.1]), "mag_r": np.array([21.8, 22.3])}

        result = wrapper(input_data)

        # Verify the result is the ensemble from the estimator
        assert result is expected_ensemble

    def test_ensemble_wrapper_returns_output_handle(self):
        """Test that ensemble wrapper returns the output handle"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()

        expected_handle = Mock()
        mock_estimator._output_handle = expected_handle

        wrapper = wrappers.CatEstimatorEnsembleWrapper(estim_name="test", cat_estimator=mock_estimator)

        result = wrapper(input_file=Path("/input.hdf5"), output_file=Path("/output.hdf5"))

        assert result is expected_handle

    def test_pdf_wrapper_preserves_numpy_array_dtype(self):
        """Test that array dtypes are preserved during conversion"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_estimator._output_handle.data = Mock(spec=qp.Ensemble)

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=["mag_g", "mag_r"]
        )

        # Use float32 array
        input_data = np.array([[22.5, 23.1], [21.8, 22.3]], dtype=np.float32)

        wrapper(input_data)

        # Get the data passed to _process_chunk
        passed_data = mock_estimator._process_chunk.call_args[0][2]

        # Verify dtype is preserved
        assert passed_data["mag_g"].dtype == np.float32
        assert passed_data["mag_r"].dtype == np.float32


class TestDocstringExamples:
    """Tests based on docstring examples"""

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.apply")
    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_docstring_example_build_wrapper(self, mock_get_stage, mock_apply):
        """Test example from build_wrapper docstring"""
        mock_class = Mock()
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_class.make_stage.return_value = mock_estimator
        mock_get_stage.return_value = mock_class

        sys.modules["rail.estimation.algos"] = Mock()

        with patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag"):
            mock_tag = Mock()
            mock_tag.band_name_dict.return_value = {"g": "mag_g"}

            with patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag", return_value=mock_tag):
                wrapper = wrappers.CatEstimatorPdfWrapper.build_wrapper(
                    estim_name="my_estimator",
                    estim_class_name="rail.estimation.algos.BPZ",
                    model_path=Path("models/bpz_model.pkl"),
                    catalog_tag="lsst_dp0",
                )

                assert isinstance(wrapper, wrappers.CatEstimatorPdfWrapper)
                assert wrapper.estim_name == "my_estimator"

    def test_docstring_example_pdf_wrapper_call_dict(self):
        """Test dict input example from __call__ docstring"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator._output_handle.data = mock_ensemble

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test",
            cat_estimator=mock_estimator,
            names=["mag_g", "mag_r", "mag_g_err", "mag_r_err"],
        )

        data = {
            "mag_g": np.array([22.5]),
            "mag_r": np.array([21.8]),
            "mag_g_err": np.array([0.1]),
            "mag_r_err": np.array([0.09]),
        }

        pdfs = wrapper(data)
        assert pdfs is mock_ensemble

    def test_docstring_example_pdf_wrapper_call_array(self):
        """Test array input example from __call__ docstring"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_ensemble = Mock(spec=qp.Ensemble)
        mock_estimator._output_handle.data = mock_ensemble

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test",
            cat_estimator=mock_estimator,
            names=["mag_g", "mag_r", "mag_g_err", "mag_r_err"],
        )

        data = np.array([[22.5], [21.8], [0.1], [0.09]])  # 4 features, 1 object
        pdfs = wrapper(data)
        assert pdfs is mock_ensemble

    @patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag")
    @patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage")
    def test_docstring_example_ensemble_wrapper(self, mock_get_stage, mock_get_tag):
        """Test example from ensemble wrapper docstring"""
        mock_class = Mock()
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_output_handle = Mock()
        mock_estimator._output_handle = mock_output_handle
        mock_class.make_stage.return_value = mock_estimator
        mock_get_stage.return_value = mock_class

        sys.modules["rail.estimation.algos"] = Mock()

        wrapper = wrappers.CatEstimatorEnsembleWrapper.build_wrapper(
            estim_name="batch_estimator",
            estim_class_name="rail.estimation.algos.FlexZBoost",
            model_path=Path("models/flexz.pkl"),
        )

        output_handle = wrapper(input_file="catalogs/input.hdf5", output_file="results/output.hdf5")

        assert output_handle is mock_output_handle


class TestSpecialCases:
    """Tests for special cases and corner conditions"""

    def test_pdf_wrapper_with_unicode_column_names(self):
        """Test PDF wrapper with unicode characters in column names"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()

        # Column names with unicode
        names = ["måg_g", "måg_r", "êrr_g", "êrr_r"]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        assert wrapper._names == names

    def test_pdf_wrapper_with_very_long_column_names(self):
        """Test PDF wrapper with very long column names"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()

        long_name = "a" * 1000
        names = [long_name]

        wrapper = wrappers.CatEstimatorPdfWrapper(
            estim_name="test", cat_estimator=mock_estimator, names=names
        )

        assert wrapper._names == [long_name]

    def test_ensemble_wrapper_with_relative_paths(self):
        """Test ensemble wrapper with relative paths"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator.config.keys = Mock(return_value=[])
        mock_estimator.config.items = Mock(return_value=[])
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()
        mock_estimator._output_handle.path = "output.hdf5"

        wrapper = wrappers.CatEstimatorEnsembleWrapper(estim_name="test", cat_estimator=mock_estimator)

        # Relative paths
        wrapper(input_file="./data/input.hdf5", output_file="../results/output.hdf5")

        mock_estimator.add_handle.assert_called_with("input", path="./data/input.hdf5")
        mock_estimator.run.assert_called_once()
        mock_estimator.finalize.assert_called_once()

    def test_build_wrapper_with_empty_kwargs(self):
        """Test building wrapper with no additional kwargs"""
        with patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage"):
            mock_class = Mock()
            mock_estimator = Mock()
            mock_estimator.config = Mock()
            mock_estimator.config.__iter__ = Mock(return_value=iter([]))
            mock_estimator.config.keys = Mock(return_value=[])
            mock_estimator.config.items = Mock(return_value=[])
            mock_estimator.data_store = Mock()
            mock_estimator._output_handle = Mock()
            mock_class.make_stage.return_value = mock_estimator

            with patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage", return_value=mock_class):
                sys.modules["rail.estimation.algos"] = Mock()

                result = wrappers.CatEstimatorEnsembleWrapper.build_wrapper(
                    estim_name="test",
                    estim_class_name="rail.estimation.algos.BPZ",
                    model_path=Path("/model.pkl"),
                )

                assert isinstance(result, wrappers.CatEstimatorEnsembleWrapper)

    def test_multiple_wrappers_dont_share_state(self):
        """Test that multiple wrappers don't share mutable state"""
        mock_estimator1 = Mock()
        mock_estimator1.config = Mock()
        mock_estimator1.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator1.config.keys = Mock(return_value=[])
        mock_estimator1.config.items = Mock(return_value=[])

        mock_estimator1.data_store = Mock()
        mock_estimator1._output_handle = Mock()

        mock_estimator2 = Mock()
        mock_estimator2.config = Mock()
        mock_estimator2.config.__iter__ = Mock(return_value=iter([]))
        mock_estimator2.config.keys = Mock(return_value=[])
        mock_estimator2.config.items = Mock(return_value=[])
        mock_estimator2.data_store = Mock()
        mock_estimator2._output_handle = Mock()

        wrapper1 = wrappers.CatEstimatorPdfWrapper(
            estim_name="wrapper1", cat_estimator=mock_estimator1, names=["mag_g"]
        )

        wrapper2 = wrappers.CatEstimatorPdfWrapper(
            estim_name="wrapper2", cat_estimator=mock_estimator2, names=["mag_r", "mag_i"]
        )

        # Verify they have different estimators and names
        assert wrapper1._estimator is not wrapper2._estimator
        assert wrapper1._names != wrapper2._names

    def test_build_wrapper_model_path_always_string(self):
        """Test that model path is always converted to string"""
        captured_kwargs = {}

        def capture(**kwargs):
            captured_kwargs.update(kwargs)
            mock = Mock()
            mock.config = Mock()
            mock.data_store = Mock()
            return mock

        with patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage"):
            mock_class = Mock()
            mock_class.make_stage = capture

            with patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage", return_value=mock_class):
                sys.modules["rail.estimation.algos"] = Mock()

                wrappers.CatEstimatorEnsembleWrapper.build_wrapper(
                    estim_name="test",
                    estim_class_name="rail.estimation.algos.BPZ",
                    model_path=Path("/path/to/model.pkl"),
                )

                # Model should be string, not Path
                assert isinstance(captured_kwargs.get("model"), str)
                assert captured_kwargs.get("model") == "/path/to/model.pkl"


class TestLogging:
    """Tests for logging behavior"""

    def test_build_wrapper_logs_module_import(self, caplog):
        """Test that module import is logged"""
        with caplog.at_level(logging.INFO):
            with patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage"):
                with patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag"):
                    module_name = "rail.estimation.test_log_module"
                    if module_name in sys.modules:
                        del sys.modules[module_name]

                    with patch("builtins.__import__"):
                        try:
                            wrappers.CatEstimatorPdfWrapper.build_wrapper(
                                estim_name="test",
                                estim_class_name=f"{module_name}.TestClass",
                                model_path=Path("/path/to/model.pkl"),
                            )
                        except Exception:
                            pass

                        assert "Importing module" in caplog.text

    def test_build_wrapper_logs_catalog_tag(self, caplog):
        """Test that catalog tag application is logged"""
        with caplog.at_level(logging.INFO):
            with patch("rail_svc.rail_funcs.wrappers.PipelineStage.get_stage"):
                with patch("rail_svc.rail_funcs.wrappers.catalog_utils.apply"):
                    with patch("rail_svc.rail_funcs.wrappers.catalog_utils.get_active_tag"):
                        sys.modules["rail.estimation.algos"] = Mock()

                        try:
                            wrappers.CatEstimatorPdfWrapper.build_wrapper(
                                estim_name="test",
                                estim_class_name="rail.estimation.algos.BPZ",
                                model_path=Path("/path/to/model.pkl"),
                                catalog_tag="lsst_dp0",
                            )
                        except Exception:
                            pass

                        assert "Applying catalog tag" in caplog.text

    def test_ensemble_wrapper_logs_processing(self, caplog):
        """Test that catalog processing is logged"""
        mock_estimator = Mock()
        mock_estimator.config = Mock()
        mock_estimator.data_store = Mock()
        mock_estimator._output_handle = Mock()

        wrapper = wrappers.CatEstimatorEnsembleWrapper(estim_name="test", cat_estimator=mock_estimator)

        with caplog.at_level(logging.INFO):
            wrapper(Path("/input.hdf5"), Path("/output.hdf5"))

            assert "Processing catalog" in caplog.text
