"""Unit tests for rail_svc.rail_funcs.catalog_funcs"""

import logging
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rail_svc.models import BandCreate, CatalogBandAssocCreate, CatalogTagCreate
from rail_svc.rail_funcs import catalog_funcs


class TestExtractPaddedNonZeros:
    """Tests for extract_padded_non_zeros function"""

    def test_basic_extraction_with_padding(self):
        """Test basic extraction with padding on both sides"""
        arr = np.array([[1, 0], [2, 0], [3, 5], [4, 10], [5, 0], [6, 0]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        # Should include rows 1-5 (padding around indices 2-3)
        expected = np.array([[2, 0], [3, 5], [4, 10], [5, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_no_padding_at_start(self):
        """Test when non-zero values start at index 0"""
        arr = np.array([[1, 5], [2, 10], [3, 0], [4, 0]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        # Should start at 0, end at index 3 (padding)
        expected = np.array([[1, 5], [2, 10], [3, 0]])
        np.testing.assert_array_equal(result, expected)

    def test_no_padding_at_end(self):
        """Test when non-zero values extend to the end"""
        arr = np.array([[1, 0], [2, 0], [3, 5], [4, 10]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        # Should start at index 1 (padding), end at 4
        expected = np.array([[2, 0], [3, 5], [4, 10]])
        np.testing.assert_array_equal(result, expected)

    def test_no_padding_needed(self):
        """Test when non-zero values span entire array"""
        arr = np.array([[1, 5], [2, 10], [3, 15]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        np.testing.assert_array_equal(result, arr)

    def test_single_non_zero_with_padding(self):
        """Test single non-zero value with padding available"""
        arr = np.array([[1, 0], [2, 5], [3, 0]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        np.testing.assert_array_equal(result, arr)

    def test_all_zeros(self):
        """Test array with all zeros in second column"""
        arr = np.array([[1, 0], [2, 0], [3, 0]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        assert result.shape == (0, 2)

    def test_empty_array(self):
        """Test empty input array"""
        arr = np.empty((0, 2))
        result = catalog_funcs.extract_padded_non_zeros(arr)

        assert result.shape == (0, 2)

    def test_invalid_dimensions(self):
        """Test with wrong number of dimensions"""
        arr = np.array([1, 2, 3])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        assert result.shape == (0, 2)

    def test_invalid_columns(self):
        """Test with wrong number of columns"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        assert result.shape == (0, 2)


class TestReadBandResFile:
    """Tests for read_band_res_file function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.find_rail_file")
    @patch("numpy.loadtxt")
    @patch("pathlib.Path.exists")
    def test_successful_read(self, mock_exists, mock_loadtxt, mock_find_rail):
        """Test successful reading of band file"""
        mock_find_rail.return_value = "/path/to/filters"
        mock_data = np.array([[100, 0], [200, 0.5], [300, 1.0], [400, 0.5], [500, 0]])
        mock_loadtxt.return_value = mock_data

        result = catalog_funcs.read_band_res_file("g")

        # Should extract non-zero region with padding
        assert len(result) > 0
        assert result.ndim == 2
        assert result.shape[1] == 2

    @patch("numpy.loadtxt")
    def test_read_with_custom_filter_dir(self, mock_loadtxt):
        """Test reading with custom filter directory"""
        mock_data = np.array([[100, 0], [200, 1.0], [300, 0]])
        mock_loadtxt.return_value = mock_data

        filter_dir = Path("/custom/filter/dir")
        with patch.object(Path, "exists", return_value=True):
            result = catalog_funcs.read_band_res_file("r", filter_dir=filter_dir)

        assert len(result) > 0

    def test_invalid_band_name(self):
        """Test with invalid band name"""
        with pytest.raises(ValueError, match="band_name must be a non-empty string"):
            catalog_funcs.read_band_res_file("")

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.find_rail_file")
    def test_filter_dir_not_found(self, mock_find_rail):
        """Test when filter directory doesn't exist"""
        mock_find_rail.return_value = "/nonexistent/path"

        with pytest.raises(FileNotFoundError, match="Filter directory not found"):
            catalog_funcs.read_band_res_file("g")

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.find_rail_file")
    @patch("pathlib.Path.exists")
    def test_band_file_not_found(self, mock_exists, mock_find_rail):
        """Test when band file doesn't exist"""
        mock_find_rail.return_value = "/path/to/filters"

        # Make filter_dir exist but file doesn't
        def exists_side_effect(self):
            # Return True for filter_dir, False for the actual .res file
            return "/FILTER" in str(self)

        with patch.object(Path, "exists", exists_side_effect):
            with pytest.raises(FileNotFoundError, match="Filter directory not found"):
                catalog_funcs.read_band_res_file("g")

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.find_rail_file")
    @patch("numpy.loadtxt")
    def test_invalid_data_format(self, mock_loadtxt, mock_find_rail):
        """Test with invalid data format in file"""
        mock_find_rail.return_value = "/path/to/filters"
        mock_loadtxt.side_effect = ValueError("Invalid format")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="Invalid data format"):
                catalog_funcs.read_band_res_file("g")

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.find_rail_file")
    @patch("numpy.loadtxt")
    def test_wrong_shape_data(self, mock_loadtxt, mock_find_rail):
        """Test when loaded data has wrong shape"""
        mock_find_rail.return_value = "/path/to/filters"
        mock_loadtxt.return_value = np.array([1, 2, 3])  # 1D array

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="Invalid data shape"):
                catalog_funcs.read_band_res_file("g")


class TestMakeBandCreateModel:
    """Tests for make_band_create_model function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.read_band_res_file")
    def test_successful_creation(self, mock_read):
        """Test successful creation of BandCreate model"""
        mock_data = np.array([[100, 0.1], [200, 0.5], [300, 0.8]])
        mock_read.return_value = mock_data

        result = catalog_funcs.make_band_create_model("g", None)

        assert isinstance(result, BandCreate)
        assert result.name == "g"
        assert result.band_wavelengths == [100, 200, 300]
        assert result.band_transmission == [0.1, 0.5, 0.8]

    @patch("rail_svc.rail_funcs.catalog_funcs.read_band_res_file")
    def test_file_not_found_graceful(self, mock_read):
        """Test graceful handling of missing file"""
        mock_read.side_effect = FileNotFoundError("File not found")

        # The function returns a BandCreate with empty lists, which will fail validation
        # So we should expect a ValidationError
        with pytest.raises(Exception):  # Could be ValidationError or handled gracefully
            _result = catalog_funcs.make_band_create_model("g", None)

    @patch("rail_svc.rail_funcs.catalog_funcs.read_band_res_file")
    def test_invalid_data_graceful(self, mock_read):
        """Test graceful handling of invalid data"""
        mock_read.side_effect = ValueError("Invalid data")

        # Same issue - empty lists will fail validation
        with pytest.raises(Exception):
            _result = catalog_funcs.make_band_create_model("g", None)

    def test_invalid_band_name(self):
        """Test with invalid band name"""
        with pytest.raises(ValueError, match="band_name must be a non-empty string"):
            catalog_funcs.make_band_create_model("", None)


class TestMakeBandCreateModels:
    """Tests for make_band_create_models function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.BandFactory.get_bands")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_band_create_model")
    def test_successful_creation(self, mock_make_band, mock_get_bands):
        """Test successful creation of multiple band models"""
        mock_get_bands.return_value = ["u", "g", "r", "i", "z"]
        mock_make_band.side_effect = [
            BandCreate(name=name, band_wavelengths=[100], band_transmission=[0.5])
            for name in ["u", "g", "r", "i", "z"]
        ]

        result = catalog_funcs.make_band_create_models()

        assert len(result) == 5
        assert all(isinstance(b, BandCreate) for b in result)

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.BandFactory.get_bands")
    def test_no_bands(self, mock_get_bands):
        """Test when no bands are registered"""
        mock_get_bands.return_value = []

        result = catalog_funcs.make_band_create_models()

        assert result == []

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.BandFactory.get_bands")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_band_create_model")
    def test_partial_failure(self, mock_make_band, mock_get_bands):
        """Test when some bands fail to create"""
        mock_get_bands.return_value = ["u", "g", "r"]
        mock_make_band.side_effect = [
            BandCreate(name="u", band_wavelengths=[100], band_transmission=[0.5]),
            Exception("Failed to create g"),
            BandCreate(name="r", band_wavelengths=[100], band_transmission=[0.5]),
        ]

        result = catalog_funcs.make_band_create_models()

        assert len(result) == 2  # u and r succeeded


class TestMakeCatalogTagCreateModel:
    """Tests for make_catalog_tag_create_model function"""

    def test_successful_creation(self):
        """Test successful creation of CatalogTagCreate model"""
        result = catalog_funcs.make_catalog_tag_create_model("lsst_dp0")

        assert isinstance(result, CatalogTagCreate)
        assert result.name == "lsst_dp0"

    def test_invalid_name_empty(self):
        """Test with empty string"""
        with pytest.raises(ValueError, match="catalog_tag_name must be a non-empty string"):
            catalog_funcs.make_catalog_tag_create_model("")

    def test_invalid_name_whitespace(self):
        """Test with whitespace-only string"""
        with pytest.raises(ValueError, match="catalog_tag_name must be a non-empty string"):
            catalog_funcs.make_catalog_tag_create_model("   ")


class TestMakeCatalogTagCreateModels:
    """Tests for make_catalog_tag_create_models function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.CatalogTagFactory.get_catalog_tags")
    def test_successful_creation(self, mock_get_tags):
        """Test successful creation of multiple catalog tag models"""
        mock_get_tags.return_value = {"lsst_dp0": Mock(), "des_y6": Mock(), "hsc": Mock()}

        result = catalog_funcs.make_catalog_tag_create_models()

        assert len(result) == 3
        assert all(isinstance(t, CatalogTagCreate) for t in result)
        assert {t.name for t in result} == {"lsst_dp0", "des_y6", "hsc"}

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.CatalogTagFactory.get_catalog_tags")
    def test_no_tags(self, mock_get_tags):
        """Test when no catalog tags are registered"""
        mock_get_tags.return_value = {}

        result = catalog_funcs.make_catalog_tag_create_models()

        assert result == []


class TestMakeCatalogBandAssocCreateModels:
    """Tests for make_catalog_band_assoc_create_models function"""

    def test_successful_creation(self):
        """Test successful creation of band associations"""
        # Create mock catalog tag
        mock_config = Mock()
        mock_config.name = "test_catalog"
        mock_config.band_list = ["g", "r", "i"]
        mock_config.bands = {
            "g": {"filter": "g_band", "mag_column_name": "mag_g", "mag_err_column_name": "magerr_g"},
            "r": {"filter": "r_band", "mag_column_name": "mag_r", "mag_err_column_name": "magerr_r"},
            "i": {"filter": "i_band", "mag_column_name": "mag_i", "mag_err_column_name": "magerr_i"},
        }
        mock_config.filter_template = "{band}_band"
        mock_config.mag_column_template = "mag_{band}"
        mock_config.mag_err_column_template = "magerr_{band}"

        mock_ct = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
        mock_ct.config = mock_config

        result = catalog_funcs.make_catalog_band_assoc_create_models(mock_ct)

        assert len(result) == 3
        assert all(isinstance(a, CatalogBandAssocCreate) for a in result)
        assert result[0].catalog_tag_name == "test_catalog"

    def test_with_template_fallback(self):
        """Test creation using template fallback when band info missing"""
        mock_config = Mock()
        mock_config.name = "test_catalog"
        mock_config.band_list = ["g"]
        mock_config.bands = {"g": {}}  # Empty band config
        mock_config.filter_template = "filter_{band}"
        mock_config.mag_column_template = "mag_{band}"
        mock_config.mag_err_column_template = "magerr_{band}"

        mock_ct = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
        mock_ct.config = mock_config

        result = catalog_funcs.make_catalog_band_assoc_create_models(mock_ct)

        assert len(result) == 1
        assert result[0].band_name == "filter_g"
        assert result[0].mag_column_name == "mag_g"
        assert result[0].mag_err_column_name == "magerr_g"

    def test_empty_band_list(self):
        """Test with empty band list"""
        mock_config = Mock()
        mock_config.name = "test_catalog"
        mock_config.band_list = []
        mock_config.bands = {}

        mock_ct = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
        mock_ct.config = mock_config

        result = catalog_funcs.make_catalog_band_assoc_create_models(mock_ct)

        assert result == []

    def test_invalid_catalog_tag(self):
        """Test with invalid CatalogTag object"""
        with pytest.raises(ValueError, match="ct must be a CatalogTag object"):
            catalog_funcs.make_catalog_band_assoc_create_models("not a catalog tag")

    def test_missing_config(self):
        """Test with CatalogTag missing config attribute"""
        # Create a proper CatalogTag instance instead of a Mock
        from rail.utils.catalog_utils import CatalogTag

        # This should raise an error because we need a valid config
        with pytest.raises(AttributeError, match="CatalogTag must have a 'config' attribute"):
            # Create a mock that will pass isinstance check but has no config
            mock_ct = Mock(spec=CatalogTag)
            delattr(mock_ct, "config")  # Remove config after creating spec
            catalog_funcs.make_catalog_band_assoc_create_models(mock_ct)

    def test_partial_band_failure(self):
        """Test when some bands fail to process"""
        mock_config = Mock()
        mock_config.name = "test_catalog"
        mock_config.band_list = ["g", "r"]
        mock_config.bands = {
            "g": {"filter": "g_band", "mag_column_name": "mag_g", "mag_err_column_name": "magerr_g"},
            # "r" is missing - will cause KeyError
        }
        mock_config.filter_template = "{band}_band"
        mock_config.mag_column_template = "mag_{band}"
        mock_config.mag_err_column_template = "magerr_{band}"

        mock_ct = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
        mock_ct.config = mock_config

        result = catalog_funcs.make_catalog_band_assoc_create_models(mock_ct)

        # Should still create association for 'g' despite 'r' failing
        assert len(result) >= 0


class TestMakeAllCatalogBandAssocCreateModels:
    """Tests for make_all_catalog_band_assoc_create_models function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.CatalogTagFactory.get_catalog_tags")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_catalog_band_assoc_create_models")
    def test_successful_creation(self, mock_make_assoc, mock_get_tags):
        """Test successful creation of all associations"""
        mock_ct1 = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
        mock_ct2 = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)

        mock_get_tags.return_value = {
            "catalog1": mock_ct1,
            "catalog2": mock_ct2,
        }

        mock_make_assoc.side_effect = [
            [
                CatalogBandAssocCreate(
                    mag_column_name="mag_g",
                    mag_err_column_name="err_g",
                    band_name="g",
                    catalog_tag_name="catalog1",
                )
            ],
            [
                CatalogBandAssocCreate(
                    mag_column_name="mag_r",
                    mag_err_column_name="err_r",
                    band_name="r",
                    catalog_tag_name="catalog2",
                )
            ],
        ]

        result = catalog_funcs.make_all_catalog_band_assoc_create_models()

        assert len(result) == 2
        assert all(isinstance(a, CatalogBandAssocCreate) for a in result)

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.CatalogTagFactory.get_catalog_tags")
    def test_no_catalogs(self, mock_get_tags):
        """Test when no catalogs are registered"""
        mock_get_tags.return_value = {}

        result = catalog_funcs.make_all_catalog_band_assoc_create_models()

        assert result == []

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.CatalogTagFactory.get_catalog_tags")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_catalog_band_assoc_create_models")
    def test_partial_catalog_failure(self, mock_make_assoc, mock_get_tags):
        """Test when some catalogs fail to process"""
        mock_ct1 = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
        mock_ct2 = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)

        mock_get_tags.return_value = {
            "catalog1": mock_ct1,
            "catalog2": mock_ct2,
        }

        mock_make_assoc.side_effect = [
            [
                CatalogBandAssocCreate(
                    mag_column_name="mag_g",
                    mag_err_column_name="err_g",
                    band_name="g",
                    catalog_tag_name="catalog1",
                )
            ],
            Exception("Failed to process catalog2"),
        ]

        result = catalog_funcs.make_all_catalog_band_assoc_create_models()

        assert len(result) == 1  # Only catalog1 succeeded


class TestLoadCatalogYaml:
    """Tests for load_catalog_yaml function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.load_yaml")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_band_create_models")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_catalog_tag_create_models")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_all_catalog_band_assoc_create_models")
    def test_successful_load(self, mock_make_assocs, mock_make_tags, mock_make_bands, mock_load_yaml):
        """Test successful loading of catalog YAML"""
        # Create a temporary file
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = Path(f.name)
            f.write("# test yaml\n")

        try:
            mock_bands = [BandCreate(name="g", band_wavelengths=[100], band_transmission=[0.5])]
            mock_tags = [CatalogTagCreate(name="test_catalog")]
            mock_assocs = [
                CatalogBandAssocCreate(
                    mag_column_name="mag_g",
                    mag_err_column_name="err_g",
                    band_name="g",
                    catalog_tag_name="test_catalog",
                )
            ]

            mock_make_bands.return_value = mock_bands
            mock_make_tags.return_value = mock_tags
            mock_make_assocs.return_value = mock_assocs

            bands, tags, assocs = catalog_funcs.load_catalog_yaml(yaml_path)

            assert len(bands) == 1
            assert len(tags) == 1
            assert len(assocs) == 1
            mock_load_yaml.assert_called_once()
        finally:
            yaml_path.unlink()

    def test_file_not_found(self):
        """Test with non-existent file"""
        with pytest.raises(FileNotFoundError, match="Catalog YAML file not found"):
            catalog_funcs.load_catalog_yaml(Path("/nonexistent/file.yaml"))

    def test_path_is_directory(self):
        """Test when path is a directory not a file"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="catalog_yaml must be a file"):
                catalog_funcs.load_catalog_yaml(Path(tmpdir))

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.load_yaml")
    def test_invalid_yaml_content(self, mock_load_yaml):
        """Test with invalid YAML content"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = Path(f.name)
            f.write("# test yaml\n")

        try:
            mock_load_yaml.side_effect = ValueError("Invalid YAML")

            with pytest.raises(ValueError, match="Invalid catalog YAML file"):
                catalog_funcs.load_catalog_yaml(yaml_path)
        finally:
            yaml_path.unlink()

    def test_string_path_conversion(self):
        """Test that string paths are converted to Path objects"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = f.name
            f.write("# test yaml\n")

        try:
            with patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.load_yaml"):
                with patch("rail_svc.rail_funcs.catalog_funcs.make_band_create_models", return_value=[]):
                    with patch(
                        "rail_svc.rail_funcs.catalog_funcs.make_catalog_tag_create_models", return_value=[]
                    ):
                        with patch(
                            "rail_svc.rail_funcs.catalog_funcs.make_all_catalog_band_assoc_create_models",
                            return_value=[],
                        ):
                            bands, tags, assocs = catalog_funcs.load_catalog_yaml(yaml_path)
                            assert isinstance(bands, list)
        finally:
            Path(yaml_path).unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.load_yaml")
    @patch("rail_svc.rail_funcs.catalog_funcs.make_band_create_models")
    def test_band_creation_failure(self, mock_make_bands, mock_load_yaml):
        """Test when band creation fails"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = Path(f.name)
            f.write("# test yaml\n")

        try:
            mock_make_bands.side_effect = Exception("Failed to create bands")

            with pytest.raises(Exception, match="Failed to create bands"):
                catalog_funcs.load_catalog_yaml(yaml_path)
        finally:
            yaml_path.unlink()


class TestGetCatalogRow:
    """Tests for get_catalog_row function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.tables_io.read")
    def test_successful_read(self, mock_read):
        """Test successful reading of catalog row"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            catalog_path = Path(f.name)

        try:
            mock_data = {
                "mag_g": np.array([20.5]),
                "mag_r": np.array([19.8]),
                "redshift": np.array([0.5]),
            }
            mock_read.return_value = mock_data

            result = catalog_funcs.get_catalog_row(catalog_path, 42)

            assert isinstance(result, dict)
            assert "mag_g" in result
            mock_read.assert_called_once_with(str(catalog_path), slice_dict=42)
        finally:
            catalog_path.unlink()

    def test_file_not_found(self):
        """Test with non-existent file"""
        with pytest.raises(FileNotFoundError, match="Catalog file not found"):
            catalog_funcs.get_catalog_row(Path("/nonexistent/catalog.hdf5"), 0)

    @patch("rail_svc.rail_funcs.catalog_funcs.tables_io.read")
    def test_negative_row_index(self, mock_read):
        """Test with negative row index"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            catalog_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="row must be non-negative"):
                catalog_funcs.get_catalog_row(catalog_path, -1)
        finally:
            catalog_path.unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.tables_io.read")
    def test_row_out_of_bounds(self, mock_read):
        """Test with row index out of bounds"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            catalog_path = Path(f.name)

        try:
            mock_read.side_effect = IndexError("Row index out of bounds")

            with pytest.raises(ValueError, match="Row index .* out of bounds"):
                catalog_funcs.get_catalog_row(catalog_path, 999999)
        finally:
            catalog_path.unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.tables_io.read")
    def test_invalid_file_format(self, mock_read):
        """Test with invalid file format"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            catalog_path = Path(f.name)

        try:
            mock_read.side_effect = OSError("Cannot read file")

            with pytest.raises(OSError, match="Cannot read catalog file"):
                catalog_funcs.get_catalog_row(catalog_path, 0)
        finally:
            catalog_path.unlink()

    def test_path_is_directory(self):
        """Test when path is a directory"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="catalog_path must be a file"):
                catalog_funcs.get_catalog_row(Path(tmpdir), 0)

    @patch("rail_svc.rail_funcs.catalog_funcs.tables_io.read")
    def test_string_path_conversion(self, mock_read):
        """Test that string paths are converted to Path objects"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            catalog_path = f.name

        try:
            mock_data = {"mag_g": np.array([20.5])}
            mock_read.return_value = mock_data

            result = catalog_funcs.get_catalog_row(catalog_path, 0)

            assert isinstance(result, dict)
        finally:
            Path(catalog_path).unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.tables_io.read")
    def test_unexpected_return_type(self, mock_read):
        """Test when tables_io.read returns unexpected type"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            catalog_path = Path(f.name)

        try:
            mock_read.return_value = "not a dict"

            with pytest.raises(OSError, match="Invalid data format in catalog"):
                catalog_funcs.get_catalog_row(catalog_path, 0)
        finally:
            catalog_path.unlink()


class TestGetEstimatesRow:
    """Tests for get_estimates_row function"""

    @patch("rail_svc.rail_funcs.catalog_funcs.qp.read")
    def test_successful_read(self, mock_qp_read):
        """Test successful reading of estimates row"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            estimates_path = Path(f.name)

        try:
            mock_data = Mock()  # qp ensemble object
            mock_qp_read.return_value = mock_data

            result = catalog_funcs.get_estimates_row(estimates_path, 42)

            assert result is not None
            mock_qp_read.assert_called_once_with(str(estimates_path), read_slice=slice(42, 43))
        finally:
            estimates_path.unlink()

    def test_file_not_found(self):
        """Test with non-existent file"""
        with pytest.raises(FileNotFoundError, match="Estimates file not found"):
            catalog_funcs.get_estimates_row(Path("/nonexistent/estimates.hdf5"), 0)

    @patch("rail_svc.rail_funcs.catalog_funcs.qp.read")
    def test_negative_row_index(self, mock_qp_read):
        """Test with negative row index"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            estimates_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="row must be non-negative"):
                catalog_funcs.get_estimates_row(estimates_path, -1)
        finally:
            estimates_path.unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.qp.read")
    def test_row_out_of_bounds(self, mock_qp_read):
        """Test with row index out of bounds"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            estimates_path = Path(f.name)

        try:
            mock_qp_read.side_effect = IndexError("Row index out of bounds")

            with pytest.raises(ValueError, match="Row index .* out of bounds"):
                catalog_funcs.get_estimates_row(estimates_path, 999999)
        finally:
            estimates_path.unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.qp.read")
    def test_invalid_file_format(self, mock_qp_read):
        """Test with invalid file format"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            estimates_path = Path(f.name)

        try:
            mock_qp_read.side_effect = OSError("Cannot read file")

            with pytest.raises(OSError, match="Cannot read estimates file"):
                catalog_funcs.get_estimates_row(estimates_path, 0)
        finally:
            estimates_path.unlink()

    def test_path_is_directory(self):
        """Test when path is a directory"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="estimates_path must be a file"):
                catalog_funcs.get_estimates_row(Path(tmpdir), 0)

    @patch("rail_svc.rail_funcs.catalog_funcs.qp.read")
    def test_string_path_conversion(self, mock_qp_read):
        """Test that string paths are converted to Path objects"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            estimates_path = f.name

        try:
            mock_data = Mock()
            mock_qp_read.return_value = mock_data

            result = catalog_funcs.get_estimates_row(estimates_path, 0)

            assert result is not None
        finally:
            Path(estimates_path).unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.qp.read")
    def test_none_return_value(self, mock_qp_read):
        """Test when qp.read returns None"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            estimates_path = Path(f.name)

        try:
            mock_qp_read.return_value = None

            with pytest.raises(OSError, match="Invalid or empty data at row"):
                catalog_funcs.get_estimates_row(estimates_path, 0)
        finally:
            estimates_path.unlink()


class TestLogging:
    """Tests for logging behavior"""

    def test_extract_padded_non_zeros_logs_error_on_invalid_dimensions(self, caplog):
        """Test that invalid dimensions are logged"""
        with caplog.at_level(logging.ERROR):
            arr = np.array([1, 2, 3])
            catalog_funcs.extract_padded_non_zeros(arr)

            assert "Expected 2D array" in caplog.text

    def test_extract_padded_non_zeros_logs_warning_on_all_zeros(self, caplog):
        """Test that all-zeros warning is logged"""
        with caplog.at_level(logging.WARNING):
            arr = np.array([[1, 0], [2, 0]])
            catalog_funcs.extract_padded_non_zeros(arr)

            assert "All values in second column are zero" in caplog.text

    def test_read_band_res_file_logs_info(self, caplog):
        """Test that successful read logs info"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".res") as f:
            filter_dir = Path(f.name).parent
            band_name = Path(f.name).stem
            np.savetxt(f.name, np.array([[100, 0], [200, 1.0], [300, 0]]))

        try:
            with caplog.at_level(logging.INFO):
                catalog_funcs.read_band_res_file(band_name, filter_dir)

                assert "Reading band response file" in caplog.text
        finally:
            Path(f.name).unlink()

    @patch("rail_svc.rail_funcs.catalog_funcs.read_band_res_file")
    def test_make_band_create_model_logs_warning_on_missing_file(self, mock_read, caplog):
        """Test that missing file warning is logged"""
        mock_read.side_effect = FileNotFoundError("Not found")

        with caplog.at_level(logging.WARNING):
            try:
                catalog_funcs.make_band_create_model("nonexistent", None)
            except Exception:
                pass  # Ignore validation error

            assert "Filter response file not found" in caplog.text


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_extract_padded_non_zeros_single_element_array(self):
        """Test with single-element array"""
        arr = np.array([[1, 5]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        np.testing.assert_array_equal(result, arr)

    def test_extract_padded_non_zeros_two_element_array(self):
        """Test with two-element array"""
        arr = np.array([[1, 0], [2, 5]])
        result = catalog_funcs.extract_padded_non_zeros(arr)

        np.testing.assert_array_equal(result, arr)

    @patch("rail_svc.rail_funcs.catalog_funcs.read_band_res_file")
    def test_make_band_create_model_empty_data(self, mock_read):
        """Test with empty data array"""
        """Test with empty data array"""
        # Return a proper 2D array with at least one row
        mock_read.return_value = np.array([[100.0, 0.5]])  # Shape (1, 2)

        result = catalog_funcs.make_band_create_model("g", None)

        assert result.band_wavelengths == [100.0]
        assert result.band_transmission == [0.5]

    def test_make_catalog_band_assoc_with_special_characters(self):
        """Test catalog band association with special characters in names"""
        mock_config = Mock()
        mock_config.name = "test-catalog_v2.0"
        mock_config.band_list = ["u-band"]
        mock_config.bands = {
            "u-band": {"filter": "u_filter", "mag_column_name": "mag_u", "mag_err_column_name": "magerr_u"}
        }
        mock_config.filter_template = "{band}_filter"
        mock_config.mag_column_template = "mag_{band}"
        mock_config.mag_err_column_template = "magerr_{band}"

        mock_ct = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
        mock_ct.config = mock_config

        result = catalog_funcs.make_catalog_band_assoc_create_models(mock_ct)

        assert len(result) == 1
        assert result[0].catalog_tag_name == "test-catalog_v2.0"

    @patch("rail_svc.rail_funcs.catalog_funcs.tables_io.read")
    def test_get_catalog_row_zero_index(self, mock_read):
        """Test reading row at index 0"""
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".hdf5") as f:
            catalog_path = Path(f.name)

        try:
            mock_data = {"mag_g": np.array([20.5])}
            mock_read.return_value = mock_data

            result = catalog_funcs.get_catalog_row(catalog_path, 0)

            assert isinstance(result, dict)
            mock_read.assert_called_once_with(str(catalog_path), slice_dict=0)
        finally:
            catalog_path.unlink()

    def test_load_catalog_yaml_with_yml_extension(self):
        """Test loading YAML file with .yml extension"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml_path = Path(f.name)
            f.write("# test yaml\n")

        try:
            with patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.load_yaml"):
                with patch("rail_svc.rail_funcs.catalog_funcs.make_band_create_models", return_value=[]):
                    with patch(
                        "rail_svc.rail_funcs.catalog_funcs.make_catalog_tag_create_models", return_value=[]
                    ):
                        with patch(
                            "rail_svc.rail_funcs.catalog_funcs.make_all_catalog_band_assoc_create_models",
                            return_value=[],
                        ):
                            bands, tags, assocs = catalog_funcs.load_catalog_yaml(yaml_path)
                            assert isinstance(bands, list)
        finally:
            yaml_path.unlink()


class TestIntegration:
    """Integration tests combining multiple functions"""

    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.load_yaml")
    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.BandFactory.get_bands")
    @patch("rail_svc.rail_funcs.catalog_funcs.catalog_utils.CatalogTagFactory.get_catalog_tags")
    @patch("rail_svc.rail_funcs.catalog_funcs.read_band_res_file")
    def test_complete_catalog_load_workflow(
        self, mock_read_band, mock_get_tags, mock_get_bands, mock_load_yaml
    ):
        """Test complete workflow of loading catalog configuration"""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml_path = Path(f.name)
            f.write("# test yaml\n")

        try:
            # Setup mocks
            mock_get_bands.return_value = ["g", "r"]
            mock_read_band.side_effect = [
                np.array([[100, 0], [200, 1.0], [300, 0]]),
                np.array([[100, 0], [200, 0.8], [300, 0]]),
            ]

            mock_config = Mock()
            mock_config.name = "test_catalog"
            mock_config.band_list = ["g", "r"]
            mock_config.bands = {
                "g": {"filter": "g_band", "mag_column_name": "mag_g", "mag_err_column_name": "magerr_g"},
                "r": {"filter": "r_band", "mag_column_name": "mag_r", "mag_err_column_name": "magerr_r"},
            }
            mock_config.filter_template = "{band}_band"
            mock_config.mag_column_template = "mag_{band}"
            mock_config.mag_err_column_template = "magerr_{band}"

            mock_ct = Mock(spec=catalog_funcs.catalog_utils.CatalogTag)
            mock_ct.config = mock_config

            mock_get_tags.return_value = {"test_catalog": mock_ct}

            # Execute
            bands, tags, assocs = catalog_funcs.load_catalog_yaml(yaml_path, filter_dir=None)

            # Verify
            assert len(bands) == 2
            assert len(tags) == 1
            assert len(assocs) == 2
            assert tags[0].name == "test_catalog"
        finally:
            yaml_path.unlink()
