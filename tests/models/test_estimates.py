"""Unit tests for the Estimates Pydantic models"""

import pytest
from pydantic import ValidationError

from rail_svc.models.estimates import Estimates, EstimatesBase, EstimatesCreate


class TestEstimatesBase:
    """Tests for EstimatesBase model"""

    def test_valid_estimates_base_with_path(self):
        """Test creating a valid EstimatesBase with qp_file_path"""
        estimates = EstimatesBase(
            qp_file_path="/data/outputs/pz_estimates_001.hdf5",
        )
        assert estimates.qp_file_path == "/data/outputs/pz_estimates_001.hdf5"

    def test_valid_estimates_base_without_path(self):
        """Test creating EstimatesBase without qp_file_path (defaults to None)"""
        estimates = EstimatesBase()
        assert estimates.qp_file_path is None

    def test_estimates_base_with_none_path(self):
        """Test creating EstimatesBase with explicit None path"""
        estimates = EstimatesBase(qp_file_path=None)
        assert estimates.qp_file_path is None

    def test_estimates_base_various_path_formats(self):
        """Test various valid file path formats"""
        paths = [
            "/absolute/path/to/file.hdf5",
            "relative/path/file.hdf5",
            "./local/file.qp",
            "../parent/file.qp",
            "s3://bucket/key/file.hdf5",
            "/data/pz_12345.hdf5",
        ]

        for path in paths:
            estimates = EstimatesBase(qp_file_path=path)
            assert estimates.qp_file_path == path


class TestEstimatesCreate:
    """Tests for EstimatesCreate model"""

    def test_valid_estimates_create(self):
        """Test creating a valid EstimatesCreate"""
        estimates = EstimatesCreate(
            qp_file_path="/outputs/result.hdf5",
            estimator_name="knn_estimator_v1",
            dataset_name="lsst_dp02_sample",
        )
        assert estimates.qp_file_path == "/outputs/result.hdf5"
        assert estimates.estimator_name == "knn_estimator_v1"
        assert estimates.dataset_name == "lsst_dp02_sample"

    def test_estimates_create_without_path(self):
        """Test creating EstimatesCreate without qp_file_path"""
        estimates = EstimatesCreate(
            estimator_name="som_estimator",
            dataset_name="test_dataset",
        )
        assert estimates.qp_file_path is None
        assert estimates.estimator_name == "som_estimator"
        assert estimates.dataset_name == "test_dataset"

    def test_estimates_create_missing_estimator_name(self):
        """Test that estimator_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            EstimatesCreate(
                dataset_name="test_dataset",
            )
        assert "estimator_name" in str(exc_info.value)

    def test_estimates_create_missing_dataset_name(self):
        """Test that dataset_name is required"""
        with pytest.raises(ValidationError) as exc_info:
            EstimatesCreate(
                estimator_name="test_estimator",
            )
        assert "dataset_name" in str(exc_info.value)


class TestEstimates:
    """Tests for Estimates model"""

    def test_valid_estimates(self):
        """Test creating a valid Estimates with all fields"""
        estimates = Estimates(
            id=1,
            qp_file_path="/data/estimates_001.hdf5",
            estimator_id=5,
            dataset_id=10,
        )
        assert estimates.id == 1
        assert estimates.qp_file_path == "/data/estimates_001.hdf5"
        assert estimates.estimator_id == 5
        assert estimates.dataset_id == 10

    def test_estimates_without_path(self):
        """Test creating Estimates without qp_file_path (in progress job)"""
        estimates = Estimates(
            id=1,
            estimator_id=5,
            dataset_id=10,
        )
        assert estimates.id == 1
        assert estimates.qp_file_path is None
        assert estimates.estimator_id == 5
        assert estimates.dataset_id == 10

    def test_estimates_id_must_be_positive(self):
        """Test that id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=0,
                estimator_id=1,
                dataset_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=-1,
                estimator_id=1,
                dataset_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_estimates_estimator_id_must_be_positive(self):
        """Test that estimator_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=1,
                estimator_id=0,
                dataset_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=1,
                estimator_id=-5,
                dataset_id=1,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_estimates_dataset_id_must_be_positive(self):
        """Test that dataset_id must be greater than 0"""
        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=1,
                estimator_id=1,
                dataset_id=0,
            )
        assert "greater than 0" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=1,
                estimator_id=1,
                dataset_id=-3,
            )
        assert "greater than 0" in str(exc_info.value)

    def test_estimates_missing_id(self):
        """Test that id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                estimator_id=1,
                dataset_id=1,
            )
        assert "id" in str(exc_info.value)

    def test_estimates_missing_estimator_id(self):
        """Test that estimator_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=1,
                dataset_id=1,
            )
        assert "estimator_id" in str(exc_info.value)

    def test_estimates_missing_dataset_id(self):
        """Test that dataset_id is required"""
        with pytest.raises(ValidationError) as exc_info:
            Estimates(
                id=1,
                estimator_id=1,
            )
        assert "dataset_id" in str(exc_info.value)

    def test_estimates_from_attributes(self):
        """Test that from_attributes config works"""

        # Simulate an ORM object with attributes
        class MockORMObject:
            id = 42
            qp_file_path = "/storage/pz_42.hdf5"
            estimator_id = 7
            dataset_id = 15

        orm_obj = MockORMObject()
        estimates = Estimates.model_validate(orm_obj)
        assert estimates.id == 42
        assert estimates.qp_file_path == "/storage/pz_42.hdf5"
        assert estimates.estimator_id == 7
        assert estimates.dataset_id == 15

    def test_estimates_from_attributes_no_path(self):
        """Test ORM object without qp_file_path"""

        class MockORMObject:
            id = 99
            qp_file_path = None
            estimator_id = 3
            dataset_id = 8

        orm_obj = MockORMObject()
        estimates = Estimates.model_validate(orm_obj)
        assert estimates.id == 99
        assert estimates.qp_file_path is None
        assert estimates.estimator_id == 3
        assert estimates.dataset_id == 8

    def test_estimates_col_names_for_table(self):
        """Test that col_names_for_table class variable is set correctly"""
        expected_cols = ["id", "estimator_id", "dataset_id", "qp_file_path"]
        assert Estimates.col_names_for_table == expected_cols

    def test_estimates_field_descriptions(self):
        """Test that field descriptions are set"""
        schema = Estimates.model_json_schema()
        assert "Path to the output qp ensemble file" in schema["properties"]["qp_file_path"]["description"]
        assert "Foreign key referencing Estimator.id" in schema["properties"]["estimator_id"]["description"]
        assert "Foreign key referencing Dataset.id" in schema["properties"]["dataset_id"]["description"]

    def test_estimates_realistic_workflow(self):
        """Test realistic workflow: create request -> pending -> completed"""
        # Step 1: Create request (no path yet)
        create_req = EstimatesCreate(
            estimator_name="knn_v1",
            dataset_name="dp02_sample",
        )
        assert create_req.qp_file_path is None

        # Step 2: Job pending (DB record created, no path)
        pending = Estimates(
            id=123,
            estimator_id=5,
            dataset_id=10,
            qp_file_path=None,
        )
        assert pending.qp_file_path is None

        # Step 3: Job completed (path added)
        completed = Estimates(
            id=123,
            estimator_id=5,
            dataset_id=10,
            qp_file_path="/outputs/pz_123.hdf5",
        )
        assert completed.qp_file_path == "/outputs/pz_123.hdf5"

    def test_estimates_various_file_extensions(self):
        """Test various qp file extensions"""
        extensions = [".hdf5", ".h5", ".qp", ".fits"]

        for idx, ext in enumerate(extensions, start=1):
            estimates = Estimates(
                id=idx,
                estimator_id=1,
                dataset_id=1,
                qp_file_path=f"/data/output{ext}",
            )
            assert estimates.qp_file_path.endswith(ext)

    def test_estimates_json_serialization(self):
        """Test that Estimates can be serialized to/from JSON"""
        original = Estimates(
            id=50,
            estimator_id=10,
            dataset_id=20,
            qp_file_path="/results/pz_50.hdf5",
        )

        # Serialize to JSON
        json_str = original.model_dump_json()

        # Deserialize from JSON
        restored = Estimates.model_validate_json(json_str)

        assert restored.id == original.id
        assert restored.estimator_id == original.estimator_id
        assert restored.dataset_id == original.dataset_id
        assert restored.qp_file_path == original.qp_file_path

    def test_estimates_multiple_jobs_same_estimator(self):
        """Test multiple estimate jobs using same estimator but different datasets"""
        estimator_id = 5

        jobs = [
            Estimates(
                id=1,
                estimator_id=estimator_id,
                dataset_id=10,
                qp_file_path="/out/job1.hdf5",
            ),
            Estimates(
                id=2,
                estimator_id=estimator_id,
                dataset_id=11,
                qp_file_path="/out/job2.hdf5",
            ),
            Estimates(
                id=3,
                estimator_id=estimator_id,
                dataset_id=12,
                qp_file_path="/out/job3.hdf5",
            ),
        ]

        for job in jobs:
            assert job.estimator_id == estimator_id
            assert job.qp_file_path is not None
