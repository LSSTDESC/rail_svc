"""Unit tests for RemoteDatabase class."""

from __future__ import annotations

from pydantic import BaseModel

from rail_svc.client.base import RemoteAPI, RemoteTableOperations
from rail_svc.client.client import TABLE_CONFIGS, RemoteDatabase


class TestRemoteDatabase:
    """Tests for RemoteDatabase class."""

    def test_initialization(self) -> None:
        """Test RemoteDatabase initialization."""
        db = RemoteDatabase(
            base_url="http://api.example.com",
            api_prefix="/api/v2",
            timeout=60.0,
            auth_token="test-token",
        )

        assert db.base_url == "http://api.example.com"
        assert db.api_prefix == "/api/v2"
        assert db.timeout == 60.0
        assert db.auth_token == "test-token"
        assert db._api is None

    def test_initialization_defaults(self) -> None:
        """Test RemoteDatabase initialization with defaults."""
        db = RemoteDatabase(base_url="http://api.example.com")

        assert db.base_url == "http://api.example.com"
        assert db.api_prefix == "/api/v1"
        assert db.timeout == 30.0
        assert db.auth_token is None

    async def test_context_manager_lifecycle(self) -> None:
        """Test that context manager properly initializes and cleans up."""
        db = RemoteDatabase("http://api.example.com")

        assert db._api is None

        async with db:
            assert db._api is not None
            assert isinstance(db._api, RemoteAPI)

        # API should still exist but be closed
        assert db._api is not None

    async def test_setup_clients_creates_all_tables(self) -> None:
        """Test that all configured tables get client attributes."""
        async with RemoteDatabase("http://api.example.com") as db:
            # Check that all table clients are created
            for table_name in TABLE_CONFIGS.keys():
                assert hasattr(db, table_name)
                client = getattr(db, table_name)
                assert isinstance(client, RemoteTableOperations)

    async def test_table_clients_have_correct_endpoints(self) -> None:
        """Test that table clients have correctly constructed endpoints."""
        async with RemoteDatabase("http://api.example.com", api_prefix="/api/v1") as db:
            # Check a few specific tables
            assert db.algorithms.endpoint == "http://api.example.com/api/v1/algorithms"
            assert db.datasets.endpoint == "http://api.example.com/api/v1/datasets"
            assert db.models.endpoint == "http://api.example.com/api/v1/models"

    async def test_table_clients_have_correct_models(self) -> None:
        """Test that table clients have correct response and create models."""
        async with RemoteDatabase("http://api.example.com") as db:
            from rail_svc import models

            # Check algorithms table
            assert db.algorithms.response_model == models.Algorithm
            assert db.algorithms.create_model == models.AlgorithmCreate

            # Check datasets table
            assert db.datasets.response_model == models.Dataset
            assert db.datasets.create_model == models.DatasetCreate

    async def test_table_clients_share_http_client(self) -> None:
        """Test that all table clients share the same HTTP client."""
        async with RemoteDatabase("http://api.example.com") as db:
            # Get a few different clients
            algo_client = db.algorithms
            data_client = db.datasets
            model_client = db.models

            # They should all share the same underlying HTTP client
            assert algo_client.client is data_client.client
            assert data_client.client is model_client.client
            assert algo_client.client is db._api.client

    def test_list_tables(self) -> None:
        """Test listing all available tables."""
        db = RemoteDatabase("http://api.example.com")

        tables = db.list_tables()

        assert isinstance(tables, list)
        assert len(tables) > 0
        assert "algorithms" in tables
        assert "datasets" in tables
        assert "models" in tables
        assert "bands" in tables
        assert "estimators" in tables

        # Should match TABLE_CONFIGS
        assert set(tables) == set(TABLE_CONFIGS.keys())

    async def test_get_client_existing_table(self) -> None:
        """Test getting a client for an existing table."""
        async with RemoteDatabase("http://api.example.com") as db:
            algo_client = db.get_client("algorithms")

            assert algo_client is not None
            assert isinstance(algo_client, RemoteTableOperations)
            assert algo_client is db.algorithms

    async def test_get_client_nonexistent_table(self) -> None:
        """Test getting a client for a non-existent table."""
        async with RemoteDatabase("http://api.example.com") as db:
            client = db.get_client("nonexistent_table")

            assert client is None

    async def test_get_client_returns_same_instance(self) -> None:
        """Test that get_client returns the same instance as direct attribute access."""
        async with RemoteDatabase("http://api.example.com") as db:
            client1 = db.algorithms
            client2 = db.get_client("algorithms")

            assert client1 is client2

    async def test_multiple_databases_independent(self) -> None:
        """Test that multiple RemoteDatabase instances are independent."""
        async with RemoteDatabase("http://api1.example.com", auth_token="token1") as db1:
            async with RemoteDatabase("http://api2.example.com", auth_token="token2") as db2:
                assert db1._api is not db2._api
                assert db1._api.client is not db2._api.client
                assert db1.algorithms.client is not db2.algorithms.client

    async def test_context_manager_handles_exceptions(self) -> None:
        """Test that context manager properly handles exceptions."""
        api_ref = None

        try:
            async with RemoteDatabase("http://api.example.com") as db:
                api_ref = db._api
                raise ValueError("Test exception")
        except ValueError:
            pass

        # API should still exist and be closed
        assert api_ref is not None
        assert api_ref.client.is_closed

    async def test_all_configured_tables_accessible(self) -> None:
        """Test that all tables in TABLE_CONFIGS are accessible."""
        async with RemoteDatabase("http://api.example.com") as db:
            for table_name, (response_model, create_model, ops_model) in TABLE_CONFIGS.items():
                # Check attribute exists
                assert hasattr(db, table_name), f"Table {table_name} not accessible"

                # Check it's a RemoteTableOperations instance
                client = getattr(db, table_name)
                if ops_model is None:
                    assert isinstance(client, RemoteTableOperations)
                else:
                    assert isinstance(client, RemoteTableOperations)

                # Check models are correct
                assert client.response_model == response_model
                assert client.create_model == create_model


class TestRemoteDatabaseIntegration:
    """Integration-style tests for RemoteDatabase usage patterns."""

    async def test_basic_workflow(self) -> None:
        """Test a basic workflow using multiple tables."""
        async with RemoteDatabase("http://api.example.com") as db:
            # Should be able to access multiple table clients
            assert db.algorithms is not None
            assert db.datasets is not None
            assert db.models is not None
            assert db.estimators is not None

            # All should share the same API instance
            assert db.algorithms.client is db.datasets.client

    async def test_table_discovery(self) -> None:
        """Test discovering available tables."""
        async with RemoteDatabase("http://api.example.com") as db:
            tables = db.list_tables()

            # Should be able to get clients for all listed tables
            for table_name in tables:
                client = db.get_client(table_name)
                assert client is not None
                assert isinstance(client, RemoteTableOperations)

    async def test_custom_configuration(self) -> None:
        """Test using RemoteDatabase with custom configuration."""
        async with RemoteDatabase(
            base_url="http://custom.api.com",
            api_prefix="/v2",
            timeout=120.0,
            auth_token="custom-token",
        ) as db:
            # Verify configuration is passed through
            assert db._api.base_url == "http://custom.api.com"
            assert db._api.api_prefix == "/v2"
            assert db._api.timeout == 120.0
            assert db._api.auth_token == "custom-token"

            # Verify clients use the correct configuration
            assert "http://custom.api.com/v2" in db.algorithms.endpoint

    async def test_accessing_all_tables_sequentially(self) -> None:
        """Test accessing all table clients sequentially."""
        async with RemoteDatabase("http://api.example.com") as db:
            clients = []

            # Access all tables
            clients.append(db.algorithms)
            clients.append(db.bands)
            clients.append(db.catalog_band_assocs)
            clients.append(db.catalog_tags)
            clients.append(db.datasets)
            clients.append(db.estimates)
            clients.append(db.estimators)
            clients.append(db.models)

            # All should be valid RemoteTableOperations instances
            assert all(isinstance(c, RemoteTableOperations) for c in clients)

            # All should share the same HTTP client
            http_clients = {c.client for c in clients}
            assert len(http_clients) == 1

    async def test_table_config_completeness(self) -> None:
        """Test that TABLE_CONFIGS matches expected tables."""
        expected_tables = {
            "algorithms",
            "bands",
            "catalog_band_assocs",
            "catalog_tags",
            "datasets",
            "estimates",
            "estimators",
            "models",
        }

        actual_tables = set(TABLE_CONFIGS.keys())

        assert actual_tables == expected_tables


class TestTableConfigurations:
    """Tests for TABLE_CONFIGS constant."""

    def test_table_configs_structure(self) -> None:
        """Test that TABLE_CONFIGS has the correct structure."""
        assert isinstance(TABLE_CONFIGS, dict)
        assert len(TABLE_CONFIGS) > 0

        for table_name, config in TABLE_CONFIGS.items():
            assert isinstance(table_name, str)
            assert isinstance(config, tuple)
            assert len(config) == 3

            response_model, create_model, _ops_model = config
            assert issubclass(response_model, BaseModel)
            assert issubclass(create_model, BaseModel)

    def test_all_models_are_pydantic(self) -> None:
        """Test that all models in TABLE_CONFIGS are Pydantic models."""
        for table_name, (response_model, create_model, _ops_model) in TABLE_CONFIGS.items():
            # Check they inherit from BaseModel
            assert issubclass(response_model, BaseModel), f"{table_name} response model not a BaseModel"
            assert issubclass(create_model, BaseModel), f"{table_name} create model not a BaseModel"

            # Check they're not the base class itself
            assert response_model is not BaseModel
            assert create_model is not BaseModel
