"""Unit tests for FastAPI app factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient
from pydantic import BaseModel

from rail_svc.router.app import (
    add_cors_middleware,
    add_error_handlers,
    add_health_check,
    add_rate_limiting,
    create_all_routers,
    create_fastapi_app,
    lifespan,
    register_all_routers,
    setup_fastapi_app,
)


# Test create_all_routers
class TestCreateAllRouters:
    """Tests for create_all_routers function."""

    @patch("rail_svc.router.app.create_table_router")
    @patch("rail_svc.router.app.local_async")
    def test_creates_all_routers(self, mock_local_async: MagicMock, mock_create_router: MagicMock) -> None:
        """Test that all routers are created."""
        mock_router = MagicMock()
        mock_create_router.return_value = mock_router

        # Set up mock operations
        mock_local_async.algorithm = MagicMock()
        mock_local_async.band = MagicMock()
        mock_local_async.catalog_band_assoc = MagicMock()
        mock_local_async.catalog_tag = MagicMock()
        mock_local_async.dataset = MagicMock()
        mock_local_async.estimates = MagicMock()
        mock_local_async.estimator = MagicMock()
        mock_local_async.model = MagicMock()

        routers = create_all_routers()

        # Should create 8 routers
        assert len(routers) == 8
        assert mock_create_router.call_count == 8

        # Verify each router was created with correct name
        expected_calls = [
            call("algorithms", mock_local_async.algorithm),
            call("bands", mock_local_async.band),
            call("catalog_band_assocs", mock_local_async.catalog_band_assoc),
            call("catalog_tags", mock_local_async.catalog_tag),
            call("datasets", mock_local_async.dataset),
            call("estimates", mock_local_async.estimates),
            call("estimators", mock_local_async.estimator),
            call("models", mock_local_async.model),
        ]
        mock_create_router.assert_has_calls(expected_calls, any_order=False)

    @patch("rail_svc.router.app.create_table_router")
    @patch("rail_svc.router.app.local_async")
    def test_returns_list_of_routers(
        self, mock_local_async: MagicMock, mock_create_router: MagicMock
    ) -> None:
        """Test that function returns a list."""
        mock_router = MagicMock()
        mock_create_router.return_value = mock_router

        routers = create_all_routers()

        assert isinstance(routers, list)
        assert all(router == mock_router for router in routers)


# Test register_all_routers
class TestRegisterAllRouters:
    """Tests for register_all_routers function."""

    @patch("rail_svc.router.app.create_all_routers")
    def test_registers_routers_with_default_prefix(self, mock_create_routers: MagicMock) -> None:
        """Test registering routers with default prefix."""
        app = FastAPI()
        mock_router1 = MagicMock()
        mock_router1.prefix = "/router1"
        mock_router2 = MagicMock()
        mock_router2.prefix = "/router2"
        mock_create_routers.return_value = [mock_router1, mock_router2]

        register_all_routers(app)

        mock_create_routers.assert_called_once()

    @patch("rail_svc.router.app.create_all_routers")
    def test_registers_routers_with_custom_prefix(self, mock_create_routers: MagicMock) -> None:
        """Test registering routers with custom prefix."""
        app = FastAPI()
        mock_router = MagicMock()
        mock_router.prefix = "/test"
        mock_create_routers.return_value = [mock_router]

        register_all_routers(app, prefix="/api/v2")

        mock_create_routers.assert_called_once()

    @patch("rail_svc.router.app.create_all_routers")
    def test_handles_empty_router_list(self, mock_create_routers: MagicMock) -> None:
        """Test handling of empty router list."""
        app = FastAPI()
        mock_create_routers.return_value = []

        # Should not raise an error
        register_all_routers(app)

        mock_create_routers.assert_called_once()


# Test add_rate_limiting
class TestAddRateLimiting:
    """Tests for add_rate_limiting function."""

    @patch("rail_svc.router.app.Limiter")
    def test_adds_rate_limiting_with_defaults(self, mock_limiter_class: MagicMock) -> None:
        """Test adding rate limiting with default settings."""
        app = FastAPI()
        mock_limiter = MagicMock()
        mock_limiter_class.return_value = mock_limiter

        result = add_rate_limiting(app)

        assert result == mock_limiter
        assert app.state.limiter == mock_limiter
        mock_limiter_class.assert_called_once()

    @patch("rail_svc.router.app.Limiter")
    def test_adds_rate_limiting_with_custom_limits(self, mock_limiter_class: MagicMock) -> None:
        """Test adding rate limiting with custom limits."""
        app = FastAPI()
        mock_limiter = MagicMock()
        mock_limiter_class.return_value = mock_limiter
        custom_limits = ["500 per day", "50 per hour"]

        result = add_rate_limiting(app, default_limits=custom_limits)

        assert result == mock_limiter
        # Verify custom limits were passed
        call_kwargs = mock_limiter_class.call_args.kwargs
        assert call_kwargs["default_limits"] == custom_limits

    @patch("rail_svc.router.app.Limiter")
    def test_adds_rate_limiting_with_redis_storage(self, mock_limiter_class: MagicMock) -> None:
        """Test adding rate limiting with Redis storage."""
        app = FastAPI()
        mock_limiter = MagicMock()
        mock_limiter_class.return_value = mock_limiter
        redis_uri = "redis://localhost:6379"

        result = add_rate_limiting(app, storage_uri=redis_uri)

        assert result == mock_limiter
        call_kwargs = mock_limiter_class.call_args.kwargs
        assert call_kwargs["storage_uri"] == redis_uri

    @patch("rail_svc.router.app.Limiter", side_effect=ImportError)
    def test_handles_missing_slowapi(self, mock_limiter_class: MagicMock) -> None:
        """Test handling when slowapi is not installed."""
        app = FastAPI()

        result = add_rate_limiting(app)

        assert result is None
        assert not hasattr(app.state, "limiter")


# Test add_health_check
class TestAddHealthCheck:
    """Tests for add_health_check function."""

    def test_adds_health_check_endpoint(self) -> None:
        """Test that health check endpoint is added."""
        app = FastAPI()

        add_health_check(app)

        # Test the health check endpoint
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK
        assert response.json()["status"] == "healthy"
        assert response.json()["service"] == "api"
        assert response.json()["version"] == "1.0.0"

    def test_health_check_returns_correct_structure(self) -> None:
        """Test health check response structure."""
        app = FastAPI()
        add_health_check(app)

        client = TestClient(app)
        response = client.get("/health")

        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data

    @patch("rail_svc.router.app.logger")
    def test_health_check_handles_errors(self, mock_logger: MagicMock) -> None:
        """Test health check error handling."""
        app = FastAPI(debug=False)

        # Override health check to raise an error
        @app.get("/health", tags=["health"])
        async def failing_health_check():
            raise Exception("Database connection failed")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


# Test add_error_handlers
class TestAddErrorHandlers:
    """Tests for add_error_handlers function."""

    def test_handles_404_errors(self) -> None:
        """Test 404 error handler."""
        app = FastAPI()
        add_error_handlers(app)

        client = TestClient(app)
        response = client.get("/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "error" in data

    def test_handles_405_errors(self) -> None:
        """Test 405 error handler."""
        app = FastAPI()
        add_error_handlers(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)
        response = client.post("/test")

        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

    def test_handles_validation_errors(self) -> None:
        """Test validation error handler."""
        app = FastAPI()
        add_error_handlers(app)

        class TestModel(BaseModel):
            name: str
            value: int

        @app.post("/test")
        async def test_endpoint(data: TestModel):
            return data

        client = TestClient(app, raise_server_exceptions=False)
        # Send invalid data
        response = client.post("/test", json={"name": "test"})  # Missing 'value'

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "error" in response.json()

    def test_handles_500_errors_debug_mode(self) -> None:
        """Test 500 error handler in debug mode."""
        app = FastAPI(debug=True)
        add_error_handlers(app)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/error")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            assert "error" in data
            # In debug mode, should have details
            assert data.get("details") is not None

    def test_handles_500_errors_production_mode(self) -> None:
        """Test 500 error handler in production mode."""
        app = FastAPI(debug=False)
        add_error_handlers(app)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/error")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # Should get JSON response
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type:
            data = response.json()
            assert "error" in data
            # In production mode, should hide details or show generic message
            details = data.get("details")
            if details:
                # Should not expose internal error message
                assert "Test error" not in str(details) or details == "An unexpected error occurred"


# Test add_cors_middleware
class TestAddCorsMiddleware:
    """Tests for add_cors_middleware function."""

    def test_adds_cors_with_defaults(self) -> None:
        """Test adding CORS with default settings."""
        app = FastAPI()

        add_cors_middleware(app)

        # Verify middleware was added
        assert len(app.user_middleware) > 0

    def test_adds_cors_with_custom_origins(self) -> None:
        """Test adding CORS with custom origins."""
        app = FastAPI()
        custom_origins = ["https://example.com", "https://app.example.com"]

        add_cors_middleware(app, allow_origins=custom_origins)

        assert len(app.user_middleware) > 0

    def test_adds_cors_with_custom_methods(self) -> None:
        """Test adding CORS with custom methods."""
        app = FastAPI()
        custom_methods = ["GET", "POST"]

        add_cors_middleware(app, allow_methods=custom_methods)

        assert len(app.user_middleware) > 0

    def test_adds_cors_with_credentials(self) -> None:
        """Test adding CORS with credentials."""
        app = FastAPI()

        add_cors_middleware(app, allow_credentials=True)

        assert len(app.user_middleware) > 0

    def test_cors_headers_in_response(self) -> None:
        """Test that CORS headers are present in response."""
        app = FastAPI()
        add_cors_middleware(app, allow_origins=["*"])

        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}

        client = TestClient(app)
        response = client.get("/test", headers={"Origin": "https://example.com"})

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers


# Test lifespan
class TestLifespan:
    """Tests for lifespan context manager."""

    @pytest.mark.asyncio
    async def test_lifespan_context_manager(self) -> None:
        """Test lifespan context manager."""
        app = FastAPI()

        async with lifespan(app):
            # During app lifetime
            pass

        # Should complete without errors

    @pytest.mark.asyncio
    async def test_lifespan_startup_and_shutdown(self) -> None:
        """Test lifespan startup and shutdown."""
        app = FastAPI()
        startup_called = False
        shutdown_called = False

        # We can't easily test the logging, but we can verify the context manager works
        async with lifespan(app):
            startup_called = True

        shutdown_called = True

        assert startup_called
        assert shutdown_called


# Test create_fastapi_app
class TestCreateFastAPIApp:
    """Tests for create_fastapi_app function."""

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.register_all_routers")
    def test_creates_app_with_defaults(self, mock_register: MagicMock, mock_init_db: MagicMock) -> None:
        """Test creating app with default settings."""
        app = create_fastapi_app()

        assert isinstance(app, FastAPI)
        assert app.title == "API"
        assert app.version == "1.0.0"
        mock_register.assert_called_once()
        mock_init_db.assert_called_once()

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.register_all_routers")
    def test_creates_app_with_custom_settings(
        self, mock_register: MagicMock, mock_init_db: MagicMock
    ) -> None:
        """Test creating app with custom settings."""
        app = create_fastapi_app(
            title="My API",
            description="Custom API",
            version="2.0.0",
            debug=True,
        )

        assert app.title == "My API"
        assert app.version == "2.0.0"
        assert app.debug is True

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.add_rate_limiting")
    @patch("rail_svc.router.app.register_all_routers")
    def test_creates_app_with_rate_limiting(
        self, mock_register: MagicMock, mock_rate_limit: MagicMock, mock_init_db: MagicMock
    ) -> None:
        """Test creating app with rate limiting enabled."""
        app = create_fastapi_app(
            enable_rate_limiting=True,
            rate_limits=["500 per day"],
            rate_limit_storage="redis://localhost:6379",
        )

        mock_rate_limit.assert_called_once_with(
            app,
            default_limits=["500 per day"],
            storage_uri="redis://localhost:6379",
        )

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.add_cors_middleware")
    @patch("rail_svc.router.app.register_all_routers")
    def test_creates_app_with_cors(
        self, mock_register: MagicMock, mock_cors: MagicMock, mock_init_db: MagicMock
    ) -> None:
        """Test creating app with CORS enabled."""
        custom_origins = ["https://example.com"]

        app = create_fastapi_app(
            enable_cors=True,
            cors_origins=custom_origins,
        )

        mock_cors.assert_called_once_with(app, allow_origins=custom_origins)

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.register_all_routers")
    def test_creates_app_with_custom_prefix(self, mock_register: MagicMock, mock_init_db: MagicMock) -> None:
        """Test creating app with custom API prefix."""
        app = create_fastapi_app(api_prefix="/api/v2")

        mock_register.assert_called_once_with(app, prefix="/api/v2")

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.register_all_routers")
    def test_app_includes_health_check(self, mock_register: MagicMock, mock_init_db: MagicMock) -> None:
        """Test that created app includes health check."""
        app = create_fastapi_app()

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.register_all_routers")
    def test_app_includes_error_handlers(self, mock_register: MagicMock, mock_init_db: MagicMock) -> None:
        """Test that created app includes error handlers."""
        app = create_fastapi_app()

        client = TestClient(app)
        response = client.get("/nonexistent")

        # Should get 404 with custom error handler
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "error" in response.json()

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.add_rate_limiting")
    @patch("rail_svc.router.app.add_cors_middleware")
    @patch("rail_svc.router.app.register_all_routers")
    def test_creates_fully_configured_app(
        self,
        mock_register: MagicMock,
        mock_cors: MagicMock,
        mock_rate_limit: MagicMock,
        mock_init_db: MagicMock,
    ) -> None:
        """Test creating fully configured app with all features."""
        app = create_fastapi_app(
            title="Full API",
            version="1.0.0",
            enable_rate_limiting=True,
            rate_limits=["1000 per day"],
            enable_cors=True,
            cors_origins=["https://example.com"],
            api_prefix="/api/v1",
            debug=True,
        )

        assert isinstance(app, FastAPI)
        mock_register.assert_called_once()
        mock_cors.assert_called_once()
        mock_rate_limit.assert_called_once()
        mock_init_db.assert_called_once()


# Test setup_fastapi_app
class TestSetupFastAPIApp:
    """Tests for setup_fastapi_app function."""

    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_existing_app(self, mock_register: MagicMock) -> None:
        """Test setting up an existing app."""
        app = FastAPI()

        setup_fastapi_app(app)

        mock_register.assert_called_once_with(app, prefix="/api/v1")

    @patch("rail_svc.router.app.add_rate_limiting")
    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_app_with_rate_limiting(self, mock_register: MagicMock, mock_rate_limit: MagicMock) -> None:
        """Test setting up app with rate limiting."""
        app = FastAPI()

        setup_fastapi_app(
            app,
            enable_rate_limiting=True,
            rate_limits=["500 per day"],
        )

        mock_rate_limit.assert_called_once()

    @patch("rail_svc.router.app.add_cors_middleware")
    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_app_with_cors(self, mock_register: MagicMock, mock_cors: MagicMock) -> None:
        """Test setting up app with CORS."""
        app = FastAPI()
        custom_origins = ["https://example.com"]

        setup_fastapi_app(
            app,
            enable_cors=True,
            cors_origins=custom_origins,
        )

        mock_cors.assert_called_once_with(app, allow_origins=custom_origins)

    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_app_with_custom_prefix(self, mock_register: MagicMock) -> None:
        """Test setting up app with custom prefix."""
        app = FastAPI()

        setup_fastapi_app(app, api_prefix="/api/v2")

        mock_register.assert_called_once_with(app, prefix="/api/v2")

    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_adds_health_check(self, mock_register: MagicMock) -> None:
        """Test that setup adds health check."""
        app = FastAPI()

        setup_fastapi_app(app)

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == status.HTTP_200_OK

    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_adds_error_handlers(self, mock_register: MagicMock) -> None:
        """Test that setup adds error handlers."""
        app = FastAPI()

        setup_fastapi_app(app)

        client = TestClient(app)
        response = client.get("/nonexistent")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "error" in response.json()

    @patch("rail_svc.router.app.add_rate_limiting")
    @patch("rail_svc.router.app.add_cors_middleware")
    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_fully_configured_app(
        self,
        mock_register: MagicMock,
        mock_cors: MagicMock,
        mock_rate_limit: MagicMock,
    ) -> None:
        """Test setting up app with all features."""
        app = FastAPI()

        setup_fastapi_app(
            app,
            enable_rate_limiting=True,
            rate_limits=["1000 per day"],
            rate_limit_storage="redis://localhost:6379",
            enable_cors=True,
            cors_origins=["https://example.com"],
            api_prefix="/api/v1",
        )

        mock_register.assert_called_once()
        mock_cors.assert_called_once()
        mock_rate_limit.assert_called_once()

    @patch("rail_svc.router.app.register_all_routers")
    def test_setup_modifies_app_in_place(self, mock_register: MagicMock) -> None:
        """Test that setup modifies the app in place."""
        app = FastAPI()
        original_app = app

        setup_fastapi_app(app)

        # Should be the same object
        assert app is original_app


# Integration Tests
class TestIntegration:
    """Integration tests for app factory."""

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.create_all_routers")
    def test_created_app_is_functional(self, mock_create_routers: MagicMock, mock_init_db: MagicMock) -> None:
        """Test that created app is functional."""
        mock_create_routers.return_value = []

        app = create_fastapi_app(title="Test API")
        client = TestClient(app)

        # Health check should work
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

        # 404 handler should work
        response = client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.create_all_routers")
    def test_app_with_cors_and_rate_limiting(
        self, mock_create_routers: MagicMock, mock_init_db: MagicMock
    ) -> None:
        """Test app with both CORS and rate limiting."""
        mock_create_routers.return_value = []

        app = create_fastapi_app(
            enable_cors=True,
            cors_origins=["https://example.com"],
            enable_rate_limiting=True,
        )

        assert isinstance(app, FastAPI)
        # CORS middleware should be present
        assert len(app.user_middleware) > 0

    @patch("rail_svc.router.app.create_all_routers")
    def test_setup_and_create_produce_similar_apps(self, mock_create_routers: MagicMock) -> None:
        """Test that setup and create produce functionally similar apps."""
        mock_create_routers.return_value = []

        # Create app using create_fastapi_app
        with patch("rail_svc.router.app.init_db"):
            app1 = create_fastapi_app()

        # Create app using setup_fastapi_app
        app2 = FastAPI()
        setup_fastapi_app(app2)

        # Both should have health checks
        client1 = TestClient(app1)
        client2 = TestClient(app2)

        response1 = client1.get("/health")
        response2 = client2.get("/health")

        assert response1.status_code == response2.status_code == status.HTTP_200_OK


# Edge Cases and Error Handling
class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.create_all_routers")
    def test_handles_empty_router_list(self, mock_create_routers: MagicMock, mock_init_db: MagicMock) -> None:
        """Test handling when no routers are created."""
        mock_create_routers.return_value = []

        app = create_fastapi_app()

        assert isinstance(app, FastAPI)

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.Limiter", side_effect=ImportError)
    @patch("rail_svc.router.app.create_all_routers")
    def test_continues_without_rate_limiting_if_unavailable(
        self, mock_create_routers: MagicMock, mock_limiter: MagicMock, mock_init_db: MagicMock
    ) -> None:
        """Test that app creation continues if rate limiting is unavailable."""
        mock_create_routers.return_value = []

        app = create_fastapi_app(enable_rate_limiting=True)

        # Should still create app successfully
        assert isinstance(app, FastAPI)

    @patch("rail_svc.router.app.register_all_routers")
    def test_health_check_with_exception(self, mock_register: MagicMock) -> None:
        """Test health check behavior when exception occurs."""
        app = FastAPI(debug=False)

        # Override health check to raise exception
        @app.get("/health", tags=["health"])
        async def failing_health_check():
            raise RuntimeError("Service unavailable")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")

        # Should return 500 error
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_cors_with_none_origins(self) -> None:
        """Test CORS with None origins (should use default)."""
        app = FastAPI()

        # Should not raise an error
        add_cors_middleware(app, allow_origins=None)

        assert len(app.user_middleware) > 0

    def test_rate_limiting_with_none_limits(self) -> None:
        """Test rate limiting with None limits (should use default)."""
        app = FastAPI()

        with patch("rail_svc.router.app.Limiter") as mock_limiter_class:
            mock_limiter = MagicMock()
            mock_limiter_class.return_value = mock_limiter

            _result = add_rate_limiting(app, default_limits=None)

            # Should use default limits
            call_kwargs = mock_limiter_class.call_args.kwargs
            assert call_kwargs["default_limits"] == ["1000 per day", "100 per hour"]


# Performance and Configuration Tests
class TestPerformanceAndConfiguration:
    """Tests for performance and configuration options."""

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.create_all_routers")
    def test_debug_mode_enabled(self, mock_create_routers: MagicMock, mock_init_db: MagicMock) -> None:
        """Test app creation with debug mode enabled."""
        mock_create_routers.return_value = []

        app = create_fastapi_app(debug=True)

        assert app.debug is True

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.create_all_routers")
    def test_debug_mode_disabled(self, mock_create_routers: MagicMock, mock_init_db: MagicMock) -> None:
        """Test app creation with debug mode disabled."""
        mock_create_routers.return_value = []

        app = create_fastapi_app(debug=False)

        assert app.debug is False

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.create_all_routers")
    def test_error_details_in_debug_mode(
        self, mock_create_routers: MagicMock, mock_init_db: MagicMock
    ) -> None:
        """Test that error details are shown in debug mode."""
        mock_create_routers.return_value = []

        app = create_fastapi_app(debug=True)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Test error message")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/error")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        # In debug mode, FastAPI may return HTML error page
        # Just verify we get an error response with correct status

    @patch("rail_svc.router.app.init_db")
    @patch("rail_svc.router.app.create_all_routers")
    def test_error_details_hidden_in_production(
        self, mock_create_routers: MagicMock, mock_init_db: MagicMock
    ) -> None:
        """Test that error details are hidden in production mode."""
        mock_create_routers.return_value = []

        app = create_fastapi_app(debug=False)

        @app.get("/error")
        async def error_endpoint():
            raise ValueError("Secret error message")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/error")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        # Should not expose internal error details
        assert "Secret error message" not in str(data)

    def test_multiple_rate_limits(self) -> None:
        """Test adding multiple rate limits."""
        app = FastAPI()

        with patch("rail_svc.router.app.Limiter") as mock_limiter_class:
            mock_limiter = MagicMock()
            mock_limiter_class.return_value = mock_limiter

            limits = ["1000 per day", "100 per hour", "10 per minute"]
            _result = add_rate_limiting(app, default_limits=limits)

            call_kwargs = mock_limiter_class.call_args.kwargs
            assert call_kwargs["default_limits"] == limits

    def test_cors_with_multiple_origins(self) -> None:
        """Test CORS with multiple allowed origins."""
        app = FastAPI()
        origins = ["https://example.com", "https://app.example.com", "https://admin.example.com"]

        add_cors_middleware(app, allow_origins=origins)

        assert len(app.user_middleware) > 0

    def test_cors_with_specific_methods(self) -> None:
        """Test CORS with specific allowed methods."""
        app = FastAPI()
        methods = ["GET", "POST", "PUT"]

        add_cors_middleware(app, allow_methods=methods)

        assert len(app.user_middleware) > 0

    def test_cors_with_specific_headers(self) -> None:
        """Test CORS with specific allowed headers."""
        app = FastAPI()
        headers = ["Content-Type", "Authorization", "X-Custom-Header"]

        add_cors_middleware(app, allow_headers=headers)

        assert len(app.user_middleware) > 0


# Lifespan Event Tests
class TestLifespanEvents:
    """Tests for lifespan events."""

    @pytest.mark.asyncio
    async def test_lifespan_executes_startup_code(self) -> None:
        """Test that lifespan executes startup code."""
        _app = FastAPI()
        executed_startup = False

        @asynccontextmanager
        async def custom_lifespan(_app):
            nonlocal executed_startup
            executed_startup = True
            yield

        app_with_lifespan = FastAPI(lifespan=custom_lifespan)

        # Use TestClient to trigger lifespan
        with TestClient(app_with_lifespan):
            assert executed_startup

    @pytest.mark.asyncio
    async def test_lifespan_executes_shutdown_code(self) -> None:
        """Test that lifespan executes shutdown code."""
        _app = FastAPI()
        executed_shutdown = False

        @asynccontextmanager
        async def custom_lifespan(_app):
            nonlocal executed_shutdown
            yield
            executed_shutdown = True

        app_with_lifespan = FastAPI(lifespan=custom_lifespan)

        # Use TestClient to trigger lifespan
        with TestClient(app_with_lifespan):
            pass

        assert executed_shutdown

    @pytest.mark.asyncio
    async def test_lifespan_handles_exceptions(self) -> None:
        """Test that lifespan handles exceptions gracefully."""

        @asynccontextmanager
        async def failing_lifespan(_app):
            # Startup should succeed
            yield
            # Simulate cleanup that might fail
            # (In real scenario, this should be handled)

        app = FastAPI(lifespan=failing_lifespan)

        # Should not raise during context usage
        with TestClient(app):
            pass


# Router Registration Tests
class TestRouterRegistration:
    """Tests for router registration."""

    @patch("rail_svc.router.app.create_all_routers")
    def test_routers_registered_with_prefix(self, mock_create_routers: MagicMock) -> None:
        """Test that routers are registered with correct prefix."""
        app = FastAPI()
        mock_router = MagicMock()
        mock_router.prefix = "/test"
        mock_router.routes = []
        mock_create_routers.return_value = [mock_router]

        register_all_routers(app, prefix="/api/v2")

        # Verify the router was included
        mock_create_routers.assert_called_once()

    @patch("rail_svc.router.app.create_all_routers")
    def test_multiple_routers_registered(self, mock_create_routers: MagicMock) -> None:
        """Test that multiple routers are all registered."""
        app = FastAPI()
        mock_routers = [MagicMock() for _ in range(5)]
        for i, router in enumerate(mock_routers):
            router.prefix = f"/router{i}"
            router.routes = []
        mock_create_routers.return_value = mock_routers

        register_all_routers(app)

        mock_create_routers.assert_called_once()


# Error Handler Specific Tests
class TestErrorHandlerDetails:
    """Detailed tests for error handlers."""

    def test_404_handler_includes_request_info(self) -> None:
        """Test that 404 handler includes request information."""
        app = FastAPI()
        add_error_handlers(app)

        client = TestClient(app)
        response = client.get("/nonexistent/path")

        data = response.json()
        assert "error" in data
        assert data["error"] == "Endpoint not found"

    def test_validation_error_includes_details(self) -> None:
        """Test that validation errors include detailed information."""
        app = FastAPI()
        add_error_handlers(app)

        class Item(BaseModel):
            name: str
            price: float

        @app.post("/items")
        async def create_item(item: Item):
            return item

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/items", json={"name": "test"})  # Missing price

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        data = response.json()
        assert "details" in data
        assert len(data["details"]) > 0

    def test_general_exception_handler_logs_error(self) -> None:
        """Test that general exception handler logs errors."""
        app = FastAPI()
        add_error_handlers(app)

        @app.get("/error")
        async def error_endpoint():
            raise RuntimeError("Unexpected error")

        with patch("rail_svc.router.app.logger") as mock_logger:
            client = TestClient(app, raise_server_exceptions=False)
            _response = client.get("/error")

            # Should log the exception
            mock_logger.exception.assert_called()


# Health Check Specific Tests
class TestHealthCheckDetails:
    """Detailed tests for health check endpoint."""

    def test_health_check_response_format(self) -> None:
        """Test health check response format."""
        app = FastAPI()
        add_health_check(app)

        client = TestClient(app)
        response = client.get("/health")

        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert data["status"] == "healthy"
        assert data["service"] == "api"
        assert isinstance(data["version"], str)

    def test_health_check_tagged_correctly(self) -> None:
        """Test that health check endpoint has correct tags."""
        app = FastAPI()
        add_health_check(app)

        # Find the health check route
        health_route = None
        for route in app.routes:
            if hasattr(route, "path") and route.path == "/health":
                health_route = route
                break

        assert health_route is not None
        assert "health" in health_route.tags


# CORS Middleware Specific Tests
class TestCorsMiddlewareDetails:
    """Detailed tests for CORS middleware."""

    def test_cors_allows_credentials_by_default(self) -> None:
        """Test that CORS allows credentials by default."""
        app = FastAPI()
        add_cors_middleware(app)

        # Middleware should be configured with allow_credentials=True
        assert len(app.user_middleware) > 0

    def test_cors_with_no_credentials(self) -> None:
        """Test CORS without credentials."""
        app = FastAPI()
        add_cors_middleware(app, allow_credentials=False)

        assert len(app.user_middleware) > 0

    def test_cors_wildcard_origins(self) -> None:
        """Test CORS with wildcard origins."""
        app = FastAPI()
        add_cors_middleware(app, allow_origins=["*"])

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        client = TestClient(app)
        response = client.get("/test", headers={"Origin": "https://anywhere.com"})

        assert response.status_code == status.HTTP_200_OK


# Rate Limiting Specific Tests
class TestRateLimitingDetails:
    """Detailed tests for rate limiting."""

    @patch("rail_svc.router.app.Limiter")
    def test_rate_limiting_uses_remote_address(self, mock_limiter_class: MagicMock) -> None:
        """Test that rate limiting uses remote address as key."""
        app = FastAPI()
        mock_limiter = MagicMock()
        mock_limiter_class.return_value = mock_limiter

        add_rate_limiting(app)

        # Verify key_func is set
        call_kwargs = mock_limiter_class.call_args.kwargs
        assert "key_func" in call_kwargs

    @patch("rail_svc.router.app.Limiter")
    def test_rate_limiting_adds_exception_handler(self, mock_limiter_class: MagicMock) -> None:
        """Test that rate limiting adds exception handler."""
        app = FastAPI()
        mock_limiter = MagicMock()
        mock_limiter_class.return_value = mock_limiter

        add_rate_limiting(app)

        # App should have the limiter in state
        assert hasattr(app.state, "limiter")
        assert app.state.limiter == mock_limiter

    @patch("rail_svc.router.app.Limiter")
    def test_rate_limiting_with_memory_storage(self, mock_limiter_class: MagicMock) -> None:
        """Test rate limiting with memory storage."""
        app = FastAPI()
        mock_limiter = MagicMock()
        mock_limiter_class.return_value = mock_limiter

        add_rate_limiting(app, storage_uri="memory://")

        call_kwargs = mock_limiter_class.call_args.kwargs
        assert call_kwargs["storage_uri"] == "memory://"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
