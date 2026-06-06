"""Unit tests for CLI serve command."""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, call, patch

import pytest
from click.testing import CliRunner

from rail_svc.cli.server.top import serve


class TestServeCommand:
    """Tests for the serve CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_uvicorn_run(self):
        """Mock uvicorn.run to prevent actual server startup."""
        with patch("rail_svc.cli.server.top.uvicorn.run") as mock:
            yield mock

    @pytest.fixture
    def mock_create_app(self):
        """Mock create_fastapi_app to prevent actual app creation."""
        with patch("rail_svc.cli.server.top.create_fastapi_app") as mock:
            mock.return_value = MagicMock()
            yield mock

    def test_serve_command_exists(self, runner: CliRunner) -> None:
        """Test that serve command can be invoked."""
        result = runner.invoke(serve, ["--help"])
        assert result.exit_code == 0
        assert "Start the FastAPI server" in result.output

    def test_serve_with_defaults(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with default options."""
        result = runner.invoke(serve)
        
        # Should complete successfully
        assert result.exit_code == 0
        
        # Should call uvicorn.run
        assert mock_uvicorn_run.called
        
        # Should create app
        assert mock_create_app.called

    def test_serve_custom_host_and_port(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with custom host and port."""
        result = runner.invoke(serve, ["--host", "0.0.0.0", "--port", "9000"])
        
        assert result.exit_code == 0
        
        # Check uvicorn was called with correct host/port
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs["host"] == "0.0.0.0"
        assert call_kwargs["port"] == 9000

    def test_serve_with_reload(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with reload enabled."""
        result = runner.invoke(serve, ["--reload"])
        
        assert result.exit_code == 0
        assert "development mode" in result.output
        assert "Auto-reload is enabled" in result.output
        
        # Check reload flag is set
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs.get("reload") is True
        assert "workers" not in call_kwargs

    def test_serve_with_workers(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with multiple workers."""
        result = runner.invoke(serve, ["--workers", "4"])
        
        assert result.exit_code == 0
        assert "production mode" in result.output
        assert "4 worker" in result.output
        
        # Check workers are set
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs.get("workers") == 4
        assert "reload" not in call_kwargs

    def test_serve_with_custom_api_prefix(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with custom API prefix."""
        result = runner.invoke(serve, ["--api-prefix", "/api/v2"])
        
        assert result.exit_code == 0
        assert "/api/v2" in result.output
        
        # Check app was created with custom prefix
        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["api_prefix"] == "/api/v2"

    def test_serve_with_debug(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with debug mode."""
        result = runner.invoke(serve, ["--debug"])
        
        assert result.exit_code == 0
        
        # Check app was created with debug=True
        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["debug"] is True

    def test_serve_with_rate_limiting(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with rate limiting enabled."""
        result = runner.invoke(
            serve,
            [
                "--enable-rate-limiting",
                "--rate-limit-storage", "redis://localhost:6379"
            ]
        )
        
        assert result.exit_code == 0
        
        # Check app was created with rate limiting
        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["enable_rate_limiting"] is True
        assert call_kwargs["rate_limit_storage"] == "redis://localhost:6379"

    def test_serve_with_cors(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with CORS enabled."""
        result = runner.invoke(
            serve,
            [
                "--enable-cors",
                "--cors-origins", "http://localhost:3000,https://example.com"
            ]
        )
        
        assert result.exit_code == 0
        
        # Check app was created with CORS
        call_kwargs = mock_create_app.call_args[1]
        assert call_kwargs["enable_cors"] is True
        assert "http://localhost:3000" in call_kwargs["cors_origins"]
        assert "https://example.com" in call_kwargs["cors_origins"]

    def test_serve_cors_origins_parsing(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test that CORS origins are correctly parsed from comma-separated string."""
        result = runner.invoke(
            serve,
            ["--cors-origins", "http://a.com, http://b.com , http://c.com"]
        )
        
        assert result.exit_code == 0
        
        # Check origins are trimmed
        call_kwargs = mock_create_app.call_args[1]
        cors_origins = call_kwargs["cors_origins"]
        assert "http://a.com" in cors_origins
        assert "http://b.com" in cors_origins
        assert "http://c.com" in cors_origins
        # No whitespace
        assert all(not origin.startswith(" ") for origin in cors_origins)

    def test_serve_log_level(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve command with custom log level."""
        result = runner.invoke(serve, ["--log-level", "debug"])
        
        assert result.exit_code == 0
        
        # Check log level is passed to uvicorn
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs["log_level"] == "debug"

    def test_serve_displays_helpful_output(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test that serve command displays helpful startup information."""
        result = runner.invoke(serve, ["--host", "localhost", "--port", "8000"])
        
        assert result.exit_code == 0
        
        # Should show helpful messages
        assert "http://localhost:8000" in result.output
        assert "/health" in result.output
        assert "/docs" in result.output

    def test_serve_app_configuration(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test that app is created with correct configuration."""
        result = runner.invoke(serve)
        
        assert result.exit_code == 0
        
        # Check app creation call
        call_kwargs = mock_create_app.call_args[1]
        assert "title" in call_kwargs
        assert "description" in call_kwargs
        assert "version" in call_kwargs
        assert call_kwargs["title"] == "Database API"

    def test_serve_uvicorn_receives_app(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test that uvicorn.run receives the created app."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app
        
        result = runner.invoke(serve)
        
        assert result.exit_code == 0
        
        # First argument to uvicorn.run should be the app
        assert mock_uvicorn_run.call_args[0][0] is mock_app


class TestServeCommandEdgeCases:
    """Tests for edge cases and combinations."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def mock_uvicorn_run(self):
        """Mock uvicorn.run."""
        with patch("rail_svc.cli.server.top.uvicorn.run") as mock:
            yield mock

    @pytest.fixture
    def mock_create_app(self):
        """Mock create_fastapi_app."""
        with patch("rail_svc.cli.server.top.create_fastapi_app") as mock:
            mock.return_value = MagicMock()
            yield mock

    def test_serve_reload_and_workers_mutually_exclusive_behavior(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test that reload takes precedence over workers."""
        result = runner.invoke(serve, ["--reload", "--workers", "4"])
        
        assert result.exit_code == 0
        
        # Reload should be set, workers should not
        call_kwargs = mock_uvicorn_run.call_args[1]
        assert call_kwargs.get("reload") is True
        assert "workers" not in call_kwargs

    def test_serve_empty_cors_origins(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve with empty CORS origins string."""
        result = runner.invoke(serve, ["--cors-origins", ""])
        
        assert result.exit_code == 0
        
        call_kwargs = mock_create_app.call_args[1]
        # Should result in a list with one empty string (after strip)
        assert isinstance(call_kwargs["cors_origins"], list)

    def test_serve_all_options_combined(
        self, runner: CliRunner, mock_uvicorn_run: Mock, mock_create_app: Mock
    ) -> None:
        """Test serve with many options combined."""
        result = runner.invoke(
            serve,
            [
                "--host", "0.0.0.0",
                "--port", "9000",
                "--workers", "2",
                "--log-level", "info",
                "--api-prefix", "/api/v3",
                "--enable-rate-limiting",
                "--rate-limit-storage", "memory://",
                "--enable-cors",
                "--cors-origins", "http://localhost:3000",
            ]
        )
        
        assert result.exit_code == 0
        
        # Verify key configurations made it through
        uvicorn_kwargs = mock_uvicorn_run.call_args[1]
        assert uvicorn_kwargs["host"] == "0.0.0.0"
        assert uvicorn_kwargs["port"] == 9000
        
        app_kwargs = mock_create_app.call_args[1]
        assert app_kwargs["api_prefix"] == "/api/v3"
        assert app_kwargs["enable_cors"] is True
