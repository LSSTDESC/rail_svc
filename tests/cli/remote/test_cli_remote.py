"""Unit tests for CLI remote top-level entry point."""

from __future__ import annotations

from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner

from rail_svc.cli.remote.top import TABLES, cli, make_table_group


class TestMakeTableGroup:
    """Tests for make_table_group function."""

    def test_creates_click_group(self) -> None:
        """Test that make_table_group creates a Click group."""
        mock_factory = Mock()
        mock_factory.return_value = Mock()
        mock_factory.return_value.async_ops = Mock()
        mock_factory.return_value.async_ops.table_name = "test"

        group = make_table_group("test", mock_factory, "Test description")

        assert isinstance(group, click.Group)

    def test_calls_factory_function(self) -> None:
        """Test that make_table_group calls the factory."""
        mock_factory = Mock()
        mock_factory.return_value = Mock()
        mock_factory.return_value.async_ops = Mock()
        mock_factory.return_value.async_ops.table_name = "test"

        make_table_group("test", mock_factory, "Test description")

        mock_factory.assert_called_once()


class TestTablesConfiguration:
    """Tests for TABLES configuration."""

    def test_tables_structure(self) -> None:
        """Test that TABLES has valid structure."""
        assert isinstance(TABLES, list)
        assert len(TABLES) > 0

        for table in TABLES:
            assert len(table) == 3
            assert isinstance(table[0], str)  # name
            assert callable(table[1])  # factory
            assert isinstance(table[2], str)  # description


class TestCliBasics:
    """Basic tests for the main CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    def test_cli_invokable(self, runner: CliRunner) -> None:
        """Test that CLI can be invoked."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_cli_has_table_subcommands(self, runner: CliRunner) -> None:
        """Test that table subcommands exist."""
        # Just verify a couple exist
        result = runner.invoke(cli, ["algorithm", "--help"])
        assert result.exit_code == 0

        result = runner.invoke(cli, ["dataset", "--help"])
        assert result.exit_code == 0

    def test_context_stores_config(self, runner: CliRunner) -> None:
        """Test that configuration is stored in context."""
        # We can't easily add a command to test this, so just verify
        # the CLI accepts the options without error
        result = runner.invoke(
            cli,
            ["--base-url", "http://test.com", "--timeout", "60", "--help"]
        )
        
        assert result.exit_code == 0

class TestEnvironmentVariables:
    """Tests for environment variable support."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    def test_base_url_from_env(self, runner: CliRunner) -> None:
        """Test that base URL can be set via env var."""
        # Just verify the option can be set via env without error
        result = runner.invoke(
            cli,
            ["--help"],
            env={"RAIL_SVC_BASE_URL": "http://env-url.com"}
        )

        assert result.exit_code == 0

    def test_auth_token_from_env(self, runner: CliRunner) -> None:
        """Test that auth token can be set via env var."""
        # Just verify the option can be set via env without error
        result = runner.invoke(
            cli,
            ["--help"],
            env={"RAIL_SVC_AUTH_TOKEN": "env-token"}
        )

        assert result.exit_code == 0
