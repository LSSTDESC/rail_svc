"""Unit tests for CLI entry point."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
import pytest
from click.testing import CliRunner

from rail_svc import __version__
from rail_svc.cli.local.top import TABLES, cli, init, make_table_group


# Fixtures
@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


# Test init command
class TestInitCommand:
    """Tests for database initialization command."""

    @patch("rail_svc.cli.local.top.create_async_engine")
    @patch("rail_svc.cli.local.top.config")
    @patch("rail_svc.cli.local.top.asyncio.run")
    def test_init_without_reset(
        self,
        mock_asyncio_run: MagicMock,
        mock_config: MagicMock,
        mock_create_engine: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test init command without reset flag."""
        mock_config.db.url = "sqlite+aiosqlite:///test.db"

        # Capture and close the coroutine to prevent warning
        def close_coro(coro):
            coro.close()
            return None

        mock_asyncio_run.side_effect = close_coro

        result = runner.invoke(init)

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch("rail_svc.cli.local.top.create_async_engine")
    @patch("rail_svc.cli.local.top.config")
    @patch("rail_svc.cli.local.top.asyncio.run")
    def test_init_with_reset(
        self,
        mock_asyncio_run: MagicMock,
        mock_config: MagicMock,
        mock_create_engine: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test init command with reset flag."""
        mock_config.db.url = "sqlite+aiosqlite:///test.db"

        # Capture and close the coroutine to prevent warning
        def close_coro(coro):
            coro.close()
            return None

        mock_asyncio_run.side_effect = close_coro

        result = runner.invoke(init, ["--reset"])

        assert result.exit_code == 0
        mock_asyncio_run.assert_called_once()

    @patch("rail_svc.cli.local.top.create_async_engine")
    @patch("rail_svc.cli.local.top.config")
    @patch("rail_svc.cli.local.top.asyncio.run")
    def test_init_command_name(
        self, mock_asyncio_run: MagicMock, mock_config: MagicMock, mock_create_engine: MagicMock
    ) -> None:
        """Test that init command has correct name."""

        # Capture and close the coroutine to prevent warning
        def close_coro(coro):
            coro.close()
            return None

        mock_asyncio_run.side_effect = close_coro
        assert init.name == "init"

    @patch("rail_svc.cli.local.top.create_async_engine")
    @patch("rail_svc.cli.local.top.config")
    @patch("rail_svc.cli.local.top.asyncio.run")
    def test_init_help_message(
        self,
        mock_asyncio_run: MagicMock,
        mock_config: MagicMock,
        mock_create_engine: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test init command help message."""

        # Capture and close the coroutine to prevent warning
        def close_coro(coro):
            coro.close()
            return None

        mock_asyncio_run.side_effect = close_coro
        result = runner.invoke(init, ["--help"])

        assert result.exit_code == 0
        assert "--reset" in result.output


# Test make_table_group function
class TestMakeTableGroup:
    """Tests for make_table_group function."""

    def test_creates_click_group(self) -> None:
        """Test that make_table_group creates a Click group."""
        mock_ops = MagicMock()

        group = make_table_group("test", mock_ops, "Test description")

        assert isinstance(group, click.Group)
        assert group.name == "test"

    def test_group_has_help_text(self) -> None:
        """Test that created group has help text."""
        mock_ops = MagicMock()

        group = make_table_group("test", mock_ops, "Test description")

        assert group.help == "Test description"


# Test TABLES constant
class TestTablesConstant:
    """Tests for TABLES constant."""

    def test_tables_is_list(self) -> None:
        """Test that TABLES is a list."""
        assert isinstance(TABLES, list)

    def test_tables_not_empty(self) -> None:
        """Test that TABLES is not empty."""
        assert len(TABLES) > 0

    def test_tables_structure(self) -> None:
        """Test that each table entry has correct structure."""
        for table in TABLES:
            assert isinstance(table, tuple)
            assert len(table) == 3
            assert isinstance(table[0], str)  # name
            # table[1] is operations object
            assert isinstance(table[2], str)  # description


# Test main CLI group
class TestMainCLI:
    """Tests for main CLI group."""

    def test_cli_is_group(self) -> None:
        """Test that cli is a Click group."""
        assert isinstance(cli, click.Group)

    def test_cli_name(self) -> None:
        """Test that CLI has correct name."""
        assert cli.name == "rail-svc-client"

    def test_cli_has_version_option(self, runner: CliRunner) -> None:
        """Test that CLI has version option."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output

    def test_cli_has_init_command(self, runner: CliRunner) -> None:
        """Test that CLI includes init command."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "init" in result.output


# Test CLI integration
class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_nonexistent_command(self, runner: CliRunner) -> None:
        """Test invoking non-existent command."""
        result = runner.invoke(cli, ["nonexistent"])

        assert result.exit_code != 0


# Test error handling
class TestErrorHandling:
    """Tests for error handling in CLI."""

    @patch("rail_svc.cli.local.top.create_async_engine")
    @patch("rail_svc.cli.local.top.config")
    @patch("rail_svc.cli.local.top.asyncio.run")
    def test_init_handles_connection_error(
        self,
        mock_asyncio_run: MagicMock,
        mock_config: MagicMock,
        mock_create_engine: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test init handles database connection errors."""
        mock_config.db.url = "sqlite+aiosqlite:///test.db"

        # Capture and close the coroutine to prevent warning
        def close_coro(coro):
            coro.close()
            return None
        mock_asyncio_run.side_effect = close_coro

        # Simulate connection error
        mock_asyncio_run.side_effect = RuntimeError("Connection failed")

        result = runner.invoke(init)

        # Should fail gracefully
        assert result.exit_code != 0


# Test module-level execution
class TestModuleExecution:
    """Tests for module-level execution."""

    def test_cli_is_callable(self) -> None:
        """Test that cli function is callable."""
        assert callable(cli)

    def test_init_is_callable(self) -> None:
        """Test that init function is callable."""
        assert callable(init)

    def test_make_table_group_is_callable(self) -> None:
        """Test that make_table_group is callable."""
        assert callable(make_table_group)


# Test version display
class TestVersionDisplay:
    """Tests for version display."""

    def test_version_option_long(self, runner: CliRunner) -> None:
        """Test long version option."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert __version__ in result.output


# Test CLI group structure
class TestCLIGroupStructure:
    """Tests for CLI group structure and organization."""

    def test_cli_has_subcommands(self) -> None:
        """Test that CLI has subcommands."""
        assert hasattr(cli, "commands")
        assert len(cli.commands) > 0

    def test_init_in_commands(self) -> None:
        """Test that init is in commands."""
        assert "init" in cli.commands


# Test command invocation
class TestCommandInvocation:
    """Tests for command invocation patterns."""

    @patch("rail_svc.cli.local.top.create_async_engine")
    @patch("rail_svc.cli.local.top.config")
    @patch("rail_svc.cli.local.top.asyncio.run")
    def test_init_can_be_invoked_standalone(
        self,
        mock_asyncio_run: MagicMock,
        mock_config: MagicMock,
        mock_create_engine: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test that init can be invoked as standalone command."""
        mock_config.db.url = "sqlite+aiosqlite:///test.db"
        
        # Capture and close the coroutine to prevent warning
        def close_coro(coro):
            coro.close()
            return None

        mock_asyncio_run.side_effect = close_coro        

        result = runner.invoke(cli, ["init"])

        assert result.exit_code == 0


# Test async execution
class TestAsyncExecution:
    """Tests for async execution patterns."""

    @patch("rail_svc.cli.local.top.create_async_engine")
    @patch("rail_svc.cli.local.top.config")
    @patch("rail_svc.cli.local.top.asyncio.run")
    def test_init_uses_asyncio_run(
        self,
        mock_asyncio_run: MagicMock,
        mock_config: MagicMock,
        mock_create_engine: MagicMock,
        runner: CliRunner,
    ) -> None:
        """Test that init uses asyncio.run."""
        mock_config.db.url = "sqlite+aiosqlite:///test.db"
        
        # Capture and close the coroutine to prevent warning
        def close_coro(coro):
            coro.close()
            return None

        mock_asyncio_run.side_effect = close_coro        

        runner.invoke(cli, ["init"])

        # asyncio.run should be called with async function
        assert mock_asyncio_run.called


# Test CLI completeness
class TestCLICompleteness:
    """Tests for CLI completeness and consistency."""

    def test_no_duplicate_table_names(self) -> None:
        """Test that there are no duplicate table names."""
        names = [t[0] for t in TABLES]
        assert len(names) == len(set(names))


# Test edge cases
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_args_list(self, runner: CliRunner) -> None:
        """Test CLI with empty args."""
        result = runner.invoke(cli, [])

        # Should succeed (show help or usage)
        assert result.exit_code == 0 or "Usage" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
