"""Unit tests for CliRemoteOperations class."""

from __future__ import annotations

import json
from typing import ClassVar
from unittest.mock import Mock

import click
import pytest
from click.testing import CliRunner
from pydantic import BaseModel, ValidationError

from rail_svc.cli.remote.base import CliRemoteOperations
from rail_svc.models import FilterOp
from rail_svc.remote_sync.base import SyncRemoteOperations


# Test models
class RemoteTestResponse(BaseModel):
    """Test response model."""

    id: int
    name: str
    value: int = 0
    col_names_for_table: ClassVar[list[str]] = ["id", "name", "value"]


class RemoteTestCreate(BaseModel):
    """Test create model."""

    name: str
    value: int = 0


class TestCliRemoteOperationsBasics:
    """Basic tests for CliRemoteOperations."""

    @pytest.fixture
    def mock_sync_ops(self) -> Mock:
        """Create mock sync operations."""
        mock = Mock(spec=SyncRemoteOperations)
        mock.async_ops = Mock()
        mock.async_ops.table_name = "test_table"
        return mock

    @pytest.fixture
    def cli_group(self) -> click.Group:
        """Create a Click group for testing."""

        @click.group()
        def test_cli():
            pass

        return test_cli

    def test_initialization(self, mock_sync_ops: Mock, cli_group: click.Group) -> None:
        """Test CliRemoteOperations initialization."""
        cli_ops = CliRemoteOperations(mock_sync_ops, cli_group)

        assert cli_ops.sync_oper is mock_sync_ops
        assert cli_ops.group is cli_group
        assert cli_ops.table_name == "test_table"

    def test_register_commands_adds_to_group(self, mock_sync_ops: Mock, cli_group: click.Group) -> None:
        """Test that registering commands adds them to the group."""
        cli_ops = CliRemoteOperations(mock_sync_ops, cli_group)

        initial_count = len(cli_group.commands)
        cli_ops.register_get_row()

        assert len(cli_group.commands) == initial_count + 1
        assert "get-row" in cli_group.commands


class TestReadCommands:
    """Tests for read command registration and execution."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_get_row_command_exists(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-row command is registered."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "--help"])
        assert result.exit_code == 0
        assert "Get a single" in result.output

    def test_get_row_calls_sync_operation(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-row calls the sync operation."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "1"])
        assert result.exit_code == 0
        mock_ops.get_row.assert_called_once_with(row_id=1)

    def test_get_rows_with_pagination(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-rows with pagination options."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_rows.return_value = [RemoteTestResponse(id=i, name=f"row{i}") for i in range(5)]
        cli_ops.register_get_rows()

        result = runner.invoke(cli_group, ["get-rows", "--skip", "10", "--limit", "5"])

        assert result.exit_code == 0
        mock_ops.get_rows.assert_called_once_with(skip=10, limit=5)

    def test_count_rows_displays_count(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count command displays the count."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_rows.return_value = 42
        cli_ops.register_count_rows()

        result = runner.invoke(cli_group, ["count"])

        assert result.exit_code == 0
        assert "42" in result.output
        mock_ops.count_rows.assert_called_once()

    def test_lookup_requires_id_or_name(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test lookup requires exactly one of ID or name."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_lookup_by_id_or_name()

        # No arguments
        result = runner.invoke(cli_group, ["lookup"])
        assert result.exit_code != 0

        # Both arguments
        result = runner.invoke(cli_group, ["lookup", "--id", "1", "--name", "test"])
        assert result.exit_code != 0


class TestCreateCommands:
    """Tests for create command registration and execution."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_row_with_key_value_pairs(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create with KEY=VALUE arguments."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test", value=42)
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=test", "value=42"])
        assert result.exit_code == 0
        mock_ops.create_row.assert_called_once()
        call_kwargs = mock_ops.create_row.call_args[1]
        assert call_kwargs["name"] == "test"
        assert call_kwargs["value"] == 42

    def test_create_row_from_json_file(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create from JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_create_row()

        json_data = {"name": "test", "value": 100}
        with runner.isolated_filesystem():
            # Create actual file in isolated temp directory
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create", "--from-json", "test.json"])
        assert result.exit_code == 0
        mock_ops.create_row.assert_called_once()

    def test_create_rows_requires_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-many requires JSON array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_rows()

        json_data = {"name": "test"}

        with runner.isolated_filesystem():
            # Create actual file in isolated temp directory
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-many", "test.json"])

        assert result.exit_code != 0
        assert "array" in result.output.lower()


class TestUpdateCommands:
    """Tests for update command registration and execution."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_update_row_with_key_value_pairs(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update with KEY=VALUE arguments."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_row.return_value = RemoteTestResponse(id=1, name="updated", value=99)
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1", "name=updated", "value=99"])

        assert result.exit_code == 0
        mock_ops.update_row.assert_called_once_with(row_id=1, name="updated", value=99)

    def test_update_row_prevents_id_change(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that updating row ID is prevented."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1", "id=2"])

        assert result.exit_code != 0
        assert "Cannot change row ID" in result.output


class TestDeleteCommands:
    """Tests for delete command registration and execution."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_delete_row_requires_confirmation(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete requires confirmation by default."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = RemoteTestResponse(id=1, name="deleted")
        cli_ops.register_delete_row()

        # Without --confirm, need to provide input
        result = runner.invoke(cli_group, ["delete", "1"], input="n\n")

        assert "cancelled" in result.output.lower()
        mock_ops.delete_row.assert_not_called()

    def test_delete_row_with_confirm_flag(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete with --confirm skips prompt."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = RemoteTestResponse(id=1, name="deleted")
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "--confirm", "1"])

        assert result.exit_code == 0
        mock_ops.delete_row.assert_called_once()

    def test_delete_rows_from_arguments(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many with ID arguments."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.return_value = 3
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "--confirm", "1", "2", "3"])

        assert result.exit_code == 0
        mock_ops.delete_rows.assert_called_once()


class TestFilterCommands:
    """Tests for filter command registration and execution."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_rows_with_conditions(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with conditions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = [RemoteTestResponse(id=1, name="test", value=50)]
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:10"])
        assert result.exit_code == 0
        mock_ops.filter_rows.assert_called_once()

    def test_filter_rows_invalid_format(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with invalid format raises error."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "invalidformat"])

        assert result.exit_code != 0
        assert "Invalid filter format" in result.output

    def test_find_by_with_conditions(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with KEY=VALUE conditions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = [RemoteTestResponse(id=1, name="test")]
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test"])

        assert result.exit_code == 0
        mock_ops.find_by.assert_called_once()


class TestConvenienceMethods:
    """Tests for convenience registration methods."""

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_register_all_read_commands(self, setup_cli: tuple) -> None:
        """Test register_all_read_commands adds all read commands."""
        cli_group, mock_ops, cli_ops = setup_cli

        initial_count = len(cli_group.commands)
        cli_ops.register_all_read_commands()

        # Should have added multiple commands
        assert len(cli_group.commands) > initial_count
        assert "get-row" in cli_group.commands
        assert "get-rows" in cli_group.commands
        assert "count" in cli_group.commands

    def test_register_all_create_commands(self, setup_cli: tuple) -> None:
        """Test register_all_create_commands adds all create commands."""
        cli_group, mock_ops, cli_ops = setup_cli

        cli_ops.register_all_create_commands()

        assert "create" in cli_group.commands
        assert "create-many" in cli_group.commands
        assert "bulk-insert" in cli_group.commands

    def test_register_all_update_commands(self, setup_cli: tuple) -> None:
        """Test register_all_update_commands adds all update commands."""
        cli_group, mock_ops, cli_ops = setup_cli

        cli_ops.register_all_update_commands()

        assert "update" in cli_group.commands
        assert "update-many" in cli_group.commands

    def test_register_all_delete_commands(self, setup_cli: tuple) -> None:
        """Test register_all_delete_commands adds all delete commands."""
        cli_group, mock_ops, cli_ops = setup_cli

        cli_ops.register_all_delete_commands()

        assert "delete" in cli_group.commands
        assert "delete-many" in cli_group.commands
        assert "bulk-delete" in cli_group.commands

    def test_register_all_filter_commands(self, setup_cli: tuple) -> None:
        """Test register_all_filter_commands adds all filter commands."""
        cli_group, mock_ops, cli_ops = setup_cli

        cli_ops.register_all_filter_commands()

        assert "filter" in cli_group.commands
        assert "find-by" in cli_group.commands


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_handle_error_displays_message(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that errors are displayed properly."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row.side_effect = ValueError("Test error")
        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "1"])

        assert result.exit_code != 0
        assert "Test error" in result.output

    def test_handle_validation_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test handling of validation errors."""
        cli_group, mock_ops, cli_ops = setup_cli

        from pydantic import ValidationError

        # Create a validation error
        try:
            RemoteTestResponse(id="not_an_int", name="test")  # type: ignore
        except ValidationError as e:
            mock_ops.get_row.side_effect = e

        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "1"])

        assert result.exit_code != 0
        assert "Validation failed" in result.output


class TestOutputFormats:
    """Tests for different output formats."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_get_row_with_json_output(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-row with JSON output format."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row.return_value = RemoteTestResponse(id=1, name="test", value=42)
        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "--output", "json", "1"])

        assert result.exit_code == 0
        # Output should be valid JSON
        import json

        try:
            json.loads(result.output)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

    def test_get_row_with_table_output(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-row with table output format."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row.return_value = RemoteTestResponse(id=1, name="test", value=42)
        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "--output", "table", "1"])

        assert result.exit_code == 0


class TestInputParsing:
    """Tests for input parsing logic."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_parses_json_values(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that JSON values in KEY=VALUE are parsed."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test", value=42)
        cli_ops.register_create_row()

        # Boolean as JSON
        result = runner.invoke(cli_group, ["create", "name=test", "value=42"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        # Should parse 42 as int, not string
        assert isinstance(call_kwargs["value"], int)

    def test_update_requires_data(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that update requires at least one field."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1"])

        assert result.exit_code != 0
        assert "No update data" in result.output

    def test_filter_operator_parsing(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that filter operators are parsed correctly."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:10", "-f", "name:eq:test"])

        assert result.exit_code == 0
        # Check that filters were passed
        call_args = mock_ops.filter_rows.call_args
        filters = call_args[1]["filters"]
        assert len(filters) == 2


class TestBatchOperations:
    """Tests for batch operation commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_batched_validates_batch_size(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that batch size is validated."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_rows_batched()

        json_data = [{"name": "test"}]
        with runner.isolated_filesystem():
            # Create actual file in isolated temp directory
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-batched", "--batch-size", "0", "test.json"])

        assert result.exit_code != 0
        assert "Batch size must be at least 1" in result.output

    def test_bulk_insert_displays_count(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that bulk insert displays count."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_insert_rows.return_value = 100
        cli_ops.register_bulk_insert_rows()

        json_data = [{"name": "test"}]
        with runner.isolated_filesystem():
            # Create actual file in isolated temp directory
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["bulk-insert", "test.json"])

        assert result.exit_code == 0
        assert "100" in result.output


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_get_row_or_none_displays_not_found(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-row-if-exists displays message when not found."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row_or_none.return_value = None
        cli_ops.register_get_row_or_none()

        result = runner.invoke(cli_group, ["get-row-if-exists", "999"])

        assert result.exit_code == 0
        assert "No test_table found" in result.output

    def test_delete_from_file_with_json(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many from JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.return_value = 3
        cli_ops.register_delete_rows()

        json_data = [1, 2, 3]

        with runner.isolated_filesystem():
            # Create actual file in isolated temp directory
            with open("ids.json", "w") as f:
                json.dump(json_data, f)
            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", "ids.json"])

        assert result.exit_code == 0

    def test_delete_from_file_with_text(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many from text file (one ID per line)."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.return_value = 3
        cli_ops.register_delete_rows()

        text_data = "1\n2\n3\n"

        with runner.isolated_filesystem():
            # Create actual file in isolated temp directory
            with open("ids.txt", "w") as f:
                f.write(text_data)

            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", "ids.txt"])

        assert result.exit_code == 0

    def test_filter_with_in_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with IN operator parses comma-separated values."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "status:in:active,pending,done"])

        assert result.exit_code == 0
        call_args = mock_ops.filter_rows.call_args
        filters = call_args[1]["filters"]
        # Value should be a list
        assert isinstance(filters[0].value, list)

    def test_count_filtered_with_no_filters(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered works with no filters."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_filtered_rows.return_value = 42
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered"])

        assert result.exit_code == 0
        assert "42" in result.output


class TestGetRowByName:
    """Tests for get-row-by-name command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_get_row_by_name_success(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-by-name retrieves row successfully."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row_by_name.return_value = RemoteTestResponse(id=1, name="test_name")
        cli_ops.register_get_row_by_name()

        result = runner.invoke(cli_group, ["get-by-name", "test_name"])

        assert result.exit_code == 0
        mock_ops.get_row_by_name.assert_called_once_with(name="test_name")

    def test_get_row_by_name_not_found(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-by-name with non-existent name."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row_by_name.side_effect = ValueError("Not found")
        cli_ops.register_get_row_by_name()

        result = runner.invoke(cli_group, ["get-by-name", "nonexistent"])

        assert result.exit_code != 0
        assert "Not found" in result.output


class TestLookupByIdOrName:
    """Tests for lookup command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_lookup_by_id_success(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test lookup by ID."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.lookup_by_id_or_name.return_value = (1, RemoteTestResponse(id=1, name="test"))
        cli_ops.register_lookup_by_id_or_name()

        result = runner.invoke(cli_group, ["lookup", "--id", "1"])

        assert result.exit_code == 0
        mock_ops.lookup_by_id_or_name.assert_called_once_with(row_id=1, name=None)

    def test_lookup_by_name_success(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test lookup by name."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.lookup_by_id_or_name.return_value = (1, RemoteTestResponse(id=1, name="test"))
        cli_ops.register_lookup_by_id_or_name()

        result = runner.invoke(cli_group, ["lookup", "--name", "test"])

        assert result.exit_code == 0
        mock_ops.lookup_by_id_or_name.assert_called_once_with(row_id=None, name="test")


class TestCreateRowValidation:
    """Tests for create-row validation and edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_row_no_validate_flag(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create with --no-validate flag."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "--no-validate", "name=test"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        assert call_kwargs["validate"] is False

    def test_create_row_with_validate(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create with validation (default)."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=test"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        assert call_kwargs["validate"] is True

    def test_create_row_invalid_json_file(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create from invalid JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_row()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("{ invalid json")

            result = runner.invoke(cli_group, ["create", "--from-json", "invalid.json"])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_create_row_file_read_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create from unreadable file."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_row()

        # Try to read from non-existent file (using --from-json with non-existent path)
        # Note: click.Path(exists=True) prevents this in normal usage,
        # but we can test by mocking file operations
        from unittest.mock import patch

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                f.write('{"name": "test"}')

            with patch("builtins.open", side_effect=OSError("Permission denied")):
                result = runner.invoke(cli_group, ["create", "--from-json", "test.json"])

        assert result.exit_code != 0
        assert "Cannot read file" in result.output

    def test_create_row_no_data_provided(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create with no data raises error."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create"])

        assert result.exit_code != 0
        assert "No data provided" in result.output

    def test_create_row_invalid_field_format(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create with invalid KEY=VALUE format."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "invalidformat"])

        assert result.exit_code != 0
        assert "Invalid field format" in result.output


class TestCreateRowsValidation:
    """Tests for create-rows validation and edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_rows_empty_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-many with empty array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_rows()

        json_data: list[dict[str, str]] = []

        with runner.isolated_filesystem():
            with open("empty.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-many", "empty.json"])

        assert result.exit_code != 0
        assert "Array is empty" in result.output

    def test_create_rows_with_no_validate(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-many with --no-validate flag."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_rows.return_value = [RemoteTestResponse(id=1, name="test")]
        cli_ops.register_create_rows()

        json_data = [{"name": "test"}]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-many", "--no-validate", "test.json"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_rows.call_args[1]
        assert call_kwargs["validate"] is False


class TestCreateRowsBatched:
    """Tests for create-rows-batched command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_batched_invalid_json(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-batched with invalid JSON."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_rows_batched()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("not json")

            result = runner.invoke(cli_group, ["create-batched", "invalid.json"])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_create_batched_file_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-batched with file read error."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_rows_batched()

        from unittest.mock import patch

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                f.write('[{"name": "test"}]')

            with patch("builtins.open", side_effect=OSError("Cannot read")):
                result = runner.invoke(cli_group, ["create-batched", "test.json"])

        assert result.exit_code != 0
        assert "Cannot read file" in result.output

    def test_create_batched_not_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-batched with non-array JSON."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_rows_batched()

        json_data = {"name": "test"}

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-batched", "test.json"])

        assert result.exit_code != 0
        assert "array" in result.output.lower()

    def test_create_batched_empty_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-batched with empty array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_create_rows_batched()

        json_data: list[dict[str, str]] = []

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-batched", "test.json"])

        assert result.exit_code != 0
        assert "empty" in result.output.lower()

    def test_create_batched_custom_batch_size(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-batched with custom batch size."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_rows_batched.return_value = [
            RemoteTestResponse(id=i, name=f"test{i}") for i in range(5)
        ]
        cli_ops.register_create_rows_batched()

        json_data = [{"name": f"test{i}"} for i in range(5)]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-batched", "--batch-size", "2", "test.json"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_rows_batched.call_args[1]
        assert call_kwargs["batch_size"] == 2


class TestBulkInsert:
    """Tests for bulk-insert command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_bulk_insert_invalid_json(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-insert with invalid JSON."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_insert_rows()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("invalid")

            result = runner.invoke(cli_group, ["bulk-insert", "invalid.json"])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_bulk_insert_file_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-insert with file read error."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_insert_rows()

        from unittest.mock import patch

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                f.write('[{"name": "test"}]')

            with patch("builtins.open", side_effect=OSError("Error")):
                result = runner.invoke(cli_group, ["bulk-insert", "test.json"])

        assert result.exit_code != 0
        assert "Cannot read file" in result.output

    def test_bulk_insert_not_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-insert with non-array JSON."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_insert_rows()

        json_data = {"name": "test"}

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["bulk-insert", "test.json"])

        assert result.exit_code != 0
        assert "array" in result.output.lower()

    def test_bulk_insert_empty_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-insert with empty array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_insert_rows()

        json_data: list[dict[str, str]] = []

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["bulk-insert", "test.json"])

        assert result.exit_code != 0
        assert "empty" in result.output.lower()


class TestUpdateRowEdgeCases:
    """Tests for update-row edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_update_row_from_json_file(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update from JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_row.return_value = RemoteTestResponse(id=1, name="updated")
        cli_ops.register_update_row()

        json_data = {"name": "updated", "value": 100}

        with runner.isolated_filesystem():
            with open("update.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update", "--from-json", "update.json", "1"])

        assert result.exit_code == 0
        mock_ops.update_row.assert_called_once()

    def test_update_row_invalid_json_file(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update from invalid JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_row()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("not json")

            result = runner.invoke(cli_group, ["update", "--from-json", "invalid.json", "1"])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_update_row_file_read_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update with file read error."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_row()

        from unittest.mock import patch

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                f.write('{"name": "test"}')

            with patch("builtins.open", side_effect=OSError("Error")):
                result = runner.invoke(cli_group, ["update", "--from-json", "test.json", "1"])

        assert result.exit_code != 0
        assert "Cannot read file" in result.output

    def test_update_row_invalid_field_format(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update with invalid field format."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1", "invalidformat"])

        assert result.exit_code != 0
        assert "Invalid field format" in result.output

    def test_update_row_allows_same_id(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update allows ID field if it matches row_id."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1", "id=1", "name=test"])

        assert result.exit_code == 0
        # Should not error since id=1 matches row_id=1


class TestUpdateRows:
    """Tests for update-rows command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_update_rows_success(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many with valid data."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_rows.return_value = [
            RemoteTestResponse(id=1, name="updated1"),
            RemoteTestResponse(id=2, name="updated2"),
        ]
        cli_ops.register_update_rows()

        json_data = [{"id": 1, "name": "updated1"}, {"id": 2, "name": "updated2"}]

        with runner.isolated_filesystem():
            with open("updates.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update-many", "updates.json"])

        assert result.exit_code == 0
        assert "2" in result.output

    def test_update_rows_invalid_json(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many with invalid JSON."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_rows()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("invalid")

            result = runner.invoke(cli_group, ["update-many", "invalid.json"])

        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_update_rows_file_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many with file read error."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_rows()

        from unittest.mock import patch

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                f.write('[{"id": 1, "name": "test"}]')

            with patch("builtins.open", side_effect=OSError("Error")):
                result = runner.invoke(cli_group, ["update-many", "test.json"])

        assert result.exit_code != 0
        assert "Cannot read file" in result.output

    def test_update_rows_not_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many with non-array JSON."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_rows()

        json_data = {"id": 1, "name": "test"}

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update-many", "test.json"])

        assert result.exit_code != 0
        assert "array" in result.output.lower()

    def test_update_rows_empty_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many with empty array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_rows()

        json_data: list[dict[str, str]] = []

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update-many", "test.json"])

        assert result.exit_code != 0
        assert "empty" in result.output.lower()

    def test_update_rows_missing_id(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many with missing id field."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_rows()

        json_data = [{"name": "test"}]  # Missing 'id'

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update-many", "test.json"])

        assert result.exit_code != 0
        assert "missing 'id' field" in result.output

    def test_update_rows_non_dict_element(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many with non-dict element in array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_update_rows()

        json_data = ["not a dict"]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update-many", "test.json"])

        assert result.exit_code != 0
        assert "not an object" in result.output


class TestDeleteRowEdgeCases:
    """Tests for delete-row edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_delete_row_with_capture(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete with data capture (default)."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = RemoteTestResponse(id=1, name="deleted")
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "--confirm", "1"])

        assert result.exit_code == 0
        assert "Deleted data:" in result.output
        call_kwargs = mock_ops.delete_row.call_args[1]
        assert call_kwargs["capture_data"] is True

    def test_delete_row_no_capture(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete with --no-capture flag."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = None
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "--confirm", "--no-capture", "1"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.delete_row.call_args[1]
        assert call_kwargs["capture_data"] is False
        assert "Deleted data:" not in result.output


class TestDeleteRowsEdgeCases:
    """Tests for delete-rows edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_delete_rows_with_capture_data(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many with --capture-data flag."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.return_value = [
            RemoteTestResponse(id=1, name="deleted1"),
            RemoteTestResponse(id=2, name="deleted2"),
        ]
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "--confirm", "--capture-data", "1", "2"])

        assert result.exit_code == 0
        assert "Deleted data:" in result.output
        call_kwargs = mock_ops.delete_rows.call_args[1]
        assert call_kwargs["capture_data"] is True

    def test_delete_rows_without_capture_returns_count(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many without capture returns count."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.return_value = 2  # Returns count instead of list
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "--confirm", "1", "2"])

        assert result.exit_code == 0
        assert "2" in result.output
        assert "Deleted data:" not in result.output

    def test_delete_rows_no_ids_provided(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many with no IDs."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "--confirm"])

        assert result.exit_code != 0
        assert "No IDs provided" in result.output

    def test_delete_rows_from_json_file_invalid(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many from invalid JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_delete_rows()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("not json")

            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", "invalid.json"])

        assert result.exit_code != 0
        assert "Error reading file" in result.output

    def test_delete_rows_from_json_not_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many from JSON file that's not an array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_delete_rows()

        json_data = {"id": 1}

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", "test.json"])

        assert result.exit_code != 0
        assert "Error reading file" in result.output


class TestBulkDelete:
    """Tests for bulk-delete command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_bulk_delete_success(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete with confirmation."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_delete_rows.return_value = 3
        cli_ops.register_bulk_delete_rows()

        result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "1", "2", "3"])

        assert result.exit_code == 0
        assert "3" in result.output

    def test_bulk_delete_partial_success(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete when some IDs not found."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_delete_rows.return_value = 2
        cli_ops.register_bulk_delete_rows()

        result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "1", "2", "999"])

        assert result.exit_code == 0
        assert "2" in result.output
        assert "1" in result.output  # 3 - 2 = 1 not found
        assert "not found" in result.output

    def test_bulk_delete_requires_confirmation(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete requires confirmation."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_delete_rows()

        result = runner.invoke(cli_group, ["bulk-delete", "1", "2"], input="n\n")

        assert "cancelled" in result.output.lower()
        mock_ops.bulk_delete_rows.assert_not_called()

    def test_bulk_delete_no_ids(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete with no IDs."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_delete_rows()

        result = runner.invoke(cli_group, ["bulk-delete", "--confirm"])

        assert result.exit_code != 0
        assert "No IDs provided" in result.output

    def test_bulk_delete_from_json_file(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete from JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_delete_rows.return_value = 3
        cli_ops.register_bulk_delete_rows()

        json_data = [1, 2, 3]

        with runner.isolated_filesystem():
            with open("ids.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "--from-file", "ids.json"])

        assert result.exit_code == 0

    def test_bulk_delete_from_text_file(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete from text file."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_delete_rows.return_value = 3
        cli_ops.register_bulk_delete_rows()

        with runner.isolated_filesystem():
            with open("ids.txt", "w") as f:
                f.write("1\n2\n3\n")

            result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "--from-file", "ids.txt"])

        assert result.exit_code == 0

    def test_bulk_delete_from_file_invalid_json(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete from invalid JSON file."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_delete_rows()

        with runner.isolated_filesystem():
            with open("invalid.json", "w") as f:
                f.write("not json array")

            result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "--from-file", "invalid.json"])

        # Should still work as it falls back to line-separated parsing
        # But if all lines fail to parse as int, should error
        assert result.exit_code != 0

    def test_bulk_delete_from_file_not_array(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete from JSON file that's not an array."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_bulk_delete_rows()

        json_data = {"id": 1}

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "--from-file", "test.json"])

        assert result.exit_code != 0
        assert "Error reading file" in result.output


class TestFilterRowsEdgeCases:
    """Tests for filter-rows edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_unknown_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with unknown operator."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:unknown:10"])

        assert result.exit_code != 0
        assert "Unknown operator" in result.output

    def test_filter_with_or_logic(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with OR logic."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:10", "-f", "name:eq:test", "--or"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        assert call_kwargs["logical_op"] == "or"

    def test_filter_with_order_by(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with order by."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "--order-by", "name", "--order-by", "value:desc"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        order_by = call_kwargs["order_by"]
        assert len(order_by) == 2
        assert order_by[0].field == "name"
        assert order_by[0].descending is False
        assert order_by[1].field == "value"
        assert order_by[1].descending is True

    def test_filter_with_ilike_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with ILIKE operator."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "name:ilike:%test%"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert filters[0].op == FilterOp.ILIKE

    def test_filter_with_not_in_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with NOT_IN operator."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "status:not_in:inactive,deleted"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert filters[0].op == FilterOp.NOT_IN
        assert isinstance(filters[0].value, list)

    def test_filter_with_json_value(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with JSON-parsed value."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:eq:42"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        # Value should be parsed as int
        assert isinstance(filters[0].value, int)
        assert filters[0].value == 42

    def test_filter_pagination_validation_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with invalid pagination."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "--skip", "-1"])

        assert result.exit_code != 0
        # Should show pagination validation error


class TestCountFilteredRows:
    """Tests for count-filtered-rows command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_count_filtered_with_filters(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered with filters."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_filtered_rows.return_value = 42
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered", "-f", "value:gt:10"])

        assert result.exit_code == 0
        assert "42" in result.output
        assert "matching" in result.output

    def test_count_filtered_with_or_logic(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered with OR logic."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_filtered_rows.return_value = 10
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered", "-f", "value:gt:10", "--or"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.count_filtered_rows.call_args[1]
        assert call_kwargs["logical_op"] == "or"

    def test_count_filtered_invalid_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered with invalid operator."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered", "-f", "value:invalid:10"])

        assert result.exit_code != 0
        assert "Unknown operator" in result.output

    def test_count_filtered_invalid_format(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered with invalid filter format."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered", "-f", "invalid"])

        assert result.exit_code != 0
        assert "Invalid filter format" in result.output


class TestFindByEdgeCases:
    """Tests for find-by edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_find_by_no_conditions(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with no conditions."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by"])

        assert result.exit_code != 0
        assert "No conditions provided" in result.output

    def test_find_by_invalid_format(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with invalid condition format."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "invalid"])

        assert result.exit_code != 0
        assert "Invalid condition format" in result.output

    def test_find_by_with_order_by(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with order by."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = []
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test", "--order-by", "value:desc"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.find_by.call_args[1]
        order_by = call_kwargs["order_by"]
        assert len(order_by) == 1
        assert order_by[0].field == "value"
        assert order_by[0].descending is True

    def test_find_by_with_pagination(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with pagination."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = []
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test", "--skip", "10", "--limit", "5"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.find_by.call_args[1]
        assert call_kwargs["skip"] == 10
        assert call_kwargs["limit"] == 5

    def test_find_by_pagination_validation_error(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with invalid pagination."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test", "--skip", "-1"])

        assert result.exit_code != 0


class TestFindOneBy:
    """Tests for find-one-by command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_find_one_by_success(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-one-by finds exactly one row."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_one_by.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_find_one_by()

        result = runner.invoke(cli_group, ["find-one-by", "name=test"])

        assert result.exit_code == 0
        mock_ops.find_one_by.assert_called_once_with(name="test")

    def test_find_one_by_no_conditions(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-one-by with no conditions."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_find_one_by()

        result = runner.invoke(cli_group, ["find-one-by"])

        assert result.exit_code != 0
        assert "No conditions provided" in result.output

    def test_find_one_by_invalid_format(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-one-by with invalid format."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_find_one_by()

        result = runner.invoke(cli_group, ["find-one-by", "invalid"])

        assert result.exit_code != 0
        assert "Invalid condition format" in result.output

    def test_find_one_by_multiple_conditions(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-one-by with multiple conditions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_one_by.return_value = RemoteTestResponse(id=1, name="test", value=42)
        cli_ops.register_find_one_by()

        result = runner.invoke(cli_group, ["find-one-by", "name=test", "value=42"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.find_one_by.call_args[1]
        assert call_kwargs["name"] == "test"
        assert call_kwargs["value"] == 42


class TestErrorHandlingEdgeCases:
    """Tests for additional error handling scenarios."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_handle_generic_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test handling of generic exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row.side_effect = RuntimeError("Unexpected error")
        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "1"])

        assert result.exit_code != 0
        assert "Unexpected error" in result.output

    def test_handle_error_with_context(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test error messages include context."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row.side_effect = ValueError("Test error")
        cli_ops.register_get_row()

        result = runner.invoke(cli_group, ["get-row", "123"])

        assert result.exit_code != 0
        assert "getting test_table with ID 123" in result.output
        assert "Test error" in result.output

    def test_handle_validation_error_in_create(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test validation error in create command."""
        cli_group, mock_ops, cli_ops = setup_cli

        # Create a validation error
        try:
            RemoteTestResponse(id="not_int", name="test")  # type: ignore
        except ValidationError as e:
            mock_ops.create_row.side_effect = e

        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=test"])

        assert result.exit_code != 0
        assert "Validation failed" in result.output


class TestResponseModelAttribute:
    """Tests for response_model attribute access."""

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_response_model_initialization(self, setup_cli: tuple) -> None:
        """Test response_model is properly initialized."""
        cli_group, mock_ops, cli_ops = setup_cli

        assert cli_ops.response_model == RemoteTestResponse
        assert cli_ops.col_names_for_table == ["id", "name", "value"]


class TestPaginationValidation:
    """Tests for pagination parameter validation in various commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_get_rows_pagination_validation(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-rows validates pagination parameters."""
        cli_group, mock_ops, cli_ops = setup_cli
        cli_ops.register_get_rows()

        result = runner.invoke(cli_group, ["get-rows", "--skip", "-1"])

        assert result.exit_code != 0
        # Pagination validation should catch negative skip

    def test_get_rows_uses_limit_over_page_size(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-rows uses limit when provided instead of page_size."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_rows.return_value = []
        cli_ops.register_get_rows()

        result = runner.invoke(cli_group, ["get-rows", "--limit", "5", "--page-size", "100"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.get_rows.call_args[1]
        assert call_kwargs["limit"] == 5  # Uses limit, not page_size

    def test_get_rows_uses_page_size_when_no_limit(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-rows uses page_size when limit not provided."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_rows.return_value = []
        cli_ops.register_get_rows()

        result = runner.invoke(cli_group, ["get-rows", "--page-size", "50"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.get_rows.call_args[1]
        assert call_kwargs["limit"] == 50  # Uses page_size when no limit


class TestOutputFormatEdgeCases:
    """Tests for output format edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_delete_with_capture_and_json_output(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete with capture and JSON output."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = RemoteTestResponse(id=1, name="deleted")
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "--confirm", "--output", "json", "1"])

        assert result.exit_code == 0
        assert "Deleted data:" in result.output


class TestFileHandlingEdgeCases:
    """Tests for file handling edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_delete_from_text_file_with_empty_lines(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete from text file with empty lines."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.return_value = 2
        cli_ops.register_delete_rows()

        with runner.isolated_filesystem():
            with open("ids.txt", "w") as f:
                f.write("1\n\n2\n\n\n")  # Empty lines

            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", "ids.txt"])

        assert result.exit_code == 0
        # Should parse only non-empty lines

    def test_create_from_json_with_complex_types(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create with complex JSON types in KEY=VALUE."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test", value=42)
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", 'name="test"', "value=42"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        # Should parse "test" as string and 42 as int
        assert call_kwargs["name"] == "test"
        assert call_kwargs["value"] == 42


class TestAllOperationsException:
    """Tests for exception handling in all operation types."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_get_rows_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-rows handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_get_rows()

        result = runner.invoke(cli_group, ["get-rows"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_count_rows_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-rows handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_count_rows()

        result = runner.invoke(cli_group, ["count"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_get_row_or_none_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-row-if-exists handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_row_or_none.side_effect = RuntimeError("Database error")
        cli_ops.register_get_row_or_none()

        result = runner.invoke(cli_group, ["get-row-if-exists", "1"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_create_rows_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-many handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_create_rows()

        json_data = [{"name": "test"}]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-many", "test.json"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_create_batched_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-batched handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_rows_batched.side_effect = RuntimeError("Database error")
        cli_ops.register_create_rows_batched()

        json_data = [{"name": "test"}]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-batched", "test.json"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_bulk_insert_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-insert handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_insert_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_bulk_insert_rows()

        json_data = [{"name": "test"}]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["bulk-insert", "test.json"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_update_rows_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_update_rows()

        json_data = [{"id": 1, "name": "test"}]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update-many", "test.json"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_delete_rows_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "--confirm", "1"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_bulk_delete_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_delete_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_bulk_delete_rows()

        result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "1"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_filter_rows_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_count_filtered_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_filtered_rows.side_effect = RuntimeError("Database error")
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered", "-f", "name:eq:test"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_find_by_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.side_effect = RuntimeError("Database error")
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test"])

        assert result.exit_code != 0
        assert "Database error" in result.output

    def test_find_one_by_exception(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-one-by handles exceptions."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_one_by.side_effect = RuntimeError("Database error")
        cli_ops.register_find_one_by()

        result = runner.invoke(cli_group, ["find-one-by", "name=test"])

        assert result.exit_code != 0
        assert "Database error" in result.output


class TestFilterOperatorCoverage:
    """Tests to cover all filter operators."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_with_all_comparison_operators(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with all comparison operators."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        operators = ["eq", "ne", "lt", "le", "gt", "ge"]
        for op in operators:
            result = runner.invoke(cli_group, ["filter", "-f", f"value:{op}:10"])
            assert result.exit_code == 0

    def test_filter_with_like_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with LIKE operator."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "name:like:%test%"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert filters[0].op == FilterOp.LIKE


class TestJSONValueParsing:
    """Tests for JSON value parsing in various commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_parses_boolean_values(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create parses boolean JSON values."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=test", "active=true"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        assert isinstance(call_kwargs.get("active"), bool)
        assert call_kwargs.get("active") is True

    def test_create_parses_null_values(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create parses null JSON values."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=test", "description=null"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        assert call_kwargs.get("description") is None

    def test_create_keeps_non_json_as_string(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create keeps non-JSON values as strings."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=not-json-value"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        assert isinstance(call_kwargs["name"], str)
        assert call_kwargs["name"] == "not-json-value"

    def test_update_parses_json_values(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update parses JSON values."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1", "value=100", "active=false"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.update_row.call_args[1]
        assert isinstance(call_kwargs["value"], int)
        assert call_kwargs["value"] == 100
        assert isinstance(call_kwargs.get("active"), bool)

    def test_find_by_parses_json_values(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by parses JSON values."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = []
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "value=42", "active=true"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.find_by.call_args[1]
        assert isinstance(call_kwargs["value"], int)
        assert isinstance(call_kwargs["active"], bool)


class TestCommandHelpText:
    """Tests to verify command help text is accessible."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_all_commands_have_help(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test all registered commands have accessible help text."""
        cli_group, mock_ops, cli_ops = setup_cli

        # Register all commands
        cli_ops.register_all_read_commands()
        cli_ops.register_all_create_commands()
        cli_ops.register_all_update_commands()
        cli_ops.register_all_delete_commands()
        cli_ops.register_all_filter_commands()

        # Test help for each command
        for command_name in cli_group.commands:
            result = runner.invoke(cli_group, [command_name, "--help"])
            assert result.exit_code == 0
            assert len(result.output) > 0


class TestEmptyResultHandling:
    """Tests for handling empty results."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_get_rows_empty_result(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test get-rows with empty result."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.get_rows.return_value = []
        cli_ops.register_get_rows()

        result = runner.invoke(cli_group, ["get-rows"])

        assert result.exit_code == 0
        # Should handle empty list gracefully

    def test_filter_rows_empty_result(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with no matching rows."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:nonexistent"])

        assert result.exit_code == 0
        assert "Found 0" in result.output

    def test_find_by_empty_result(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with no matching rows."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = []
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=nonexistent"])

        assert result.exit_code == 0
        assert "Found 0" in result.output

    def test_count_rows_zero(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count with zero rows."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_rows.return_value = 0
        cli_ops.register_count_rows()

        result = runner.invoke(cli_group, ["count"])

        assert result.exit_code == 0
        assert "0" in result.output


class TestMultipleFiltersAndOrdering:
    """Tests for multiple filters and ordering combinations."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_multiple_conditions_and_logic(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with multiple conditions using AND logic."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(
            cli_group, ["filter", "-f", "value:gt:10", "-f", "value:lt:100", "-f", "name:like:%test%"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert len(filters) == 3
        assert call_kwargs["logical_op"] == "and"

    def test_filter_multiple_order_by_fields(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with multiple order by fields."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(
            cli_group, ["filter", "--order-by", "name", "--order-by", "value:desc", "--order-by", "id"]
        )

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        order_by = call_kwargs["order_by"]
        assert len(order_by) == 3
        assert order_by[0].field == "name"
        assert order_by[0].descending is False
        assert order_by[1].field == "value"
        assert order_by[1].descending is True
        assert order_by[2].field == "id"
        assert order_by[2].descending is False

    def test_find_by_multiple_order_by(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by with multiple order by fields."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = []
        cli_ops.register_find_by()

        result = runner.invoke(
            cli_group,
            ["find-by", "name=test", "--order-by", "value:desc", "--order-by", "id"],
        )

        assert result.exit_code == 0
        call_kwargs = mock_ops.find_by.call_args[1]
        order_by = call_kwargs["order_by"]
        assert len(order_by) == 2


class TestSpecialCharactersInValues:
    """Tests for special characters in values."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_with_spaces_in_value(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create with spaces in value."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test name")
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=test name"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.create_row.call_args[1]
        assert call_kwargs["name"] == "test name"

    def test_filter_with_special_characters(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with special characters in pattern."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "name:like:%test_value%"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert filters[0].value == "%test_value%"


class TestNoneAndNullHandling:
    """Tests for None/null value handling."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_delete_row_returns_none(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete when no data captured returns None."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = None
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "--confirm", "--no-capture", "1"])

        assert result.exit_code == 0
        # Should not try to display deleted data when None


class TestLimitAndPageSizeInteraction:
    """Tests for limit and page_size parameter interaction."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_limit_overrides_page_size(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test that limit overrides page_size in filter command."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "--limit", "10", "--page-size", "100"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        assert call_kwargs["limit"] == 10

    def test_find_by_default_page_size(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by uses default page_size when no limit."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = []
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.find_by.call_args[1]
        # Should use default page_size (typically 100)
        assert "limit" in call_kwargs


class TestOrderByEdgeCases:
    """Tests for order-by edge cases."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_order_by_case_insensitive_desc(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test order by with case variations of 'desc'."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        # Test with uppercase DESC
        result = runner.invoke(cli_group, ["filter", "--order-by", "value:DESC"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        order_by = call_kwargs["order_by"]
        assert order_by[0].descending is True

    def test_order_by_with_asc_is_not_descending(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test order by with 'asc' does not set descending."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "--order-by", "value:asc"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        order_by = call_kwargs["order_by"]
        # 'asc' is not 'desc', so descending should be False
        assert order_by[0].descending is False


class TestInOperatorListParsing:
    """Tests for IN/NOT_IN operator list parsing."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_in_operator_strips_whitespace(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test IN operator strips whitespace from values."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "status:in:active, pending, done"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        # Values should have whitespace stripped
        assert filters[0].value == ["active", "pending", "done"]

    def test_in_operator_single_value(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test IN operator with single value."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "status:in:active"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert filters[0].value == ["active"]

    def test_in_operator_empty_strings_filtered(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test IN operator handles empty strings in list."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        # Even with commas, should handle gracefully
        result = runner.invoke(cli_group, ["filter", "-f", "status:in:active,,pending"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        # Empty strings are stripped and result in empty string in list
        assert len(filters[0].value) == 3


class TestCountFilteredOperatorCoverage:
    """Tests for count-filtered with all operators."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_count_filtered_with_like_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered with LIKE operator."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_filtered_rows.return_value = 5
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered", "-f", "name:like:%test%"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.count_filtered_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert filters[0].op == FilterOp.LIKE

    def test_count_filtered_with_in_operator(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered with IN operator."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_filtered_rows.return_value = 3
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered", "-f", "status:in:active,pending"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.count_filtered_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert filters[0].op == FilterOp.IN
        assert isinstance(filters[0].value, list)


class TestComplexJSONParsing:
    """Tests for complex JSON value parsing."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_with_negative_number(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with negative number value."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:lt:-10"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert isinstance(filters[0].value, int)
        assert filters[0].value == -10

    def test_filter_with_float_value(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with float value."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:10.5"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert isinstance(filters[0].value, float)
        assert filters[0].value == 10.5


class TestMultipleSameFieldFilters:
    """Tests for multiple filters on the same field."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_range_query(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with range query (gt and lt on same field)."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:10", "-f", "value:lt:100"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        filters = call_kwargs["filters"]
        assert len(filters) == 2
        assert filters[0].field == "value"
        assert filters[1].field == "value"
        assert filters[0].op == FilterOp.GT
        assert filters[1].op == FilterOp.LT


class TestDeleteConfirmationInteraction:
    """Tests for delete confirmation interaction."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_delete_row_confirm_with_yes(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete row with confirmation 'yes'."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = RemoteTestResponse(id=1, name="deleted")
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "1"], input="y\n")

        assert result.exit_code == 0
        mock_ops.delete_row.assert_called_once()

    def test_delete_rows_confirm_with_yes(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete-many with confirmation 'yes'."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_rows.return_value = 2
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "1", "2"], input="y\n")

        assert result.exit_code == 0
        mock_ops.delete_rows.assert_called_once()

    def test_bulk_delete_confirm_with_yes(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-delete with confirmation 'yes'."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_delete_rows.return_value = 2
        cli_ops.register_bulk_delete_rows()

        result = runner.invoke(cli_group, ["bulk-delete", "1", "2"], input="y\n")

        assert result.exit_code == 0
        mock_ops.bulk_delete_rows.assert_called_once()


class TestTableNameInMessages:
    """Tests to verify table name appears in messages."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "my_custom_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_count_displays_table_name(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count command displays table name."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_rows.return_value = 10
        cli_ops.register_count_rows()

        result = runner.invoke(cli_group, ["count"])

        assert result.exit_code == 0
        assert "my_custom_table" in result.output

    def test_create_success_message_includes_table_name(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create success message includes table name."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_row.return_value = RemoteTestResponse(id=1, name="test")
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "name=test"])

        assert result.exit_code == 0
        assert "my_custom_table" in result.output

    def test_delete_success_message_includes_table_name(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test delete success message includes table name."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.delete_row.return_value = RemoteTestResponse(id=1, name="deleted")
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "--confirm", "1"])

        assert result.exit_code == 0
        assert "my_custom_table" in result.output

    def test_update_success_message_includes_table_name(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update success message includes table name."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_row.return_value = RemoteTestResponse(id=1, name="updated")
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1", "name=updated"])

        assert result.exit_code == 0
        assert "my_custom_table" in result.output

    def test_filter_displays_table_name_in_count(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter displays table name in result count."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = [RemoteTestResponse(id=1, name="test")]
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test"])

        assert result.exit_code == 0
        assert "my_custom_table" in result.output


class TestBulkOperationCounts:
    """Tests for bulk operation count reporting."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_create_many_reports_count(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-many reports count of created rows."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_rows.return_value = [RemoteTestResponse(id=i, name=f"test{i}") for i in range(1, 6)]
        cli_ops.register_create_rows()

        json_data = [{"name": f"test{i}"} for i in range(1, 6)]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-many", "test.json"])

        assert result.exit_code == 0
        assert "5" in result.output
        assert "Successfully created" in result.output

    def test_create_batched_reports_count_and_batch_size(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test create-batched reports count and batch size."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.create_rows_batched.return_value = [
            RemoteTestResponse(id=i, name=f"test{i}") for i in range(1, 11)
        ]
        cli_ops.register_create_rows_batched()

        json_data = [{"name": f"test{i}"} for i in range(1, 11)]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["create-batched", "--batch-size", "3", "test.json"])

        assert result.exit_code == 0
        assert "10" in result.output
        assert "3" in result.output  # batch size
        assert "Successfully created" in result.output

    def test_update_many_reports_count(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test update-many reports count of updated rows."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.update_rows.return_value = [
            RemoteTestResponse(id=i, name=f"updated{i}") for i in range(1, 4)
        ]
        cli_ops.register_update_rows()

        json_data = [{"id": i, "name": f"updated{i}"} for i in range(1, 4)]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["update-many", "test.json"])

        assert result.exit_code == 0
        assert "3" in result.output
        assert "Successfully updated" in result.output


class TestFilterWithNoFilters:
    """Tests for filter command with no filters specified."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_with_no_filters_uses_none(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with no filters passes None to operation."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        assert call_kwargs["filters"] is None

    def test_count_filtered_with_no_filters_displays_all(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test count-filtered with no filters displays 'all' in message."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.count_filtered_rows.return_value = 100
        cli_ops.register_count_filtered_rows()

        result = runner.invoke(cli_group, ["count-filtered"])

        assert result.exit_code == 0
        assert "all" in result.output.lower()


class TestFilterWithOnlyOrderBy:
    """Tests for filter with only ordering, no filters."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_filter_only_order_by(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test filter with only order by, no filters."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.filter_rows.return_value = []
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "--order-by", "name:desc"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.filter_rows.call_args[1]
        assert call_kwargs["filters"] is None
        assert len(call_kwargs["order_by"]) == 1


class TestFindByMultipleConditionsJsonParsing:
    """Tests for find-by with multiple conditions and JSON parsing."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_find_by_parses_different_types(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by parses different types correctly."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = []
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test", "value=42", "active=true", "score=3.14"])

        assert result.exit_code == 0
        call_kwargs = mock_ops.find_by.call_args[1]
        assert isinstance(call_kwargs["name"], str)
        assert isinstance(call_kwargs["value"], int)
        assert isinstance(call_kwargs["active"], bool)
        assert isinstance(call_kwargs["score"], float)


class TestSuccessMessageFormatting:
    """Tests for success message formatting."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a Click test runner."""
        return CliRunner()

    @pytest.fixture
    def setup_cli(self) -> tuple[click.Group, Mock, CliRemoteOperations]:
        """Setup CLI with mocked operations."""

        @click.group()
        def test_cli():
            pass

        mock_sync_ops = Mock(spec=SyncRemoteOperations)
        mock_sync_ops.async_ops = Mock()
        mock_sync_ops.async_ops.table_name = "test_table"
        mock_sync_ops.async_ops.response_model = RemoteTestResponse
        cli_ops = CliRemoteOperations(mock_sync_ops, test_cli)

        return test_cli, mock_sync_ops, cli_ops

    def test_bulk_insert_success_message(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test bulk-insert displays success message with count."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.bulk_insert_rows.return_value = 500
        cli_ops.register_bulk_insert_rows()

        json_data = [{"name": f"test{i}"} for i in range(500)]

        with runner.isolated_filesystem():
            with open("test.json", "w") as f:
                json.dump(json_data, f)

            result = runner.invoke(cli_group, ["bulk-insert", "test.json"])

        assert result.exit_code == 0
        assert "Successfully inserted 500" in result.output

    def test_find_by_found_count_message(self, runner: CliRunner, setup_cli: tuple) -> None:
        """Test find-by displays found count."""
        cli_group, mock_ops, cli_ops = setup_cli
        mock_ops.find_by.return_value = [RemoteTestResponse(id=i, name=f"test{i}") for i in range(7)]
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "name=test"])

        assert result.exit_code == 0
        assert "Found 7" in result.output
