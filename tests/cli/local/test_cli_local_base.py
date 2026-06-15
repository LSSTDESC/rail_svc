"""Unit tests for CLI operations base class."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import anyio
import click
import pytest
from click.testing import CliRunner
from pydantic import BaseModel, ValidationError
from sqlalchemy.exc import IntegrityError

from rail_svc.cli.local.base import CliOperations, handle_database_error
from rail_svc.local_sync.base import SyncOperations
from rail_svc.models import FilterOp
from rail_svc.models.utils import OutputEnum


# Test Models
class CliResponse(BaseModel):
    """Test response model."""

    id: int
    name: str
    value: int


class CliCreate(BaseModel):
    """Test create model."""

    name: str
    value: int


# Fixtures
@pytest.fixture
def mock_sync_ops() -> MagicMock:
    """Create mock SyncOperations instance."""
    sync_ops = MagicMock(spec=SyncOperations)

    # Mock the context
    mock_async_ops = MagicMock()
    mock_table_ops = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.class_string = "test_item"
    mock_table_ops.ctx = mock_ctx
    mock_async_ops._table_ops = mock_table_ops
    sync_ops.async_ops = mock_async_ops

    # Set up sync operation methods
    sync_ops.get_row = Mock()
    sync_ops.get_row_by_name = Mock()
    sync_ops.get_rows = Mock()
    sync_ops.get_row_or_none = Mock()
    sync_ops.count_rows = Mock()
    sync_ops.lookup_by_id_or_name = Mock()
    sync_ops.create_row = Mock()
    sync_ops.create_rows = Mock()
    sync_ops.create_rows_batched = Mock()
    sync_ops.bulk_insert_rows = Mock()
    sync_ops.update_row = Mock()
    sync_ops.update_rows = Mock()
    sync_ops.delete_row = Mock()
    sync_ops.delete_rows = Mock()
    sync_ops.bulk_delete_rows = Mock()
    sync_ops.filter_rows = Mock()
    sync_ops.count_filtered_rows = Mock()
    sync_ops.find_by = Mock()
    sync_ops.find_one_by = Mock()

    return sync_ops


@pytest.fixture
def cli_group() -> click.Group:
    """Create a test CLI group."""

    @click.group()
    def test_cli():
        pass

    return test_cli


@pytest.fixture
def cli_ops(mock_sync_ops: MagicMock, cli_group: click.Group) -> CliOperations:
    """Create CliOperations instance."""
    return CliOperations(mock_sync_ops, cli_group)


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


# Test Utility Methods
class TestUtilityMethods:
    """Tests for utility methods."""

    @pytest.mark.asyncio
    async def test_load_json_file_valid(self, cli_ops: CliOperations) -> None:
        """Test loading valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "test", "value": 1}], f)
            temp_path = f.name

        try:
            data = await cli_ops._load_json_file(temp_path)
            assert data == [{"name": "test", "value": 1}]
        finally:
            await anyio.Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_json_file_invalid_json(self, cli_ops: CliOperations) -> None:
        """Test loading invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            temp_path = f.name

        try:
            with pytest.raises(click.Abort):
                await cli_ops._load_json_file(temp_path)
        finally:
            await anyio.Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_json_file_not_array(self, cli_ops: CliOperations) -> None:
        """Test loading JSON that is not an array."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"name": "test"}, f)
            temp_path = f.name

        try:
            with pytest.raises(click.Abort):
                await cli_ops._load_json_file(temp_path)
        finally:
            await anyio.Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_json_file_empty_array(self, cli_ops: CliOperations) -> None:
        """Test loading empty JSON array."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            temp_path = f.name

        try:
            with pytest.raises(click.Abort):
                await cli_ops._load_json_file(temp_path)
        finally:
            await anyio.Path(temp_path).unlink()

    @pytest.mark.asyncio
    async def test_load_json_file_not_exists(self, cli_ops: CliOperations) -> None:
        """Test loading non-existent file."""
        with pytest.raises(click.Abort):
            await cli_ops._load_json_file("/nonexistent/file.json")

    def test_handle_database_error_validation(self, cli_ops: CliOperations) -> None:
        """Test handling validation error."""
        exc = ValidationError.from_exception_data(
            "TestModel", [{"type": "missing", "loc": ("field",), "msg": "Field required", "input": {}}]
        )

        with pytest.raises(click.Abort):
            handle_database_error(exc, "during test")

    def test_handle_database_error_integrity(self, cli_ops: CliOperations) -> None:
        """Test handling integrity error."""
        exc = IntegrityError("statement", {}, "orig")

        with pytest.raises(click.Abort):
            handle_database_error(exc, "during test")

    def test_handle_database_error_value(self, cli_ops: CliOperations) -> None:
        """Test handling value error."""
        exc = ValueError("Invalid value")

        with pytest.raises(click.Abort):
            handle_database_error(exc, "during test")

    def test_handle_database_error_generic(self, cli_ops: CliOperations) -> None:
        """Test handling generic error."""
        exc = RuntimeError("Something went wrong")

        with pytest.raises(click.Abort):
            handle_database_error(exc)


# Test Read Commands
class TestReadCommands:
    """Tests for read command registration."""

    @patch("rail_svc.db.session.init_db")
    def test_get_row_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-row command success."""
        cli_ops.register_get_row()

        mock_sync_ops.get_row.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["get-row", "1"])

        assert result.exit_code == 0
        mock_sync_ops.get_row.assert_called_once_with(row_id=1)
        mock_init_db.called

    @patch("rail_svc.db.session.init_db")
    def test_get_row_not_found(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-row command when row not found."""
        cli_ops.register_get_row()

        mock_sync_ops.get_row.side_effect = ValueError("Row not found")

        result = runner.invoke(cli_group, ["get-row", "999"])

        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_get_row_by_name_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-by-name command success."""
        cli_ops.register_get_row_by_name()

        mock_sync_ops.get_row_by_name.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["get-by-name", "test"])

        assert result.exit_code == 0
        mock_sync_ops.get_row_by_name.assert_called_once_with(name="test")

    @patch("rail_svc.db.session.init_db")
    def test_get_rows_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-rows command success."""
        cli_ops.register_get_rows()

        mock_sync_ops.get_rows.return_value = [
            CliResponse(id=1, name="test1", value=100),
            CliResponse(id=2, name="test2", value=200),
        ]

        result = runner.invoke(cli_group, ["get-rows"])

        assert result.exit_code == 0
        mock_sync_ops.get_rows.assert_called_once()

    @patch("rail_svc.db.session.init_db")
    def test_get_rows_with_pagination(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-rows with pagination options."""
        cli_ops.register_get_rows()

        mock_sync_ops.get_rows.return_value = []

        result = runner.invoke(cli_group, ["get-rows", "--skip", "10", "--limit", "20"])

        assert result.exit_code == 0
        mock_sync_ops.get_rows.assert_called_once_with(skip=10, limit=20)

    @patch("rail_svc.db.session.init_db")
    def test_get_row_or_none_found(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-row-if-exists when row exists."""
        cli_ops.register_get_row_or_none()

        mock_sync_ops.get_row_or_none.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["get-row-if-exists", "1"])
        assert result.exit_code == 0

    @patch("rail_svc.db.session.init_db")
    def test_get_row_or_none_not_found(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-row-if-exists when row doesn't exist."""
        cli_ops.register_get_row_or_none()

        mock_sync_ops.get_row_or_none.return_value = None

        result = runner.invoke(cli_group, ["get-row-if-exists", "999"])

        assert result.exit_code == 0
        assert "No" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_count_rows_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test count command success."""
        cli_ops.register_count_rows()

        mock_sync_ops.count_rows.return_value = 42

        result = runner.invoke(cli_group, ["count"])

        assert result.exit_code == 0
        assert "42" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_lookup_by_id_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test lookup by ID."""
        cli_ops.register_lookup_by_id_or_name()

        mock_sync_ops.lookup_by_id_or_name.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["lookup", "--id", "1"])

        assert result.exit_code == 0
        mock_sync_ops.lookup_by_id_or_name.assert_called_once_with(row_id=1, name=None)

    @patch("rail_svc.db.session.init_db")
    def test_lookup_by_name_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test lookup by name."""
        cli_ops.register_lookup_by_id_or_name()

        mock_sync_ops.lookup_by_id_or_name.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["lookup", "--name", "test"])

        assert result.exit_code == 0
        mock_sync_ops.lookup_by_id_or_name.assert_called_once_with(row_id=None, name="test")

    @patch("rail_svc.db.session.init_db")
    def test_lookup_without_params(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test lookup without ID or name."""
        cli_ops.register_lookup_by_id_or_name()

        result = runner.invoke(cli_group, ["lookup"])

        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_lookup_with_both_params(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test lookup with both ID and name."""
        cli_ops.register_lookup_by_id_or_name()

        result = runner.invoke(cli_group, ["lookup", "--id", "1", "--name", "test"])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_register_all_read_commands(self, cli_ops: CliOperations, cli_group: click.Group) -> None:
        """Test registering all read commands."""
        cli_ops.register_all_read_commands()

        # Check that commands were registered
        assert "get-row" in cli_group.commands
        assert "get-by-name" in cli_group.commands
        assert "get-rows" in cli_group.commands
        assert "get-row-if-exists" in cli_group.commands
        assert "count" in cli_group.commands
        assert "lookup" in cli_group.commands


# Test Create Commands
class TestCreateCommands:
    """Tests for create command registration."""

    @patch("rail_svc.db.session.init_db")
    def test_create_row_with_args(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command with arguments."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["create", "name=test", "value=100"])

        assert result.exit_code == 0
        mock_sync_ops.create_row.assert_called_once_with(validate=True, name="test", value=100)

    @patch("rail_svc.db.session.init_db")
    def test_create_row_from_json(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command from JSON file."""
        cli_ops.register_create_row()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"name": "test", "value": 100}, f)
            temp_path = f.name

        try:
            mock_sync_ops.create_row.return_value = CliResponse(id=1, name="test", value=100)

            result = runner.invoke(cli_group, ["create", "--from-json", temp_path])

            assert result.exit_code == 0
            mock_sync_ops.create_row.assert_called_once()
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_create_row_no_validate(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command without validation."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["create", "--no-validate", "name=test", "value=100"])

        assert result.exit_code == 0
        mock_sync_ops.create_row.assert_called_once_with(validate=False, name="test", value=100)

    @patch("rail_svc.db.session.init_db")
    def test_create_row_invalid_format(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test create command with invalid field format."""
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "invalid_format"])

        assert result.exit_code != 0
        assert "Invalid field format" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_create_row_no_data(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test create command without data."""
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create"])

        assert result.exit_code != 0
        assert "No data provided" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_create_rows_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create-many command success."""
        cli_ops.register_create_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "test1", "value": 100}, {"name": "test2", "value": 200}], f)
            temp_path = f.name

        try:
            mock_sync_ops.create_rows.return_value = [
                CliResponse(id=1, name="test1", value=100),
                CliResponse(id=2, name="test2", value=200),
            ]

            result = runner.invoke(cli_group, ["create-many", temp_path])

            assert result.exit_code == 0
            assert "2" in result.output
            mock_sync_ops.create_rows.assert_called_once()
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_create_rows_not_array(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test create-many with non-array JSON."""
        cli_ops.register_create_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"name": "test"}, f)
            temp_path = f.name

        try:
            result = runner.invoke(cli_group, ["create-many", temp_path])

            assert result.exit_code != 0
            assert "must contain an array" in result.output
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_create_rows_batched_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create-batched command success."""
        cli_ops.register_create_rows_batched()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": f"test{i}", "value": i} for i in range(5)], f)
            temp_path = f.name

        try:
            mock_sync_ops.create_rows_batched.return_value = [
                CliResponse(id=i, name=f"test{i}", value=i) for i in range(5)
            ]

            result = runner.invoke(cli_group, ["create-batched", "--batch-size", "2", temp_path])

            assert result.exit_code == 0
            mock_sync_ops.create_rows_batched.assert_called_once()
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_create_rows_batched_invalid_size(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test create-batched with invalid batch size."""
        cli_ops.register_create_rows_batched()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "test"}], f)
            temp_path = f.name

        try:
            result = runner.invoke(cli_group, ["create-batched", "--batch-size", "0", temp_path])

            assert result.exit_code != 0
            assert "at least 1" in result.output
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_bulk_insert_rows_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test bulk-insert command success."""
        cli_ops.register_bulk_insert_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "test1", "value": 100}, {"name": "test2", "value": 200}], f)
            temp_path = f.name

        try:
            mock_sync_ops.bulk_insert_rows.return_value = 2

            result = runner.invoke(cli_group, ["bulk-insert", temp_path])

            assert result.exit_code == 0
            assert "2" in result.output
        finally:
            Path(temp_path).unlink()

    def test_register_all_create_commands(self, cli_ops: CliOperations, cli_group: click.Group) -> None:
        """Test registering all create commands."""
        cli_ops.register_all_create_commands()

        assert "create" in cli_group.commands
        assert "create-many" in cli_group.commands
        assert "create-batched" in cli_group.commands
        assert "bulk-insert" in cli_group.commands


# Test Update Commands
class TestUpdateCommands:
    """Tests for update command registration."""

    @patch("rail_svc.db.session.init_db")
    def test_update_row_with_args(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test update command with arguments."""
        cli_ops.register_update_row()

        mock_sync_ops.update_row.return_value = CliResponse(id=1, name="updated", value=200)

        result = runner.invoke(cli_group, ["update", "1", "name=updated", "value=200"])

        assert result.exit_code == 0
        mock_sync_ops.update_row.assert_called_once_with(row_id=1, name="updated", value=200)

    @patch("rail_svc.db.session.init_db")
    def test_update_row_from_json(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test update command from JSON file."""
        cli_ops.register_update_row()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"name": "updated", "value": 200}, f)
            temp_path = f.name

        try:
            mock_sync_ops.update_row.return_value = CliResponse(id=1, name="updated", value=200)

            result = runner.invoke(cli_group, ["update", "--from-json", temp_path, "1"])

            assert result.exit_code == 0
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_update_row_no_data(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test update command without data."""
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1"])

        assert result.exit_code != 0
        assert "No update data" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_update_row_prevent_id_change(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test that ID changes are prevented."""
        cli_ops.register_update_row()

        result = runner.invoke(cli_group, ["update", "1", "id=2", "name=test"])

        assert result.exit_code != 0
        assert "Cannot change row ID" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_update_rows_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test update-many command success."""
        cli_ops.register_update_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"id": 1, "name": "updated1"}, {"id": 2, "name": "updated2"}], f)
            temp_path = f.name

        try:
            mock_sync_ops.update_rows.return_value = [
                CliResponse(id=1, name="updated1", value=100),
                CliResponse(id=2, name="updated2", value=200),
            ]

            result = runner.invoke(cli_group, ["update-many", temp_path])

            assert result.exit_code == 0
            assert "2" in result.output
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_update_rows_missing_id(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test update-many with missing ID field."""
        cli_ops.register_update_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "updated"}], f)
            temp_path = f.name

        try:
            result = runner.invoke(cli_group, ["update-many", temp_path])

            assert result.exit_code != 0
            assert "missing 'id' field" in result.output
        finally:
            Path(temp_path).unlink()

    def test_register_all_update_commands(self, cli_ops: CliOperations, cli_group: click.Group) -> None:
        """Test registering all update commands."""
        cli_ops.register_all_update_commands()

        assert "update" in cli_group.commands
        assert "update-many" in cli_group.commands


# Test Delete Commands
class TestDeleteCommands:
    """Tests for delete command registration."""

    @patch("rail_svc.db.session.init_db")
    def test_delete_row_with_confirmation(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete command with confirmation."""
        cli_ops.register_delete_row()

        mock_sync_ops.delete_row.return_value = {"id": 1, "name": "test", "value": 100}

        result = runner.invoke(cli_group, ["delete", "--confirm", "1"])
        assert result.exit_code == 0
        mock_sync_ops.delete_row.assert_called_once_with(row_id=1, capture_data=True)

    @patch("rail_svc.db.session.init_db")
    def test_delete_row_with_prompt_yes(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete command with confirmation prompt (yes)."""
        cli_ops.register_delete_row()

        mock_sync_ops.delete_row.return_value = {"id": 1, "name": "test", "value": 100}

        result = runner.invoke(cli_group, ["delete", "1"], input="y\n")

        assert result.exit_code == 0
        mock_sync_ops.delete_row.assert_called_once()

    @patch("rail_svc.db.session.init_db")
    def test_delete_row_with_prompt_no(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete command with confirmation prompt (no)."""
        cli_ops.register_delete_row()

        result = runner.invoke(cli_group, ["delete", "1"], input="n\n")

        assert result.exit_code == 0
        assert "cancelled" in result.output
        mock_sync_ops.delete_row.assert_not_called()

    @patch("rail_svc.db.session.init_db")
    def test_delete_row_no_capture(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete command without data capture."""
        cli_ops.register_delete_row()

        mock_sync_ops.delete_row.return_value = None

        result = runner.invoke(cli_group, ["delete", "--confirm", "--no-capture", "1"])

        assert result.exit_code == 0
        mock_sync_ops.delete_row.assert_called_once_with(row_id=1, capture_data=False)

    @patch("rail_svc.db.session.init_db")
    def test_delete_rows_with_args(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete-many command with arguments."""
        cli_ops.register_delete_rows()

        mock_sync_ops.delete_rows.return_value = None

        result = runner.invoke(cli_group, ["delete-many", "--confirm", "1", "2", "3"])

        assert result.exit_code == 0
        mock_sync_ops.delete_rows.assert_called_once_with(row_ids=[1, 2, 3], capture_data=False)

    @patch("rail_svc.db.session.init_db")
    def test_delete_rows_from_file_json(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete-many from JSON file."""
        cli_ops.register_delete_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            temp_path = f.name

        try:
            mock_sync_ops.delete_rows.return_value = None

            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", temp_path])

            assert result.exit_code == 0
            mock_sync_ops.delete_rows.assert_called_once()
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_delete_rows_from_file_text(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete-many from text file."""
        cli_ops.register_delete_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1\n2\n3\n")
            temp_path = f.name

        try:
            mock_sync_ops.delete_rows.return_value = None

            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", temp_path])

            assert result.exit_code == 0
            mock_sync_ops.delete_rows.assert_called_once()
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_delete_rows_no_ids(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test delete-many without IDs."""
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "--confirm"])

        assert result.exit_code != 0
        assert "No IDs provided" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_delete_rows_with_capture(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete-many with data capture."""
        cli_ops.register_delete_rows()

        mock_sync_ops.delete_rows.return_value = [
            {"id": 1, "name": "test1", "value": 100},
            {"id": 2, "name": "test2", "value": 200},
        ]

        result = runner.invoke(cli_group, ["delete-many", "--confirm", "--capture-data", "1", "2"])

        assert result.exit_code == 0
        mock_sync_ops.delete_rows.assert_called_once_with(row_ids=[1, 2], capture_data=True)

    @patch("rail_svc.db.session.init_db")
    def test_bulk_delete_rows_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test bulk-delete command success."""
        cli_ops.register_bulk_delete_rows()

        mock_sync_ops.bulk_delete_rows.return_value = 3

        result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "1", "2", "3"])

        assert result.exit_code == 0
        assert "3" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_bulk_delete_rows_partial_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test bulk-delete when some IDs not found."""
        cli_ops.register_bulk_delete_rows()

        mock_sync_ops.bulk_delete_rows.return_value = 2  # Only 2 out of 3 deleted

        result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "1", "2", "999"])

        assert result.exit_code == 0
        assert "2" in result.output
        assert "not found" in result.output

    def test_register_all_delete_commands(self, cli_ops: CliOperations, cli_group: click.Group) -> None:
        """Test registering all delete commands."""
        cli_ops.register_all_delete_commands()

        assert "delete" in cli_group.commands
        assert "delete-many" in cli_group.commands
        assert "bulk-delete" in cli_group.commands


# Test Filter Commands
class TestFilterCommands:
    """Tests for filter command registration."""

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_single_condition(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter command with single condition."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = [
            CliResponse(id=1, name="test", value=100),
        ]

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test"])

        assert result.exit_code == 0
        mock_sync_ops.filter_rows.assert_called_once()

        # Check that the filter was created correctly
        call_args = mock_sync_ops.filter_rows.call_args
        filters = call_args.kwargs["filters"]
        assert len(filters) == 1
        assert filters[0].field == "name"
        assert filters[0].op == FilterOp.EQ

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_multiple_conditions_and(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with multiple AND conditions."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test", "-f", "value:gt:50"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        assert call_args.kwargs["logical_op"] == "and"

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_multiple_conditions_or(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with multiple OR conditions."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test1", "-f", "name:eq:test2", "--or"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        assert call_args.kwargs["logical_op"] == "or"

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_with_order_by(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with order by."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test", "--order-by", "created_at:desc"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        order_by = call_args.kwargs["order_by"]
        assert len(order_by) == 1
        assert order_by[0].field == "created_at"
        assert order_by[0].descending is True

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_invalid_format(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test filter with invalid format."""
        cli_ops.register_filter_rows()

        result = runner.invoke(cli_group, ["filter", "-f", "invalid_format"])

        assert result.exit_code != 0
        assert "Invalid filter format" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_invalid_operator(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test filter with invalid operator."""
        cli_ops.register_filter_rows()
        result = runner.invoke(cli_group, ["filter", "-f", "name:invalid_op:test"])

        assert result.exit_code != 0
        assert "Unknown operator" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_in_operator(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with IN operator."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "status:in:active,pending,done"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        filters = call_args.kwargs["filters"]
        assert filters[0].op == FilterOp.IN
        assert filters[0].value == ["active", "pending", "done"]

    @patch("rail_svc.db.session.init_db")
    def test_count_filtered_rows_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test count-filtered command success."""
        cli_ops.register_count_filtered_rows()

        mock_sync_ops.count_filtered_rows.return_value = 42

        result = runner.invoke(cli_group, ["count-filtered", "-f", "name:eq:test"])

        assert result.exit_code == 0
        assert "42" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_count_filtered_rows_no_filters(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test count-filtered without filters."""
        cli_ops.register_count_filtered_rows()

        mock_sync_ops.count_filtered_rows.return_value = 100

        result = runner.invoke(cli_group, ["count-filtered"])

        assert result.exit_code == 0
        assert "100" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_find_by_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test find-by command success."""
        cli_ops.register_find_by()

        mock_sync_ops.find_by.return_value = [
            CliResponse(id=1, name="test", value=100),
        ]

        result = runner.invoke(cli_group, ["find-by", "name=test", "value=100"])

        assert result.exit_code == 0
        mock_sync_ops.find_by.assert_called_once()

    @patch("rail_svc.db.session.init_db")
    def test_find_by_no_conditions(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test find-by without conditions."""
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by"])

        assert result.exit_code != 0
        assert "No conditions provided" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_find_by_invalid_format(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test find-by with invalid condition format."""
        cli_ops.register_find_by()

        result = runner.invoke(cli_group, ["find-by", "invalid"])

        assert result.exit_code != 0
        assert "Invalid condition format" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_find_by_with_order_by(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test find-by with ordering."""
        cli_ops.register_find_by()

        mock_sync_ops.find_by.return_value = []

        result = runner.invoke(
            cli_group, ["find-by", "status=active", "--order-by", "name", "--order-by", "created_at:desc"]
        )

        assert result.exit_code == 0
        call_args = mock_sync_ops.find_by.call_args
        order_by = call_args.kwargs["order_by"]
        assert len(order_by) == 2

    @patch("rail_svc.db.session.init_db")
    def test_find_one_by_success(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test find-one-by command success."""
        cli_ops.register_find_one_by()

        mock_sync_ops.find_one_by.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["find-one-by", "name=test"])

        assert result.exit_code == 0
        mock_sync_ops.find_one_by.assert_called_once()

    @patch("rail_svc.db.session.init_db")
    def test_find_one_by_no_conditions(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test find-one-by without conditions."""
        cli_ops.register_find_one_by()

        result = runner.invoke(cli_group, ["find-one-by"])

        assert result.exit_code != 0
        assert "No conditions provided" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_find_one_by_not_found(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test find-one-by when row not found."""
        cli_ops.register_find_one_by()

        mock_sync_ops.find_one_by.side_effect = ValueError("No rows found")

        result = runner.invoke(cli_group, ["find-one-by", "name=nonexistent"])

        assert result.exit_code != 0
        assert "Error" in result.output

    def test_register_all_filter_commands(self, cli_ops: CliOperations, cli_group: click.Group) -> None:
        """Test registering all filter commands."""
        cli_ops.register_all_filter_commands()

        assert "filter" in cli_group.commands
        assert "count-filtered" in cli_group.commands
        assert "find-by" in cli_group.commands
        assert "find-one-by" in cli_group.commands


# Test Complex JSON Value Parsing
class TestJSONValueParsing:
    """Tests for JSON value parsing in commands."""

    @patch("rail_svc.db.session.init_db")
    def test_create_with_json_values(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command with JSON-parseable values."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="test", value=100)

        # Pass a boolean as JSON
        result = runner.invoke(cli_group, ["create", "name=test", "value=100", "active=true"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.create_row.call_args
        assert call_args.kwargs["active"] is True

    @patch("rail_svc.db.session.init_db")
    def test_create_with_string_values(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command with string values that aren't valid JSON."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["create", "name=test name", "value=100"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.create_row.call_args
        assert call_args.kwargs["name"] == "test name"

    @patch("rail_svc.db.session.init_db")
    def test_filter_with_numeric_value(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with numeric value."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:50"])
        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        filters = call_args.kwargs["filters"]
        # The value should be parsed as a number
        assert isinstance(filters[0].value, int)
        assert filters[0].value == 50


# Test Output Formats
class TestOutputFormats:
    """Tests for different output format options."""

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.output_pydantic")
    def test_get_rows_json_output(
        self,
        mock_output: MagicMock,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-rows with JSON output."""
        cli_ops.register_get_rows()

        mock_sync_ops.get_rows.return_value = []
        mock_output.return_value = "[]"

        result = runner.invoke(cli_group, ["get-rows", "--output", "json"])

        assert result.exit_code == 0
        mock_output.assert_called_once()
        assert mock_output.call_args[0][1] == OutputEnum.json

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.output_pydantic")
    def test_get_rows_table_output(
        self,
        mock_output: MagicMock,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-rows with table output."""
        cli_ops.register_get_rows()

        mock_sync_ops.get_rows.return_value = []
        mock_output.return_value = ""

        result = runner.invoke(cli_group, ["get-rows", "--output", "table"])

        assert result.exit_code == 0
        assert mock_output.call_args[0][1] == OutputEnum.table

    @patch("rail_svc.db.session.init_db")
    @patch("rail_svc.cli.local.base.output_pydantic")
    def test_get_rows_yaml_output(
        self,
        mock_output: MagicMock,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-rows with table output."""
        cli_ops.register_get_rows()

        mock_sync_ops.get_rows.return_value = []
        mock_output.return_value = ""

        result = runner.invoke(cli_group, ["get-rows", "--output", "yaml"])

        assert result.exit_code == 0
        assert mock_output.call_args[0][1] == OutputEnum.yaml


# Test Error Handling
class TestErrorHandling:
    """Tests for error handling in various scenarios."""

    @patch("rail_svc.db.session.init_db")
    def test_create_with_validation_error(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command with validation error."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.side_effect = ValidationError.from_exception_data(
            "CliCreate", [{"type": "missing", "loc": ("value",), "msg": "Field required", "input": {}}]
        )

        result = runner.invoke(cli_group, ["create", "name=test"])

        assert result.exit_code != 0
        assert "Validation failed" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_create_with_integrity_error(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command with integrity error."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.side_effect = IntegrityError("statement", {}, "duplicate key")

        result = runner.invoke(cli_group, ["create", "name=test", "value=100"])

        assert result.exit_code != 0
        assert "Integrity constraint" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_update_with_value_error(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test update command with value error."""
        cli_ops.register_update_row()

        mock_sync_ops.update_row.side_effect = ValueError("Invalid value for field")

        result = runner.invoke(cli_group, ["update", "1", "name=test"])

        assert result.exit_code != 0
        assert "Error" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_filter_with_generic_error(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter command with generic error."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.side_effect = RuntimeError("Unexpected database error")

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test"])

        assert result.exit_code != 0
        assert "Error" in result.output


# Test File Operations
class TestFileOperations:
    """Tests for file-based operations."""

    @patch("rail_svc.db.session.init_db")
    def test_create_from_json_invalid_file(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test create from non-existent JSON file."""
        cli_ops.register_create_row()

        result = runner.invoke(cli_group, ["create", "--from-json", "/nonexistent/file.json"])

        assert result.exit_code != 0
        assert "Usage" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_create_rows_invalid_json(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test create-many with invalid JSON."""
        cli_ops.register_create_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {")
            temp_path = f.name

        try:
            result = runner.invoke(cli_group, ["create-many", temp_path])

            assert result.exit_code != 0
            assert "Invalid JSON" in result.output
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_delete_from_file_mixed_format(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete from file with mixed text/empty lines."""
        cli_ops.register_delete_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("1\n\n2\n3\n\n")
            temp_path = f.name

        try:
            mock_sync_ops.delete_rows.return_value = None

            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", temp_path])

            assert result.exit_code == 0
            # Should parse only valid lines
            call_args = mock_sync_ops.delete_rows.call_args
            assert call_args.kwargs["row_ids"] == [1, 2, 3]
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_update_from_json_not_dict(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test update with JSON that's not a dict."""
        cli_ops.register_update_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)  # Array instead of object
            temp_path = f.name

        try:
            result = runner.invoke(cli_group, ["update-many", temp_path])

            assert result.exit_code != 0
            assert "is not an object" in result.output
        finally:
            Path(temp_path).unlink()


# Test Pagination
class TestPagination:
    """Tests for pagination parameters."""

    @patch("rail_svc.db.session.init_db")
    def test_get_rows_negative_skip(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test get-rows with negative skip."""
        cli_ops.register_get_rows()

        result = runner.invoke(cli_group, ["get-rows", "--skip", "-1"])

        # Click should validate this before our code
        assert result.exit_code != 0

    @patch("rail_svc.db.session.init_db")
    def test_get_rows_with_page_size(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test get-rows with page size."""
        cli_ops.register_get_rows()

        mock_sync_ops.get_rows.return_value = []

        result = runner.invoke(cli_group, ["get-rows", "--page-size", "50"])

        assert result.exit_code == 0
        # Should use page_size when limit not specified
        call_args = mock_sync_ops.get_rows.call_args
        assert call_args.kwargs["limit"] == 50

    @patch("rail_svc.db.session.init_db")
    def test_filter_rows_pagination(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with pagination."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test", "--skip", "10", "--limit", "20"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        assert call_args.kwargs["skip"] == 10
        assert call_args.kwargs["limit"] == 20


# Test Context Manager and Initialization
class TestInitialization:
    """Tests for CLI operations initialization."""

    def test_cli_ops_initialization(self, mock_sync_ops: MagicMock, cli_group: click.Group) -> None:
        """Test CliOperations initialization."""
        cli_ops = CliOperations(mock_sync_ops, cli_group)

        assert cli_ops.sync_oper == mock_sync_ops
        assert cli_ops.group == cli_group
        assert cli_ops.ctx is not None
        assert cli_ops.ctx.class_string == "test_item"

    def test_cli_ops_context_extraction(self, mock_sync_ops: MagicMock, cli_group: click.Group) -> None:
        """Test that context is correctly extracted from sync operations."""
        cli_ops = CliOperations(mock_sync_ops, cli_group)

        # Verify the context chain
        assert cli_ops.ctx == mock_sync_ops.async_ops._table_ops.ctx


# Test Special Cases
class TestSpecialCases:
    """Tests for special cases and edge conditions."""

    @patch("rail_svc.db.session.init_db")
    def test_delete_with_empty_file(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test delete from empty file."""
        cli_ops.register_delete_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            result = runner.invoke(cli_group, ["delete-many", "--confirm", "--from-file", temp_path])

            assert result.exit_code != 0
            assert "No IDs provided" in result.output
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_bulk_delete_from_json_not_array(
        self, mock_init_db: MagicMock, cli_ops: CliOperations, cli_group: click.Group, runner: CliRunner
    ) -> None:
        """Test bulk-delete from JSON that's not an array."""
        cli_ops.register_bulk_delete_rows()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"id": 1}, f)
            temp_path = f.name

        try:
            result = runner.invoke(cli_group, ["bulk-delete", "--confirm", "--from-file", temp_path])

            assert result.exit_code != 0
            assert "JSON must be an array" in result.output
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_filter_with_like_operator(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with LIKE operator."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "name:like:%test%"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        filters = call_args.kwargs["filters"]
        assert filters[0].op == FilterOp.LIKE
        assert filters[0].value == ["%test%"]

    @patch("rail_svc.db.session.init_db")
    def test_find_by_with_json_array_value(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test find-by with JSON array value."""
        cli_ops.register_find_by()

        mock_sync_ops.find_by.return_value = []

        result = runner.invoke(cli_group, ["find-by", 'tags=["tag1","tag2"]'])

        assert result.exit_code == 0
        call_args = mock_sync_ops.find_by.call_args
        # Should parse as JSON array
        assert isinstance(call_args.kwargs["tags"], list)

    @patch("rail_svc.db.session.init_db")
    def test_create_with_null_value(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create with null JSON value."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="test", value=0)

        result = runner.invoke(cli_group, ["create", "name=test", "description=null"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.create_row.call_args
        assert call_args.kwargs["description"] is None


# Test Command Registration Groups
class TestCommandRegistrationGroups:
    """Tests for bulk command registration methods."""

    def test_all_commands_registered(self, mock_sync_ops: MagicMock, cli_group: click.Group) -> None:
        """Test that all command groups register correctly."""
        cli_ops = CliOperations(mock_sync_ops, cli_group)

        cli_ops.register_all_read_commands()
        cli_ops.register_all_create_commands()
        cli_ops.register_all_update_commands()
        cli_ops.register_all_delete_commands()
        cli_ops.register_all_filter_commands()

        # Verify all expected commands are present
        expected_commands = [
            "get-row",
            "get-by-name",
            "get-rows",
            "get-row-if-exists",
            "count",
            "lookup",
            "create",
            "create-many",
            "create-batched",
            "bulk-insert",
            "update",
            "update-many",
            "delete",
            "delete-many",
            "bulk-delete",
            "filter",
            "count-filtered",
            "find-by",
            "find-one-by",
        ]

        for cmd in expected_commands:
            assert cmd in cli_group.commands, f"Command '{cmd}' not registered"


# Test Order By Parsing
class TestOrderByParsing:
    """Tests for order by parameter parsing."""

    @patch("rail_svc.db.session.init_db")
    def test_filter_order_by_ascending(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with ascending order."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:0", "--order-by", "name"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        order_by = call_args.kwargs["order_by"]
        assert order_by[0].field == "name"
        assert order_by[0].descending is False

    @patch("rail_svc.db.session.init_db")
    def test_filter_order_by_descending(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with descending order."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "value:gt:0", "--order-by", "created_at:desc"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        order_by = call_args.kwargs["order_by"]
        assert order_by[0].field == "created_at"
        assert order_by[0].descending is True

    @patch("rail_svc.db.session.init_db")
    def test_find_by_multiple_order_by(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test find-by with multiple order by clauses."""
        cli_ops.register_find_by()

        mock_sync_ops.find_by.return_value = []

        result = runner.invoke(
            cli_group, ["find-by", "status=active", "--order-by", "priority:desc", "--order-by", "name"]
        )

        assert result.exit_code == 0
        call_args = mock_sync_ops.find_by.call_args
        order_by = call_args.kwargs["order_by"]
        assert len(order_by) == 2
        assert order_by[0].field == "priority"
        assert order_by[0].descending is True
        assert order_by[1].field == "name"
        assert order_by[1].descending is False


# Test Confirmation Prompts
class TestConfirmationPrompts:
    """Tests for confirmation prompt behavior."""

    @patch("rail_svc.db.session.init_db")
    def test_delete_many_prompt_cancelled(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete-many with cancelled prompt."""
        cli_ops.register_delete_rows()

        result = runner.invoke(cli_group, ["delete-many", "1", "2"], input="n\n")

        assert result.exit_code == 0
        assert "cancelled" in result.output
        mock_sync_ops.delete_rows.assert_not_called()

    @patch("rail_svc.db.session.init_db")
    def test_delete_many_prompt_confirmed(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete-many with confirmed prompt."""
        cli_ops.register_delete_rows()

        mock_sync_ops.delete_rows.return_value = None

        result = runner.invoke(cli_group, ["delete-many", "1", "2"], input="y\n")

        assert result.exit_code == 0
        mock_sync_ops.delete_rows.assert_called_once()

    @patch("rail_svc.db.session.init_db")
    def test_bulk_delete_prompt_with_warning(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test bulk-delete shows warning in prompt."""
        cli_ops.register_bulk_delete_rows()

        result = runner.invoke(cli_group, ["bulk-delete", "1", "2"], input="n\n")

        assert result.exit_code == 0
        assert "WARNING" in result.output
        assert "without calling hooks" in result.output


# Test Complex Filter Scenarios
class TestComplexFilterScenarios:
    """Tests for complex filtering scenarios."""

    @patch("rail_svc.db.session.init_db")
    def test_filter_multiple_operators(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with multiple different operators."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(
            cli_group,
            [
                "filter",
                "-f",
                "name:like:%test%",
                "-f",
                "value:gt:10",
                "-f",
                "value:lt:100",
                "-f",
                "status:in:active,pending",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        filters = call_args.kwargs["filters"]
        assert len(filters) == 4
        assert filters[0].op == FilterOp.LIKE
        assert filters[1].op == FilterOp.GT
        assert filters[2].op == FilterOp.LT
        assert filters[3].op == FilterOp.IN

    @patch("rail_svc.db.session.init_db")
    def test_filter_with_all_options(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with all available options."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(
            cli_group,
            [
                "filter",
                "-f",
                "status:eq:active",
                "--or",
                "--order-by",
                "priority:desc",
                "--order-by",
                "name",
                "--skip",
                "10",
                "--limit",
                "20",
                "--output",
                "json",
            ],
        )

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        assert call_args.kwargs["logical_op"] == "or"
        assert call_args.kwargs["skip"] == 10
        assert call_args.kwargs["limit"] == 20
        assert len(call_args.kwargs["order_by"]) == 2


# Test Edge Cases in Value Parsing
class TestValueParsingEdgeCases:
    """Tests for edge cases in value parsing."""

    @patch("rail_svc.db.session.init_db")
    def test_create_with_empty_string(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create with empty string value."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="", value=100)

        result = runner.invoke(cli_group, ["create", "name=", "value=100"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.create_row.call_args
        assert call_args.kwargs["name"] == ""

    @patch("rail_svc.db.session.init_db")
    def test_create_with_equals_in_value(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create with equals sign in value."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="a=b", value=100)

        result = runner.invoke(cli_group, ["create", "name=a=b", "value=100"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.create_row.call_args
        assert call_args.kwargs["name"] == "a=b"

    @patch("rail_svc.db.session.init_db")
    def test_filter_with_special_characters(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test filter with special characters in value."""
        cli_ops.register_filter_rows()

        mock_sync_ops.filter_rows.return_value = []

        result = runner.invoke(cli_group, ["filter", "-f", "name:eq:test@example.com"])

        assert result.exit_code == 0
        call_args = mock_sync_ops.filter_rows.call_args
        filters = call_args.kwargs["filters"]
        assert filters[0].value == ["test@example.com"]


# Test Batch Size Validation
class TestBatchSizeValidation:
    """Tests for batch size validation."""

    @patch("rail_svc.db.session.init_db")
    def test_create_batched_default_batch_size(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create-batched uses default batch size."""
        cli_ops.register_create_rows_batched()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "test"}], f)
            temp_path = f.name

        try:
            mock_sync_ops.create_rows_batched.return_value = [CliResponse(id=1, name="test", value=100)]

            result = runner.invoke(cli_group, ["create-batched", temp_path])

            assert result.exit_code == 0
            call_args = mock_sync_ops.create_rows_batched.call_args
            assert call_args.kwargs["batch_size"] == 1000  # Default
        finally:
            Path(temp_path).unlink()

    @patch("rail_svc.db.session.init_db")
    def test_create_batched_custom_batch_size(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create-batched with custom batch size."""
        cli_ops.register_create_rows_batched()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"name": "test"}], f)
            temp_path = f.name

        try:
            mock_sync_ops.create_rows_batched.return_value = [CliResponse(id=1, name="test", value=100)]

            result = runner.invoke(cli_group, ["create-batched", "--batch-size", "500", temp_path])

            assert result.exit_code == 0
            call_args = mock_sync_ops.create_rows_batched.call_args
            assert call_args.kwargs["batch_size"] == 500
        finally:
            Path(temp_path).unlink()


# Test Success Messages
class TestSuccessMessages:
    """Tests for success message output."""

    @patch("rail_svc.db.session.init_db")
    def test_create_success_message(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test create command shows success message."""
        cli_ops.register_create_row()

        mock_sync_ops.create_row.return_value = CliResponse(id=1, name="test", value=100)

        result = runner.invoke(cli_group, ["create", "name=test", "value=100"])

        assert result.exit_code == 0
        assert "Created" in result.output
        assert "successfully" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_update_success_message(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test update command shows success message."""
        cli_ops.register_update_row()

        mock_sync_ops.update_row.return_value = CliResponse(id=1, name="updated", value=200)

        result = runner.invoke(cli_group, ["update", "1", "name=updated"])

        assert result.exit_code == 0
        assert "Successfully updated" in result.output

    @patch("rail_svc.db.session.init_db")
    def test_delete_success_message(
        self,
        mock_init_db: MagicMock,
        cli_ops: CliOperations,
        mock_sync_ops: MagicMock,
        cli_group: click.Group,
        runner: CliRunner,
    ) -> None:
        """Test delete command shows success message."""
        cli_ops.register_delete_row()

        mock_sync_ops.delete_row.return_value = {"id": 1, "name": "test", "value": 100}

        result = runner.invoke(cli_group, ["delete", "--confirm", "1"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
