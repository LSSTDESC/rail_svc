"""Unit tests for output formatting functions"""

import json

import pytest
import yaml
from pydantic import BaseModel

from rail_svc.models.utils import (OutputEnum, display_table, format_output,
                                   output_json, output_pydantic,
                                   output_pydantic_list,
                                   output_pydantic_single)


class TestOutputEnum:
    """Tests for OutputEnum"""

    def test_output_enum_values(self):
        """Test that OutputEnum has expected values"""
        assert OutputEnum.yaml
        assert OutputEnum.json
        assert OutputEnum.table

    def test_output_enum_name_attribute(self):
        """Test that OutputEnum members have name attribute"""
        assert OutputEnum.json.name == "json"
        assert OutputEnum.yaml.name == "yaml"
        assert OutputEnum.table.name == "table"


class TestDisplayTable:
    """Tests for display_table function"""

    def test_display_table_basic(self):
        """Test basic table display"""
        data = [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}]
        col_names = ["name", "age"]
        result = display_table(data, col_names)
        assert "Alice" in result
        assert "Bob" in result
        assert "25" in result
        assert "30" in result

    def test_display_table_empty_data(self):
        """Test display_table with empty data"""
        result = display_table([], ["name", "age"])
        assert result == ""

    def test_display_table_missing_keys(self):
        """Test display_table with missing keys in data"""
        data = [{"name": "Alice", "age": 25}, {"name": "Bob"}]  # missing age
        col_names = ["name", "age"]
        result = display_table(data, col_names)
        assert "Alice" in result
        assert "Bob" in result

    def test_display_table_extra_keys(self):
        """Test display_table ignores extra keys not in col_names"""
        data = [{"name": "Alice", "age": 25, "extra": "ignored"}]
        col_names = ["name", "age"]
        result = display_table(data, col_names)
        assert "Alice" in result
        assert "25" in result
        assert "ignored" not in result


class TestFormatOutput:
    """Tests for format_output function"""

    def test_format_output_json_dict(self):
        """Test formatting dict as JSON"""
        data = {"name": "Alice", "age": 25}
        result = format_output(data, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed == data

    def test_format_output_json_list(self):
        """Test formatting list as JSON"""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = format_output(data, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed == data

    def test_format_output_yaml_dict(self):
        """Test formatting dict as YAML"""
        data = {"name": "Alice", "age": 25}
        result = format_output(data, OutputEnum.yaml)
        parsed = yaml.safe_load(result)
        assert parsed == data

    def test_format_output_yaml_list(self):
        """Test formatting list as YAML"""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = format_output(data, OutputEnum.yaml)
        parsed = yaml.safe_load(result)
        assert parsed == data

    def test_format_output_invalid_format(self):
        """Test format_output with invalid format raises ValueError"""
        data = {"test": "data"}
        with pytest.raises(ValueError, match="Unknown output format"):
            format_output(data, OutputEnum.table)


class TestOutputJson:
    """Tests for output_json function"""

    def test_output_json_with_dict(self):
        """Test output_json with dict input"""
        data = {"name": "Alice", "age": 25}
        result = output_json(data, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed == data

    def test_output_json_with_list(self):
        """Test output_json with list input"""
        data = [{"name": "Alice"}, {"name": "Bob"}]
        result = output_json(data, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed == data

    def test_output_json_with_json_string(self):
        """Test output_json with JSON string input"""
        data = {"name": "Alice", "age": 25}
        json_str = json.dumps(data)
        result = output_json(json_str, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed == data

    def test_output_json_yaml_format(self):
        """Test output_json with YAML output format"""
        data = {"name": "Alice", "age": 25}
        result = output_json(data, OutputEnum.yaml)
        parsed = yaml.safe_load(result)
        assert parsed == data

    def test_output_json_invalid_json_string(self):
        """Test output_json with invalid JSON string raises error"""
        with pytest.raises(json.JSONDecodeError):
            output_json("not valid json", OutputEnum.json)


class TestOutputPydanticList:
    """Tests for output_pydantic_list function"""

    def setup_method(self):
        """Set up test fixtures"""

        class User(BaseModel):
            name: str
            age: int

        self.User = User
        self.users = [User(name="Alice", age=25), User(name="Bob", age=30)]

    def test_output_pydantic_list_json(self):
        """Test output_pydantic_list with JSON format"""
        result = output_pydantic_list(self.users, OutputEnum.json)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Alice"
        assert parsed[1]["name"] == "Bob"

    def test_output_pydantic_list_yaml(self):
        """Test output_pydantic_list with YAML format"""
        result = output_pydantic_list(self.users, OutputEnum.yaml)
        parsed = yaml.safe_load(result)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Alice"

    def test_output_pydantic_list_table(self):
        """Test output_pydantic_list with table format"""
        result = output_pydantic_list(self.users, OutputEnum.table, col_names=["name", "age"])
        assert "Alice" in result
        assert "Bob" in result
        assert "25" in result
        assert "30" in result

    def test_output_pydantic_list_table_no_col_names(self):
        """Test output_pydantic_list table format without col_names raises error"""
        with pytest.raises(ValueError, match="Table output requires column names"):
            output_pydantic_list(self.users, OutputEnum.table)

    def test_output_pydantic_list_empty(self):
        """Test output_pydantic_list with empty list"""
        result = output_pydantic_list([], OutputEnum.json)
        assert result == "[]"


class TestOutputPydanticSingle:
    """Tests for output_pydantic_single function"""

    def setup_method(self):
        """Set up test fixtures"""

        class User(BaseModel):
            name: str
            age: int

        self.User = User
        self.user = User(name="Alice", age=25)

    def test_output_pydantic_single_json(self):
        """Test output_pydantic_single with JSON format"""
        result = output_pydantic_single(self.user, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed["name"] == "Alice"
        assert parsed["age"] == 25

    def test_output_pydantic_single_yaml(self):
        """Test output_pydantic_single with YAML format"""
        result = output_pydantic_single(self.user, OutputEnum.yaml)
        parsed = yaml.safe_load(result)
        assert parsed["name"] == "Alice"
        assert parsed["age"] == 25

    def test_output_pydantic_single_table(self):
        """Test output_pydantic_single with table format"""
        result = output_pydantic_single(self.user, OutputEnum.table, col_names=["name", "age"])
        assert "Alice" in result
        assert "25" in result

    def test_output_pydantic_single_table_no_col_names(self):
        """Test output_pydantic_single table format without col_names raises error"""
        with pytest.raises(ValueError, match="Table output requires column names"):
            output_pydantic_single(self.user, OutputEnum.table)


class TestOutputPydantic:
    """Tests for output_pydantic function"""

    def setup_method(self):
        """Set up test fixtures"""

        class User(BaseModel):
            name: str
            age: int

        self.User = User
        self.user = User(name="Alice", age=25)
        self.users = [User(name="Alice", age=25), User(name="Bob", age=30)]

    def test_output_pydantic_single_model_json(self):
        """Test output_pydantic with single model and JSON format"""
        result = output_pydantic(self.user, OutputEnum.json)
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "Alice"

    def test_output_pydantic_list_models_json(self):
        """Test output_pydantic with list of models and JSON format"""
        result = output_pydantic(self.users, OutputEnum.json)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Alice"
        assert parsed[1]["name"] == "Bob"

    def test_output_pydantic_single_model_yaml(self):
        """Test output_pydantic with single model and YAML format"""
        result = output_pydantic(self.user, OutputEnum.yaml)
        parsed = yaml.safe_load(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "Alice"

    def test_output_pydantic_list_models_yaml(self):
        """Test output_pydantic with list of models and YAML format"""
        result = output_pydantic(self.users, OutputEnum.yaml)
        parsed = yaml.safe_load(result)
        assert len(parsed) == 2

    def test_output_pydantic_single_model_table(self):
        """Test output_pydantic with single model and table format"""
        result = output_pydantic(self.user, OutputEnum.table, col_names=["name", "age"])
        assert "Alice" in result
        assert "25" in result

    def test_output_pydantic_list_models_table(self):
        """Test output_pydantic with list of models and table format"""
        result = output_pydantic(self.users, OutputEnum.table, col_names=["name", "age"])
        assert "Alice" in result
        assert "Bob" in result

    def test_output_pydantic_table_no_col_names(self):
        """Test output_pydantic table format without col_names raises error"""
        with pytest.raises(ValueError, match="Table output requires column names"):
            output_pydantic(self.user, OutputEnum.table)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_format_output_nested_structures(self):
        """Test formatting deeply nested data structures"""
        data = {
            "users": [
                {"name": "Alice", "metadata": {"role": "admin"}},
                {"name": "Bob", "metadata": {"role": "user"}},
            ]
        }
        result = format_output(data, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed == data

    def test_output_pydantic_with_optional_fields(self):
        """Test output_pydantic with models containing optional fields"""

        class User(BaseModel):
            name: str
            age: int | None = None

        user = User(name="Alice")
        result = output_pydantic(user, OutputEnum.json)
        parsed = json.loads(result)
        assert parsed[0]["name"] == "Alice"
        assert parsed[0]["age"] is None

    def test_display_table_with_numeric_values(self):
        """Test display_table with various numeric types"""
        data = [{"int": 42, "float": 3.14, "bool": True}]
        col_names = ["int", "float", "bool"]
        result = display_table(data, col_names)
        assert "42" in result
        assert "3.14" in result

    def test_output_json_with_empty_dict(self):
        """Test output_json with empty dict"""
        result = output_json({}, OutputEnum.json)
        assert result == "{}"

    def test_output_json_with_empty_list(self):
        """Test output_json with empty list"""
        result = output_json([], OutputEnum.json)
        assert result == "[]"
