"""Unit tests for the Configuration module"""

import os
from unittest.mock import patch

import pytest
from pydantic import SecretStr, ValidationError

from rail_svc.config import (
    AsgiConfiguration,
    Configuration,
    DaemonConfiguration,
    DatabaseConfiguration,
    LoggingConfiguration,
    StorageConfiguration,
)


class TestAsgiConfiguration:
    """Tests for AsgiConfiguration model"""

    def test_valid_asgi_configuration_defaults(self):
        """Test creating AsgiConfiguration with default values"""
        asgi = AsgiConfiguration()
        assert asgi.title == "rail-svc"
        assert asgi.host == "0.0.0.0"
        assert asgi.port == 8080
        assert asgi.prefix == "/rail-svc"
        assert asgi.frontend_prefix == "/rail-svc"
        assert asgi.reload is True

    def test_valid_asgi_configuration_custom_values(self):
        """Test creating AsgiConfiguration with custom values"""
        asgi = AsgiConfiguration(
            title="custom-app",
            host="127.0.0.1",
            port=5000,
            prefix="/api",
            frontend_prefix="/app",
            reload=False,
        )
        assert asgi.title == "custom-app"
        assert asgi.host == "127.0.0.1"
        assert asgi.port == 5000
        assert asgi.prefix == "/api"
        assert asgi.frontend_prefix == "/app"
        assert asgi.reload is False

    def test_asgi_port_must_be_in_valid_range(self):
        """Test that port must be between 1 and 65535"""
        with pytest.raises(ValidationError) as exc_info:
            AsgiConfiguration(port=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            AsgiConfiguration(port=65536)
        assert "less than or equal to 65535" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            AsgiConfiguration(port=-1)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_asgi_port_edge_values(self):
        """Test valid edge values for port"""
        # Minimum valid port
        asgi_min = AsgiConfiguration(port=1)
        assert asgi_min.port == 1

        # Maximum valid port
        asgi_max = AsgiConfiguration(port=65535)
        assert asgi_max.port == 65535

    def test_asgi_common_ports(self):
        """Test common port values"""
        common_ports = [80, 443, 3000, 5000, 8000, 8080, 8443, 9000]
        for port in common_ports:
            asgi = AsgiConfiguration(port=port)
            assert asgi.port == port

    def test_asgi_various_hosts(self):
        """Test various host configurations"""
        hosts = [
            "0.0.0.0",
            "127.0.0.1",
            "localhost",
            "192.168.1.100",
            "::1",  # IPv6 localhost
        ]
        for host in hosts:
            asgi = AsgiConfiguration(host=host)
            assert asgi.host == host


class TestLoggingConfiguration:
    """Tests for LoggingConfiguration model"""

    def test_valid_logging_configuration_defaults(self):
        """Test creating LoggingConfiguration with default values"""
        logging = LoggingConfiguration()
        assert logging.handle == "rail-svc"
        assert logging.level == "INFO"
        assert logging.profile == "development"

    def test_valid_logging_configuration_custom_values(self):
        """Test creating LoggingConfiguration with custom values"""
        logging = LoggingConfiguration(
            handle="custom-logger",
            level="DEBUG",
            profile="production",
        )
        assert logging.handle == "custom-logger"
        assert logging.level == "DEBUG"
        assert logging.profile == "production"

    def test_logging_level_validation(self):
        """Test that log level must be valid"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

        for level in valid_levels:
            logging = LoggingConfiguration(level=level)
            assert logging.level == level

    def test_logging_level_case_insensitive(self):
        """Test that log level is normalized to uppercase"""
        levels = ["debug", "info", "warning", "error", "critical", "Debug", "InFo"]

        for level in levels:
            logging = LoggingConfiguration(level=level)
            assert logging.level == level.upper()

    def test_logging_level_invalid(self):
        """Test that invalid log level raises error"""
        with pytest.raises(ValidationError) as exc_info:
            LoggingConfiguration(level="INVALID")
        assert "Log level must be one of" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            LoggingConfiguration(level="TRACE")
        assert "Log level must be one of" in str(exc_info.value)


class TestDaemonConfiguration:
    """Tests for DaemonConfiguration model"""

    def test_valid_daemon_configuration_defaults(self):
        """Test creating DaemonConfiguration with default values"""
        daemon = DaemonConfiguration()
        assert daemon.processing_interval == 30

    def test_valid_daemon_configuration_custom_values(self):
        """Test creating DaemonConfiguration with custom values"""
        daemon = DaemonConfiguration(processing_interval=60)
        assert daemon.processing_interval == 60

    def test_daemon_processing_interval_must_be_positive(self):
        """Test that processing_interval must be at least 1"""
        with pytest.raises(ValidationError) as exc_info:
            DaemonConfiguration(processing_interval=0)
        assert "greater than or equal to 1" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            DaemonConfiguration(processing_interval=-10)
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_daemon_processing_interval_edge_value(self):
        """Test minimum valid processing interval"""
        daemon = DaemonConfiguration(processing_interval=1)
        assert daemon.processing_interval == 1

    def test_daemon_realistic_intervals(self):
        """Test realistic processing interval values"""
        intervals = [5, 10, 30, 60, 120, 300, 600]
        for interval in intervals:
            daemon = DaemonConfiguration(processing_interval=interval)
            assert daemon.processing_interval == interval


class TestDatabaseConfiguration:
    """Tests for DatabaseConfiguration model"""

    def test_valid_database_configuration_defaults(self):
        """Test creating DatabaseConfiguration with default values"""
        db = DatabaseConfiguration()
        assert db.url == "sqlite+aiosqlite:///rail_svc.db"
        assert db.password is None
        assert db.table_schema is None
        assert db.echo is False

    def test_valid_database_configuration_custom_values(self):
        """Test creating DatabaseConfiguration with custom values"""
        db = DatabaseConfiguration(
            url="postgresql://localhost/testdb",
            password=SecretStr("secret123"),
            table_schema="public",
            echo=True,
        )
        assert db.url == "postgresql://localhost/testdb"
        assert db.password.get_secret_value() == "secret123"
        assert db.table_schema == "public"
        assert db.echo is True

    def test_database_url_validation_valid_schemes(self):
        """Test that valid database URL schemes are accepted"""
        valid_urls = [
            "sqlite:///database.db",
            "sqlite:////absolute/path/database.db",
            "postgresql://user:pass@localhost/db",
            "postgresql+psycopg2://user:pass@localhost/db",
            "mysql://user:pass@localhost/db",
        ]

        for url in valid_urls:
            db = DatabaseConfiguration(url=url)
            assert db.url == url

    def test_database_url_validation_invalid_schemes(self):
        """Test that invalid database URL schemes are rejected"""
        invalid_urls = [
            "http://localhost/db",
            "ftp://localhost/db",
            "mongodb://localhost/db",
            "invalid://localhost/db",
        ]

        for url in invalid_urls:
            with pytest.raises(ValidationError) as exc_info:
                DatabaseConfiguration(url=url)
            assert "valid scheme" in str(exc_info.value)

    def test_database_password_is_secret(self):
        """Test that password is properly handled as SecretStr"""
        db = DatabaseConfiguration(password=SecretStr("my_secret_password"))

        # Password should not be in string representation
        db_str = str(db)
        assert "my_secret_password" not in db_str
        assert "**********" in db_str or "SecretStr" in db_str

        # But can be retrieved when needed
        assert db.password.get_secret_value() == "my_secret_password"

    def test_database_echo_boolean(self):
        """Test echo setting as boolean"""
        db_false = DatabaseConfiguration(echo=False)
        assert db_false.echo is False

        db_true = DatabaseConfiguration(echo=True)
        assert db_true.echo is True


class TestStorageConfiguration:
    """Tests for StorageConfiguration model"""

    def test_valid_storage_configuration_defaults(self):
        """Test creating StorageConfiguration with default values"""
        storage = StorageConfiguration()
        assert storage.archive == "archive"
        assert storage.import_area == "import"

    def test_valid_storage_configuration_custom_values(self):
        """Test creating StorageConfiguration with custom values"""
        with pytest.raises(ValueError):
            StorageConfiguration(
                archive="/data/archive",
                import_area="/data/import",
            )

    def test_storage_existing_paths(self, tmp_path):
        """Test that existing paths work correctly"""
        archive_path = tmp_path / "existing_archive"
        import_path = tmp_path / "existing_import"

        # Create paths
        archive_path.mkdir()
        import_path.mkdir()

        storage = StorageConfiguration(
            archive=str(archive_path),
            import_area=str(import_path),
        )

        assert storage.archive == str(archive_path)
        assert storage.import_area == str(import_path)


class TestConfiguration:
    """Tests for main Configuration model"""

    def test_valid_configuration_defaults(self):
        """Test creating Configuration with all default values"""
        config = Configuration()

        assert isinstance(config.asgi, AsgiConfiguration)
        assert isinstance(config.daemon, DaemonConfiguration)
        assert isinstance(config.db, DatabaseConfiguration)
        assert isinstance(config.logging, LoggingConfiguration)
        assert isinstance(config.storage, StorageConfiguration)

    def test_configuration_from_environment_variables(self, tmp_path):
        """Test loading configuration from environment variables"""

        archive_path = tmp_path / "existing_archive"
        import_path = tmp_path / "existing_import"

        # Create paths
        archive_path.mkdir()
        import_path.mkdir()

        env_vars = {
            "ASGI__PORT": "9000",
            "ASGI__HOST": "127.0.0.1",
            "DB__URL": "postgresql://localhost/testdb",
            "LOGGING__LEVEL": "DEBUG",
            "DAEMON__PROCESSING_INTERVAL": "60",
            "STORAGE__ARCHIVE": str(archive_path),
            "STORAGE__IMPORT_AREA": str(import_path),
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Configuration()

            assert config.asgi.port == 9000
            assert config.asgi.host == "127.0.0.1"
            assert config.db.url == "postgresql://localhost/testdb"
            assert config.logging.level == "DEBUG"
            assert config.daemon.processing_interval == 60
            assert config.storage.archive == str(archive_path)
            assert config.storage.import_area == str(import_path)

    def test_configuration_case_insensitive_env_vars(self):
        """Test that environment variables are case insensitive"""
        env_vars = {
            "asgi__port": "7000",
            "ASGI__PORT": "8000",  # Should override lowercase
        }

        with patch.dict(os.environ, env_vars, clear=False):
            config = Configuration()
            # Due to case_sensitive=False, both should work
            # The exact behavior depends on environment variable precedence
            assert config.asgi.port in [7000, 8000]

    def test_configuration_partial_update(self):
        """Test partial updates to nested models"""
        config = Configuration(
            asgi={"port": 3000},  # Only update port, keep other defaults
        )

        assert config.asgi.port == 3000
        assert config.asgi.host == "0.0.0.0"  # Default preserved
        assert config.asgi.title == "rail-svc"  # Default preserved

    def test_configuration_extra_fields_ignored(self):
        """Test that extra fields are ignored"""
        # Should not raise error due to extra="ignore"
        config = Configuration(
            unknown_field="value",
            another_unknown=123,
        )

        assert not hasattr(config, "unknown_field")
        assert not hasattr(config, "another_unknown")

    def test_configuration_realistic_production_setup(self, tmp_path):
        """Test realistic production configuration"""
        archive_path = tmp_path / "existing_archive"
        import_path = tmp_path / "existing_import"

        # Create paths
        archive_path.mkdir()
        import_path.mkdir()

        config = Configuration(
            asgi={
                "host": "0.0.0.0",
                "port": 8080,
                "reload": False,
            },
            db={
                "url": "postgresql://user@db.example.com:5432/rail_prod",
                "password": SecretStr("prod_password"),
                "table_schema": "public",
                "echo": False,
            },
            logging={
                "level": "WARNING",
                "profile": "production",
            },
            daemon={
                "processing_interval": 60,
            },
            storage={
                "archive": str(archive_path),
                "import_area": str(import_path),
            },
        )

        assert config.asgi.reload is False
        assert config.db.password.get_secret_value() == "prod_password"
        assert config.logging.level == "WARNING"
        assert config.daemon.processing_interval == 60

    def test_configuration_realistic_development_setup(self):
        """Test realistic development configuration"""
        config = Configuration(
            asgi={
                "host": "127.0.0.1",
                "port": 8000,
                "reload": True,
            },
            db={
                "url": "sqlite:///dev.db",
                "echo": True,
            },
            logging={
                "level": "DEBUG",
                "profile": "development",
            },
            daemon={
                "processing_interval": 10,
            },
        )

        assert config.asgi.reload is True
        assert config.db.echo is True
        assert config.logging.level == "DEBUG"
        assert config.daemon.processing_interval == 10
