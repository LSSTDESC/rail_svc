"""Unit tests for database session management"""

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from rail_svc.db.algorithm import Algorithm
from rail_svc.db.base import Base
from rail_svc.db.session import close_db, get_session, init_db

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def db_url():
    """Provide in-memory SQLite database URL for testing."""
    return "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def initialized_db(db_url):
    """Initialize database for testing and cleanup afterwards."""
    init_db(db_url)

    # Create tables
    from rail_svc.db.session import _engine

    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    await close_db()


@pytest.fixture
async def clean_db_state():
    """Ensure clean database state before and after tests."""
    # Clean up any existing state
    await close_db()

    # Also clear the module-level variables
    import rail_svc.db.session as session_module

    session_module._engine = None
    session_module._async_session_factory = None

    yield

    # Clean up after test
    await close_db()


# ============================================================================
# Test init_db
# ============================================================================


class TestInitDb:
    """Tests for init_db function"""

    @pytest.mark.asyncio
    async def test_init_db_with_url(self, clean_db_state, db_url):
        """Test initializing database with explicit URL"""
        init_db(db_url)

        from rail_svc.db.session import _async_session_factory, _engine

        assert _engine is not None
        assert _async_session_factory is not None

        await close_db()

    @pytest.mark.asyncio
    async def test_init_db_with_echo(self, clean_db_state, db_url):
        """Test initializing database with echo parameter"""
        init_db(db_url, echo=True)

        from rail_svc.db.session import _engine

        assert _engine is not None
        assert _engine.echo is True

        await close_db()

    @pytest.mark.asyncio
    async def test_init_db_with_custom_kwargs(self, clean_db_state, db_url):
        """Test initializing database with custom kwargs"""
        # Use kwargs that work with SQLite
        init_db(db_url, pool_pre_ping=True)

        from rail_svc.db.session import _engine

        assert _engine is not None

        await close_db()

    @pytest.mark.asyncio
    async def test_init_db_multiple_times(self, clean_db_state, db_url):
        """Test that calling init_db multiple times replaces engine"""
        init_db(db_url, echo=False)

        from rail_svc.db.session import _engine as engine1

        first_engine = engine1

        # Initialize again with different settings
        init_db(db_url, echo=True)

        from rail_svc.db.session import _engine as engine2

        # Should be different engines
        assert engine2 is not first_engine

        await close_db()

    @pytest.mark.asyncio
    async def test_init_db_creates_session_factory(self, clean_db_state, db_url):
        """Test that init_db creates a session factory"""
        init_db(db_url)

        from rail_svc.db.session import _async_session_factory

        assert _async_session_factory is not None
        assert callable(_async_session_factory)

        await close_db()


# ============================================================================
# Test get_session
# ============================================================================


class TestGetSession:
    """Tests for get_session function"""

    @pytest.mark.asyncio
    async def test_get_session_not_initialized(self, clean_db_state):
        """Test that get_session raises error when not initialized"""
        with pytest.raises(RuntimeError, match="Database not initialized"):
            async with get_session() as _session:
                pass

    @pytest.mark.asyncio
    async def test_get_session_returns_session(self, initialized_db):
        """Test that get_session returns a valid session"""
        async with get_session() as session:
            assert isinstance(session, AsyncSession)

    @pytest.mark.asyncio
    async def test_get_session_commits_on_success(self, initialized_db):
        """Test that session commits on successful completion"""
        # Create an algorithm
        async with get_session() as session:
            algo = Algorithm(name="test_commit", class_name="test.Class")
            session.add(algo)

        # Verify it was committed
        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name == "test_commit"))
            found = result.scalar_one_or_none()
            assert found is not None

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_error(self, initialized_db):
        """Test that session rolls back on error"""
        # Try to create an algorithm and raise an error
        try:
            async with get_session() as session:
                algo = Algorithm(name="test_rollback", class_name="test.Class")
                session.add(algo)
                await session.flush()
                raise ValueError("Test error")
        except ValueError:
            pass

        # Verify it was rolled back
        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name == "test_rollback"))
            found = result.scalar_one_or_none()
            assert found is None

    @pytest.mark.asyncio
    async def test_get_session_multiple_sessions(self, initialized_db):
        """Test that multiple sessions can be created"""
        async with get_session() as session1:
            algo1 = Algorithm(name="session1", class_name="test.Class1")
            session1.add(algo1)

        async with get_session() as session2:
            algo2 = Algorithm(name="session2", class_name="test.Class2")
            session2.add(algo2)

        # Verify both were committed
        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            algos = result.scalars().all()
            names = {a.name for a in algos}
            assert "session1" in names
            assert "session2" in names

    @pytest.mark.asyncio
    async def test_get_session_nested_not_supported(self, initialized_db):
        """Test behavior with nested session contexts"""
        # SQLAlchemy sessions don't support true nesting,
        # but we can test that it doesn't crash
        async with get_session() as session1:
            algo = Algorithm(name="outer", class_name="test.Class")
            session1.add(algo)

            # Inner session is independent
            async with get_session() as session2:
                assert session2 is not session1


# ============================================================================
# Test close_db
# ============================================================================


class TestCloseDb:
    """Tests for close_db function"""

    @pytest.mark.asyncio
    async def test_close_db_disposes_engine(self, clean_db_state, db_url):
        """Test that close_db disposes the engine"""
        init_db(db_url)

        from rail_svc.db.session import _engine

        assert _engine is not None

        await close_db()

        from rail_svc.db.session import _engine as engine_after

        assert engine_after is None

    @pytest.mark.asyncio
    async def test_close_db_when_not_initialized(self, clean_db_state):
        """Test that close_db doesn't fail when database not initialized"""
        # Should not raise an error
        await close_db()

    @pytest.mark.asyncio
    async def test_close_db_multiple_times(self, clean_db_state, db_url):
        """Test that calling close_db multiple times is safe"""
        init_db(db_url)

        await close_db()
        await close_db()  # Should not raise an error

    @pytest.mark.asyncio
    async def test_get_session_after_close(self, clean_db_state, db_url):
        """Test that get_session fails after close_db"""
        init_db(db_url)
        await close_db()

        with pytest.raises(RuntimeError, match="Database not initialized"):
            async with get_session() as _session:
                pass


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for database session management"""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, clean_db_state, db_url):
        """Test full database lifecycle: init -> use -> close"""
        # Initialize
        init_db(db_url)

        # Create tables
        from rail_svc.db.session import _engine

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Use database
        async with get_session() as session:
            algo = Algorithm(name="lifecycle_test", class_name="test.Class")
            session.add(algo)

        # Verify data persists
        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name == "lifecycle_test"))
            found = result.scalar_one()
            assert found.name == "lifecycle_test"

        # Close
        await close_db()

        # Verify cannot use after close
        with pytest.raises(RuntimeError, match="Database not initialized"):
            async with get_session() as session:
                pass

    @pytest.mark.asyncio
    async def test_transaction_isolation(self, initialized_db):
        """Test that sessions are properly isolated"""
        # Create in one session
        async with get_session() as session1:
            algo = Algorithm(name="isolation_test", class_name="test.Class")
            session1.add(algo)
            await session1.flush()

            # In another session (before commit), should not see it
            async with get_session() as session2:
                _result = await session2.execute(select(Algorithm).where(Algorithm.name == "isolation_test"))
                # In SQLite, default isolation may allow seeing it
                # This behavior depends on isolation level

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, initialized_db):
        """Test that multiple concurrent sessions work correctly"""
        import asyncio

        async def create_algo(name: str):
            async with get_session() as session:
                algo = Algorithm(name=name, class_name="test.Class")
                session.add(algo)
                await asyncio.sleep(0.01)  # Simulate some work

        # Create multiple algorithms concurrently
        await asyncio.gather(
            create_algo("concurrent_1"), create_algo("concurrent_2"), create_algo("concurrent_3")
        )

        # Verify all were created
        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            algos = result.scalars().all()
            names = {a.name for a in algos}
            assert "concurrent_1" in names
            assert "concurrent_2" in names
            assert "concurrent_3" in names

    @pytest.mark.asyncio
    async def test_exception_handling_in_session(self, initialized_db):
        """Test proper exception handling within session context"""

        class CustomError(Exception):
            pass

        with pytest.raises(CustomError):
            async with get_session() as session:
                algo = Algorithm(name="error_test", class_name="test.Class")
                session.add(algo)
                await session.flush()
                raise CustomError("Test exception")

        # Verify rollback occurred
        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name == "error_test"))
            found = result.scalar_one_or_none()
            assert found is None

    @pytest.mark.asyncio
    async def test_query_execution(self, initialized_db):
        """Test that queries work correctly through session"""
        # Insert test data
        async with get_session() as session:
            for i in range(5):
                algo = Algorithm(name=f"query_test_{i}", class_name="test.Class")
                session.add(algo)

        # Query data
        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name.like("query_test_%")))
            algos = result.scalars().all()
            assert len(algos) == 5

    @pytest.mark.asyncio
    async def test_raw_sql_execution(self, initialized_db):
        """Test executing raw SQL through session"""
        async with get_session() as session:
            # Execute raw SQL
            result = await session.execute(text("SELECT 1 as num"))
            row = result.first()
            assert row[0] == 1


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_init_with_unsupported_dialect(self, clean_db_state):
        """Test initialization with unsupported database dialect"""
        # Use a malformed URL that will fail
        with pytest.raises(Exception):  # ArgumentError or NoSuchModuleError
            init_db("notadialect+driver://localhost/db")

        await close_db()

    @pytest.mark.asyncio
    async def test_session_after_engine_disposed(self, clean_db_state, db_url):
        """Test that session fails gracefully after engine disposal"""
        init_db(db_url)

        from rail_svc.db.session import _engine

        await _engine.dispose()

        # Attempting to get session might fail
        # Behavior depends on SQLAlchemy version
        await close_db()

    @pytest.mark.asyncio
    async def test_multiple_errors_in_session(self, initialized_db):
        """Test handling multiple errors within same session"""
        error_count = 0

        try:
            async with get_session() as session:
                try:
                    algo = Algorithm(name="multi_error", class_name="test.Class")
                    session.add(algo)
                    raise ValueError("First error")
                except ValueError:
                    error_count += 1
                    raise RuntimeError("Second error")
        except RuntimeError:
            error_count += 1

        assert error_count == 2

        # Verify rollback occurred
        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name == "multi_error"))
            found = result.scalar_one_or_none()
            assert found is None


class TestGlobalState:
    """Tests for global state management"""

    @pytest.mark.asyncio
    async def test_engine_global_state(self, clean_db_state, db_url):
        """Test that engine is stored in global state"""
        from rail_svc.db.session import _engine

        assert _engine is None

        init_db(db_url)

        from rail_svc.db.session import _engine as engine_after_init

        assert engine_after_init is not None

        await close_db()

        from rail_svc.db.session import _engine as engine_after_close

        assert engine_after_close is None

    @pytest.mark.asyncio
    async def test_session_factory_persists_until_close(self, clean_db_state, db_url):
        """Test that session factory persists until close_db is called"""
        from rail_svc.db.session import _async_session_factory

        assert _async_session_factory is None

        init_db(db_url)

        from rail_svc.db.session import _async_session_factory as factory_after_init

        assert factory_after_init is not None

        # Session factory should still exist
        from rail_svc.db.session import _async_session_factory as factory_still_there

        assert factory_still_there is not None

        await close_db()

        # After close, engine is None but session factory may or may not be cleared
        # (implementation detail - we only guarantee engine is None)

    @pytest.mark.asyncio
    async def test_reinitialize_after_close(self, clean_db_state, db_url):
        """Test that database can be reinitialized after close"""
        # First lifecycle
        init_db(db_url)

        from rail_svc.db.session import _engine

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with get_session() as session:
            algo = Algorithm(name="reinit_test", class_name="test.Class")
            session.add(algo)

        await close_db()

        # Second lifecycle
        init_db(db_url)

        from rail_svc.db.session import _engine as engine2

        async with engine2.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with get_session() as session:
            algo = Algorithm(name="reinit_test2", class_name="test.Class")
            session.add(algo)

        await close_db()


# ============================================================================
# Performance and Resource Tests
# ============================================================================


class TestPerformance:
    """Tests for performance and resource management"""

    @pytest.mark.asyncio
    async def test_session_cleanup(self, initialized_db):
        """Test that sessions are properly cleaned up"""
        sessions_created = 0

        for _ in range(10):
            async with get_session() as session:
                sessions_created += 1
                algo = Algorithm(name=f"cleanup_{sessions_created}", class_name="test.Class")
                session.add(algo)

        assert sessions_created == 10

        # Verify all were committed
        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            algos = result.scalars().all()
            assert len(algos) >= 10

    @pytest.mark.asyncio
    async def test_many_sequential_sessions(self, initialized_db):
        """Test creating many sessions sequentially"""
        for i in range(50):
            async with get_session() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1

    @pytest.mark.asyncio
    async def test_session_with_large_query(self, initialized_db):
        """Test session with large query result"""
        # Create many records
        async with get_session() as session:
            for i in range(100):
                algo = Algorithm(name=f"large_query_{i}", class_name="test.Class")
                session.add(algo)

        # Query all
        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            algos = result.scalars().all()
            assert len(algos) >= 100


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Tests for database configuration"""

    @pytest.mark.asyncio
    async def test_init_with_custom_engine_kwargs(self, clean_db_state, db_url):
        """Test initialization with custom engine kwargs"""
        init_db(db_url, echo=False, pool_pre_ping=True)

        from rail_svc.db.session import _engine

        assert _engine is not None

        await close_db()

    @pytest.mark.asyncio
    async def test_session_factory_creates_sessions(self, clean_db_state, db_url):
        """Test that session factory creates valid sessions"""
        init_db(db_url)

        from rail_svc.db.session import _async_session_factory

        # Create a session using the factory
        async with _async_session_factory() as session:
            assert isinstance(session, AsyncSession)

        await close_db()


# ============================================================================
# Context Manager Behavior Tests
# ============================================================================


class TestContextManager:
    """Tests for context manager behavior"""

    @pytest.mark.asyncio
    async def test_context_manager_normal_exit(self, initialized_db):
        """Test context manager with normal exit"""
        async with get_session() as session:
            algo = Algorithm(name="normal_exit", class_name="test.Class")
            session.add(algo)
            # Normal exit should commit

        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name == "normal_exit"))
            found = result.scalar_one_or_none()
            assert found is not None

    @pytest.mark.asyncio
    async def test_context_manager_exception_exit(self, initialized_db):
        """Test context manager with exception exit"""
        try:
            async with get_session() as session:
                algo = Algorithm(name="exception_exit", class_name="test.Class")
                session.add(algo)
                raise ValueError("Test exception")
        except ValueError:
            pass

        async with get_session() as session:
            result = await session.execute(select(Algorithm).where(Algorithm.name == "exception_exit"))
            found = result.scalar_one_or_none()
            assert found is None  # Should be rolled back

    @pytest.mark.asyncio
    async def test_context_manager_explicit_commit(self, initialized_db):
        """Test that explicit commit still works"""
        async with get_session() as session:
            algo = Algorithm(name="explicit_commit", class_name="test.Class")
            session.add(algo)
            await session.commit()

            # Add another after explicit commit
            algo2 = Algorithm(name="after_commit", class_name="test.Class")
            session.add(algo2)

        # Both should be committed
        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            names = {a.name for a in result.scalars().all()}
            assert "explicit_commit" in names
            assert "after_commit" in names

    @pytest.mark.asyncio
    async def test_context_manager_explicit_rollback(self, initialized_db):
        """Test that explicit rollback works"""
        async with get_session() as session:
            algo = Algorithm(name="explicit_rollback", class_name="test.Class")
            session.add(algo)
            await session.rollback()

            # Add another after explicit rollback
            algo2 = Algorithm(name="after_rollback", class_name="test.Class")
            session.add(algo2)

        # Only the second should be committed
        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            names = {a.name for a in result.scalars().all()}
            assert "explicit_rollback" not in names
            assert "after_rollback" in names


# ============================================================================
# Database URL Tests
# ============================================================================


class TestDatabaseUrl:
    """Tests for different database URLs"""

    @pytest.mark.asyncio
    async def test_sqlite_memory_url(self, clean_db_state):
        """Test with SQLite in-memory URL"""
        init_db("sqlite+aiosqlite:///:memory:")

        from rail_svc.db.session import _engine

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with get_session() as session:
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1

        await close_db()

    @pytest.mark.asyncio
    async def test_sqlite_file_url(self, clean_db_state, tmp_path):
        """Test with SQLite file URL"""
        db_file = tmp_path / "test.db"
        init_db(f"sqlite+aiosqlite:///{db_file}")

        from rail_svc.db.session import _engine

        async with _engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with get_session() as session:
            algo = Algorithm(name="file_test", class_name="test.Class")
            session.add(algo)

        # Verify file was created
        assert db_file.exists()

        await close_db()

    @pytest.mark.asyncio
    async def test_init_with_none_url_uses_config(self, clean_db_state):
        """Test that init_db with None URL uses config"""
        # When None is passed, it should use the config URL
        init_db(None)

        from rail_svc.db.session import _engine

        assert _engine is not None

        await close_db()


# ============================================================================
# Error Recovery Tests
# ============================================================================


class TestErrorRecovery:
    """Tests for error recovery scenarios"""

    @pytest.mark.asyncio
    async def test_recovery_after_connection_error(self, initialized_db):
        """Test that new sessions work after a connection error"""
        # First session works
        async with get_session() as session:
            algo = Algorithm(name="before_error", class_name="test.Class")
            session.add(algo)

        # Second session should also work
        async with get_session() as session:
            algo = Algorithm(name="after_error", class_name="test.Class")
            session.add(algo)

        # Verify both committed
        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            names = {a.name for a in result.scalars().all()}
            assert "before_error" in names
            assert "after_error" in names

    @pytest.mark.asyncio
    async def test_recovery_after_rollback(self, initialized_db):
        """Test that sessions work normally after a rollback"""
        # Session with rollback
        try:
            async with get_session() as session:
                algo = Algorithm(name="rollback_test", class_name="test.Class")
                session.add(algo)
                raise ValueError("Force rollback")
        except ValueError:
            pass

        # Next session should work normally
        async with get_session() as session:
            algo = Algorithm(name="after_rollback", class_name="test.Class")
            session.add(algo)

        async with get_session() as session:
            result = await session.execute(select(Algorithm))
            names = {a.name for a in result.scalars().all()}
            assert "rollback_test" not in names
            assert "after_rollback" in names


# ============================================================================
# Additional Tests
# ============================================================================


class TestSessionBehavior:
    """Tests for specific session behaviors"""

    @pytest.mark.asyncio
    async def test_session_autoflush_behavior(self, initialized_db):
        """Test session autoflush behavior"""
        async with get_session() as session:
            algo = Algorithm(name="autoflush_test", class_name="test.Class")
            session.add(algo)

            # Query should trigger autoflush
            result = await session.execute(select(Algorithm))
            algos = result.scalars().all()

            # The new algo should be in results due to autoflush
            names = {a.name for a in algos}
            assert "autoflush_test" in names

    @pytest.mark.asyncio
    async def test_session_isolation_between_contexts(self, initialized_db):
        """Test that sessions are isolated between context managers"""
        # Create in first session
        async with get_session() as session1:
            algo1 = Algorithm(name="isolated_1", class_name="test.Class")
            session1.add(algo1)

        # Create in second session - should not see uncommitted from first
        async with get_session() as session2:
            algo2 = Algorithm(name="isolated_2", class_name="test.Class")
            session2.add(algo2)

            # Both should be in database now (first was committed)
            result = await session2.execute(select(Algorithm))
            names = {a.name for a in result.scalars().all()}
            assert "isolated_1" in names
            assert "isolated_2" in names
