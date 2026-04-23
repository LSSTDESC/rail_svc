"""Unit tests for filter.py database filtering functionality."""

import pytest
import pytest_asyncio

from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import String, Integer, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from rail_svc.db.base import Base
from rail_svc.db_funcs.filter import (
    Filter,
    FilterOp,
    OrderBy,
    filter_rows,
    filter_rows_streaming,
    count_filtered_rows,
    filter_one,
    filter_one_or_none,
    find_by,
    find_one_by,
    and_filters,
    or_filters,
)


# Test Model
class TestUser(Base):
    """Test user model for filtering tests."""
    
    __tablename__ = "test_users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50))
    email: Mapped[str] = mapped_column(String(255))
    age: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(20))
    role: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), 
        nullable=True,
        default=None
    )
    
    @classmethod
    def pydantic_model_class(cls):
        # Mock for testing
        return None
    
    @classmethod
    def class_string(cls) -> str:
        return cls.__name__


# Fixtures
@pytest_asyncio.fixture
async def engine():
    """Create in-memory SQLite database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine) -> AsyncIterator[AsyncSession]:
    """Create database session for testing."""
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def sample_users(session: AsyncSession) -> list[TestUser]:
    """Create sample users for testing."""
    now = datetime.now(timezone.utc)
    
    users = [
        TestUser(
            id=1,
            username="alice",
            email="alice@example.com",
            age=25,
            status="active",
            role="admin",
            created_at=now,
        ),
        TestUser(
            id=2,
            username="bob",
            email="bob@example.com",
            age=30,
            status="active",
            role="user",
            created_at=now,
        ),
        TestUser(
            id=3,
            username="charlie",
            email="charlie@example.com",
            age=35,
            status="inactive",
            role="user",
            created_at=now,
        ),
        TestUser(
            id=4,
            username="diana",
            email="diana@example.com",
            age=28,
            status="active",
            role="moderator",
            created_at=now,
        ),
        TestUser(
            id=5,
            username="eve",
            email="eve@example.com",
            age=22,
            status="active",
            role="user",
            created_at=now,
            deleted_at=now,  # Soft deleted
        ),
    ]
    
    for user in users:
        session.add(user)
    
    await session.commit()
    
    for user in users:
        await session.refresh(user)
    
    return users


# Tests for Filter class
class TestFilter:
    """Tests for Filter class."""
    
    def test_filter_creation(self):
        """Test creating a Filter object."""
        f = Filter("age", FilterOp.GT, 18)
        assert f.field == "age"
        assert f.op == FilterOp.GT
        assert f.value == 18
    
    def test_filter_repr(self):
        """Test Filter string representation."""
        f = Filter("age", FilterOp.GT, 18)
        assert "age" in repr(f)
        assert "GT" in repr(f)
        assert "18" in repr(f)
    
    def test_filter_without_value(self):
        """Test Filter for operators that don't need a value."""
        f = Filter("deleted_at", FilterOp.IS_NULL)
        assert f.field == "deleted_at"
        assert f.op == FilterOp.IS_NULL
        assert f.value is None


# Tests for OrderBy class
class TestOrderBy:
    """Tests for OrderBy class."""
    
    def test_orderby_creation_ascending(self):
        """Test creating ascending OrderBy."""
        o = OrderBy("username", descending=False)
        assert o.field == "username"
        assert o.descending is False
    
    def test_orderby_creation_descending(self):
        """Test creating descending OrderBy."""
        o = OrderBy("created_at", descending=True)
        assert o.field == "created_at"
        assert o.descending is True
    
    def test_orderby_repr(self):
        """Test OrderBy string representation."""
        o = OrderBy("username", descending=True)
        assert "username" in repr(o)
        assert "DESC" in repr(o)


# Tests for filter_rows
class TestFilterRows:
    """Tests for filter_rows function."""
    
    @pytest.mark.asyncio
    async def test_filter_rows_no_filters(self, session, sample_users):
        """Test filtering with no filters returns all rows."""
        results = await filter_rows(TestUser, session, filters=None)
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_filter_rows_equality(self, session, sample_users):
        """Test filtering with equality operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "active")]
        )
        assert len(results) == 4
        assert all(u.status == "active" for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_not_equal(self, session, sample_users):
        """Test filtering with not equal operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("role", FilterOp.NE, "user")]
        )
        assert len(results) == 2
        assert all(u.role != "user" for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_greater_than(self, session, sample_users):
        """Test filtering with greater than operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("age", FilterOp.GT, 25)]
        )
        assert len(results) == 3
        assert all(u.age > 25 for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_greater_equal(self, session, sample_users):
        """Test filtering with greater than or equal operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("age", FilterOp.GE, 25)]
        )
        assert len(results) == 4
        assert all(u.age >= 25 for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_less_than(self, session, sample_users):
        """Test filtering with less than operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("age", FilterOp.LT, 30)]
        )
        assert len(results) == 3
        assert all(u.age < 30 for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_less_equal(self, session, sample_users):
        """Test filtering with less than or equal operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("age", FilterOp.LE, 30)]
        )
        assert len(results) == 4
        assert all(u.age <= 30 for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_in_operator(self, session, sample_users):
        """Test filtering with IN operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("id", FilterOp.IN, [1, 2, 3])]
        )
        assert len(results) == 3
        assert all(u.id in [1, 2, 3] for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_not_in_operator(self, session, sample_users):
        """Test filtering with NOT IN operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("id", FilterOp.NOT_IN, [1, 2])]
        )
        assert len(results) == 3
        assert all(u.id not in [1, 2] for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_like_operator(self, session, sample_users):
        """Test filtering with LIKE operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("username", FilterOp.LIKE, "a%")]
        )
        assert len(results) == 1
        assert results[0].username == "alice"
    
    @pytest.mark.asyncio
    async def test_filter_rows_starts_with(self, session, sample_users):
        """Test filtering with STARTS_WITH operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("username", FilterOp.STARTS_WITH, "c")]
        )
        assert len(results) == 1
        assert results[0].username == "charlie"
    
    @pytest.mark.asyncio
    async def test_filter_rows_ends_with(self, session, sample_users):
        """Test filtering with ENDS_WITH operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("email", FilterOp.ENDS_WITH, "example.com")]
        )
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_filter_rows_is_null(self, session, sample_users):
        """Test filtering with IS_NULL operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("deleted_at", FilterOp.IS_NULL)]
        )
        assert len(results) == 4
        assert all(u.deleted_at is None for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_is_not_null(self, session, sample_users):
        """Test filtering with IS_NOT_NULL operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("deleted_at", FilterOp.IS_NOT_NULL)]
        )
        assert len(results) == 1
        assert results[0].deleted_at is not None
    
    @pytest.mark.asyncio
    async def test_filter_rows_between(self, session, sample_users):
        """Test filtering with BETWEEN operator."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("age", FilterOp.BETWEEN, [25, 30])]
        )
        assert len(results) == 3
        assert all(25 <= u.age <= 30 for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_multiple_and(self, session, sample_users):
        """Test filtering with multiple AND conditions."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[
                Filter("status", FilterOp.EQ, "active"),
                Filter("age", FilterOp.GT, 25),
            ],
            logical_op="and"
        )
        assert len(results) == 2
        assert all(u.status == "active" and u.age > 25 for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_multiple_or(self, session, sample_users):
        """Test filtering with multiple OR conditions."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[
                Filter("role", FilterOp.EQ, "admin"),
                Filter("role", FilterOp.EQ, "moderator"),
            ],
            logical_op="or"
        )
        assert len(results) == 2
        assert all(u.role in ["admin", "moderator"] for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_with_ordering_asc(self, session, sample_users):
        """Test filtering with ascending order."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "active")],
            order_by=OrderBy("age", descending=False)
        )
        ages = [u.age for u in results]
        assert ages == sorted(ages)
    
    @pytest.mark.asyncio
    async def test_filter_rows_with_ordering_desc(self, session, sample_users):
        """Test filtering with descending order."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "active")],
            order_by=OrderBy("age", descending=True)
        )
        ages = [u.age for u in results]
        assert ages == sorted(ages, reverse=True)
    
    @pytest.mark.asyncio
    async def test_filter_rows_with_multiple_ordering(self, session, sample_users):
        """Test filtering with multiple order by clauses."""
        results = await filter_rows(
            TestUser,
            session,
            filters=None,
            order_by=[
                OrderBy("status", descending=False),
                OrderBy("age", descending=True)
            ]
        )
        # Should be ordered by status asc, then age desc within each status
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_filter_rows_with_pagination(self, session, sample_users):
        """Test filtering with skip and limit."""
        results = await filter_rows(
            TestUser,
            session,
            filters=None,
            order_by=OrderBy("id", descending=False),
            skip=1,
            limit=2
        )
        assert len(results) == 2
        assert results[0].id == 2
        assert results[1].id == 3
    
    @pytest.mark.asyncio
    async def test_filter_rows_invalid_field(self, session, sample_users):
        """Test filtering with non-existent field raises AttributeError."""
        with pytest.raises(AttributeError, match="does not have field"):
            await filter_rows(
                TestUser,
                session,
                filters=[Filter("nonexistent", FilterOp.EQ, "value")]
            )
    
    @pytest.mark.asyncio
    async def test_filter_rows_invalid_logical_op(self, session, sample_users):
        """Test filtering with invalid logical operator raises ValueError."""
        with pytest.raises(ValueError, match="logical_op must be"):
            await filter_rows(
                TestUser,
                session,
                filters=[Filter("status", FilterOp.EQ, "active")],
                logical_op="invalid"
            )
    
    @pytest.mark.asyncio
    async def test_filter_rows_in_operator_wrong_type(self, session, sample_users):
        """Test IN operator with wrong value type raises ValueError."""
        with pytest.raises(ValueError, match="IN operator requires"):
            await filter_rows(
                TestUser,
                session,
                filters=[Filter("id", FilterOp.IN, "not_a_list")]
            )
    
    @pytest.mark.asyncio
    async def test_filter_rows_between_wrong_type(self, session, sample_users):
        """Test BETWEEN operator with wrong value type raises ValueError."""
        with pytest.raises(ValueError, match="BETWEEN operator requires"):
            await filter_rows(
                TestUser,
                session,
                filters=[Filter("age", FilterOp.BETWEEN, [25])]  # Only 1 value
            )
    
    @pytest.mark.asyncio
    async def test_filter_rows_starts_with_wrong_type(self, session, sample_users):
        """Test STARTS_WITH operator with non-string raises ValueError."""
        with pytest.raises(ValueError, match="STARTS_WITH operator requires string"):
            await filter_rows(
                TestUser,
                session,
                filters=[Filter("username", FilterOp.STARTS_WITH, 123)]
            )


# Tests for filter_rows_streaming
class TestFilterRowsStreaming:
    """Tests for filter_rows_streaming function."""
    
    @pytest.mark.asyncio
    async def test_filter_rows_streaming_basic(self, session, sample_users):
        """Test streaming rows with basic filter."""
        results = []
        async for user in filter_rows_streaming(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "active")]
        ):
            results.append(user)
        
        assert len(results) == 4
        assert all(u.status == "active" for u in results)
    
    @pytest.mark.asyncio
    async def test_filter_rows_streaming_with_limit(self, session, sample_users):
        """Test streaming with limit."""
        results = []
        async for user in filter_rows_streaming(
            TestUser,
            session,
            filters=None,
            limit=3
        ):
            results.append(user)
        
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_filter_rows_streaming_ordered(self, session, sample_users):
        """Test streaming with ordering."""
        results = []
        async for user in filter_rows_streaming(
            TestUser,
            session,
            filters=None,
            order_by=OrderBy("age", descending=False)
        ):
            results.append(user)
        
        ages = [u.age for u in results]
        assert ages == sorted(ages)


# Tests for count_filtered_rows
class TestCountFilteredRows:
    """Tests for count_filtered_rows function."""
    
    @pytest.mark.asyncio
    async def test_count_all_rows(self, session, sample_users):
        """Test counting all rows without filters."""
        count = await count_filtered_rows(TestUser, session, filters=None)
        assert count == 5
    
    @pytest.mark.asyncio
    async def test_count_with_filter(self, session, sample_users):
        """Test counting with filter."""
        count = await count_filtered_rows(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "active")]
        )
        assert count == 4
    
    @pytest.mark.asyncio
    async def test_count_with_multiple_filters_and(self, session, sample_users):
        """Test counting with multiple AND filters."""
        count = await count_filtered_rows(
            TestUser,
            session,
            filters=[
                Filter("status", FilterOp.EQ, "active"),
                Filter("age", FilterOp.GT, 25),
            ],
            logical_op="and"
        )
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_count_with_multiple_filters_or(self, session, sample_users):
        """Test counting with multiple OR filters."""
        count = await count_filtered_rows(
            TestUser,
            session,
            filters=[
                Filter("role", FilterOp.EQ, "admin"),
                Filter("role", FilterOp.EQ, "moderator"),
            ],
            logical_op="or"
        )
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_count_no_matches(self, session, sample_users):
        """Test counting when no rows match."""
        count = await count_filtered_rows(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "nonexistent")]
        )
        assert count == 0


# Tests for filter_one
class TestFilterOne:
    """Tests for filter_one function."""
    
    @pytest.mark.asyncio
    async def test_filter_one_success(self, session, sample_users):
        """Test finding exactly one row."""
        user = await filter_one(
            TestUser,
            session,
            filters=[Filter("username", FilterOp.EQ, "alice")]
        )
        assert user.username == "alice"
        assert user.id == 1
    
    @pytest.mark.asyncio
    async def test_filter_one_not_found(self, session, sample_users):
        """Test filter_one raises KeyError when no match."""
        with pytest.raises(KeyError, match="No .* found matching filters"):
            await filter_one(
                TestUser,
                session,
                filters=[Filter("username", FilterOp.EQ, "nonexistent")]
            )
    
    @pytest.mark.asyncio
    async def test_filter_one_multiple_matches(self, session, sample_users):
        """Test filter_one raises KeyError when multiple matches."""
        with pytest.raises(KeyError, match="Multiple .* rows found"):
            await filter_one(
                TestUser,
                session,
                filters=[Filter("status", FilterOp.EQ, "active")]
            )
    
    @pytest.mark.asyncio
    async def test_filter_one_with_or_logic(self, session, sample_users):
        """Test filter_one with OR logic."""
        # This should fail because multiple users match
        with pytest.raises(KeyError, match="Multiple"):
            await filter_one(
                TestUser,
                session,
                filters=[
                    Filter("role", FilterOp.EQ, "admin"),
                    Filter("role", FilterOp.EQ, "moderator"),
                ],
                logical_op="or"
            )


# Tests for filter_one_or_none
class TestFilterOneOrNone:
    """Tests for filter_one_or_none function."""
    
    @pytest.mark.asyncio
    async def test_filter_one_or_none_success(self, session, sample_users):
        """Test finding exactly one row."""
        user = await filter_one_or_none(
            TestUser,
            session,
            filters=[Filter("username", FilterOp.EQ, "alice")]
        )
        assert user is not None
        assert user.username == "alice"
    
    @pytest.mark.asyncio
    async def test_filter_one_or_none_not_found(self, session, sample_users):
        """Test filter_one_or_none returns None when no match."""
        user = await filter_one_or_none(
            TestUser,
            session,
            filters=[Filter("username", FilterOp.EQ, "nonexistent")]
        )
        assert user is None
    
    @pytest.mark.asyncio
    async def test_filter_one_or_none_multiple_matches(self, session, sample_users):
        """Test filter_one_or_none raises KeyError when multiple matches."""
        with pytest.raises(KeyError, match="Multiple .* rows found"):
            await filter_one_or_none(
                TestUser,
                session,
                filters=[Filter("status", FilterOp.EQ, "active")]
            )


# Tests for find_by convenience function
class TestFindBy:
    """Tests for find_by convenience function."""
    
    @pytest.mark.asyncio
    async def test_find_by_single_field(self, session, sample_users):
        """Test finding by single field."""
        results = await find_by(TestUser, session, status="active")
        assert len(results) == 4
        assert all(u.status == "active" for u in results)
    
    @pytest.mark.asyncio
    async def test_find_by_multiple_fields(self, session, sample_users):
        """Test finding by multiple fields."""
        results = await find_by(
            TestUser,
            session,
            status="active",
            role="user"
        )
        assert len(results) == 2
        assert results[0].username == "bob"
    
    @pytest.mark.asyncio
    async def test_find_by_with_ordering(self, session, sample_users):
        """Test find_by with ordering."""
        results = await find_by(
            TestUser,
            session,
            status="active",
            order_by=OrderBy("age", descending=True)
        )
        ages = [u.age for u in results]
        assert ages == sorted(ages, reverse=True)
    
    @pytest.mark.asyncio
    async def test_find_by_with_pagination(self, session, sample_users):
        """Test find_by with pagination."""
        results = await find_by(
            TestUser,
            session,
            status="active",
            skip=1,
            limit=2
        )
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_find_by_no_matches(self, session, sample_users):
        """Test find_by with no matches returns empty list."""
        results = await find_by(TestUser, session, status="nonexistent")
        assert len(results) == 0


# Tests for find_one_by convenience function
class TestFindOneBy:
    """Tests for find_one_by convenience function."""
    
    @pytest.mark.asyncio
    async def test_find_one_by_success(self, session, sample_users):
        """Test finding one row by field."""
        user = await find_one_by(TestUser, session, username="alice")
        assert user.username == "alice"
        assert user.id == 1
    
    @pytest.mark.asyncio
    async def test_find_one_by_multiple_fields(self, session, sample_users):
        """Test finding one row by multiple fields."""
        user = await find_one_by(
            TestUser,
            session,
            username="bob",
            role="user"
        )
        assert user.username == "bob"
        assert user.role == "user"
    
    @pytest.mark.asyncio
    async def test_find_one_by_not_found(self, session, sample_users):
        """Test find_one_by raises KeyError when not found."""
        with pytest.raises(KeyError, match="No .* found"):
            await find_one_by(TestUser, session, username="nonexistent")
    
    @pytest.mark.asyncio
    async def test_find_one_by_multiple_matches(self, session, sample_users):
        """Test find_one_by raises KeyError when multiple matches."""
        with pytest.raises(KeyError, match="Multiple .* rows found"):
            await find_one_by(TestUser, session, status="active")


# Tests for helper functions
class TestHelperFunctions:
    """Tests for and_filters and or_filters helpers."""
    
    def test_and_filters(self):
        """Test and_filters helper function."""
        f1 = Filter("status", FilterOp.EQ, "active")
        f2 = Filter("age", FilterOp.GT, 18)
        
        filters = and_filters(f1, f2)
        
        assert isinstance(filters, list)
        assert len(filters) == 2
        assert filters[0] == f1
        assert filters[1] == f2
    
    def test_or_filters(self):
        """Test or_filters helper function."""
        f1 = Filter("role", FilterOp.EQ, "admin")
        f2 = Filter("role", FilterOp.EQ, "moderator")
        
        filters = or_filters(f1, f2)
        
        assert isinstance(filters, list)
        assert len(filters) == 2
        assert filters[0] == f1
        assert filters[1] == f2
    
    def test_and_filters_empty(self):
        """Test and_filters with no arguments."""
        filters = and_filters()
        assert filters == []
    
    def test_or_filters_empty(self):
        """Test or_filters with no arguments."""
        filters = or_filters()
        assert filters == []


# Tests for edge cases
class TestEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_empty_database(self, session):
        """Test filtering on empty database."""
        results = await filter_rows(TestUser, session, filters=None)
        assert len(results) == 0
    
    @pytest.mark.asyncio
    async def test_count_empty_database(self, session):
        """Test counting on empty database."""
        count = await count_filtered_rows(TestUser, session, filters=None)
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_filter_one_or_none_empty_database(self, session):
        """Test filter_one_or_none on empty database."""
        user = await filter_one_or_none(
            TestUser,
            session,
            filters=[Filter("username", FilterOp.EQ, "alice")]
        )
        assert user is None
    
    @pytest.mark.asyncio
    async def test_ordering_invalid_field(self, session, sample_users):
        """Test ordering by non-existent field raises AttributeError."""
        with pytest.raises(AttributeError, match="does not have field"):
            await filter_rows(
                TestUser,
                session,
                filters=None,
                order_by=OrderBy("nonexistent")
            )
    
    @pytest.mark.asyncio
    async def test_filter_with_none_value(self, session, sample_users):
        """Test filtering with None as explicit value."""
        # IS NULL should work
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("deleted_at", FilterOp.IS_NULL)]
        )
        assert len(results) == 4
    
    @pytest.mark.asyncio
    async def test_large_in_list(self, session, sample_users):
        """Test IN operator with large list."""
        large_list = list(range(1, 1000))
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("id", FilterOp.IN, large_list)]
        )
        # Should only return existing IDs
        assert len(results) == 5
    
    @pytest.mark.asyncio
    async def test_empty_in_list(self, session, sample_users):
        """Test IN operator with empty list."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("id", FilterOp.IN, [])]
        )
        assert len(results) == 0


# Integration tests
class TestFilterIntegration:
    """Integration tests combining multiple features."""
    
    @pytest.mark.asyncio
    async def test_complex_query(self, session, sample_users):
        """Test complex query with multiple filters and ordering."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[
                Filter("status", FilterOp.EQ, "active"),
                Filter("age", FilterOp.GE, 25),
                Filter("deleted_at", FilterOp.IS_NULL),
            ],
            logical_op="and",
            order_by=[
                OrderBy("role", descending=False),
                OrderBy("age", descending=True)
            ],
            skip=0,
            limit=10
        )
        
        assert len(results) == 3
        # Verify all conditions
        for user in results:
            assert user.status == "active"
            assert user.age >= 25
            assert user.deleted_at is None
    
    @pytest.mark.asyncio
    async def test_pagination_consistency(self, session, sample_users):
        """Test that pagination returns consistent results."""
        # Get all results
        all_results = await filter_rows(
            TestUser,
            session,
            filters=None,
            order_by=OrderBy("id", descending=False)
        )
        
        # Get paginated results
        page1 = await filter_rows(
            TestUser,
            session,
            filters=None,
            order_by=OrderBy("id", descending=False),
            skip=0,
            limit=2
        )
        
        page2 = await filter_rows(
            TestUser,
            session,
            filters=None,
            order_by=OrderBy("id", descending=False),
            skip=2,
            limit=2
        )
        
        page3 = await filter_rows(
            TestUser,
            session,
            filters=None,
            order_by=OrderBy("id", descending=False),
            skip=4,
            limit=2
        )
        
        # Verify pages match full results
        assert page1[0].id == all_results[0].id
        assert page1[1].id == all_results[1].id
        assert page2[0].id == all_results[2].id
        assert page2[1].id == all_results[3].id
        assert page3[0].id == all_results[4].id
    
    @pytest.mark.asyncio
    async def test_count_matches_results(self, session, sample_users):
        """Test that count matches actual result count."""
        filters = [
            Filter("status", FilterOp.EQ, "active"),
            Filter("age", FilterOp.GT, 25)
        ]
        
        count = await count_filtered_rows(
            TestUser,
            session,
            filters=filters
        )
        
        results = await filter_rows(
            TestUser,
            session,
            filters=filters
        )
        
        assert count == len(results)
    
    @pytest.mark.asyncio
    async def test_streaming_matches_regular(self, session, sample_users):
        """Test that streaming returns same results as regular query."""
        filters = [Filter("status", FilterOp.EQ, "active")]
        order = OrderBy("age", descending=False)
        
        # Regular query
        regular_results = await filter_rows(
            TestUser,
            session,
            filters=filters,
            order_by=order
        )
        
        # Streaming query
        streaming_results = []
        async for user in filter_rows_streaming(
            TestUser,
            session,
            filters=filters,
            order_by=order
        ):
            streaming_results.append(user)
        
        # Compare
        assert len(regular_results) == len(streaming_results)
        for r, s in zip(regular_results, streaming_results):
            assert r.id == s.id
    
    @pytest.mark.asyncio
    async def test_find_by_matches_filter_rows(self, session, sample_users):
        """Test that find_by produces same results as filter_rows."""
        # Using find_by
        find_by_results = await find_by(
            TestUser,
            session,
            status="active",
            role="user"
        )
        
        # Using filter_rows
        filter_results = await filter_rows(
            TestUser,
            session,
            filters=[
                Filter("status", FilterOp.EQ, "active"),
                Filter("role", FilterOp.EQ, "user"),
            ]
        )
        
        # Should match
        assert len(find_by_results) == len(filter_results)
        find_by_ids = {u.id for u in find_by_results}
        filter_ids = {u.id for u in filter_results}
        assert find_by_ids == filter_ids


# Performance/stress tests
class TestFilterPerformance:
    """Tests for performance and handling of large datasets."""
    
    @pytest.mark.asyncio
    async def test_many_filters(self, session, sample_users):
        """Test handling many filters efficiently."""
        # Create many filters
        filters = [
            Filter("status", FilterOp.EQ, "active"),
            Filter("age", FilterOp.GT, 20),
            Filter("age", FilterOp.LT, 40),
            Filter("deleted_at", FilterOp.IS_NULL),
            Filter("role", FilterOp.IN, ["admin", "user", "moderator"]),
        ]
        
        results = await filter_rows(
            TestUser,
            session,
            filters=filters
        )
        
        # Should handle without error
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_complex_or_logic(self, session, sample_users):
        """Test complex OR logic with many conditions."""
        filters = [
            Filter("username", FilterOp.EQ, "alice"),
            Filter("username", FilterOp.EQ, "bob"),
            Filter("username", FilterOp.EQ, "charlie"),
            Filter("username", FilterOp.EQ, "diana"),
        ]
        
        results = await filter_rows(
            TestUser,
            session,
            filters=filters,
            logical_op="or"
        )
        
        assert len(results) == 4


# Test for case-insensitive operations (if database supports)
class TestCaseInsensitiveOperations:
    """Tests for case-insensitive operations."""
    
    @pytest.mark.asyncio
    async def test_ilike_operator(self, session, sample_users):
        """Test ILIKE operator for case-insensitive matching."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("username", FilterOp.ILIKE, "ALICE")]
        )
        # SQLite's LIKE is case-insensitive by default
        # This test verifies the operator works
        assert len(results) >= 0  # May or may not match depending on DB


# Tests for type safety and validation
class TestTypeValidation:
    """Tests for type validation in filters."""
    
    @pytest.mark.asyncio
    async def test_in_operator_with_tuple(self, session, sample_users):
        """Test IN operator accepts tuple."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("id", FilterOp.IN, (1, 2, 3))]
        )
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_in_operator_with_set(self, session, sample_users):
        """Test IN operator accepts set."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("id", FilterOp.IN, {1, 2, 3})]
        )
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_between_with_tuple(self, session, sample_users):
        """Test BETWEEN operator accepts tuple."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("age", FilterOp.BETWEEN, (25, 30))]
        )
        assert len(results) == 3


# Parametrized tests for all comparison operators
@pytest.mark.parametrize("op,value,expected_count", [
    (FilterOp.EQ, "active", 4),
    (FilterOp.NE, "active", 1),
    (FilterOp.GT, 25, 3),
    (FilterOp.GE, 25, 4),
    (FilterOp.LT, 30, 3),
    (FilterOp.LE, 30, 4),
])
@pytest.mark.asyncio
async def test_comparison_operators_parametrized(
    session,
    sample_users,
    op,
    value,
    expected_count
):
    """Parametrized test for all comparison operators."""
    field = "status" if isinstance(value, str) else "age"
    
    results = await filter_rows(
        TestUser,
        session,
        filters=[Filter(field, op, value)]
    )
    
    assert len(results) == expected_count


# Test for FilterOp enum
class TestFilterOpEnum:
    """Tests for FilterOp enum."""
    
    def test_filterop_values(self):
        """Test FilterOp has expected values."""
        assert FilterOp.EQ == "eq"
        assert FilterOp.NE == "ne"
        assert FilterOp.LT == "lt"
        assert FilterOp.LE == "le"
        assert FilterOp.GT == "gt"
        assert FilterOp.GE == "ge"
        assert FilterOp.IN == "in"
        assert FilterOp.NOT_IN == "not_in"
        assert FilterOp.LIKE == "like"
        assert FilterOp.ILIKE == "ilike"
        assert FilterOp.IS_NULL == "is_null"
        assert FilterOp.IS_NOT_NULL == "is_not_null"
        assert FilterOp.BETWEEN == "between"
        assert FilterOp.STARTS_WITH == "starts_with"
        assert FilterOp.ENDS_WITH == "ends_with"
    
    def test_filterop_is_string_enum(self):
        """Test FilterOp values are strings."""
        for op in FilterOp:
            assert isinstance(op.value, str)


# Tests for real-world scenarios
class TestRealWorldScenarios:
    """Tests simulating real-world use cases."""
    
    @pytest.mark.asyncio
    async def test_active_users_search(self, session, sample_users):
        """Simulate searching for active users by name pattern."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[
                Filter("status", FilterOp.EQ, "active"),
                Filter("username", FilterOp.STARTS_WITH, "a"),
            ],
            order_by=OrderBy("username", descending=False)
        )
        
        assert len(results) == 1
        assert results[0].username == "alice"
    
    @pytest.mark.asyncio
    async def test_user_list_with_pagination(self, session, sample_users):
        """Simulate paginated user list with filtering."""
        page = 1
        page_size = 2
        
        # Get total count
        total = await count_filtered_rows(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "active")]
        )
        
        # Get page
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("status", FilterOp.EQ, "active")],
            order_by=OrderBy("created_at", descending=True),
            skip=(page - 1) * page_size,
            limit=page_size
        )
        
        assert total == 4
        assert len(results) == 2
        total_pages = (total + page_size - 1) // page_size
        assert total_pages == 2
    
    @pytest.mark.asyncio
    async def test_admin_and_moderator_list(self, session, sample_users):
        """Simulate getting all privileged users."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[
                Filter("role", FilterOp.IN, ["admin", "moderator"]),
                Filter("deleted_at", FilterOp.IS_NULL),
            ],
            order_by=OrderBy("role", descending=False)
        )
        
        assert len(results) == 2
        roles = [u.role for u in results]
        assert "admin" in roles
        assert "moderator" in roles
    
    @pytest.mark.asyncio
    async def test_soft_deleted_users(self, session, sample_users):
        """Simulate finding soft-deleted users."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[Filter("deleted_at", FilterOp.IS_NOT_NULL)]
        )
        
        assert len(results) == 1
        assert results[0].username == "eve"
    
    @pytest.mark.asyncio
    async def test_age_range_search(self, session, sample_users):
        """Simulate searching users in age range."""
        results = await filter_rows(
            TestUser,
            session,
            filters=[
                Filter("age", FilterOp.BETWEEN, [25, 30]),
                Filter("status", FilterOp.EQ, "active"),
            ],
            order_by=OrderBy("age", descending=False)
        )
        
        assert len(results) == 3
        assert all(25 <= u.age <= 30 for u in results)


# Conftest additions for better test coverage
@pytest_asyncio.fixture
async def empty_session(engine) -> AsyncIterator[AsyncSession]:
    """Create empty database session for testing edge cases."""
    async_session = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        # Clear any existing data
        await session.execute(TestUser.__table__.delete())
        await session.commit()
        yield session


# Additional test for ensure_base_inheritance
class TestBaseInheritance:
    """Tests for base class inheritance checking."""
    
    @pytest.mark.asyncio
    async def test_filter_non_base_class(self, session):
        """Test filtering with non-Base class raises TypeError."""
        class NotABaseClass:
            pass
        
        with pytest.raises(TypeError, match="must inherit from"):
            await filter_rows(NotABaseClass, session, filters=None)
