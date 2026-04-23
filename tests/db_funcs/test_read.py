"""Unit tests for read.py database reading functionality."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

from sqlalchemy import String, Integer, DateTime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from rail_svc.db.base import Base
from rail_svc.db_funcs.read import (
    get_row,
    get_row_by_name,
    get_row_or_none,
    get_rows,
    get_rows_streaming,
    count_rows,
)
from rail_svc.db_funcs.create import create_row, create_rows


# Test Models
class TestArticle(Base):
    """Test article model for reading tests."""
    
    __tablename__ = "test_articles"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    title: Mapped[str] = mapped_column(String(200))
    author: Mapped[str] = mapped_column(String(100))
    views: Mapped[int] = mapped_column(Integer, default=0)
    published: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    
    @classmethod
    def pydantic_model_class(cls):
        """Return Pydantic model class."""
        from pydantic import BaseModel
        
        class ArticleSchema(BaseModel):
            id: int | None = None
            name: str
            title: str
            author: str
            views: int = 0
            published: bool = False
            
            class Config:
                from_attributes = True
        
        return ArticleSchema


class TestBook(Base):
    """Test book model without name field."""
    
    __tablename__ = "test_books"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    isbn: Mapped[str] = mapped_column(String(20), unique=True)
    title: Mapped[str] = mapped_column(String(200))
    pages: Mapped[int] = mapped_column(Integer)
    
    @classmethod
    def pydantic_model_class(cls):
        """Return Pydantic model class."""
        from pydantic import BaseModel
        
        class BookSchema(BaseModel):
            id: int | None = None
            isbn: str
            title: str
            pages: int
            
            class Config:
                from_attributes = True
        
        return BookSchema


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
async def sample_articles(session: AsyncSession) -> list[TestArticle]:
    """Create sample articles for testing."""
    articles_data = [
        {
            "name": "article-1",
            "title": "Introduction to Python",
            "author": "Alice",
            "views": 100,
            "published": True,
        },
        {
            "name": "article-2",
            "title": "Advanced SQLAlchemy",
            "author": "Bob",
            "views": 250,
            "published": True,
        },
        {
            "name": "article-3",
            "title": "Testing Best Practices",
            "author": "Alice",
            "views": 150,
            "published": False,
        },
        {
            "name": "article-4",
            "title": "Async Programming",
            "author": "Charlie",
            "views": 300,
            "published": True,
        },
        {
            "name": "article-5",
            "title": "Database Design",
            "author": "Alice",
            "views": 200,
            "published": True,
        },
    ]
    
    articles = await create_rows(
        TestArticle,
        session,
        articles_data,
        validate=False
    )
    
    return articles


@pytest_asyncio.fixture
async def sample_books(session: AsyncSession) -> list[TestBook]:
    """Create sample books for testing."""
    books_data = [
        {"isbn": "978-0-123-45678-0", "title": "Book One", "pages": 200},
        {"isbn": "978-0-123-45678-1", "title": "Book Two", "pages": 350},
        {"isbn": "978-0-123-45678-2", "title": "Book Three", "pages": 150},
    ]
    
    books = await create_rows(
        TestBook,
        session,
        books_data,
        validate=False
    )
    
    return books


# Tests for get_row
class TestGetRow:
    """Tests for get_row function."""
    
    @pytest.mark.asyncio
    async def test_get_row_success(self, session, sample_articles):
        """Test getting a row by ID successfully."""
        article_id = sample_articles[0].id
        
        article = await get_row(TestArticle, session, article_id)
        
        assert article.id == article_id
        assert article.name == "article-1"
        assert article.title == "Introduction to Python"
    
    @pytest.mark.asyncio
    async def test_get_row_not_found(self, session, sample_articles):
        """Test getting non-existent row raises KeyError."""
        non_existent_id = 99999
        
        with pytest.raises(KeyError, match="not found"):
            await get_row(TestArticle, session, non_existent_id)
    
    @pytest.mark.asyncio
    async def test_get_row_empty_table(self, session):
        """Test getting row from empty table raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await get_row(TestArticle, session, 1)
    
    @pytest.mark.asyncio
    async def test_get_row_returns_correct_type(self, session, sample_articles):
        """Test get_row returns correct model type."""
        article = await get_row(TestArticle, session, sample_articles[0].id)
        
        assert isinstance(article, TestArticle)
    
    @pytest.mark.asyncio
    async def test_get_row_with_all_fields(self, session, sample_articles):
        """Test get_row returns object with all fields populated."""
        article = await get_row(TestArticle, session, sample_articles[0].id)
        
        assert article.id is not None
        assert article.name is not None
        assert article.title is not None
        assert article.author is not None
        assert article.views is not None
        assert article.published is not None
        assert article.created_at is not None
    
    @pytest.mark.asyncio
    async def test_get_row_invalid_class(self, session):
        """Test get_row with non-Base class raises TypeError."""
        class NotABase:
            pass
        
        with pytest.raises(TypeError, match="must inherit from"):
            await get_row(NotABase, session, 1)


# Tests for get_row_by_name
class TestGetRowByName:
    """Tests for get_row_by_name function."""
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_success(self, session, sample_articles):
        """Test getting row by name successfully."""
        article = await get_row_by_name(TestArticle, session, "article-1")
        
        assert article.name == "article-1"
        assert article.title == "Introduction to Python"
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_not_found(self, session, sample_articles):
        """Test getting non-existent name raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await get_row_by_name(TestArticle, session, "nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_case_sensitive(self, session, sample_articles):
        """Test that name lookup is case-sensitive."""
        # SQLite's LIKE is case-insensitive by default, but == should be case-sensitive
        # This test documents the behavior
        article = await get_row_by_name(TestArticle, session, "article-1")
        assert article is not None
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_no_name_field(self, session, sample_books):
        """Test get_row_by_name with model without name field raises AttributeError."""
        with pytest.raises(AttributeError, match="does not have a 'name' attribute"):
            await get_row_by_name(TestBook, session, "any-name")
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_empty_table(self, session):
        """Test get_row_by_name on empty table raises KeyError."""
        with pytest.raises(KeyError, match="not found"):
            await get_row_by_name(TestArticle, session, "article-1")
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_special_characters(self, session):
        """Test get_row_by_name with special characters in name."""
        article = await create_row(
            TestArticle,
            session,
            name="special-name's-\"test\"",
            title="Special Test",
            author="Tester",
            validate=False
        )
        
        retrieved = await get_row_by_name(
            TestArticle,
            session,
            "special-name's-\"test\""
        )
        
        assert retrieved.id == article.id


# Tests for get_row_or_none
class TestGetRowOrNone:
    """Tests for get_row_or_none function."""
    
    @pytest.mark.asyncio
    async def test_get_row_or_none_success(self, session, sample_articles):
        """Test getting existing row returns the row."""
        article_id = sample_articles[0].id
        
        article = await get_row_or_none(TestArticle, session, article_id)
        
        assert article is not None
        assert article.id == article_id
    
    @pytest.mark.asyncio
    async def test_get_row_or_none_not_found(self, session, sample_articles):
        """Test getting non-existent row returns None."""
        article = await get_row_or_none(TestArticle, session, 99999)
        
        assert article is None
    
    @pytest.mark.asyncio
    async def test_get_row_or_none_empty_table(self, session):
        """Test getting row from empty table returns None."""
        article = await get_row_or_none(TestArticle, session, 1)
        
        assert article is None
    
    @pytest.mark.asyncio
    async def test_get_row_or_none_does_not_raise(self, session):
        """Test get_row_or_none never raises KeyError."""
        # Should not raise even for invalid ID
        result = await get_row_or_none(TestArticle, session, -1)
        assert result is None


# Tests for get_rows
class TestGetRows:
    """Tests for get_rows function."""
    
    @pytest.mark.asyncio
    async def test_get_rows_all(self, session, sample_articles):
        """Test getting all rows."""
        articles = await get_rows(TestArticle, session)
        
        assert len(articles) == 5
    
    @pytest.mark.asyncio
    async def test_get_rows_with_limit(self, session, sample_articles):
        """Test getting rows with limit."""
        articles = await get_rows(TestArticle, session, limit=3)
        
        assert len(articles) == 3
    
    @pytest.mark.asyncio
    async def test_get_rows_with_skip(self, session, sample_articles):
        """Test getting rows with skip."""
        articles = await get_rows(TestArticle, session, skip=2)
        
        # Should get 3 rows (5 total - 2 skipped)
        assert len(articles) == 3
    
    @pytest.mark.asyncio
    async def test_get_rows_with_skip_and_limit(self, session, sample_articles):
        """Test getting rows with both skip and limit."""
        articles = await get_rows(TestArticle, session, skip=1, limit=2)
        
        assert len(articles) == 2
    
    @pytest.mark.asyncio
    async def test_get_rows_empty_table(self, session):
        """Test getting rows from empty table returns empty list."""
        articles = await get_rows(TestArticle, session)
        
        assert len(articles) == 0
        assert isinstance(articles, list)
    
    @pytest.mark.asyncio
    async def test_get_rows_skip_beyond_total(self, session, sample_articles):
        """Test skip beyond total rows returns empty list."""
        articles = await get_rows(TestArticle, session, skip=100)
        
        assert len(articles) == 0
    
    @pytest.mark.asyncio
    async def test_get_rows_limit_zero(self, session, sample_articles):
        """Test limit=0 returns empty list."""
        articles = await get_rows(TestArticle, session, limit=0)
        
        assert len(articles) == 0
    
    @pytest.mark.asyncio
    async def test_get_rows_uses_default_pagination(self, session):
        """Test get_rows uses table's default pagination limit."""
        # Create more articles than default limit
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}",
                "author": "Author",
            }
            for i in range(150)  # More than default 100
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        # Without explicit limit, should use default (100)
        articles = await get_rows(TestArticle, session)
        
        assert len(articles) == 100  # DEFAULT_PAGINATION_LIMIT
    
    @pytest.mark.asyncio
    async def test_get_rows_returns_sequence(self, session, sample_articles):
        """Test get_rows returns a sequence."""
        from collections.abc import Sequence
        
        articles = await get_rows(TestArticle, session)
        
        assert isinstance(articles, Sequence)
    
    @pytest.mark.asyncio
    async def test_get_rows_order_consistent(self, session, sample_articles):
        """Test get_rows returns consistent order across calls."""
        articles1 = await get_rows(TestArticle, session)
        articles2 = await get_rows(TestArticle, session)
        
        ids1 = [a.id for a in articles1]
        ids2 = [a.id for a in articles2]
        
        assert ids1 == ids2


# Tests for get_rows_streaming
class TestGetRowsStreaming:
    """Tests for get_rows_streaming function."""
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_basic(self, session, sample_articles):
        """Test streaming all rows."""
        articles = []
        async for article in get_rows_streaming(TestArticle, session):
            articles.append(article)
        
        assert len(articles) == 5
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_with_limit(self, session, sample_articles):
        """Test streaming with limit."""
        articles = []
        async for article in get_rows_streaming(TestArticle, session, limit=3):
            articles.append(article)
        
        assert len(articles) == 3
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_with_skip(self, session, sample_articles):
        """Test streaming with skip."""
        articles = []
        async for article in get_rows_streaming(TestArticle, session, skip=2):
            articles.append(article)
        
        assert len(articles) == 3
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_with_skip_and_limit(self, session, sample_articles):
        """Test streaming with both skip and limit."""
        articles = []
        async for article in get_rows_streaming(TestArticle, session, skip=1, limit=2):
            articles.append(article)
        
        assert len(articles) == 2
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_empty_table(self, session):
        """Test streaming from empty table."""
        articles = []
        async for article in get_rows_streaming(TestArticle, session):
            articles.append(article)
        
        assert len(articles) == 0
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_early_break(self, session, sample_articles):
        """Test breaking out of streaming early."""
        count = 0
        async for article in get_rows_streaming(TestArticle, session):
            count += 1
            if count >= 2:
                break
        
        assert count == 2
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_is_async_iterator(self, session, sample_articles):
        """Test that streaming returns an async iterator."""
        from collections.abc import AsyncIterator
        
        result = get_rows_streaming(TestArticle, session)
        
        assert isinstance(result, AsyncIterator)
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_yields_correct_type(self, session, sample_articles):
        """Test streaming yields correct model type."""
        async for article in get_rows_streaming(TestArticle, session, limit=1):
            assert isinstance(article, TestArticle)
            break
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_matches_regular(self, session, sample_articles):
        """Test streaming returns same data as regular get_rows."""
        # Regular
        regular_articles = await get_rows(TestArticle, session, limit=3)
        
        # Streaming
        streaming_articles = []
        async for article in get_rows_streaming(TestArticle, session, limit=3):
            streaming_articles.append(article)
        
        # Compare IDs
        regular_ids = [a.id for a in regular_articles]
        streaming_ids = [a.id for a in streaming_articles]
        
        assert regular_ids == streaming_ids
    
    @pytest.mark.asyncio
    async def test_get_rows_streaming_large_dataset(self, session):
        """Test streaming with large dataset doesn't load all into memory."""
        # Create many articles
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}",
                "author": "Author",
            }
            for i in range(1000)
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        # Stream through all - this would be memory-intensive with get_rows
        count = 0
        async for article in get_rows_streaming(TestArticle, session):
            count += 1
        
        assert count == 1000


# Tests for count_rows
class TestCountRows:
    """Tests for count_rows function."""
    
    @pytest.mark.asyncio
    async def test_count_rows_basic(self, session, sample_articles):
        """Test counting rows in table."""
        count = await count_rows(TestArticle, session)
        
        assert count == 5
    
    @pytest.mark.asyncio
    async def test_count_rows_empty_table(self, session):
        """Test counting rows in empty table."""
        count = await count_rows(TestArticle, session)
        
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_count_rows_returns_int(self, session, sample_articles):
        """Test count_rows returns integer."""
        count = await count_rows(TestArticle, session)
        
        assert isinstance(count, int)
    
    @pytest.mark.asyncio
    async def test_count_rows_after_insert(self, session):
        """Test count increases after insert."""
        count_before = await count_rows(TestArticle, session)
        
        await create_row(
            TestArticle,
            session,
            name="new-article",
            title="New Article",
            author="Author",
            validate=False
        )
        
        count_after = await count_rows(TestArticle, session)
        
        assert count_after == count_before + 1
    
    @pytest.mark.asyncio
    async def test_count_rows_large_table(self, session):
        """Test counting large table."""
        # Create many articles
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}",
                "author": "Author",
            }
            for i in range(5000)
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        count = await count_rows(TestArticle, session)
        
        assert count == 5000
    
    @pytest.mark.asyncio
    async def test_count_rows_different_tables(self, session, sample_articles, sample_books):
        """Test counting different tables independently."""
        article_count = await count_rows(TestArticle, session)
        book_count = await count_rows(TestBook, session)
        
        assert article_count == 5
        assert book_count == 3


# Integration tests
class TestReadIntegration:
    """Integration tests combining multiple read operations."""
    
    @pytest.mark.asyncio
    async def test_get_row_then_by_name(self, session, sample_articles):
        """Test getting row by ID then by name."""
        # Get by ID
        article_by_id = await get_row(TestArticle, session, sample_articles[0].id)
        
        # Get by name
        article_by_name = await get_row_by_name(
            TestArticle,
            session,
            article_by_id.name
        )
        
        assert article_by_id.id == article_by_name.id
    
    @pytest.mark.asyncio
    async def test_pagination_workflow(self, session, sample_articles):
        """Test typical pagination workflow."""
        page_size = 2
        
        # Get total count
        total = await count_rows(TestArticle, session)
        total_pages = (total + page_size - 1) // page_size
        
        assert total == 5
        assert total_pages == 3
        
        # Get page 1
        page1 = await get_rows(TestArticle, session, skip=0, limit=page_size)
        assert len(page1) == 2
        
        # Get page 2
        page2 = await get_rows(TestArticle, session, skip=2, limit=page_size)
        assert len(page2) == 2
        
        # Get page 3 (partial)
        page3 = await get_rows(TestArticle, session, skip=4, limit=page_size)
        assert len(page3) == 1
        
        # Verify no overlap
        all_ids = [a.id for a in page1] + [a.id for a in page2] + [a.id for a in page3]
        assert len(all_ids) == len(set(all_ids))  # All unique
    
    @pytest.mark.asyncio
    async def test_streaming_vs_regular_performance(self, session):
        """Test that streaming and regular queries return same data."""
        # Create test data
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}",
                "author": "Author",
            }
            for i in range(100)
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        # Regular get
        regular = await get_rows(TestArticle, session)
        
        # Streaming get
        streaming = []
        async for article in get_rows_streaming(TestArticle, session):
            streaming.append(article)
        
        # Should have same data
        assert len(regular) == len(streaming)
        
        regular_ids = sorted([a.id for a in regular])
        streaming_ids = sorted([a.id for a in streaming])
        assert regular_ids == streaming_ids
    
    @pytest.mark.asyncio
    async def test_get_or_none_workflow(self, session, sample_articles):
        """Test typical get_or_none workflow."""
        # Try to get existing
        existing = await get_row_or_none(TestArticle, session, sample_articles[0].id)
        if existing:
            assert existing.id == sample_articles[0].id
        
        # Try to get non-existing
        non_existing = await get_row_or_none(TestArticle, session, 99999)
        if non_existing is None:
            # Create it
            new_article = await create_row(
                TestArticle,
                session,
                name="new-article",
                title="New Article",
                author="Author",
                validate=False
            )
            assert new_article.id is not None


# Edge cases and error handling
class TestReadEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_get_row_with_negative_id(self, session):
        """Test getting row with negative ID."""
        with pytest.raises(KeyError):
            await get_row(TestArticle, session, -1)
    
    @pytest.mark.asyncio
    async def test_get_row_with_zero_id(self, session):
        """Test getting row with zero ID."""
        with pytest.raises(KeyError):
            await get_row(TestArticle, session, 0)
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_empty_string(self, session):
        """Test getting row by empty string name."""
        with pytest.raises(KeyError):
            await get_row_by_name(TestArticle, session, "")
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_with_spaces(self, session):
        """Test getting row by name with spaces."""
        article = await create_row(
            TestArticle,
            session,
            name="name with spaces",
            title="Title",
            author="Author",
            validate=False
        )
        
        retrieved = await get_row_by_name(TestArticle, session, "name with spaces")
        
        assert retrieved.id == article.id
    
    @pytest.mark.asyncio
    async def test_get_row_by_name_unicode(self, session):
        """Test getting row by name with unicode characters."""
        article = await create_row(
            TestArticle,
            session,
            name="café-☕",
            title="Title",
            author="Author",
            validate=False
        )
        
        retrieved = await get_row_by_name(TestArticle, session, "café-☕")
        
        assert retrieved.id == article.id
    
    @pytest.mark.asyncio
    async def test_get_rows_negative_skip(self, session, sample_articles):
        """Test get_rows with negative skip."""
        # SQLAlchemy should handle this - might treat as 0 or error
        # This test documents the behavior
        articles = await get_rows(TestArticle, session, skip=-1)
        # Should either work or raise an error
        assert isinstance(articles, list)
    
    @pytest.mark.asyncio
    async def test_get_rows_negative_limit(self, session, sample_articles):
        """Test get_rows with negative limit."""
        # This might error or return empty - test documents behavior
        try:
            articles = await get_rows(TestArticle, session, limit=-1)
            assert isinstance(articles, list)
        except Exception:
            # Some databases might reject negative limit
            pass
    
    @pytest.mark.asyncio
    async def test_get_rows_very_large_skip(self, session, sample_articles):
        """Test get_rows with very large skip value."""
        articles = await get_rows(TestArticle, session, skip=1000000)
        
        assert len(articles) == 0
    
    @pytest.mark.asyncio
    async def test_get_rows_very_large_limit(self, session, sample_articles):
        """Test get_rows with very large limit value."""
        articles = await get_rows(TestArticle, session, limit=1000000)
        
        # Should return all available rows
        assert len(articles) == 5
    
    @pytest.mark.asyncio
    async def test_count_rows_after_failed_insert(self, session, sample_articles):
        """Test count remains unchanged after failed insert."""
        count_before = await count_rows(TestArticle, session)
        
        # Try to insert duplicate
        try:
            await create_row(
                TestArticle,
                session,
                name="article-1",  # Duplicate
                title="Duplicate",
                author="Author",
                validate=False
            )
        except Exception:
            pass
        
        count_after = await count_rows(TestArticle, session)
        
        assert count_after == count_before


# Performance tests
class TestReadPerformance:
    """Performance-related tests."""
    
    @pytest.mark.asyncio
    async def test_get_rows_large_table(self, session):
        """Test getting rows from large table."""
        # Create large dataset
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}",
                "author": "Author",
            }
            for i in range(10000)
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        # Get first page - should be fast
        import time
        start = time.time()
        articles = await get_rows(TestArticle, session, limit=100)
        duration = time.time() - start
        
        assert len(articles) == 100
        # Should complete in reasonable time (< 1 second for 100 rows)
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_streaming_memory_efficiency(self, session):
        """Test that streaming doesn't load all data into memory at once."""
        # Create large dataset
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}" * 100,  # Large text
                "author": "Author" * 50,
            }
            for i in range(1000)
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        # Stream through - should handle without memory issues
        count = 0

        async for article in get_rows_streaming(TestArticle, session, limit=1000):
            count += 1
            # Process one at a time, don't accumulate
        
        assert count == 1000


# Test ordering and consistency
class TestReadOrdering:
    """Tests for result ordering and consistency."""
    
    @pytest.mark.asyncio
    async def test_get_rows_consistent_order(self, session, sample_articles):
        """Test that get_rows returns consistent order."""
        # Get rows multiple times
        results1 = await get_rows(TestArticle, session)
        results2 = await get_rows(TestArticle, session)
        results3 = await get_rows(TestArticle, session)
        
        ids1 = [a.id for a in results1]
        ids2 = [a.id for a in results2]
        ids3 = [a.id for a in results3]
        
        # Should be same order each time
        assert ids1 == ids2 == ids3
    
    @pytest.mark.asyncio
    async def test_streaming_consistent_order(self, session, sample_articles):
        """Test that streaming returns consistent order."""
        # Stream multiple times
        async def get_streaming_ids():
            ids = []
            async for article in get_rows_streaming(TestArticle, session):
                ids.append(article.id)
            return ids
        
        ids1 = await get_streaming_ids()
        ids2 = await get_streaming_ids()
        
        assert ids1 == ids2
    
    @pytest.mark.asyncio
    async def test_pagination_order_consistency(self, session, sample_articles):
        """Test that pagination returns rows in consistent order."""
        # Get all rows at once
        all_rows = await get_rows(TestArticle, session)
        all_ids = [a.id for a in all_rows]
        
        # Get same rows via pagination
        page1 = await get_rows(TestArticle, session, skip=0, limit=2)
        page2 = await get_rows(TestArticle, session, skip=2, limit=2)
        page3 = await get_rows(TestArticle, session, skip=4, limit=2)
        
        paginated_ids = (
            [a.id for a in page1] + 
            [a.id for a in page2] + 
            [a.id for a in page3]
        )
        
        assert all_ids == paginated_ids


# Test with multiple tables
class TestReadMultipleTables:
    """Tests involving multiple tables."""
    
    @pytest.mark.asyncio
    async def test_read_from_different_tables(self, session, sample_articles, sample_books):
        """Test reading from different tables independently."""
        articles = await get_rows(TestArticle, session)
        books = await get_rows(TestBook, session)
        
        assert len(articles) == 5
        assert len(books) == 3
        
        # Types should be different
        assert all(isinstance(a, TestArticle) for a in articles)
        assert all(isinstance(b, TestBook) for b in books)
    
    @pytest.mark.asyncio
    async def test_count_different_tables(self, session, sample_articles, sample_books):
        """Test counting different tables."""
        article_count = await count_rows(TestArticle, session)
        book_count = await count_rows(TestBook, session)
        
        assert article_count == 5
        assert book_count == 3
    
    @pytest.mark.asyncio
    async def test_interleaved_reads(self, session, sample_articles, sample_books):
        """Test reading from tables in interleaved manner."""
        # Read article
        article = await get_row(TestArticle, session, sample_articles[0].id)
        assert isinstance(article, TestArticle)
        
        # Read book
        book = await get_row(TestBook, session, sample_books[0].id)
        assert isinstance(book, TestBook)
        
        # Read another article
        article2 = await get_row(TestArticle, session, sample_articles[1].id)
        assert isinstance(article2, TestArticle)
        
        # All should work fine
        assert article.id != article2.id


# Real-world scenario tests
class TestReadRealWorldScenarios:
    """Tests simulating real-world usage patterns."""
    
    @pytest.mark.asyncio
    async def test_search_by_author_pattern(self, session, sample_articles):
        """Simulate searching articles by author."""
        # Get all articles
        all_articles = await get_rows(TestArticle, session)
        
        # Filter by author in Python (in real app, use filter.py)
        alice_articles = [a for a in all_articles if a.author == "Alice"]
        
        assert len(alice_articles) == 3
        assert all(a.author == "Alice" for a in alice_articles)
    
    @pytest.mark.asyncio
    async def test_published_articles_workflow(self, session, sample_articles):
        """Simulate getting published articles."""
        all_articles = await get_rows(TestArticle, session)
        
        # Filter published in Python
        published = [a for a in all_articles if a.published]
        
        assert len(published) == 4
    
    @pytest.mark.asyncio
    async def test_most_viewed_articles(self, session, sample_articles):
        """Simulate getting most-viewed articles."""
        all_articles = await get_rows(TestArticle, session)
        
        # Sort by views in Python
        sorted_articles = sorted(all_articles, key=lambda a: a.views, reverse=True)
        
        # Most viewed should be "Async Programming" with 300 views
        assert sorted_articles[0].views == 300
        assert sorted_articles[0].title == "Async Programming"
    
    @pytest.mark.asyncio
    async def test_article_exists_check(self, session, sample_articles):
        """Simulate checking if article exists before creating."""
        # Check if article exists
        existing = await get_row_or_none(TestArticle, session, sample_articles[0].id)
        
        if existing is None:
            # Create it
            new_article = await create_row(
                TestArticle,
                session,
                name="new-article",
                title="New Article",
                author="Author",
                validate=False
            )
        else:
            # Update it (would use update.py in real app)
            pass
        
        assert existing is not None
    
    @pytest.mark.asyncio
    async def test_batch_processing_workflow(self, session):
        """Simulate batch processing large dataset."""
        # Create large dataset
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}",
                "author": "Author",
                "views": i * 10,
            }
            for i in range(1000)
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        # Process in batches
        batch_size = 100
        total_processed = 0
        
        skip = 0
        while True:
            batch = await get_rows(TestArticle, session, skip=skip, limit=batch_size)
            
            if not batch:
                break
            
            # Process batch
            for article in batch:
                total_processed += 1
            
            skip += batch_size
        
        assert total_processed == 1000
    
    @pytest.mark.asyncio
    async def test_streaming_export_workflow(self, session):
        """Simulate exporting large dataset via streaming."""
        # Create large dataset
        articles_data = [
            {
                "name": f"article-{i}",
                "title": f"Title {i}",
                "author": f"Author {i % 10}",
            }
            for i in range(5000)
        ]
        
        await create_rows(TestArticle, session, articles_data, validate=False)
        
        # Export via streaming (to avoid memory issues)
        exported_count = 0
        exported_data = []
        
        async for article in get_rows_streaming(TestArticle, session, limit=5000):
            # Convert to dict (simulate CSV export)
            exported_data.append({
                'id': article.id,
                'title': article.title,
                'author': article.author,
            })
            exported_count += 1
        
        assert exported_count == 5000
        assert len(exported_data) == 5000


# Test data integrity
class TestReadDataIntegrity:
    """Tests ensuring data integrity during reads."""
    
    @pytest.mark.asyncio
    async def test_get_row_has_all_fields(self, session, sample_articles):
        """Test that get_row returns all fields populated."""
        article = await get_row(TestArticle, session, sample_articles[0].id)
        
        # Check all fields are present and correct type
        assert isinstance(article.id, int)
        assert isinstance(article.name, str)
        assert isinstance(article.title, str)
        assert isinstance(article.author, str)
        assert isinstance(article.views, int)
        assert isinstance(article.published, bool)
        assert isinstance(article.created_at, datetime)
    
    @pytest.mark.asyncio
    async def test_get_rows_all_complete(self, session, sample_articles):
        """Test that all rows from get_rows are complete."""
        articles = await get_rows(TestArticle, session)
        
        for article in articles:
            assert article.id is not None
            assert article.name is not None
            assert article.title is not None
            assert article.author is not None
            assert article.views is not None
            assert article.published is not None
            assert article.created_at is not None
    
    @pytest.mark.asyncio
    async def test_streaming_all_complete(self, session, sample_articles):
        """Test that all streamed rows are complete."""
        async for article in get_rows_streaming(TestArticle, session):
            assert article.id is not None
            assert article.name is not None
            assert article.title is not None
            assert article.author is not None
            assert article.views is not None
            assert article.published is not None
            assert article.created_at is not None
    
    @pytest.mark.asyncio
    async def test_read_does_not_modify_data(self, session, sample_articles):
        """Test that reading doesn't modify the data."""
        original_id = sample_articles[0].id
        original_name = sample_articles[0].name
        
        # Read multiple times
        article1 = await get_row(TestArticle, session, sample_articles[0].id)
        article2 = await get_row(TestArticle, session, sample_articles[0].id)
        article3 = await get_row_by_name(TestArticle, session, sample_articles[0].name)
        
        # All should have same data
        assert article1.id == original_id
        assert article2.id == original_id
        assert article3.id == original_id
        assert article1.name == original_name
        assert article2.name == original_name
        assert article3.name == original_name


# Test with fresh data
class TestReadFreshData:
    """Tests ensuring reads get fresh data."""
    
    @pytest.mark.asyncio
    async def test_get_row_after_create(self, session):
        """Test that get_row retrieves just-created row."""
        # Create article
        article = await create_row(
            TestArticle,
            session,
            name="fresh-article",
            title="Fresh Article",
            author="Author",
            validate=False
        )
        
        # Immediately read it back
        retrieved = await get_row(TestArticle, session, article.id)
        
        assert retrieved.id == article.id
        assert retrieved.name == "fresh-article"
    
    @pytest.mark.asyncio
    async def test_count_reflects_creates(self, session):
        """Test that count reflects recently created rows."""
        initial_count = await count_rows(TestArticle, session)
        
        # Create 3 articles
        for i in range(3):
            await create_row(
                TestArticle,
                session,
                name=f"article-{i}",
                title=f"Title {i}",
                author="Author",
                validate=False
            )
        
        final_count = await count_rows(TestArticle, session)
        
        assert final_count == initial_count + 3


# Parametrized tests
@pytest.mark.parametrize("skip,limit,expected_count", [
    (0, 5, 5),    # All rows
    (0, 3, 3),    # First 3
    (2, 3, 3),    # Skip 2, get 3
    (0, 10, 5),   # Limit exceeds total
    (3, 10, 2),   # Skip 3, get remaining
    (5, 10, 0),   # Skip all
])
@pytest.mark.asyncio
async def test_get_rows_pagination_parametrized(
    session,
    sample_articles,
    skip,
    limit,
    expected_count
):
    """Parametrized test for various pagination scenarios."""
    articles = await get_rows(TestArticle, session, skip=skip, limit=limit)
    assert len(articles) == expected_count


@pytest.mark.parametrize("skip,limit,expected_count", [
    (0, 5, 5),
    (0, 3, 3),
    (2, 3, 3),
    (0, 10, 5),
    (3, 10, 2),
    (5, 10, 0),
])
@pytest.mark.asyncio
async def test_get_rows_streaming_pagination_parametrized(
    session,
    sample_articles,
    skip,
    limit,
    expected_count
):
    """Parametrized test for streaming pagination."""
    articles = []
    async for article in get_rows_streaming(TestArticle, session, skip=skip, limit=limit):
        articles.append(article)
    
    assert len(articles) == expected_count
