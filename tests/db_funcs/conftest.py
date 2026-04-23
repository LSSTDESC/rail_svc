import pytest_asyncio

from datetime import datetime, UTC
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from rail_svc.db.base import Base
from rail_svc.db_funcs.create import create_rows

from fake_db import (
    DbTestUser,
    DbTestArticle,
    DbTestBook,
)


@pytest_asyncio.fixture
async def engine():
    """Create in-memory SQLite database engine."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def session(engine) -> AsyncIterator[AsyncSession]:
    """Create database session for testing."""
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session


@pytest_asyncio.fixture
async def sample_users(session: AsyncSession) -> list[DbTestUser]:
    """Create sample users for testing."""
    now = datetime.now(UTC)

    users = [
        DbTestUser(
            id=1,
            username="alice",
            email="alice@example.com",
            age=25,
            status="active",
            role="admin",
            created_at=now,
        ),
        DbTestUser(
            id=2,
            username="bob",
            email="bob@example.com",
            age=30,
            status="active",
            role="user",
            created_at=now,
        ),
        DbTestUser(
            id=3,
            username="charlie",
            email="charlie@example.com",
            age=35,
            status="inactive",
            role="user",
            created_at=now,
        ),
        DbTestUser(
            id=4,
            username="diana",
            email="diana@example.com",
            age=28,
            status="active",
            role="moderator",
            created_at=now,
        ),
        DbTestUser(
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


@pytest_asyncio.fixture
async def sample_articles(session: AsyncSession) -> list[DbTestArticle]:
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

    articles = await create_rows(DbTestArticle, session, articles_data, validate=False)

    return articles


@pytest_asyncio.fixture
async def sample_books(session: AsyncSession) -> list[DbTestBook]:
    """Create sample books for testing."""
    books_data = [
        {"isbn": "978-0-123-45678-0", "title": "Book One", "pages": 200},
        {"isbn": "978-0-123-45678-1", "title": "Book Two", "pages": 350},
        {"isbn": "978-0-123-45678-2", "title": "Book Three", "pages": 150},
    ]

    books = await create_rows(DbTestBook, session, books_data, validate=False)

    return books
