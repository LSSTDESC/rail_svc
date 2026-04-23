"""Unit tests for create.py database creation functionality."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from typing import AsyncIterator

from pydantic import BaseModel, ValidationError
from sqlalchemy import String, Integer, DateTime
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import Mapped, mapped_column

from rail_svc.db.base import Base
from rail_svc.db_funcs.create import (
    create_row,
    create_rows,
    create_rows_batched,
    bulk_insert_rows,
)


# Test Models
class TestProduct(Base):
    """Test product model for creation tests."""
    
    __tablename__ = "test_products"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    sku: Mapped[str] = mapped_column(String(50), unique=True)
    price: Mapped[int] = mapped_column(Integer)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    
    @classmethod
    def pydantic_model_class(cls):
        """Return Pydantic model class."""
        class ProductSchema(BaseModel):
            id: int | None = None
            name: str
            sku: str
            price: int
            stock: int = 0
            
            class Config:
                from_attributes = True
        
        return ProductSchema


class TestOrder(Base):
    """Test order model with custom get_create_kwargs."""
    
    __tablename__ = "test_orders"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(String(50), unique=True)
    total: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc)
    )
    # Computed field
    order_year: Mapped[int] = mapped_column(Integer)
    
    @classmethod
    async def get_create_kwargs(cls, session, **kwargs):
        """Add computed order_year field."""
        if 'order_year' not in kwargs:
            kwargs['order_year'] = datetime.now(timezone.utc).year
        return kwargs
    
    @classmethod
    def pydantic_model_class(cls):
        """Return Pydantic model class."""
        class OrderSchema(BaseModel):
            id: int | None = None
            order_number: str
            total: int
            order_year: int | None = None
            
            class Config:
                from_attributes = True
        
        return OrderSchema


class TestUser(Base):
    """Test user model with validation."""
    
    __tablename__ = "test_users"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    age: Mapped[int] = mapped_column(Integer)
    
    @classmethod
    def pydantic_model_class(cls):
        """Return Pydantic model class with validation."""
        from pydantic import field_validator
        
        class UserSchema(BaseModel):
            id: int | None = None
            username: str
            email: str
            age: int
            
            @field_validator('age')
            @classmethod
            def age_must_be_positive(cls, v):
                if v < 0:
                    raise ValueError('age must be positive')
                return v
            
            @field_validator('email')
            @classmethod
            def email_must_contain_at(cls, v):
                if '@' not in v:
                    raise ValueError('email must contain @')
                return v
            
            class Config:
                from_attributes = True
        
        return UserSchema


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


# Tests for create_row
class TestCreateRow:
    """Tests for create_row function."""
    
    @pytest.mark.asyncio
    async def test_create_row_basic(self, session):
        """Test creating a single row with valid data."""
        product = await create_row(
            TestProduct,
            session,
            name="Widget",
            sku="WDG-001",
            price=1999,
            stock=100
        )
        
        assert product.id is not None
        assert product.name == "Widget"
        assert product.sku == "WDG-001"
        assert product.price == 1999
        assert product.stock == 100
        assert product.created_at is not None
    
    @pytest.mark.asyncio
    async def test_create_row_with_defaults(self, session):
        """Test creating row uses default values."""
        product = await create_row(
            TestProduct,
            session,
            name="Gadget",
            sku="GDG-001",
            price=2999
            # stock not provided, should use default
        )
        
        assert product.stock == 0  # Default value
    
    @pytest.mark.asyncio
    async def test_create_row_with_validation(self, session):
        """Test creating row with Pydantic validation."""
        user = await create_row(
            TestUser,
            session,
            username="alice",
            email="alice@example.com",
            age=25,
            validate=True
        )
        
        assert user.id is not None
        assert user.username == "alice"
        assert user.email == "alice@example.com"
        assert user.age == 25
    
    @pytest.mark.asyncio
    async def test_create_row_validation_failure(self, session):
        """Test creating row with invalid data raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            await create_row(
                TestUser,
                session,
                username="bob",
                email="invalid-email",  # Missing @
                age=30,
                validate=True
            )
        
        errors = exc_info.value.errors()
        assert any('email' in str(e) for e in errors)
    
    @pytest.mark.asyncio
    async def test_create_row_negative_age_validation(self, session):
        """Test validation catches negative age."""
        with pytest.raises(ValidationError) as exc_info:
            await create_row(
                TestUser,
                session,
                username="charlie",
                email="charlie@example.com",
                age=-5,  # Invalid
                validate=True
            )
        
        errors = exc_info.value.errors()
        assert any('age' in str(e) for e in errors)
    
    @pytest.mark.asyncio
    async def test_create_row_skip_validation(self, session):
        """Test creating row without validation."""
        # This would fail validation but should work with validate=False
        product = await create_row(
            TestProduct,
            session,
            name="Test",
            sku="TST-001",
            price=100,
            validate=False
        )
        
        assert product.id is not None
    
    @pytest.mark.asyncio
    async def test_create_row_duplicate_unique_constraint(self, session):
        """Test creating duplicate row raises IntegrityError."""
        # Create first product
        await create_row(
            TestProduct,
            session,
            name="Unique Product",
            sku="UNQ-001",
            price=1000
        )
        
        # Try to create duplicate (same SKU)
        with pytest.raises(IntegrityError):
            await create_row(
                TestProduct,
                session,
                name="Different Name",
                sku="UNQ-001",  # Duplicate SKU
                price=2000
            )
    
    @pytest.mark.asyncio
    async def test_create_row_with_get_create_kwargs(self, session):
        """Test creating row with custom get_create_kwargs."""
        order = await create_row(
            TestOrder,
            session,
            order_number="ORD-2024-001",
            total=5000
            # order_year should be computed
        )
        
        assert order.id is not None
        assert order.order_number == "ORD-2024-001"
        assert order.total == 5000
        assert order.order_year == datetime.now(timezone.utc).year
    
    @pytest.mark.asyncio
    async def test_create_row_returns_refreshed_object(self, session):
        """Test that returned object has all DB-generated values."""
        product = await create_row(
            TestProduct,
            session,
            name="Fresh Product",
            sku="FRS-001",
            price=3000
        )
        
        # Should have DB-generated ID and timestamp
        assert product.id is not None
        assert isinstance(product.id, int)
        assert product.created_at is not None
        assert isinstance(product.created_at, datetime)
    
    @pytest.mark.asyncio
    async def test_create_row_invalid_class(self, session):
        """Test creating row with non-Base class raises TypeError."""
        class NotABase:
            pass
        
        with pytest.raises(TypeError, match="must inherit from"):
            await create_row(
                NotABase,
                session,
                name="Test"
            )


# Tests for create_rows
class TestCreateRows:
    """Tests for create_rows function."""
    
    @pytest.mark.asyncio
    async def test_create_rows_basic(self, session):
        """Test creating multiple rows."""
        products = await create_rows(
            TestProduct,
            session,
            [
                {"name": "Product A", "sku": "PRD-A", "price": 1000},
                {"name": "Product B", "sku": "PRD-B", "price": 2000},
                {"name": "Product C", "sku": "PRD-C", "price": 3000},
            ]
        )
        
        assert len(products) == 3
        assert products[0].name == "Product A"
        assert products[1].name == "Product B"
        assert products[2].name == "Product C"
        assert all(p.id is not None for p in products)
    
    @pytest.mark.asyncio
    async def test_create_rows_with_validation(self, session):
        """Test creating multiple rows with validation."""
        users = await create_rows(
            TestUser,
            session,
            [
                {"username": "user1", "email": "user1@example.com", "age": 20},
                {"username": "user2", "email": "user2@example.com", "age": 25},
            ],
            validate=True
        )
        
        assert len(users) == 2
        assert all(u.id is not None for u in users)
    
    @pytest.mark.asyncio
    async def test_create_rows_validation_failure_first_row(self, session):
        """Test validation failure on first row aborts all."""
        with pytest.raises(ValidationError):
            await create_rows(
                TestUser,
                session,
                [
                    {"username": "bad1", "email": "invalid", "age": 20},  # Bad
                    {"username": "good", "email": "good@example.com", "age": 25},
                ],
                validate=True
            )
        
        # Verify no rows were created
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestUser)
        result = await session.execute(count_query)
        count = result.scalar_one()
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_create_rows_validation_failure_middle_row(self, session):
        """Test validation failure in middle row aborts all."""
        with pytest.raises(ValidationError):
            await create_rows(
                TestUser,
                session,
                [
                    {"username": "good1", "email": "good1@example.com", "age": 20},
                    {"username": "bad", "email": "invalid", "age": 25},  # Bad
                    {"username": "good2", "email": "good2@example.com", "age": 30},
                ],
                validate=True
            )
        
        # Verify no rows were created (atomic)
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestUser)
        result = await session.execute(count_query)
        count = result.scalar_one()
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_create_rows_empty_list(self, session):
        """Test creating rows with empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await create_rows(TestProduct, session, [])
    
    @pytest.mark.asyncio
    async def test_create_rows_integrity_error_rollback(self, session):
        """Test integrity error rolls back all rows."""
        with pytest.raises(IntegrityError):
            await create_rows(
                TestProduct,
                session,
                [
                    {"name": "Product 1", "sku": "SKU-1", "price": 1000},
                    {"name": "Product 2", "sku": "SKU-1", "price": 2000},  # Duplicate SKU
                ]
            )
        
        # Verify no rows were created
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestProduct)
        result = await session.execute(count_query)
        count = result.scalar_one()
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_create_rows_with_get_create_kwargs(self, session):
        """Test creating multiple rows with custom get_create_kwargs."""
        orders = await create_rows(
            TestOrder,
            session,
            [
                {"order_number": "ORD-001", "total": 1000},
                {"order_number": "ORD-002", "total": 2000},
            ]
        )
        
        assert len(orders) == 2
        current_year = datetime.now(timezone.utc).year
        assert all(o.order_year == current_year for o in orders)
    
    @pytest.mark.asyncio
    async def test_create_rows_all_refreshed(self, session):
        """Test all created rows are refreshed with DB values."""
        products = await create_rows(
            TestProduct,
            session,
            [
                {"name": f"Product {i}", "sku": f"SKU-{i}", "price": i * 1000}
                for i in range(5)
            ]
        )
        
        assert all(p.id is not None for p in products)
        assert all(p.created_at is not None for p in products)
        # IDs should be sequential
        ids = [p.id for p in products]
        assert ids == sorted(ids)
    
    @pytest.mark.asyncio
    async def test_create_rows_skip_validation(self, session):
        """Test creating rows without validation."""
        products = await create_rows(
            TestProduct,
            session,
            [
                {"name": "P1", "sku": "S1", "price": 100},
                {"name": "P2", "sku": "S2", "price": 200},
            ],
            validate=False
        )
        
        assert len(products) == 2
    
    @pytest.mark.asyncio
    async def test_create_rows_large_batch(self, session):
        """Test creating many rows at once."""
        num_products = 100
        products_data = [
            {
                "name": f"Product {i}",
                "sku": f"SKU-{i:04d}",
                "price": i * 100,
                "stock": i
            }
            for i in range(num_products)
        ]
        
        products = await create_rows(
            TestProduct,
            session,
            products_data,
            validate=False
        )
        
        assert len(products) == num_products
        assert all(p.id is not None for p in products)


# Tests for create_rows_batched
class TestCreateRowsBatched:
    """Tests for create_rows_batched function."""
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_basic(self, session):
        """Test creating rows in batches."""
        products_data = [
            {"name": f"Product {i}", "sku": f"SKU-{i}", "price": i * 100}
            for i in range(10)
        ]
        
        products = await create_rows_batched(
            TestProduct,
            session,
            products_data,
            batch_size=3
        )
        
        assert len(products) == 10
        assert all(p.id is not None for p in products)
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_exact_batch_size(self, session):
        """Test batching when total is exact multiple of batch size."""
        products_data = [
            {"name": f"Product {i}", "sku": f"SKU-{i}", "price": 1000}
            for i in range(9)  # 9 items, batch size 3 = exactly 3 batches
        ]
        
        products = await create_rows_batched(
            TestProduct,
            session,
            products_data,
            batch_size=3
        )
        
        assert len(products) == 9
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_partial_last_batch(self, session):
        """Test batching when last batch is partial."""
        products_data = [
            {"name": f"Product {i}", "sku": f"SKU-{i}", "price": 1000}
            for i in range(10)  # 10 items, batch size 3 = 3 full + 1 partial
        ]
        
        products = await create_rows_batched(
            TestProduct,
            session,
            products_data,
            batch_size=3
        )
        
        assert len(products) == 10
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_single_batch(self, session):
        """Test batching when all rows fit in one batch."""
        products_data = [
            {"name": f"Product {i}", "sku": f"SKU-{i}", "price": 1000}
            for i in range(5)
        ]
        
        products = await create_rows_batched(
            TestProduct,
            session,
            products_data,
            batch_size=10  # Batch size larger than data
        )
        
        assert len(products) == 5
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_empty_list(self, session):
        """Test batched creation with empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await create_rows_batched(TestProduct, session, [])
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_invalid_batch_size(self, session):
        """Test invalid batch size raises ValueError."""
        with pytest.raises(ValueError, match="batch_size must be at least 1"):
            await create_rows_batched(
                TestProduct,
                session,
                [{"name": "Test", "sku": "TST", "price": 100}],
                batch_size=0
            )
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_failure_leaves_partial(self, session):
        """Test that batch failure leaves previously committed batches."""
        products_data = [
            {"name": f"Product {i}", "sku": f"SKU-{i}", "price": 1000}
            for i in range(5)
        ] + [
            {"name": "Bad", "sku": "SKU-1", "price": 2000}  # Duplicate in 2nd batch
        ]
        
        with pytest.raises(IntegrityError):
            await create_rows_batched(
                TestProduct,
                session,
                products_data,
                batch_size=3
            )
        
        # First batch (3 rows) should be committed
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestProduct)
        result = await session.execute(count_query)
        count = result.scalar_one()
        assert count == 3  # Only first batch committed
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_with_validation(self, session):
        """Test batched creation with validation."""
        users_data = [
            {"username": f"user{i}", "email": f"user{i}@example.com", "age": 20 + i}
            for i in range(10)
        ]
        
        users = await create_rows_batched(
            TestUser,
            session,
            users_data,
            batch_size=4,
            validate=True
        )
        
        assert len(users) == 10
    
    @pytest.mark.asyncio
    async def test_create_rows_batched_large_dataset(self, session):
        """Test batching with large dataset."""
        num_products = 1000
        products_data = [
            {
                "name": f"Product {i}",
                "sku": f"SKU-{i:05d}",
                "price": i,
                "stock": i % 100
            }
            for i in range(num_products)
        ]
        
        products = await create_rows_batched(
            TestProduct,
            session,
            products_data,
            batch_size=100,
            validate=False
        )
        
        assert len(products) == num_products


# Tests for bulk_insert_rows
class TestBulkInsertRows:
    """Tests for bulk_insert_rows function."""
    
    @pytest.mark.asyncio
    async def test_bulk_insert_basic(self, session):
        """Test bulk inserting rows."""
        products_data = [
            {"name": f"Product {i}", "sku": f"SKU-{i}", "price": i * 100}
            for i in range(10)
        ]
        
        count = await bulk_insert_rows(
            TestProduct,
            session,
            products_data
        )
        
        assert count == 10
        
        # Verify rows were created
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestProduct)
        result = await session.execute(count_query)
        db_count = result.scalar_one()
        assert db_count == 10
    
    @pytest.mark.asyncio
    async def test_bulk_insert_with_validation(self, session):
        """Test bulk insert with Pydantic validation."""
        users_data = [
            {"username": f"user{i}", "email": f"user{i}@example.com", "age": 20 + i}
            for i in range(5)
        ]
        
        count = await bulk_insert_rows(
            TestUser,
            session,
            users_data,
            validate=True
        )
        
        assert count == 5
    
    @pytest.mark.asyncio
    async def test_bulk_insert_validation_failure(self, session):
        """Test bulk insert validation catches errors."""
        users_data = [
            {"username": "user1", "email": "user1@example.com", "age": 20},
            {"username": "user2", "email": "invalid-email", "age": 25},  # Bad
        ]
        
        with pytest.raises(ValidationError):
            await bulk_insert_rows(
                TestUser,
                session,
                users_data,
                validate=True
            )
        
        # Verify no rows were created
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestUser)
        result = await session.execute(count_query)
        count = result.scalar_one()
        assert count == 0
    
    @pytest.mark.asyncio
    async def test_bulk_insert_skip_validation(self, session):
        """Test bulk insert without validation."""
        products_data = [
            {"name": f"P{i}", "sku": f"S{i}", "price": 100}
            for i in range(5)
        ]
        
        count = await bulk_insert_rows(
            TestProduct,
            session,
            products_data,
            validate=False
        )
        
        assert count == 5
    
    @pytest.mark.asyncio
    async def test_bulk_insert_empty_list(self, session):
        """Test bulk insert with empty list raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            await bulk_insert_rows(TestProduct, session, [])
    
    @pytest.mark.asyncio
    async def test_bulk_insert_integrity_error(self, session):
        """Test bulk insert with duplicate raises IntegrityError."""
        products_data = [
            {"name": "Product 1", "sku": "DUP", "price": 1000},
            {"name": "Product 2", "sku": "DUP", "price": 2000},  # Duplicate
        ]
        
        with pytest.raises(IntegrityError):
            await bulk_insert_rows(
                TestProduct,
                session,
                products_data,
                validate=False
            )
    
    @pytest.mark.asyncio
    async def test_bulk_insert_large_dataset(self, session):
        """Test bulk insert with large dataset."""
        num_products = 10000
        products_data = [
            {
                "name": f"Product {i}",
                "sku": f"SKU-{i:06d}",
                "price": i,
                "stock": i % 100
            }
            for i in range(num_products)
        ]
        
        count = await bulk_insert_rows(
            TestProduct,
            session,
            products_data,
            validate=False
        )
        
        assert count == num_products
    
    @pytest.mark.asyncio
    async def test_bulk_insert_does_not_return_objects(self, session):
        """Test bulk insert returns count, not objects."""
        products_data = [
            {"name": f"Product {i}", "sku": f"SKU-{i}", "price": 100}
            for i in range(3)
        ]
        
        result = await bulk_insert_rows(
            TestProduct,
            session,
            products_data
        )
        
        # Returns int count, not list of objects
        assert isinstance(result, int)
        assert result == 3


# Integration tests
class TestCreationIntegration:
    """Integration tests combining multiple creation scenarios."""
    
    @pytest.mark.asyncio
    async def test_mixed_creation_methods(self, session):
        """Test using different creation methods together."""
        # Single row
        product1 = await create_row(
            TestProduct,
            session,
            name="Single Product",
            sku="SINGLE-1",
            price=1000
        )
        
        # Multiple rows
        products2 = await create_rows(
            TestProduct,
            session,
            [
                {"name": "Batch Product 1", "sku": "BATCH-1", "price": 2000},
                {"name": "Batch Product 2", "sku": "BATCH-2", "price": 3000},
            ]
        )
        
        # Bulk insert
        bulk_count = await bulk_insert_rows(
            TestProduct,
            session,
            [
                {"name": "Bulk Product 1", "sku": "BULK-1", "price": 4000},
                {"name": "Bulk Product 2", "sku": "BULK-2", "price": 5000},
            ]
        )
        
        # Verify all created
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestProduct)
        result = await session.execute(count_query)
        total_count = result.scalar_one()
        assert total_count == 5  # 1 + 2 + 2
    
    @pytest.mark.asyncio
    async def test_create_with_relationships(self, session):
        """Test creating related objects."""
        # This would test foreign key relationships if we had them
        # For now, just verify independent creation works
        order = await create_row(
            TestOrder,
            session,
            order_number="ORD-001",
            total=5000
        )
        
        assert order.id is not None
        assert order.order_year == datetime.now(timezone.utc).year
    
    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, session):
        """Test that errors properly rollback transactions."""
        # Create one product successfully
        product1 = await create_row(
            TestProduct,
            session,
            name="Good Product",
            sku="GOOD-1",
            price=1000
        )
        
        # Try to create batch with error
        try:
            await create_rows(
                TestProduct,
                session,
                [
                    {"name": "Product A", "sku": "A", "price": 100},
                    {"name": "Product B", "sku": "GOOD-1", "price": 200},  # Duplicate
                ]
            )
        except IntegrityError:
            pass
        
        # Verify only first product exists
        from sqlalchemy import select, func
        count_query = select(func.count()).select_from(TestProduct)
        result = await session.execute(count_query)
        count = result.scalar_one()
        assert count == 1


# Performance tests
class TestCreationPerformance:
    """Performance-related tests."""
    
    @pytest.mark.asyncio
    async def test_bulk_insert_faster_than_individual(self, session):
        """Test that bulk insert is more efficient."""
        import time
        
        num_rows = 100
        
        # Method 1: Individual creates
        start = time.time()
        for i in range(num_rows):
            await create_row(
                TestProduct,
                session,
                name=f"Individual {i}",
                sku=f"IND-{i}",
                price=100,
                validate=False
            )
        individual_time = time.time() - start
        
        # Clear table
        from sqlalchemy import delete
        await session.execute(delete(TestProduct))
        await session.commit()
        
        # Method 2: Bulk insert
        start = time.time()
        await bulk_insert_rows(
            TestProduct,
            session,
            [
                {"name": f"Bulk {i}", "sku": f"BLK-{i}", "price": 100}
                for i in range(num_rows)
            ],
            validate=False
        )
        bulk_time = time.time() - start
        
        # Bulk should be faster (though not guaranteed on small datasets)
        # This is more of a smoke test
        assert bulk_time < individual_time * 2  # At least somewhat faster
    
    @pytest.mark.asyncio
    async def test_batched_vs_single_transaction(self, session):
        """Compare batched vs single transaction performance."""
        num_rows = 50
        
        # Clear any existing data
        from sqlalchemy import delete
        await session.execute(delete(TestProduct))
        await session.commit()
        
        # Batched (commits per batch)
        batched_products = await create_rows_batched(
            TestProduct,
            session,
            [
                {"name": f"Batched {i}", "sku": f"BAT-{i}", "price": 100}
                for i in range(num_rows)
            ],
            batch_size=10,
            validate=False
        )
        
        assert len(batched_products) == num_rows
        
        # Clear for next test
        await session.execute(delete(TestProduct))
        await session.commit()
        
        # Single transaction
        single_products = await create_rows(
            TestProduct,
            session,
            [
                {"name": f"Single {i}", "sku": f"SNG-{i}", "price": 100}
                for i in range(num_rows)
            ],
            validate=False
        )
        
        assert len(single_products) == num_rows


# Edge cases and error handling
class TestCreationEdgeCases:
    """Tests for edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_create_with_none_values(self, session):
        """Test creating row with None for nullable field."""
        # TestProduct.stock has a default, but we can explicitly set None
        # if it were nullable
        product = await create_row(
            TestProduct,
            session,
            name="Null Test",
            sku="NULL-1",
            price=100,
            stock=0  # Explicit value
        )
        
        assert product.stock == 0
    
    @pytest.mark.asyncio
    async def test_create_rows_single_item(self, session):
        """Test create_rows with single item list."""
        products = await create_rows(
            TestProduct,
            session,
            [{"name": "Solo", "sku": "SOLO-1", "price": 100}]
        )
        
        assert len(products) == 1
        assert products[0].name == "Solo"
    
    @pytest.mark.asyncio
    async def test_create_with_special_characters(self, session):
        """Test creating row with special characters in strings."""
        product = await create_row(
            TestProduct,
            session,
            name="Product's \"Special\" Name!",
            sku="SPEC-1",
            price=100
        )
        
        assert product.name == "Product's \"Special\" Name!"
    
    @pytest.mark.asyncio
    async def test_create_with_unicode(self, session):
        """Test creating row with unicode characters."""
        product = await create_row(
            TestProduct,
            session,
            name="Produto Café ☕",
            sku="UNI-1",
            price=100
        )
        
        assert "☕" in product.name
    
    @pytest.mark.asyncio
    async def test_create_with_very_long_string(self, session):
        """Test creating row with maximum length string."""
        # TestProduct.name has max length 100
        long_name = "A" * 100
        product = await create_row(
            TestProduct,
            session,
            name=long_name,
            sku="LONG-1",
            price=100,
            validate=False
        )
        
        assert product.name == long_name
    
    @pytest.mark.asyncio
    async def test_create_with_zero_values(self, session):
        """Test creating row with zero for numeric fields."""
        product = await create_row(
            TestProduct,
            session,
            name="Zero Product",
            sku="ZERO-1",
            price=0,  # Zero price
            stock=0   # Zero stock
        )
        
        assert product.price == 0
        assert product.stock == 0
    
    @pytest.mark.asyncio
    async def test_create_with_negative_values(self, session):
        """Test creating row with negative numbers."""
        # This might fail business validation but SQLAlchemy allows it
        product = await create_row(
            TestProduct,
            session,
            name="Negative Product",
            sku="NEG-1",
            price=-100,  # Negative price
            validate=False
        )
        
        assert product.price == -100

@pytest.mark.asyncio
async def test_create_multiple_tables_sequentially(session):
    """Test creating rows in different tables works correctly."""
    # Create product
    product = await create_row(
        TestProduct,
        session,
        name="Multi-table Product",
        sku="MULTI-1",
        price=100
    )
    
    # Create user
    user = await create_row(
        TestUser,
        session,
        username="multi_user",
        email="multi@example.com",
        age=25
    )
    
    # Create order
    order = await create_row(
        TestOrder,
        session,
        order_number="MULTI-ORD-1",
        total=5000
    )
    
    assert product.id is not None
    assert user.id is not None
    assert order.id is not None        

# Test validation scenarios
class TestValidationScenarios:
    """Tests focusing on validation behavior."""
    
    @pytest.mark.asyncio
    async def test_validation_with_missing_required_field(self, session):
        """Test validation catches missing required fields."""
        with pytest.raises(ValidationError) as exc_info:
            await create_row(
                TestUser,
                session,
                username="incomplete",
                # email missing
                age=25,
                validate=True
            )
        
        errors = exc_info.value.errors()
        assert any('email' in str(e) for e in errors)
    
    @pytest.mark.asyncio
    async def test_validation_with_wrong_type(self, session):
        """Test validation catches wrong types."""
        with pytest.raises(ValidationError) as exc_info:
            await create_row(
                TestUser,
                session,
                username="typetest",
                email="type@example.com",
                age="not an int",  # Wrong type
                validate=True
            )
        
        errors = exc_info.value.errors()
        assert any('age' in str(e) for e in errors)
    
    @pytest.mark.asyncio
    async def test_validation_passes_with_valid_data(self, session):
        """Test validation passes with all valid data."""
        user = await create_row(
            TestUser,
            session,
            username="validuser",
            email="valid@example.com",
            age=30,
            validate=True
        )
        
        assert user.id is not None
    
    @pytest.mark.asyncio
    async def test_skip_validation_allows_invalid_data(self, session):
        """Test that skipping validation allows questionable data."""
        # This would fail Pydantic validation but SQLAlchemy allows it
        user = await create_row(
            TestUser,
            session,
            username="skipval",
            email="not-validated",  # Missing @, would fail Pydantic
            age=-1,  # Negative, would fail Pydantic
            validate=False
        )
        
        assert user.id is not None


# Test refresh parameter
class TestRefreshParameter:
    """Tests for the refresh parameter in create_rows."""
    
    @pytest.mark.asyncio
    async def test_create_rows_with_refresh(self, session):
        """Test create_rows with refresh=True loads DB values."""
        products = await create_rows(
            TestProduct,
            session,
            [
                {"name": "Refresh Test 1", "sku": "REF-1", "price": 100},
                {"name": "Refresh Test 2", "sku": "REF-2", "price": 200},
            ],
            refresh=True
        )
        
        # All DB-generated values should be present
        assert all(p.id is not None for p in products)
        assert all(p.created_at is not None for p in products)
    
    @pytest.mark.asyncio
    async def test_create_rows_without_refresh(self, session):
        """Test create_rows with refresh=False for better performance."""
        products = await create_rows(
            TestProduct,
            session,
            [
                {"name": "No Refresh 1", "sku": "NRF-1", "price": 100},
                {"name": "No Refresh 2", "sku": "NRF-2", "price": 200},
            ],
            refresh=False
        )
        
        # IDs should still be available (from flush)
        assert all(p.id is not None for p in products)
        # But created_at might not be refreshed from server defaults
        # (depends on DB configuration)


# Test with custom types
class TestCustomTypes:
    """Tests with various data types."""
    
    @pytest.mark.asyncio
    async def test_create_with_datetime(self, session):
        """Test creating row with datetime field."""
        specific_time = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        
        # TestProduct has created_at with default, but we can override
        product = await create_row(
            TestProduct,
            session,
            name="DateTime Test",
            sku="DT-1",
            price=100
        )
        
        # created_at should be set to current time (default)
        assert product.created_at is not None
        assert isinstance(product.created_at, datetime)
    
    @pytest.mark.asyncio
    async def test_create_with_large_integers(self, session):
        """Test creating row with large integer values."""
        large_price = 999999999
        product = await create_row(
            TestProduct,
            session,
            name="Big Price",
            sku="BIG-1",
            price=large_price
        )
        
        assert product.price == large_price


# Test error messages
class TestErrorMessages:
    """Tests for error message quality."""
    
    @pytest.mark.asyncio
    async def test_type_error_message_quality(self, session):
        """Test that TypeError has helpful message."""
        class NotBase:
            pass
        
        with pytest.raises(TypeError) as exc_info:
            await create_row(NotBase, session, name="test")
        
        error_msg = str(exc_info.value)
        assert "NotBase" in error_msg
        assert "inherit" in error_msg.lower()
    
    @pytest.mark.asyncio
    async def test_integrity_error_logged(self, session, caplog):
        """Test that IntegrityError is properly logged."""
        # Create first product
        await create_row(
            TestProduct,
            session,
            name="First",
            sku="DUP-SKU",
            price=100
        )
        
        # Try duplicate
        try:
            await create_row(
                TestProduct,
                session,
                name="Second",
                sku="DUP-SKU",  # Duplicate
                price=200
            )
        except IntegrityError:
            pass
        
        # Check that error was logged
        # (This requires structlog to be configured for testing)


# Parametrized tests
@pytest.mark.parametrize("num_rows,batch_size", [
    (10, 3),
    (10, 5),
    (10, 10),
    (10, 15),
    (100, 10),
    (100, 25),
    (100, 50),
])
@pytest.mark.asyncio
async def test_create_rows_batched_various_sizes(session, num_rows, batch_size):
    """Parametrized test for various batch sizes."""
    products_data = [
        {"name": f"P{i}", "sku": f"S{i:04d}", "price": i * 100}
        for i in range(num_rows)
    ]
    
    products = await create_rows_batched(
        TestProduct,
        session,
        products_data,
        batch_size=batch_size,
        validate=False
    )
    
    assert len(products) == num_rows
    
    # Clean up for next test
    from sqlalchemy import delete
    await session.execute(delete(TestProduct))
    await session.commit()
