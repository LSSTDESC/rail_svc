from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from rail_svc.db.base import Base


class DbTestProduct(Base):
    """Test product model for creation tests."""

    __tablename__ = "test_products"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    sku: Mapped[str] = mapped_column(String(50), unique=True)
    price: Mapped[int] = mapped_column(Integer)
    stock: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

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


class DbTestOrder(Base):
    """Test order model with custom get_create_kwargs."""

    __tablename__ = "test_orders"

    id: Mapped[int] = mapped_column(primary_key=True)
    order_number: Mapped[str] = mapped_column(String(50), unique=True)
    total: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    # Computed field
    order_year: Mapped[int] = mapped_column(Integer)

    @classmethod
    async def get_create_kwargs(cls, session, **kwargs):
        """Add computed order_year field."""
        if "order_year" not in kwargs:
            kwargs["order_year"] = datetime.now(UTC).year
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


class DbTestUser(Base):
    """Test user model with validation."""

    __tablename__ = "test_users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)
    age: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(50), default="")
    role: Mapped[str] = mapped_column(String(50), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, default=None)

    @classmethod
    def pydantic_model_class(cls):
        """Return Pydantic model class with validation."""
        from pydantic import field_validator

        class UserSchema(BaseModel):
            id: int | None = None
            username: str
            email: str
            age: int

            @field_validator("age")
            @classmethod
            def age_must_be_positive(cls, v):
                if v < 0:
                    raise ValueError("age must be positive")
                return v

            @field_validator("email")
            @classmethod
            def email_must_contain_at(cls, v):
                if "@" not in v:
                    raise ValueError("email must contain @")
                return v

            class Config:
                from_attributes = True

        return UserSchema


class DbTestArticle(Base):
    """Test article model for reading tests."""

    __tablename__ = "test_articles"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    title: Mapped[str] = mapped_column(String(200))
    author: Mapped[str] = mapped_column(String(100))
    views: Mapped[int] = mapped_column(Integer, default=0)
    published: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

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


class DbTestBook(Base):
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
