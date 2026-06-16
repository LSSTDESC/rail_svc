"""Unit tests for CatalogTag database model"""

import pytest
from sqlalchemy import select

from rail_svc.db.catalog_tag import CatalogTag
from rail_svc.models import CatalogTag as CatalogTagPydantic
from rail_svc.models import CatalogTagCreate

# ============================================================================
# CatalogTag Class Tests
# ============================================================================


class TestCatalogTag:
    """Tests for CatalogTag database model"""

    def test_catalog_tag_tablename(self):
        """Test CatalogTag has correct table name"""
        assert CatalogTag.__tablename__ == "catalog_tag"

    def test_catalog_tag_class_string(self):
        """Test CatalogTag.class_string() returns table name"""
        assert CatalogTag.class_string() == "catalog_tag"

    def test_pydantic_create_class(self):
        """Test CatalogTag.pydantic_create_class() returns correct model"""
        assert CatalogTag.pydantic_create_class() == CatalogTagCreate

    def test_pydantic_model_class(self):
        """Test CatalogTag.pydantic_model_class() returns correct model"""
        assert CatalogTag.pydantic_model_class() == CatalogTagPydantic

    @pytest.mark.asyncio
    async def test_create_catalog_tag(self, session):
        """Test creating a CatalogTag instance"""
        tag = CatalogTag(name="des_y6")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)

        assert tag.id_ is not None
        assert tag.name == "des_y6"

    @pytest.mark.asyncio
    async def test_catalog_tag_unique_name(self, session, sample_catalog_tag):
        """Test that catalog tag name must be unique"""
        duplicate = CatalogTag(name=sample_catalog_tag.name)
        session.add(duplicate)

        with pytest.raises(Exception):  # IntegrityError or similar
            await session.commit()

    @pytest.mark.asyncio
    async def test_query_catalog_tag_by_name(self, session, sample_catalog_tag):
        """Test querying catalog tag by name"""
        result = await session.execute(select(CatalogTag).where(CatalogTag.name == sample_catalog_tag.name))
        tag = result.scalar_one()
        assert tag.id_ == sample_catalog_tag.id_
        assert tag.name == sample_catalog_tag.name

    @pytest.mark.asyncio
    async def test_query_catalog_tag_by_id(self, session, sample_catalog_tag):
        """Test querying catalog tag by id"""
        tag = await session.get(CatalogTag, sample_catalog_tag.id_)
        assert tag is not None
        assert tag.name == sample_catalog_tag.name

    @pytest.mark.asyncio
    async def test_update_catalog_tag(self, session, sample_catalog_tag):
        """Test updating a CatalogTag"""
        new_name = "updated_catalog"
        sample_catalog_tag.name = new_name
        await session.commit()
        await session.refresh(sample_catalog_tag)

        assert sample_catalog_tag.name == new_name

    @pytest.mark.asyncio
    async def test_delete_catalog_tag(self, session, sample_catalog_tag):
        """Test deleting a CatalogTag"""
        tag_id = sample_catalog_tag.id_
        await session.delete(sample_catalog_tag)
        await session.commit()

        result = await session.get(CatalogTag, tag_id)
        assert result is None

    def test_catalog_tag_repr(self, sample_catalog_tag):
        """Test CatalogTag __repr__ method"""
        repr_str = repr(sample_catalog_tag)
        assert "CatalogTag" in repr_str
        assert str(sample_catalog_tag.id_) in repr_str
        assert sample_catalog_tag.name in repr_str

    def test_catalog_tag_str(self, sample_catalog_tag):
        """Test CatalogTag __str__ method"""
        assert str(sample_catalog_tag) == sample_catalog_tag.name


class TestCatalogTagValidation:
    """Tests for CatalogTag field validation"""

    @pytest.mark.asyncio
    async def test_catalog_tag_requires_name(self, session):
        """Test that CatalogTag requires a name"""
        with pytest.raises(Exception):  # IntegrityError
            tag = CatalogTag()
            session.add(tag)
            await session.commit()

    @pytest.mark.asyncio
    async def test_catalog_tag_name_indexed(self, session):
        """Test that name field is indexed"""
        name_column = CatalogTag.__table__.c.name
        assert name_column.index is True
        assert name_column.unique is True


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================


# ============================================================================
# Batch Operations Tests
# ============================================================================


class TestCatalogTagBatch:
    """Tests for batch operations"""

    @pytest.mark.asyncio
    async def test_bulk_insert(self, session):
        """Test inserting multiple catalog tags at once"""
        tags = [CatalogTag(name=f"catalog_{i}") for i in range(10)]

        session.add_all(tags)
        await session.commit()

        for tag in tags:
            await session.refresh(tag)

        assert all(tag.id_ is not None for tag in tags)
        assert len(tags) == 10

    @pytest.mark.asyncio
    async def test_bulk_query(self, session, multiple_catalog_tags):
        """Test querying multiple catalog tags"""
        result = await session.execute(select(CatalogTag))
        tags = result.scalars().all()

        assert len(tags) >= 3
        names = {tag.name for tag in tags}
        assert "roman" in names
        assert "rubin" in names
        assert "hsc_pdr3" in names

    @pytest.mark.asyncio
    async def test_bulk_update(self, session, multiple_catalog_tags):
        """Test updating multiple catalog tags"""
        for tag in multiple_catalog_tags:
            tag.name = f"updated_{tag.name}"

        await session.commit()

        for tag in multiple_catalog_tags:
            await session.refresh(tag)
            assert tag.name.startswith("updated_")

    @pytest.mark.asyncio
    async def test_bulk_delete(self, session, multiple_catalog_tags):
        """Test deleting multiple catalog tags"""
        tag_ids = [tag.id_ for tag in multiple_catalog_tags]

        for tag in multiple_catalog_tags:
            await session.delete(tag)

        await session.commit()

        for tag_id in tag_ids:
            result = await session.get(CatalogTag, tag_id)
            assert result is None


class TestCatalogTagQueries:
    """Tests for various query patterns"""

    @pytest.mark.asyncio
    async def test_query_by_name_pattern(self, session, multiple_catalog_tags):
        """Test querying catalog tags with name pattern matching"""
        result = await session.execute(select(CatalogTag).where(CatalogTag.name.like("r%")))
        tags = result.scalars().all()

        assert len(tags) >= 2  # roman, rubin
        names = {tag.name for tag in tags}
        assert "roman" in names or "rubin" in names

    @pytest.mark.asyncio
    async def test_query_order_by_name(self, session, multiple_catalog_tags):
        """Test querying catalog tags ordered by name"""
        result = await session.execute(select(CatalogTag).order_by(CatalogTag.name))
        tags = result.scalars().all()

        # Check that results are ordered
        names = [tag.name for tag in tags]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_query_with_limit(self, session, multiple_catalog_tags):
        """Test querying catalog tags with limit"""
        result = await session.execute(select(CatalogTag).limit(2))
        tags = result.scalars().all()

        assert len(tags) <= 2

    @pytest.mark.asyncio
    async def test_count_catalog_tags(self, session, multiple_catalog_tags):
        """Test counting total number of catalog tags"""
        from sqlalchemy import func

        result = await session.execute(select(func.count()).select_from(CatalogTag))
        count = result.scalar()

        assert count >= 3

    @pytest.mark.asyncio
    async def test_query_case_sensitive_name(self, session):
        """Test that name queries are case-sensitive"""
        tag1 = CatalogTag(name="TestCatalog")
        tag2 = CatalogTag(name="testcatalog")
        session.add_all([tag1, tag2])
        await session.commit()

        result = await session.execute(select(CatalogTag).where(CatalogTag.name == "TestCatalog"))
        found_tag = result.scalar_one()

        assert found_tag.name == "TestCatalog"


class TestCatalogTagDataIntegrity:
    """Tests for data integrity and consistency"""

    @pytest.mark.asyncio
    async def test_catalog_tag_persistence(self, session, sample_catalog_tag):
        """Test that catalog tag data persists correctly"""
        tag_id = sample_catalog_tag.id_
        tag_name = sample_catalog_tag.name

        # Clear session
        await session.commit()
        session.expire_all()

        # Query fresh from database
        result = await session.get(CatalogTag, tag_id)
        assert result is not None
        assert result.name == tag_name

    @pytest.mark.asyncio
    async def test_catalog_tag_name_modification_persistence(self, session):
        """Test that name modifications persist correctly"""
        tag = CatalogTag(name="original_name")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)

        tag.name = "modified_name"
        await session.commit()

        # Query fresh from database
        result = await session.get(CatalogTag, tag.id_)
        assert result.name == "modified_name"


class TestCatalogTagPydanticValidation:
    """Tests for Pydantic model integration and validation"""

    @pytest.mark.asyncio
    async def test_to_pydantic_list(self, multiple_catalog_tags):
        """Test converting multiple catalog tags to Pydantic list"""
        pydantic_list = CatalogTag.to_pydantic_list(multiple_catalog_tags)

        assert len(pydantic_list) == 3
        assert all(isinstance(obj, CatalogTagPydantic) for obj in pydantic_list)
        assert pydantic_list[0].name == "roman"
        assert pydantic_list[1].name == "rubin"
        assert pydantic_list[2].name == "hsc_pdr3"

    @pytest.mark.asyncio
    async def test_to_pydantic_dict_list(self, multiple_catalog_tags):
        """Test converting multiple catalog tags to dict list"""
        dict_list = CatalogTag.to_pydantic_dict_list(multiple_catalog_tags)

        assert len(dict_list) == 3
        assert all(isinstance(d, dict) for d in dict_list)
        assert all("name" in d for d in dict_list)
        assert all("id_" in d for d in dict_list)


class TestCatalogTagNaming:
    """Tests for catalog tag naming conventions"""

    @pytest.mark.asyncio
    async def test_catalog_tag_with_uppercase(self, session):
        """Test CatalogTag with uppercase letters"""
        tag = CatalogTag(name="LSST")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)

        assert tag.name == "LSST"

    @pytest.mark.asyncio
    async def test_catalog_tag_with_mixed_case(self, session):
        """Test CatalogTag with mixed case"""
        tag = CatalogTag(name="RomanSpaceTelescope")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)

        assert tag.name == "RomanSpaceTelescope"

    @pytest.mark.asyncio
    async def test_catalog_tag_with_dots(self, session):
        """Test CatalogTag name with dots"""
        tag = CatalogTag(name="dp.0.2")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)

        assert tag.name == "dp.0.2"

    @pytest.mark.asyncio
    async def test_catalog_tag_with_hyphens(self, session):
        """Test CatalogTag name with hyphens"""
        tag = CatalogTag(name="data-release-1")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)

        assert tag.name == "data-release-1"


class TestCatalogTagExistence:
    """Tests for checking catalog tag existence"""

    @pytest.mark.asyncio
    async def test_check_catalog_tag_exists_by_name(self, session, sample_catalog_tag):
        """Test checking if catalog tag exists by name"""
        result = await session.execute(select(CatalogTag).where(CatalogTag.name == sample_catalog_tag.name))
        tag = result.scalar_one_or_none()

        assert tag is not None
        assert tag.id_ == sample_catalog_tag.id_

    @pytest.mark.asyncio
    async def test_check_catalog_tag_not_exists(self, session):
        """Test checking for non-existent catalog tag"""
        result = await session.execute(select(CatalogTag).where(CatalogTag.name == "nonexistent_catalog"))
        tag = result.scalar_one_or_none()

        assert tag is None


class TestCatalogTagFiltering:
    """Tests for filtering catalog tags"""

    @pytest.mark.asyncio
    async def test_filter_multiple_conditions(self, session, multiple_catalog_tags):
        """Test filtering with multiple conditions"""
        # Add some tags with specific patterns
        tag1 = CatalogTag(name="test_prod_v1")
        tag2 = CatalogTag(name="test_prod_v2")
        tag3 = CatalogTag(name="test_dev_v1")
        session.add_all([tag1, tag2, tag3])
        await session.commit()

        result = await session.execute(select(CatalogTag).where(CatalogTag.name.like("test_prod%")))
        tags = result.scalars().all()

        assert len(tags) == 2
        names = {tag.name for tag in tags}
        assert "test_prod_v1" in names
        assert "test_prod_v2" in names

    @pytest.mark.asyncio
    async def test_filter_exclude_pattern(self, session, multiple_catalog_tags):
        """Test filtering to exclude a pattern"""
        result = await session.execute(select(CatalogTag).where(~CatalogTag.name.like("roman%")))
        tags = result.scalars().all()

        # Should get tags that don't start with "roman"
        names = {tag.name for tag in tags}
        assert "roman" not in names or all(not name.startswith("roman") for name in names)


class TestCatalogTagSorting:
    """Tests for sorting catalog tags"""

    @pytest.mark.asyncio
    async def test_sort_ascending(self, session, multiple_catalog_tags):
        """Test sorting catalog tags in ascending order"""
        result = await session.execute(select(CatalogTag).order_by(CatalogTag.name.asc()))
        tags = result.scalars().all()

        names = [tag.name for tag in tags]
        assert names == sorted(names)

    @pytest.mark.asyncio
    async def test_sort_descending(self, session, multiple_catalog_tags):
        """Test sorting catalog tags in descending order"""
        result = await session.execute(select(CatalogTag).order_by(CatalogTag.name.desc()))
        tags = result.scalars().all()

        names = [tag.name for tag in tags]
        assert names == sorted(names, reverse=True)

    @pytest.mark.asyncio
    async def test_sort_by_id(self, session, multiple_catalog_tags):
        """Test sorting catalog tags by id"""
        result = await session.execute(select(CatalogTag).order_by(CatalogTag.id_))
        tags = result.scalars().all()

        ids = [tag.id_ for tag in tags]
        assert ids == sorted(ids)


class TestCatalogTagPagination:
    """Tests for pagination"""

    @pytest.mark.asyncio
    async def test_pagination_first_page(self, session):
        """Test getting first page of results"""
        # Create many tags
        tags = [CatalogTag(name=f"page_test_{i}") for i in range(20)]
        session.add_all(tags)
        await session.commit()

        result = await session.execute(select(CatalogTag).order_by(CatalogTag.name).limit(10).offset(0))
        page1 = result.scalars().all()

        assert len(page1) == 10

    @pytest.mark.asyncio
    async def test_pagination_second_page(self, session):
        """Test getting second page of results"""
        # Create many tags
        tags = [CatalogTag(name=f"page_test_{i:02d}") for i in range(20)]
        session.add_all(tags)
        await session.commit()

        result = await session.execute(select(CatalogTag).order_by(CatalogTag.name).limit(10).offset(10))
        page2 = result.scalars().all()

        assert len(page2) == 10

    @pytest.mark.asyncio
    async def test_pagination_boundary(self, session):
        """Test pagination at boundary"""
        # Create exact number of tags
        tags = [CatalogTag(name=f"boundary_{i}") for i in range(15)]
        session.add_all(tags)
        await session.commit()

        result = await session.execute(select(CatalogTag).order_by(CatalogTag.name).limit(10).offset(10))
        page = result.scalars().all()

        assert len(page) == 5  # Only 5 remaining


class TestCatalogTagReprStr:
    """Tests for __repr__ and __str__ edge cases"""

    @pytest.mark.asyncio
    async def test_repr_with_special_characters(self, session):
        """Test __repr__ with special characters in name"""
        tag = CatalogTag(name="test'catalog\"with\\special")
        session.add(tag)
        await session.commit()
        await session.refresh(tag)

        repr_str = repr(tag)
        assert "CatalogTag" in repr_str
        assert str(tag.id_) in repr_str

    @pytest.mark.asyncio
    async def test_str_returns_name_only(self, sample_catalog_tag):
        """Test that __str__ returns only the name"""
        str_repr = str(sample_catalog_tag)
        assert str_repr == sample_catalog_tag.name
        assert "CatalogTag" not in str_repr
        assert str(sample_catalog_tag.id_) not in str_repr
