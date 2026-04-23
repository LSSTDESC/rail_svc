# test_async_setup.py
import pytest
import pytest_asyncio

import asyncio

@pytest_asyncio.fixture
async def simple_fixture():
    return "test"


@pytest.mark.asyncio
async def test_async_works(simple_fixture):
    assert simple_fixture == "test"
    await asyncio.sleep(0)  # Verify async works
