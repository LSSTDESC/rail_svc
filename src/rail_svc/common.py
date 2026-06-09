"""Common utilities and functions"""

from typing import Any

def unexpected(condition: Any) -> Any:
    """A simple wrapper to mark evaluations that we don't expected to fail"""
    return condition
