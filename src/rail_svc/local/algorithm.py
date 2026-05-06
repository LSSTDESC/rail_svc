"""Local operations for Algorithm table.

Provides simplified API with automatic session management.
"""

from .. import db_oper  # or dp_oper if that's correct
from .base import create_local_operations

# Create local operations - this has all the methods via dynamic binding!
algorithm = create_local_operations(db_oper.algorithm)

# Re-export for convenience
__all__ = ['algorithm']
