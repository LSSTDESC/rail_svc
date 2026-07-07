Architecture
============

Framework Dependency
--------------------

``pz-rail-svc`` uses the `macon <https://github.com/KIPAC/macon>`_ package for
all generic database/service framework functionality:

- ORM base class, session management, CRUD functions
- Table operations with lifecycle hooks and file validation
- Local async/sync operation wrappers
- FastAPI router factory
- Remote HTTP client and sync wrappers
- Configuration, filtering models, CLI scaffolding

This repository contains only **domain-specific code**: table definitions,
pydantic models, photo-z estimation logic, and application assembly.

Layer Stack
-----------

.. code-block:: text

   ┌─────────────────────────────────────────────────────┐
   │  CLI (local / remote / server)                      │
   ├─────────────────────────────────────────────────────┤
   │  local_sync / remote_sync                           │
   ├─────────────────────────────────────────────────────┤
   │  local_async / remote_async                         │
   ├───────────────────────┬─────────────────────────────┤
   │  db_oper              │  client (httpx)             │
   ├───────────────────────┤                             │
   │  macon.db_funcs       │                             │
   ├───────────────────────┼─────────────────────────────┤
   │  db (SQLAlchemy ORM)  │  router (FastAPI)           │
   ├───────────────────────┴─────────────────────────────┤
   │  macon (framework)                                  │
   └─────────────────────────────────────────────────────┘

Domain Layers
^^^^^^^^^^^^^

1. **db/** — SQLAlchemy ORM models inheriting from ``macon.db.base.Base``.
2. **models/** — Pydantic models (``FooCreate`` for input, ``Foo`` for response).
3. **db_oper/** — Domain operations: foreign key resolution, file validation, composite workflows.
4. **local_async/** — Domain ``LocalOperations`` subclasses with load/read_slice.
5. **local_sync/** — Sync wrappers for domain operations.
6. **router/** — Domain routers plus custom load/download/read_slice endpoints.
7. **client/** — Domain remote client subclasses with read_slice.
8. **remote_async/** / **remote_sync/** — Domain remote operation subclasses.
9. **cli/** — Click-based CLIs for local, remote, and server modes.
10. **rail_funcs/** — Photo-z domain logic (RAIL wrappers, catalog I/O).

Domain Tables
-------------

.. list-table::
   :header-rows: 1

   * - Entity
     - Description
     - Extended Operations
   * - Algorithm
     - Python class implementing a p(z) estimation algorithm
     -
   * - Band
     - Photometric band with wavelength/transmission data
     -
   * - Sed
     - Spectral energy distribution
     -
   * - FilterAB
     - Redshift-dependent AB flux for a (band, sed) pair
     -
   * - CatalogTag
     - Named catalog configuration (column mappings)
     -
   * - CatalogBandAssoc
     - Associates bands with catalog tags
     -
   * - Dataset
     - Catalog data file with n_objects
     - load, read_slice, download
   * - DatasetAssoc
     - Links matched datasets to component datasets
     -
   * - Model
     - Trained model file for an algorithm + catalog tag
     - load, download
   * - Estimator
     - Algorithm + Model + config parameters
     -
   * - Estimates
     - Output p(z) file for an estimator + dataset
     - load, read_slice, download

Key Design Patterns
-------------------

Imports from macon
^^^^^^^^^^^^^^^^^^

Domain code imports framework classes directly from macon::

    from macon.db.base import Base
    from macon.db_oper.base import TableOperations, FileValidatedOperations, TableContext
    from macon.local_async.base import LocalOperations, with_session
    from macon.config import config as global_config

Generic Type Parameters
^^^^^^^^^^^^^^^^^^^^^^^

Operations classes use PEP 695 generics:
``[T: Base, ResponseT: BaseModel, CreateT: BaseModel]``

Parallel Interfaces
^^^^^^^^^^^^^^^^^^^

``local_async``, ``local_sync``, ``remote_async``, ``remote_sync`` all expose
identical CRUD methods per table (``get_row``, ``get_rows``, ``create_row``,
``update_row``, ``delete_row``, ``filter_rows``, ``find_by``, etc.).

Router Factory
^^^^^^^^^^^^^^

``macon.router.base.create_table_router(name, operations)`` generates a complete
CRUD + filter router from a ``LocalOperations`` instance.

Configuration
^^^^^^^^^^^^^

``macon.config`` provides pydantic-settings with ``__`` delimiter for nested env vars.
