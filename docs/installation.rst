Installation
============

Requirements
------------

- Python 3.13+
- `macon <https://github.com/KIPAC/macon>`_ (installed automatically as dependency)
- SQLite (for local mode) or PostgreSQL (for production)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/LSSTDESC/rail_pz_svc.git
   cd rail_pz_svc
   pip install -e '.[all]'

Optional dependency groups:

- ``[all]`` — Everything (recommended for development)
- ``[dev]`` — Development tools (pytest, ruff, mypy)
- ``[docs]`` — Documentation (sphinx)

The core dependencies (database, server, client) are provided transitively
through ``macon[db,server,client]``.

Configuration
-------------

Environment variables (use ``__`` as nested delimiter):

.. list-table::
   :header-rows: 1

   * - Variable
     - Description
     - Default
   * - ``DB__URL``
     - Database connection URL
     - ``sqlite+aiosqlite:///rail_svc.db``
   * - ``STORAGE__ARCHIVE``
     - Path for archived data files
     - ``archive``
   * - ``STORAGE__IMPORT_AREA``
     - Path for import staging
     - ``import``
   * - ``PZ_RAIL_SERVICE``
     - Remote service URL (client mode)
     -
   * - ``PZ_RAIL_TOKEN``
     - Auth token (client mode)
     -
