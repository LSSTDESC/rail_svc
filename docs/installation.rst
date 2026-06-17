Installation
============

Requirements
------------

- Python 3.13+
- SQLite (for local mode) or PostgreSQL (for production)

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/LSSTDESC/rail_pz_svc.git
   cd rail_pz_svc
   pip install -e '.[all]'

Optional dependency groups:

- ``[db]`` — Database support (SQLAlchemy, aiosqlite)
- ``[server]`` — FastAPI server (includes ``[db]``)
- ``[client]`` — HTTP client (httpx)
- ``[dev]`` — Development tools (pytest, ruff, mypy)
- ``[docs]`` — Documentation (sphinx)
- ``[all]`` — Everything

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
