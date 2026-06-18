# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pz-rail-svc` is a service for managing photometric redshift (photo-z) estimation workflows for the LSST DESC (Dark Energy Science Collaboration). It provides a database-backed catalog of algorithms, models, datasets, estimators, and photo-z estimates, accessible via local Python API, REST API, or CLI.

## Build & Development Commands

```bash
# Bootstrap (creates .venv via uv, installs deps)
make init

# Run all tests (uses in-memory SQLite)
pytest

# Run a single test file
pytest tests/db/test_db_algorithm.py -vvv

# Run a single test
pytest tests/db/test_db_algorithm.py::test_function_name -vvv

# Lint (pre-commit: ruff format + ruff lint + isort + yamllint)
make lint

# Type checking
make typing

# Start the FastAPI server (local SQLite)
DB__URL=sqlite+aiosqlite:////$(pwd)/rail_svc.db pz-rail-svc-server

# CLI (local database operations)
pz-rail-svc-local --help

# CLI (remote server operations)
pz-rail-svc-remote --help

# Build documentation
cd docs && make html
```

## Architecture

The codebase has a layered architecture with parallel sync/async and local/remote access paths:

### Layer Stack (bottom to top)

1. **`db/`** — SQLAlchemy ORM models inheriting from `db.base.Base`. Each table class implements `pydantic_create_class()` and `pydantic_model_class()` plus optional lifecycle hooks (pre/after create/update/delete).

2. **`models/`** — Pydantic models for validation and serialization. Each table has a `FooCreate` (input) and `Foo` (response) model. `models/web.py` has request/response types for the API.

3. **`db_funcs/`** — Pure async database functions (CRUD, filtering) that operate on `(db_class, session, ...)`. Stateless, reusable across layers.

4. **`db_oper/`** — `TableOperations` class that binds a db class + session and delegates to `db_funcs`. Adds lifecycle hook execution and file handling (load/download). Uses `@forward_to_db_funcs` decorator pattern.

5. **`local_async/`** — `LocalOperations` wraps `TableOperations` with automatic session management via `@with_session` decorator. Module-level singletons (e.g., `local_async.algorithm`).

6. **`local_sync/`** — `SyncOperations` wraps `local_async` with `asyncio.run()` via `@sync_wrapper` decorator. Used by CLI.

7. **`router/`** — FastAPI routers. `create_table_router()` generates full CRUD+filter endpoints for any table. Custom endpoints for `dataset`, `estimates`, `model` (load, read_slice, download).

8. **`client/`** — `RemoteTableOperations` (httpx async client mirroring the router API). `RemoteAPI` manages shared httpx client.

9. **`remote_async/`** — `AsyncRemoteOperations` wraps `client/` with context-manager-based lifecycle. Module-level singletons.

10. **`remote_sync/`** — Sync wrappers around `remote_async/`, same pattern as `local_sync/`.

11. **`cli/`** — Click-based CLIs: `local/` (direct DB), `remote/` (via HTTP), `server/` (starts uvicorn).

### Key Design Patterns

- **Generic type parameters**: Operations classes use `[T: Base, ResponseT: BaseModel, CreateT: BaseModel]` throughout for type safety.
- **Parallel interfaces**: `local_async`, `local_sync`, `remote_async`, `remote_sync` all expose the same CRUD methods per table.
- **Router factory**: `create_table_router(name, operations)` generates a complete CRUD router from a `LocalOperations` instance.
- **Configuration**: `config.py` uses pydantic-settings with `__` delimiter for nested env vars (e.g., `DB__URL`, `STORAGE__ARCHIVE`).

### Domain Tables

Algorithm, Band, CatalogTag, CatalogBandAssoc, Dataset, DatasetAssoc, Estimates, Estimator, FilterAB, Model, Sed. The `Dataset`, `Estimates`, and `Model` tables have extended operations for file handling (load, read_slice, download). The `FilterAB` table has a `get_create_kwargs` that resolves Band and Sed by name or ID.

### Spectral Data Tables

- **`Band`** — Filter transmission curves. Loaded from two-column `.res` files (wavelength, transmission) via `rail_funcs.catalog_funcs.read_band_res_file()`.
- **`Sed`** — Spectral energy distributions. Loaded from two-column `.sed` files (wavelength, sed_value) via `rail_funcs.catalog_funcs.read_sed_file()`. Batch loading via `make_sed_create_models(sed_dir, names_file=...)`.
- **`FilterAB`** — Redshift-dependent AB fluxes for a (band, sed) pair. Loaded from two-column `.AB` files (redshift, flux) with naming convention `{sed}.{band}.AB`. Batch loading via `make_filter_ab_create_models(filter_ab_dir)`. Foreign keys to Band and Sed are resolved by name at creation time.

### Composite Operations (`db_oper/catalog_funcs.py`, `local_async/funcs.py`, `local_sync/funcs.py`)

- `load_catalog_yaml(catalog_yaml, filter_dir)` — loads bands, catalog tags, and associations from YAML
- `load_seds(sed_dir, names=, names_file=)` — loads .sed files into Sed table
- `load_filter_abs(filter_ab_dir, names=)` — loads .AB files into FilterAB table
- `create_matched_dataset(...)` — creates a collection dataset with component associations
- `get_data_and_estimates_data(dataset_id, row)` — retrieves catalog + all photo-z estimates for one object

## Configuration

Environment variables with `__` as nested delimiter:
- `DB__URL` — database URL (default: `sqlite+aiosqlite:///rail_svc.db`)
- `STORAGE__ARCHIVE` — path for archived data files
- `STORAGE__IMPORT_AREA` — path for import files
- `PZ_RAIL_SERVICE` — remote service URL for client
- `PZ_RAIL_TOKEN` — auth token for remote client

## Testing

- Tests use in-memory SQLite via async fixtures in `tests/conftest.py`
- pytest-asyncio with `asyncio_mode = "auto"` — all async test functions run automatically
- Coverage is collected on `src/` by default (`--cov=src` in pytest addopts)
- Test structure mirrors `src/rail_svc/` (e.g., `tests/db/`, `tests/models/`, `tests/router/`)

## Code Style

- Line length: 110 (ruff)
- Ruff with pycodestyle, pyflakes, pep8-naming, pyupgrade, flake8-async rules
- Naming conventions relaxed: camelCase in variables and non-lowercase function names are allowed (N802, N803, N806, N815, N816 ignored)
- Python 3.13+ required (uses PEP 695 generics syntax)
- Docstrings: numpy convention
- mypy strict mode with sqlalchemy and pydantic plugins

## Key Patterns for New Code

- **Sync wrappers**: `local_sync/funcs.py` and `remote_sync/funcs.py` use `__getattr__` to auto-generate sync versions of async functions. Don't add explicit wrapper functions — just add the async version and it's available synchronously.
- **`with_session` decorator**: Detects methods vs standalone functions via `_is_method()` in `local_async/base.py`. For standalone functions, session is injected as the first arg. For methods, `self` is preserved.
- **CLI load/read_slice/download**: Use factories in `cli/load_commands.py` rather than copy-pasting command definitions per entity.
- **Remote client extended ops**: `RemoteFileOperations` base class in `client/base.py` provides `load()` and `download()`. Subclasses only override `read_slice()`.
- **Remote sync operations**: `_make_sync_method` + `__init_subclass__` in `remote_sync/base.py` auto-generates sync wrappers. Add extra methods via `_extra_methods` class variable.
- **Parametrized tests**: `tests/db/test_db_shared.py`, `tests/db_oper/test_db_oper_shared.py`, and `tests/models/test_models_shared.py` test common entity patterns. Add new entities to `ENTITY_CONFIGS` rather than creating new test files.
- **Adding a new table**: Add `db/{name}.py`, `models/{name}.py`, `db_oper/{name}.py` (with singleton), then register in: `db/__init__`, `models/__init__`, `db_oper/__init__`, `local_async/base.py` (LocalOperations subclass), `local_async/__init__`, `local_sync/base.py` (SyncOperations subclass), `local_sync/__init__`, `router/app.py`, `client/client.py` (TABLE_CONFIGS), `remote_async/__init__`, `remote_sync/base.py` (SyncRemoteOperations subclass), `remote_sync/__init__`, and both CLI `top.py` files. Add fixtures in `tests/conftest.py`.
- **Foreign key resolution at create time**: Tables with foreign keys (e.g., FilterAB, CatalogBandAssoc) implement `get_create_kwargs()` in their `db_oper` module. This allows callers to pass `band_name=` or `band_id=` and the method resolves via `db_funcs.read.lookup_by_id_or_name()`.

## Documentation

Docs are built with Sphinx and hosted on ReadTheDocs.

```bash
# Build docs locally
cd docs && make html

# Output goes to docs/_build/html/
```

- **Config**: `docs/conf.py` — uses sphinx-autoapi, sphinx-click, sphinx_rtd_theme
- **ReadTheDocs**: `.readthedocs.yaml` — builds on RTD with Python 3.13
- **CI**: `.github/workflows/docs.yml` — test-builds docs on PRs touching `docs/`, `src/`, or config
- **Auto-generated**: `docs/autoapi/` is generated at build time and gitignored

## Related Projects

- **`live-rail`** (`/Users/echarles/software/DESC/live-rail`) — Unified Dash dashboard that uses `pz-rail-svc` for data access. Provides CRUD management, estimation workflows, and interactive visualizations.
