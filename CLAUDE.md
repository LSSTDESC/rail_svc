# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`pz-rail-svc` is a service for managing photometric redshift (photo-z) estimation workflows for the LSST DESC (Dark Energy Science Collaboration). It provides a database-backed catalog of algorithms, models, datasets, estimators, and photo-z estimates, accessible via local Python API, REST API, or CLI.

The generic database/service framework (CRUD operations, session management, router factories, remote clients, config) lives in the [`macon`](https://github.com/KIPAC/macon) package. This repo contains only the domain-specific code: ORM table definitions, pydantic models, table operations with foreign key resolution, file validation, photo-z estimation logic, and CLI/API assembly.

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

### Dependency on macon

All generic framework code is provided by `macon`:
- `macon.db.base` — `Base` ORM class, lifecycle hooks, pydantic conversion
- `macon.db.session` — `init_db()`, `get_session()`, `close_db()`
- `macon.db_funcs` — Generic CRUD/filter/update/delete functions
- `macon.db_oper.base` — `TableOperations`, `FileValidatedOperations`, `TableContext`
- `macon.local_async.base` — `LocalOperations`, `@with_session`
- `macon.local_sync.base` — `SyncOperations`, `@sync_wrapper`
- `macon.remote_async.base` — `AsyncRemoteOperations`
- `macon.remote_sync.base` — `SyncRemoteOperations`
- `macon.router.base` — `create_table_router()` factory
- `macon.client.base` — `RemoteTableOperations`, `RemoteFileOperations`, `RemoteAPI`
- `macon.common` — `LoadType`, file handling, slice utilities
- `macon.config` — Configuration classes, `config` singleton
- `macon.models.filtering` — `Filter`, `FilterOp`, `OrderBy`
- `macon.models.web` — Shared API response/request types

Domain code in this repo imports directly from macon (e.g., `from macon.db.base import Base`).

### Layer Stack (domain code in this repo)

1. **`db/`** — SQLAlchemy ORM models (Algorithm, Band, Dataset, etc.) inheriting from `macon.db.base.Base`. Each implements `pydantic_create_class()`, `pydantic_model_class()`, and optional lifecycle hooks.

2. **`models/`** — Pydantic models for validation/serialization. `FooCreate` (input) and `Foo` (response) per table. `models/web.py` has domain-specific request/response types (EstimatePdfRequest, LoadCatalogYamlResponse, etc.).

3. **`db_oper/`** — Domain table operations. Simple tables (algorithm, band, etc.) just instantiate `TableOperations`. File-backed tables (dataset, estimates, model) subclass `FileValidatedOperations` and implement `get_create_kwargs()` with foreign key resolution and `get_file_length()`. Composite operations in `catalog_funcs.py` and `estimation_funcs.py`.

4. **`local_async/`** — Domain `LocalOperations` subclasses adding `load()` and `read_slice()` for file-backed tables. Module-level singletons.

5. **`local_sync/`** — Domain `SyncOperations` subclasses with sync wrappers for load/read_slice.

6. **`router/`** — Domain routers created via `create_table_router()`, plus custom endpoints for dataset/estimates/model load/download/read_slice. `router/funcs.py` adds estimation and catalog function endpoints.

7. **`client/`** — Domain remote client subclasses (`RemoteDatasetOperations`, `RemoteEstimatesOperations`, `RemoteModelOperations`) with `read_slice()` overrides.

8. **`remote_async/`** and **`remote_sync/`** — Domain subclasses adding load/read_slice/download methods.

9. **`cli/`** — Click-based CLIs assembling domain operations into commands.

10. **`rail_funcs/`** — Photo-z domain logic: catalog YAML parsing, HDF5/qp file I/O, RAIL estimation wrappers.

### Domain Tables

Algorithm, Band, CatalogTag, CatalogBandAssoc, Dataset, DatasetAssoc, Estimates, Estimator, FilterAB, Model, Sed.

File-backed tables (Dataset, Estimates, Model) have extended operations: load, read_slice, download.

### Composite Operations (`db_oper/catalog_funcs.py`, `db_oper/estimation_funcs.py`)

- `load_catalog_yaml(session, catalog_yaml, filter_dir)` — loads bands, catalog tags, and associations from YAML
- `load_seds(session, sed_dir, names=, names_file=)` — loads .sed files into Sed table
- `load_filter_abs(session, filter_ab_dir, names=)` — loads .AB files into FilterAB table
- `create_matched_dataset(session, ...)` — creates a collection dataset with component associations
- `get_data_and_estimates_data(session, dataset_id, row)` — retrieves catalog + all photo-z estimates for one object
- `estimate_pdf(session, estimator_id, dataset_id, row)` — single-object photo-z estimation
- `estimate_ensemble(session, estimator_id, dataset_id, output_file_path)` — batch estimation
- `estimate_dataset(session, estimator_id, dataset_id)` — full estimation workflow with record creation

## Configuration

Environment variables with `__` as nested delimiter (provided by macon's config system):
- `DB__URL` — database URL (default: `sqlite+aiosqlite:///rail_svc.db`)
- `STORAGE__ARCHIVE` — path for archived data files
- `STORAGE__IMPORT_AREA` — path for import files
- `PZ_RAIL_SERVICE` — remote service URL for client
- `PZ_RAIL_TOKEN` — auth token for remote client

## Testing

- Tests use in-memory SQLite via `macon.db.session.init_db()` in `tests/conftest.py`
- pytest-asyncio with `asyncio_mode = "auto"` — all async test functions run automatically
- Test fixtures use `create()` and `create_all()` helpers for concise ORM object setup
- Tests mock only external I/O (HDF5/qp file reading, RAIL library wrappers); DB operations use real in-memory SQLite
- Framework behavior (CRUD, filtering, routing, session management) is tested in macon's own test suite — not duplicated here

## Code Style

- Line length: 110 (ruff)
- Ruff with pycodestyle, pyflakes, pep8-naming, pyupgrade, flake8-async rules
- Naming conventions relaxed: camelCase in variables and non-lowercase function names are allowed (N802, N803, N806, N815, N816 ignored)
- Python 3.13+ required (uses PEP 695 generics syntax)
- Docstrings: numpy convention
- mypy strict mode with sqlalchemy and pydantic plugins

## Key Patterns for New Code

- **Adding a new table**: Add `db/{name}.py` (ORM, inherits `macon.db.base.Base`), `models/{name}.py` (Pydantic), `db_oper/{name}.py` (instantiate `TableOperations` or subclass `FileValidatedOperations`). Register in: `db/__init__`, `models/__init__`, `db_oper/__init__`, `local_async/base.py` (add subclass), `local_async/__init__`, `local_sync/base.py` (add subclass), `local_sync/__init__`, `router/base.py` (create router), `client/client.py` (TABLE_CONFIGS), `remote_async/__init__`, `remote_sync/base.py` (add subclass), `remote_sync/__init__`, and both CLI `top.py` files. Add fixtures in `tests/conftest.py`.
- **Foreign key resolution at create time**: Tables with foreign keys implement `get_create_kwargs()` in their `db_oper` module. This resolves names to IDs via `macon.db_funcs.read.lookup_by_id_or_name()`.
- **File-backed tables**: Subclass `macon.db_oper.base.FileValidatedOperations`, implement `get_file_length(path)` and `get_subdirectory()`. Call `self._validate_path_security(path)` and `self.validate_data_for_path(fullpath, ref_obj)` in `get_create_kwargs()`.
- **Sync wrappers**: `local_sync/funcs.py` and `remote_sync/funcs.py` use `__getattr__` to auto-generate sync versions. Just add the async version.
- **CLI load/read_slice/download**: Use factories in `cli/load_commands.py`.
- **Parametrized tests**: `tests/db/test_db_shared.py`, `tests/db_oper/test_db_oper_shared.py`, and `tests/models/test_models_shared.py` test common patterns. Add new entities to `ENTITY_CONFIGS`.

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

## Related Projects

- **`macon`** (`/Users/echarles/software/KIPAC/macon`) — Generic database service framework providing CRUD operations, session management, router factories, remote clients, and CLI scaffolding. All framework code in this repo comes from macon.
- **`live-rail`** (`/Users/echarles/software/DESC/live-rail`) — Unified Dash dashboard that uses `pz-rail-svc` for data access. Provides CRUD management, estimation workflows, and interactive visualizations.
