Quick Start
===========

Local Database Mode
-------------------

Initialize the database and start working with photo-z data directly:

.. code-block:: bash

   # Start the local CLI
   pz-rail-svc-local --help

   # Create an algorithm
   pz-rail-svc-local algorithm create --name knn --class-name rail.estimation.algos.k_nearneigh.KNearNeighEstimator

   # List algorithms
   pz-rail-svc-local algorithm get-rows

Python API (local)
------------------

.. code-block:: python

   from rail_svc import local_sync

   # Initialize the database
   from rail_svc.db.session import init_db
   init_db()

   # CRUD operations
   algos = local_sync.algorithm.get_rows()
   ds = local_sync.dataset.get_row(1)

   # Estimation functions
   result = local_sync.funcs.estimate_pdf(
       estimator_id=1, dataset_id=1, row=0
   )

Server Mode
-----------

Start the FastAPI server:

.. code-block:: bash

   DB__URL=sqlite+aiosqlite:///rail_svc.db pz-rail-svc-server

Then use the remote CLI or Python client:

.. code-block:: bash

   PZ_RAIL_SERVICE=http://localhost:8000 pz-rail-svc-remote algorithm get-rows

.. code-block:: python

   from rail_svc import remote_sync

   algos = remote_sync.algorithm().get_rows()
