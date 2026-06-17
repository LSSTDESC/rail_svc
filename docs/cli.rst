CLI Reference
=============

pz-rail-svc provides three CLI entry points:

Local CLI
---------

Direct database operations without a server.

.. click:: rail_svc.cli.local.top:cli
   :prog: pz-rail-svc-local
   :nested: full

Remote CLI
----------

Operations via HTTP against a running server.

.. click:: rail_svc.cli.remote.top:cli
   :prog: pz-rail-svc-remote
   :nested: full

Server
------

Start the FastAPI server.

.. click:: rail_svc.cli.server.top:serve
   :prog: pz-rail-svc-server
   :nested: full
