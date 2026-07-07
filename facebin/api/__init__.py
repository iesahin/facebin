"""HTTP API and mobile web app for Facebin.

Exposes the recognition pipeline to phones and other remote clients:
authenticated REST endpoints for appearance history, people, and cameras,
an MJPEG live stream of annotated frames, and a mobile-first progressive
web app served from :mod:`facebin.api.static`.

Requires the optional ``api`` dependencies (``pip install 'facebin[api]'``).
Start it with ``facebin api``, or set ``enabled = true`` in the ``[api]``
section of ``facebin.toml`` so ``facebin run`` / ``facebin server`` start
it alongside the other worker processes.
"""
