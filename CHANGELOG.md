# 0.3.0

- **Mobile support**: new HTTP API server (`facebin api`, or
  `[api] enabled = true` to have `facebin run`/`facebin server` start and
  supervise it) exposing authenticated REST endpoints for history,
  people, cameras, and pipeline status, plus MJPEG live streams of the
  annotated camera feeds. Served at `/` is a mobile-first progressive
  web app (installable on Android via "Add to Home screen") with Live,
  History, People, and Status screens. See docs/MOBILE.md.
- New `api` dependency extra (FastAPI + uvicorn) and `[api]` section in
  facebin.toml (enabled, host, port, session_ttl).
- 15 new API tests (auth, streaming, history/people endpoints, PWA
  shell) — 102 tests total.

# 0.2.0

- **Single executable**: the new `facebin` command (also `python -m
  facebin`) starts Redis (optional), all worker processes, and the GUI,
  and supervises them. Subcommands: `run`, `server`, `gui`,
  `init-config`, `init-db`, `check`.
- **TOML configuration**: all settings (Redis, database, server process
  counts, history/video recording, model paths, cameras) moved from
  scattered INI files into one validated `facebin.toml`
  (see docs/CONFIGURATION.md). `facebin init-config` writes a commented
  starter file.
- **Upgraded dependencies**: Python >= 3.11, NumPy >= 1.26, redis-py >= 5
  (`hmset`/`tostring` migrated to `hset(mapping=...)`/`tobytes`),
  OpenCV >= 4.9, PyAV >= 12, TensorFlow 2 (through `tf.compat.v1`),
  PySide6 (with PySide2 fallback). Packaging moved to `pyproject.toml`
  with `ml`, `ui`, and `dev` extras.
- **Meaningful errors**: new `FacebinError` hierarchy with actionable
  hints; configuration errors name the file, key, and valid values;
  database and Redis failures explain what to do instead of silently
  returning empty results.
- **Test suite**: 87 pytest tests covering configuration, the database
  layer, Redis queue helpers (fakeredis), the process supervisor, camera
  controllers, and the CLI — all runnable without cameras, Redis,
  TensorFlow, or Qt.
- **Documentation**: rewritten README, new docs/ARCHITECTURE.md and
  docs/CONFIGURATION.md, and module docstrings throughout.
- Fixes along the way: broken imports left over from the 0.1.1 package
  reorganization, an unusable `FacebinServer` (undefined names), swapped
  width/height columns in `insert_image`, `del_frame` calling `hdel`
  without fields, and hard-coded camera credentials in the database
  initialization statements.

# 0.1.2

- Adding a Dockerfile for recognition server based on nvidia/cuda:10.2-base


# 0.1.1

- Created a python package for the server and ui functionality
- Wrote a [README.md](./README.md) file
- Moved binary files to s3 and added downloads to install.sh
- Arranged the files into directories
