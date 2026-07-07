# Facebin

Facebin is a desktop application and server that detects and recognizes
faces in video streams and photos. It reads frames from one or more cameras
(local devices, RTSP streams, or video files), finds faces with a
TensorFlow detector, recognizes them with VGGFace embeddings, records every
appearance to a SQLite database, and shows live annotated streams and an
appearance history in a Qt GUI.

## Quick start

```sh
# 1. Install (Python >= 3.11). Pick the extras you need:
pip install -e ".[ml,ui,dev]"

# 2. Create a configuration file and adapt it to your cameras:
facebin init-config
$EDITOR facebin.toml

# 3. Verify the environment (Redis, cameras, model files):
facebin check

# 4. Start everything (Redis if needed, all worker processes, and the GUI):
facebin
```

`facebin` is a single executable that starts and supervises all processes.
The subcommands:

| Command               | What it does                                                       |
| --------------------- | ------------------------------------------------------------------ |
| `facebin` / `facebin run` | Start Redis (when `autostart` is on), all worker processes, and the GUI |
| `facebin run --no-gui`    | The same, without the GUI                                      |
| `facebin server`      | Start only the headless worker processes                           |
| `facebin gui`         | Start only the GUI (attach to a running server)                    |
| `facebin api`         | Start only the HTTP API / mobile web app server                    |
| `facebin init-config` | Write a commented default `facebin.toml`                           |
| `facebin init-db`     | Create the SQLite schema and default admin user (idempotent)       |
| `facebin check`       | Validate configuration, Redis connectivity, and referenced paths   |

`python -m facebin` is equivalent to `facebin`.

## Installation details

The dependencies are split so that each deployment installs only what it
needs:

- **Core** (always installed): `numpy`, `opencv-contrib-python-headless`,
  `redis`, `av`. Enough for the supervisor, queues, database, and camera
  readers.
- **`ml` extra**: TensorFlow, scikit-learn, pandas, Pillow, and
  `keras-vggface` — required for the face detection and recognition
  workers.
- **`ui` extra**: PySide6 for the desktop GUI (PySide2 is still supported
  as a fallback at runtime).
- **`api` extra**: FastAPI and uvicorn for the HTTP API and the mobile
  web app (see [docs/MOBILE.md](docs/MOBILE.md)).
- **`dev` extra**: `pytest` and `fakeredis` for the test suite.

You also need:

- A **Redis server** (`apt install redis-server`). With
  `autostart = true` in the config, `facebin run` starts one for you.
- **Model files** for the TensorFlow face detector
  (`frozen_inference_graph_face.pb` and `face_label_map.pbtxt`, from the
  [tensorflow-face-detection](https://github.com/yeephycho/tensorflow-face-detection)
  project), placed in the `[models]` directory configured in
  `facebin.toml`.
- **ffmpeg** if you use camera helper commands (e.g. RTSP-to-v4l2 relays)
  or keyframe extraction.

## Configuration

All settings live in a single TOML file, looked up in this order:

1. `--config/-c` command line option
2. the `FACEBIN_CONFIG` environment variable
3. `./facebin.toml`
4. `~/.config/facebin/facebin.toml`

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for every setting, and
`facebin init-config` for a commented starter file. A minimal example:

```toml
[redis]
autostart = true

[[camera]]
id = "camera1"
name = "Entrance"
device = "rtsp://user:password@192.168.1.65:554/live"
fps = 25
```

## Using Facebin from your phone

Enable the `[api]` section in `facebin.toml` and install the `api` extra;
`facebin run` then also serves a mobile web app (installable as a PWA on
Android) with live camera streams, appearance history, and people
management. See [docs/MOBILE.md](docs/MOBILE.md).

## Architecture

Facebin runs as a set of cooperating processes connected by Redis queues:
camera readers push frames, recognizer processes annotate them with
detected/recognized faces, the GUI displays them, and history recorders
aggregate appearances into the SQLite database. The `facebin` executable
starts all of them and restarts any process that dies.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full picture,
including the queue layout, the database schema, and the module map.

## Development

```sh
pip install -e ".[dev]"
python -m pytest        # no Redis, camera, TensorFlow, or Qt required
```

The test suite covers the configuration loader, the database layer, the
Redis queue helpers (against `fakeredis`), the process supervisor (against
dummy workers), and the CLI.

Repository layout:

| Path                | Contents                                              |
| ------------------- | ----------------------------------------------------- |
| `facebin/cli.py`    | The `facebin` executable                              |
| `facebin/config.py` | TOML configuration loading and validation             |
| `facebin/errors.py` | Exception hierarchy (`FacebinError` and subclasses)   |
| `facebin/server/`   | Headless pipeline: supervisor, camera readers, detection, recognition, history, database |
| `facebin/api/`      | HTTP API (FastAPI) and the mobile web app (PWA)       |
| `facebin/ui/`       | Qt GUI: main window, dialogs, Qt compatibility layer  |
| `facebin/models/`   | Model directory helpers (`label_map_util`)            |
| `tests/`            | Pytest suite                                          |
| `init/`             | Requirements files and legacy install script          |
| `legacy/`           | Old, unused module versions kept for reference        |

## History

Version 0.1.x used INI files, per-host config names, TensorFlow 1.x, and a
zoo of shell scripts. Version 0.2.0 modernized the project: single TOML
configuration, a single `facebin` executable, upgraded dependencies
(TensorFlow 2 via `tf.compat.v1`, redis-py 5, PySide6, NumPy >= 1.26), a
test suite, and meaningful error messages throughout. See
[CHANGELOG.md](CHANGELOG.md).
