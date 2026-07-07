# Facebin Architecture

Facebin is a multi-process pipeline. Processes communicate through Redis;
persistent data (people, images, appearance history) lives in SQLite.
The `facebin` executable (`facebin/cli.py`) starts everything and the
`FacebinServer` supervisor (`facebin/server/__init__.py`) keeps the worker
processes alive.

## Process and data flow

```
                       ┌────────────────────────────────────────────────┐
                       │                facebin run                     │
                       │  (starts Redis*, workers, GUI; supervises all) │
                       └────────────────────────────────────────────────┘

┌──────────────┐  frames   ┌───────────────┐  annotated   ┌─────────────┐
│ camera reader│──────────▶│  recognizer   │─────────────▶│     GUI     │
│  (1/camera)  │  CAMERA_  │ (N/camera)    │ RECOGNIZER_  │ live view + │
│   PyAV       │  QUEUE    │ TF detector + │ QUEUE(cam)   │ appearance  │
└──────────────┘           │ VGGFace + kNN │              │ table       │
                           └───────┬───────┘              └──────┬──────┘
                                   │ faces                       │ viewed
                                   │                             ▼ frames
                                   │                        HISTORY_QUEUE
                                   │                             │
                                   ▼                             ▼
                           ┌──────────────────────────────────────────┐
                           │             history recorder             │
                           │  merges consecutive appearances of the   │
                           │  same person; writes snapshots + rows    │
                           └───────────────────┬──────────────────────┘
                                               │
                                               ▼
                                     SQLite (facebin.db)
                                     PNG snapshots on disk

* Redis is started automatically when `[redis] autostart = true`.
```

## Redis queues

Queues are sorted sets scored by frame timestamp; frame payloads are Redis
hashes with a `frame:{camera_id}:{dts}` key and a TTL
(`redis_queue_utils.STANDARD_EXPIRATION`) so stale frames evaporate.

| Queue                     | Producer         | Consumer          | Content                     |
| ------------------------- | ---------------- | ----------------- | --------------------------- |
| `framekeys` (CAMERA_QUEUE)| camera readers   | recognizers       | raw frames                  |
| `recognizer:{camera_id}`  | recognizers      | GUI               | annotated frames            |
| `history` (HISTORY_QUEUE) | GUI              | history recorders | frames with recognized faces|
| `record` (HISTORY_RECORDING_QUEUE) | history recorders | GUI appearance table, DB writer | aggregated person appearances |

The hash field names (image data/shape/dtype, per-face coordinates,
encodings, person ids, ...) are defined once in
`facebin/server/redis_queue_utils.py` and used by every producer and
consumer.

## Worker processes

- **Camera reader** (`server/camera_reader.py`) — one per `[[camera]]`
  block. Opens the device with PyAV, throttles to the configured fps,
  and pushes BGR frames to the camera queue. Backs off when the queue is
  full and exits (to be restarted) when the stream ends.
- **Recognizer** (`server/face_recognition_v6.py`) — `recognizers_per_camera`
  × number of cameras. Detects faces (`server/face_detection.py`,
  TensorFlow frozen graph via `tf.compat.v1`), encodes them with the fc7
  layer of VGGFace (4096-dim), matches against the known feature set by
  L2 distance, draws annotations, and republishes the frame.
- **History recorder** (`server/history_recorder.py`) — merges consecutive
  sightings of the same person (same person id, or encoding distance below
  a threshold for unknown faces), keeps the largest face image, and
  periodically writes finished appearances to SQLite together with face
  and camera snapshots.
- **Video recorder** (`server/video_recorder.py`) — writes frames to
  rolling footage files (one file per `video_seconds_per_file` seconds per
  camera).

`FacebinServer` starts these with `multiprocessing.Process`, checks
liveness every `health_check_interval` seconds, and restarts anything that
died. Worker entry points are imported lazily and can be injected, so the
supervisor is testable and usable without TensorFlow installed.

## Database schema

`facebin/server/database_api.py` owns all SQL. Tables:

- `person(id, name, title, notes)`
- `person_image(id, person_id, path, is_face, super_image_id, width,
  height, feature_id)` — both full photos and cropped faces; `feature_id`
  indexes into the on-disk feature array of the dataset manager.
- `history(id, camera_id, person_id, time, camera_image_filename,
  face_image_filename, video_filename, original_person_id,
  original_record_change_timestamp, original_record_change_user)` —
  one row per aggregated appearance; `person_id < 0` means unknown.
- `login(username, password)` and `permissions(username, permission)` —
  GUI users.
- `key_value_string` / `key_value_int` — small settings area.

`facebin init-db` creates missing tables idempotently; `reset_db()` is the
destructive variant.

## Training dataset

`server/dataset_manager_v3.py` keeps person images under the `[dataset]`
directory and their VGGFace features in `facebin.feature.gz.npz`.
Recognition compares live encodings against this array; adding a photo
from the GUI appends a feature and restarts the recognizers.

## GUI

`facebin/ui/main_window.py` shows one `VideoPresentationWidget` per camera
(reading annotated frames from Redis) plus a `HistoryTable` fed by the
aggregation queue. Dialogs: camera configuration (writes back to
`facebin.toml`), person management, history browsing.
`facebin/ui/qt_compat.py` selects PySide6 or PySide2 at import time.

## Error handling

All Facebin errors derive from `facebin.errors.FacebinError` and carry an
optional hint that is printed with the message (`Error: ...\nHint: ...`).
The CLI catches `FacebinError` at the top level and exits with status 2;
inside workers, errors propagate so the supervisor can restart the
process and the cause lands in the process log under
`$FACEBIN_LOG_DIR` (default `/tmp/facebin-logs`).
