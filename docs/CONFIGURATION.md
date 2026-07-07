# Facebin Configuration Reference

Facebin reads a single TOML file. `facebin init-config` writes a
commented starter file; every command accepts `-c/--config PATH`.

Lookup order:

1. `-c/--config` command line option
2. `FACEBIN_CONFIG` environment variable
3. `./facebin.toml`
4. `~/.config/facebin/facebin.toml`
5. Built-in defaults (with a warning)

Paths may contain `~` and environment variables (`$HOME/...`); they are
expanded when used. Unknown sections or keys are rejected with an error
naming the offending key and the valid alternatives, so typos cannot be
silently ignored.

## `[redis]`

| Key         | Default       | Description                                            |
| ----------- | ------------- | ------------------------------------------------------ |
| `host`      | `"localhost"` | Redis host                                             |
| `port`      | `6379`        | Redis TCP port (1–65535)                               |
| `db`        | `0`           | Redis database number                                  |
| `autostart` | `false`       | Start a local `redis-server` when none is reachable    |

## `[database]`

| Key    | Default                     | Description               |
| ------ | --------------------------- | ------------------------- |
| `path` | `"~/facebin-data/facebin.db"` | SQLite database file    |

## `[server]`

| Key                       | Default | Description                                        |
| ------------------------- | ------- | -------------------------------------------------- |
| `recognizers_per_camera`  | `1`     | Face recognition worker processes per camera (≥ 0) |
| `history_recorders`       | `1`     | History recorder processes (≥ 0)                   |
| `health_check_interval`   | `1.0`   | Seconds between worker liveness checks (> 0)       |
| `flush_redis_on_start`    | `true`  | Clear Redis queues when the server starts          |

## `[history]`

| Key                   | Default                        | Description                                                  |
| --------------------- | ------------------------------ | ------------------------------------------------------------ |
| `image_record_dir`    | `"~/facebin-data/image-store"` | Where face/camera snapshots are written                      |
| `image_record_period` | `10`                           | Seconds within which sightings of a person are merged (> 0)  |

## `[video]`

| Key                      | Default                        | Description                              |
| ------------------------ | ------------------------------ | ---------------------------------------- |
| `video_record_dir`       | `"~/facebin-data/video-store"` | Where footage files are written          |
| `video_seconds_per_file` | `600`                          | Length of each rolling footage file (> 0)|

## `[models]`

| Key                | Default                             | Description                          |
| ------------------ | ----------------------------------- | ------------------------------------ |
| `dir`              | `"~/facebin-data/models"`           | Directory containing the model files |
| `detection_model`  | `"frozen_inference_graph_face.pb"`  | TensorFlow frozen detection graph    |
| `detection_labels` | `"face_label_map.pbtxt"`            | Label map for the detector           |

Download the two detector files from the
[tensorflow-face-detection](https://github.com/yeephycho/tensorflow-face-detection)
project into `dir`. `facebin check` verifies they exist.

## `[dataset]`

| Key   | Default                                  | Description                                |
| ----- | ---------------------------------------- | ------------------------------------------ |
| `dir` | `"~/facebin-data/dataset-images/user"`   | Root of the training images per person     |

## `[[camera]]` (repeatable)

One block per camera. `id` and `device` are required; ids must be unique.

| Key       | Default | Description                                                        |
| --------- | ------- | ------------------------------------------------------------------ |
| `id`      | —       | Unique camera identifier, e.g. `"camera1"`                         |
| `name`    | `""`    | Human-readable name shown in the GUI                               |
| `device`  | —       | `/dev/videoN`, an RTSP/HTTP URL, or a video file path              |
| `command` | `""`    | Optional helper command started before reading (e.g. ffmpeg relay) |
| `fps`     | `25`    | Maximum frames per second read from this camera (> 0)              |

Example with an ffmpeg relay that converts an RTSP stream into a v4l2
device:

```toml
[[camera]]
id = "camera1"
name = "Entrance"
device = "/dev/video1"
command = "ffmpeg -rtsp_transport tcp -i rtsp://user:pass@192.168.1.65:554/live -c:v rawvideo -pix_fmt yuv420p -f v4l2 /dev/video1"
fps = 25
```

> Note: the configuration file contains camera credentials in RTSP URLs —
> keep it out of version control and readable only by the service user.

## Environment variables

| Variable                 | Effect                                                     |
| ------------------------ | ---------------------------------------------------------- |
| `FACEBIN_CONFIG`         | Path of the configuration file                             |
| `FACEBIN_LOG_DIR`        | Debug log directory (default `/tmp/facebin-logs`)          |
| `FACEBIN_CAMERA_LOG_DIR` | Camera helper log directory (default `/tmp/facebin-cam-logs/`) |

## Migrating from 0.1.x INI files

The old `facebin.ini`, per-host `facebin-config-$(hostname).ini`, and
`camera-config-*.ini` files are no longer read. Their settings map to:

| Old INI setting                       | New TOML setting                 |
| ------------------------------------- | -------------------------------- |
| `[history] image_record_dir/period`   | `[history]` (same keys)          |
| `[video] video_record_dir/...`        | `[video]` (same keys)            |
| `[general] dataset-dir`               | `[dataset] dir`                  |
| `[cameraN] id/name/device/command`    | one `[[camera]]` block per camera|

Run `facebin init-config` and copy the values over.
