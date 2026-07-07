"""TOML configuration for Facebin.

All runtime settings live in a single ``facebin.toml`` file.  The file is
located with the following precedence:

1. An explicit path passed to :func:`load_config`.
2. The ``FACEBIN_CONFIG`` environment variable.
3. ``./facebin.toml`` in the current working directory.
4. ``~/.config/facebin/facebin.toml``.

When no file is found, built-in defaults are used (a warning is logged).
Use ``facebin init-config`` (or :func:`write_default_config`) to create a
commented starter file.

The parsed configuration is exposed as plain dataclasses so that worker
processes can receive it through ``multiprocessing`` without re-reading the
file.
"""

import dataclasses
import logging
import os
import tomllib
from dataclasses import dataclass, field
from typing import List, Optional

from .errors import ConfigError

log = logging.getLogger("facebin")

ENV_VAR = "FACEBIN_CONFIG"
DEFAULT_LOCATIONS = (
    "facebin.toml",
    os.path.join("~", ".config", "facebin", "facebin.toml"),
)


def _expand(path: str) -> str:
    return os.path.expanduser(os.path.expandvars(path))


@dataclass
class RedisConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    autostart: bool = False


@dataclass
class DatabaseConfig:
    path: str = "~/facebin-data/facebin.db"

    def resolved_path(self) -> str:
        return _expand(self.path)


@dataclass
class ServerConfig:
    recognizers_per_camera: int = 1
    history_recorders: int = 1
    health_check_interval: float = 1.0
    flush_redis_on_start: bool = True


@dataclass
class HistoryConfig:
    image_record_dir: str = "~/facebin-data/image-store"
    image_record_period: int = 10

    def resolved_dir(self) -> str:
        return _expand(self.image_record_dir)


@dataclass
class VideoConfig:
    video_record_dir: str = "~/facebin-data/video-store"
    video_seconds_per_file: int = 600

    def resolved_dir(self) -> str:
        return _expand(self.video_record_dir)


@dataclass
class ModelsConfig:
    dir: str = "~/facebin-data/models"
    detection_model: str = "frozen_inference_graph_face.pb"
    detection_labels: str = "face_label_map.pbtxt"

    def resolved_dir(self) -> str:
        return _expand(self.dir)

    def detection_model_path(self) -> str:
        return os.path.join(self.resolved_dir(), self.detection_model)

    def detection_labels_path(self) -> str:
        return os.path.join(self.resolved_dir(), self.detection_labels)


@dataclass
class DatasetConfig:
    dir: str = "~/facebin-data/dataset-images/user"

    def resolved_dir(self) -> str:
        return _expand(self.dir)


@dataclass
class CameraConfig:
    id: str
    name: str = ""
    device: str = ""
    command: str = ""
    fps: int = 25


@dataclass
class Config:
    redis: RedisConfig = field(default_factory=RedisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    history: HistoryConfig = field(default_factory=HistoryConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cameras: List[CameraConfig] = field(default_factory=list)
    # Path the configuration was loaded from; None when defaults are used.
    source: Optional[str] = None

    def camera_by_id(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise ConfigError(
            "No camera with id '{}' is defined in {}.".format(
                camera_id, self.source or "the default configuration"),
            hint="Add a [[camera]] table with id = \"{}\" to your "
            "facebin.toml.".format(camera_id))


_SECTION_TYPES = {
    "redis": RedisConfig,
    "database": DatabaseConfig,
    "server": ServerConfig,
    "history": HistoryConfig,
    "video": VideoConfig,
    "models": ModelsConfig,
    "dataset": DatasetConfig,
}


def _build_section(name, cls, data, source):
    """Build a section dataclass from a TOML table, with strict validation."""
    if not isinstance(data, dict):
        raise ConfigError(
            "Section [{}] in {} must be a table, got {}.".format(
                name, source, type(data).__name__))
    fields = {f.name: f for f in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        if key not in fields:
            raise ConfigError(
                "Unknown key '{}' in section [{}] of {}.".format(
                    key, name, source),
                hint="Valid keys are: {}.".format(", ".join(sorted(fields))))
        expected = fields[key].type
        expected_name = (expected if isinstance(expected, str)
                         else getattr(expected, "__name__", str(expected)))
        # tomllib gives us str/int/float/bool; check against the annotation.
        type_map = {"str": str, "int": int, "float": (int, float),
                    "bool": bool}
        expected_types = type_map.get(expected_name, object)
        if not isinstance(value, expected_types) or (
                expected_name in ("int", "str") and isinstance(value, bool)):
            raise ConfigError(
                "Key '{}' in section [{}] of {} must be of type {}, "
                "got {} ({!r}).".format(key, name, source, expected_name,
                                        type(value).__name__, value))
        if expected_name == "float":
            value = float(value)
        kwargs[key] = value
    return cls(**kwargs)


def _build_cameras(data, source):
    if not isinstance(data, list):
        raise ConfigError(
            "'camera' in {} must be an array of tables.".format(source),
            hint="Define cameras as [[camera]] blocks, one per camera.")
    cameras = []
    seen_ids = set()
    for i, item in enumerate(data):
        cam = _build_section("camera[{}]".format(i), CameraConfig,
                             {**{"id": ""}, **item}, source)
        if not cam.id:
            raise ConfigError(
                "Camera #{} in {} has no 'id'.".format(i + 1, source),
                hint="Every [[camera]] table needs a unique id, "
                "e.g. id = \"camera1\".")
        if cam.id in seen_ids:
            raise ConfigError(
                "Duplicate camera id '{}' in {}.".format(cam.id, source),
                hint="Camera ids must be unique.")
        if not cam.device:
            raise ConfigError(
                "Camera '{}' in {} has no 'device'.".format(cam.id, source),
                hint="Set device to a video device (/dev/video0), an RTSP "
                "URL, or a video file path.")
        seen_ids.add(cam.id)
        cameras.append(cam)
    return cameras


def _validate(config: Config):
    src = config.source or "the default configuration"
    if not (0 < config.redis.port < 65536):
        raise ConfigError(
            "Redis port {} in {} is out of range.".format(
                config.redis.port, src),
            hint="Use a TCP port between 1 and 65535 (default 6379).")
    if config.server.recognizers_per_camera < 0:
        raise ConfigError(
            "server.recognizers_per_camera must be >= 0 in {}.".format(src))
    if config.server.history_recorders < 0:
        raise ConfigError(
            "server.history_recorders must be >= 0 in {}.".format(src))
    if config.server.health_check_interval <= 0:
        raise ConfigError(
            "server.health_check_interval must be > 0 in {}.".format(src))
    if config.history.image_record_period <= 0:
        raise ConfigError(
            "history.image_record_period must be > 0 in {}.".format(src))
    if config.video.video_seconds_per_file <= 0:
        raise ConfigError(
            "video.video_seconds_per_file must be > 0 in {}.".format(src))
    for cam in config.cameras:
        if cam.fps <= 0:
            raise ConfigError(
                "Camera '{}' has fps = {} in {}; it must be > 0.".format(
                    cam.id, cam.fps, src))


def parse_config(text: str, source: str = "<string>") -> Config:
    """Parse TOML text into a validated :class:`Config`."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(
            "Cannot parse {} as TOML: {}".format(source, e),
            hint="Check the syntax; strings need quotes and sections use "
            "[section] or [[camera]] headers.") from e

    known = set(_SECTION_TYPES) | {"camera"}
    unknown = set(data) - known
    if unknown:
        raise ConfigError(
            "Unknown section(s) {} in {}.".format(
                ", ".join("[{}]".format(u) for u in sorted(unknown)), source),
            hint="Valid sections are: {} and [[camera]].".format(
                ", ".join("[{}]".format(s) for s in sorted(_SECTION_TYPES))))

    kwargs = {}
    for name, cls in _SECTION_TYPES.items():
        if name in data:
            kwargs[name] = _build_section(name, cls, data[name], source)
    if "camera" in data:
        kwargs["cameras"] = _build_cameras(data["camera"], source)

    config = Config(source=source, **kwargs)
    _validate(config)
    return config


def find_config_file(path: Optional[str] = None) -> Optional[str]:
    """Locate the configuration file; returns None when nothing is found."""
    if path is not None:
        expanded = _expand(path)
        if not os.path.exists(expanded):
            raise ConfigError(
                "Configuration file '{}' does not exist.".format(path),
                hint="Create one with `facebin init-config --path {}`."
                .format(path))
        return expanded
    env_path = os.environ.get(ENV_VAR)
    if env_path:
        expanded = _expand(env_path)
        if not os.path.exists(expanded):
            raise ConfigError(
                "{} points to '{}' but that file does not exist.".format(
                    ENV_VAR, env_path),
                hint="Fix the environment variable or create the file with "
                "`facebin init-config --path {}`.".format(env_path))
        return expanded
    for candidate in DEFAULT_LOCATIONS:
        expanded = _expand(candidate)
        if os.path.exists(expanded):
            return expanded
    return None


def load_config(path: Optional[str] = None) -> Config:
    """Load and validate the configuration.

    Falls back to built-in defaults when no configuration file exists and no
    explicit path was requested.
    """
    found = find_config_file(path)
    if found is None:
        log.warning(
            "No facebin.toml found; using built-in defaults. "
            "Run `facebin init-config` to create one.")
        return Config(source=None)
    try:
        with open(found, "rb") as f:
            text = f.read().decode("utf-8")
    except OSError as e:
        raise ConfigError(
            "Cannot read configuration file '{}': {}".format(found, e),
            hint="Check the file permissions.") from e
    return parse_config(text, source=found)


DEFAULT_CONFIG_TOML = '''\
# Facebin configuration.
# Location: ./facebin.toml, ~/.config/facebin/facebin.toml, or set
# the FACEBIN_CONFIG environment variable. Paths may use ~ and $VARS.

[redis]
host = "localhost"
port = 6379
db = 0
# Start a local redis-server automatically when it is not reachable.
autostart = true

[database]
path = "~/facebin-data/facebin.db"

[server]
# Number of face-recognition worker processes per camera.
recognizers_per_camera = 1
# Number of history-recorder processes.
history_recorders = 1
# Seconds between worker-process health checks.
health_check_interval = 1.0
# Clear Redis queues when the server starts.
flush_redis_on_start = true

[history]
image_record_dir = "~/facebin-data/image-store"
# Group appearances of the same person within this many seconds.
image_record_period = 10

[video]
video_record_dir = "~/facebin-data/video-store"
video_seconds_per_file = 600

[models]
dir = "~/facebin-data/models"
detection_model = "frozen_inference_graph_face.pb"
detection_labels = "face_label_map.pbtxt"

[dataset]
dir = "~/facebin-data/dataset-images/user"

# One [[camera]] block per camera. `device` may be a local video device,
# an RTSP URL, or a video file. `command` optionally starts a helper
# process (e.g. an ffmpeg relay) before reading from the device.
[[camera]]
id = "camera1"
name = "Default Camera"
device = "/dev/video0"
command = ""
fps = 25
'''


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"{}"'.format(str(value).replace("\\", "\\\\").replace('"', '\\"'))


def dump_toml(config: Config) -> str:
    """Serialize a :class:`Config` back to TOML text."""
    lines = []
    for name, cls in _SECTION_TYPES.items():
        section = getattr(config, name)
        lines.append("[{}]".format(name))
        for f in dataclasses.fields(cls):
            lines.append("{} = {}".format(
                f.name, _toml_value(getattr(section, f.name))))
        lines.append("")
    for cam in config.cameras:
        lines.append("[[camera]]")
        for f in dataclasses.fields(CameraConfig):
            lines.append("{} = {}".format(
                f.name, _toml_value(getattr(cam, f.name))))
        lines.append("")
    return "\n".join(lines)


def save_config(config: Config, path: str):
    """Write a configuration back to disk as TOML."""
    expanded = _expand(path)
    os.makedirs(os.path.dirname(os.path.abspath(expanded)), exist_ok=True)
    try:
        with open(expanded, "w", encoding="utf-8") as f:
            f.write(dump_toml(config))
    except OSError as e:
        raise ConfigError(
            "Cannot write configuration to '{}': {}".format(path, e),
            hint="Check that the directory exists and is writable.") from e
    config.source = expanded


def write_default_config(path: str, overwrite: bool = False) -> str:
    """Create a commented default configuration file and return its path."""
    expanded = _expand(path)
    if os.path.exists(expanded) and not overwrite:
        raise ConfigError(
            "Refusing to overwrite existing configuration '{}'.".format(path),
            hint="Pass --force to overwrite it.")
    os.makedirs(os.path.dirname(os.path.abspath(expanded)), exist_ok=True)
    with open(expanded, "w", encoding="utf-8") as f:
        f.write(DEFAULT_CONFIG_TOML)
    return expanded
