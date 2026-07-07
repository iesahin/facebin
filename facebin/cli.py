"""The ``facebin`` command line interface.

A single executable that starts every part of the system:

- ``facebin run``: start Redis (if configured), the server worker
  processes, and the GUI.  This is the default command.
- ``facebin server``: start only the headless worker processes.
- ``facebin gui``: start only the GUI (expects a running server).
- ``facebin init-config``: write a commented default ``facebin.toml``.
- ``facebin init-db``: create the SQLite schema and the default admin user.
- ``facebin check``: validate the configuration and the environment.

Run ``facebin <command> --help`` for the options of each command.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

from . import __version__
from .config import load_config, write_default_config
from .errors import ConfigError, FacebinError, RedisConnectionError


def _add_config_argument(parser):
    parser.add_argument(
        "-c", "--config",
        default=None,
        metavar="PATH",
        help="Path to facebin.toml (default: $FACEBIN_CONFIG, "
        "./facebin.toml, or ~/.config/facebin/facebin.toml)")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="facebin",
        description="Face detection and recognition for video streams.",
        epilog="Run `facebin <command> --help` for command options.")
    parser.add_argument("--version", action="version",
                        version="facebin {}".format(__version__))
    sub = parser.add_subparsers(dest="command")

    p_run = sub.add_parser(
        "run", help="Start all processes: Redis (optional), server "
        "workers, and the GUI.")
    _add_config_argument(p_run)
    p_run.add_argument("--no-gui", action="store_true",
                       help="Do not start the GUI (same as `facebin "
                       "server`).")

    p_server = sub.add_parser(
        "server", help="Start the headless server processes only.")
    _add_config_argument(p_server)

    p_gui = sub.add_parser(
        "gui", help="Start the GUI only (expects a running server).")
    _add_config_argument(p_gui)

    p_api = sub.add_parser(
        "api", help="Start the HTTP API / mobile web app server only "
        "(expects a running server).")
    _add_config_argument(p_api)

    p_init_config = sub.add_parser(
        "init-config", help="Write a commented default facebin.toml.")
    p_init_config.add_argument(
        "--path", default="facebin.toml",
        help="Where to write the file (default: ./facebin.toml)")
    p_init_config.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing file.")

    p_init_db = sub.add_parser(
        "init-db", help="Create the database schema and default admin "
        "user (idempotent).")
    _add_config_argument(p_init_db)

    p_check = sub.add_parser(
        "check", help="Validate configuration, Redis connectivity, and "
        "referenced paths.")
    _add_config_argument(p_check)

    return parser


def ensure_redis(config):
    """Make sure Redis is reachable, starting a local server if allowed."""
    from .server import redis_queue_utils as rqu
    rqu.configure(config.redis)
    try:
        rqu.get_redis()
        return None
    except RedisConnectionError:
        if not config.redis.autostart:
            raise

    if shutil.which("redis-server") is None:
        raise RedisConnectionError(
            "Redis is not reachable at {}:{} and `redis-server` is not "
            "installed, so it cannot be started automatically.".format(
                config.redis.host, config.redis.port),
            hint="Install Redis (e.g. `apt install redis-server`) or point "
            "the [redis] section of facebin.toml at a running instance.")

    print("Starting a local redis-server on port {} ...".format(
        config.redis.port))
    proc = subprocess.Popen(
        ["redis-server", "--port", str(config.redis.port),
         "--save", "", "--appendonly", "no"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
    deadline = time.time() + 10
    last_error = None
    while time.time() < deadline:
        try:
            rqu.configure(config.redis)  # reset the failed connection
            rqu.get_redis()
            return proc
        except RedisConnectionError as e:
            last_error = e
            time.sleep(0.2)
    proc.terminate()
    raise RedisConnectionError(
        "Started redis-server (pid {}) but it did not become reachable "
        "within 10 seconds.".format(proc.pid),
        hint="Check `redis-server --port {}` manually; the original "
        "error was: {}".format(config.redis.port, last_error))


def cmd_server(args, with_gui=False):
    from .server import FacebinServer
    from .server import database_api as db

    config = load_config(args.config)
    redis_proc = ensure_redis(config)
    db.configure(config)
    db.init_db()

    server = FacebinServer(config)
    server.start()

    try:
        if with_gui:
            rc = _run_gui(config, server)
        else:
            print("Facebin server is running; press Ctrl-C to stop.")
            server.join()
            rc = 0
    except KeyboardInterrupt:
        print("\nShutting down ...")
        rc = 0
    finally:
        server.stop()
        if redis_proc is not None:
            redis_proc.terminate()
            redis_proc.wait()
    return rc


def _run_gui(config, server=None):
    from facebin.ui.main_window import main as gui_main
    return gui_main(config=config, server=server) or 0


def cmd_gui(args):
    config = load_config(args.config)
    from .server import redis_queue_utils as rqu
    from .server import database_api as db
    rqu.configure(config.redis)
    rqu.get_redis()  # fail early with a helpful message
    db.configure(config)
    return _run_gui(config)


def cmd_api(args):
    from facebin.api.server import serve
    from .server import database_api as db
    config = load_config(args.config)
    redis_proc = ensure_redis(config)
    db.configure(config)
    db.init_db()
    try:
        serve(config)
    except KeyboardInterrupt:
        pass
    finally:
        if redis_proc is not None:
            redis_proc.terminate()
            redis_proc.wait()
    return 0


def cmd_init_config(args):
    path = write_default_config(args.path, overwrite=args.force)
    print("Wrote default configuration to {}".format(path))
    print("Edit the [[camera]] sections to match your cameras, then run "
          "`facebin check`.")
    return 0


def cmd_init_db(args):
    from .server import database_api as db
    config = load_config(args.config)
    db.configure(config)
    db.init_db()
    print("Database ready at {}".format(db.get_database_path()))
    return 0


def cmd_check(args):
    config = load_config(args.config)
    problems = []
    print("Configuration: {}".format(config.source or "built-in defaults"))

    from .server import redis_queue_utils as rqu
    rqu.configure(config.redis)
    try:
        rqu.get_redis()
        print("Redis: OK ({}:{})".format(config.redis.host,
                                         config.redis.port))
    except RedisConnectionError as e:
        if config.redis.autostart:
            print("Redis: not running; will be started automatically "
                  "(autostart = true).")
        else:
            problems.append(str(e))

    db_path = config.database.resolved_path()
    if os.path.exists(db_path):
        print("Database: OK ({})".format(db_path))
    else:
        print("Database: {} does not exist yet; it will be created by "
              "`facebin init-db` or `facebin run`.".format(db_path))

    for cam in config.cameras:
        if cam.device.startswith("/dev/") and not os.path.exists(cam.device):
            problems.append(
                "Camera '{}': device {} does not exist. Check the "
                "connection or the [[camera]] block.".format(
                    cam.id, cam.device))
        else:
            print("Camera '{}': {}".format(cam.id, cam.device))
    if not config.cameras:
        print("Cameras: none configured.")

    model_path = config.models.detection_model_path()
    if os.path.exists(model_path):
        print("Detection model: OK ({})".format(model_path))
    else:
        problems.append(
            "Detection model '{}' does not exist. Download the model "
            "files into the [models] directory (see docs/CONFIGURATION.md)."
            .format(model_path))

    if problems:
        print("\nProblems found:")
        for p in problems:
            print(" - {}".format(p))
        return 1
    print("\nEverything looks good.")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"
    if args.command is None:
        # Re-parse so `facebin` alone behaves exactly like `facebin run`.
        args = parser.parse_args(["run"] + (argv or sys.argv[1:]))

    try:
        if command == "run":
            return cmd_server(args, with_gui=not args.no_gui)
        if command == "server":
            return cmd_server(args, with_gui=False)
        if command == "gui":
            return cmd_gui(args)
        if command == "api":
            return cmd_api(args)
        if command == "init-config":
            return cmd_init_config(args)
        if command == "init-db":
            return cmd_init_db(args)
        if command == "check":
            return cmd_check(args)
        parser.error("Unknown command: {}".format(command))
    except FacebinError as e:
        print("Error: {}".format(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
