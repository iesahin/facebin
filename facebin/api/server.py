"""FastAPI application exposing Facebin to remote (mobile) clients.

Routes (all under ``/api``, token-authenticated except ``/api/login``):

- ``POST /api/login`` — exchange username/password for a bearer token.
- ``GET /api/me`` / ``POST /api/logout`` — session management.
- ``GET /api/status`` — pipeline health: Redis, database, queue lengths.
- ``GET /api/cameras`` — configured cameras.
- ``GET /api/stream/{camera_id}`` — MJPEG live stream of annotated frames.
- ``GET /api/history`` — appearance history with filters.
- ``GET /api/history/{id}/face|camera`` — recorded snapshots.
- ``GET/POST /api/persons`` and ``GET /api/persons/{id}`` — people.
- ``GET /api/persons/{id}/image`` — a face image of the person.

The progressive web app in :mod:`facebin.api.static` is served at ``/``.

Authentication is a bearer token kept in server memory (the app runs on a
trusted LAN; there is no external identity provider).  Streams embedded in
``<img>`` tags cannot send headers, so the token is also accepted as a
``?token=`` query parameter.
"""

import os
import secrets
import time as time_mod

import cv2
import numpy as np

from facebin.config import Config, load_config
from facebin.errors import DependencyError, FacebinError
from facebin.server import database_api as db
from facebin.server import redis_queue_utils as rqu
from facebin.server.utils import init_logging

try:
    from fastapi import (Depends, FastAPI, HTTPException, Query, Request,
                         Response)
    from fastapi.responses import (FileResponse, JSONResponse,
                                   StreamingResponse)
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
except ImportError as e:
    raise DependencyError(
        "The Facebin API requires FastAPI, which is not installed ({})."
        .format(e),
        hint="Install the API dependencies with "
        "`pip install 'facebin[api]'`.") from e

log = init_logging()

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "static")

STREAM_BOUNDARY = "facebin-frame"
STREAM_IDLE_DELAY = 0.05
JPEG_QUALITY = 80


class TokenStore:
    """In-memory bearer tokens with expiry."""

    def __init__(self, ttl_seconds):
        self.ttl = ttl_seconds
        self._tokens = {}

    def create(self, username):
        token = secrets.token_urlsafe(32)
        self._tokens[token] = (username, time_mod.time() + self.ttl)
        return token

    def username_for(self, token):
        entry = self._tokens.get(token)
        if entry is None:
            return None
        username, expires = entry
        if time_mod.time() > expires:
            del self._tokens[token]
            return None
        return username

    def revoke(self, token):
        self._tokens.pop(token, None)


class LoginRequest(BaseModel):
    username: str
    password: str


class PersonRequest(BaseModel):
    name: str
    title: str = ""
    notes: str = ""


def _jpeg_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image,
                           [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise FacebinError("Could not encode a frame as JPEG.")
    return buf.tobytes()


def create_app(config: Config = None) -> FastAPI:
    """Build the FastAPI application for a loaded configuration."""
    if config is None:
        config = load_config()

    rqu.configure(config.redis)
    db.configure(config)

    app = FastAPI(title="Facebin API", version="0.3.0", docs_url="/api/docs",
                  openapi_url="/api/openapi.json")
    tokens = TokenStore(config.api.session_ttl)
    app.state.config = config
    app.state.tokens = tokens

    @app.exception_handler(FacebinError)
    async def facebin_error_handler(request: Request, exc: FacebinError):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    def require_auth(request: Request) -> str:
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else \
            request.query_params.get("token")
        username = tokens.username_for(token) if token else None
        if username is None:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated: send `Authorization: Bearer "
                "<token>` (from POST /api/login) or a ?token= parameter.")
        return username

    # --- Session -----------------------------------------------------------

    @app.post("/api/login")
    def login(body: LoginRequest):
        if not db.login(body.username, body.password):
            raise HTTPException(
                status_code=401,
                detail="Unknown username or wrong password.")
        return {"token": tokens.create(body.username),
                "username": body.username}

    @app.post("/api/logout")
    def logout(request: Request, username: str = Depends(require_auth)):
        auth = request.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else \
            request.query_params.get("token")
        tokens.revoke(token)
        return {"ok": True}

    @app.get("/api/me")
    def me(username: str = Depends(require_auth)):
        return {"username": username,
                "permissions": [p for _, p in db.permissions(username)]}

    # --- Status ------------------------------------------------------------

    @app.get("/api/status")
    def status(username: str = Depends(require_auth)):
        queues = {"camera": rqu.queue_length(rqu.CAMERA_QUEUE),
                  "history": rqu.queue_length(rqu.HISTORY_QUEUE),
                  "recording": rqu.queue_length(rqu.HISTORY_RECORDING_QUEUE)}
        for cam in config.cameras:
            queues["recognizer:{}".format(cam.id)] = rqu.queue_length(
                rqu.RECOGNIZER_QUEUE(cam.id))
        return {"version": app.version,
                "database": db.get_database_path(),
                "queues": queues}

    # --- Cameras and live streams -------------------------------------------

    @app.get("/api/cameras")
    def cameras(username: str = Depends(require_auth)):
        return [{"id": cam.id, "name": cam.name or cam.id, "fps": cam.fps,
                 "stream_url": "/api/stream/{}".format(cam.id)}
                for cam in config.cameras]

    def mjpeg_frames(camera_id, max_frames=None):
        """Yield MJPEG parts from the recognizer output queue.

        Consumes keys the way the desktop GUI does: pop from the camera's
        recognizer queue and forward to the history queue so appearance
        recording keeps working when the phone is the only viewer.
        """
        input_queue = rqu.RECOGNIZER_QUEUE(camera_id)
        served = 0
        while max_frames is None or served < max_frames:
            key, score = rqu.get_next_key(input_queue)
            if key is None:
                time_mod.sleep(STREAM_IDLE_DELAY)
                continue
            image = rqu.get_frame_image(key, name="processed_image")
            if image is None:
                image = rqu.get_frame_image(key, name="image")
            rqu.R.zrem(input_queue, key)
            rqu.R.zadd(rqu.HISTORY_QUEUE, {key: score})
            if image is None:
                continue
            payload = _jpeg_bytes(image)
            served += 1
            yield (b"--" + STREAM_BOUNDARY.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(payload)).encode() +
                   b"\r\n\r\n" + payload + b"\r\n")

    @app.get("/api/stream/{camera_id}")
    def stream(camera_id: str,
               frames: int = Query(default=None, gt=0),
               username: str = Depends(require_auth)):
        known = {cam.id for cam in config.cameras}
        if camera_id not in known:
            raise HTTPException(
                status_code=404,
                detail="Unknown camera '{}'. Configured cameras: {}.".format(
                    camera_id, ", ".join(sorted(known)) or "none"))
        return StreamingResponse(
            mjpeg_frames(camera_id, max_frames=frames),
            media_type="multipart/x-mixed-replace; boundary={}".format(
                STREAM_BOUNDARY))

    # --- History -------------------------------------------------------------

    def history_record_json(rec):
        return {
            "id": rec.row_id,
            "camera_id": rec.camera_id,
            "person_id": rec.person_id,
            "time": rec.time,
            "person_name": rec.person_name,
            "person_title": rec.person_title,
            "known": (rec.person_id or -1) >= 0,
            "face_image_url": "/api/history/{}/face".format(rec.row_id),
            "camera_image_url": "/api/history/{}/camera".format(rec.row_id),
        }

    @app.get("/api/history")
    def history(username: str = Depends(require_auth),
                person_id: int = None,
                camera_id: int = None,
                begin: float = None,
                end: float = None,
                unknown_only: bool = False,
                limit: int = Query(default=100, gt=0, le=1000)):
        records = db.history_query(person_id=person_id,
                                   camera_id=camera_id,
                                   datetime_begin=begin,
                                   datetime_end=end,
                                   max_elements=limit)
        if unknown_only:
            records = [r for r in records if (r.person_id or -1) < 0]
        return [history_record_json(r) for r in records]

    def _history_image(history_id, which):
        recs = db.history_by_id(history_id)
        if not recs:
            raise HTTPException(
                status_code=404,
                detail="No history record with id {}.".format(history_id))
        rec = recs[0]
        path = (rec.face_image_filename if which == "face"
                else rec.camera_image_filename)
        if not path or not os.path.exists(path):
            raise HTTPException(
                status_code=404,
                detail="The {} snapshot for history record {} is not on "
                "disk (expected at '{}'). Snapshots may have been cleaned "
                "up.".format(which, history_id, path))
        return FileResponse(path, media_type="image/png")

    @app.get("/api/history/{history_id}/face")
    def history_face(history_id: int,
                     username: str = Depends(require_auth)):
        return _history_image(history_id, "face")

    @app.get("/api/history/{history_id}/camera")
    def history_camera(history_id: int,
                       username: str = Depends(require_auth)):
        return _history_image(history_id, "camera")

    # --- Persons --------------------------------------------------------------

    @app.get("/api/persons")
    def persons(username: str = Depends(require_auth)):
        result = []
        for pid, name, title, notes in db.person_list():
            faces = db.person_face_images_by_person_id(pid)
            result.append({
                "id": pid, "name": name, "title": title, "notes": notes,
                "face_count": len(faces),
                "image_url": ("/api/persons/{}/image".format(pid)
                              if faces else None),
            })
        return result

    @app.post("/api/persons", status_code=201)
    def create_person(body: PersonRequest,
                      username: str = Depends(require_auth)):
        if not body.name.strip():
            raise HTTPException(status_code=422,
                                detail="A person needs a non-empty name.")
        pid = db.insert_person(body.title, body.name.strip(), body.notes)
        return {"id": pid, "name": body.name.strip(), "title": body.title,
                "notes": body.notes}

    @app.get("/api/persons/{person_id}")
    def person_detail(person_id: int,
                      username: str = Depends(require_auth)):
        recs = db.person_by_id(person_id)
        if not recs:
            raise HTTPException(
                status_code=404,
                detail="No person with id {}.".format(person_id))
        pid, name, title, notes = recs[0]
        images = db.person_images_by_person_id(pid)
        return {
            "id": pid, "name": name, "title": title, "notes": notes,
            "images": [{"id": i[0], "path": i[2], "is_face": bool(i[3])}
                       for i in images],
            "history": [history_record_json(r)
                        for r in db.history_by_person(pid, max_elements=20)],
        }

    @app.get("/api/persons/{person_id}/image")
    def person_image(person_id: int,
                     username: str = Depends(require_auth)):
        faces = db.person_face_images_by_person_id(person_id)
        for face in faces:
            path = face[2]
            if path and os.path.exists(path):
                return FileResponse(path)
        raise HTTPException(
            status_code=404,
            detail="No face image on disk for person {}.".format(person_id))

    # --- Web app ---------------------------------------------------------------

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True),
              name="webapp")

    return app


def serve(config: Config = None):
    """Run the API server (blocking).  Entry point for `facebin api`."""
    if config is None:
        config = load_config()
    try:
        import uvicorn
    except ImportError as e:
        raise DependencyError(
            "The Facebin API requires uvicorn, which is not installed.",
            hint="Install the API dependencies with "
            "`pip install 'facebin[api]'`.") from e
    log.warning("Facebin API listening on http://%s:%s/ "
                "(open this address in your phone's browser)",
                config.api.host, config.api.port)
    uvicorn.run(create_app(config), host=config.api.host,
                port=config.api.port, log_level="warning")


def api_loop(config=None):
    """Worker entry point used by the FacebinServer supervisor."""
    serve(config)
