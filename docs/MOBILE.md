# Facebin on your phone

Facebin ships a mobile web app (a PWA) served by the Facebin API server.
The recognition pipeline keeps running on your server — the phone is a
remote monitor: live annotated camera streams, the appearance history,
the people list, and pipeline status.

This deliberately is *not* a native Android app: the heavy parts
(TensorFlow, camera capture, Redis) cannot run on the phone anyway, and a
web app needs no Play Store, no Android SDK, and works on iPhones and
desktops too. The API it uses (`/api/...`, see below) is exactly what a
native app would need, so a Kotlin/Flutter client can be added later
without server changes.

## Setup

1. Install the API dependencies on the server:

   ```sh
   pip install -e ".[api]"
   ```

2. Enable the API in `facebin.toml`:

   ```toml
   [api]
   enabled = true      # started by `facebin run` / `facebin server`
   host = "0.0.0.0"
   port = 8420
   ```

3. Start Facebin as usual (`facebin` or `facebin server`), or start only
   the API against an already-running server with `facebin api`.

4. On your Android phone (same network), open
   `http://<server-address>:8420/` in Chrome, sign in (default
   `admin`/`admin` — change it), and choose **Add to Home screen** when
   prompted. The app then launches full-screen like a native app.

## What you get

- **Live**: every configured camera as a live MJPEG stream with the
  recognizer's annotations (names and boxes) drawn in.
- **History**: recent appearances with face snapshots, filterable to
  unknown people only.
- **People**: the person dataset with face thumbnails; add new person
  records from the phone.
- **Status**: queue lengths and pipeline health at a glance.

## Security notes

- The app is designed for a **trusted LAN**. If you expose it beyond
  that, put it behind a reverse proxy with TLS (Caddy/nginx) — the
  bearer token and the MJPEG `?token=` parameter are plaintext over
  plain HTTP.
- Logins are checked against Facebin's `login` table, which stores
  passwords in plaintext (a legacy of the 0.1.x schema). Change the
  default admin password and treat the database file accordingly.
- Tokens live in server memory and expire after `session_ttl` seconds
  (default: one day); restarting the API server signs everyone out.

## API reference (for native clients)

All endpoints are under `/api` and need `Authorization: Bearer <token>`
(or `?token=` for streams/images embedded in `<img>` tags).
Interactive docs are served at `/api/docs`.

| Endpoint | Description |
| -------- | ----------- |
| `POST /api/login` `{username, password}` | Returns `{token, username}` |
| `POST /api/logout` | Revokes the token |
| `GET /api/me` | Current user and permissions |
| `GET /api/status` | Version, database path, queue lengths |
| `GET /api/cameras` | Configured cameras and their stream URLs |
| `GET /api/stream/{camera_id}` | MJPEG stream (`multipart/x-mixed-replace`) |
| `GET /api/history?person_id&camera_id&begin&end&unknown_only&limit` | Appearance records |
| `GET /api/history/{id}/face` / `.../camera` | Recorded snapshots (PNG) |
| `GET /api/persons` / `POST /api/persons` | List / create people |
| `GET /api/persons/{id}` | Person detail with images and recent history |
| `GET /api/persons/{id}/image` | A face image of the person |

Note on streams: the stream endpoint consumes frames from the same
recognizer output queue as the desktop GUI and forwards them to the
history queue (that hand-off is what feeds appearance recording). Run
either the desktop GUI or phone viewers as the primary consumer of a
camera — several simultaneous viewers of the same camera will each get a
subset of the frames.
