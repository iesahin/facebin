# Headless Facebin recognition server.
# Build:  docker build -t facebin .
# Run:    docker run --gpus all -v $HOME/facebin-data:/data \
#             -v $PWD/facebin.toml:/etc/facebin/facebin.toml facebin
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        redis-server ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/facebin
COPY . .

RUN pip install --no-cache-dir ".[ml]"

ENV FACEBIN_CONFIG=/etc/facebin/facebin.toml

CMD ["facebin", "server"]
