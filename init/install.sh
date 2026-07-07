#!/bin/bash
# Install Facebin on a Debian/Ubuntu machine.
# Installs system packages, creates a virtual environment, installs the
# Python package with its extras, downloads the model files, and writes a
# starter configuration.
set -euo pipefail

FACEBIN_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENVIRONMENT_DIR="$FACEBIN_DIR/.venv"
MODEL_DIR="${MODEL_DIR:-$HOME/facebin-data/models}"
MODEL_DOWNLOAD_PREFIX="https://facebin-artifacts.s3.eu-central-1.amazonaws.com/models"

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip redis-server ffmpeg pkg-config

if [[ ! -d $ENVIRONMENT_DIR ]] ; then
    python3 -m venv "$ENVIRONMENT_DIR"
fi

# shellcheck disable=SC1091
source "$ENVIRONMENT_DIR/bin/activate"
pip install --upgrade pip

if [[ -d /usr/local/cuda ]] ; then
    pip install -r "$FACEBIN_DIR/init/requirements-gpu.txt"
    pip install -e "$FACEBIN_DIR" --no-deps
else
    pip install -e "$FACEBIN_DIR[ml,ui]"
fi

## Download model files

mkdir -p "$MODEL_DIR"

for f in face_label_map.pbtxt frozen_inference_graph_face.pb \
         haarcascade_frontalface_default.xml mmod_human_face_detector.dat \
         vgg_face_weights.h5 ; do
    if [[ ! -f "$MODEL_DIR/$f" ]] ; then
        echo "Downloading $f"
        curl -fL "$MODEL_DOWNLOAD_PREFIX/$f" -o "$MODEL_DIR/$f"
    fi
done

# The generated protobuf module belongs next to label_map_util.py inside
# the package.
PB2="$FACEBIN_DIR/facebin/models/string_int_label_map_pb2.py"
if [[ ! -f "$PB2" ]] ; then
    echo "Downloading string_int_label_map_pb2.py"
    curl -fL "$MODEL_DOWNLOAD_PREFIX/string_int_label_map_pb2.py" -o "$PB2"
fi

## Create a starter configuration if none exists

if [[ ! -f "$FACEBIN_DIR/facebin.toml" ]] ; then
    facebin init-config --path "$FACEBIN_DIR/facebin.toml"
    echo "Edit $FACEBIN_DIR/facebin.toml, set [models] dir = \"$MODEL_DIR\","
    echo "then run: facebin check && facebin"
fi
