#!/bin/bash

facebin_DIR=$HOME/Repository/facebin
DLIB_DIR=$HOME/Software/dlib
OPENCV_DIR=$HOME/Software/opencv
ENVIRONMENT_DIR=$HOME/Repository/facebin/env
MODEL_DIR=${facebin_DIR}/models
MODEL_DOWNLOAD_PREFIX="https://facebin-artifacts.s3.eu-central-1.amazonaws.com/models"

sudo apt install pkg-config
sudo apt install python3-venv python3-tk python3-pip
sudo apt install redis
sudo apt install ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev libavfilter-dev
sudo apt install cmake
sudo apt install zsh tmux 


if [[ ! -d $facebin_DIR ]] ; then
    echo Clone the repository
    git clone https://bitbucket.org/teknokrat/facebin $facebin_DIR
else
    echo Pull the repository
    git -C $facebin_DIR pull
fi

if [[ ! -d $ENVIRONMENT_DIR ]] ; then
    python3 -m venv $ENVIRONMENT_DIR -p facebin
fi

source $ENVIRONMENT_DIR/bin/activate

if [[ -d /usr/local/cuda ]] ; then
    pip3 install -r $facebin_DIR/requirements-gpu.txt
else
    pip3 install -r $facebin_DIR/requirements.txt
fi

## DOWNLOAD MODEL FILES

if [[ ! -d $MODEL_DIR ]] ; then
    mkdir -p $MODEL_DIR
fi

for f in string_int_label_map_pb2.py face_label_map.pbtxt frozen_inference_graph_face.pb haarcascade_frontalface_default.xml keras_vggface mmod_human_face_detector.dat vgg_face_weights.h5 ; do
    if [[ ! -f $MODEL_DIR/${f} ]] ; then
        curl $MODEL_DOWNLOAD_PREFIX/${f} -o $MODEL_DIR/${f}
    fi
done

python3 $MODEL_DIR/keras_vggface/setup.py install

## CC=/usr/bin/cc pip3 install -r $facebin_DIR/requirements.txt
## 
## if [[ ! -d $OPENCV_DIR ]] ; then
##     echo Clone OpenCV
##     git clone https://github.com/opencv/opencv.git $OPENCV_DIR
## else
##     echo Pull OpenCV
##     git -C $OPENCV_DIR pull
## fi
## 
OPENCV_C=$(ls -1 $ENVIRONMENT_DIR/lib/*/site-packages/cv2.* | wc -l)

if [[ $OPENCV_C -eq 0 ]] ; then
    OPENCV_BUILD=$OPENCV_DIR/build-$(date "+%F-%H-%M-%S")
    mkdir $OPENCV_BUILD
    cd $OPENCV_BUILD
    sudo apt install libv4l-dev
    # cmake -C $OPENCV_BUILD -DWITH_CUDA=ON -DWITH_FFMPEG=ON -DWITH_LIBV4L=ON ..
    cmake -C $OPENCV_BUILD -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D WITH_CUDA=ON -D WITH_CUBLAS=ON -D WITH_TBB=ON -D WITH_V4L=ON -D WITH_QT=ON -D WITH_OPENGL=ON -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES" ..
    make -C $OPENCV_BUILD -j4 
    for d in $ENVIRONMENT_DIR/lib/*/site-packages/ ; do
        cp $OPENCV_BUILD/lib/python3/cv2* $d
    done
fi

## if [[ ! -e /dev/video1 ]] ; then
##    sudo apt install v4l2loopback-dkms v4l2loopback-utils
##    sudo modprobe v4l2loopback devices=4
## fi
## 
