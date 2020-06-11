#!/bin/zsh

FCBF=$HOME/Repository/facebin
ENV=$FCBF/env/bin/
cd $FCBF
source $ENV/activate

git -C $FCBF pull

LOGDIR=$FCBF/logs/$(date +"%F-%H-%M-%S")
mkdir -p $LOGDIR

$ENV/python3 $FCBF/facebin_gui.py 1>$LOGDIR/facebin_gui.out 2>$LOGDIR/facebin_gui.err
