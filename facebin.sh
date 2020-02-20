#!/bin/zsh

MKMF=$HOME/Repository/facebin
ENV=$MKMF/env/bin/
cd $MKMF
source $ENV/activate

git -C $MKMF pull

LOGDIR=$MKMF/logs/$(date +"%F-%H-%M-%S")
mkdir -p $LOGDIR

$ENV/python3 $MKMF/main_gui.py 1>$LOGDIR/main_gui.out 2>$LOGDIR/main_gui.err
