#!/bin/zsh

repodir=${0:A:h}

. $HOME/.profile
. $repodir/env/bin/activate

sha1f=/tmp/facebin-sha1
if [[ -f $sha1f ]] ; then
    facebin_CURRENT_SHA1=$(cat $sha1f)
else 
    echo "Cannot find facebin Current Sha1" 
    facebin_CURRENT_SHA1=
fi
git -C $repodir pull -q
NEW_SHA1=$(git -C $repodir log -n 1 | head -n 1)
echo "$NEW_SHA1"
if [[ $NEW_SHA1 != $facebin_CURRENT_SHA1 ]] ; then
    outfile=/tmp/facebin-output-${RANDOM}.txt
    CUDA_VISIBLE_DEVICES=1 python3 /home/iesahin/Repository/facebin/face_recognition_v3.py > $outfile
    ntfy -t "facebin done: $(date +%H:%M)" send "$(tail $outfile)"
    debugfile=$(ls -t1 /tmp/facebin-debug-* | head -n 1)
    filename="facebin ${NEW_SHA1[8,14]} @$(hostname) Test Results.cres"
    mv $outfile "$HOME/org/INBOX/${filename}"
    mv $debugfile "$HOME/org/INBOX/debug-${filename}"
    echo $NEW_SHA1 > $sha1f
    SLEEP_COUNTER=10
else
    echo "No Changes in the Repository"
    SLEEP_COUNTER=$(($RANDOM % 100))
fi
echo "Sleeping for $SLEEP_COUNTER seconds"
sleep $SLEEP_COUNTER

