#!/bin/zsh

IMAGE_STORE=$HOME/facebin-data/image-store
SENT_FILES=$HOME/facebin-data/sent-files.txt

touch $SENT_FILES

fswatch -0 $IMAGE_STORE | while read -d "" path ; do
    print $path
    if [[ "$path" =~ '.*/face-.*' ]] ; then
        print "face"
        if [[ x"$(/bin/grep $path $SENT_FILES)" = x ]] ; then
            print "not found"
            /usr/bin/mutt -s $path -a $path -- i.emre.sahin@gmail.com <<EOF
EOF
            echo $path >> $SENT_FILES
        fi
    fi
done

