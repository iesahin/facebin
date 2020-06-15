import redis
import rq

import av
import av.datasets

import camera_controller as cc
import time

R = redis.Redis(host='localhost', port=6379)
history_queue = rq.Queue("history", connection=R)
video_queue = rq.Queue("video", connection=R)
cams = cc.get_camera_controllers()

containers = {}

def init_container()

for k, c in cams.items():
    containers[k] = av.open(c.device)
    # TODO: Make any necessary configuration here
    # This is faster
    containers[k].streams.video[0].thread_type = 'AUTO'
    # Get only the keyframes
    containers[k].streams.video[0].codec_context.skip_frame = 'NONKEY'

    for k, c in containers.items():
        for frame in c.decode(video=0):
            frame.to_image().save('/tmp/frame-{}-{}.jpg'.format(
                frame.pts, time.time()))

def read_from_container(
