import camera_controller as cc
import configparser as cp
import redis
import time

import sys
import os
import datetime as dt
import av

import redis_queue_utils as rqu

from utils import *

log = init_logging()

current_video_outstream = None
current_video_output = None
current_video_filename = None
current_video_index = None

cfg = cp.ConfigParser()
cfg.read("facebin.ini")

video_record_dir = os.path.expandvars(cfg['video']['video_record_dir'])
video_record_period = int(cfg['video']['video_seconds_per_file'])

R = redis.Redis(host='localhost', port=6379)


def get_video_index(t):
    return (t // video_record_period) * video_record_period


def get_video_filename(t, cam_id):
    log.debug("t: %s", t)
    log.debug("cam_id: %s", cam_id)
    try:
        video_index = get_video_index(t)
        tstr = dt.datetime.fromtimestamp(video_index).strftime("%F-%H-%M-%S")
    except ValueError:
        video_index = get_video_index(time.time())
        tstr = dt.datetime.fromtimestamp(video_index).strftime("%F-%H-%M-%S")

    fn = "{}/camera-{}-index-{}.mp4".format(video_record_dir, cam_id, tstr)
    return fn


def reader_loop(camera_id, camera_device, max_fps=25):
    outfile = open(
        "/tmp/facebin-camera-reader-out-pid-{}.txt".format(os.getpid()), "a")
    sys.stdout = outfile
    sys.stderr = outfile
    container = av.open(camera_device, 'r')
    # FASTER
    container.streams.video[0].thread_type = 'AUTO'
    # SKIP NONKEY FRAMES
    # container.streams.video[0].codec_context.skip_frame = 'NONKEY'
    camera_prev_time = time.time()
    skip_threshold = 10
    skips_before_quit = 100
    skips = 0
    min_delay = 1.0 / max_fps
    for frame in container.decode(video=0):
        if rqu.queue_length(rqu.CAMERA_QUEUE) > skip_threshold:
            log.warning("Waiting other elements to catch up")
            time.sleep(min_delay)
            skips += 1
            if skips >= skips_before_quit:
                log.warning("Seems no one is cleaning up the queue")
                exit()
        else:
            skips = 0

        if frame.dts is None:
            continue
        camera_current_time = time.time()
        camera_diff = camera_current_time - camera_prev_time
        if camera_diff < min_delay:
            time.sleep(min_delay - camera_diff)
            continue

        camera_diff = camera_current_time - camera_prev_time
        fps = 1 / (camera_diff)
        # log.debug("Cam Time %s: %s", camera_id, camera_diff)
        # log.debug("FPS %s: %s", camera_id, fps)
        # log.debug("frame.dts: %s", frame.dts)

        current_video_filename = get_video_filename(frame.dts, camera_id)

        redis_before = time.time()
        rqu.init_frame(
            frame.to_ndarray(format='bgr24'), frame.dts,
            current_video_filename, camera_id)
        redis_after = time.time()

        log.debug("Redis time for %s: %s", camera_id,
                  (redis_after - redis_before))

        camera_prev_time = time.time()


def reader_record_loop(camera_id):
    global current_video_index
    global current_video_output
    global current_video_outstream
    global current_video_filename

    while True:
        for packet in cam.container.demux(video=0):
            log.debug("packet: %s", packet)
            if packet.dts is None:
                log.debug("dts is None")
                continue

            vi = get_video_index(packet.dts)
            if vi != current_video_index:
                if current_video_output is not None:
                    current_video_output.close()
                current_video_filename = get_video_filename(
                    packet.dts, cam.camera_id)
                current_video_output = av.open(current_video_filename, "w")
                try:
                    current_video_outstream = current_video_output.add_stream(
                        template=cam.container.streams.video[0])
                except ValueError:
                    current_video_outstream = current_video_output.add_stream(
                        codec_name=av.Codec('mpeg4', 'w'))

                log.debug('current_video_outstream: %s',
                          current_video_outstream)
            # current_video_outstream.pix_fmt = cam.container.streams.video[
            #     0].pix_fmt
            packet.stream = current_video_outstream
            current_video_output.mux(packet)

            frame = packet.decode()
            rqu.add_frame_to_redis(
                frame.to_ndarray(format='bgr24'), frame.dts,
                current_video_filename, cam.camera_id)


if __name__ == '__main__':
    reader_loop('camera1', '/dev/video0')
