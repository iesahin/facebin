import facebin.server as fs

import argparse
import sys


def main():
    parser = argparse.ArgumentParser("Facebin server to control processes")
    parser.add_argument('-C', '--num-camera-readers', default=1)
    parser.add_argument('-H', '--num-history-writers', default=1)
    parser.add_argument('-R', '--num-recognizers', default=1)
    parser.add_argument("--camera-config",
                        default="../config/camera-config.ini")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", default=6379)
    parser.add_argument("--health-check-interval", default=1)
    a = parser.parse_args()
    print(a)

    server = fs.FacebinServer(num_camera_reader=a.num_camera_readers,
                              num_recognizer=a.num_recognizers,
                              num_history=a.num_history_writers,
                              health_check_interval=a.health_check_interval,
                              camera_config_file=a.camera_config
                              )


if __name__ == '__main__':
    main()
