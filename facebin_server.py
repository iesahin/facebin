import facebin.server as fs

import argparse


def main(args):
    global R
    parser = argparse.ArgumentParser("Facebin server to control processes")
    parser.add_argument(['-c', '--num-camera-readers'], default=1)
    parser.add_argument(['-h', '--num-history-writers'], default=1)
    parser.add_argument(['-r', '--num-recognizers'], default=1)
    parser.add_argument(["--camera-config"],
                        default="../config/camera-config.ini")
    parser.add_argument(["--redis-host"], default="localhost")
    parser.add_argument(["--redis-port"], default=6379)
    args = parser.parse_args()

    server = fs.FacebinServer()
