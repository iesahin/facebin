import sys, os

os.environ['CPATH'] = "/usr/local/cuda/:/usr/local/cuda/lib64/" + (
    os.environ['CPATH'] if 'CPATH' in os.environ else "")
os.environ['LIBRARY_PATH'] = "/usr/local/cuda/:/usr/local/cuda/lib64/:" + (
    os.environ['LIBRARY_PATH'] if 'LIBRARY_PATH' in os.environ else "")
os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/:/usr/local/cuda/lib64/:" + (
    os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ else "")
os.environ['PATH'] = "/usr/local/cuda/:/usr/local/cuda/lib64/:" + (
    os.environ['PATH'] if 'PATH' in os.environ else "")

import tensorflow
