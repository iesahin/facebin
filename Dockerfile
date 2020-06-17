FROM nvidia/cuda:10.2-base

RUN apt-get update && apt-get upgrade -y && apt-get install -y python3 python3-pip redis

COPY . /root/facebin

RUN pip3 install -r /root/facebin/init/requirements-gpu.txt

CMD ['/root/facebin/facebin-server -H 0 -R 1 -C 0']



