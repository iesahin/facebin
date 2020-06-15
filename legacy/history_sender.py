#!/usr/bin/env python
import pika
import sys
import cv2
import numpy as np

connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.exchange_declare(exchange='logs', exchange_type='fanout')

for j in range(100):
    message = (np.zeros(10000000), j)
    channel.basic_publish(exchange='logs', routing_key='', body=message)
    print(" [x] Sent %r" % message)

connection.close()
