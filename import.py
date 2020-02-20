"Installation Procedure for facebin on a new machine"

import database_api
import os
import cv2
import time
import face_recognition_v6 as fr6
import dataset_manager_v3 as dm3
import face_detection as fd
import sys

from utils import *

log = init_logging()

if len(sys.argv) > 1:
    USER_SOURCE = sys.argv[1]
else:
    USER_SOURCE = os.path.expandvars("dataset-images/user-makim-v1")

if len(sys.argv) > 2:
    DM_DIR = sys.argv[2]
else:
    DM_DIR = "dataset-images/user-{}".format(time.time())

database_api.init_db()
version_key = "v3"
DM = dm3.DatasetManager_v3(user_root=DM_DIR, encoder=fr6.FaceRecognizer_v6())

if not os.path.exists(USER_SOURCE):
    print("Cannot Find: {}".format(USER_SOURCE))
    exit()

for d in sorted(os.scandir(USER_SOURCE), key=lambda d: d.name):
    if d.is_dir():
        fields = d.name.split('--')
        log.debug(fields)
        person_name = fields[0]
        person_title = fields[1] if len(fields) > 1 else ""
        person_notes = fields[2] if len(fields) > 2 else ""

        person_id = database_api.insert_person(person_title, person_name,
                                               person_notes)
        log.debug("ID: {} Name: {} Title: {} Notes: {}".format(
            person_id, person_name, person_title, person_notes))
        for f in os.scandir(d):
            print(f.path)
            img = cv2.imread(f.path)
            if img is not None:
                DM.add_user_image(person_id, img, add_face_also=True)
