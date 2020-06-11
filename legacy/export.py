import database_api
import os
import cv2
import time
import face_recognition_v6 as fr6
import face_detection as fd

import dataset_manager_v3 as dm3

DM = dm3.DatasetManager_v3()

person_list = database_api.person_list()
export_dir = "facebin-artifacts/export-{}/".format(time.time())

for p in person_list:
    person_id, title, name, notes = p
    dirname = "{}--{}--{}".format(name, title, notes)
    images = DM.images_by_id(person_id)
    outdir = os.path.join(export_dir, dirname)
    os.makedirs(outdir)
    for i in range(images.shape[0]):
        outfile = os.path.join(outdir, "{}.png".format(i))
        cv2.imwrite(outfile, images[i])
        print("Wrote: {}".format(outfile))
