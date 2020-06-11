
This software is a Python Desktop Application and Server that uses Tensorflow, Redis, FFMPEG and such libraries to detect and recognize faces in video streams and photos. 

# Todo

- [ ] ADD model downloads to `install.sh` 
- [ ] COLLECT all model files to a `models` directory ignored by git. 
- [ ] CREATE a server python file that runs only detection and recognition modules

# Files

`install.sh`: Install necessary packages and libraries on a Ubuntu 18.04 installation and creates virtual environment. 

`facebin.sh`: Runs the desktop application by activating the environment and running python command. Keeps logs in `FACEBIN_DIR/logs/$(date)`

`facebin_gui.py`: Creates the gui and runs the servers. Main entrance of the software. 

`camera_controller.py`: A class to manage cameras and start/stop them using `libav`

`camera_dialog.py`: A Qt dialog to control up to 4 cameras

`camera_reader.py`: Uses `libav` to read frames from cameras and puts them to Redis server in localhost

`database_api.py`: Provides database access functions

`dataset_manager_v3.py`: Provides training dataset access functions. 

`export.py`: Exports the dataset to image files to `~/facebin-artifacts/export-$(time.time)`

`face_detection.py`: Contains Tensorflow and Haar based classes for face detection. Receives image files from Redis queue, runs face detection on them and stores there again. 

`face_recognition_v6.py`: Receives 

`facebin_init.py`: It adds `CUDA` paths to `$PATH` and imports tensorflow afterwards. 
