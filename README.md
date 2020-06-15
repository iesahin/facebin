
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

`facebin.db`: SQLite database that contains basic information about recognized persons and their images. Also some configuration is done here. 

`face_recognition_v6.py`: Receives faces from Redis, recognizes them using Tensorflow model and puts their information to redis

`redis_queue_utils.py`: Higher level functions to manage image queues in Redis. 

`requirements-gpu.txt`: Python pip requirements file for GPU based recognition libraries

`requirements.txt`: Python pip requirements file for CPU based recognition libraries

`show_image_dialog.py`: Simple QT dialog to show Numpy or QImage based images

`history_dialog.py`: Shows timestamp, person, image information for the recognized faces. It retrieves these from the database. 

`history_recorder.py`: One of the elements of server processes. Retrieves the recognized faces from Redis and records them to database. 

`import.py`: Imports the initial dataset from a directory by creating necessary DatasetManager object and importing the information.

`utils.py`: various utilities 

`person_dialog.py`: Shows information about a particular person.

`video_recorder.py`: Records footage videos to a directory

`visualization_utils_color.py`: To draw on images 

`watch-image-dir.sh`: Watches the changes to a directory and sends the new images using email

`facebin_init.py`: It adds `CUDA` paths to `$PATH` and imports tensorflow afterwards. 
