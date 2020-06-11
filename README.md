
This software is a Python Desktop Application and Server that uses Tensorflow, Redis, FFMPEG and such libraries to detect and recognize faces in video streams and photos. 

# Todo

- [ ] ADD model downloads to `install.sh` 
- [ ] COLLECT all model files to a `models` directory ignored by git. 
- [ ] CREATE a server python file that runs only detection and recognition modules

# Files

`install.sh`: Install necessary packages and libraries on a Ubuntu 18.04 installation and creates virtual environment. 

`facebin.sh`: Runs the desktop application by activating the environment and running python command. Keeps logs in `FACEBIN_DIR/logs/$(date)`

`facebin_gui.py`: Creates the gui and runs the servers  