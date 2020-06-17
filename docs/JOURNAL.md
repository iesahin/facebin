# TIL 2020-06-17 17:03:55+0300

`docker run` has a `--gpus` parameter. Also it's possible to map the devices (like cameras) from
host to container using `-v` or `--device`. 

# 2020-06-17 16:53:10+0300

My current problem is dividing the program into separate servers each running into its own docker
container. 

- Redis: The central node where each of the other containers use as a queue. 
- Recognition-CPU: The CPU version of the recognition server. It will use tensorflow lib and fetch
    data from redis, run recognition and push to redis
- Recognition-GPU: The GPU version of the recognition server. It will use tensorflow-gpu lib and
    fetch the data from redis, run recognition and push to redis. 
- Camera: Reads camera data from a single camera, resizes it and pushes to redis. For each camera in
    the system we will run a separate version of the file
- History: Will read the recognition information from redis, build records in a central database and delete info from redis. 
