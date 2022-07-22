#!/bin/bash
username=$(id -u -n)
tag=sr-camera
imagename=cuda114-ubuntu2004
port=41234
docker run -it --rm \
        --gpus all \
        -p ${port}:${port} \
        -p 8501:8501 \
        -u $(id -u $username) \
        --name ${imagename}-$1 \
        --privileged=true \
        -v /mnt/workspace2021:/mnt/workspace2021 \
        -v /mnt/workspace2022:/mnt/workspace2022 \
        repo-luna.ist.osaka-u.ac.jp:5000/${username}/${imagename}:${tag} bash
        # jupyter notebook --ip 0.0.0.0 --port ${port} --allow-root --NotebookApp.token=''