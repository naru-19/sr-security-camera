#!/bin/bash
username=$(id -u -n)
uid=$(id -u)
groupname=$(id -g -n)
gid=$(id -g)
tag=sr-camera
imagename=cuda114-ubuntu2004
docker build --build-arg USERNAME=${username} \
       --build-arg UID=${uid} \
       --build-arg GROUPNAME=${groupname} \
       --build-arg GID=${gid} \
       -t repo-luna.ist.osaka-u.ac.jp:5000/${username}/${imagename}:${tag} .