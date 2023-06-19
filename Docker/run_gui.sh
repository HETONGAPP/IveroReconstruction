#!/bin/bash

set -e

while getopts d:f: flag
do
    case "${flag}" in
        d) workdir=${OPTARG};;
    esac
done

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ] ; do SOURCE="$(readlink "$SOURCE")"; done
DOCKER_DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
REPO_ROOT="$( cd -P "$( dirname "$DOCKER_DIR" )" && pwd )"

DOCKER_VERSION=$(docker version -f "{{.Server.Version}}")
DOCKER_MAJOR=$(echo "$DOCKER_VERSION"| cut -d'.' -f 1)

# no need to do `xhost +` anymore
XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

# specify the image, very important
IMAGE="ivero-recons"
RUNTIME=nvidia

docker run --runtime=${RUNTIME} --gpus=all --memory=16000m --cpus=12 --privileged --rm \
	  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
            --volume=${REPO_ROOT}/:/opt/src/ \
            --volume=$XSOCK:$XSOCK:rw \
            --volume=$XAUTH:$XAUTH:rw \
            --volume=/dev:/dev:rw \
            -v $workdir:/Data \
            -w /Data \
            -m 8000m \
            --cpus=12 \
	   --device /dev/dri \
            --net=host \
	   --ipc=host \
            --shm-size=1gb \
            --env="XAUTHORITY=${XAUTH}" \
            --env="DISPLAY=unix$DISPLAY" \
            ${IMAGE}
