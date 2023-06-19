FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ARG MASTER
ARG USER_ID
ARG GROUP_ID

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Toronto
ENV SUDO=command

RUN apt-get update && apt-get install -y \
    git  \
    wget \
    curl \
    cmake \
    pip \
    build-essential \
    pkg-config

RUN git clone https://github.com/isl-org/Open3D && cd Open3D &&\
    util/install_deps_ubuntu.sh assume-yes

RUN python3 -m pip install --upgrade pip && pip3 install --ignore-installed PyYAML
RUN python3 -m pip install opencv-python tensorflow==2.8.4 open3d==0.17.0

RUN python3 -c "import open3d.ml.tf as ml3d"

WORKDIR /app

COPY ../ ./

RUN addgroup --gid $GROUP_ID user &&\
	adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

USER user

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
CMD [ "/app/run.sh" ]