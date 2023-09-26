FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

ENV DEBCONF_NOWARNINGS=yes
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
  && echo $TZ > /etc/timezone

RUN apt update && apt upgrade -y \
  && apt install -y \
    git \
    curl \
    vim \
    build-essential \
    python3 \
    python3-pip \
    python3-tk \
    python3-dev \
    python3-venv \
    python-is-python3 \
    libgl1-mesa-dev \
    python3-pyqt5 \
    python3-pyqt5.qtsvg \
    pyqt5-dev-tools \
    qttools5-dev-tools \
  && apt remove -y python3-blinker

RUN pip3 install pip --upgrade \
  && pip3 install torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu113
