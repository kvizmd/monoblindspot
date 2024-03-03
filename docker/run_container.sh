#!/bin/bash

image_name=$1
container_name=$2
data_dir=$3
output_dir=$4

if [ -z "$image_name" ] || [ -z "$container_name" ] || [ -z "$data_dir" ] || [ -z "$output_dir" ]; then
  echo "Usage: run_container.sh [image_name] [container_name] [data_dir] [output_dir]"
  exit
fi

current_dir=`pwd -P`

docker run \
  -it -d \
  --gpus all \
  --name $container_name \
  -e TZ=Asia/Tokyo \
  -v $data_dir:/root/work/data:ro \
  -v $output_dir:/root/work/outputs \
  -v /dev/shm:/dev/shm \
  -v $current_dir:/root/work \
  -v $output_dir/mlruns:/root/work/mlruns \
  -w /root/work \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $HOME/.Xauthority:/root/.Xauthority \
  -v /var/run/dbus:/var/run/dbus \
  -e DBUS_SESSION_BUS_ADDRESS=/var/dbus/bus \
  --privileged \
  --net host \
  $image_name \
  bash

docker start $container_name
docker exec -it $container_name pip install -r requirements.txt
docker exec -it $container_name bash
