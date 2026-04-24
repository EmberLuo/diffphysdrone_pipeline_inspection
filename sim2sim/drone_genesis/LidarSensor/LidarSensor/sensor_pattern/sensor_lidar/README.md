# sensor_lidar

## LidarSensor/LidarSensor/sensor_pattern/sensor_lidar/lidar_vis_ros1.py

安装nvidia-container-toolkit

```bash
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

```bash
sudo systemctl restart docker
```

拉取基础镜像

```bash
docker pull osrf/ros:noetic-desktop-full
```

启动新容器

```bash
xhost +local:root
```

```bash
docker run -it \
  --name ros1_noetic_gpu \
  --net=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/ember/GitHub/drones_nav_genesis:/ws \
  osrf/ros:noetic-desktop-full bash
```

容器内安装依赖

```bash
sudo apt-get update
```

```bash
apt-get install -y python3-pip
```

```bash
python3 -m pip install --upgrade pip setuptools wheel && \
python3 -m pip install --no-cache-dir mujoco taichi pynput && \
python3 -m pip install --no-cache-dir numpy==1.24.4 scipy==1.10.1
```

新开一个终端

```bash
docker commit ros1_noetic_gpu ros1_noetic:lidar_v1
```

原来的容器退出：

```bash
docker stop ros1_noetic_gpu
```

之后直接用新镜像启动：

```bash
docker run -it \
  --name ros1_noetic_gpu \
  --net=host \
  --gpus all \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/ember/GitHub/drones_nav_genesis:/ws \
  ros1_noetic:lidar_v1 bash
```

```bash
source /opt/ros/noetic/setup.bash
```

启动roscore

```bash
roscore
```

再新开两个终端，分别执行：

```bash
docker exec -it ros1_noetic_gpu bash
source /opt/ros/noetic/setup.bash
cd /ws/LidarSensor/LidarSensor/sensor_pattern/sensor_lidar
python3 lidar_vis_ros1.py
```

```bash
docker exec -it ros1_noetic_gpu bash
source /opt/ros/noetic/setup.bash
rviz
```
