# Overview
This repository contains the simulation and deployment for Unitree Go1 based on [unitree_guide](https://github.com/unitreerobotics/unitree_guide). The code has been modified to work for the sim-to-sim and sim-to-real of the Deep Reinforcement Learning control method [ALARM: Safe Reinforcement Learning with Reliable Mimicry for Robust Legged Locomotion](https://sucro-legged.github.io/ALARM/).

## Dependencies
- unbuntu20.04
- ros1 noetic
- eigen
- lcm-1.4.0
- onnxruntime-1.16.3

## Usage
1. clone this repository
2. build
```shell
cd unitree_ws
catkin_make
```
3. start simulation
```shell
./scripts/sim.sh
```
4. start control
```shell
./scripts/ctrl.sh
```

## Issues
1. Could not find a configuration file for package move_base_msgs.
```shell
sudo apt-get install ros-noetic-navigation
```
2. error while loading shared libraries: libxmlrpcpp.so: cannot open shared object file: No such file or directory
```shell
sudo gedit /etc/ld.so.conf
```
add the following lines to the file

> /opt/ros/noetic/lib

> /usr/local/lib

```shell
sudo ldconfig
```

## References
- unitree_guide: https://github.com/unitreerobotics/unitree_guide