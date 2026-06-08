#!/usr/bin/env python3
"""Show a small visual-only quadrotor marker that follows the Gazebo iris."""

from __future__ import annotations

import argparse
import copy

import rospy
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import DeleteModel, SpawnModel, SetModelState
from geometry_msgs.msg import Pose


MARKER_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{marker_name}">
    <static>false</static>
    <allow_auto_disable>false</allow_auto_disable>
    <link name="visual_quadrotor">
      <gravity>false</gravity>
      <visual name="body">
        <pose>0 0 0 0 0 0</pose>
        <geometry><sphere><radius>0.28</radius></sphere></geometry>
        <material>
          <ambient>1 0.02 0.02 1</ambient>
          <diffuse>1 0.03 0.03 1</diffuse>
          <emissive>0.45 0 0 1</emissive>
        </material>
      </visual>
      <visual name="x_arm">
        <pose>0 0 0 0 1.5708 0</pose>
        <geometry><cylinder><radius>0.055</radius><length>1.8</length></cylinder></geometry>
        <material>
          <ambient>1 0.02 0.02 1</ambient>
          <diffuse>1 0.03 0.03 1</diffuse>
          <emissive>0.35 0 0 1</emissive>
        </material>
      </visual>
      <visual name="y_arm">
        <pose>0 0 0 1.5708 0 0</pose>
        <geometry><cylinder><radius>0.055</radius><length>1.8</length></cylinder></geometry>
        <material>
          <ambient>1 0.02 0.02 1</ambient>
          <diffuse>1 0.03 0.03 1</diffuse>
          <emissive>0.35 0 0 1</emissive>
        </material>
      </visual>
      <visual name="rotor_front">
        <pose>0.9 0 0.04 0 0 0</pose>
        <geometry><cylinder><radius>0.22</radius><length>0.035</length></cylinder></geometry>
        <material><ambient>0.03 0.03 0.03 1</ambient><diffuse>0.03 0.03 0.03 1</diffuse></material>
      </visual>
      <visual name="rotor_back">
        <pose>-0.9 0 0.04 0 0 0</pose>
        <geometry><cylinder><radius>0.22</radius><length>0.035</length></cylinder></geometry>
        <material><ambient>0.03 0.03 0.03 1</ambient><diffuse>0.03 0.03 0.03 1</diffuse></material>
      </visual>
      <visual name="rotor_left">
        <pose>0 0.9 0.04 0 0 0</pose>
        <geometry><cylinder><radius>0.22</radius><length>0.035</length></cylinder></geometry>
        <material><ambient>0.03 0.03 0.03 1</ambient><diffuse>0.03 0.03 0.03 1</diffuse></material>
      </visual>
      <visual name="rotor_right">
        <pose>0 -0.9 0.04 0 0 0</pose>
        <geometry><cylinder><radius>0.22</radius><length>0.035</length></cylinder></geometry>
        <material><ambient>0.03 0.03 0.03 1</ambient><diffuse>0.03 0.03 0.03 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""


class DroneVisibilityMarker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source_pose: Pose | None = None
        self.spawned = False
        rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_states_cb, queue_size=1)
        rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=args.timeout)
        rospy.wait_for_service("/gazebo/set_model_state", timeout=args.timeout)
        self.spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        self.set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        self.delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)

    def _model_states_cb(self, msg: ModelStates) -> None:
        if self.args.source_model in msg.name:
            self.source_pose = msg.pose[msg.name.index(self.args.source_model)]

    def marker_pose(self) -> Pose:
        if self.source_pose is None:
            raise RuntimeError("Source pose is not available")
        pose = copy.deepcopy(self.source_pose)
        pose.position.z += self.args.z_offset
        return pose

    def spawn(self) -> None:
        deadline = rospy.Time.now() + rospy.Duration(self.args.timeout)
        rate = rospy.Rate(self.args.rate)
        while not rospy.is_shutdown() and self.source_pose is None:
            if rospy.Time.now() > deadline:
                raise TimeoutError(f"Timed out waiting for Gazebo model {self.args.source_model!r}")
            rate.sleep()

        self.delete_model(self.args.marker_model)
        sdf = MARKER_SDF.format(marker_name=self.args.marker_model)
        self.spawn_model(self.args.marker_model, sdf, "", self.marker_pose(), "world")
        self.spawned = True
        rospy.loginfo("Spawned visibility marker %s following %s", self.args.marker_model, self.args.source_model)

    def run(self) -> None:
        self.spawn()
        rate = rospy.Rate(self.args.rate)
        while not rospy.is_shutdown():
            if self.source_pose is not None:
                state = ModelState()
                state.model_name = self.args.marker_model
                state.pose = self.marker_pose()
                state.reference_frame = "world"
                self.set_model_state(state)
            rate.sleep()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", default="iris")
    parser.add_argument("--marker-model", default="drone_visibility_marker")
    parser.add_argument("--z-offset", type=float, default=0.35)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args(rospy.myargv()[1:])


def main() -> None:
    rospy.init_node("gazebo_drone_visibility_marker")
    DroneVisibilityMarker(parse_args()).run()


if __name__ == "__main__":
    main()
