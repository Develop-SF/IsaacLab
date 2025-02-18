# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS interface for JointState control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

# ROS2
import rclpy
import rclpy.node
from sensor_msgs.msg import JointState

# IsaacLab
from ..device_base import DeviceBase


class JointStateRosTopic(DeviceBase):
    """ROS interface for JointState control."""

    def __init__(self):
        super().__init__()
        self.node = rclpy.node.Node('isaaclab_jointstate_bridge')
        # sub joint state command
        self.sub_joint_state_cmd = self.node.create_subscription(JointState, 'joint_state_cmd', self.joint_state_cmd_callback, 10)
        # pub joint state
        self.pub_joint_state = self.node.create_publisher(JointState, 'isaaclab_joint_state', 10)

    def joint_state_cmd_callback(self, msg):
        pass
    
    def advance(self):
        pass

    def reset(self):
        pass

    # def add_callback(self, key: Any, func: Callable):
    #     pass
