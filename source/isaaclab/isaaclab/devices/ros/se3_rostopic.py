# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS interface for SE(3) control."""

import numpy as np
import weakref
from collections.abc import Callable
from scipy.spatial.transform import Rotation

# ROS2
try:
    import rclpy
    import rclpy.node
    from geometry_msgs.msg import TwistStamped
    from std_msgs.msg import Bool
    from rclpy.parameter import Parameter
    HAS_ROS = True
except (ImportError, ModuleNotFoundError):
    HAS_ROS = False
    rclpy = None
    TwistStamped = None
    Bool = None


import torch
from dataclasses import dataclass


# IsaacLab
from ..device_base import DeviceBase, DeviceCfg


@dataclass
class Se3RosTopicCfg(DeviceCfg):
    """Configuration for SE3 ROS topic devices."""
    pos_sensitivity: float = 0.1
    rot_sensitivity: float = 0.2
    timeout: float = 0.01
    class_type: type[DeviceBase] = None  # Set later to avoid circular reference



class Se3RosTopic(DeviceBase):
    """ROS interface for SE(3) control."""

    def __init__(self, cfg: Se3RosTopicCfg):
        """Initialize the ROS interface.
        
        Args:
            cfg: Configuration object for ROS settings.
        """
        if not HAS_ROS:
            raise ImportError(
                "The 'rclpy' package is not installed or it is incompatible with your Python version. "
                "Please ensure ROS2 Humble (Python 3.10) is installed and sourced, "
                "or that you are running with a compatible Python environment."
            )
        super().__init__()

        # store inputs
        self.pos_sensitivity = cfg.pos_sensitivity
        self.rot_sensitivity = cfg.rot_sensitivity
        self.timeout = cfg.timeout
        self._sim_device = cfg.sim_device

        # Initialize ROS node

        if not rclpy.ok():
            try:
                rclpy.init(args=[])
            except Exception as e:
                # If rclpy is already initialized, we don't need to do anything
                if not rclpy.ok():
                    raise e

        self.node = rclpy.node.Node('isaaclab_se3_bridge')
        
        # Declare parameters with default values
        self.node.declare_parameter('pos_sensitivity', self.pos_sensitivity)
        self.node.declare_parameter('rot_sensitivity', self.rot_sensitivity)

        
        # sub twist command
        self.sub_twist_cmd = self.node.create_subscription(TwistStamped, 'twist_cmd', self.twist_cmd_callback, 1)
        # sub gripper command
        self.sub_gripper_cmd = self.node.create_subscription(Bool, 'gripper_cmd', self.gripper_cmd_callback, 1)
        # create timer for command timeout
        self.timer = self.node.create_timer(self.timeout, self.timer_callback)


        # default flags

        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        # dictionary for additional callbacks
        self._additional_callbacks = dict()
        
        # flag to track if new twist command received
        self._new_twist_received = False

    def __del__(self):
        """Cleanup ROS node."""
        if HAS_ROS:
            if hasattr(self, 'node'):
                self.node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()

    
    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"ROS Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\tIt takes ROS2 geometry_msgs/Twist messages for pos and rot commands.\n"
        msg += "\tIt also takes ROS2 std_msgs/Bool messages for gripper commands.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tExample usage:\n"
        msg += "\t\tros2 topic pub /twist_cmd geometry_msgs/msg/TwistStamped '{twist: {linear: {x: -1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}}' -r 1 \n"
        msg += "\t\tros2 topic pub /gripper_cmd std_msgs/Bool '{data: true}' -once \n"
        return msg

    def twist_cmd_callback(self, msg):
        """Callback for twist command messages.
        
        The message contains:
        - linear: Vector3(x, y, z) for delta position commands
        - angular: Vector3(x, y, z) for delta rotation commands (roll, pitch, yaw)
        """
        # Get current sensitivity values from parameters
        pos_sensitivity = self.node.get_parameter('pos_sensitivity').value
        rot_sensitivity = self.node.get_parameter('rot_sensitivity').value
        
        # Update position deltas (x, y, z) with sensitivity scaling
        self._delta_pos = np.array([
            msg.twist.linear.x * pos_sensitivity,
            msg.twist.linear.y * pos_sensitivity,
            msg.twist.linear.z * pos_sensitivity
        ])
        
        # Update rotation deltas (roll, pitch, yaw) with sensitivity scaling
        self._delta_rot = np.array([
            msg.twist.angular.x * rot_sensitivity,
            msg.twist.angular.y * rot_sensitivity,
            msg.twist.angular.z * rot_sensitivity
        ])
        
        # Set flag for new message
        self._new_twist_received = True

    def gripper_cmd_callback(self, msg):
        """Callback for gripper command messages."""
        self._close_gripper = msg.data

    def timer_callback(self):
        """Timer callback to reset commands if no new messages received."""
        if not self._new_twist_received:
            self._delta_pos = np.zeros(3)
            self._delta_rot = np.zeros(3)
        # Reset flag for next timer period
        self._new_twist_received = False

    def advance(self) -> torch.Tensor:
        """Provides the latest command state from ROS messages.

        Returns:
            torch.Tensor: A 7-element tensor containing:
                - delta pose: First 6 elements as [x, y, z, rx, ry, rz] in meters and radians.
                - gripper command: Last element as a binary value (+1.0 for open, -1.0 for close).
        """
        # Process any pending callbacks
        rclpy.spin_once(self.node, timeout_sec=0)
        
        # Convert euler angles to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # Return the command and gripper state
        command = np.concatenate([self._delta_pos, rot_vec])
        gripper_value = -1.0 if self._close_gripper else 1.0
        command = np.append(command, gripper_value)

        return torch.tensor(command, dtype=torch.float32, device=self._sim_device)


    def reset(self):
        """Reset the device state."""
        self._close_gripper = False
        self._delta_pos = np.zeros(3)  # (x, y, z)
        self._delta_rot = np.zeros(3)  # (roll, pitch, yaw)
        self._new_twist_received = False

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to ROS messages.

        Args:
            key: The key to bind the callback to (e.g., "RESET")
            func: The function to call. The callback function should not
                take any arguments.
        """
        self._additional_callbacks[key] = func


# Set the class type in the configuration
Se3RosTopicCfg.class_type = Se3RosTopic