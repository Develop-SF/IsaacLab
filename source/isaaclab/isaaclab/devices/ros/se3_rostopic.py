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
import rclpy
import rclpy.node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from rclpy.parameter import Parameter

# IsaacLab
from ..device_base import DeviceBase


class Se3RosTopic(DeviceBase):
    """ROS interface for SE(3) control."""

    def __init__(self, pos_sensitivity: float = 0.1, rot_sensitivity: float = 0.2, timeout: float = 0.01):
        """Initialize the ROS interface.
        
        Args:
            pos_sensitivity: Magnitude of input position command scaling. Defaults to 0.4.
            rot_sensitivity: Magnitude of scale input rotation commands scaling. Defaults to 0.8.
            timeout: Time in seconds after which commands are reset if no new messages 
                    are received. Defaults to 0.01 (10ms).
        """
        super().__init__()
        # Initialize ROS node
        rclpy.init()
        self.node = rclpy.node.Node('isaaclab_se3_bridge')
        
        # Declare parameters with default values
        self.node.declare_parameter('pos_sensitivity', pos_sensitivity)
        self.node.declare_parameter('rot_sensitivity', rot_sensitivity)
        
        # sub twist command
        self.sub_twist_cmd = self.node.create_subscription(TwistStamped, 'twist_cmd', self.twist_cmd_callback, 1)
        # sub gripper command
        self.sub_gripper_cmd = self.node.create_subscription(Bool, 'gripper_cmd', self.gripper_cmd_callback, 1)
        # create timer for command timeout
        self.timer = self.node.create_timer(timeout, self.timer_callback)

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
        self.node.destroy_node()
        rclpy.shutdown()
    
    def __str__(self) -> str:
        """Returns: A string containing the information of joystick."""
        msg = f"ROS Controller for SE(3): {self.__class__.__name__}\n"
        msg += "\tIt takes ROS2 geometry_msgs/Twist messages for pos and rot commands.\n"
        msg += "\tIt also takes ROS2 std_msgs/Bool messages for gripper commands.\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tExample usage:\n"
        msg += "\t\tros2 topic pub /twist_cmd geometry_msgs/msg/Twist '{linear: {x: -1.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' -r 1 \n"
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

    def advance(self) -> tuple[np.ndarray, bool]:
        """Provides the latest command state from ROS messages.

        Returns:
            A tuple containing the delta pose command and gripper commands.
        """
        # Process any pending callbacks
        rclpy.spin_once(self.node, timeout_sec=0)
        
        # Convert euler angles to rotation vector
        rot_vec = Rotation.from_euler("XYZ", self._delta_rot).as_rotvec()
        # Return the command and gripper state
        return np.concatenate([self._delta_pos, rot_vec]), self._close_gripper

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