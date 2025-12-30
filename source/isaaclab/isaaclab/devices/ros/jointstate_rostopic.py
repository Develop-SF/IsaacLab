# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS interface for JointState control."""

import numpy as np
import weakref
from collections.abc import Callable
from typing import Optional
from scipy.spatial.transform import Rotation
import logging

# ROS2
try:
    import rclpy
    import rclpy.node
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory
    HAS_ROS = True
except (ImportError, ModuleNotFoundError):
    HAS_ROS = False
    rclpy = None
    JointState = None
    JointTrajectory = None


import torch
from dataclasses import dataclass


# IsaacLab
from ..device_base import DeviceBase, DeviceCfg


@dataclass
class JointStateRosTopicCfg(DeviceCfg):
    """Configuration for JointState ROS topic devices."""
    class_type: type[DeviceBase] = None  # Set later


logger = logging.getLogger(__name__)

class JointStateRosTopic(DeviceBase):
    """ROS interface for JointState control."""

    def __init__(self, cfg: JointStateRosTopicCfg):
        """Initialize the ROS interface for joint trajectory control.
        
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

        self._sim_device = cfg.sim_device
        logger.info("Initializing JointStateRosTopic...")

        
        # Initialize ROS node
        if not rclpy.ok():
            logger.info("Initializing ROS context...")
            try:
                rclpy.init(args=[])
            except Exception as e:
                # If rclpy is already initialized, we don't need to do anything
                if not rclpy.ok():
                    raise e

        
        self.node = rclpy.node.Node('isaaclab_jointtrajectory_bridge')
        logger.info("Created ROS node: %s", self.node.get_name())
        
        # Initialize joint state storage
        self._current_joint_positions = None
        self._new_command_received = False
        
        # Subscribe to joint trajectory commands
        self.sub_joint_state_cmd = self.node.create_subscription(
            JointTrajectory, 
            'joint_trajectory_cmd', 
            self.joint_trajectory_cmd_callback, 
            10
        )
        logger.info("Subscribed to topic: joint_trajectory_cmd")
        
        # Dictionary for additional callbacks
        self._additional_callbacks = dict()
        logger.info("JointStateRosTopic initialization complete")

    def __del__(self):
        """Cleanup ROS node."""
        if HAS_ROS:
            logger.info("Cleaning up ROS node...")
            try:
                self.node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception as e:
                logger.error("Error during ROS cleanup: %s", str(e))


    def joint_trajectory_cmd_callback(self, msg):
        """Callback for joint trajectory command messages."""
        try:
            if len(msg.points) > 0:
                logger.debug("Received trajectory message with %d points", len(msg.points))
                # Take the first point from the trajectory
                self._current_joint_positions = np.array(msg.points[0].positions)
                self._new_command_received = True
                logger.debug("Updated joint positions: %s", self._current_joint_positions)
            else:
                logger.warning("Received empty trajectory message")
        except Exception as e:
            logger.error("Error in joint trajectory callback: %s", str(e))
    
    def advance(self) -> Optional[torch.Tensor]:
        """Process ROS messages and return the latest joint positions."""
        try:
            # Process any pending callbacks
            rclpy.spin_once(self.node, timeout_sec=0)
            
            # Return the current joint positions
            if self._new_command_received:
                logger.debug("Returning new joint positions")
                self._new_command_received = False
                return torch.tensor(self._current_joint_positions, dtype=torch.float32, device=self._sim_device)
            return None
        except Exception as e:
            logger.error("Error in advance: %s", str(e))
            return None


    def reset(self):
        """Reset the device state."""
        logger.info("Resetting device state")
        self._current_joint_positions = None
        self._new_command_received = False

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to ROS messages."""
        logger.info("Adding callback for key: %s", key)
        self._additional_callbacks[key] = func

    def __str__(self) -> str:
        """Returns a string representation of the interface."""
        return f"""JointStateRosTopic:
    Node name: {self.node.get_name()}
    Subscribed topic: joint_trajectory_cmd
    ROS context initialized: {rclpy.ok()}"""


# Set the class type in the configuration
JointStateRosTopicCfg.class_type = JointStateRosTopic

