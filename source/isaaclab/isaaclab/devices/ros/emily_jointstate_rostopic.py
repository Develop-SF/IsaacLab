# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ROS interface for Emily robot joint state control with multiple trajectory controllers."""

import numpy as np
import weakref
from collections.abc import Callable
from collections import defaultdict
from typing import Optional, Dict, List, Set
import logging
import threading
import math

# ROS2
import rclpy
import rclpy.node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from rclpy.executors import MultiThreadedExecutor
# IsaacLab
from ..device_base import DeviceBase

logger = logging.getLogger(__name__)

class EmilyJointStateRosTopic(DeviceBase):
    """ROS interface for Emily robot joint state control with multiple trajectory controllers."""

    def __init__(self, trajectory_point="last"):
        """Initialize the ROS interface for Emily robot joint trajectory control."""
        super().__init__()
        logger.info("Initializing EmilyJointStateRosTopic...")
        
        # Store which trajectory point to use
        self._trajectory_point = trajectory_point
        logger.info(f"Using trajectory point: {self._trajectory_point}")
        
        # For trajectory following
        self._current_trajectories = {}  # Store active trajectories by controller
        self._trajectory_start_times = {}  # Store when each trajectory started
        self._last_advance_time = None  # Track last advance time for trajectory execution
        
        # Initialize ROS node
        if not rclpy.ok():
            logger.info("Initializing ROS context...")
            rclpy.init()
        
        # Create a node
        self.node = rclpy.node.Node('isaaclab_emily_jointtrajectory_bridge')
        logger.info("Created ROS node: %s", self.node.get_name())
        
        # Define Emily's controller topics
        self.controller_names = [
            'la_trajectory_controller',
            'lh_trajectory_controller',
            'lh_wr_trajectory_controller',
            'ra_trajectory_controller',
            'rh_trajectory_controller',
            'rh_wr_trajectory_controller'
        ]
        
        # Initialize joint state storage
        self._joint_positions = {}  # Dictionary to store positions by joint name
        self._new_commands = defaultdict(bool)  # Track which controllers have new commands
        self._subscribers = []  # Store subscribers
        
        # Add tracking for joint command history
        self._joint_command_history = {}  # Track last command for each joint
        self._controller_joint_map = {}  # Map controllers to their joints
        
        # Define joint transformations for coordinate system differences
        # between MoveIt and Isaac Sim
        self._joint_transformations = {
            "la_shoulder_pan_joint": lambda x: x,
            # Other joints may need transformations too
            "la_shoulder_lift_joint": lambda x: x,
            "la_elbow_joint": lambda x: x,
            "la_wrist_1_joint": lambda x: x,
            "la_wrist_2_joint": lambda x: x,
            "la_wrist_3_joint": lambda x: x,
            "ra_shoulder_pan_joint": lambda x: x,
            "ra_shoulder_lift_joint": lambda x: x,
            "ra_elbow_joint": lambda x: x,
            "ra_wrist_1_joint": lambda x: x,
            "ra_wrist_2_joint": lambda x: x,
            "ra_wrist_3_joint": lambda x: x,
            # Left hand joints
            "lh_WRJ1": lambda x: x, "lh_WRJ2": lambda x: x,
            "lh_FFJ1": lambda x: x, "lh_FFJ2": lambda x: x, "lh_FFJ3": lambda x: x, "lh_FFJ4": lambda x: x,
            "lh_LFJ1": lambda x: x, "lh_LFJ2": lambda x: x, "lh_LFJ3": lambda x: x, "lh_LFJ4": lambda x: x, "lh_LFJ5": lambda x: x,
            "lh_MFJ1": lambda x: x, "lh_MFJ2": lambda x: x, "lh_MFJ3": lambda x: x, "lh_MFJ4": lambda x: x,
            "lh_RFJ1": lambda x: x, "lh_RFJ2": lambda x: x, "lh_RFJ3": lambda x: x, "lh_RFJ4": lambda x: x,
            "lh_THJ1": lambda x: x, "lh_THJ2": lambda x: x, "lh_THJ3": lambda x: x, "lh_THJ4": lambda x: x, "lh_THJ5": lambda x: x,
            # Right hand joints
            "rh_WRJ1": lambda x: x, "rh_WRJ2": lambda x: x,
            "rh_FFJ1": lambda x: x, "rh_FFJ2": lambda x: x, "rh_FFJ3": lambda x: x, "rh_FFJ4": lambda x: x,
            "rh_LFJ1": lambda x: x, "rh_LFJ2": lambda x: x, "rh_LFJ3": lambda x: x, "rh_LFJ4": lambda x: x, "rh_LFJ5": lambda x: x,
            "rh_MFJ1": lambda x: x, "rh_MFJ2": lambda x: x, "rh_MFJ3": lambda x: x, "rh_MFJ4": lambda x: x,
            "rh_RFJ1": lambda x: x, "rh_RFJ2": lambda x: x, "rh_RFJ3": lambda x: x, "rh_RFJ4": lambda x: x,
            "rh_THJ1": lambda x: x, "rh_THJ2": lambda x: x, "rh_THJ3": lambda x: x, "rh_THJ4": lambda x: x, "rh_THJ5": lambda x: x,
        }
        
        # Subscribe to all controller topics
        for controller_name in self.controller_names:
            topic = f'/{controller_name}/command'
            sub = self.node.create_subscription(
                JointTrajectory, 
                topic, 
                lambda msg, topic=topic: self.joint_trajectory_cmd_callback(msg, topic), 
                10
            )
            self._subscribers.append(sub)
            logger.info(f"Subscribed to topic: {topic}")
        
        # Subscribe to joint states to monitor actual robot state
        self.joint_state_sub = self.node.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        logger.info("Subscribed to topic: /joint_states")
        self._current_joint_states = {}  # Store current joint states
        
        # Dictionary for additional callbacks
        self._additional_callbacks = dict()
        
        # Create a separate thread for spinning the node
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._spin_node, daemon=True)
        self._spin_thread.start()
        
        # Lock for thread safety when updating joint positions
        self._joint_lock = threading.Lock()
        
        logger.info("EmilyJointStateRosTopic initialization complete")

    def _spin_node(self):
        """Spin the ROS node in a separate thread."""
        try:
            self._executor.spin()
        except Exception as e:
            logger.error(f"Error in ROS spin thread: {str(e)}")
        finally:
            logger.info("ROS spin thread exiting")

    def __del__(self):
        """Cleanup ROS node."""
        logger.info("Cleaning up ROS node...")
        try:
            # Destroy node
            self.node.destroy_node()
            
            # Shutdown ROS if we're the last node
            if rclpy.ok():
                rclpy.shutdown()
        except Exception as e:
            logger.error("Error during ROS cleanup: %s", str(e))

    def joint_state_callback(self, msg):
        """Callback for joint state messages to monitor actual robot state."""
        try:
            # Update current joint states
            with self._joint_lock:
                for i, joint_name in enumerate(msg.name):
                    if i < len(msg.position):
                        self._current_joint_states[joint_name] = msg.position[i]
            
            # Log joint state differences compared to commands (every 50 messages to avoid spam)
            if hasattr(self, '_joint_state_counter'):
                self._joint_state_counter += 1
            else:
                self._joint_state_counter = 0
                
            if self._joint_state_counter % 50 == 0:
                self._log_joint_differences()
                
        except Exception as e:
            logger.error(f"Error in joint state callback: {str(e)}")
    
    def _log_joint_differences(self):
        """Log differences between commanded and actual joint positions."""
        with self._joint_lock:
            if not self._joint_positions or not self._current_joint_states:
                return
                
            # Find joints that exist in both dictionaries
            common_joints = set(self._joint_positions.keys()) & set(self._current_joint_states.keys())
            
            # Calculate differences
            differences = {}
            for joint in common_joints:
                cmd_pos = self._joint_positions[joint]
                actual_pos = self._current_joint_states[joint]
                diff = cmd_pos - actual_pos
                if abs(diff) > 0.01:  # Only log significant differences
                    differences[joint] = (cmd_pos, actual_pos, diff)
            
            if differences:
                logger.info("=== Joint Command vs Actual State Differences ===")
                for joint, (cmd, actual, diff) in differences.items():
                    logger.info(f"Joint {joint}: Command={cmd:.4f}, Actual={actual:.4f}, Diff={diff:.4f}")
                logger.info("================================================")

    def joint_trajectory_cmd_callback(self, msg, topic, is_action=False):
        """Callback for joint trajectory command messages from different controllers."""
        try:
            if len(msg.points) > 0:
                # Use print for critical information to ensure it's always visible
                logger.info(f"Received trajectory on {topic} with {len(msg.points)} points")
                
                # Get joint names and positions from the message
                joint_names = msg.joint_names
                
                # Extract controller name from topic
                controller_name = topic.split('/')[-2]  # Extract controller name from topic
                
                # Get controller prefix for strict joint matching
                controller_prefix = controller_name.split('_')[0]  # e.g., 'la', 'ra', 'lh', 'rh'
                
                # Store the joints associated with this controller
                if controller_name not in self._controller_joint_map:
                    self._controller_joint_map[controller_name] = set()
                self._controller_joint_map[controller_name].update(joint_names)
                
                # If we're in "follow" mode, store the trajectory for execution in advance()
                if self._trajectory_point == "follow":
                    logger.info(f"Storing trajectory with {len(msg.points)} points for sequential execution")
                    
                    # Store the trajectory for execution
                    with self._joint_lock:
                        # Store the trajectory with its start time
                        current_time = self.node.get_clock().now().to_msg()
                        self._current_trajectories[controller_name] = {
                            'trajectory': msg,
                            'current_point_index': 0,
                            'start_time': current_time
                        }
                        self._trajectory_start_times[controller_name] = current_time
                        
                        # IMPORTANT: Set the initial position from the first point
                        # This ensures the robot starts moving immediately
                        first_positions = msg.points[0].positions
                        
                        # Update the joint positions dictionary with the first point
                        updated_joints = []
                        for i, joint_name in enumerate(joint_names):
                            if i < len(first_positions):
                                # Only update joints that match the controller's prefix
                                joint_prefix = joint_name.split('_')[0]
                                
                                if joint_prefix != controller_prefix:
                                    continue
                                
                                # Apply any necessary transformations
                                position = first_positions[i]
                                if joint_name in self._joint_transformations:
                                    position = self._joint_transformations[joint_name](position)
                                
                                # Store the position
                                self._joint_positions[joint_name] = position
                                self._joint_command_history[joint_name] = {
                                    'position': position,
                                    'controller': controller_name,
                                    'is_action': is_action,
                                    'timestamp': current_time,
                                    'trajectory_point': 0  # First point
                                }
                                
                                updated_joints.append(joint_name)
                        
                        # Mark this controller as having new commands
                        self._new_commands[controller_name] = True
                        
                        logger.info(f"Set initial positions from first point for {len(updated_joints)} joints")
                    
                    return
                
                # For non-follow modes, determine which point in the trajectory to use based on the configuration
                if self._trajectory_point == "first":
                    point_index = 0
                    point_label = "FIRST POINT"
                elif self._trajectory_point == "last":
                    point_index = -1
                    point_label = "LAST POINT"
                else:
                    try:
                        # Try to parse as an integer index
                        point_index = int(self._trajectory_point)
                        if point_index < 0:
                            # Convert negative index to positive
                            point_index = len(msg.points) + point_index
                        # Ensure index is within bounds
                        point_index = max(0, min(point_index, len(msg.points) - 1))
                        point_label = f"POINT {point_index}"
                    except ValueError:
                        # Default to last point if parsing fails
                        logger.warning(f"Invalid trajectory_point value: {self._trajectory_point}, using last point")
                        point_index = -1
                        point_label = "LAST POINT (default)"
                
                # Get the positions from the specified point
                positions = msg.points[point_index].positions
                
                logger.info(f"Using {point_label} from trajectory with {len(msg.points)} points")
                
                # Track changes in joint positions
                with self._joint_lock:
                    updated_joints = []
                    ignored_joints = []
                    
                    # Update the joint positions dictionary
                    for i, joint_name in enumerate(joint_names):
                        if i < len(positions):
                            # Check if this joint belongs to the arm/hand being controlled
                            # STRICT MATCHING: Only update joints that match the controller's prefix
                            joint_prefix = joint_name.split('_')[0]  # e.g., 'la', 'ra', 'lh', 'rh'
                            
                            if joint_prefix != controller_prefix:
                                # Use WARNING level for important messages
                                logger.warning(f"Ignoring joint {joint_name} as it doesn't match controller {controller_name} prefix")
                                ignored_joints.append(joint_name)
                                continue
                            
                            # Apply any necessary transformations to match Isaac Sim's coordinate system
                            position = positions[i]
                            
                            # Apply joint-specific transformations if needed
                            if joint_name in self._joint_transformations:
                                position = self._joint_transformations[joint_name](position)
                            
                            # Store the position and update command history
                            self._joint_positions[joint_name] = position
                            self._joint_command_history[joint_name] = {
                                'position': position,
                                'controller': controller_name,
                                'is_action': is_action,
                                'timestamp': self.node.get_clock().now().to_msg()
                            }
                            
                            # Track that this joint was updated
                            updated_joints.append(joint_name)
                    
                    # Mark this controller as having new commands
                    self._new_commands[controller_name] = True
                    
                    logger.info(f"Updated {len(updated_joints)} joints, ignored {len(ignored_joints)} joints")
                
                # Call any additional callbacks - FIXED to avoid calling RESET and R callbacks
                for callback_key, callback_func in self._additional_callbacks.items():
                    # Skip special callbacks that don't accept message arguments
                    if callback_key not in ["RESET", "R"]:
                        callback_func(msg, topic)
            else:
                logger.warning(f"Received empty trajectory message on {topic}")
        except Exception as e:
            logger.error(f"Error in joint trajectory callback for {topic}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def advance(self) -> Optional[Dict[str, float]]:
        """Process ROS messages and return the latest joint positions as a dictionary."""
        try:
            # For trajectory following mode, process active trajectories
            if self._trajectory_point == "follow" and self._current_trajectories:
                current_time = self.node.get_clock().now().to_msg()
                
                # Initialize the last advance time if not set
                if self._last_advance_time is None:
                    self._last_advance_time = current_time
                
                with self._joint_lock:
                    has_updates = False
                    controllers_to_remove = []
                    
                    # Process each active trajectory
                    for controller_name, trajectory_data in self._current_trajectories.items():
                        trajectory = trajectory_data['trajectory']
                        current_point_index = trajectory_data['current_point_index']
                        start_time = trajectory_data['start_time']
                        
                        # Skip if we've reached the end of this trajectory
                        if current_point_index >= len(trajectory.points):
                            controllers_to_remove.append(controller_name)
                            continue
                        
                        # Calculate elapsed time since trajectory start
                        elapsed_sec = (current_time.sec - start_time.sec) + ((current_time.nanosec - start_time.nanosec) / 1e9)
                        
                        # Find the appropriate point based on elapsed time
                        next_point_index = current_point_index
                        while next_point_index < len(trajectory.points):
                            point = trajectory.points[next_point_index]
                            point_time = point.time_from_start.sec + (point.time_from_start.nanosec / 1e9)
                            
                            if point_time <= elapsed_sec:
                                next_point_index += 1
                            else:
                                break
                        
                        # If we found a new point to execute (or reached the end)
                        if next_point_index > current_point_index:
                            # Use the previous point (the one we just passed in time)
                            point_to_execute = next_point_index - 1
                            
                            # Update the current point index
                            trajectory_data['current_point_index'] = next_point_index
                            
                            # Get joint names and positions
                            joint_names = trajectory.joint_names
                            positions = trajectory.points[point_to_execute].positions
                            
                            logger.info(f"Executing trajectory point {point_to_execute}/{len(trajectory.points)-1} for {controller_name}")
                            
                            # Determine which arm this controller is for
                            controller_prefix = controller_name.split('_')[0]  # e.g., 'la', 'ra', 'lh', 'rh'
                            
                            # Update joint positions
                            updated_joints = []
                            for i, joint_name in enumerate(joint_names):
                                if i < len(positions):
                                    # Only update joints that match the controller's prefix
                                    joint_prefix = joint_name.split('_')[0]
                                    
                                    if joint_prefix != controller_prefix:
                                        continue
                                    
                                    # Apply any necessary transformations
                                    position = positions[i]
                                    if joint_name in self._joint_transformations:
                                        position = self._joint_transformations[joint_name](position)
                                    
                                    # Store the position
                                    self._joint_positions[joint_name] = position
                                    self._joint_command_history[joint_name] = {
                                        'position': position,
                                        'controller': controller_name,
                                        'is_action': False,
                                        'timestamp': current_time,
                                        'trajectory_point': point_to_execute
                                    }
                                    
                                    updated_joints.append(joint_name)
                            
                            has_updates = True
                            
                            # If we've reached the last point, mark for removal
                            if next_point_index >= len(trajectory.points):
                                logger.info(f"Completed trajectory for {controller_name}")
                                controllers_to_remove.append(controller_name)
                    
                    # Remove completed trajectories
                    for controller in controllers_to_remove:
                        del self._current_trajectories[controller]
                    
                    # Update the last advance time
                    self._last_advance_time = current_time
                    
                    # Return joint positions if we have updates
                    if has_updates:
                        # Mark all controllers as having new commands to ensure the positions are sent
                        for controller in self.controller_names:
                            self._new_commands[controller] = True
                        return dict(self._joint_positions)
            
            # For non-follow modes or if no trajectory updates
            with self._joint_lock:
                # Check if we have any new commands
                if any(self._new_commands.values()):
                    # Reset the new command flags
                    for key in self._new_commands:
                        self._new_commands[key] = False
                    
                    # Return a copy of the current joint positions dictionary
                    return dict(self._joint_positions)
            
            return None
        except Exception as e:
            logger.error(f"Error in advance: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def reset(self):
        """Reset the device state."""
        logger.info("Resetting device state")
        with self._joint_lock:
            self._joint_positions = {}
            self._new_commands = defaultdict(bool)
            self._joint_command_history = {}
            self._current_joint_states = {}
            # Clear trajectory data
            self._current_trajectories = {}
            self._trajectory_start_times = {}
            self._last_advance_time = None

    def add_callback(self, key: str, func: Callable):
        """Add additional functions to bind to ROS messages."""
        logger.info(f"Adding callback for key: {key}")
        self._additional_callbacks[key] = func

    def __str__(self) -> str:
        """Returns a string representation of the interface."""
        controller_joint_info = []
        for controller, joints in self._controller_joint_map.items():
            controller_joint_info.append(f"  {controller}: {', '.join(joints)}")
        
        return f"""EmilyJointStateRosTopic:
    Node name: {self.node.get_name()}
    Subscribed topics: {', '.join(['/' + name + '/command' for name in self.controller_names])}
    Trajectory point mode: {self._trajectory_point}
    ROS context initialized: {rclpy.ok()}
    Controller-joint mappings:
{chr(10).join(controller_joint_info)}""" 