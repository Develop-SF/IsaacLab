# Copyright (c) 2025, ShennongShi
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gamepad joy topic to twist topic ROS interface."""
"""Run the command 'ros2 run joy joy_node' first."""

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
import subprocess
import time

class XboxController(Node):
    def __init__(self):
        super().__init__('xbox_controller')
        
        # Create callback groups
        self.cb_group = MutuallyExclusiveCallbackGroup()
        
        # Initialize variables
        self.teleop_enabled = False
        self.servo_start = True
        self.robot_parts_commands = [[0.0] * 6]  # 6 zeros for twist commands
        self.scale_factor = 0.5 # to make the movement less sensitive
        
        # Create subscription with callback group
        self.joy_sub = self.create_subscription(
            Joy,
            'joy',  # Replace with your joy topic
            self.joy_callback,
            1,
            callback_group=self.cb_group
        )
        
        # Create publisher
        self.command_publisher = self.create_publisher(
            TwistStamped,
            'twist_cmd',  # Replace with your twist topic
            1
        )
        
        # Create timer for publishing commands
        self.timer = self.create_timer(
            1.0 / 100.0,  # Adjust rate as needed
            self.publish_command,
            callback_group=self.cb_group
        )

        # print the instructions
        self.get_logger().warn(self.__str__())
    
    def __str__(self) -> str:
        """Print the controller mapping"""
        msg = f"\n\nJoy to End Effector Twist: {self.__class__.__name__}\n"
        msg += "\t----------------------------------------------\n"
        msg += "\tLB\t\t: Trigger Motion(hold)\n"
        msg += "\tLT\t\t: Down\n"
        msg += "\tRT\t\t: Up\n"
        msg += "\tL Thumbstick\t: ↑/↓ & ←/→\n"
        msg += "\tR Thumbstick\t: Roll/Pitch\n"
        msg += "\tDPAD Left/Right\t: Yaw\n"
        return msg

    def joy_callback(self, msg):
        # self.get_logger().info(f'joy axes: {[f"{x:.1f}" for x in msg.axes]}', throttle_duration_sec=1.0)
        # self.get_logger().info(f'joy buttons: {[f"{x}" for x in msg.buttons]}', throttle_duration_sec=1.0)
        # Assuming standard Xbox controller button mapping
        trigger_button = 4  # Left bumper, adjust as needed
        
        if msg.buttons[trigger_button] and self.servo_start:
            self.teleop_enabled = True
            # Linear velocities - planner
            self.robot_parts_commands[0][0] = msg.axes[1] * self.scale_factor  # x movement
            self.robot_parts_commands[0][1] = msg.axes[0] * self.scale_factor  # y movement
            # Linear velocities - vertical
            # if both 2 and 5 are pressed(greater than 0.2), then linear z is 0
            if msg.axes[2] < 0.9 and msg.axes[5] < 0.9:
                self.get_logger().warn('Both up and down are pressed, linear z is 0')
                self.robot_parts_commands[0][2] = 0.0
            else:
                # the default value of 2 and 5 are 1.0, and range is 1 to -1
                # normalize the value to 0 to 1
                z_up = -(msg.axes[5] - 1.0) / 2.0
                z_down = (msg.axes[2] - 1.0) / 2.0
                self.robot_parts_commands[0][2] = (z_up + z_down) * self.scale_factor  # linear z
            
            # Angular velocities
            self.robot_parts_commands[0][3] = -msg.axes[3] * self.scale_factor  # roll
            self.robot_parts_commands[0][4] = msg.axes[4] * self.scale_factor  # pitch
            # Yaw control from axis 6
            self.robot_parts_commands[0][5] = msg.axes[6] * self.scale_factor  # yaw
        else:
            self.teleop_enabled = False
            # Reset all commands to zero
            self.robot_parts_commands[0] = [0.0] * 6

    def publish_command(self):
        if self.teleop_enabled:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            
            # Set linear velocities
            msg.twist.linear.x = self.robot_parts_commands[0][0]
            msg.twist.linear.y = self.robot_parts_commands[0][1]
            msg.twist.linear.z = self.robot_parts_commands[0][2]
            
            # Set angular velocities
            msg.twist.angular.x = self.robot_parts_commands[0][3]
            msg.twist.angular.y = self.robot_parts_commands[0][4]
            msg.twist.angular.z = self.robot_parts_commands[0][5]
            
            self.command_publisher.publish(msg)

def main(args=None):
    # Start joy_node in a separate process
    try:
        joy_process = subprocess.Popen(['ros2', 'run', 'joy', 'joy_node'])
        # Wait a bit to ensure joy_node is running
        time.sleep(2)
        
        # Initialize and run our node
        rclpy.init(args=args)
        xbox_controller = XboxController()
        try:
            rclpy.spin(xbox_controller)
        finally:
            xbox_controller.destroy_node()
            rclpy.shutdown()
            # Cleanup: terminate joy_node process
            joy_process.terminate()
            joy_process.wait()
    except Exception as e:
        print(f"Error occurred: {e}")
        if 'joy_process' in locals():
            joy_process.terminate()
            joy_process.wait()

if __name__ == '__main__':
    main()
