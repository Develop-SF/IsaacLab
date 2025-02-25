import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_publisher')
        # Create a publisher for JointState messages on the 'joint_states' topic
        self.publisher_ = self.create_publisher(JointState, 'isaac_joint_commands', 10)
        # Create a timer that triggers the callback every 1.0 second
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('JointStatePublisher node started.')

    def timer_callback(self):
        msg = JointState()
        # Set header with current time
        msg.header.stamp = self.get_clock().now().to_msg()
        # Define joint names and dummy positions for example purposes
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7','panda_finger_joint1', 'panda_finger_joint2']
        msg.position = [0.05, -0.3, 0.0, -1.3, 0.0, 0.5, 0.05, 0.0, 0.0]
        # Optionally, you can include velocity and effort if needed
        msg.velocity = []
        msg.effort = []
        # Publish the message
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing JointState message')

def main(args=None):
    rclpy.init(args=args)
    joint_state_publisher = JointStatePublisher()
    try:
        rclpy.spin(joint_state_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        joint_state_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
