import rclpy
import numpy as np
from rclpy.node import Node

from driving_swarm_messages.msg import Range
from geometry_msgs.msg import PointStamped


class LocatorNode(Node):

    def __init__(self):
        super().__init__('locator_node')
        self.anchor_ranges = []
        self.create_subscription(Range, 'range', self.range_cb, 10)
        self.position_pub = self.create_publisher(PointStamped, 'position', 10)
        self.initialized = False
        self.create_timer(1.0, self.timer_cb)
        self.get_logger().info('locator node started')

        self.last_position = np.array([0.0, 0.0, 0.0])
        
    def range_cb(self, msg):
        self.anchor_ranges.append(msg)
        self.anchor_ranges = self.anchor_ranges[-10:]
        if not self.initialized:
            self.initialized = True
            self.get_logger().info('first range received')

    def timer_cb(self):
        if not self.initialized:
            return
        msg = PointStamped()
        msg.point.x, msg.point.y, msg.point.z = self.calculate_position()
        msg.header.frame_id = 'world'
        self.position_pub.publish(msg)
    
    def calculate_position(self):
        if not len(self.anchor_ranges):
            return 0.0, 0.0, 0.0

        # YOUR CODE GOES HERE:
        x_guess = self.last_position
        for i in range(10):
            gradiants = []
            errs = []
            for a in self.anchor_ranges:
                diff = x_guess - np.array([a.anchor.x, a.anchor.y, a.anchor.z])
                errs.append(a.range - np.linalg.norm(diff))
                gradiants.append(-diff / np.linalg.norm(diff))
            R = np.array(errs)
            delta_R = np.array(gradiants)
            x_guess -= np.linalg.pinv(delta_R) @ R
            self.get_logger().info("-!"*15)
            self.get_logger().info("R:" + str(R))
            self.get_logger().info("delta_R:" + str(delta_R))
            if max(R) < 0.001:
                break

        self.get_logger().info(str(x_guess))
        
        self.last_position = x_guess
        return x_guess[0], x_guess[1], x_guess[2] 


def main(args=None):
    rclpy.init(args=args)

    node = LocatorNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
