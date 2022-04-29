import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from numpy import argmin

class VelocityController(Node):

    def __init__(self):
        super().__init__('velocity_controller')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.create_subscription(LaserScan, 'scan', self.laser_cb, rclpy.qos.qos_profile_sensor_data)
        self.create_timer(0.1, self.timer_cb)
        self.get_logger().info('controller node started')

        self.target_distance = 0.2
        self.dist_tolerance = 0.1
        self.angle_tolerance = 15

        self.forward_distance = 0.0
        self.shortest_direction = 0
        self.shortest_distance = 0.0

        self.step = 0
        self.start_wait = 100
        self.inc_wait = 3750
        self.inc = 0.2
        
    def timer_cb(self):
        self.step += 1
        msg = Twist()

        if self.step < self.start_wait:
            # delays start of bot to let sensors calibrate
            # not sure if this is only relevant to the simulation
            self.get_logger().info("###### WAITING #####")
            self.publisher.publish(msg)
            return
        elif self.step % self.inc_wait == 0:
            # increase target distance after a while,
            # inc_wait is set so that the bot can finish a round before first increase
            self.get_logger().info("#" * 30)
            self.get_logger().info("|"*10 + " INCREASING " + "|"*10)
            self.get_logger().info("#" * 30)
            self.target_distance += self.inc
            # time per round decreases as the bot draws smaller and smaller circles
            self.inc_wait -= 300
        
        if self.wall_is_left():
            # drive forward
            msg.linear.x = 0.1
        else:
            # correct angle
            msg.angular.z = self.turn_val()
        
        self.publisher.publish(msg) 

    def in_dist(self):
        return self.target_distance < self.shortest_direction < self.target_distance + self.dist_tolerance

    # determines if the shortest distance is on the left side of the bot (90°)
    # acceptance skewed so that the bot self correct towards target distance.
    # and changes to steeper angles, once the bot leaves the tolerance zone.
    def wall_is_left(self):
        if self.shortest_distance < self.target_distance:
            if self.shortest_distance < self.target_distance - self.dist_tolerance:
                return 120+self.angle_tolerance*2 >= self.shortest_direction >= 120
            else:
                return 90+self.angle_tolerance >= self.shortest_direction >= 90
        else:
            if self.shortest_distance > self.target_distance + self.dist_tolerance:
                return 60 >= self.shortest_direction >= 60-self.angle_tolerance*2
            else:
                return 90 >= self.shortest_direction >= 90-self.angle_tolerance

    # helper function for turn_val
    # determines which angle left wall should be
    def target_angle(self):
        if self.shortest_distance < self.target_distance - self.dist_tolerance:
            return 120
        elif self.shortest_distance > self.target_distance + self.dist_tolerance:
            return 60
        else:
            return 90

    # returns turn direction for fastest possible turn
    # TODO: I think this might be wrong and just works by accident
    def turn_val(self):
        a = self.target_angle()
        b = (a + 180) % 360
        low = min(a,b)
        high = max(a,b)
        if low < self.shortest_direction < high:
            return 0.2
        else:
            return -0.2
    
    def laser_cb(self, msg):
        forward_ranges = msg.ranges[0:FRONT_RANGE] + msg.ranges[-FRONT_RANGE:]
        self.forward_distance = min(forward_ranges)
        self.shortest_direction = argmin(msg.ranges)
        self.shortest_distance = msg.ranges[self.shortest_direction]
        

FRONT_RANGE = 15

def main(args=None):
    rclpy.init(args=args)

    node = VelocityController()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
