from enum import Enum, auto
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from sensor_msgs.msg import LaserScan


def cos_deg(a, b):
    return np.arccos( (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) )

def calc_rotation_params(speed, theta):
    time = round(theta / speed)
    speed = theta / time 
    return time, speed

def val_in_range(target_value, test_value, tolerance):
    if test_value > (target_value + tolerance):
        return False
    if test_value < (target_value - tolerance):
        return False
    return True


FRONT_RANGE = 15
MAX_POSITION_CACHE = 10

DEFAULT_ROTATION_SPEED = np.pi/8
MIN_SPEED = 0.05
MAX_SPEED = 0.1
SPEED_INCREASE = 0.01

TURN_TOLERANCE = 0.15 # 0.15rad ~ 8.5deg
GOAL_POSITION_TOLERANCE = 0.2

class RobotState(Enum):
    STARTUP = auto()
    SLEEPING = auto()
    ROTATING = auto()
    DRIVING = auto()
    WAITING_ON_GOAL = auto()


class VelocityController(Node):

    def __init__(self):
        super().__init__('velocity_controller')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        
        self.state = None
        self.state_timer = 0
        self.startup()

        self.goal = None
        self.positions = []

        self.forward_speed = MIN_SPEED
        self.rotation_speed = 0
        self.rotation_timer = None

        self.forward_distance = 0
        self.shortest_direction = 0
        self.shortest_distance = 0

        self.create_subscription(LaserScan, 'scan', self.laser_cb, rclpy.qos.qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'nav/goal', self.goal_cb, 10)
        self.create_subscription(PointStamped, 'position', self.position_cb, 10)
        self.create_timer(0.1, self.timer_cb)
        self.get_logger().info('controller node started')

    def startup(self):
        self.forward_speed = MIN_SPEED
        self.state_timer = 10
        self.state = RobotState.STARTUP

    def sleep(self, next_state: RobotState, timer = 5):
        self.post_sleep_state = next_state
        self.state_timer = timer
        self.state = RobotState.SLEEPING


    def timer_cb(self):
        self.get_logger().debug(f"[State|StateInfo] Current state: '{self.state}' with timer '{self.state_timer}'")
        if self.state_timer > 0:
            self.state_timer -= 1
        
        if self.state == RobotState.STARTUP:
            if self.state_timer == 0:
                next_state = RobotState.DRIVING
                self.get_logger().info(f"[State|Startup] Switching to '{next_state} state")
                self.state = next_state
            else:
                speed = MIN_SPEED/2
                self.get_logger().info(f"[State|Startup] Moving with speed '{speed}'")
                self.drive(linear=MIN_SPEED/2)

        elif self.state == RobotState.ROTATING:
            return

        elif self.state == RobotState.SLEEPING:
            if self.state_timer == 0:
                self.get_logger().info(f"[State|Sleeping] Switching to '{self.post_sleep_state}'")
                self.state = self.post_sleep_state
                self.positions = []
            else:
                self.drive(0., 0.)
            return

        elif self.state == RobotState.WAITING_ON_GOAL:
            self.drive(0., 0.)
            return

        elif self.state == RobotState.DRIVING:
            # check if goal is reached
            if self.has_reached_goal():
                self.get_logger().info("Goal reached!")
                self.forward_speed = MIN_SPEED
                self.state = RobotState.WAITING_ON_GOAL
                return
            
            if len(self.positions) >= MAX_POSITION_CACHE:
                rotation_angle = self.calc_goal_rotation_angle()
                if np.abs(rotation_angle) > TURN_TOLERANCE:
                    self.get_logger().info(f"Rotation by {rotation_angle}rad required")
                    self.rotate(rotation_angle)
                    return

                # increase forward speed if position cache is full
                elif self.forward_speed < MAX_SPEED:
                    self.forward_speed += SPEED_INCREASE

            self.drive(linear=self.forward_speed)

        return

        # check whether in rotation state
        if self.rotating:
            if self.rotation_time > 0:
                self.rotation_time -= 1
                self.drive(angular=self.rotation_speed)
                return
            else:
                self.rotating = False
                self.positions = []

    def calc_goal_rotation_angle(self):
        vec_to_pos = self.positions[-1] - self.positions[0]
        vec_pos_to_goal = self.goal - self.positions[-1]
        
        cross_z = np.cross(vec_pos_to_goal, vec_to_pos)[2]
        turn_angle = cos_deg(vec_pos_to_goal, vec_to_pos)

        return turn_angle if cross_z < 0 else -turn_angle


    def rotate(self, angle, speed = DEFAULT_ROTATION_SPEED):
        self.state = RobotState.ROTATING
        time = np.abs(angle)/speed
        self.drive(angular=speed if angle > 0 else -speed)
        self.rotation_timer = self.create_timer(time, self.end_rotation)
        self.positions = []

    def end_rotation(self):
        self.drive()
        if not self.rotation_timer.is_canceled():
            self.rotation_timer.cancel()
        self.sleep(RobotState.DRIVING, 3)

    def drive(self, linear = 0.0, angular = 0.0):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)

    def has_reached_goal(self):
        if self.goal is None or len(self.positions) == 0:
            return False

        if not val_in_range(self.goal[0], self.positions[-1][0], GOAL_POSITION_TOLERANCE):
            return False
        if not val_in_range(self.goal[1], self.positions[-1][1], GOAL_POSITION_TOLERANCE):
            return False
        return True


    def goal_cb(self, msg):
        goal = np.array((msg.pose.position.x, msg.pose.position.y, 0))
        if any(self.goal != goal):
            self.get_logger().info(f'received a new goal: (x={goal[0]}, y={goal[1]})')
            self.goal = goal
        
            # restart robot driving loop
            if self.state == RobotState.WAITING_ON_GOAL:
                self.state = RobotState.STARTUP

    def laser_cb(self, msg):
        forward_ranges = msg.ranges[0:FRONT_RANGE] + msg.ranges[-FRONT_RANGE:]
        self.forward_distance = min(forward_ranges)
        self.shortest_direction = np.argmin(msg.ranges)
        self.shortest_distance = msg.ranges[self.shortest_direction]

    def position_cb(self, msg):
        self.positions.append(np.array([msg.point.x, msg.point.y, 0]))
        self.positions = self.positions[-MAX_POSITION_CACHE:]


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
