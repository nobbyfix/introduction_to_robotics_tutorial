from ctypes import Union
from enum import Enum, auto
from typing import Callable
import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from sensor_msgs.msg import LaserScan


STATE_INTERVAL = 10

FRONT_RANGE = 15
MAX_POSITION_CACHE = 10

DEFAULT_ROTATION_SPEED = np.pi/8
FORWARD_SPEED_MIN = 0.05
FORWARD_SPEED_MAX = 0.1
SPEED_INCREASE = 0.01

TURN_TOLERANCE = 0.15 # 0.15rad ~ 8.5deg


def cos_deg(a, b):
    return np.arccos( (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)) )

def state_interval(seconds: int):
    interval = seconds * STATE_INTERVAL
    if interval < 1:
        return 1
    return round(interval)


class RobotState(Enum):
    STARTUP = auto()
    SLEEPING = auto()
    ROTATING = auto()
    DRIVING = auto()


class VelocityController(Node):

    def __init__(self):
        super().__init__('velocity_controller')
        self.publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.goal = None
        self.positions = []

        self.forward_speed = 0
        self.forward_distance = 0
        self.shortest_direction = 0
        self.shortest_distance = 0

        self.state = None
        self.state_timer = 0
        self.startup()

        self.create_subscription(LaserScan, 'scan', self.laser_cb, rclpy.qos.qos_profile_sensor_data)
        self.create_subscription(PoseStamped, 'nav/goal', self.goal_cb, 10)
        self.create_subscription(PointStamped, 'position', self.position_cb, 10)
        self.create_timer(1/STATE_INTERVAL, self.timer_cb)
        self.get_logger().info('controller node started')


    def startup(self):
        self.get_logger().info("[Startup] Entered 'startup' state")
        # reset speed to minimum if switching from max speed after reaching goal
        self.forward_speed = FORWARD_SPEED_MIN
        self.state_timer = state_interval(1)
        self.state = RobotState.STARTUP

    def sleep(self, state_change: Callable, timer: int = 0.5):
        """
        Stops the robot for *timer* seconds.
        """
        self.get_logger().info("[Sleeping] Entered 'sleeping' state")
        self.post_sleep_state = state_change
        self.state_timer = state_interval(timer)
        self.state = RobotState.SLEEPING
        self.get_logger().info(f"[Sleeping] Sleeping for '{timer}s', interval '{self.state_timer}'")

    def driving(self):
        self.get_logger().info("[Driving] Entered 'driving' state")
        # clear position cache after changing direction or standing
        self.positions = []
        self.state = RobotState.DRIVING

    def rotate(self, angle: float, speed: float = DEFAULT_ROTATION_SPEED):
        """
        Creates a timer to make the robot rotate the specified *angle* in constant *speed*.
        """
        self.get_logger().info("[Rotation] Entered 'rotation' state")
        time = np.abs(angle)/speed
        self.drive(angular=(speed if angle > 0 else -speed))
        self.rotation_timer = self.create_timer(time, self.end_rotation)
        self.state = RobotState.ROTATING
        self.get_logger().info(f"[Rotation] Rotating for '{time}s' with speed '{speed}rad'")

    def end_rotation(self):
        # explicitly stop movement because the state timer will execute sleeping state a bit later,
        # causing a further rotation
        self.drive(0., 0.)

        # stop the rotation timer, it is only supposed to execute a single time
        if not self.rotation_timer.is_canceled():
            self.rotation_timer.cancel()

        # delay the start of driving by half a second 
        self.sleep(self.driving, 0.5)


    def timer_cb(self):
        self.get_logger().debug(f"[StateInfo] Current state: '{self.state}' with timer '{self.state_timer}'")
        
        # reduce state timer on every state execution
        # reaching 0 (and other values) is handled by the state itself
        if self.state_timer > 0:
            self.state_timer -= 1

        if self.state == RobotState.STARTUP:
            if self.state_timer == 0:
                # start normal driving as default state after startup
                self.get_logger().info(f"[Startup] Switching to 'driving' state")
                self.driving()
            else:
                # drive with half speed to help prevent driving into walls
                self.drive(linear=FORWARD_SPEED_MIN/2)
            return

        elif self.state == RobotState.ROTATING:
            # rotation is handled with a seperate timer, nothing to do here
            return

        elif self.state == RobotState.SLEEPING:
            if self.state_timer == 0:
                self.get_logger().info(f"[Sleeping] Switching to '{self.post_sleep_state.__name__}' state")
                self.post_sleep_state()
            else:
                # send drive message with 0 values to make sure the robot is staying still while sleeping
                self.drive(0., 0.)
            return

        elif self.state == RobotState.DRIVING:
            if len(self.positions) >= MAX_POSITION_CACHE:
                # check if correction of driving direction is neccessary
                rotation_angle = self.calc_goal_rotation_angle()
                if np.abs(rotation_angle) > TURN_TOLERANCE:
                    self.get_logger().warn(f"[Driving] Rotation by '{rotation_angle}rad' required")
                    # sleep before switching to rotation state
                    self.sleep(lambda: self.rotate(rotation_angle), 0.5)
                    return

                # increase forward speed if position cache is full
                elif self.forward_speed < FORWARD_SPEED_MAX:
                    self.forward_speed += SPEED_INCREASE
                    self.get_logger().info(f"[Driving] Increased speed to '{self.forward_distance}'")

            # keep driving forward in driving state
            self.drive(linear=self.forward_speed)
            return

    def drive(self, linear: float = 0., angular: float = 0.):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self.publisher.publish(msg)

    def calc_goal_rotation_angle(self):
        vec_to_pos = self.positions[-1] - self.positions[0]
        vec_pos_to_goal = self.goal - self.positions[-1]
        
        cross_z = np.cross(vec_pos_to_goal, vec_to_pos)[2]
        turn_angle = cos_deg(vec_pos_to_goal, vec_to_pos)

        return turn_angle if cross_z < 0 else -turn_angle

    def goal_cb(self, msg):
        goal = np.array((msg.pose.position.x, msg.pose.position.y, 0))
        if any(self.goal != goal):
            self.get_logger().info(f'[GoalCB] Received a new goal: (x={goal[0]}, y={goal[1]})')
            self.goal = goal
            self.startup()

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
