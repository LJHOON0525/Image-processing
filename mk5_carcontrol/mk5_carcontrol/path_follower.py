import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import numpy as np
import math

class DynamicPathFollower(Node):
    def __init__(self):
        super().__init__('dynamic_path_follower')
        self.get_logger().info("Dynamic Path Follower Started")

        # 라이다 구독
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        # 계획된 경로 구독
        self.path_sub = self.create_subscription(Float32MultiArray, '/planned_path', self.path_callback, 10)
        # cmd_vel publish
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.path = []
        self.current_index = 0
        self.min_obstacle_dist = 0.3  # m, 장애물 최소 거리

        self.latest_scan = None

    def path_callback(self, msg):
        coords = msg.data
        self.path = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        self.current_index = 0

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.follow_path()

    def follow_path(self):
        if not self.path or not self.latest_scan:
            return

        twist = Twist()

        # 가장 가까운 장애물 확인
        ranges = np.array(self.latest_scan.ranges)
        ranges = ranges[np.isfinite(ranges)]  # inf 제거
        min_dist = np.min(ranges) if len(ranges) > 0 else np.inf

        if min_dist < self.min_obstacle_dist:
            # 장애물이 너무 가까우면 회전
            twist.linear.x = 0.0
            twist.angular.z = 0.5  # 시계 방향 회전
        else:
            # 다음 waypoint 향해 이동
            if self.current_index < len(self.path):
                x, y = self.path[self.current_index]
                angle = math.atan2(y, x)
                twist.linear.x = 0.2
                twist.angular.z = angle
                self.current_index += 1

        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = DynamicPathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
