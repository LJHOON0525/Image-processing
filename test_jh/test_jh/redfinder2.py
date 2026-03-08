#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from geometry_msgs.msg import TransformStamped, Quaternion, Twist
from nav_msgs.msg import Odometry
import tf_transformations
import tf2_ros
import math
from std_msgs.msg import Float32MultiArray

class FakeOdometryNode(Node):
    def __init__(self):
        super().__init__('fake_odometry_node')

        qos_profile = QoSProfile(depth=10)

        # Nav2 제어 명령 구독 (/cmd_vel)
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self.on_cmd_vel, qos_profile
        )

        # (옵션) 조이스틱 직접 제어도 허용
        self.joy_sub = self.create_subscription(
            Float32MultiArray, 'joycmd', self.on_joycmd, qos_profile
        )
        self.use_nav2 = True  # Nav2 제어 우선

        # 퍼블리셔
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos_profile)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Timer
        self.create_timer(0.1, self.publish_odometry)

        # 내부 상태
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.omega = 0.0
        self.wheel_distance = 0.3

        self.get_logger().info("✅ FakeOdometryNode started (listening to /cmd_vel).")

    # Nav2 속도 입력
    def on_cmd_vel(self, msg: Twist):
        if self.use_nav2:
            self.v = msg.linear.x
            self.omega = msg.angular.z

    # 조이스틱 입력
    def on_joycmd(self, msg: Float32MultiArray):
        if not self.use_nav2 and len(msg.data) >= 2:
            v_l, v_r = msg.data[0], msg.data[1]
            self.v = (v_r + v_l) / 2.0
            self.omega = (v_r - v_l) / self.wheel_distance

    def publish_odometry(self):
        dt = 0.1
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.omega * dt

        # Odometry 메시지
        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y

        q = tf_transformations.quaternion_from_euler(0, 0, self.yaw)
        odom.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.omega

        self.odom_pub.publish(odom)

        # TF (odom → base_link)
        t = TransformStamped()
        t.header.stamp = odom.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

        self.get_logger().info(
            f"Publishing Odom: x={self.x:.2f}, y={self.y:.2f}, yaw={math.degrees(self.yaw):.1f}°"
        )

def main(args=None):
    rclpy.init(args=args)
    node = FakeOdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()