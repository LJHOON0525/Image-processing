
#원인이 무엇일지
#1.엔코더 처리 방식 : pos_estimate vs. count_in_cpr => 부동 소수점 오차 누적 문제 의심
#2. 데이터 동기화 문제 : 별도의 타이머에 의해 호출돼서 오도메트리 메시지 발행 
# -> 엔코더 데이터가 업데이트되어도 타이머가 실행될 때까지 기다려야 하므로, 위치 데이터가 실시간으로 반영되지 않고 지연이 발생합니다. 이는 SLAM 알고리즘이 로봇의 움직임을 정확하게 파악하는 것을 방해

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, TransformStamped
from std_msgs.msg import Float32MultiArray
import tf_transformations
import math
import tf2_ros

class OdomPublisher(Node):
    def __init__(self):
        super().__init__('odom_publisher')

        # Odometry 퍼블리셔
        self.publisher_ = self.create_publisher(Odometry, 'odom', 10)

        # 엔코더 데이터 구독자
        self.encoder_subscriber = self.create_subscription(
            Float32MultiArray, 
            '/encoder_data',
            self.encoder_callback,
            10
        )
        
        # 동적 TF 발행을 위한 TF2 broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # 초기 값들
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.prev_left_pulse = 0.0
        self.prev_right_pulse = 0.0

        # -------------- 차체 관련 파라미터 --------------
        # 실제 로봇 스펙에 맞게 정확히 측정하여 입력해야 합니다.
        self.wheel_radius = 0.1185  #2000 반지름 (m)
        self.CountsPerRevolution = 2000  # ODrive Incremental Encoder의 일반적인 CPR 값으로 가정 (정확한 값으로 수정 필요)
        self.wheel_base = 0.465  # 양쪽 바퀴 간 거리 (m)

        # base_link -> laser 간의 변환
        self.laser_x = 0.32
        self.laser_y = -0.185
        self.laser_z = 0.0
        self.laser_roll = 0.0
        self.laser_pitch = 0.0
        self.laser_yaw = 0.0
        
        # 속도 변수
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        
        # timestamp 저장용 변수
        self.last_time = self.get_clock().now()

        # 정적 TF는 초기화 시 단 한 번만 발행
        self.publish_static_tf()

    def encoder_callback(self, msg):
        left_pulse = msg.data[0]
        right_pulse = msg.data[1]

        # 첫 수신
        if not hasattr(self, 'initialized'):
            self.prev_left_pulse = left_pulse
            self.prev_right_pulse = right_pulse
            self.last_time = self.get_clock().now()
            self.initialized = True
            self.get_logger().info("Encoder initialized.")
            return

        # 시간 계산
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt < 1e-3:
            dt = 1e-3
        self.last_time = current_time

        # 펄스 변화량
        delta_left_pulse = left_pulse - self.prev_left_pulse
        delta_right_pulse = right_pulse - self.prev_right_pulse

        # 펄스 -> 거리
        delta_left = delta_left_pulse * (2 * math.pi * self.wheel_radius) / self.CountsPerRevolution
        delta_right = delta_right_pulse * (2 * math.pi * self.wheel_radius) / self.CountsPerRevolution

        # 1️⃣ delta 값 clamp
        max_delta = 0.05  # 한 스텝 최대 이동(m)
        delta_left = max(min(delta_left, max_delta), -max_delta)
        delta_right = max(min(delta_right, max_delta), -max_delta)

        # 2️⃣ 오른쪽 바퀴 안정화 보정
        Kp = 2.0
        delta_correction = Kp * (delta_left - delta_right)
        max_correction = 0.01
        delta_correction = max(min(delta_correction, max_correction), -max_correction)
        delta_right += delta_correction

        # 엔코더 갱신
        self.prev_left_pulse = left_pulse
        self.prev_right_pulse = right_pulse

        # 이동 거리 및 회전량
        delta_s = (delta_left + delta_right) / 2.0
        delta_theta = (delta_right - delta_left) / self.wheel_base

        # 속도 계산 (dt가 0에 가까운 경우를 대비한 예외 처리)
        if dt > 0:
            self.linear_velocity = delta_s / dt
            self.angular_velocity = delta_theta / dt
        else:
            self.linear_velocity = 0.0
            self.angular_velocity = 0.0

        # 위치 갱신
        self.x += delta_s * math.cos(self.theta + delta_theta / 2)
        self.y += delta_s * math.sin(self.theta + delta_theta / 2)
        self.theta += delta_theta

        # Odometry와 TF 메시지 발행
        self.publish_odometry_and_tf(current_time)

    def publish_odometry_and_tf(self, timestamp):
        now = timestamp.to_msg()

        # Odometry 메시지 생성
        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"   # 🔹 base_footprint → base_link

        odom_msg.pose.pose.position.x = self.x
        odom_msg.pose.pose.position.y = self.y

        q = tf_transformations.quaternion_from_euler(0, 0, self.theta)
        odom_msg.pose.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

        odom_msg.twist.twist.linear.x = self.linear_velocity
        odom_msg.twist.twist.angular.z = self.angular_velocity

        # Covariance 값 조정 (오도메트리 신뢰도를 현실적으로 조정)
        odom_msg.pose.covariance = [
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 99999.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 99999.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 99999.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.5
        ]
        odom_msg.twist.covariance = [
            0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 99999.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 99999.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 99999.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.5
        ]

        self.publisher_.publish(odom_msg)

        # TF 브로드캐스트
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'   # 🔹 base_footprint → base_link

        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

    def publish_static_tf(self):
        self.get_logger().info("✅ Static TF being published: base_link -> laser")  # 🔹 수정됨
        static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        
        static_transform = TransformStamped()
        
        static_transform.header.stamp = rclpy.time.Time().to_msg()
        static_transform.header.frame_id = 'base_link'   # 🔹 base_footprint → base_link
        static_transform.child_frame_id = 'laser'

        static_transform.transform.translation.x = self.laser_x
        static_transform.transform.translation.y = self.laser_y
        static_transform.transform.translation.z = self.laser_z

        q = tf_transformations.quaternion_from_euler(self.laser_roll, self.laser_pitch, self.laser_yaw)
        static_transform.transform.rotation.x = q[0]
        static_transform.transform.rotation.y = q[1]
        static_transform.transform.rotation.z = q[2]
        static_transform.transform.rotation.w = q[3]

        static_broadcaster.sendTransform(static_transform)

def main(args=None):
    rclpy.init(args=args)
    node = OdomPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
