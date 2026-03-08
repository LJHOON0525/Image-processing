import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray
import pyrealsense2 as rs
import numpy as np
import math
import cv2


class CameraPoseCirculate(Node):
    def __init__(self):
        super().__init__('imu_node')
        qos_profile = QoSProfile(depth=2)
        self.camera_pose = self.create_publisher(
            Float32MultiArray, 
            'imu_data', 
            qos_profile)
        
        self.sample_freq = 1/15
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.accel)
        self.pipeline.start(self.config)
                
        self.timer = self.create_timer(self.sample_freq, self.circulate_pose)
        
    def circulate_pose(self) :
        msg = Float32MultiArray()
        
        frames = self.pipeline.wait_for_frames()
        accel_frame = frames[0].as_motion_frame()
        
        if accel_frame:
            accel_datas = accel_frame.get_motion_data()
            accel_data = [accel_datas.x, accel_datas.y, accel_datas.z]
            
            norm = math.sqrt(accel_data[0]**2 + accel_data[1]**2 + accel_data[2]**2)
            if norm == 0:
                return

            x_angle = np.arccos(accel_data[0]/norm) / np.pi * 180 #이게
            y_angle = np.arccos(-accel_data[1]/norm) / np.pi * 180 #이게
            z_angle = np.arccos(accel_data[2]/norm) / np.pi * 180 #이게 나타낸건데

            msg.data = [x_angle, y_angle, z_angle]
            self.get_logger().info(f'{x_angle:.2f}  {y_angle:.2f}  {z_angle:.2f}')
            self.camera_pose.publish(msg)

            # HUD 표시
            self.display_hud(x_angle, y_angle, z_angle)

    def display_hud(self, x_angle, y_angle, z_angle):
        # 빈 이미지 생성
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        center = (320, 240)

        # 텍스트로 각도 표시
        cv2.putText(image, f"X Angle: {x_angle:.2f}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(image, f"Y Angle: {y_angle:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, f"Z Angle: {z_angle:.2f}", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 선형 시각화 (예시: 길이로 표현)
        cv2.line(image, center, (center[0] + int((x_angle - 90) * 2), center[1] - 60), (0, 0, 255), 3)
        cv2.line(image, center, (center[0] + int((y_angle - 90) * 2), center[1]), (0, 255, 0), 3)
        cv2.line(image, center, (center[0] + int((z_angle - 90) * 2), center[1] + 60), (255, 0, 0), 3)

        # 중심 마커
        cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 10, 2)

        cv2.imshow("IMU HUD", image)
        cv2.waitKey(1)

    def destroy_node(self):
        super().destroy_node()
        self.pipeline.stop()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = CameraPoseCirculate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
