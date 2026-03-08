import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoMarkerPublisher(Node):
    def __init__(self):
        super().__init__('aruco_marker_publisher')
        
        self.publisher = self.create_publisher(Image, 'image_raw', 10)
        self.cv_bridge = CvBridge()
        
        # 아루코 마커 사전 생성 (4x4 50개 마커)
        self.aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50,200)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # 마커 크기 설정
        self.marker_size = 0.05  # 예시: 5cm 크기의 마커

        # 10개의 아루코 마커 생성
        self.generate_aruco_markers()

        # 타이머로 주기적으로 퍼블리시
        self.timer = self.create_timer(1.0, self.publish_marker)

        self.marker_id = 0  # 처음에는 첫 번째 마커부터 퍼블리시

    def generate_aruco_markers(self):
        # 10개의 아루코 마커 생성 후 리스트에 저장
        self.marker_images = []
        for marker_id in range(10):
            marker_image = np.zeros((500, 500), dtype=np.uint8)
            # `cv2.aruco.drawMarker` 대신 `cv2.drawMarker` 사용
            cv2.aruco.drawMarker(self.aruco_dict, marker_id, 500, marker_image)
            self.marker_images.append(marker_image)

    def publish_marker(self):
        if self.marker_id < 10:
            # 현재 마커 이미지를 퍼블리시
            marker_image = self.marker_images[self.marker_id]
            ros_image = self.cv_bridge.cv2_to_imgmsg(marker_image, encoding="mono8")
            self.publisher.publish(ros_image)
            self.get_logger().info(f"Publishing ArUco Marker ID: {self.marker_id}")
            
            # 다음 마커로 이동
            self.marker_id += 1
        else:
            self.get_logger().info("All markers have been published.")
            self.destroy_timer(self.timer)  # 마커 10개를 모두 퍼블리시한 후 타이머 종료

def main(args=None):
    rclpy.init(args=args)
    node = ArucoMarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
