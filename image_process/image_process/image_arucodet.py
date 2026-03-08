import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        
        # ROS2에서 이미지 구독
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.image_callback,
            10)
        
        # CvBridge 객체
        self.cv_bridge = CvBridge()
        
        # 아루코 마커 사전 생성 (4x4 50개 마커)
        self.aruco_dict = cv2.aruco.Dictionary(cv2.aruco.DICT_4X4_50,5)  # 뒤의 5는 마커 사이즈
        self.parameters = cv2.aruco.DetectorParameters()  # 카메라 파라미터 조정
        
        # 결과 이미지를 퍼블리시할 퍼블리셔
        self.publisher = self.create_publisher(Image, 'image_aruco', 10)

    def image_callback(self, msg):
        # ROS 메시지를 OpenCV 이미지로 변환
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # 그레이스케일로 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 아루코 마커 감지
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if len(corners) > 0:
            # 감지된 마커에 사각형 그리기
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # 각 마커의 ID와 위치 출력
            for i, corner in zip(ids, corners):
                self.get_logger().info(f"Marker ID: {i[0]}")
                self.get_logger().info(f"Marker Corners: {corner}")
        
        # 결과 이미지를 퍼블리시
        image_msg = self.cv_bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.publisher.publish(image_msg)
        
        # 결과를 화면에 출력 (테스트용으로 추가, 실제 실행시 제거 가능)
        cv2.imshow('Aruco Marker Detection', frame)
        cv2.waitKey(1)

    def destroy_node(self):
        # 노드 종료 시 윈도우 자원 해제
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
