import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2.aruco as aruco

class ArucoDetection(Node):
    def __init__(self):
        super().__init__('aruco_detection')
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.publisher = self.create_publisher(Image, 'side_camera', qos_profile)

        self.img_size_x = 640
        self.img_size_y = 480
        self.frame_rate = 10

        # ArUco 사전 생성 (2개의 마커)
        self.aruco_dict = aruco.custom_dictionary(2, 5)
        self.aruco_dict.bytesList = np.empty(shape=(2, 4, 4), dtype=np.uint8)

        # START 마커 (5x5 비트 패턴 예시)
        start_bits = [
            [1,0,1,0,1],
            [1,1,0,1,1],
            [1,0,1,0,1],
            [1,1,0,1,1],
            [1,0,1,0,1]
        ]

        # FINISH 마커 (5x5 비트 패턴 예시)
        finish_bits = [
            [1,1,1,1,1],
            [1,0,1,0,1],
            [1,1,1,1,1],
            [1,0,1,0,1],
            [1,0,1,0,1]
        ]

        self.markers = [start_bits, finish_bits]
        self.marker_labels = ["START", "FINISH"]

        # 바이트 리스트로 변환
        for i, bits in enumerate(self.markers):
            mybits = np.array(bits, dtype=np.uint8)
            self.aruco_dict.bytesList[i] = aruco.Dictionary_getByteListFromBits(mybits)

        self.aruco_param = aruco.DetectorParameters_create()

        # 카메라 2번만 사용
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)

        self.timer_finder = self.create_timer(1/self.frame_rate, self.image_callback)

    def image_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Camera cannot connect")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_param)
        frame = aruco.drawDetectedMarkers(frame, corners, ids)

        if ids is not None:
            for i, marker_id in enumerate(ids):
                label = self.marker_labels[marker_id[0]]
                x_min, y_min = map(int, np.min(corners[i][0], axis=0))
                cv2.putText(frame, label, (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if label == "START":
                    print("🚀 START detected!")
                elif label == "FINISH":
                    print("🏁 FINISH detected!")

        self.p
