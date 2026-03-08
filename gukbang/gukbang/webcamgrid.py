import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2.aruco as aruco
from std_msgs.msg import String  # 로그용 퍼블리시 (선택 사항)


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
        # 마커 로그를 퍼블리시할 토픽 생성 (선택)
        self.log_pub = self.create_publisher(String, 'aruco_log', qos_profile)

        self.img_size_x = 640
        self.img_size_y = 480
        self.frame_rate = 10
        self.threshold = 120

        self.aruco_dict = aruco.custom_dictionary(9, 5)
        self.aruco_dict.bytesList = np.empty(shape=(9, 4, 4), dtype=np.uint8)

        self.markers = [
            [[1,0,0,0,1],[1,0,0,1,0],[1,1,1,0,0],[1,0,0,1,0],[1,0,0,0,1]], #K
            [[0,1,1,1,0],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[0,1,1,1,0]], #O
            [[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,1,0],[1,0,0,0,1]], #R
            [[1,1,1,1,1],[1,0,0,0,0],[1,1,1,1,0],[1,0,0,0,0],[1,1,1,1,1]], #E
            [[0,0,1,0,0],[0,1,0,1,0],[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1]], #A
            [[1,1,1,1,0],[1,0,0,0,1],[1,1,1,1,0],[1,0,0,1,0],[1,0,0,0,1]], #R
            [[1,0,0,0,1],[1,1,0,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,0,0,0,1]], #M
            [[1,0,0,0,1],[0,1,0,1,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,1,0,0]], #Y
            [[0,1,0,1,0],[1,1,1,1,1],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]] #♥
        ]

        for i, bits in enumerate(self.markers):
            mybits = np.array(bits, dtype=np.uint8)
            self.aruco_dict.bytesList[i] = aruco.Dictionary_getByteListFromBits(mybits)

        self.marker_chars = ["K", "O", "R", "E", "A", "R", "M", "Y", "Heart"]
        self.aruco_param = aruco.DetectorParameters_create()

        self.cap0 = cv2.VideoCapture(2)
        self.cap1 = cv2.VideoCapture(4)
        self.cvbrid = CvBridge()

        self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_size_x)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_size_y)
        self.cap0.set(cv2.CAP_PROP_FPS, self.frame_rate)
        self.cap1.set(cv2.CAP_PROP_FPS, self.frame_rate)

        self.timer_finder = self.create_timer(1/self.frame_rate, self.image_callback)

    def image_callback(self):
        ret0, img0 = self.cap0.read()
        ret1, img1 = self.cap1.read()

        if not (ret0 and ret1):
            if not ret0:
                print('cam0 cannot connect')
            if not ret1:
                print('cam1 cannot connect')
            return

        gray0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        corners0, ids0, _ = aruco.detectMarkers(gray0, self.aruco_dict, parameters=self.aruco_param)
        corners1, ids1, _ = aruco.detectMarkers(gray1, self.aruco_dict, parameters=self.aruco_param)

        img0 = aruco.drawDetectedMarkers(img0, corners0)
        img1 = aruco.drawDetectedMarkers(img1, corners1)

        # --- 로그 출력 추가 ---
        if ids0 is not None:
            for i, corner in enumerate(corners0):
                marker_id = ids0[i][0]
                label = self.marker_chars[marker_id] if 0 <= marker_id < len(self.marker_chars) else str(marker_id)
                x_min, y_min = map(int, np.min(corner[0], axis=0))
                x_max, y_max = map(int, np.max(corner[0], axis=0))
                log_msg = f"Cam0 - Marker: {label})"
                self.get_logger().info(log_msg)
                self.log_pub.publish(String(data=log_msg))  # 퍼블리시

        if ids1 is not None:
            for i, corner in enumerate(corners1):
                marker_id = ids1[i][0]
                label = self.marker_chars[marker_id] if 0 <= marker_id < len(self.marker_chars) else str(marker_id)
                x_min, y_min = map(int, np.min(corner[0], axis=0))
                x_max, y_max = map(int, np.max(corner[0], axis=0))
                log_msg = f"Cam1 - Marker: {label})"
                self.get_logger().info(log_msg)
                self.log_pub.publish(String(data=log_msg))  # 퍼블리시

        frame = np.hstack((img0, img1))
        resized = cv2.resize(frame, (int(self.img_size_x/2), int(self.img_size_y)), interpolation=cv2.INTER_AREA)
        self.publisher.publish(self.cvbrid.cv2_to_imgmsg(resized))

        cv2.imshow('frame', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()