import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class WebcamYellowNode(Node):
    def __init__(self):
        super().__init__('webcam_yellow_node')

        # 웹캠 설정 (2번)
        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.bridge = CvBridge()

        # ROS 퍼블리셔
        self.yellow_pub = self.create_publisher(Image, 'yellow_mask', 10)
        self.color_pub = self.create_publisher(Image, 'color_image', 10)

        # 반복 타이머 (30Hz)
        self.timer = self.create_timer(1.0 / 30, self.timer_callback)

        # 노란색 HSV 범위 (H:0~179, S,V:0~255)
        self.lower_yellow = np.array([0, 0, 130])
        self.upper_yellow = np.array([180, 30, 255])

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('No frame received from webcam')
            return

        # 노이즈 제거
        blurred = cv2.GaussianBlur(frame, (5, 5), 1)
        denoised = cv2.bilateralFilter(blurred, d=9, sigmaColor=75, sigmaSpace=75)

        # BGR → HSV 변환
        hsv_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)

        # 노란색 마스크 생성
        yellow_mask = cv2.inRange(hsv_image, self.lower_yellow, self.upper_yellow)

        # 컨투어 검출
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0
        max_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour

        display_img = denoised.copy()

        if max_contour is not None and max_area > 100:
            # 컨투어 그리기
            cv2.drawContours(display_img, [max_contour], -1, (0, 0, 255), 2)

            # 해당 영역의 마스크
            contour_mask = np.zeros(yellow_mask.shape, dtype=np.uint8)
            cv2.drawContours(contour_mask, [max_contour], -1, 255, thickness=cv2.FILLED)

            # 평균 HSV 값 계산
            mean_val = cv2.mean(hsv_image, mask=contour_mask)
            mean_h, mean_s, mean_v = mean_val[:3]

            # 중심좌표 계산
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 20, 20

            # 화면에 HSV 값 표시
            cv2.putText(display_img, f"H:{mean_h:.0f} S:{mean_s:.0f} V:{mean_v:.0f}", 
                        (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # OpenCV 창으로 출력
        cv2.imshow("Yellow Mask", yellow_mask)
        cv2.imshow("Detected Region", display_img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 종료
            self.get_logger().info("ESC pressed. Shutting down...")
            rclpy.shutdown()

        # ROS 퍼블리시
        try:
            mask_msg = self.bridge.cv2_to_imgmsg(yellow_mask, encoding='mono8')
            self.yellow_pub.publish(mask_msg)

            color_msg = self.bridge.cv2_to_imgmsg(display_img, encoding='bgr8')
            self.color_pub.publish(color_msg)
        except Exception as e:
            self.get_logger().error(f"Image publish failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = WebcamYellowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt, shutting down.')
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
