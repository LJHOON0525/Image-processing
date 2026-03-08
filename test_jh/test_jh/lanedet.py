import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LaneDetectNode(Node):
    def __init__(self):
        super().__init__('lane_detect_node')
        self.publisher = self.create_publisher(Image, 'lane_detect_output', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(2)  # USB Camera (Video 2)
        self.timer = self.create_timer(0.1, self.detect_lane)

    def pre_treatment_img(self, origin_img):
        """Pre-process the image with black color filtering and low-light enhancement."""
        hsv_color = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
        hsv_lower = np.array([0, 0, 0])
        hsv_upper = np.array([180, 255, 50])
        hsv_mask = cv2.inRange(hsv_color, hsv_lower, hsv_upper)
        
        hsv_filtered = cv2.bitwise_and(hsv_color, hsv_color, mask=hsv_mask)
        color_filtered = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)
        
        gray = cv2.cvtColor(color_filtered, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)  # 커널 크기 수정
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        return gray

    def detect_lane(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image")
            return
        
        processed_img = self.pre_treatment_img(frame)

        # Canny edge detection 적용 (전처리된 이미지 사용)
        edges = cv2.Canny(processed_img, 100, 200)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        cleaned_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w = processed_img.shape  # 전처리 이미지 크기 기반
        cx, cy = w // 2, h // 2
        rect_width = 640 
        rect_height = 75  
        rect_top_left = (cx - rect_width // 2, h - rect_height * 2)
        rect_bottom_right = (cx + rect_width // 2, h - rect_height)

        # 원본 프레임에 상자 그리기
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, (255, 0, 0), 2)
        cv2.line(frame, (cx, rect_top_left[1]), (cx, rect_bottom_right[1]), (0, 255, 255), 2)
        
        # 빨간색 점을 y=365 
        y_position = 365
        
        # 왼쪽 끝 40픽셀 
        cv2.circle(frame, (rect_top_left[0] + 40, y_position), 10, (0, 0, 255), -1)

        # 오른쪽 끝 40픽셀 
        cv2.circle(frame, (rect_bottom_right[0] - 40, y_position), 10, (0, 0, 255), -1)

        left_red_x = rect_top_left[0] + 40  
        right_red_x = rect_bottom_right[0] - 40 
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # 경계 영역 설정
                for point in contour:
                    x, y = point[0]
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # 파란색 점

        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)  # 윤곽선 그리기

        # 파란색 경계와 빨간색 점들을 비교하여 메세지 판단
        if cx < left_red_x:
            self.get_logger().info("Lane is to the left")
        elif cx > right_red_x:
            self.get_logger().info("Lane is to the right")
        else:
            self.get_logger().info("Lane is straight")

        img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher.publish(img_msg)
        
        # Show images (optional, for debugging)
        cv2.imshow("Lane Detection", frame)
        cv2.imshow("Processed Image", processed_img)  # 전처리 이미지 표시
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            self.destroy_node()
    
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
