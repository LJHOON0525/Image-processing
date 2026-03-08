import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LaneUIDisplay(Node):
    def __init__(self):
        super().__init__('lane_ui_display')
        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, 'lane_ui', 10)
        self.timer = self.create_timer(0.033, self.process_frame)  # 30 FPS
        self.cap = cv2.VideoCapture(2)
    
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Failed to capture frame')
            return

        # UI 그리기
        frame = self.draw_ui(frame)
        
        # ROS2 메시지로 변환 후 퍼블리싱
        image_message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(image_message)
        
        # OpenCV 창에 표시
        cv2.imshow("Lane UI", frame)
        cv2.waitKey(1)
    
    def draw_ui(self, image):
        height, width, _ = image.shape
        overlay = image.copy()
        
        # 파란색 네모 박스
        box_top_left = (width // 12, height * 2 // 3)  # 파란색 영역 늘리기
        box_bottom_right = (width * 11 // 12, height - 50) # 파란색 영역 늘리기
        cv2.rectangle(overlay, box_top_left, box_bottom_right, (255, 0, 0), 2)
        
        # 노란색 수평선 (중앙 흰색 선을 기준으로 좌우 분리)
        line_y = height - 100
        center_x = width // 2
        offset = 155  # 중앙선 기준 좌우 간격
        left_x1 = center_x - offset - 60 # 짧을시 좌측 노란색으로 늘림
        left_x2 = center_x - offset
        right_x1 = center_x + offset
        right_x2 = center_x + offset + 60 # 짧을시 우측 노란색으로 늘림
        cv2.line(overlay, (left_x1, line_y), (left_x2, line_y), (0, 255, 255), 2)
        cv2.line(overlay, (right_x1, line_y), (right_x2, line_y), (0, 255, 255), 2)
        
        # 중앙 십자선
        cv2.line(overlay, (center_x, line_y - 10), (center_x, line_y + 10), (255, 255, 255), 2)
        cv2.line(overlay, (center_x - 10, line_y), (center_x + 10, line_y), (255, 255, 255), 2)
        
        # 텍스트 추가
        cv2.putText(overlay, "Start", (int(width// 2.25) , height - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return overlay
    
    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LaneUIDisplay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
