import rclpy
from rclpy.node import Node
import cv2
import pytesseract
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageOcroiMouse(Node):
    def __init__(self):
        super().__init__("image_ocr")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/image_raw", self.image_callback, 10)
        self.get_logger().info("OCR Node started")

        # ROI와 드래그 상태 초기화
        self.roi = None
        self.drawing = False
        self.x1, self.y1 = -1, -1

        # OpenCV 윈도우 및 마우스 이벤트 콜백 등록
        cv2.namedWindow("Original")
        cv2.setMouseCallback("Original", self.mouse_callback)

    def image_callback(self, msg):
        # ROS 2 Image 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 실시간 영상 송출
        if self.roi:
            x, y, w, h = self.roi
            if w > 0 and h > 0:
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # ROI 영역 추출
                roi_image = cv_image[y:y+h, x:x+w]

                # 📌 전처리 (그레이스케일 & 이진화)
                gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

                # 📌 OCR 실행
                custom_config = r'--oem 3 --psm 6'  # OCR 설정
                text = pytesseract.image_to_string(thresh, config=custom_config, lang="eng")

                # 📌 결과 출력
                self.get_logger().info(f"OCR Result: {text}")

                # OCR 화면 표시 (디버깅용)
                cv2.imshow("OCR Processed Image", thresh)

        # 실시간 영상 원본 송출
        cv2.imshow("Original", cv_image)
        cv2.waitKey(1)  # 화면 갱신

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 클릭 시작
            self.drawing = True
            self.x1, self.y1 = x, y  # 시작 좌표 저장

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:  # 드래그 중
            self.roi = (self.x1, self.y1, x - self.x1, y - self.y1)  # ROI 갱신

        elif event == cv2.EVENT_LBUTTONUP:  # 마우스 버튼 떼면 ROI 확정
            self.drawing = False
            x2, y2 = x, y
            self.roi = (self.x1, self.y1, x2 - self.x1, y2 - self.y1)  # 최종 ROI 설정
            self.get_logger().info(f"ROI selected: {self.roi}")  # 선택된 ROI 출력

def main(args=None):
    rclpy.init(args=args)
    node = ImageOcroiMouse()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # 창 닫기

if __name__ == "__main__":
    main()
