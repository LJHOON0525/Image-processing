import rclpy
from rclpy.node import Node
import cv2
import pytesseract
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class ImageOCRNode(Node):
    def __init__(self):
        super().__init__("image_ocr")
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, "/image_raw", self.image_callback, 10)
        self.get_logger().info("OCR Node started")

    def image_callback(self, msg):
        # ROS 2 Image 메시지를 OpenCV 이미지로 변환
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # 📌 전처리 (그레이스케일 & 이진화)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # 📌 OCR 실행 (한글: lang="kor", 영어: lang="eng")
        custom_config = r'--oem 3 --psm 6' # OCR 설정
        text = pytesseract.image_to_string(thresh, config=custom_config, lang="eng")

        # 📌 결과 출력
        self.get_logger().info(f"OCR Result: {text}")

        # OCR 화면 표시 (디버깅용)
        cv2.imshow("Original",cv_image)
        cv2.imshow("OCR Processed Image", thresh)
        cv2.waitKey(1)  # 화면 갱신

def main(args=None):
    rclpy.init(args=args)
    node = ImageOCRNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()  # 창 닫기

if __name__ == "__main__":
    main()
