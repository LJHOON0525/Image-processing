import rclpy
from rclpy.node import Node
import cv2
from ultralytics import YOLO

class YoloImage(Node):
    def __init__(self):
        super().__init__('yolo_image_detector')
        self.declare_parameter('image_path', 'bus.jpg')  # 기본값: bus.jpg

        # model
        self.model = YOLO('yolov8n.pt')

        # image 읽기
        image_path = self.get_parameter('image_path').value
        image = cv2.imread(image_path)

        if image is None:
            self.get_logger().error(f"이미지를 찾을 수 없습니다: {image_path}")
            return

        # 실행
        results = self.model(image)
        annotated_image = results[0].plot()

        cv2.imshow("YOLO Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = YoloImage()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
