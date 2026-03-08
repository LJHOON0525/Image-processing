import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageRoiExtracion(Node):
    def __init__(self):
        super().__init__('image_roiextraction')
        self.subscription = self.create_subscription(Image, 'image_raw', self.process_frame, 10)
        self.bridge = CvBridge()
        self.frame = None
        self.roi = None  # ROI 좌표 (x, y, w, h)
        self.drawing = False  # ROI 드래그 상태

        # OpenCV 윈도우 생성 및 마우스 이벤트 콜백 등록
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", self.mouse_callback)

    def process_frame(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        if self.roi:
            x, y, w, h = self.roi
            if w > 0 and h > 0:  # ROI 영역이 유효한 경우만 표시
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # ROI 표시

                # ROI 영역 추출
                roi_frame = self.frame[y:y+h, x:x+w]
                self.newframe(roi_frame)  # 새 창에 ROI 영역 표시
                

        cv2.imshow("Select ROI", self.frame)  # 원본 영상
        cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 클릭 시작
            self.drawing = True
            self.roi = [x, y, 0, 0]  # 시작 좌표 설정, w와 h는 0으로 설정

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:  # 드래그 중
            w = x - self.roi[0]  # 너비 계산
            h = y - self.roi[1]  # 높이 계산
            self.roi[2], self.roi[3] = w, h  # w, h 업데이트

        elif event == cv2.EVENT_LBUTTONUP:  # 마우스 버튼을 떼면 ROI 확정
            self.drawing = False
            x, y, w, h = self.roi
            print(f"Selected ROI: ({x}, {y}) -> ({x + w}, {y + h})")  # 선택된 ROI 좌표 출력

    def newframe(self, roi_frame):  # 추출한 ROI 영역을 새로운 창에 표시
        if roi_frame is not None and roi_frame.size > 0:  # ROI가 유효한지 확인
            # ROI 영역의 크기를 변경 (예: 2배 확대)
            enlarged_roi = cv2.resize(roi_frame, (roi_frame.shape[0] * 2, roi_frame.shape[1] * 2))  # 높이와 너비를 2배로 변경
            cv2.imshow("ROI Extracted", enlarged_roi)  # 새로운 창에 ROI 부분 표시
            cv2.waitKey(1)  # 키 입력 대기

    def destroy_node(self):
        cv2.destroyAllWindows()  # OpenCV 윈도우 종료
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageRoiExtracion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
