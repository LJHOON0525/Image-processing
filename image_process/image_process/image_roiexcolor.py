import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class ImageRoiExtraction(Node):
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
                self.newframe(roi_frame)  # 새 창에 ROI 영역 표시 및 색 추출
                
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

    def newframe(self, roi_frame):  # 추출한 ROI 영역을 새로운 창에 표시 및 RGB 색상 추출
        if roi_frame is not None and roi_frame.size > 0:  # ROI가 유효한지 확인
            #enlarged_roi = cv2.resize(roi_frame, (roi_frame.shape[0] * 2, roi_frame.shape[1] * 2))  # 높이와 너비를 2배로 변경
            # ROI 색 검출
            # 가우시안 블러링 적용 (커널 크기 3x3)
            blurred_roi = cv2.GaussianBlur(roi_frame, (3, 3), 0)
            hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

            # 파란색 범위
            lower_blue = np.array([100, 150, 50])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            result_blue = cv2.bitwise_and(roi_frame, roi_frame, mask=mask_blue)

            # 초록색 범위
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([90, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            result_green = cv2.bitwise_and(roi_frame, roi_frame, mask=mask_green)

            # 빨간색 범위 (두 개의 범위 필요)
            lower_red1 = np.array([0, 150, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 150, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask_red = mask_red1 | mask_red2
            result_red = cv2.bitwise_and(roi_frame, roi_frame, mask=mask_red)

            # ROI 및 색상 감지된 결과 창 출력
            cv2.imshow("ROI Extracted", roi_frame)
            cv2.imshow("Blue Detection", result_blue)
            cv2.imshow("Green Detection", result_green)
            cv2.imshow("Red Detection", result_red)
            cv2.waitKey(1)

    def destroy_node(self):
        cv2.destroyAllWindows()  # OpenCV 윈도우 종료
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = ImageRoiExtraction()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
