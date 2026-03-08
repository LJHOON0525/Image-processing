import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        
        # 이미지 퍼블리셔 초기화
        self.publisher_ = self.create_publisher(Image, 'lane_detection_image', QoSProfile(depth=10))
        
        # 카메라 캡처 객체
        self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            self.get_logger().error("웹캠을 열 수 없습니다.")
            exit()

        self.bridge = CvBridge()

        # 차선 감지 관련 변수
        self.img_size_x = 640
        self.img_size_y = 480
        self.W_H = 640
        self.W_L = 480
        self.W_H_ratio = 0.6
        self.height_ratio = 0.5
        self.warped_remain_ratio = 1.0

        # 주기적으로 콜백 호출
        self.timer = self.create_timer(0.1, self.timer_callback)

    def warp(self, img):
        # 원근 왜곡 보정 (Perspective transform)
        src = np.float32([[0, 0],   
                        [int(self.W_H), 0],     
                        [int(self.W_H - (self.W_H-self.W_L)/2), int(self.img_size_y * (1-self.W_H_ratio))*self.height_ratio],
                        [int((self.W_H-self.W_L)/2), int(self.img_size_y * (1-self.W_H_ratio))*self.height_ratio]     
                        ])
        dst = np.float32([[0, int(self.img_size_y * self.W_H_ratio)],
                        [self.img_size_x, int(self.img_size_y * self.W_H_ratio)],
                        [self.img_size_x, self.img_size_y],
                        [0, self.img_size_y]])

        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        # Perspective Transform
        binary_warped = cv2.warpPerspective(img, Minv, (int(self.W_H), int(self.img_size_y * (1-self.W_H_ratio)*self.height_ratio*self.warped_remain_ratio)), flags=cv2.INTER_LINEAR)
        
        return binary_warped

    def process_image(self, img):
        # 1. 색상 기반 차선 감지 (검정색 차선만 추출)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])  # 검정색 범위 (조정 가능)
        upper_black = np.array([180, 255, 50])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        
        result = cv2.bitwise_and(img, img, mask=mask)
        
        # 2. 검정색이 아닌 부분을 가우시안 블러로 처리
        blurred_result = cv2.GaussianBlur(result, (3, 3), 0)
        
        # 3. 그레이스케일 변환 후, 히스토그램 평활화로 잡음 제거
        gray = cv2.cvtColor(blurred_result, cv2.COLOR_BGR2GRAY)
        # 히스토그램 평활화
        hist_eq = cv2.equalizeHist(gray)
        
        # 4. 모폴로지 연산 (침식 및 팽창)
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(hist_eq, kernel, iterations=1)  # 침식
        dilated = cv2.dilate(eroded, kernel, iterations=1)  # 팽창

        # 5. 차선 감지 후, 결과 반환
        return dilated

    def detect_lane(self, img):
        # 왜곡 보정 후 차선 감지
        warped_img = self.warp(img)
        
        # 차선 감지 및 처리
        processed_img = self.process_image(warped_img)
        
        # 차선의 좌표 찾기 (여기서부터 다항식 피팅)
        return self.find_lane_pixels(processed_img)

    def find_lane_pixels(self, binary_warped):
        # OpenCV 이미지를 NumPy 배열로 변환
        binary_warped = np.array(binary_warped)

        # 히스토그램을 사용하여 차선의 초기 위치를 추정
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        
        # 왼쪽, 오른쪽 차선의 시작점
        midpoint = int(binary_warped.shape[1] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # 차선의 위치 추적 (슬라이딩 윈도우 기법 사용)
        nwindows = 9
        window_height = int(binary_warped.shape[0] / nwindows)
        margin = 100
        minpix = 50
        leftx_current = leftx_base
        rightx_current = rightx_base

        leftx = []
        lefty = []
        rightx = []
        righty = []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # 왼쪽 차선, 오른쪽 차선 픽셀 찾기
            good_left_inds = ((binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high] > 0).nonzero())
            good_right_inds = ((binary_warped[win_y_low:win_y_high, win_xright_low:win_xright_high] > 0).nonzero())

            # 찾은 픽셀 좌표 저장
            leftx.append(good_left_inds[1] + win_xleft_low)
            lefty.append(good_left_inds[0] + win_y_low)
            rightx.append(good_right_inds[1] + win_xright_low)
            righty.append(good_right_inds[0] + win_y_low)

            if len(good_left_inds[0]) > minpix:
                leftx_current = int(np.mean(good_left_inds[1] + win_xleft_low))
            if len(good_right_inds[0]) > minpix:
                rightx_current = int(np.mean(good_right_inds[1] + win_xright_low))

        leftx = np.concatenate(leftx)
        lefty = np.concatenate(lefty)
        rightx = np.concatenate(rightx)
        righty = np.concatenate(righty)

        # 다항식 피팅
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # 차선의 곡률 계산
        left_curverad, right_curverad = self.measure_curvature(leftx, rightx, lefty)

        # 값 로그로 출력
        self.get_logger().info(f"Left fit: {left_fit}")
        self.get_logger().info(f"Right fit: {right_fit}")
        self.get_logger().info(f"Left curvature: {left_curverad}")
        self.get_logger().info(f"Right curvature: {right_curverad}")

        return left_fit, right_fit, left_curverad, right_curverad

    def measure_curvature(self, leftx, rightx, ploty):
        ym_per_pix = 1  # m/pixel, 실제 거리로 변환하는 비율
        xm_per_pix = 1  # m/pixel, 실제 거리로 변환하는 비율

        # 곡률 계산을 위한 다항식 피팅
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        # 차선의 곡률 반경 계산
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def timer_callback(self):
        # 카메라에서 프레임 읽기
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("프레임을 읽을 수 없습니다.")
            return

        # 차선 감지
        left_fit, right_fit, left_curverad, right_curverad = self.detect_lane(frame)

        # 결과를 ROS 메시지로 변환하여 퍼블리시
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"이미지 변환 에러: {e}")
        
        # 차선 감지 결과를 화면에 표시
        cv2.imshow("Lane Detection", frame)

        # 'q' 키를 눌러 창을 닫을 수 있도록 처리
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("프로그램 종료")
            self.cap.release()
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)
    lane_detection_node.cap.release()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
