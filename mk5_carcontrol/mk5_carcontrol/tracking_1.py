import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32MultiArray
import time

class TrackControl1(Node):
    def __init__(self):
        super().__init__('TrackControl1')

        qos_profile = QoSProfile(depth=10)
        
        # ODrive Command 퍼블리셔
        self.control_publisher = self.create_publisher(Float32MultiArray, 'Odrive_control', qos_profile)
        self.drive_command = Float32MultiArray()

        self.odrive_mode = 1.0 # Ramped Velocity

        # Lidar Distance 구독
        self.lidar_sub = self.create_subscription(
            Float32MultiArray,
            '/lidar_distance',
            self.algorithm_control,
            qos_profile
        )

    # --------------- Parameters ---------------
        #라이다
        # 좌/전방/우 거리 측정값 초기화
        self.left = 0.0
        self.front = 0.0
        self.right = 0.0
        # 라이다 전방 안전 거리 (m)
        self.safe_distance = 1.0    
        
        # 기본 직진 속도
        self.base_speed = 6.0       
        # 바퀴 속도 초기화
        self.car_left = 0.0
        self.car_right = 0.0


        # 측면 균형 비례 상수 => STATE 1에서 거리 조절할 때 !
        self.kp = 1.8               
        self.max_turn = 1.5        # 최대 회전 속도 제한
        self.max_speed = 10.0   

        # 라이다 센서 값 히스토리
        self.left_history = []
        self.right_history = []

    #----------------- Algorithm -------------------

    def algorithm_control(self, msg: Float32MultiArray):
        self.left, self.front, self.right = msg.data

        # STATE1: 전방 안전 거리 확보
        if self.front > self.safe_distance:
            self.state1_drive()

        # STATE2: 전방 장애물 감지 시 회피
        else:
            self.state2_avoid()

        # ODrive 퍼블리시
        self.drive_command.data = [1.0,self.odrive_mode,self.car_left, self.car_right]
        self.control_publisher.publish(self.drive_command)


    #---------------- STATE Functions ----------------
    # STATE1: 측면 균형 유지 + 직진 (속도 제한 + 미세 조정)
    def state1_drive(self):
        # 최근 5개 센서 평균
        self.left_history.append(self.left)
        self.right_history.append(self.right)
        if len(self.left_history) > 5:
            self.left_history.pop(0)
            self.right_history.pop(0)
        avg_left = sum(self.left_history)/len(self.left_history)
        avg_right = sum(self.right_history)/len(self.right_history)

        # 좌우 거리 차이 비율 계산
        if max(avg_left, avg_right) > 0:
            ratio_error = (avg_left - avg_right) / max(avg_left, avg_right)
        else:
            ratio_error = 0.0

        # 미세 회전 계산
        correction = ratio_error * self.base_speed * 0.3
        correction = max(min(correction, self.max_turn), -self.max_turn)

        # 속도 적용 및 제한
        self.car_left = self.base_speed - correction
        self.car_right = self.base_speed + correction
        # self.car_left = -5.0
        # self.car_right = -5.0

        self.get_logger().info(
            f"STATE1: FRONT CLEAR | L={self.left:.2f} R={self.right:.2f} | Correction={correction:.2f} | "
            f"Car_L={self.car_left:.2f} Car_R={self.car_right:.2f}"
        )

    # STATE2: 전방 장애물 회피 (제자리 회전)
    def state2_avoid(self):
        self.get_logger().info(f"STATE2: OBSTACLE AHEAD - Turning Right | FRONT={self.front:.2f}")
        self.right2front()

        # self.car_left = self.base_speed
        # self.car_right = self.base_speed

    #----------------- Manual Functions -------------------
    def stop(self):
        self.get_logger().info("STOP")
        self.car_left = 0.0
        self.car_right = 0.0
    
    def right2front(self):
        self.car_left = self.base_speed
        self.car_right = -self.base_speed
        time.sleep(1)
        self.car_left = self.base_speed *0.5
        self.car_right = self.base_speed *0.5




def main(args=None):
    rclpy.init(args=args)
    node = TrackControl1()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
