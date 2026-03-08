import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String,Bool
import odrive
from odrive.enums import AXIS_STATE_FULL_CALIBRATION_SEQUENCE, AXIS_STATE_CLOSED_LOOP_CONTROL, CONTROL_MODE_POSITION_CONTROL
import time

class FlipperControl(Node):
    def __init__(self):
        super().__init__('FlipperControl')

        qos_profile = QoSProfile(depth=10)
        self.subscription_flip = self.create_subscription(
            String,
            'Odrive_Flip_control',
            self.subscribe_flip_message,
            qos_profile)
        
        #FLIP동작 플래그
        self.flipON_done_pub = self.create_publisher(Bool, 'flipON_done', qos_profile)
        self.flipINIT_done_pub = self.create_publisher(Bool, 'flipINIT_done', qos_profile)

        self.get_logger().info("Connecting to FLIPPER")
        self.flip_drive = odrive.find_any(serial_number="205C327D4D31") # 연결된 ODrive 자동 검색
        self.get_logger().info("ODrive connected!")

        self.calibration()

        # 현재 명령과 목표 위치 초기화
        self.left_target_pos= None
        self.right_target_pos= None
        self.current_command = None

    # ------------------------- 캘리브레이션 -------------------------
    def calibration(self):
        self.get_logger().info("Calibration START")
        #------ RIGHT FLIPPER 
        self.flip_drive.axis0.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.flip_drive.axis0.current_state != odrive.enums.AXIS_STATE_IDLE:  # IDLE 상태 대기
            time.sleep(0.1)

        self.flip_drive.axis0.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.flip_drive.axis0.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
        time.sleep(0.1)
        self.get_logger().info("---------- RIGHT Cali DONE ------------")

        #------ LEFT FLIPPER 
        self.flip_drive.axis1.requested_state = AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.flip_drive.axis1.current_state != odrive.enums.AXIS_STATE_IDLE:  # IDLE 상태 대기
            time.sleep(0.1)

        self.flip_drive.axis1.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        self.flip_drive.axis1.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
        time.sleep(0.1)
        self.get_logger().info("---------- LEFT Cali DONE ---------- ")

        # 속도 제한
        self.flip_drive.axis0.controller.config.vel_limit = 8
        self.flip_drive.axis1.controller.config.vel_limit = 8 

        # Trap Trajectory 모드 적용
        self.flip_drive.axis0.controller.config.input_mode = 5  # INPUT_MODE_TRAP_TRAJ
        self.flip_drive.axis1.controller.config.input_mode = 5

        # 가속/감속 제한 설정
        self.flip_drive.axis0.trap_traj.config.accel_limit = 20   # 상황에 따라 10~30 조절 가능
        self.flip_drive.axis0.trap_traj.config.decel_limit = 18
        self.flip_drive.axis1.trap_traj.config.accel_limit = 20
        self.flip_drive.axis1.trap_traj.config.decel_limit = 18

        # 초기 위치 0 설정
        self.flip_drive.axis0.encoder.set_linear_count(0)
        self.flip_drive.axis1.encoder.set_linear_count(-10)
        self.get_logger().info("Initial Position Set to 0")

        self.current_pos_left = 0.0
        self.current_pos_right = 0.0

    # ------------------------- 메시지 수신 -------------------------
    def subscribe_flip_message(self, msg):
        command = msg.data.upper()

        if command == "FLIPON":
            # 항상 같은 목표 위치 (누적 X)
            self.left_target_pos = 140
            self.right_target_pos = -110

            self.current_command = "FLIPON"
            # 동시에 명령
            self.flip_drive.axis0.controller.input_pos = self.right_target_pos
            self.flip_drive.axis1.controller.input_pos = self.left_target_pos
            self.get_logger().info(f"FLIPON -> Left={self.left_target_pos}, Right={self.right_target_pos}")

        elif command == "FLIPINIT":
            self.left_target_pos = 0
            self.right_target_pos = 0
    
            self.current_command = "FLIPINIT"
            self.flip_drive.axis0.controller.input_pos = self.right_target_pos
            self.flip_drive.axis1.controller.input_pos = self.left_target_pos
            self.get_logger().info(f"FLIPINIT -> Left={self.left_target_pos}, Right={self.right_target_pos}")

        else:
            self.get_logger().warn(f"Unknown command: {command}")
            return

        # 이동 완료 확인용 타이머
        if not hasattr(self, "done_timer"):
            self.done_timer = self.create_timer(0.05, self.check_position_done)

    def check_position_done(self):
        if self.current_command is None:
            return

        self.current_pos_left = self.flip_drive.axis1.encoder.pos_estimate
        self.current_pos_right = self.flip_drive.axis0.encoder.pos_estimate

        if (abs(self.current_pos_left - self.left_target_pos) < 1 and
            abs(self.current_pos_right - self.right_target_pos) < 1):
            done_msg = Bool()
            done_msg.data = True

            if self.current_command == "FLIPON":
                self.flipON_done_pub.publish(done_msg)
                self.get_logger().info("FLIPON movement DONE")
            elif self.current_command == "FLIPINIT":
                self.flipINIT_done_pub.publish(done_msg)
                self.get_logger().info("FLIPINIT movement DONE")

            self.current_command = None


def main(args=None):
    rclpy.init(args=args)
    flipper_node = FlipperControl()
    try:
        rclpy.spin(flipper_node)
    except KeyboardInterrupt:
        flipper_node.get_logger().info("Keyboard Interrupt, shutting down")
    finally:
        flipper_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
