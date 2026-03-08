import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Float32MultiArray
import odrive
from odrive.enums import InputMode, CONTROL_MODE_VELOCITY_CONTROL, INPUT_MODE_VEL_RAMP
from sensor_msgs.msg import Joy
import time

class Odrive_processing(Node):
    def __init__(self):
        super().__init__('test_car_sub')


################################################# JOYSTICK #################################################

        qos_profile = QoSProfile(depth=10)
        #Joystick

        #self.joy_subscriber = self.create_subscription(Joy, 'joy', self.joy_msg_sampling, qos_profile)


        #Motor Controller
        self.car_drive = odrive.find_any(serial_number="3678387D3333")  # 구동모터 오드라이브

        self.calibration()

        

        self.subscription = self.create_subscription(
            Float32MultiArray,
            'Odrive_control',
            self.subscribe_topic_message,
            qos_profile)

        self.publisher = self.create_publisher(
            Float32MultiArray,
            'Odrive_encoder', 
            qos_profile
        )
        self.timer = self.create_timer(0.1, self.encoder_callback) 
        self.mode = 'pos'
        self.cur_mode = 'pos'
        self.pos_axis0 = 0.0
        self.pos_axis1 = 0.0
        
        #Joy Stick
        self.joy_status = False
        self.joy_stick_data = [0, 0]

    def calibration(self):
        self.get_logger().info('Calibration START')
        self.get_logger().info('-------------------------------')

        # CAR Axis 0
        self.car_drive.axis0.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis0.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
     
        self.car_drive.axis0.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('CAR RIGHT Calibration COMPLETE.')

        self.get_logger().info('-------------------------------')

        #CAR Axis 1
        self.car_drive.axis1.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis1.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
     
        self.car_drive.axis1.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL       


        self.get_logger().info('CAR LEFT Calibration COMPLETE.')
        self.get_logger().info('-------------------------------')

    def subscribe_topic_message(self, msg):
        self.get_logger().info('Received data : {}'.format(msg.data))
        movedata = msg.data[1:]  
        odrivemode = msg.data[0] 

        if odrivemode == 1:    #Ramped 속도 제어 모드
            self.mode = 'vel'
            
        elif odrivemode == 2:  #상대 위치제어 모드 : Trajectory
            self.mode = 'pos'
        
        elif odrivemode == 0:
            self.mode = 'stop'

        else:
            self.get_logger().warn('Invalid control mode received: {}'.format(msg.data[0]))

        self.motor_mode_set()
        self.motor_run(movedata)


    def motor_mode_set(self):
        if self.mode == "vel" and self.cur_mode != "vel":

            self.car_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.car_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.car_drive.axis0.controller.config.vel_ramp_rate = 10
            self.car_drive.axis1.controller.config.vel_ramp_rate = 10
            self.car_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            self.car_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            self.get_logger().info('Change mode to velocity')
            self.cur_mode = "vel"
            
        elif self.mode == "pos" and self.cur_mode != "pos":

            #PASSTHROUGH Mode

            self.car_drive.axis0.controller.config.input_mode = 1
            self.car_drive.axis1.controller.config.input_mode = 1
         
            ############################ ENCODER Offset ##################################
            self.pos_axis0_offset = self.car_drive.axis0.encoder.pos_estimate
            self.pos_axis1_offset = self.car_drive.axis1.encoder.pos_estimate
            
            # INPUT POS에 현재 엔코더 값 넣기
            self.car_drive.axis0.controller.input_pos = self.pos_axis0_offset 
            self.car_drive.axis1.controller.input_pos = self.pos_axis1_offset
            self.get_logger().info(f'Change Offset : {self.pos_axis0_offset} , {self.pos_axis1_offset}')  

            
            self.car_drive.axis0.controller.config.control_mode = 3
            self.car_drive.axis1.controller.config.control_mode = 3

            self.car_drive.axis0.controller.config.input_mode = InputMode.TRAP_TRAJ
            self.car_drive.axis1.controller.config.input_mode = InputMode.TRAP_TRAJ
            self.get_logger().info('Change mode to position')
            self.cur_mode = "pos"


    def motor_run(self, data):
        
        if self.joy_status:
            self.joy_msg_control()
        
        else:

            if self.cur_mode == "pos":          #trajectory Control  #상대 위치 설정
                self.car_drive.axis0.controller.input_pos = data[1] + self.pos_axis0_offset
                self.car_drive.axis1.controller.input_pos = data[0] + self.pos_axis1_offset
            # self.get_logger().info(f'Position control set : axis0 = {data[0]}, axis1 = {data[1]}')
             
            elif self.cur_mode == "vel":                            # Ramped Velocity Control  #속도 조절
                self.car_drive.axis0.controller.input_vel = -data[1] #RIGHT
                self.car_drive.axis1.controller.input_vel = data[0] #LEFT
            # self.get_logger().info(f'Velocity control set : axis0 = {data[0]}, axis1 = {data[1]}')
            else : 
                self.get_logger().fatal(f'Invalid mode!! reset to vel mode')
                self.mode = "vel"
            self.encoder_check()
        
########################### Motor Encoder Feedback ###########################
    def encoder_check(self):
        self.pos_axis0 = self.car_drive.axis0.encoder.pos_estimate
        self.pos_axis1 = self.car_drive.axis1.encoder.pos_estimate
        # self.get_logger().info('Encoder positions: axis0 = {}, axis1 = {}'.format(self.pos_axis0, self.pos_axis1))





    def encoder_callback(self):
        msg = Float32MultiArray()
        msg.data = [self.pos_axis0, self.pos_axis1]
        self.publisher.publish(msg)
        # self.get_logger().info(f'ENCODER Value: {self.pos_axis0} , {self.pos_axis1}')  


########################### JOY STICK ###########################
#     def joy_msg_sampling(self, msg):
#         self.axes = msg.axes
#         # btn = msg.buttons
#         #fix
#         btn = msg.buttons
#         self.joy_status = btn[7] == 1 and btn[8] != 1

#     def joy_msg_control(self):
# #odrive set
#         self.car_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
#         self.car_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
#         self.car_drive.axis0.controller.config.vel_ramp_rate = 10
#         self.car_drive.axis1.controller.config.vel_ramp_rate = 10
#         self.car_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
#         self.car_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP
#         self.get_logger().info('JOYSTICK MODE')
        
#         if self.btn[7] == 1 and self.btn[8] == 1:
#             self.car_drive.axis0.controller.input_vel = 0.0
#             self.car_drive.axis1.controller.input_vel = 0.0
#         else:
#             self.car_drive.axis0.controller.input_vel = self.axes[1] #RIGHT
#             self.car_drive.axis1.controller.input_vel = self.count_publishersaxes[3] #LEFT


def main(args=None):
    rclpy.init(args=args)
    sub = Odrive_processing()
    rclpy.spin(sub)
    sub.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
