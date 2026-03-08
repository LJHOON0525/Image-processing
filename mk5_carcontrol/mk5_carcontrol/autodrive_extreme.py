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
        self.flip_drive = odrive.find_any(serial_number="205C327D4D31")

        self.calibration()

        

        self.subscription_car = self.create_subscription(
            Float32MultiArray,
            'Odrive_car_control',
            self.subscribe_car_message,
            qos_profile)

        self.subscription_flip = self.create_subscription(
            Float32MultiArray,
            'Odrive_flip_control',
            self.subscribe_flip_message,
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
        # CAR Axis 0
        self.car_drive.axis0.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis0.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
     
        self.car_drive.axis0.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('CAR RIGHT Calibration COMPLETE.')

        # #CAR Axis 1
        self.car_drive.axis1.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.car_drive.axis1.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
     
        self.car_drive.axis1.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL       


        self.get_logger().info('CAR LEFT Calibration COMPLETE.')

        #Flipper  Axis0

        self.flip_drive.axis0.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.flip_drive.axis0.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)
        self.flip_drive.axis0.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('Flipper RIGHT COMPLETE.')

        #Flipper Axis1
        self.flip_drive.axis1.requested_state = odrive.enums.AXIS_STATE_FULL_CALIBRATION_SEQUENCE
        while self.flip_drive.axis1.current_state != odrive.enums.AXIS_STATE_IDLE:
            time.sleep(0.1)         
        
        self.flip_drive.axis1.requested_state = odrive.enums.AXIS_STATE_CLOSED_LOOP_CONTROL
        self.get_logger().info('Flipper LEFT COMPLETE.')



#----------------------------- 차량 제어 ------------------

    def subscribe_car_message(self, msg):
        self.get_logger().info('Received CAR data : {}'.format(msg.data))
        car_movedata = msg.data[1:]  
        car_mode = msg.data[0] 

        if car_mode == 1:    #Ramped 속도 제어 모드
            self.car_mode = 'vel'
            
        elif car_mode == 2:  #상대 위치제어 모드 : Trajectory
            self.car_mode = 'pos'
        
        elif car_mode == 0:
            self.car_mode = 'stop'

        else:
            self.get_logger().warn('Invalid CAR mode received: {}'.format(msg.data[0]))

        self.car_mode_set()
        self.car_run(car_movedata)


    def subscribe_flip_message(self, msg):
        self.get_logger().info('Received FLIPPER data : {}'.format(msg.data))
        flip_movedata = msg.data[1:]  
        flip_mode = msg.data[0] 

        if flip_mode == 1:    #Ramped 속도 제어 모드
            self.flip_mode = 'vel'
            
        elif flip_mode == 2:  #상대 위치제어 모드 : Trajectory
            self.flip_mode = 'pos'
        
        elif flip_mode == 0:
            self.flip_mode = 'stop'

        else:
            self.get_logger().warn('Invalid FLIPPER mode received: {}'.format(msg.data[0]))

        self.flip_mode_set()
        self.flip_run(flip_movedata)


# 구동 모터 모드 세팅

    def car_mode_set(self):
        if self.car_mode == "vel" and self.car_cur_mode != "vel":

            self.car_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.car_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.car_drive.axis0.controller.config.vel_ramp_rate = 10
            self.car_drive.axis1.controller.config.vel_ramp_rate = 10
            self.car_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            self.car_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            self.get_logger().info('Change CAR mode to velocity')
            self.car_cur_mode = "vel"
            
        elif self.car_mode == "pos" and self.car_cur_mode != "pos":

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
            self.car_cur_mode = "pos"

#플리퍼 모드 세팅


    def flip_mode_set(self):
        if self.flip_mode == "vel" and self.flip_cur_mode != "vel":

            self.flip_drive.axis0.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.flip_drive.axis1.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
            self.flip_drive.axis0.controller.config.vel_ramp_rate = 10
            self.flip_drive.axis1.controller.config.vel_ramp_rate = 10
            self.flip_drive.axis0.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            self.flip_drive.axis1.controller.config.input_mode = INPUT_MODE_VEL_RAMP
            self.get_logger().info('Change mode to velocity')
            self.flip_cur_mode = "vel"
            
        elif self.flip_mode == "pos" and self.flip_cur_mode != "pos":

            #PASSTHROUGH Mode

            self.flip_drive.axis0.controller.config.input_mode = 1
            self.flip_drive.axis1.controller.config.input_mode = 1
         
            ############################ ENCODER Offset ##################################
            self.pos_axis0_offset = self.flip_drive.axis0.encoder.pos_estimate
            self.pos_axis1_offset = self.flip_drive.axis1.encoder.pos_estimate
            
            # INPUT POS에 현재 엔코더 값 넣기
            self.flip_drive.axis0.controller.input_pos = self.pos_axis0_offset 
            self.flip_drive.axis1.controller.input_pos = self.pos_axis1_offset
            self.get_logger().info(f'Change Offset : {self.pos_axis0_offset} , {self.pos_axis1_offset}')  

            
            self.flip_drive.axis0.controller.config.control_mode = 3
            self.flip_drive.axis1.controller.config.control_mode = 3

            self.flip_drive.axis0.controller.config.input_mode = InputMode.TRAP_TRAJ
            self.flip_drive.axis1.controller.config.input_mode = InputMode.TRAP_TRAJ
            self.get_logger().info('Change mode to position')
            self.flip_cur_mode = "pos"




    def car_run(self, data):
        
        if self.joy_status:
            self.joy_msg_control()
        
        else:

            if self.car_cur_mode == "pos":          #trajectory Control  #상대 위치 설정
                self.car_drive.axis0.controller.input_pos = data[1] + self.pos_axis0_offset
                self.car_drive.axis1.controller.input_pos = data[0] + self.pos_axis1_offset
            # self.get_logger().info(f'Position control set : axis0 = {data[0]}, axis1 = {data[1]}')
             
            elif self.car_cur_mode == "vel":                            # Ramped Velocity Control  #속도 조절
                self.car_drive.axis0.controller.input_vel = -data[1] #RIGHT
                self.car_drive.axis1.controller.input_vel = data[0] #LEFT
            # self.get_logger().info(f'Velocity control set : axis0 = {data[0]}, axis1 = {data[1]}')
            else : 
                self.get_logger().fatal(f'Invalid mode!! reset to vel mode')
                self.mode = "vel"
            self.encoder_car_check()



    def flip_run(self, data):
        
        if self.joy_status:
            self.joy_msg_control()
        
        else:

            if self.flip_cur_mode == "pos":          #trajectory Control  #상대 위치 설정
                self.flip_drive.axis0.controller.input_pos = data[1] + self.pos_axis0_offset
                self.flip_drive.axis1.controller.input_pos = data[0] + self.pos_axis1_offset
            # self.get_logger().info(f'Position control set : axis0 = {data[0]}, axis1 = {data[1]}')
             
            elif self.flip_cur_mode == "vel":                            # Ramped Velocity Control  #속도 조절
                self.flip_drive.axis0.controller.input_vel = -data[1] #RIGHT
                self.flip_drive.axis1.controller.input_vel = data[0] #LEFT
            # self.get_logger().info(f'Velocity control set : axis0 = {data[0]}, axis1 = {data[1]}')
            else : 
                self.get_logger().fatal(f'Invalid mode!! reset to vel mode')
                self.mode = "vel"
            self.encoder_flip_check()

########################### Motor Encoder Feedback ###########################
    def encoder_car_check(self):
        self.pos_axis0 = self.car_drive.axis0.encoder.pos_estimate
        self.pos_axis1 = self.car_drive.axis1.encoder.pos_estimate
        # self.get_logger().info('Encoder positions: axis0 = {}, axis1 = {}'.format(self.pos_axis0, self.pos_axis1))


    def encoder_flip_check(self):
        self.pos_axis0 = self.flip_drive.axis0.encoder.pos_estimate
        self.pos_axis1 = self.flip_drive.axis1.encoder.pos_estimate
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
