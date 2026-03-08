#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import subprocess
import signal
import time
import os

class D435iNav2Node(Node):
    def __init__(self):
        super().__init__('d435i_nav2_node')
        self.get_logger().info("Starting D435i Camera + RGBD Odometry + Nav2...")

        self.processes = []
        self.env = os.environ.copy()

        # 순차 실행
        self.start_camera()
        self.start_static_tf_map_odom()
        self.start_static_tf_camera()
        self.start_rgbd_odometry()
        self.start_imu_filter()
        self.start_nav2()
        self.start_rviz()
        self.get_logger().info("✅ All processes started.")

    def wait_for_process(self, name, check_cmd=None, timeout=10):
        if not check_cmd:
            time.sleep(2)
            return True
        self.get_logger().info(f"Waiting for {name} to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                result = subprocess.run(check_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
                if result.stdout.decode().strip():
                    self.get_logger().info(f"{name} is ready!")
                    return True
            except Exception:
                pass
            time.sleep(0.5)
        self.get_logger().warn(f"{name} readiness check timed out!")
        return False

    def start_process(self, cmd, name="Process"):
        try:
            p = subprocess.Popen(cmd, env=self.env)
            self.processes.append(p)
            self.get_logger().info(f"{name} started.")
            return p
        except Exception as e:
            self.get_logger().error(f"Failed to start {name}: {e}")
            return None

    def start_camera(self):
        cmd = [
            "ros2", "launch", "realsense2_camera", "rs_launch.py",
            "enable_gyro:=true","enable_accel:=true",
            "unite_imu_method:=1","enable_sync:=true"
        ]
        self.start_process(cmd, "D435i Camera")
        self.wait_for_process("D435i Camera", check_cmd=["ros2","topic","list"])

    def start_static_tf_map_odom(self):
        cmd = ["ros2","run","tf2_ros","static_transform_publisher",
               "0","0","0","0","0","0","map","odom"]
        self.start_process(cmd, "Map → Odom TF")
        time.sleep(1)

    def start_static_tf_camera(self):
        cmd = ["ros2","run","tf2_ros","static_transform_publisher",
               "0.2","0","0","0","0","0","base_link","camera_link"]
        self.start_process(cmd, "Base → Camera TF")
        time.sleep(1)

    def start_rgbd_odometry(self):
        cmd = [
            "ros2","run","rtabmap_odom","rgbd_odometry",
            "--ros-args",
            "-p","frame_id:=odom",
            "-p","child_frame_id:=camera_link",   # 🔹 base_link 아님
            "-p","subscribe_depth:=true",
            "-p","subscribe_odom_info:=true",
            "-p","approx_sync:=true",
            "-p","publish_tf:=false",             # 🔹 odom → base_link 발행 끔
            "-p","qos:=1",
            "--remap","imu:=/camera/camera/imu",
            "--remap","rgb/image:=/camera/camera/color/image_raw",
            "--remap","rgb/camera_info:=/camera/camera/color/camera_info",
            "--remap","depth/image:=/camera/camera/aligned_depth_to_color/image_raw"
        ]
        self.start_process(cmd, "RGBD Odometry")
        self.wait_for_process("RGBD Odometry", check_cmd=["ros2","topic","list"])

    def start_imu_filter(self):
        cmd = [
            "ros2","run","imu_filter_madgwick","imu_filter_madgwick_node",
            "--ros-args",
            "-p","use_mag:=false",
            "-p","world_frame:=map",
            "-p","publish_tf:=false",
            "--remap","imu/data_raw:=/camera/camera/imu"
        ]
        self.start_process(cmd, "IMU Filter")
        self.wait_for_process("IMU Filter", check_cmd=["ros2","topic","list"])

    def start_nav2(self):
        cmd = ["ros2","launch","nav2_bringup","bringup_launch.py",
               "map:=/home/ljh/.ros/rtabmap.yaml"]
        self.start_process(cmd, "Nav2 Bringup")
        self.wait_for_process("Nav2 Bringup", timeout=15)

    def start_rviz(self):
        cmd = ["ros2","run","rviz2","rviz2",
               "-d","/opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz"]
        self.start_process(cmd, "RViz")
        time.sleep(1)

    def destroy_node(self):
        self.get_logger().info("Shutting down all subprocesses...")
        for p in self.processes:
            try:
                p.send_signal(signal.SIGINT)
                p.wait()
            except Exception:
                pass
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = D435iNav2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
