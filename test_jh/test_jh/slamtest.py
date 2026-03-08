#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import subprocess
import signal
import time

class D435iNav2Node(Node):
    def __init__(self):
        super().__init__('d435i_nav2_node')
        self.get_logger().info("Starting D435i Camera + RGBD Odometry + Nav2...")

        self.processes = []

        # 1️⃣ Realsense 카메라 + IMU
        camera_cmd = [
            "ros2", "launch", "realsense2_camera", "rs_launch.py",
            "enable_gyro:=true",
            "enable_accel:=true",
            "unite_imu_method:=1",
            "enable_sync:=true"
        ]
        self.start_process(camera_cmd, "D435i Camera")
        time.sleep(3)

        # 2️⃣ TF: map → odom (static) 
        tf_map_odom_cmd = [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "0","0","0","0","0","0",
            "map","odom"
        ]
        self.start_process(tf_map_odom_cmd, "Map → Odom TF")
        time.sleep(0.5)

        # ⛔️ [삭제됨] odom → base_link static TF
        # 이 부분은 FakeOdometryNode가 동적으로 퍼블리시하도록 대체

        # 3️⃣ TF: base_link → camera_link (static)
        tf_camera_cmd = [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "0.2","0","0","0","0","0",
            "base_link","camera_link"
        ]
        self.start_process(tf_camera_cmd, "Base → Camera TF")
        time.sleep(0.5)

        # 4️⃣ RGBD Odometry (odom → base_link, pose 추정용)
        odometry_cmd = [
            "ros2", "run", "rtabmap_odom", "rgbd_odometry",
            "--ros-args",
            "-p", "frame_id:=odom",
            "-p", "child_frame_id:=base_link",
            "-p", "subscribe_depth:=true",
            "-p", "subscribe_odom_info:=true",
            "-p", "approx_sync:=true",
            "-p", "publish_tf:=true",   # ⚠️ TF는 FakeOdometryNode에서 발행
            "-p", "qos:=1",
            "--remap", "imu:=/imu/data",
            "--remap", "rgb/image:=/camera/camera/color/image_raw",
            "--remap", "rgb/camera_info:=/camera/camera/color/camera_info",
            "--remap", "depth/image:=/camera/camera/realigned_depth_to_color/image_raw"
        ]
        self.start_process(odometry_cmd, "RGBD Odometry")
        time.sleep(3)

        # 5️⃣ Fake Odometry Node (Nav2 <-> /cmd_vel -> /odom, TF)
        # fake_odom_cmd = [
        #     "ros2", "run", "test_jh", "redfinder2"
        # ]
        # self.start_process(fake_odom_cmd, "Fake Odometry Node")
        # time.sleep(0.5)

        # 6️⃣ IMU Filter
        imu_cmd = [
            "ros2", "run", "imu_filter_madgwick", "imu_filter_madgwick_node",
            "--ros-args",
            "-p", "use_mag:=false",
            "-p", "world_frame:=map",
            "-p", "publish_tf:=false",
            "--remap", "imu/data_raw:=/camera/camera/imu"
        ]
        self.start_process(imu_cmd, "IMU Filter")
        time.sleep(1)

        # 7️⃣ Nav2 Bringup
        nav2_cmd = [
            "ros2", "launch", "nav2_bringup", "bringup_launch.py",
            "map:=/home/ljh/.ros/rtabmap.yaml"
        ]
        self.start_process(nav2_cmd, "Nav2 Bringup")
        time.sleep(3)

        # 8️⃣ RViz
        rviz_cmd = [
            "ros2", "run", "rviz2", "rviz2",
            "-d", "/opt/ros/humble/share/nav2_bringup/rviz/nav2_default_view.rviz"
        ]
        self.start_process(rviz_cmd, "RViz")

        self.get_logger().info("✅ All processes started. Use RViz to set initial pose & goal.")

    def start_process(self, cmd, name="Process"):
        try:
            p = subprocess.Popen(cmd)
            self.processes.append(p)
            self.get_logger().info(f"{name} started.")
            time.sleep(0.5)
        except Exception as e:
            self.get_logger().error(f"Failed to start {name}: {e}")

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