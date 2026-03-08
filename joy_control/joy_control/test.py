import rclpy
from rclpy.node import Node
import subprocess
import signal
import time
import os

class D435iRtabmapNav2Node(Node):
    def __init__(self):
        super().__init__('d435i_rtabmap_nav2_node')
        self.get_logger().info("Starting D435i + RTAB-Map + Nav2 + RViz nodes...")

        self.processes = []

        # 1️⃣ Realsense D435i Camera
        camera_cmd = [
            "ros2", "launch", "realsense2_camera", "rs_launch.py",
            "enable_gyro:=true",
            "enable_accel:=true",
            "unite_imu_method:=1",
            "enable_sync:=true"
        ]
        self.start_process(camera_cmd, "D435i Camera")
        time.sleep(2)

        # 2️⃣ RGBD Odometry
        odometry_cmd = [
            "ros2", "run", "rtabmap_odom", "rgbd_odometry",
            "--ros-args",
            "-p", "frame_id:=camera_link",
            "-p", "subscribe_depth:=true",
            "-p", "subscribe_odom_info:=true",
            "-p", "approx_sync:=false",
            "-p", "publish_tf:=true",
            "-p", "qos:=1",
            "-p", "use_action_for_goal:=true",
            "--remap", "imu:=/mavros/imu/data",
            "--remap", "rgb/image:=/camera/camera/color/image_raw",
            "--remap", "rgb/camera_info:=/camera/camera/color/camera_info",
            "--remap", "depth/image:=/camera/camera/aligned_depth_to_color/image_raw",
            "--remap", "odom:=/odom/camera"
        ]
        self.start_process(odometry_cmd, "RGBD Odometry")

        # 3️⃣ Point Cloud
        pointcloud_cmd = [
            "ros2", "run", "rtabmap_util", "point_cloud_xyz",
            "--ros-args",
            "-p", "approx_sync:=false",
            "--remap", "depth/image:=/camera/camera/depth/image_rect_raw",
            "--remap", "depth/camera_info:=/camera/camera/depth/camera_info",
            "--remap", "cloud:=/camera/camera/cloud_from_depth"
        ]
        self.start_process(pointcloud_cmd, "Point Cloud")

        # 4️⃣ Aligned Depth
        aligned_depth_cmd = [
            "ros2", "run", "rtabmap_util", "pointcloud_to_depthimage",
            "--ros-args",
            "-p", "decimation:=2",
            "-p", "fixed_frame_id:=camera_link",
            "-p", "fill_holes_size:=1",
            "--remap", "camera_info:=/camera/camera/color/camera_info",
            "--remap", "cloud:=/camera/camera/cloud_from_depth",
            "--remap", "image_raw:=/camera/camera/aligned_depth_to_color/image_raw"
        ]
        self.start_process(aligned_depth_cmd, "Aligned Depth")

        # 5️⃣ IMU Filter
        imu_cmd = [
            "ros2", "run", "imu_filter_madgwick", "imu_filter_madgwick_node",
            "--ros-args",
            "-p", "use_mag:=false",
            "-p", "world_frame:=map",
            "-p", "publish_tf:=false",
            "--remap", "imu/data_raw:=/camera/camera/imu"
        ]
        self.start_process(imu_cmd, "IMU Filter")

        # 6️⃣ IMU TF
        tf_cmd = [
            "ros2", "run", "tf2_ros", "static_transform_publisher",
            "0","0","0","0","0","0",
            "camera_gyro_optical_frame","camera_imu_optical_frame"
        ]
        self.start_process(tf_cmd, "IMU TF")

        # 7️⃣ RTAB-Map SLAM
        rtabmap_cmd = [
            "ros2", "run", "rtabmap_slam", "rtabmap",
            "--ros-args",
            "-p", "frame_id:=camera_link",
            "-p", "subscribe_depth:=true",
            "-p", "subscribe_odom_info:=true",
            "-p", "approx_sync:=false",
            "-p", "publish_tf:=true",
            "-p", "qos:=1",
            "-p", "use_action_for_goal:=true",
            "--remap", "rgb/image:=/camera/camera/color/image_raw",
            "--remap", "rgb/camera_info:=/camera/camera/color/camera_info",
            "--remap", "depth/image:=/camera/camera/aligned_depth_to_color/image_raw",
            "--remap", "odom:=/odom/camera"
        ]
        self.start_process(rtabmap_cmd, "RTAB-Map SLAM")

        # 8️⃣ Nav2 (Autonomous Navigation)
        # 반드시 nav2 params 파일 경로를 정확하게 설정해야 함
        nav2_params_file = os.path.expanduser("~/rtabmap_ws/src/rtabmap_drone_example/param/nav2_params.yaml")
        nav2_cmd = [
            "ros2", "launch", "nav2_bringup", "bringup_launch.py",
            f"map:=/rtabmap/grid_map",
            f"params_file:={nav2_params_file}",
            "use_sim_time:=false"
        ]
        self.start_process(nav2_cmd, "Nav2")

        # 9️⃣ RViz (SLAM + Nav2 시각화)
        rviz_cmd = [
            "ros2", "run", "rviz2", "rviz2",
            "-d", "/opt/ros/humble/share/rtabmap_viz/config/rtabmap.rviz"
        ]
        self.start_process(rviz_cmd, "RViz")

        self.get_logger().info("All nodes including Nav2 started.")

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
    node = D435iRtabmapNav2Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt (SIGINT)")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()