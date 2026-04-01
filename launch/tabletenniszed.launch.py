"""
Launch file for the Table Tennis ZED prediction node.

Usage:
    ros2 launch tabletenniszed tabletenniszed.launch.py
    ros2 launch tabletenniszed tabletenniszed.launch.py udp_host:=192.168.1.10 udp_port:=8888
    ros2 launch tabletenniszed tabletenniszed.launch.py params_file:=/path/to/my_params.yaml

All hyperparameters (filter coefficients, coordinate transforms, etc.) are loaded
from config/params.yaml at runtime.  Edit that file and re-launch to change them
without recompiling.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("tabletenniszed")

    # ----------------------------------------------------------------
    # Declare launch arguments
    # ----------------------------------------------------------------
    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(pkg_share, "config", "params.yaml"),
        description="Path to the YAML parameter file",
    )

    udp_host_arg = DeclareLaunchArgument(
        "udp_host",
        default_value="172.16.2.101",
        description="UDP host address (overrides value in params_file)",
    )

    udp_port_arg = DeclareLaunchArgument(
        "udp_port",
        default_value="8888",
        description="UDP port (overrides value in params_file)",
    )

    planning_rate_arg = DeclareLaunchArgument(
        "planning_rate",
        default_value="100",
        description="Planning loop update rate in Hz (overrides value in params_file)",
    )

    # ----------------------------------------------------------------
    # Table tennis predictor node
    # ----------------------------------------------------------------
    table_tennis_node = Node(
        package="tabletenniszed",
        executable="table_tennis_node",
        name="table_tennis_predictor",
        output="screen",
        parameters=[
            # 1. Load all params from YAML file
            LaunchConfiguration("params_file"),
            # 2. Command-line overrides (take priority over YAML)
            {
                "udp_host":     LaunchConfiguration("udp_host"),
                "udp_port":     LaunchConfiguration("udp_port"),
                "planning_rate": LaunchConfiguration("planning_rate"),
            },
        ],
        remappings=[
            # Example remappings (uncomment to override topic names):
            # ("predicted_ball_position", "/robot/ball_position"),
            # ("predicted_ball_velocity", "/robot/ball_velocity"),
            # ("predicted_racket_normal", "/robot/racket_normal"),
            # ("predicted_racket_velocity", "/robot/racket_velocity"),
            # ("torso_pose_origin_zed", "/robot/torso_pose"),
        ],
    )

    return LaunchDescription([
        params_file_arg,
        udp_host_arg,
        udp_port_arg,
        planning_rate_arg,
        table_tennis_node,
    ])
