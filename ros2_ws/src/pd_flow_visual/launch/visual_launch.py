import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(  
                package="pd_flow_visual",
                executable="display_combined_image",
                name="combined_image",
                output="screen",
               
            ),
        Node(  
                package="pd_flow_visual",
                executable="display_color_from_speed",
                name="color_from_speed",
                output="screen",
                
            ),
    ])