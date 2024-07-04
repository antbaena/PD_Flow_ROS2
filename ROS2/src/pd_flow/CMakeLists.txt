cmake_minimum_required(VERSION 3.8)
project(pd_flow)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
find_package(pcl_conversions REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

include_directories(
        include
        ${PCL_INCLUDE_DIRS}
)

add_executable(talker src/publisher_member_function.cpp)
ament_target_dependencies(talker rclcpp std_msgs)

add_executable(listener src/subscriber_member_function.cpp)
ament_target_dependencies(listener rclcpp std_msgs)

add_executable(point_cloud_viewer src/point_cloud.cpp)
ament_target_dependencies(point_cloud_viewer
                            "rclcpp"
                            "sensor_msgs"
                            "PCL"
                            "pcl_conversions")

target_link_libraries(point_cloud_viewer
        ${Boost_SYSTEM_LIBRARY}
)


install(TARGETS
        talker
        listener
        point_cloud_viewer
        DESTINATION lib/${PROJECT_NAME})

ament_package()
