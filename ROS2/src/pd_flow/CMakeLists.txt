cmake_minimum_required(VERSION 3.8)
project(pd_flow)

# Requerimientos del compilador
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif ()

# Buscar paquetes necesarios
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(pd_flow_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

# Definir los directorios de inclusión
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

# Definir los ejecutables y sus dependencias
add_executable(Image_combiner src/Image_combiner.cpp)
ament_target_dependencies(Image_combiner
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  pd_flow_msgs
)

add_executable(PD_flow src/PD_flow.cpp)
ament_target_dependencies(PD_flow
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  pd_flow_msgs
  Eigen3
)

# Enlazar bibliotecas necesarias
target_link_libraries(PD_flow
  ${OpenCV_LIBRARIES}
  ${cv_bridge_LIBRARIES}
)

# Instalación de ejecutables
install(TARGETS
  PD_flow
  Image_combiner
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
