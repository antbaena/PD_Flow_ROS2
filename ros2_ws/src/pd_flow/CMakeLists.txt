cmake_minimum_required(VERSION 3.8)
project(pd_flow)

# Requerimientos del compilador
if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Buscar paquetes necesarios
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV  REQUIRED)
find_package(pd_flow_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)

# Definir los directorios de inclusión
include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# Definir los ejecutables y sus dependencias
add_executable(image_combiner src/image_combiner_node.cpp)
ament_target_dependencies(image_combiner
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  pd_flow_msgs
)

add_executable(image_viewer src/image_viewer_node.cpp)
ament_target_dependencies(image_viewer
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  pd_flow_msgs
)

add_executable(VectorFieldPublisher src/VectorFieldPublisher.cpp)
ament_target_dependencies(VectorFieldPublisher
  rclcpp
  visualization_msgs
)


# Configuración de CUDA
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode arch=compute_61,code=sm_61")

cuda_add_executable(pd_flow
  src/pd_flow_node.cpp
  src/scene_flow_visualization.cpp
  src/pdflow_cudalib.cu
)

set_target_properties(pd_flow PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

ament_target_dependencies(pd_flow
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  pd_flow_msgs
  Eigen3
  visualization_msgs
)

target_link_libraries(pd_flow
  ${OpenCV_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${CUDA_LIBRARIES}
)

# Instalación de ejecutables
install(TARGETS
pd_flow
VectorFieldPublisher
  image_combiner
  image_viewer
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
