cmake_minimum_required(VERSION 3.8)
project(eu_tests)

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
find_package(CUDA REQUIRED)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${cv_bridge_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

# Añadir archivos CUDA si es necesario (ejemplo)
cuda_add_library(pdflow_cudalib src/pdflow_cudalib.cu)

# Agregar el ejecutable scene_flow_node y sus dependencias
add_executable(scene_flow_node 
  src/scene_flow_node.cpp 
  src/scene_flow_impair.cpp 
)
ament_target_dependencies(scene_flow_node
  rclcpp
  sensor_msgs
  cv_bridge
  OpenCV
  pd_flow_msgs
)
target_link_libraries(scene_flow_node
  pdflow_cudalib  # Si se necesita CUDA
  ${OpenCV_LIBRARIES}
  ${cv_bridge_LIBRARIES}
  ${CUDA_LIBRARIES}  # Si se necesita CUDA
)

# Instalación de ejecutables
install(TARGETS
  scene_flow_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
