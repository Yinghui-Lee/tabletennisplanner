cmake_minimum_required(VERSION 3.14)
project(tabletenniszed VERSION 1.0.0 LANGUAGES CXX)

# ============================================================
# C++ standard
# ============================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Find ROS2 packages
# ============================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ============================================================
# Find Eigen3
# ============================================================
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ============================================================
# Find nlohmann_json (for UDP JSON parsing)
# ============================================================
find_package(nlohmann_json REQUIRED)

# ============================================================
# Include directories
# ============================================================
include_directories(include)

# ============================================================
# Library: core algorithms (no ROS dependency)
# ============================================================
add_library(tt_core STATIC
    src/kalman_filter_3d.cpp
    src/adaptive_kalman_filter_3d.cpp
    src/ball_predictor.cpp
    src/planner.cpp
)

target_include_directories(tt_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tt_core
    Eigen3::Eigen
)

# ============================================================
# Library: ZED receiver (ROS2 + network)
# ============================================================
add_library(tt_zed_receiver STATIC
    src/zed_receiver.cpp
)

target_include_directories(tt_zed_receiver PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(tt_zed_receiver
    tt_core
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

ament_target_dependencies(tt_zed_receiver
    rclcpp
    geometry_msgs
)

# ============================================================
# Executable: main table tennis node
# ============================================================
add_executable(table_tennis_node
    src/table_tennis_node.cpp
)

target_link_libraries(table_tennis_node
    tt_core
    tt_zed_receiver
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

ament_target_dependencies(table_tennis_node
    rclcpp
    geometry_msgs
    std_msgs
)

add_executable(test_zed_udp tools/test_zed_udp.cpp)
target_link_libraries(test_zed_udp PRIVATE nlohmann_json::nlohmann_json)

# ============================================================
# Install targets
# Static libraries: install to lib/ for potential downstream reuse
# Executable: install to lib/${PROJECT_NAME}/ (ROS2 convention)
# ============================================================
install(TARGETS tt_core tt_zed_receiver
    ARCHIVE DESTINATION lib
)

install(TARGETS table_tennis_node test_zed_udp
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

# ============================================================
# Finalize
# ============================================================
ament_package()
