#include "tabletenniszed/zed_receiver.hpp"

#include <nlohmann/json.hpp>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <sys/time.h>
#include <cerrno>
#include <cstring>
#include <cmath>
#include <chrono>
#include <iostream>

namespace tabletenniszed {

// ============================================================
// Construction / Destruction
// ============================================================

ZedReceiver::ZedReceiver(rclcpp::Node::SharedPtr node, const Params& params)
    : node_(node)
    , frame_id_origin_("origin_frame_w")
    , udp_host_(params.zed_udp_host)
    , udp_port_(params.zed_udp_port)
    , socket_fd_(-1)
    , running_(false)
    , zed_update_count_(0)
    , last_print_time_(0.0)
    , process_update_rate_(static_cast<int>(params.akf_fps))
    , use_adaptive_kalman_(params.use_adaptive_kalman)
{
    // Precompute coordinate transforms
    table_in_world_  = params.table_in_world;
    origin_in_world_ = params.origin_in_world;
    camera_to_torso_ = params.camera_to_torso;
    table_in_origin_ = params.table_in_origin;

    // Spatial filter bounds (stored for processBallData)
    marker_filter_x_min_   = params.marker_filter_x_min;
    marker_filter_x_max_   = params.marker_filter_x_max;
    marker_filter_y_range_ = params.marker_filter_y_range;
    marker_filter_z_min_   = params.marker_filter_z_min;
    marker_filter_z_max_   = params.marker_filter_z_max;
    table_length_half_     = params.table_length / 2.0;
    table_width_half_      = params.table_width  / 2.0;

    // Create torso pose publisher
    torso_pub_ = node_->create_publisher<geometry_msgs::msg::PoseStamped>(
        "torso_pose_origin_zed", 10);

    // Initialize Kalman filter
    if (params.kalman_enable) {
        if (params.use_adaptive_kalman) {
            adaptive_kalman_filter_ = std::make_unique<AdaptiveKalmanFilter3D>(params);
            RCLCPP_INFO(node_->get_logger(), "[ZedReceiver] Adaptive Kalman Filter (AKF) initialized");
        } else {
            kalman_filter_ = std::make_unique<KalmanFilter3D>(
                params.kalman_q_pos_x, params.kalman_q_pos_y, params.kalman_q_pos_z,
                params.kalman_q_vel_x, params.kalman_q_vel_y, params.kalman_q_vel_z,
                params.kalman_r_pos_x, params.kalman_r_pos_y, params.kalman_r_pos_z,
                params.kalman_p0_pos_x, params.kalman_p0_pos_y, params.kalman_p0_pos_z,
                params.kalman_p0_vel_x, params.kalman_p0_vel_y, params.kalman_p0_vel_z,
                params.kalman_dt_max, params.kalman_reset_threshold
            );
            RCLCPP_INFO(node_->get_logger(), "[ZedReceiver] Standard Kalman Filter (KF) initialized");
        }
    }
}

ZedReceiver::~ZedReceiver() {
    stop();
}

// ============================================================
// Start / Stop
// ============================================================

void ZedReceiver::start() {
    RCLCPP_INFO(node_->get_logger(), "[ZedReceiver] Starting...");

    if (running_.load()) {
        RCLCPP_WARN(node_->get_logger(), "[ZedReceiver] Already running");
        return;
    }

    // Create UDP socket
    socket_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
        RCLCPP_ERROR(node_->get_logger(),
            "[ZedReceiver] Failed to create socket: %s", strerror(errno));
        return;
    }

    // Allow address reuse
    int optval = 1;
    setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    // 1-second receive timeout
    struct timeval timeout;
    timeout.tv_sec  = 1;
    timeout.tv_usec = 0;
    setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    // Bind
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port   = htons(static_cast<uint16_t>(udp_port_));
    if (inet_pton(AF_INET, udp_host_.c_str(), &server_addr.sin_addr) <= 0) {
        RCLCPP_ERROR(node_->get_logger(),
            "[ZedReceiver] Invalid UDP host address: %s", udp_host_.c_str());
        close(socket_fd_);
        socket_fd_ = -1;
        return;
    }

    if (bind(socket_fd_, reinterpret_cast<struct sockaddr*>(&server_addr),
             sizeof(server_addr)) < 0) {
        RCLCPP_ERROR(node_->get_logger(),
            "[ZedReceiver] UDP bind failed %s:%d — %s",
            udp_host_.c_str(), udp_port_, strerror(errno));
        close(socket_fd_);
        socket_fd_ = -1;
        return;
    }

    // Reset stats
    {
        std::lock_guard<std::mutex> lk(zed_freq_lock_);
        zed_update_count_ = 0;
        auto now = std::chrono::system_clock::now();
        last_print_time_ = std::chrono::duration<double>(
            now.time_since_epoch()).count();
        last_recv_timestamp_.reset();
    }
    last_ball_timestamp_.reset();
    last_torso_timestamp_.reset();

    running_.store(true);

    receive_thread_       = std::thread(&ZedReceiver::udpReceiveLoop,      this);
    ball_process_thread_  = std::thread(&ZedReceiver::processBallDataLoop,  this);
    torso_process_thread_ = std::thread(&ZedReceiver::processTorsoDataLoop, this);

    RCLCPP_INFO(node_->get_logger(),
        "[ZedReceiver] All threads started (UDP %s:%d)",
        udp_host_.c_str(), udp_port_);
}

void ZedReceiver::stop() {
    running_.store(false);

    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }

    if (receive_thread_.joinable())       receive_thread_.join();
    if (ball_process_thread_.joinable())  ball_process_thread_.join();
    if (torso_process_thread_.joinable()) torso_process_thread_.join();

    RCLCPP_INFO(node_->get_logger(), "[ZedReceiver] All threads stopped");
}

// ============================================================
// Thread 1: UDP receive loop
// ============================================================

void ZedReceiver::udpReceiveLoop() {
    static constexpr int kUdpBufSize = 4096;
    char buf[kUdpBufSize];

    while (running_.load()) {
        ssize_t n = recv(socket_fd_, buf, sizeof(buf) - 1, 0);

        if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;  // Timeout — normal
            }
            if (running_.load()) {
                RCLCPP_WARN(node_->get_logger(),
                    "[ZedReceiver] UDP recv error: %s", strerror(errno));
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            continue;
        }

        buf[n] = '\0';
        auto parsed = parseUdpData(buf, static_cast<int>(n));
        if (!parsed.has_value()) continue;

        {
            std::lock_guard<std::mutex> lk(queue_lock_);
            if (parsed_data_queue_.size() >= 2) {
                parsed_data_queue_.pop_front();
            }
            parsed_data_queue_.push_back(*parsed);
        }
    }
}

// ============================================================
// UDP packet parser
// ============================================================

std::optional<ParsedUdpData> ZedReceiver::parseUdpData(const char* buf, int len) {
    try {
        nlohmann::json j = nlohmann::json::parse(buf, buf + len);

        ParsedUdpData d;

        // Timestamp
        auto now = std::chrono::system_clock::now();
        double sys_time = std::chrono::duration<double>(now.time_since_epoch()).count();
        d.timestamp = j.value("timestamp", sys_time);

        // Ball in camera frame
        auto& bc = j["ball_camera"];
        d.ball_camera = Eigen::Vector3d(
            bc.value("x", 0.0),
            bc.value("y", 0.0),
            bc.value("z", 0.0));
        d.ball_camera_valid = bc.value("valid", false);

        // Ball in table frame
        auto& bt = j["ball_table"];
        d.ball_point_table = Eigen::Vector3d(
            bt.value("x", 0.0),
            bt.value("y", 0.0),
            bt.value("z", 0.0));
        d.ball_table_valid = bt.value("valid", false);

        // Camera pose from AprilTag
        auto& cp = j["camera_pose"];
        d.apriltag_pose_valid = cp.value("valid", false);

        if (d.apriltag_pose_valid) {
            auto& pos = cp["position"];
            d.camera_pos_in_table = Eigen::Vector3d(
                pos.value("x", 0.0),
                pos.value("y", 0.0),
                pos.value("z", 0.0));

            // Rotation matrix as 9-element flat array
            auto& rot = cp["rotation"];
            Eigen::Matrix3d R;
            for (int i = 0; i < 3; ++i)
                for (int j2 = 0; j2 < 3; ++j2)
                    R(i, j2) = rot[i * 3 + j2].get<double>();
            d.camera_orientation_in_table = R;
        }

        // Update frequency statistics
        {
            std::lock_guard<std::mutex> lk(zed_freq_lock_);
            if (!last_recv_timestamp_.has_value() || d.timestamp != *last_recv_timestamp_) {
                ++zed_update_count_;
                last_recv_timestamp_ = d.timestamp;
            }
            double now_sec = sys_time;
            double elapsed = now_sec - last_print_time_;
            if (elapsed >= 1.0) {
                double freq = static_cast<double>(zed_update_count_) / elapsed;
                RCLCPP_DEBUG(node_->get_logger(),
                    "[ZedReceiver] Update freq: %.2f Hz (count: %d)",
                    freq, zed_update_count_);
                zed_update_count_ = 0;
                last_print_time_  = now_sec;
            }
        }

        return d;

    } catch (const std::exception& e) {
        RCLCPP_WARN(node_->get_logger(),
            "[ZedReceiver] Parse error: %s", e.what());
        return std::nullopt;
    }
}

// ============================================================
// Thread 2: Ball data processing loop (60 Hz)
// ============================================================

void ZedReceiver::processBallDataLoop() {
    const auto interval = std::chrono::duration<double>(
        1.0 / process_update_rate_);

    while (running_.load()) {
        auto start = std::chrono::steady_clock::now();

        // Get latest parsed data
        std::optional<ParsedUdpData> latest;
        {
            std::lock_guard<std::mutex> lk(queue_lock_);
            if (!parsed_data_queue_.empty()) {
                latest = parsed_data_queue_.back();
            }
        }

        if (latest.has_value()) {
            double ts = latest->timestamp;
            if (!last_ball_timestamp_.has_value() || ts != *last_ball_timestamp_) {
                processBallData(*latest);
                last_ball_timestamp_ = ts;
            }
        }

        // Rate control
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep_dur = interval - elapsed;
        if (sleep_dur > std::chrono::duration<double>::zero()) {
            std::this_thread::sleep_for(sleep_dur);
        }
    }
}

void ZedReceiver::processBallData(const ParsedUdpData& data) {
    if (!data.ball_table_valid) return;

    Eigen::Vector3d pos_table = data.ball_point_table;
    Eigen::Vector3d pos_camera = data.ball_camera;
    double timestamp = data.timestamp;

    // Transform table -> world for the world-frame record
    Eigen::Vector4d pos_table_h(pos_table(0), pos_table(1), pos_table(2), 1.0);
    Eigen::Vector4d pos_world_h = table_in_world_ * pos_table_h;
    Eigen::Vector3d pos_world = pos_world_h.head<3>();

    // Spatial filter (in table frame)
    bool in_cube =
        (pos_table(0) >= marker_filter_x_min_) &&
        (pos_table(0) <= marker_filter_x_max_) &&
        (std::abs(pos_table(1)) <= marker_filter_y_range_) &&
        (pos_table(2) >= marker_filter_z_min_) &&
        (pos_table(2) <= marker_filter_z_max_);

    if (!in_cube) return;

    // Kalman filtering
    Eigen::Vector3d pos_filtered;
    Eigen::Vector3d vel_filtered;
    bool filter_ok = false;

    double distance = pos_camera.norm();

    if (use_adaptive_kalman_ && adaptive_kalman_filter_) {
        auto [pf, vf] = adaptive_kalman_filter_->update(pos_table, timestamp, distance);
        if (pf.has_value() && vf.has_value()) {
            pos_filtered = *pf;
            vel_filtered = *vf;
            filter_ok = true;
        }
    } else if (kalman_filter_) {
        pos_filtered = kalman_filter_->update(pos_table, timestamp);
        auto vf = kalman_filter_->getEstimatedVelocity();
        if (vf.has_value()) {
            vel_filtered = *vf;
            filter_ok = true;
        }
    }

    if (!filter_ok) return;

    // Reject ball if it is on the table surface (z < 0.08 m while in table bounds)
    if (std::abs(pos_filtered(0)) < table_length_half_ &&
        std::abs(pos_filtered(1)) < table_width_half_ &&
        pos_filtered(2) < 0.08) {
        return;
    }

    // Store filtered data
    {
        std::lock_guard<std::mutex> lk(ball_data_lock_);
        BallData bd;
        bd.pos_camera    = pos_camera;
        bd.pos_table     = pos_filtered;
        bd.vel_table     = vel_filtered;
        bd.pos_table_raw = pos_table;
        bd.pos_world     = pos_world;
        bd.timestamp     = timestamp;
        ball_data_ = bd;
    }
}

// ============================================================
// Thread 3: Torso data processing loop (60 Hz)
// ============================================================

void ZedReceiver::processTorsoDataLoop() {
    const auto interval = std::chrono::duration<double>(
        1.0 / process_update_rate_);

    while (running_.load()) {
        auto start = std::chrono::steady_clock::now();

        std::optional<ParsedUdpData> latest;
        {
            std::lock_guard<std::mutex> lk(queue_lock_);
            if (!parsed_data_queue_.empty()) {
                latest = parsed_data_queue_.back();
            }
        }

        if (latest.has_value()) {
            double ts = latest->timestamp;
            if (!last_torso_timestamp_.has_value() || ts != *last_torso_timestamp_) {
                processTorsoData(*latest);
                last_torso_timestamp_ = ts;
            }
        }

        auto elapsed = std::chrono::steady_clock::now() - start;
        auto sleep_dur = interval - elapsed;
        if (sleep_dur > std::chrono::duration<double>::zero()) {
            std::this_thread::sleep_for(sleep_dur);
        }
    }
}

void ZedReceiver::processTorsoData(const ParsedUdpData& data) {
    if (!data.apriltag_pose_valid) return;
    if (!data.camera_pos_in_table.has_value()) return;
    if (!data.camera_orientation_in_table.has_value()) return;

    // Build camera-in-table homogeneous transform
    Eigen::Matrix4d camera_in_table = Eigen::Matrix4d::Identity();
    camera_in_table.block<3,3>(0,0) = *data.camera_orientation_in_table;
    camera_in_table.block<3,1>(0,3) = *data.camera_pos_in_table;

    // camera -> torso (apply inverse of camera_to_torso)
    Eigen::Matrix4d torso_in_table = camera_in_table * invertTransform(camera_to_torso_);

    // table -> origin
    Eigen::Matrix4d torso_in_origin = table_in_origin_ * torso_in_table;

    Eigen::Vector3d torso_pos = torso_in_origin.block<3,1>(0,3);
    Eigen::Matrix3d torso_rot = torso_in_origin.block<3,3>(0,0);
    Eigen::Vector4d torso_quat = normalizeQuaternion(matrixToQuat(torso_rot));

    publishTorsoPose(torso_pos, torso_quat);
}

// ============================================================
// ROS2 publishing
// ============================================================

void ZedReceiver::publishTorsoPose(const Eigen::Vector3d& position,
                                    const Eigen::Vector4d& quaternion_xyzw) {
    geometry_msgs::msg::PoseStamped msg;
    msg.header.stamp    = node_->now();
    msg.header.frame_id = frame_id_origin_;
    msg.pose.position.x = position(0);
    msg.pose.position.y = position(1);
    msg.pose.position.z = position(2);
    msg.pose.orientation.x = quaternion_xyzw(0);
    msg.pose.orientation.y = quaternion_xyzw(1);
    msg.pose.orientation.z = quaternion_xyzw(2);
    msg.pose.orientation.w = quaternion_xyzw(3);
    torso_pub_->publish(msg);
}

// ============================================================
// Public data access
// ============================================================

std::optional<BallData> ZedReceiver::getLatestBallPosTable() {
    std::lock_guard<std::mutex> lk(ball_data_lock_);
    return ball_data_;
}

// ============================================================
// Static math helpers
// ============================================================

Eigen::Matrix4d ZedReceiver::invertTransform(const Eigen::Matrix4d& T) {
    // For rigid-body transforms: inv = [R^T, -R^T*t; 0, 1]
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    Eigen::Matrix4d T_inv = Eigen::Matrix4d::Identity();
    T_inv.block<3,3>(0,0) = R.transpose();
    T_inv.block<3,1>(0,3) = -R.transpose() * t;
    return T_inv;
}

Eigen::Vector4d ZedReceiver::matrixToQuat(const Eigen::Matrix3d& m) {
    // Shepperd method (returns [x, y, z, w])
    double trace = m.trace();
    double x, y, z, w;

    if (trace > 0.0) {
        double s = std::sqrt(trace + 1.0) * 2.0;  // s = 4*w
        w = 0.25 * s;
        x = (m(2,1) - m(1,2)) / s;
        y = (m(0,2) - m(2,0)) / s;
        z = (m(1,0) - m(0,1)) / s;
    } else if (m(0,0) > m(1,1) && m(0,0) > m(2,2)) {
        double s = std::sqrt(1.0 + m(0,0) - m(1,1) - m(2,2)) * 2.0;  // s = 4*x
        w = (m(2,1) - m(1,2)) / s;
        x = 0.25 * s;
        y = (m(0,1) + m(1,0)) / s;
        z = (m(0,2) + m(2,0)) / s;
    } else if (m(1,1) > m(2,2)) {
        double s = std::sqrt(1.0 + m(1,1) - m(0,0) - m(2,2)) * 2.0;  // s = 4*y
        w = (m(0,2) - m(2,0)) / s;
        x = (m(0,1) + m(1,0)) / s;
        y = 0.25 * s;
        z = (m(1,2) + m(2,1)) / s;
    } else {
        double s = std::sqrt(1.0 + m(2,2) - m(0,0) - m(1,1)) * 2.0;  // s = 4*z
        w = (m(1,0) - m(0,1)) / s;
        x = (m(0,2) + m(2,0)) / s;
        y = (m(1,2) + m(2,1)) / s;
        z = 0.25 * s;
    }

    return Eigen::Vector4d(x, y, z, w);
}

Eigen::Vector4d ZedReceiver::normalizeQuaternion(const Eigen::Vector4d& q, double eps) {
    double norm = q.norm();
    if (norm < eps) {
        return Eigen::Vector4d(0.0, 0.0, 0.0, 1.0);
    }
    return q / norm;
}

} // namespace tabletenniszed
