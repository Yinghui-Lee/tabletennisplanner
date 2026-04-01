#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <Eigen/Dense>
#include <thread>
#include <mutex>
#include <deque>
#include <atomic>
#include <optional>
#include <memory>
#include <string>

#include "tabletenniszed/params.hpp"
#include "tabletenniszed/kalman_filter_3d.hpp"
#include "tabletenniszed/adaptive_kalman_filter_3d.hpp"

namespace tabletenniszed {

// ============================================================
// Data structures
// ============================================================

/** Latest processed ball state, ready for consumption by the planning node. */
struct BallData {
    Eigen::Vector3d pos_camera;     ///< Ball position in camera frame
    Eigen::Vector3d pos_table;      ///< Kalman-filtered ball position in table frame
    Eigen::Vector3d vel_table;      ///< Estimated ball velocity in table frame
    Eigen::Vector3d pos_table_raw;  ///< Raw (unfiltered) ball position in table frame
    Eigen::Vector3d pos_world;      ///< Ball position in world frame
    double timestamp;               ///< Data timestamp (seconds)
};

/** Intermediate structure from UDP packet parsing. */
struct ParsedUdpData {
    Eigen::Vector3d ball_camera;
    bool ball_camera_valid;
    Eigen::Vector3d ball_point_table;
    bool ball_table_valid;
    bool apriltag_pose_valid;
    std::optional<Eigen::Vector3d> camera_pos_in_table;
    std::optional<Eigen::Matrix3d> camera_orientation_in_table;
    double timestamp;
};

// ============================================================
// ZedReceiver class
// ============================================================

/**
 * Receives ZED camera data over UDP, filters it, and makes it available
 * to the planning node.
 *
 * Architecture: three background threads
 *   1. UDP receive thread:    receives and JSON-parses raw packets
 *   2. Ball process thread:   60 Hz — filters ball detections, runs Kalman filter
 *   3. Torso process thread:  60 Hz — computes torso pose from AprilTag, publishes to ROS2
 */
class ZedReceiver {
public:
    /**
     * @param node   Shared pointer to the owning ROS2 node (used for publishing)
     * @param params Runtime parameters (network addresses, filter coefficients, etc.)
     */
    ZedReceiver(rclcpp::Node::SharedPtr node, const Params& params);

    ~ZedReceiver();

    /** Bind UDP socket and launch background threads. */
    void start();

    /** Signal threads to stop and join them. */
    void stop();

    /**
     * Thread-safe access to the latest validated ball data.
     * @return Latest BallData, or nullopt if no valid data has arrived yet.
     */
    std::optional<BallData> getLatestBallPosTable();

private:
    // ---- Thread entry points ----
    void udpReceiveLoop();
    void processBallDataLoop();
    void processTorsoDataLoop();

    // ---- Data processing ----
    std::optional<ParsedUdpData> parseUdpData(const char* buf, int len);
    void processBallData(const ParsedUdpData& data);
    void processTorsoData(const ParsedUdpData& data);

    // ---- ROS2 publishing ----
    void publishTorsoPose(const Eigen::Vector3d& position,
                          const Eigen::Vector4d& quaternion_xyzw);

    // ---- Static math helpers ----
    static Eigen::Matrix4d invertTransform(const Eigen::Matrix4d& T);
    static Eigen::Vector4d matrixToQuat(const Eigen::Matrix3d& m);
    static Eigen::Vector4d normalizeQuaternion(const Eigen::Vector4d& q, double eps = 1e-8);

    // ---- ROS2 ----
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr torso_pub_;
    std::string frame_id_origin_;

    // ---- Network ----
    std::string udp_host_;
    int         udp_port_;
    int         socket_fd_;
    int         process_update_rate_;

    // ---- Threads ----
    std::thread      receive_thread_;
    std::thread      ball_process_thread_;
    std::thread      torso_process_thread_;
    std::atomic<bool> running_;

    // ---- Shared data between receive and process threads ----
    std::deque<ParsedUdpData> parsed_data_queue_;   // maxlen = 2
    std::mutex                queue_lock_;

    // ---- Latest ball data (output side) ----
    std::optional<BallData> ball_data_;
    std::mutex               ball_data_lock_;

    // ---- Per-thread last-seen timestamps ----
    std::optional<double> last_ball_timestamp_;
    std::optional<double> last_torso_timestamp_;

    // ---- Frequency statistics ----
    int    zed_update_count_;
    double last_print_time_;
    std::mutex              zed_freq_lock_;
    std::optional<double>   last_recv_timestamp_;

    // ---- Coordinate transforms (precomputed) ----
    Eigen::Matrix4d table_in_world_;
    Eigen::Matrix4d origin_in_world_;
    Eigen::Matrix4d camera_to_torso_;
    Eigen::Matrix4d table_in_origin_;

    // ---- Spatial filter bounds ----
    double marker_filter_x_min_;
    double marker_filter_x_max_;
    double marker_filter_y_range_;
    double marker_filter_z_min_;
    double marker_filter_z_max_;
    double table_length_half_;
    double table_width_half_;

    // ---- Kalman filters ----
    std::unique_ptr<KalmanFilter3D>         kalman_filter_;
    std::unique_ptr<AdaptiveKalmanFilter3D> adaptive_kalman_filter_;
    bool use_adaptive_kalman_;
};

} // namespace tabletenniszed
