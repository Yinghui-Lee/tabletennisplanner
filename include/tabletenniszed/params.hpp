#pragma once

#include <Eigen/Dense>
#include <string>

namespace tabletenniszed {

/**
 * Runtime hyperparameters for the table tennis system.
 *
 * Default values match the original compile-time constants.
 * Load from the ROS2 parameter server via loadParamsFromNode()
 * defined in table_tennis_node.cpp.
 */
struct Params {
    // ============================================================
    // Communication
    // ============================================================
    std::string zed_udp_host        = "172.16.2.101";
    int         zed_udp_port        = 8888;
    int         planning_update_rate = 100;

    // ============================================================
    // Table dimensions (meters)
    // ============================================================
    double table_length = 2.74;
    double table_width  = 1.525;
    double table_height = 0.76;

    // ============================================================
    // Physics
    // ============================================================
    double gravity           = 9.8;   // Used by AKF
    double gravity_predictor = 9.81;  // Used by BallPredictor and Planner
    double k_drag            = 0.20;
    double ch                = 0.85;  // Horizontal restitution
    double cv                = 0.93;  // Vertical restitution

    // ============================================================
    // Marker filter (table frame, meters)
    // ============================================================
    double marker_filter_x_min   = -1.7;
    double marker_filter_x_max   =  1.5;
    double marker_filter_y_range =  0.8;
    double marker_filter_z_min   = -0.2;
    double marker_filter_z_max   =  1.2;

    // ============================================================
    // Strike region (origin frame, meters)
    // ============================================================
    double          strike_region_x_min    =  0.4;
    double          strike_region_x_max    =  0.5;
    double          strike_region_y_min    = -0.75;
    double          strike_region_y_max    =  0.75;
    double          strike_region_z_min    =  0.85;
    double          strike_region_z_max    =  1.4;
    Eigen::Vector3d strike_region_center   = {0.45, 0.0, 1.0};

    // ============================================================
    // Default output values (origin frame)
    // ============================================================
    Eigen::Vector3d default_pos_origin      = {0.5, 0.0, 1.0};
    Eigen::Vector3d default_vel_origin      = {0.0, 0.0, 0.0};
    Eigen::Vector3d default_racket_normal   = {0.9319, 0.0061, 0.3628};
    Eigen::Vector3d default_racket_velocity = {1.8866, 0.0123, 0.7345};

    // ============================================================
    // Planning control
    // ============================================================
    double default_time_to_strike  = -0.5;
    double time_to_strike_min      = -0.5;
    double time_to_strike_max      =  0.6;
    double start_detect_time_min   =  0.3;
    double start_detect_time_max   =  0.8;
    double prediction_time_window  =  0.2;
    double prediction_time_min     =  0.0;
    double prediction_time_max     =  0.5;
    double prediction_time_step    =  0.005;
    double t_flight_default        =  0.5;
    double racket_restitution      =  0.4;

    // ============================================================
    // Incoming ball velocity limits
    // ============================================================
    double ball_vel_x_min = -7.0;
    double ball_vel_x_max = -1.0;
    double ball_vel_y_min = -2.0;
    double ball_vel_y_max =  2.0;
    double ball_vel_z_min = -1.0;
    double ball_vel_z_max =  2.0;

    double hit_ball_vel_x_min = -4.0;
    double hit_ball_vel_x_max = -1.0;
    double hit_ball_vel_y_min = -1.0;
    double hit_ball_vel_y_max =  1.0;
    double hit_ball_vel_z_min = -1.0;
    double hit_ball_vel_z_max =  1.0;

    // ============================================================
    // Standard Kalman filter
    // ============================================================
    bool   kalman_enable          = true;
    double kalman_q_pos_x         = 0.01;
    double kalman_q_pos_y         = 0.01;
    double kalman_q_pos_z         = 0.01;
    double kalman_q_vel_x         = 0.1;
    double kalman_q_vel_y         = 0.1;
    double kalman_q_vel_z         = 0.1;
    double kalman_r_pos_x         = 0.000025;
    double kalman_r_pos_y         = 0.000025;
    double kalman_r_pos_z         = 0.0001;
    double kalman_p0_pos_x        = 0.1;
    double kalman_p0_pos_y        = 0.1;
    double kalman_p0_pos_z        = 0.1;
    double kalman_p0_vel_x        = 1.0;
    double kalman_p0_vel_y        = 1.0;
    double kalman_p0_vel_z        = 1.0;
    double kalman_dt_max          = 0.1;
    double kalman_reset_threshold = 0.5;

    // ============================================================
    // Adaptive Kalman filter (AKF)
    // ============================================================
    bool            use_adaptive_kalman = true;
    double          akf_q_pos_base      = 2e-4;
    double          akf_q_vel_base      = 2e-3;
    double          akf_r_pos           = 2e-4;
    double          akf_dt_max          = 0.2;
    double          akf_fps             = 60.0;
    double          akf_collision_z     = 0.03;
    bool            akf_debug           = true;
    double          akf_tennis_table_x  = 1.37;
    double          akf_tennis_table_y  = 0.76;
    double          akf_hit_threshold   = 0.2;
    Eigen::Vector3d akf_vel_init        = {-3.0, 0.0, 0.0};
    double          akf_p_init_pos      = 1e-2;  // Initial covariance diagonal (position)
    double          akf_p_init_vel      = 1e2;   // Initial covariance diagonal (velocity)

    // ============================================================
    // Coordinate transforms (loaded from YAML as row-major flat arrays)
    // ============================================================
    Eigen::Matrix4d table_in_world    = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d camera_to_torso   = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d t_origin_to_table = Eigen::Matrix4d::Identity();

    // Derived transforms — computed by loadParamsFromNode() after matrices are loaded
    Eigen::Matrix4d origin_in_world   = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d table_in_origin   = Eigen::Matrix4d::Identity();

    // ============================================================
    // Helper: build the 6×6 AKF initial covariance matrix
    // ============================================================
    Eigen::Matrix<double, 6, 6> buildAkfPInit() const {
        Eigen::Matrix<double, 6, 6> P = Eigen::Matrix<double, 6, 6>::Zero();
        P(0,0) = akf_p_init_pos; P(1,1) = akf_p_init_pos; P(2,2) = akf_p_init_pos;
        P(3,3) = akf_p_init_vel; P(4,4) = akf_p_init_vel; P(5,5) = akf_p_init_vel;
        return P;
    }
};

} // namespace tabletenniszed
