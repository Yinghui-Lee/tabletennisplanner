#include "tabletenniszed/adaptive_kalman_filter_3d.hpp"
#include <cmath>
#include <iostream>

namespace tabletenniszed {

AdaptiveKalmanFilter3D::AdaptiveKalmanFilter3D(const Params& params)
    : dt_max_(params.akf_dt_max)
    , fps_(params.akf_fps)
    , dt_base_(1.0 / params.akf_fps)
    , k_drag_(params.k_drag)
    , Ch_(params.ch)
    , Cv_(params.cv)
    , debug_(params.akf_debug)
    , collision_z_(params.akf_collision_z)
    , tennis_table_x_(params.akf_tennis_table_x)
    , tennis_table_y_(params.akf_tennis_table_y)
    , Q_pos_base_(params.akf_q_pos_base)
    , Q_vel_base_(params.akf_q_vel_base)
    , R_pos_base_(params.akf_r_pos)
    , hit_threshold_(params.akf_hit_threshold)
    , gravity_(params.gravity)
    , vel_init_(params.akf_vel_init)
    , p_init_(params.buildAkfPInit())
    , last_timestamp_(0.0)
    , initialized_(false)
{
    // Observation matrix: only observe position (rows 0-2 of state)
    H_.setZero();
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;

    x_.setZero();
    P_.setZero();
}

std::pair<std::optional<Eigen::Vector3d>, std::optional<Eigen::Vector3d>>
AdaptiveKalmanFilter3D::update(
    const Eigen::Vector3d& pos_observed,
    double timestamp,
    double distance)
{
    // First initialization
    if (!initialized_) {
        reset(pos_observed, timestamp);
        if (debug_) {
            std::cout << "[AKF] initialized with pos_observed=["
                      << pos_observed.transpose() << "], timestamp=" << timestamp << std::endl;
        }
        return {std::nullopt, std::nullopt};
    }

    double dt = timestamp - last_timestamp_;

    if (dt <= 0.0) {
        if (debug_) {
            std::cout << "[AKF] Warning: dt <= 0, returning current state" << std::endl;
        }
        return {x_.head<3>(), x_.tail<3>()};
    }

    // dt too large -> reset
    if (dt > dt_max_) {
        if (debug_) {
            std::cout << "[AKF] dt=" << dt << "s > dt_max=" << dt_max_
                      << "s, resetting filter" << std::endl;
        }
        reset(pos_observed, timestamp);
        return {std::nullopt, std::nullopt};
    }

    // Physics-based prediction step
    predict(dt);

    // Detect return hit: predicted x moved significantly forward but observed x went backward
    if ((x_(0) - pos_observed(0)) > hit_threshold_ && x_(3) > 0.0) {
        if (debug_) {
            std::cout << "[AKF] Detected return hit: x_pred=" << x_(0)
                      << " - x_obs=" << pos_observed(0) << " > threshold="
                      << hit_threshold_ << ", vx=" << x_(3) << std::endl;
        }
        reset(pos_observed, timestamp);
        return {std::nullopt, std::nullopt};
    }

    // Kalman update step
    Eigen::Vector3d z = pos_observed;
    Eigen::Vector3d y = z - H_ * x_;

    Eigen::Matrix3d S = H_ * P_ * H_.transpose() + buildR(distance);
    Eigen::Matrix<double, 6, 3> K = P_ * H_.transpose() * S.inverse();

    x_ = x_ + K * y;
    Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
    P_ = (I - K * H_) * P_;

    last_timestamp_ = timestamp;
    return {x_.head<3>(), x_.tail<3>()};
}

void AdaptiveKalmanFilter3D::reset(
    const std::optional<Eigen::Vector3d>& pos,
    const std::optional<double>& timestamp)
{
    initialized_ = false;
    x_.setZero();
    P_.setZero();

    if (pos.has_value() && timestamp.has_value()) {
        x_.head<3>() = pos.value();
        x_.tail<3>() = vel_init_;
        P_ = p_init_;
        last_timestamp_ = timestamp.value();
        initialized_ = true;
    }
}

void AdaptiveKalmanFilter3D::predict(double dt) {
    double px = x_(0), py = x_(1), pz = x_(2);
    double vx = x_(3), vy = x_(4), vz = x_(5);

    double v_norm = std::sqrt(vx*vx + vy*vy + vz*vz) + 1e-8;

    // Quadratic drag + gravity
    double ax = -k_drag_ * v_norm * vx;
    double ay = -k_drag_ * v_norm * vy;
    double az = -k_drag_ * v_norm * vz - gravity_;

    // Euler integration: p += v*dt + 0.5*a*dt^2; v += a*dt
    x_(0) = px + vx * dt + 0.5 * ax * dt * dt;
    x_(1) = py + vy * dt + 0.5 * ay * dt * dt;
    x_(2) = pz + vz * dt + 0.5 * az * dt * dt;
    x_(3) = vx + ax * dt;
    x_(4) = vy + ay * dt;
    x_(5) = vz + az * dt;

    // Table collision: reflect if inside table bounds, z <= collision threshold, vz < 0
    bool in_table = (std::abs(x_(0)) < tennis_table_x_) &&
                    (std::abs(x_(1)) < tennis_table_y_);
    if (in_table && x_(2) <= collision_z_ && x_(5) < 0.0) {
        x_(3) *= Ch_;
        x_(4) *= Ch_;
        x_(5) = -x_(5) * Cv_;
        x_(2) = 2.0 * collision_z_ - x_(2);
        if (debug_) {
            std::cout << "[AKF] Collision bounce: x=" << x_(0) << " y=" << x_(1)
                      << " z=" << x_(2) << std::endl;
        }
    }

    // Compute Jacobian for covariance propagation
    Eigen::Matrix<double, 6, 6> Fj = computeJacobian(dt, vx, vy, vz, v_norm);

    // Covariance predict: P = Fj * P * Fj^T + Q
    Eigen::Matrix<double, 6, 6> Q = buildQ(dt);
    P_ = Fj * P_ * Fj.transpose() + Q;
}

Eigen::Matrix<double, 6, 6> AdaptiveKalmanFilter3D::buildQ(double dt) const {
    double dt_normalized = dt / dt_base_;
    double q_pos = Q_pos_base_ * dt_normalized * dt_normalized;
    double q_vel = Q_vel_base_ * dt_normalized;

    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Zero();
    Q(0,0) = q_pos; Q(1,1) = q_pos; Q(2,2) = q_pos;
    Q(3,3) = q_vel; Q(4,4) = q_vel; Q(5,5) = q_vel;
    return Q;
}

Eigen::Matrix3d AdaptiveKalmanFilter3D::buildR(double distance) const {
    double r_pos = R_pos_base_ * (1.0 + 10.0 * distance);
    return Eigen::Matrix3d::Identity() * r_pos;
}

Eigen::Matrix<double, 6, 6> AdaptiveKalmanFilter3D::computeJacobian(
    double dt, double vx, double vy, double vz, double v_norm) const
{
    Eigen::Matrix<double, 6, 6> Fj = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d B = Eigen::Matrix3d::Identity() * dt;

    double k = k_drag_;
    double v_norm_safe = std::max(v_norm, 1e-6);

    if (k > 0.0 && v_norm_safe > 1e-6) {
        Eigen::Vector3d v_vec(vx, vy, vz);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                double term = v_vec(i) * v_vec(j) / v_norm_safe;
                if (i == j) {
                    A(i, j) = 1.0 - k * dt * (v_norm_safe + term);
                    B(i, j) = dt - 0.5 * k * dt * dt * (v_norm_safe + term);
                } else {
                    A(i, j) = -k * dt * term;
                    B(i, j) = -0.5 * k * dt * dt * term;
                }
            }
        }
    }

    Fj.block<3, 3>(0, 3) = B;
    Fj.block<3, 3>(3, 3) = A;
    return Fj;
}

std::optional<Eigen::Vector3d> AdaptiveKalmanFilter3D::getEstimatedVelocity() const {
    if (!initialized_) return std::nullopt;
    return x_.tail<3>();
}

bool AdaptiveKalmanFilter3D::isInitialized() const {
    return initialized_;
}

} // namespace tabletenniszed
