#include "tabletenniszed/kalman_filter_3d.hpp"
#include <cmath>
#include <iostream>

namespace tabletenniszed {

KalmanFilter3D::KalmanFilter3D(
    double q_pos_x, double q_pos_y, double q_pos_z,
    double q_vel_x, double q_vel_y, double q_vel_z,
    double r_pos_x, double r_pos_y, double r_pos_z,
    double p0_pos_x, double p0_pos_y, double p0_pos_z,
    double p0_vel_x, double p0_vel_y, double p0_vel_z,
    double dt_max,
    double reset_threshold)
    : dt_max_(dt_max)
    , reset_threshold_(reset_threshold)
    , last_timestamp_(0.0)
    , initialized_(false)
{
    // Build process noise covariance Q (6x6 diagonal)
    Q_.setZero();
    Q_(0,0) = q_pos_x; Q_(1,1) = q_pos_y; Q_(2,2) = q_pos_z;
    Q_(3,3) = q_vel_x; Q_(4,4) = q_vel_y; Q_(5,5) = q_vel_z;

    // Build observation noise covariance R (3x3 diagonal)
    R_.setZero();
    R_(0,0) = r_pos_x; R_(1,1) = r_pos_y; R_(2,2) = r_pos_z;

    // Build initial covariance P0 (6x6 diagonal)
    P0_.setZero();
    P0_(0,0) = p0_pos_x; P0_(1,1) = p0_pos_y; P0_(2,2) = p0_pos_z;
    P0_(3,3) = p0_vel_x; P0_(4,4) = p0_vel_y; P0_(5,5) = p0_vel_z;

    // Observation matrix H: maps state to position observation
    H_.setZero();
    H_(0,0) = 1.0;  // px
    H_(1,1) = 1.0;  // py
    H_(2,2) = 1.0;  // pz

    x_.setZero();
    P_.setZero();
}

Eigen::Vector3d KalmanFilter3D::update(const Eigen::Vector3d& pos_observed, double timestamp) {
    // First initialization
    if (!initialized_) {
        initialize(pos_observed, timestamp);
        return pos_observed;
    }

    double dt = timestamp - last_timestamp_;

    // Timestamp went backwards -> reset
    if (dt < 0.0) {
        std::cout << "[KalmanFilter3D] Warning: timestamp reversal, resetting filter" << std::endl;
        initialize(pos_observed, timestamp);
        return pos_observed;
    }

    // Time gap too large (data loss) -> reset
    if (dt > dt_max_) {
        std::cout << "[KalmanFilter3D] Warning: dt=" << dt << "s > dt_max=" << dt_max_
                  << "s, resetting filter" << std::endl;
        initialize(pos_observed, timestamp);
        return pos_observed;
    }

    // Predict forward
    if (dt > 0.0) {
        predictStep(dt);
    }

    // Check for large position jump
    Eigen::Vector3d pos_predicted = x_.head<3>();
    double pos_diff = (pos_observed - pos_predicted).norm();
    if (pos_diff > reset_threshold_) {
        std::cout << "[KalmanFilter3D] Warning: position jump " << pos_diff
                  << "m > " << reset_threshold_ << "m, resetting filter" << std::endl;
        initialize(pos_observed, timestamp);
        return pos_observed;
    }

    // Measurement update
    updateStep(pos_observed);
    last_timestamp_ = timestamp;

    return x_.head<3>();
}

void KalmanFilter3D::initialize(const Eigen::Vector3d& pos_observed, double timestamp) {
    x_.setZero();
    x_.head<3>() = pos_observed;
    // Velocity initialized to zero (already done by setZero)

    P_ = P0_;
    last_timestamp_ = timestamp;
    initialized_ = true;
}

void KalmanFilter3D::predictStep(double dt) {
    // State transition matrix F (6x6):
    // F = [ I3   dt*I3 ]
    //     [ 0    I3    ]
    Eigen::Matrix<double, 6, 6> F = Eigen::Matrix<double, 6, 6>::Identity();
    F(0, 3) = dt;  // px += dt * vx
    F(1, 4) = dt;  // py += dt * vy
    F(2, 5) = dt;  // pz += dt * vz

    x_ = F * x_;
    P_ = F * P_ * F.transpose() + Q_;
}

void KalmanFilter3D::updateStep(const Eigen::Vector3d& pos_observed) {
    // Innovation
    Eigen::Vector3d y = pos_observed - H_ * x_;

    // Innovation covariance
    Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;

    // Kalman gain (6x3)
    Eigen::Matrix<double, 6, 3> K = P_ * H_.transpose() * S.inverse();

    // State update
    x_ = x_ + K * y;

    // Covariance update (numerically stable form: P = (I - KH)*P)
    Eigen::Matrix<double, 6, 6> I = Eigen::Matrix<double, 6, 6>::Identity();
    P_ = (I - K * H_) * P_;
}

std::optional<Eigen::Vector3d> KalmanFilter3D::getEstimatedVelocity() const {
    if (!initialized_) {
        return std::nullopt;
    }
    return x_.tail<3>();
}

bool KalmanFilter3D::isInitialized() const {
    return initialized_;
}

void KalmanFilter3D::reset(const std::optional<Eigen::Vector3d>& pos) {
    Eigen::Vector3d reset_pos;
    if (pos.has_value()) {
        reset_pos = pos.value();
    } else if (initialized_) {
        reset_pos = x_.head<3>();
    } else {
        reset_pos = Eigen::Vector3d::Zero();
    }
    initialized_ = false;
    initialize(reset_pos, 0.0);
}

} // namespace tabletenniszed
