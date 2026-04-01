#pragma once

#include "tabletenniszed/params.hpp"
#include <Eigen/Dense>
#include <optional>
#include <utility>

namespace tabletenniszed {

/**
 * Adaptive 3D Kalman Filter (AKF).
 *
 * State vector:       x = [px, py, pz, vx, vy, vz]^T
 * Observation vector: z = [px, py, pz]^T
 *
 * Uses a physics-based prediction model (gravity + quadratic drag + table
 * collision reflection) and adapts process/measurement noise at runtime.
 */
class AdaptiveKalmanFilter3D {
public:
    explicit AdaptiveKalmanFilter3D(const Params& params);

    /**
     * Run predict + update cycle.
     *
     * @param pos_observed  Observed ball position [x, y, z]
     * @param timestamp     Observation time (seconds)
     * @param distance      Ball-to-camera distance (m), used to scale R
     * @return (filtered_pos, filtered_vel)
     *         Both are nullopt on the first call, after a reset triggered by
     *         large dt, or when a return-hit is detected.
     */
    std::pair<std::optional<Eigen::Vector3d>, std::optional<Eigen::Vector3d>>
    update(const Eigen::Vector3d& pos_observed, double timestamp, double distance);

    /**
     * Reset the filter.
     * @param pos       Optional initial position; if provided together with
     *                  timestamp, the filter is immediately re-initialized.
     * @param timestamp Optional timestamp for immediate re-initialization.
     */
    void reset(const std::optional<Eigen::Vector3d>& pos = std::nullopt,
               const std::optional<double>& timestamp = std::nullopt);

    std::optional<Eigen::Vector3d> getEstimatedVelocity() const;
    bool isInitialized() const;

private:
    // Physics prediction step (gravity + drag + collision handling)
    void predict(double dt);

    // Build adaptive process noise matrix Q
    Eigen::Matrix<double, 6, 6> buildQ(double dt) const;

    // Build adaptive measurement noise matrix R (distance-dependent)
    Eigen::Matrix3d buildR(double distance) const;

    // Compute Jacobian of the nonlinear state transition
    Eigen::Matrix<double, 6, 6> computeJacobian(
        double dt, double vx, double vy, double vz, double v_norm) const;

    // Parameters
    double dt_max_;
    double fps_;
    double dt_base_;
    double k_drag_;
    double Ch_, Cv_;
    bool   debug_;
    double collision_z_;
    double tennis_table_x_, tennis_table_y_;
    double Q_pos_base_, Q_vel_base_, R_pos_base_;
    double hit_threshold_;
    double gravity_;
    Eigen::Vector3d              vel_init_;
    Eigen::Matrix<double, 6, 6>  p_init_;

    // Observation matrix H (3x6)
    Eigen::Matrix<double, 3, 6> H_;

    // Filter state
    Eigen::Matrix<double, 6, 1>  x_;   // [px, py, pz, vx, vy, vz]
    Eigen::Matrix<double, 6, 6>  P_;   // Error covariance

    double last_timestamp_;
    bool   initialized_;
};

} // namespace tabletenniszed
