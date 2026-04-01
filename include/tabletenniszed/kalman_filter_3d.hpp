#pragma once

#include <Eigen/Dense>
#include <optional>

namespace tabletenniszed {

/**
 * 3D Kalman Filter with constant velocity model.
 *
 * State vector:       x = [px, py, pz, vx, vy, vz]^T
 * Observation vector: z = [px, py, pz]^T
 *
 * Supports per-axis noise tuning to accommodate different sensor
 * characteristics along each axis.
 */
class KalmanFilter3D {
public:
    /**
     * Constructor.
     * @param q_pos_{x,y,z}  Position process noise (m^2), per axis
     * @param q_vel_{x,y,z}  Velocity process noise (m^2/s^2), per axis
     * @param r_pos_{x,y,z}  Position observation noise (m^2), per axis
     * @param p0_pos_{x,y,z} Initial position uncertainty (m^2), per axis
     * @param p0_vel_{x,y,z} Initial velocity uncertainty (m^2/s^2), per axis
     * @param dt_max         Max allowed time interval (s); resets filter if exceeded
     * @param reset_threshold Position jump threshold (m); resets filter if exceeded
     */
    KalmanFilter3D(
        double q_pos_x = 0.01,    double q_pos_y = 0.01,    double q_pos_z = 0.01,
        double q_vel_x = 0.1,     double q_vel_y = 0.1,     double q_vel_z = 0.1,
        double r_pos_x = 0.000025, double r_pos_y = 0.000025, double r_pos_z = 0.0001,
        double p0_pos_x = 0.1,    double p0_pos_y = 0.1,    double p0_pos_z = 0.1,
        double p0_vel_x = 1.0,    double p0_vel_y = 1.0,    double p0_vel_z = 1.0,
        double dt_max = 0.1,
        double reset_threshold = 0.5
    );

    /**
     * Run predict + update cycle with a new position observation.
     * @return Filtered position [x, y, z]
     */
    Eigen::Vector3d update(const Eigen::Vector3d& pos_observed, double timestamp);

    /** Get current estimated velocity, or nullopt if not yet initialized. */
    std::optional<Eigen::Vector3d> getEstimatedVelocity() const;

    /** Returns true if filter has been initialized with at least one observation. */
    bool isInitialized() const;

    /**
     * Reset filter to an uninitialized state.
     * @param pos Optional initial position; if not provided, uses current state position.
     */
    void reset(const std::optional<Eigen::Vector3d>& pos = std::nullopt);

private:
    void initialize(const Eigen::Vector3d& pos_observed, double timestamp);
    void predictStep(double dt);
    void updateStep(const Eigen::Vector3d& pos_observed);

    double dt_max_;
    double reset_threshold_;

    // Noise covariance matrices
    Eigen::Matrix<double, 6, 6> Q_;   // Process noise
    Eigen::Matrix3d              R_;   // Observation noise
    Eigen::Matrix<double, 6, 6> P0_;  // Initial covariance

    // Observation matrix H (3x6): only position rows
    Eigen::Matrix<double, 3, 6> H_;

    // State
    Eigen::Matrix<double, 6, 1> x_;   // [px, py, pz, vx, vy, vz]
    Eigen::Matrix<double, 6, 6> P_;   // Covariance

    double last_timestamp_;
    bool   initialized_;
};

} // namespace tabletenniszed
