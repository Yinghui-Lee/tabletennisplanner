#pragma once

#include "tabletenniszed/params.hpp"
#include <Eigen/Dense>
#include <vector>

namespace tabletenniszed {

/**
 * Physics-based ball trajectory predictor.
 *
 * Simulates ball flight under quadratic air drag and gravity, and handles
 * table surface bounces with horizontal (Ch) and vertical (Cv) restitution.
 *
 * Integration method: Euler (forward).
 */
class BallPredictor {
public:
    explicit BallPredictor(const Params& params);

    // ----------------------------------------------------------------
    // Result types
    // ----------------------------------------------------------------

    /** Result of a single-time prediction. */
    struct PredictResult {
        Eigen::Vector3d initial_pos;
        Eigen::Vector3d initial_vel;
        Eigen::Vector3d final_pos;
        Eigen::Vector3d final_vel;
        int collision_count;
    };

    /** Result of a multi-time prediction. */
    struct PredictResultList {
        Eigen::Vector3d initial_pos;
        Eigen::Vector3d initial_vel;
        std::vector<Eigen::Vector3d> final_pos_list;  // ordered by target_times
        std::vector<Eigen::Vector3d> final_vel_list;
        int collision_count;
    };

    // ----------------------------------------------------------------
    // Core prediction API
    // ----------------------------------------------------------------

    /**
     * Predict complete trajectory until ball exits bounds or max_time reached.
     * @return (trajectory_points, collision_points)
     */
    std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
    predictTrajectory(const Eigen::Vector3d& initial_pos,
                      const Eigen::Vector3d& initial_vel,
                      double dt = 0.01,
                      double max_time = 3.0) const;

    /**
     * Predict state at a single future time.
     * @param initial_pos Starting position
     * @param initial_vel Starting velocity
     * @param target_time Time offset from initial state (seconds)
     * @param dt          Integration step (seconds)
     */
    PredictResult predictAtTime(const Eigen::Vector3d& initial_pos,
                                const Eigen::Vector3d& initial_vel,
                                double target_time,
                                double dt = 0.002) const;

    /**
     * Predict state at multiple future times in a single simulation pass.
     * target_times need not be sorted; results are returned in the same order.
     * @param dt Integration step (seconds)
     */
    PredictResultList predictAtTimes(const Eigen::Vector3d& initial_pos,
                                     const Eigen::Vector3d& initial_vel,
                                     const std::vector<double>& target_times,
                                     double dt = 0.002) const;

    // ----------------------------------------------------------------
    // Utility
    // ----------------------------------------------------------------

    /** Returns true if position is at/near the table surface. */
    bool checkTableCollision(const Eigen::Vector3d& pos) const;

    // ----------------------------------------------------------------
    // Physics parameters (public for external tuning)
    // ----------------------------------------------------------------
    double k;   // Air resistance coefficient
    double Ch;  // Horizontal restitution coefficient
    double Cv;  // Vertical restitution coefficient

private:
    Eigen::Vector3d gravity_;  // [0, 0, -9.81]

    double table_x_half_;   // TABLE_LENGTH / 2
    double table_y_half_;   // TABLE_WIDTH  / 2
};

} // namespace tabletenniszed
