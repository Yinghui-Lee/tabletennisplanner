#pragma once

#include <Eigen/Dense>
#include <vector>

namespace tabletenniszed {

/**
 * Hit plan output: everything needed to command the racket.
 */
struct HitPlan {
    Eigen::Vector3d v_out;             ///< Required ball exit velocity
    Eigen::Vector3d racket_normal;     ///< Racket surface normal (unit vector)
    Eigen::Vector3d racket_velocity;   ///< Racket velocity vector Vr = v_n * n
    double          racket_speed_normal; ///< Scalar speed along normal
};

/**
 * Racket pose solution from the collision inverse model.
 */
struct RacketPose {
    Eigen::Vector3d n;    ///< Racket surface normal (unit vector)
    Eigen::Vector3d Vr;   ///< Racket velocity vector = v_n * n
    double          v_n;  ///< Scalar speed along normal
};

// ============================================================
// Main planning function
// ============================================================

/**
 * Plan a ping-pong hit.
 *
 * Given the impact point, incoming ball velocity, desired flight time to the
 * target, and the target landing position, computes the required racket
 * surface normal and velocity.
 *
 * @param p_hit    Impact point position (m)
 * @param v_in     Incoming ball velocity (m/s)
 * @param T_flight Flight time from hit to landing (s)
 * @param p_target Target landing position (m)
 * @param k_drag   Aerodynamic drag coefficient
 * @param e        Coefficient of restitution for racket-ball collision
 */
HitPlan planPingpongHit(
    const Eigen::Vector3d& p_hit,
    const Eigen::Vector3d& v_in,
    double T_flight,
    const Eigen::Vector3d& p_target,
    double k_drag = 0.35,
    double e = 0.9);

// ============================================================
// Sub-functions (exposed for testing and modularity)
// ============================================================

/**
 * Analytical inverse dynamics: find exit velocity v_out such that the ball
 * travels from p_hit to p_target in exactly T seconds under linear drag
 * a = -k*v + g (per-axis decoupled approximation).
 */
Eigen::Vector3d solveOutgoingVelocityFixedTime(
    const Eigen::Vector3d& p_hit,
    const Eigen::Vector3d& p_target,
    double T,
    double k_drag = 0.35);

/**
 * Inverse collision model: given v_in and v_out, find racket normal n and
 * racket speed v_n along normal (frictionless model, restitution e along n).
 */
RacketPose solveRacketPoseAndSpeed(
    const Eigen::Vector3d& v_in,
    const Eigen::Vector3d& v_out,
    double e = 0.9);

/**
 * Forward simulation for T seconds under a = -k*||v||*v + g.
 * Uses midpoint (Heun) integration for accuracy.
 *
 * @return Trajectory as a list of positions at each integration step.
 */
std::vector<Eigen::Vector3d> simulateFlightFixedTime(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& v0,
    double T,
    double k_drag = 0.35,
    double dt = 0.001);

} // namespace tabletenniszed
