#include "tabletenniszed/planner.hpp"
#include <cmath>
#include <algorithm>

namespace tabletenniszed {

// Standard gravity for ping-pong flight simulation
static const Eigen::Vector3d g_gravity(0.0, 0.0, -9.81);

// ============================================================
// Forward simulation
// ============================================================
std::vector<Eigen::Vector3d> simulateFlightFixedTime(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& v0,
    double T,
    double k_drag,
    double dt)
{
    Eigen::Vector3d p = p0;
    Eigen::Vector3d v = v0;

    int n_steps = std::max(1, static_cast<int>(std::ceil(T / dt)));
    double dt_eff = T / static_cast<double>(n_steps);

    std::vector<Eigen::Vector3d> traj;
    traj.reserve(n_steps + 1);
    traj.push_back(p);

    for (int s = 0; s < n_steps; ++s) {
        double speed = v.norm();
        Eigen::Vector3d a_drag = -k_drag * speed * v;
        Eigen::Vector3d a = a_drag + g_gravity;

        // Midpoint (Heun) integration
        Eigen::Vector3d v_mid = v + 0.5 * a * dt_eff;
        Eigen::Vector3d p_mid = p + 0.5 * v * dt_eff;
        double speed_mid = v_mid.norm();
        Eigen::Vector3d a_mid = -k_drag * speed_mid * v_mid + g_gravity;

        v = v + a_mid * dt_eff;
        p = p + v_mid * dt_eff;
        traj.push_back(p);
    }

    return traj;
}

// ============================================================
// Solve exit velocity (analytical, per-axis linear drag approx)
// ============================================================
Eigen::Vector3d solveOutgoingVelocityFixedTime(
    const Eigen::Vector3d& p_hit,
    const Eigen::Vector3d& p_target,
    double T,
    double k_drag)
{
    double k = std::max(k_drag, 1e-6);
    double exp_term = std::exp(-k * T);
    double a = (1.0 - exp_term) / k;

    Eigen::Vector3d v_out;

    // X / Y axes (no gravity)
    for (int i = 0; i < 2; ++i) {
        double denom = std::max(1.0 - exp_term, 1e-8);
        v_out(i) = (p_target(i) - p_hit(i)) * k / denom;
    }

    // Z axis (gravity)
    double g_z = g_gravity(2);
    double numerator = (p_target(2) - p_hit(2)) - (g_z * T / k);
    v_out(2) = g_z / k + numerator / std::max(a, 1e-8);

    // Fallback to zero-drag analytic formula when k is negligibly small
    if (k_drag < 1e-4) {
        v_out = (p_target - p_hit - 0.5 * g_gravity * T * T) / T;
    }

    return v_out;
}

// ============================================================
// Solve racket pose from v_in and v_out (frictionless collision model)
// ============================================================
RacketPose solveRacketPoseAndSpeed(
    const Eigen::Vector3d& v_in,
    const Eigen::Vector3d& v_out,
    double e)
{
    Eigen::Vector3d delta = v_out - v_in;
    double norm_delta = delta.norm();

    Eigen::Vector3d n;
    if (norm_delta < 1e-6) {
        // v_in ≈ v_out: pick any reasonable normal
        double v_in_norm = v_in.norm();
        if (v_in_norm < 1e-6) {
            n = Eigen::Vector3d(0.0, 0.0, 1.0);
        } else {
            n = v_in / v_in_norm;
        }
    } else {
        // In the frictionless model, v_out - v_in must be parallel to n
        n = delta / norm_delta;
    }

    double v_in_n  = v_in.dot(n);
    double v_out_n = v_out.dot(n);

    // 1D normal collision: v_out_n = v_in_n - (1+e)*(v_in_n - v_n)
    // => v_n = v_in_n - (v_in_n - v_out_n) / (1 + e)
    double v_n = v_in_n - (v_in_n - v_out_n) / (1.0 + e);
    Eigen::Vector3d Vr = v_n * n;

    return RacketPose{n, Vr, v_n};
}

// ============================================================
// Main planning function
// ============================================================
HitPlan planPingpongHit(
    const Eigen::Vector3d& p_hit,
    const Eigen::Vector3d& v_in,
    double T_flight,
    const Eigen::Vector3d& p_target,
    double k_drag,
    double e)
{
    // Step 1: inverse flight dynamics -> exit velocity
    Eigen::Vector3d v_out = solveOutgoingVelocityFixedTime(p_hit, p_target, T_flight, k_drag);

    // Step 2: inverse collision model -> racket normal and speed
    RacketPose rp = solveRacketPoseAndSpeed(v_in, v_out, e);

    return HitPlan{v_out, rp.n, rp.Vr, rp.v_n};
}

} // namespace tabletenniszed
