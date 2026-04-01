#include "tabletenniszed/ball_predictor.hpp"
#include <algorithm>
#include <cmath>
#include <map>
#include <iostream>

namespace tabletenniszed {

BallPredictor::BallPredictor(const Params& params)
    : k(params.k_drag)
    , Ch(params.ch)
    , Cv(params.cv)
    , gravity_(0.0, 0.0, -params.gravity_predictor)
    , table_x_half_(params.table_length / 2.0)
    , table_y_half_(params.table_width  / 2.0)
{
}

bool BallPredictor::checkTableCollision(const Eigen::Vector3d& pos) const {
    bool in_x = (pos(0) >= -table_x_half_) && (pos(0) <= table_x_half_);
    bool in_y = (pos(1) >= -table_y_half_) && (pos(1) <= table_y_half_);
    bool near_surface = std::abs(pos(2)) < 0.03;
    return in_x && in_y && near_surface;
}

std::pair<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>>
BallPredictor::predictTrajectory(
    const Eigen::Vector3d& initial_pos,
    const Eigen::Vector3d& initial_vel,
    double dt,
    double max_time) const
{
    std::vector<Eigen::Vector3d> trajectory;
    std::vector<Eigen::Vector3d> collision_points;

    Eigen::Vector3d pos = initial_pos;
    Eigen::Vector3d vel = initial_vel;
    trajectory.push_back(pos);

    double time = 0.0;
    int bounce_count = 0;
    const int max_bounces = 3;

    // Collision matrix: C = diag(Ch, Ch, -Cv)
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    C(0,0) = Ch; C(1,1) = Ch; C(2,2) = -Cv;

    while (time < max_time && bounce_count < max_bounces) {
        // Handle collision (ball moving downward)
        if (checkTableCollision(pos) && vel(2) < 0.0) {
            vel = C * vel;
            collision_points.push_back(pos);
            ++bounce_count;
            std::cout << "Collision #" << bounce_count << " at ["
                      << pos.transpose() << "]" << std::endl;
        }

        // Quadratic drag + gravity
        double speed = vel.norm();
        Eigen::Vector3d drag_accel = -k * speed * vel;
        Eigen::Vector3d accel = drag_accel + gravity_;

        // Euler integration
        vel = vel + accel * dt;
        pos = pos + vel * dt;
        trajectory.push_back(pos);
        time += dt;

        // Stop if ball exits reasonable bounds
        if (pos(2) < -1.0 || std::abs(pos(0)) > 5.0 || std::abs(pos(1)) > 3.0) {
            break;
        }
    }

    return {trajectory, collision_points};
}

BallPredictor::PredictResult BallPredictor::predictAtTime(
    const Eigen::Vector3d& initial_pos,
    const Eigen::Vector3d& initial_vel,
    double target_time,
    double dt) const
{
    auto result_list = predictAtTimes(initial_pos, initial_vel, {target_time}, dt);
    return PredictResult{
        result_list.initial_pos,
        result_list.initial_vel,
        result_list.final_pos_list[0],
        result_list.final_vel_list[0],
        result_list.collision_count
    };
}

BallPredictor::PredictResultList BallPredictor::predictAtTimes(
    const Eigen::Vector3d& initial_pos,
    const Eigen::Vector3d& initial_vel,
    const std::vector<double>& target_times,
    double dt) const
{
    // Sort target_times ascending while retaining original order for output
    std::vector<double> sorted_times = target_times;
    std::sort(sorted_times.begin(), sorted_times.end());
    double max_target_time = sorted_times.back();

    Eigen::Vector3d pos = initial_pos;
    Eigen::Vector3d vel = initial_vel;

    double time = 0.0;
    int collision_count = 0;
    const int max_bounces = 10;

    // Collision matrix: C = diag(Ch, Ch, -Cv)
    Eigen::Matrix3d C = Eigen::Matrix3d::Zero();
    C(0,0) = Ch; C(1,1) = Ch; C(2,2) = -Cv;

    // Map from sorted time -> (pos, vel) snapshot
    std::map<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>> snapshots;
    std::size_t next_target_idx = 0;

    while (time < max_target_time && collision_count < max_bounces) {
        // Record any target times that fall in [time, time+dt)
        while (next_target_idx < sorted_times.size()) {
            double target_t = sorted_times[next_target_idx];
            if (time <= target_t && target_t < time + dt) {
                snapshots[target_t] = {pos, vel};
                ++next_target_idx;
            } else if (time >= target_t) {
                snapshots[target_t] = {pos, vel};
                ++next_target_idx;
            } else {
                break;
            }
        }

        double remaining = max_target_time - time;
        double current_dt = std::min(dt, remaining);

        // Table collision
        if (checkTableCollision(pos) && vel(2) < 0.0) {
            vel = C * vel;
            ++collision_count;
            pos(2) = std::max(pos(2), 0.001);
        }

        // Quadratic drag + gravity
        double speed = vel.norm();
        Eigen::Vector3d drag_accel = -k * speed * vel;
        Eigen::Vector3d accel = drag_accel + gravity_;

        // Euler integration
        vel = vel + accel * current_dt;
        pos = pos + vel * current_dt;
        time += current_dt;

        // Stop if ball exits bounds
        if (pos(2) < -1.0 || std::abs(pos(0)) > 10.0 || std::abs(pos(1)) > 10.0) {
            break;
        }

        if (time >= max_target_time) break;
    }

    // Fill any remaining target times with the last simulated state
    while (next_target_idx < sorted_times.size()) {
        double target_t = sorted_times[next_target_idx];
        if (snapshots.find(target_t) == snapshots.end()) {
            snapshots[target_t] = {pos, vel};
        }
        ++next_target_idx;
    }

    // Build output in original target_times order
    PredictResultList result;
    result.initial_pos = initial_pos;
    result.initial_vel = initial_vel;
    result.collision_count = collision_count;

    for (double t : target_times) {
        // Find closest key in snapshots
        auto it = snapshots.find(t);
        if (it != snapshots.end()) {
            result.final_pos_list.push_back(it->second.first);
            result.final_vel_list.push_back(it->second.second);
        } else {
            // Fallback: nearest key
            auto lower = snapshots.lower_bound(t);
            if (lower == snapshots.end()) {
                --lower;
            } else if (lower != snapshots.begin()) {
                auto prev = std::prev(lower);
                if (std::abs(prev->first - t) < std::abs(lower->first - t)) {
                    lower = prev;
                }
            }
            result.final_pos_list.push_back(lower->second.first);
            result.final_vel_list.push_back(lower->second.second);
        }
    }

    return result;
}

} // namespace tabletenniszed
