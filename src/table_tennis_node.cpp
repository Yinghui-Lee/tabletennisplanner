#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>

#include <Eigen/Dense>

#include <chrono>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <algorithm>

#include "tabletenniszed/params.hpp"
#include "tabletenniszed/ball_predictor.hpp"
#include "tabletenniszed/planner.hpp"
#include "tabletenniszed/zed_receiver.hpp"

namespace tabletenniszed {

// ============================================================
// Load all ROS2 parameters into a Params struct.
// Must be called after declare_parameter() is valid (i.e. inside a Node).
// ============================================================

/** Load a 4×4 matrix declared as a flat 16-element double[] parameter. */
static Eigen::Matrix4d loadMatrix4d(rclcpp::Node& node,
                                    const std::string& name,
                                    const Eigen::Matrix4d& default_val)
{
    std::vector<double> def_flat(16);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            def_flat[i * 4 + j] = default_val(i, j);

    auto flat = node.declare_parameter<std::vector<double>>(name, def_flat);
    Eigen::Matrix4d m;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            m(i, j) = flat[i * 4 + j];
    return m;
}

/** Load a Vector3d declared as a 3-element double[] parameter. */
static Eigen::Vector3d loadVec3(rclcpp::Node& node,
                                const std::string& name,
                                const Eigen::Vector3d& default_val)
{
    std::vector<double> def = {default_val(0), default_val(1), default_val(2)};
    auto v = node.declare_parameter<std::vector<double>>(name, def);
    return Eigen::Vector3d(v[0], v[1], v[2]);
}

static Params loadParamsFromNode(rclcpp::Node& node)
{
    Params p;  // starts with default values

    // --- Communication ---
    p.zed_udp_host        = node.declare_parameter<std::string>("udp_host",        p.zed_udp_host);
    p.zed_udp_port        = node.declare_parameter<int>        ("udp_port",        p.zed_udp_port);
    p.planning_update_rate = node.declare_parameter<int>       ("planning_rate",   p.planning_update_rate);

    // --- Table dimensions ---
    p.table_length = node.declare_parameter<double>("table_length", p.table_length);
    p.table_width  = node.declare_parameter<double>("table_width",  p.table_width);
    p.table_height = node.declare_parameter<double>("table_height", p.table_height);

    // --- Physics ---
    p.gravity           = node.declare_parameter<double>("gravity",           p.gravity);
    p.gravity_predictor = node.declare_parameter<double>("gravity_predictor", p.gravity_predictor);
    p.k_drag            = node.declare_parameter<double>("k_drag",            p.k_drag);
    p.ch                = node.declare_parameter<double>("ch",                p.ch);
    p.cv                = node.declare_parameter<double>("cv",                p.cv);

    // --- Marker filter ---
    p.marker_filter_x_min   = node.declare_parameter<double>("marker_filter_x_min",   p.marker_filter_x_min);
    p.marker_filter_x_max   = node.declare_parameter<double>("marker_filter_x_max",   p.marker_filter_x_max);
    p.marker_filter_y_range = node.declare_parameter<double>("marker_filter_y_range", p.marker_filter_y_range);
    p.marker_filter_z_min   = node.declare_parameter<double>("marker_filter_z_min",   p.marker_filter_z_min);
    p.marker_filter_z_max   = node.declare_parameter<double>("marker_filter_z_max",   p.marker_filter_z_max);

    // --- Strike region ---
    p.strike_region_x_min  = node.declare_parameter<double>("strike_region_x_min",  p.strike_region_x_min);
    p.strike_region_x_max  = node.declare_parameter<double>("strike_region_x_max",  p.strike_region_x_max);
    p.strike_region_y_min  = node.declare_parameter<double>("strike_region_y_min",  p.strike_region_y_min);
    p.strike_region_y_max  = node.declare_parameter<double>("strike_region_y_max",  p.strike_region_y_max);
    p.strike_region_z_min  = node.declare_parameter<double>("strike_region_z_min",  p.strike_region_z_min);
    p.strike_region_z_max  = node.declare_parameter<double>("strike_region_z_max",  p.strike_region_z_max);
    p.strike_region_center = loadVec3(node, "strike_region_center", p.strike_region_center);

    // --- Defaults ---
    p.default_pos_origin      = loadVec3(node, "default_pos_origin",      p.default_pos_origin);
    p.default_vel_origin      = loadVec3(node, "default_vel_origin",      p.default_vel_origin);
    p.default_racket_normal   = loadVec3(node, "default_racket_normal",   p.default_racket_normal);
    p.default_racket_velocity = loadVec3(node, "default_racket_velocity", p.default_racket_velocity);

    // --- Planning control ---
    p.default_time_to_strike = node.declare_parameter<double>("default_time_to_strike", p.default_time_to_strike);
    p.time_to_strike_min     = node.declare_parameter<double>("time_to_strike_min",     p.time_to_strike_min);
    p.time_to_strike_max     = node.declare_parameter<double>("time_to_strike_max",     p.time_to_strike_max);
    p.start_detect_time_min  = node.declare_parameter<double>("start_detect_time_min",  p.start_detect_time_min);
    p.start_detect_time_max  = node.declare_parameter<double>("start_detect_time_max",  p.start_detect_time_max);
    p.prediction_time_window = node.declare_parameter<double>("prediction_time_window", p.prediction_time_window);
    p.prediction_time_min    = node.declare_parameter<double>("prediction_time_min",    p.prediction_time_min);
    p.prediction_time_max    = node.declare_parameter<double>("prediction_time_max",    p.prediction_time_max);
    p.prediction_time_step   = node.declare_parameter<double>("prediction_time_step",   p.prediction_time_step);
    p.t_flight_default       = node.declare_parameter<double>("t_flight_default",       p.t_flight_default);
    p.racket_restitution     = node.declare_parameter<double>("racket_restitution",     p.racket_restitution);

    // --- Ball velocity limits ---
    p.ball_vel_x_min = node.declare_parameter<double>("ball_vel_x_min", p.ball_vel_x_min);
    p.ball_vel_x_max = node.declare_parameter<double>("ball_vel_x_max", p.ball_vel_x_max);
    p.ball_vel_y_min = node.declare_parameter<double>("ball_vel_y_min", p.ball_vel_y_min);
    p.ball_vel_y_max = node.declare_parameter<double>("ball_vel_y_max", p.ball_vel_y_max);
    p.ball_vel_z_min = node.declare_parameter<double>("ball_vel_z_min", p.ball_vel_z_min);
    p.ball_vel_z_max = node.declare_parameter<double>("ball_vel_z_max", p.ball_vel_z_max);
    p.hit_ball_vel_x_min = node.declare_parameter<double>("hit_ball_vel_x_min", p.hit_ball_vel_x_min);
    p.hit_ball_vel_x_max = node.declare_parameter<double>("hit_ball_vel_x_max", p.hit_ball_vel_x_max);
    p.hit_ball_vel_y_min = node.declare_parameter<double>("hit_ball_vel_y_min", p.hit_ball_vel_y_min);
    p.hit_ball_vel_y_max = node.declare_parameter<double>("hit_ball_vel_y_max", p.hit_ball_vel_y_max);
    p.hit_ball_vel_z_min = node.declare_parameter<double>("hit_ball_vel_z_min", p.hit_ball_vel_z_min);
    p.hit_ball_vel_z_max = node.declare_parameter<double>("hit_ball_vel_z_max", p.hit_ball_vel_z_max);

    // --- Standard Kalman filter ---
    p.kalman_enable          = node.declare_parameter<bool>  ("kalman_enable",          p.kalman_enable);
    p.kalman_q_pos_x         = node.declare_parameter<double>("kalman_q_pos_x",         p.kalman_q_pos_x);
    p.kalman_q_pos_y         = node.declare_parameter<double>("kalman_q_pos_y",         p.kalman_q_pos_y);
    p.kalman_q_pos_z         = node.declare_parameter<double>("kalman_q_pos_z",         p.kalman_q_pos_z);
    p.kalman_q_vel_x         = node.declare_parameter<double>("kalman_q_vel_x",         p.kalman_q_vel_x);
    p.kalman_q_vel_y         = node.declare_parameter<double>("kalman_q_vel_y",         p.kalman_q_vel_y);
    p.kalman_q_vel_z         = node.declare_parameter<double>("kalman_q_vel_z",         p.kalman_q_vel_z);
    p.kalman_r_pos_x         = node.declare_parameter<double>("kalman_r_pos_x",         p.kalman_r_pos_x);
    p.kalman_r_pos_y         = node.declare_parameter<double>("kalman_r_pos_y",         p.kalman_r_pos_y);
    p.kalman_r_pos_z         = node.declare_parameter<double>("kalman_r_pos_z",         p.kalman_r_pos_z);
    p.kalman_p0_pos_x        = node.declare_parameter<double>("kalman_p0_pos_x",        p.kalman_p0_pos_x);
    p.kalman_p0_pos_y        = node.declare_parameter<double>("kalman_p0_pos_y",        p.kalman_p0_pos_y);
    p.kalman_p0_pos_z        = node.declare_parameter<double>("kalman_p0_pos_z",        p.kalman_p0_pos_z);
    p.kalman_p0_vel_x        = node.declare_parameter<double>("kalman_p0_vel_x",        p.kalman_p0_vel_x);
    p.kalman_p0_vel_y        = node.declare_parameter<double>("kalman_p0_vel_y",        p.kalman_p0_vel_y);
    p.kalman_p0_vel_z        = node.declare_parameter<double>("kalman_p0_vel_z",        p.kalman_p0_vel_z);
    p.kalman_dt_max          = node.declare_parameter<double>("kalman_dt_max",          p.kalman_dt_max);
    p.kalman_reset_threshold = node.declare_parameter<double>("kalman_reset_threshold", p.kalman_reset_threshold);

    // --- Adaptive Kalman filter ---
    p.use_adaptive_kalman = node.declare_parameter<bool>  ("use_adaptive_kalman", p.use_adaptive_kalman);
    p.akf_q_pos_base      = node.declare_parameter<double>("akf_q_pos_base",      p.akf_q_pos_base);
    p.akf_q_vel_base      = node.declare_parameter<double>("akf_q_vel_base",      p.akf_q_vel_base);
    p.akf_r_pos           = node.declare_parameter<double>("akf_r_pos",           p.akf_r_pos);
    p.akf_dt_max          = node.declare_parameter<double>("akf_dt_max",          p.akf_dt_max);
    p.akf_fps             = node.declare_parameter<double>("akf_fps",             p.akf_fps);
    p.akf_collision_z     = node.declare_parameter<double>("akf_collision_z",     p.akf_collision_z);
    p.akf_debug           = node.declare_parameter<bool>  ("akf_debug",           p.akf_debug);
    p.akf_tennis_table_x  = node.declare_parameter<double>("akf_tennis_table_x",  p.akf_tennis_table_x);
    p.akf_tennis_table_y  = node.declare_parameter<double>("akf_tennis_table_y",  p.akf_tennis_table_y);
    p.akf_hit_threshold   = node.declare_parameter<double>("akf_hit_threshold",   p.akf_hit_threshold);
    p.akf_vel_init        = loadVec3(node, "akf_vel_init", p.akf_vel_init);
    p.akf_p_init_pos      = node.declare_parameter<double>("akf_p_init_pos",      p.akf_p_init_pos);
    p.akf_p_init_vel      = node.declare_parameter<double>("akf_p_init_vel",      p.akf_p_init_vel);

    // --- Coordinate transforms ---
    // Default values for matrices (current calibration)
    Eigen::Matrix4d def_tiw;
    def_tiw <<  8.95988499e-03, -9.99923793e-01,  8.49282644e-03, -1.22892347e+00,
               -1.86844317e-03,  8.47641143e-03,  9.99962329e-01,  7.66285777e-01,
               -9.99958114e-01, -8.97541583e-03, -1.79235311e-03,  2.20226499e+00,
                0.0,             0.0,             0.0,             1.0;
    Eigen::Matrix4d def_ctt;
    def_ctt <<  0.0,  0.0,  1.0,  0.05956,
               -1.0,  0.0,  0.0,  0.05950,
                0.0, -1.0,  0.0,  0.48791,
                0.0,  0.0,  0.0,  1.0;
    Eigen::Matrix4d def_ott;
    def_ott << 1.0, 0.0, 0.0, -2.1,
               0.0, 1.0, 0.0,  0.0,
               0.0, 0.0, 1.0, -7.64800267e-01,
               0.0, 0.0, 0.0,  1.0;

    p.table_in_world    = loadMatrix4d(node, "table_in_world",    def_tiw);
    p.camera_to_torso   = loadMatrix4d(node, "camera_to_torso",   def_ctt);
    p.t_origin_to_table = loadMatrix4d(node, "t_origin_to_table", def_ott);

    // Compute derived transforms
    p.origin_in_world = p.table_in_world * p.t_origin_to_table;
    p.table_in_origin = p.origin_in_world.inverse() * p.table_in_world;

    return p;
}


// ============================================================
// Helper: current time in seconds
// ============================================================
static double nowSec() {
    return std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ============================================================
// Helper: element-wise clamp for Eigen::Vector3d
// ============================================================
static Eigen::Vector3d clampVec3(const Eigen::Vector3d& v,
                                  const Eigen::Vector3d& lo,
                                  const Eigen::Vector3d& hi) {
    return Eigen::Vector3d(
        std::clamp(v(0), lo(0), hi(0)),
        std::clamp(v(1), lo(1), hi(1)),
        std::clamp(v(2), lo(2), hi(2)));
}

// ============================================================
// TableTennisNode
// ============================================================

class TableTennisNode : public rclcpp::Node {
public:
    TableTennisNode()
        : Node("table_tennis_predictor")
        , running_(false)
    {
        p_ = loadParamsFromNode(*this);
        planning_update_rate_ = p_.planning_update_rate;
        initHyperparameters();
        initCoordinateTransforms();
        initBallPredictor();
        initPredictionControl();
        initDataLogging();
        initRos2Publishers();
        // NOTE: ZedReceiver and the planning timer are created in initialize(),
        // which must be called AFTER this object is owned by a shared_ptr
        // (i.e. after std::make_shared returns in main()).
    }

    /**
     * Must be called once from main(), immediately after the node is wrapped
     * in a shared_ptr, because ZedReceiver needs shared_from_this().
     */
    void initialize() {
        initZedReceiver();
        zed_receiver_->start();

        planning_timer_ = create_wall_timer(
            std::chrono::duration<double>(1.0 / planning_update_rate_),
            std::bind(&TableTennisNode::executePlanning, this));

        RCLCPP_INFO(get_logger(),
            "[TableTennisNode] Started at %d Hz", planning_update_rate_);
    }

    ~TableTennisNode() {
        if (zed_receiver_) zed_receiver_->stop();
        stopDataLogging();
    }

private:
    // ============================================================
    // Initialization helpers
    // ============================================================

    void initHyperparameters() {
        default_pos_origin_    = p_.default_pos_origin;
        default_vel_origin_    = p_.default_vel_origin;
        default_racket_normal_ = p_.default_racket_normal;
        default_racket_vel_    = p_.default_racket_velocity;

        last_published_pos_    = default_pos_origin_;
        last_published_vel_    = default_vel_origin_;
        last_published_normal_ = default_racket_normal_;
        last_published_racket_ = default_racket_vel_;
        last_published_ts_     = nowSec();
        last_time_to_strike_   = p_.default_time_to_strike;
    }

    void initCoordinateTransforms() {
        table_in_world_  = p_.table_in_world;
        origin_in_world_ = p_.origin_in_world;
        table_in_origin_ = p_.table_in_origin;
    }

    void initBallPredictor() {
        ball_predictor_ = std::make_unique<BallPredictor>(p_);
    }

    void initPredictionControl() {
        cross_detected_  = false;
        time_to_strike_  = p_.default_time_to_strike;
        last_zed_ts_     = std::nullopt;
        current_pos_table_.setZero();
        current_vel_table_.setZero();
        missed_steps_ = 0;
    }

    void initDataLogging() {
        // Build a unique filename: prediction_data{N}.csv
        namespace fs = std::filesystem;
        std::string base_dir  = "data";
        std::string name_part = "prediction_data";
        std::string ext_part  = ".csv";

        fs::create_directories(base_dir);

        int next_index = 1;
        for (auto& entry : fs::directory_iterator(base_dir)) {
            std::string fname = entry.path().filename().string();
            if (fname.rfind(name_part, 0) == 0 &&
                fname.size() > name_part.size() + ext_part.size()) {
                std::string middle = fname.substr(
                    name_part.size(),
                    fname.size() - name_part.size() - ext_part.size());
                try {
                    int idx = std::stoi(middle);
                    next_index = std::max(next_index, idx + 1);
                } catch (...) {}
            }
        }

        csv_path_ = base_dir + "/" + name_part +
                    std::to_string(next_index) + ext_part;
        csv_file_.open(csv_path_);
        if (csv_file_.is_open()) {
            csv_file_ << "timestamp,"
                         "pos_table_raw_x,pos_table_raw_y,pos_table_raw_z,"
                         "pos_x,pos_y,pos_z,"
                         "pos_origin_raw_x,pos_origin_raw_y,pos_origin_raw_z,"
                         "pos_origin_x,pos_origin_y,pos_origin_z,"
                         "vel_x,vel_y,vel_z,"
                         "predicted_pos_x,predicted_pos_y,predicted_pos_z,"
                         "predicted_vel_x,predicted_vel_y,predicted_vel_z,"
                         "predicted_pos_origin_x,predicted_pos_origin_y,predicted_pos_origin_z,"
                         "predicted_vel_origin_x,predicted_vel_origin_y,predicted_vel_origin_z,"
                         "racket_velocity_origin_x,racket_velocity_origin_y,racket_velocity_origin_z,"
                         "predict_time\n";
            RCLCPP_INFO(get_logger(),
                "[TableTennisNode] Logging predictions to: %s", csv_path_.c_str());
        }
    }

    void stopDataLogging() {
        if (csv_file_.is_open()) {
            csv_file_.flush();
            csv_file_.close();
            RCLCPP_INFO(get_logger(),
                "[TableTennisNode] Prediction data saved to: %s", csv_path_.c_str());
        }
    }

    void initRos2Publishers() {
        pos_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
            "predicted_ball_position", 10);
        vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
            "predicted_ball_velocity", 10);
        predict_time_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
            "predicted_ball_predict_time", 10);
        racket_normal_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>(
            "predicted_racket_normal", 10);
        racket_vel_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>(
            "predicted_racket_velocity", 10);

        RCLCPP_INFO(get_logger(), "[TableTennisNode] ROS2 publishers created:");
        RCLCPP_INFO(get_logger(), "  /predicted_ball_position     (PointStamped)");
        RCLCPP_INFO(get_logger(), "  /predicted_ball_velocity     (TwistStamped)");
        RCLCPP_INFO(get_logger(), "  /predicted_ball_predict_time (PointStamped)");
        RCLCPP_INFO(get_logger(), "  /predicted_racket_normal     (Vector3Stamped)");
        RCLCPP_INFO(get_logger(), "  /predicted_racket_velocity   (TwistStamped)");
    }

    void initZedReceiver() {
        zed_receiver_ = std::make_unique<ZedReceiver>(shared_from_this(), p_);
    }

    // ============================================================
    // Coordinate transforms
    // ============================================================

    /** Transform a position from table frame to origin frame. */
    Eigen::Vector3d transformTableToOriginPos(const Eigen::Vector3d& pos_table) const {
        Eigen::Vector4d h(pos_table(0), pos_table(1), pos_table(2), 1.0);
        return (table_in_origin_ * h).head<3>();
    }

    /** Transform a velocity (vector) from table frame to origin frame. */
    Eigen::Vector3d transformTableToOriginVel(const Eigen::Vector3d& vel_table) const {
        return table_in_origin_.block<3,3>(0,0) * vel_table;
    }

    /** Transform both position and velocity. */
    std::pair<Eigen::Vector3d, Eigen::Vector3d>
    transformTableToOrigin(const Eigen::Vector3d& pos_table,
                            const Eigen::Vector3d& vel_table) const {
        return {transformTableToOriginPos(pos_table),
                transformTableToOriginVel(vel_table)};
    }

    // ============================================================
    // Prediction helpers
    // ============================================================

    /**
     * Predict ball position and velocity at a single future time.
     */
    std::pair<Eigen::Vector3d, Eigen::Vector3d>
    predictFuturePosition(double prediction_time) const {
        auto result = ball_predictor_->predictAtTime(
            current_pos_table_, current_vel_table_, prediction_time, 0.002);
        return {result.final_pos, result.final_vel};
    }

    /**
     * Predict ball position and velocity at multiple future times.
     */
    BallPredictor::PredictResultList
    predictFuturePositions(const std::vector<double>& times) const {
        return ball_predictor_->predictAtTimes(
            current_pos_table_, current_vel_table_, times, 0.002);
    }

    // ============================================================
    // Detect start of incoming ball
    // ============================================================

    void detectStart(double timestamp) {
        // Require ball to be moving toward our side (vx < -0.1) and not too fast
        if (current_vel_table_(0) >= -0.1 || current_vel_table_(0) < -5.0) return;
        if (current_pos_table_(0) < -0.5) return;

        // Predict trajectory over [start_detect_time_min, start_detect_time_max]
        std::vector<double> t_list;
        for (double t = p_.start_detect_time_min;
             t < p_.start_detect_time_max; t += 0.01) {
            t_list.push_back(t);
        }

        auto pred_result = predictFuturePositions(t_list);

        // Transform predicted positions to origin frame and check strike region
        Eigen::Vector3d center = p_.strike_region_center;
        double best_dist = 1e9;
        int    best_idx  = -1;
        Eigen::Vector3d best_pos_origin;

        for (std::size_t i = 0; i < pred_result.final_pos_list.size(); ++i) {
            Eigen::Vector3d pos_origin =
                transformTableToOriginPos(pred_result.final_pos_list[i]);

            bool in_x = (pos_origin(0) >= p_.strike_region_x_min) &&
                        (pos_origin(0) <= p_.strike_region_x_max);
            bool in_y = (pos_origin(1) >= p_.strike_region_y_min) &&
                        (pos_origin(1) <= p_.strike_region_y_max);
            bool in_z = (pos_origin(2) >= p_.strike_region_z_min) &&
                        (pos_origin(2) <= p_.strike_region_z_max);

            if (in_x && in_y && in_z) {
                double dist = std::abs(pos_origin(0) - center(0));
                if (dist < best_dist) {
                    best_dist      = dist;
                    best_idx       = static_cast<int>(i);
                    best_pos_origin = pos_origin;
                }
            }
        }

        if (best_idx >= 0) {
            time_to_strike_ = t_list[static_cast<std::size_t>(best_idx)];
            cross_detected_ = true;
            RCLCPP_INFO(get_logger(),
                "[TableTennisNode] Ball detected: ts=%.3f, pos_table=[%.3f,%.3f,%.3f]"
                " vel=[%.3f,%.3f,%.3f] t_strike=%.3f hit=[%.3f,%.3f,%.3f]",
                timestamp,
                current_pos_table_(0), current_pos_table_(1), current_pos_table_(2),
                current_vel_table_(0), current_vel_table_(1), current_vel_table_(2),
                time_to_strike_,
                best_pos_origin(0), best_pos_origin(1), best_pos_origin(2));
        }
    }

    // ============================================================
    // Main planning loop (called by timer)
    // ============================================================

    void executePlanning() {
        double current_timestamp = nowSec();

        // ---- Fetch ZED data ----
        std::optional<BallData> ball_data;
        if (zed_receiver_) {
            ball_data = zed_receiver_->getLatestBallPosTable();
        }

        // Ignore data with unchanged timestamp
        if (ball_data.has_value()) {
            if (last_zed_ts_.has_value() &&
                ball_data->timestamp == *last_zed_ts_) {
                ball_data.reset();
            } else {
                last_zed_ts_ = ball_data->timestamp;
            }
        }

        double last_time_to_strike = time_to_strike_;

        // ---- State variables for this iteration ----
        Eigen::Vector3d predicted_pos_origin  = default_pos_origin_;
        Eigen::Vector3d predicted_vel_origin  = default_vel_origin_;
        Eigen::Vector3d racket_normal_origin  = default_racket_normal_;
        Eigen::Vector3d racket_vel_origin     = default_racket_vel_;

        // ---- Process new ZED data ----
        if (ball_data.has_value()) {
            current_pos_table_     = ball_data->pos_table;
            current_vel_table_     = ball_data->vel_table;
            current_pos_table_raw_ = ball_data->pos_table_raw;
            current_pos_origin_    = transformTableToOriginPos(current_pos_table_);
            current_pos_origin_raw_ = transformTableToOriginPos(current_pos_table_raw_);
            current_timestamp      = ball_data->timestamp;
            missed_steps_          = 0;

            // Detect incoming ball if not already tracking
            if (!cross_detected_) {
                detectStart(current_timestamp);
            }

            // Active prediction & planning
            if (cross_detected_ && time_to_strike_ > 0.0 &&
                current_pos_origin_(0) > p_.strike_region_center(0)) {

                // Build list of times around current time_to_strike
                std::vector<double> t_list;
                for (double t = time_to_strike_ - p_.prediction_time_window;
                     t < time_to_strike_ + p_.prediction_time_window;
                     t += p_.prediction_time_step) {
                    if (t >= p_.prediction_time_min &&
                        t <= p_.prediction_time_max) {
                        t_list.push_back(t);
                    }
                }

                if (!t_list.empty()) {
                    auto batch = predictFuturePositions(t_list);

                    // Find time when x (origin) is closest to strike-region center x
                    Eigen::Vector3d center = p_.strike_region_center;
                    double best_dist = 1e9;
                    int    best_idx  = -1;

                    for (std::size_t i = 0; i < batch.final_pos_list.size(); ++i) {
                        Eigen::Vector3d pos_o =
                            transformTableToOriginPos(batch.final_pos_list[i]);
                        double dist = std::abs(pos_o(0) - center(0));
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_idx  = static_cast<int>(i);
                        }
                    }

                    if (best_idx >= 0) {
                        time_to_strike_ = t_list[static_cast<std::size_t>(best_idx)];
                    }
                }

                // Predict at updated time_to_strike
                auto [pred_pos, pred_vel] = predictFuturePosition(time_to_strike_);
                pred_vel = clampVec3(pred_vel,
                    Eigen::Vector3d(p_.ball_vel_x_min, p_.ball_vel_y_min, p_.ball_vel_z_min),
                    Eigen::Vector3d(p_.ball_vel_x_max, p_.ball_vel_y_max, p_.ball_vel_z_max));

                auto [po, vo] = transformTableToOrigin(pred_pos, pred_vel);
                predicted_pos_origin = po;
                predicted_vel_origin = vo;

                // Hit planning
                Eigen::Vector3d plan_vel_origin = clampVec3(predicted_vel_origin,
                    Eigen::Vector3d(p_.hit_ball_vel_x_min, p_.hit_ball_vel_y_min, p_.hit_ball_vel_z_min),
                    Eigen::Vector3d(p_.hit_ball_vel_x_max, p_.hit_ball_vel_y_max, p_.hit_ball_vel_z_max));

                try {
                    Eigen::Vector3d p_target(2.8, 0.0, 0.8);
                    HitPlan plan = planPingpongHit(
                        predicted_pos_origin,
                        plan_vel_origin,
                        p_.t_flight_default,
                        p_target,
                        ball_predictor_->k,
                        p_.racket_restitution);

                    racket_normal_origin = transformTableToOriginVel(plan.racket_normal);
                    racket_vel_origin    = transformTableToOriginVel(plan.racket_velocity);

                    RCLCPP_INFO(get_logger(),
                        "ts=%.3f | t_strike=%.3f | pos_origin=[%.3f,%.3f,%.3f]"
                        " | vel_origin=[%.3f,%.3f,%.3f]",
                        current_timestamp, time_to_strike_,
                        predicted_pos_origin(0), predicted_pos_origin(1), predicted_pos_origin(2),
                        predicted_vel_origin(0), predicted_vel_origin(1), predicted_vel_origin(2));
                } catch (const std::exception& ex) {
                    RCLCPP_WARN(get_logger(),
                        "[TableTennisNode] Hit planning failed: %s", ex.what());
                }

                // Save prediction to CSV
                savePredictionData(
                    current_timestamp,
                    current_pos_table_raw_,
                    current_pos_table_,
                    current_pos_origin_raw_,
                    current_pos_origin_,
                    current_vel_table_,
                    pred_pos, pred_vel,
                    predicted_pos_origin, predicted_vel_origin,
                    racket_vel_origin,
                    time_to_strike_);

                // Update defaults for next cycle
                default_pos_origin_    = predicted_pos_origin;
                default_vel_origin_    = predicted_vel_origin;
                default_racket_normal_ = racket_normal_origin;
                default_racket_vel_    = racket_vel_origin;
            }
        }

        // ---- Clip predicted position to strike region ----
        predicted_pos_origin = clampVec3(predicted_pos_origin,
            Eigen::Vector3d(p_.strike_region_x_min, p_.strike_region_y_min, p_.strike_region_z_min),
            Eigen::Vector3d(p_.strike_region_x_max, p_.strike_region_y_max, p_.strike_region_z_max));

        // ---- Time-to-strike decay ----
        if (last_time_to_strike == time_to_strike_) {
            // No update this cycle — decrement
            time_to_strike_ -= 1.0 / static_cast<double>(planning_update_rate_);
        }
        time_to_strike_ = std::clamp(time_to_strike_,
            p_.time_to_strike_min, p_.time_to_strike_max);

        // Reset detection after hit is complete
        if (time_to_strike_ < -0.2) {
            cross_detected_ = false;
        }

        // ---- Data-unchanged check (reuse previous if nothing changed) ----
        bool data_unchanged =
            predicted_pos_origin.isApprox(last_published_pos_) &&
            predicted_vel_origin.isApprox(last_published_vel_) &&
            racket_normal_origin.isApprox(last_published_normal_) &&
            racket_vel_origin.isApprox(last_published_racket_);

        double current_time_to_strike;
        if (data_unchanged) {
            predicted_pos_origin  = last_published_pos_;
            predicted_vel_origin  = last_published_vel_;
            racket_normal_origin  = last_published_normal_;
            racket_vel_origin     = last_published_racket_;
            current_timestamp     = last_published_ts_;
            current_time_to_strike = last_time_to_strike_;
        } else {
            current_time_to_strike   = time_to_strike_;
            last_published_pos_      = predicted_pos_origin;
            last_published_vel_      = predicted_vel_origin;
            last_published_normal_   = racket_normal_origin;
            last_published_racket_   = racket_vel_origin;
            last_published_ts_       = current_timestamp;
            last_time_to_strike_     = current_time_to_strike;
        }

        // ---- Publish to ROS2 ----
        if (time_to_strike_ >= p_.time_to_strike_min &&
            time_to_strike_ <= p_.time_to_strike_max) {
            publishPrediction(predicted_pos_origin, predicted_vel_origin,
                              current_timestamp, current_time_to_strike);
            publishRacket(racket_normal_origin, racket_vel_origin, current_timestamp);
        }
    }

    // ============================================================
    // ROS2 publishing helpers
    // ============================================================

    void publishPrediction(const Eigen::Vector3d& pos_origin,
                            const Eigen::Vector3d& vel_origin,
                            double timestamp,
                            double predict_time) {
        rclcpp::Time ros_time(
            static_cast<int32_t>(static_cast<int64_t>(timestamp)),
            static_cast<uint32_t>((timestamp - std::floor(timestamp)) * 1e9));

        // Position
        geometry_msgs::msg::PointStamped pos_msg;
        pos_msg.header.stamp    = ros_time;
        pos_msg.header.frame_id = "origin_frame_w";
        pos_msg.point.x = pos_origin(0);
        pos_msg.point.y = pos_origin(1);
        pos_msg.point.z = pos_origin(2);
        pos_pub_->publish(pos_msg);

        // Velocity
        geometry_msgs::msg::TwistStamped vel_msg;
        vel_msg.header.stamp    = ros_time;
        vel_msg.header.frame_id = "origin_frame_w";
        vel_msg.twist.linear.x  = vel_origin(0);
        vel_msg.twist.linear.y  = vel_origin(1);
        vel_msg.twist.linear.z  = vel_origin(2);
        vel_pub_->publish(vel_msg);

        // Predict time (encoded in point.x)
        if (predict_time >= -0.5) {
            geometry_msgs::msg::PointStamped pt_msg;
            pt_msg.header.stamp    = ros_time;
            pt_msg.header.frame_id = "origin_frame_w";
            pt_msg.point.x = predict_time;
            pt_msg.point.y = 0.0;
            pt_msg.point.z = 0.0;
            predict_time_pub_->publish(pt_msg);
        }
    }

    void publishRacket(const Eigen::Vector3d& racket_normal,
                        const Eigen::Vector3d& racket_vel,
                        double timestamp) {
        rclcpp::Time ros_time(
            static_cast<int32_t>(static_cast<int64_t>(timestamp)),
            static_cast<uint32_t>((timestamp - std::floor(timestamp)) * 1e9));

        // Racket normal
        geometry_msgs::msg::Vector3Stamped n_msg;
        n_msg.header.stamp    = ros_time;
        n_msg.header.frame_id = "origin_frame_w";
        n_msg.vector.x = racket_normal(0);
        n_msg.vector.y = racket_normal(1);
        n_msg.vector.z = racket_normal(2);
        racket_normal_pub_->publish(n_msg);

        // Racket velocity
        geometry_msgs::msg::TwistStamped v_msg;
        v_msg.header.stamp    = ros_time;
        v_msg.header.frame_id = "origin_frame_w";
        v_msg.twist.linear.x  = racket_vel(0);
        v_msg.twist.linear.y  = racket_vel(1);
        v_msg.twist.linear.z  = racket_vel(2);
        racket_vel_pub_->publish(v_msg);
    }

    // ============================================================
    // CSV logging
    // ============================================================

    void savePredictionData(
        double timestamp,
        const Eigen::Vector3d& pos_table_raw,
        const Eigen::Vector3d& pos_table,
        const Eigen::Vector3d& pos_origin_raw,
        const Eigen::Vector3d& pos_origin,
        const Eigen::Vector3d& vel_table,
        const Eigen::Vector3d& predicted_pos,
        const Eigen::Vector3d& predicted_vel,
        const Eigen::Vector3d& predicted_pos_origin,
        const Eigen::Vector3d& predicted_vel_origin,
        const Eigen::Vector3d& racket_vel_origin,
        double predict_time)
    {
        if (!csv_file_.is_open()) return;

        csv_file_ << timestamp << ","
                  << pos_table_raw(0) << "," << pos_table_raw(1) << "," << pos_table_raw(2) << ","
                  << pos_table(0) << "," << pos_table(1) << "," << pos_table(2) << ","
                  << pos_origin_raw(0) << "," << pos_origin_raw(1) << "," << pos_origin_raw(2) << ","
                  << pos_origin(0) << "," << pos_origin(1) << "," << pos_origin(2) << ","
                  << vel_table(0) << "," << vel_table(1) << "," << vel_table(2) << ","
                  << predicted_pos(0) << "," << predicted_pos(1) << "," << predicted_pos(2) << ","
                  << predicted_vel(0) << "," << predicted_vel(1) << "," << predicted_vel(2) << ","
                  << predicted_pos_origin(0) << "," << predicted_pos_origin(1) << "," << predicted_pos_origin(2) << ","
                  << predicted_vel_origin(0) << "," << predicted_vel_origin(1) << "," << predicted_vel_origin(2) << ","
                  << racket_vel_origin(0) << "," << racket_vel_origin(1) << "," << racket_vel_origin(2) << ","
                  << predict_time << "\n";

        csv_file_.flush();
    }

    // ============================================================
    // Member variables
    // ============================================================

    // Runtime parameters (loaded from ROS2 parameter server / YAML)
    Params p_;

    // Planning rate
    int planning_update_rate_;
    bool running_;

    // ZED receiver
    std::unique_ptr<ZedReceiver> zed_receiver_;

    // Ball predictor
    std::unique_ptr<BallPredictor> ball_predictor_;

    // Coordinate transforms
    Eigen::Matrix4d table_in_world_;
    Eigen::Matrix4d origin_in_world_;
    Eigen::Matrix4d table_in_origin_;

    // Current ball state (table frame)
    Eigen::Vector3d current_pos_table_;
    Eigen::Vector3d current_vel_table_;
    Eigen::Vector3d current_pos_table_raw_;

    // Current ball state (origin frame)
    Eigen::Vector3d current_pos_origin_;
    Eigen::Vector3d current_pos_origin_raw_;

    int missed_steps_;
    std::optional<double> last_zed_ts_;

    // Prediction control
    bool   cross_detected_;
    double time_to_strike_;

    // Default/last-published values
    Eigen::Vector3d default_pos_origin_;
    Eigen::Vector3d default_vel_origin_;
    Eigen::Vector3d default_racket_normal_;
    Eigen::Vector3d default_racket_vel_;

    Eigen::Vector3d last_published_pos_;
    Eigen::Vector3d last_published_vel_;
    Eigen::Vector3d last_published_normal_;
    Eigen::Vector3d last_published_racket_;
    double          last_published_ts_;
    double          last_time_to_strike_;

    // ROS2
    rclcpp::TimerBase::SharedPtr planning_timer_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr    pos_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr    vel_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr    predict_time_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr  racket_normal_pub_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr    racket_vel_pub_;

    // CSV logging
    std::ofstream csv_file_;
    std::string   csv_path_;
};

} // namespace tabletenniszed

// ============================================================
// main()
// ============================================================

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<tabletenniszed::TableTennisNode>();
    node->initialize();  // must be called after shared_ptr is established

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
