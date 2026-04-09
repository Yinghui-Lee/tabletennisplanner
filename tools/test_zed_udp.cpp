#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace {

constexpr int kBufSize = 65536;
constexpr double kUdpTimeoutSec = 1.0;
constexpr double kStatsIntervalSec = 1.0;
constexpr std::size_t kFreqWindow = 10;

struct Options {
    std::string bind_addr = "0.0.0.0";
    int port = 8888;
    bool quiet = false;
    bool print_data = true;
    bool print_stats = true;
};

bool parse_args(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0)
            return false;
        if (std::strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            o.bind_addr = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            o.port = std::atoi(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--quiet") == 0) {
            o.quiet = true;
            o.print_data = false;
            continue;
        }
        if (std::strcmp(argv[i], "--no-data") == 0) {
            o.print_data = false;
            continue;
        }
        if (std::strcmp(argv[i], "--no-stats") == 0) {
            o.print_stats = false;
            continue;
        }
        std::fprintf(stderr, "Unknown argument: %s\n", argv[i]);
        return false;
    }
    return true;
}

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [--host BIND_ADDR] [--port PORT] [--quiet] [--no-data] [--no-stats]\n"
        "  Bind must be a local IP. Sender dst must match: same machine -> 127.0.0.1 or\n"
        "  `sudo ip addr add 172.16.2.101/32 dev lo` if sender is hardcoded to that IP.\n"
        "  Stop table_tennis_node if port 8888 is already in use.\n",
        prog);
}

std::atomic<bool> g_run{true};

void on_signal(int) {
    g_run = false;
}

std::optional<double> json_timestamp(const nlohmann::json& j) {
    if (j.contains("timestamp") && j["timestamp"].is_number())
        return j["timestamp"].get<double>();
    return std::nullopt;
}

void print_packet_line(const nlohmann::json& j, int n, double recv_wall,
    const char* from_ip, int from_port,
    std::optional<double> dt_ts, std::optional<double> inst_hz) {
    auto ts = json_timestamp(j);
    if (j.contains("ball_camera") && j["ball_camera"].is_object()) {
        const auto& bc = j["ball_camera"];
        bool v = bc.value("valid", false);
        double x = bc.value("x", 0.0);
        double y = bc.value("y", 0.0);
        double z = bc.value("z", 0.0);
        std::printf(
            "[ZedTest] #%d from %s:%d ts=%.6f recv=%.6f ball_cam valid=%d [%.4f,%.4f,%.4f]",
            n, from_ip, from_port, ts.value_or(-1.0), recv_wall, v ? 1 : 0, x, y, z);
    } else if (j.contains("x") && j.contains("y") && j.contains("z")) {
        double x = j.value("x", 0.0);
        double y = j.value("y", 0.0);
        double z = j.value("z", 0.0);
        std::printf("[ZedTest] #%d from %s:%d ts=%.6f recv=%.6f pos [%.4f,%.4f,%.4f]",
            n, from_ip, from_port, ts.value_or(-1.0), recv_wall, x, y, z);
    } else {
        std::printf("[ZedTest] #%d from %s:%d recv=%.6f JSON keys: ", n, from_ip, from_port, recv_wall);
        bool first = true;
        for (auto it = j.begin(); it != j.end(); ++it) {
            if (!first) std::printf(", ");
            first = false;
            std::printf("%s", it.key().c_str());
        }
    }
    if (dt_ts.has_value())
        std::printf(" d_ts=%.6fs", *dt_ts);
    if (inst_hz.has_value())
        std::printf(" ~%.1fHz", *inst_hz);
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }
    if (opt.quiet) {
        opt.print_data = false;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGQUIT, on_signal);

    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) {
        std::perror("socket");
        return 1;
    }
    int one = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    timeval tv;
    tv.tv_sec = static_cast<time_t>(kUdpTimeoutSec);
    tv.tv_usec = static_cast<suseconds_t>((kUdpTimeoutSec - tv.tv_sec) * 1e6);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(opt.port));
    if (inet_pton(AF_INET, opt.bind_addr.c_str(), &addr.sin_addr) <= 0) {
        std::fprintf(stderr, "Invalid bind address: %s\n", opt.bind_addr.c_str());
        close(fd);
        return 1;
    }
    if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::fprintf(stderr, "bind %s:%d failed: %s\n", opt.bind_addr.c_str(), opt.port, std::strerror(errno));
        if (errno == EADDRNOTAVAIL)
            std::fprintf(stderr,
                "  This IP is not on any interface. Use 0.0.0.0 or add the address (e.g. lo alias).\n");
        if (errno == EADDRINUSE)
            std::fprintf(stderr, "  Port busy — stop table_tennis_node or other process on this port.\n");
        close(fd);
        return 1;
    }

    if (!opt.quiet) {
        std::printf("[ZedTest] listening UDP %s:%d (Ctrl+C to quit)\n", opt.bind_addr.c_str(), opt.port);
        std::printf(
            "[ZedTest] If sender targets 172.16.2.101 but this host has no such IP, packets never arrive.\n"
            "[ZedTest] Fix: send to 127.0.0.1, or: sudo ip addr add 172.16.2.101/32 dev lo\n");
    }

    std::deque<double> ts_intervals;
    std::deque<double> rx_intervals;
    std::deque<double> ts_win;
    std::deque<double> rx_win;

    long long count = 0;
    long long err_count = 0;
    double last_ts = 0.0;
    bool have_last_ts = false;
    double last_rx = 0.0;
    bool have_last_rx = false;

    auto last_stats_tp = std::chrono::steady_clock::now();
    auto last_hb_tp = std::chrono::steady_clock::now();

    char buf[kBufSize];

    while (g_run.load()) {
        sockaddr_in peer{};
        socklen_t peerlen = sizeof(peer);
        ssize_t nread = recvfrom(fd, buf, sizeof(buf) - 1, 0,
            reinterpret_cast<sockaddr*>(&peer), &peerlen);
        if (nread < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                auto now = std::chrono::steady_clock::now();
                if (!opt.quiet &&
                    std::chrono::duration<double>(now - last_hb_tp).count() >= 5.0) {
                    std::fprintf(stderr,
                        "[ZedTest] (no UDP yet, still listening on %s:%d — check sender dst IP/port)\n",
                        opt.bind_addr.c_str(), opt.port);
                    last_hb_tp = now;
                }
                continue;
            }
            std::perror("recvfrom");
            ++err_count;
            continue;
        }
        char from_ip[INET_ADDRSTRLEN]{};
        inet_ntop(AF_INET, &peer.sin_addr, from_ip, sizeof(from_ip));
        int from_port = ntohs(peer.sin_port);
        buf[nread] = '\0';

        nlohmann::json j;
        try {
            j = nlohmann::json::parse(buf, buf + nread);
        } catch (...) {
            ++err_count;
            if (!opt.quiet)
                std::fprintf(stderr, "[ZedTest] JSON parse error (%zd bytes)\n", nread);
            continue;
        }

        double recv_wall = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::optional<double> ts = json_timestamp(j);
        if (!ts.has_value())
            ts = recv_wall;

        std::optional<double> dt_ts;
        std::optional<double> inst_hz;

        if (have_last_ts) {
            dt_ts = *ts - last_ts;
            ts_intervals.push_back(*dt_ts);
            if (ts_intervals.size() > 1000)
                ts_intervals.pop_front();
            ts_win.push_back(*dt_ts);
            if (ts_win.size() > kFreqWindow)
                ts_win.pop_front();
            if (ts_win.size() > 1) {
                double s = 0.0;
                for (double x : ts_win)
                    s += x;
                double avg = s / static_cast<double>(ts_win.size());
                if (avg > 0.0)
                    inst_hz = 1.0 / avg;
            }
        }
        have_last_ts = true;
        last_ts = *ts;

        const double recv_mono = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (have_last_rx) {
            double dri = recv_mono - last_rx;
            rx_intervals.push_back(dri);
            if (rx_intervals.size() > 1000)
                rx_intervals.pop_front();
            rx_win.push_back(dri);
            if (rx_win.size() > kFreqWindow)
                rx_win.pop_front();
        }
        have_last_rx = true;
        last_rx = recv_mono;

        ++count;

        if (opt.print_data)
            print_packet_line(j, static_cast<int>(count), recv_wall, from_ip, from_port, dt_ts, inst_hz);

        if (opt.print_stats) {
            auto now_tp = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now_tp - last_stats_tp).count() >= kStatsIntervalSec) {
                double f_ts = 0.0;
                if (!ts_intervals.empty()) {
                    double sum = 0.0;
                    for (double x : ts_intervals)
                        sum += x;
                    double avg = sum / static_cast<double>(ts_intervals.size());
                    if (avg > 0.0)
                        f_ts = 1.0 / avg;
                }
                double f_rx = 0.0;
                if (!rx_intervals.empty()) {
                    double sum = 0.0;
                    for (double x : rx_intervals)
                        sum += x;
                    double avg = sum / static_cast<double>(rx_intervals.size());
                    if (avg > 0.0)
                        f_rx = 1.0 / avg;
                }
                std::printf("[ZedTest] [Hz] by_timestamp=%.2f by_recv=%.2f packets=%lld\n",
                    f_ts, f_rx, count);
                last_stats_tp = now_tp;
            }
        }
    }
    close(fd);
    if (!opt.quiet)
        std::printf("[ZedTest] exit packets=%lld errors=%lld\n", count, err_count);
    return 0;
}
