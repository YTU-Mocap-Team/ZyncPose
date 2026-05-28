/**
 * edge_pipeline.cpp
 * -----------------
 * Real-time inference pipeline for the AMD Kria KV260.
 *
 * Pipeline:
 *   AR1335 camera (via GStreamer + AP1302 ISP)
 *     -> NV12 frame (960x540)
 *     -> OpenCV: NV12 -> BGR -> resize 368x368
 *     -> Scale x0.5 (uint8 -> INT8, fix_point_in = -1)
 *     -> DPU DPUCZDX8G B3136: ResNet-101 INT8 backbone
 *     -> Output tensor [1x23x23x2048], INT8
 *     -> TCP socket: send to host PC
 *
 * TCP Packet structure (sent per frame):
 *   [4B] total_size   : total packet size in bytes (uint32, little-endian)
 *   [4B] jpeg_size    : JPEG frame size in bytes   (uint32, little-endian)
 *   [N B] jpeg_frame  : JPEG-compressed 640x360 BGR frame (for visualization)
 *   [1,083,392 B] tensor : raw INT8 feature tensor [1x23x23x2048]
 *
 * Build (on KV260, inside Docker):
 *   g++ -std=c++17 -O2 -o edge_pipeline edge_pipeline.cpp \
 *       $(pkg-config --cflags --libs opencv4) \
 *       -lvart-runner -lxir -lvitis_ai_library-dpu_task -lpthread
 *
 * Run:
 *   ./edge_pipeline --model ./dlc_human_fpga_v2.xmodel \
 *                   --host  192.168.137.1 \
 *                   --port  5000
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <cstring>
#include <cassert>

// Networking
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>

// OpenCV
#include <opencv2/opencv.hpp>

// Vitis AI Runtime
#include <vart/runner.hpp>
#include <vart/runner_ext.hpp>
#include <xir/graph/graph.hpp>
#include <xir/tensor/tensor.hpp>

// ── Constants ────────────────────────────────────────────────────────────────
constexpr int    CAM_WIDTH      = 960;
constexpr int    CAM_HEIGHT     = 540;
constexpr int    DPU_INPUT_SIZE = 368;
constexpr int    TENSOR_SIZE    = 23 * 23 * 2048;   // 1,083,392 bytes (INT8)
constexpr float  INPUT_SCALE    = 0.5f;              // fix_point_in = -1
constexpr int    JPEG_WIDTH     = 640;
constexpr int    JPEG_HEIGHT    = 360;
constexpr int    JPEG_QUALITY   = 60;
constexpr int    NUM_RUNNERS    = 3;                 // parallel DPU runners

// ── GStreamer pipeline for AR1335 via AP1302 ISP ─────────────────────────────
const std::string GSTREAMER_PIPELINE =
    "mediasrcbin media-device=/dev/media0 "
    "v4l2src0::io-mode=dmabuf v4l2src0::stride-align=256 "
    "! video/x-raw,width=960,height=540,format=NV12,framerate=30/1 "
    "! appsink drop=true max-buffers=1 sync=false";

// ── Shared state for producer/consumer threads ────────────────────────────────
struct FramePayload {
    std::vector<uint8_t> jpeg_data;
    std::vector<int8_t>  tensor_data;
    bool                 valid = false;
};

std::mutex          g_payload_mutex;
FramePayload        g_latest_payload;
std::atomic<bool>   g_running{true};

// ── FPS counter ───────────────────────────────────────────────────────────────
std::atomic<int>    g_frames_sent{0};

// ── Connect TCP socket to host PC ─────────────────────────────────────────────
int connect_tcp(const std::string& host_ip, int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) { perror("socket"); return -1; }

    // Disable Nagle's algorithm for low latency
    int yes = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons(port);
    inet_pton(AF_INET, host_ip.c_str(), &addr.sin_addr);

    if (connect(sock, (sockaddr*)&addr, sizeof(addr)) < 0) {
        perror("connect");
        close(sock);
        return -1;
    }

    std::cout << "Connected to " << host_ip << ":" << port << std::endl;
    return sock;
}

// ── Send exactly n bytes, blocking ───────────────────────────────────────────
bool send_all(int sock, const void* data, size_t n) {
    const char* ptr = static_cast<const char*>(data);
    size_t sent = 0;
    while (sent < n) {
        ssize_t r = send(sock, ptr + sent, n - sent, MSG_NOSIGNAL);
        if (r <= 0) return false;
        sent += r;
    }
    return true;
}

// ── Network thread: sends payloads from the shared buffer ────────────────────
void network_thread(const std::string& host_ip, int port) {
    int sock = -1;

    while (g_running) {
        // Reconnect if disconnected
        if (sock < 0) {
            std::cout << "[NET] Connecting to host ..." << std::endl;
            sock = connect_tcp(host_ip, port);
            if (sock < 0) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
                continue;
            }
        }

        // Grab latest payload
        FramePayload payload;
        {
            std::lock_guard<std::mutex> lock(g_payload_mutex);
            if (!g_latest_payload.valid) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            payload = g_latest_payload;
            g_latest_payload.valid = false;
        }

        // Build packet
        uint32_t jpeg_size  = payload.jpeg_data.size();
        uint32_t total_size = 4 + jpeg_size + TENSOR_SIZE;  // 4B = jpeg_size field

        // Send: [total_size][jpeg_size][jpeg_bytes][tensor_bytes]
        if (!send_all(sock, &total_size, 4)              ||
            !send_all(sock, &jpeg_size,  4)              ||
            !send_all(sock, payload.jpeg_data.data(), jpeg_size) ||
            !send_all(sock, payload.tensor_data.data(), TENSOR_SIZE)) {
            std::cerr << "[NET] Send failed. Reconnecting ..." << std::endl;
            close(sock);
            sock = -1;
        } else {
            g_frames_sent++;
        }
    }

    if (sock >= 0) close(sock);
}

// ── DPU inference thread ─────────────────────────────────────────────────────
void inference_thread(const std::string& model_path,
                      const std::string& host_ip, int port) {
    // Load xmodel
    auto graph    = xir::Graph::deserialize(model_path);
    auto root     = graph->get_root_subgraph();
    auto children = root->get_children();

    // Find the DPU subgraph
    xir::Subgraph* dpu_subgraph = nullptr;
    for (auto* child : children) {
        if (child->get_attr<std::string>("device") == "DPU") {
            dpu_subgraph = child;
            break;
        }
    }
    if (!dpu_subgraph) {
        throw std::runtime_error("No DPU subgraph found in .xmodel");
    }

    // Create NUM_RUNNERS parallel RunnerExt instances
    std::vector<std::unique_ptr<vart::RunnerExt>> runners;
    for (int i = 0; i < NUM_RUNNERS; i++) {
        runners.push_back(
            vart::RunnerExt::create_runner(dpu_subgraph, "run")
        );
    }

    // Get input/output tensor shapes from the first runner
    auto in_tensors  = runners[0]->get_input_tensors();
    auto out_tensors = runners[0]->get_output_tensors();
    assert(in_tensors.size()  >= 1);
    assert(out_tensors.size() >= 1);

    // Open camera
    cv::VideoCapture cap(GSTREAMER_PIPELINE, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        throw std::runtime_error(
            "Failed to open camera. Check GStreamer pipeline and overlay."
        );
    }
    std::cout << "Camera opened. Starting inference at 1080p ..." << std::endl;

    // Start network thread
    std::thread net_thread(network_thread, host_ip, port);

    // FPS timer
    auto fps_timer = std::chrono::high_resolution_clock::now();
    int  fps_count = 0;

    int runner_idx = 0;

    while (g_running) {
        cv::Mat frame_nv12;
        if (!cap.read(frame_nv12)) continue;

        // NV12 -> BGR
        cv::Mat frame_bgr;
        cv::cvtColor(frame_nv12, frame_bgr, cv::COLOR_YUV2BGR_NV12);

        // Resize to DPU input size (368x368)
        cv::Mat frame_resized;
        cv::resize(frame_bgr, frame_resized,
                   cv::Size(DPU_INPUT_SIZE, DPU_INPUT_SIZE),
                   0, 0, cv::INTER_LINEAR);

        // BGR -> RGB
        cv::Mat frame_rgb;
        cv::cvtColor(frame_resized, frame_rgb, cv::COLOR_BGR2RGB);

        // ── Preprocess: scale uint8 -> INT8 using fix_point_in = -1 ──────────
        // in_ptr[i] = (int8_t)((float)src[i] * 2^fix_point_in)
        //           = (int8_t)(src[i] * 0.5)
        // ──────────────────────────────────────────────────────────────────────
        auto& runner = runners[runner_idx % NUM_RUNNERS];
        runner_idx++;

        auto in_buffers  = runner->get_inputs();
        auto out_buffers = runner->get_outputs();

        int8_t* in_ptr = reinterpret_cast<int8_t*>(
            in_buffers[0]->data(std::vector<int>{0, 0, 0, 0}).first
        );

        const uint8_t* src = frame_rgb.data;
        int total_pixels = DPU_INPUT_SIZE * DPU_INPUT_SIZE * 3;
        for (int i = 0; i < total_pixels; i++) {
            in_ptr[i] = static_cast<int8_t>(
                static_cast<float>(src[i]) * INPUT_SCALE
            );
        }

        // ── Run DPU inference ─────────────────────────────────────────────────
        auto v = runner->execute_async(in_buffers, out_buffers);
        runner->wait(v.first, -1);

        // ── Sync output tensor from DPU memory to CPU memory ─────────────────
        out_buffers[0]->sync_for_read(0, TENSOR_SIZE);

        const int8_t* out_ptr = reinterpret_cast<const int8_t*>(
            out_buffers[0]->data(std::vector<int>{0, 0, 0, 0}).first
        );

        // ── Package payload for network thread ────────────────────────────────
        // JPEG-compress a small preview frame (640x360)
        cv::Mat preview;
        cv::resize(frame_bgr, preview, cv::Size(JPEG_WIDTH, JPEG_HEIGHT));
        std::vector<uint8_t> jpeg_buf;
        cv::imencode(".jpg", preview, jpeg_buf,
                     {cv::IMWRITE_JPEG_QUALITY, JPEG_QUALITY});

        // Copy tensor
        std::vector<int8_t> tensor_buf(out_ptr, out_ptr + TENSOR_SIZE);

        // Store as latest payload (drop policy: newest replaces old)
        {
            std::lock_guard<std::mutex> lock(g_payload_mutex);
            g_latest_payload.jpeg_data   = std::move(jpeg_buf);
            g_latest_payload.tensor_data = std::move(tensor_buf);
            g_latest_payload.valid       = true;
        }

        // ── FPS counter ───────────────────────────────────────────────────────
        fps_count++;
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - fps_timer).count();
        if (elapsed >= 2.0) {
            double fps = fps_count / elapsed;
            std::cout << "[DPU] FPS: " << std::fixed
                      << std::setprecision(1) << fps
                      << "  Sent: " << g_frames_sent.load() << " frames"
                      << std::endl;
            fps_count = 0;
            fps_timer = now;
        }
    }

    cap.release();
    net_thread.join();
}


int main(int argc, char* argv[]) {
    // ── Parse arguments ───────────────────────────────────────────────────────
    std::string model_path = "./dlc_human_fpga_v2.xmodel";
    std::string host_ip    = "192.168.137.1";
    int         port       = 5000;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--host" && i + 1 < argc) host_ip = argv[++i];
        else if (arg == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    }

    std::cout << "=== KV260 DeepLabCut Edge Pipeline ===" << std::endl;
    std::cout << "Model  : " << model_path << std::endl;
    std::cout << "Host   : " << host_ip << ":" << port << std::endl;
    std::cout << "Runners: " << NUM_RUNNERS << std::endl;
    std::cout << "=======================================" << std::endl;

    try {
        inference_thread(model_path, host_ip, port);
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
