cmake_minimum_required(VERSION 3.10)
project(edge_pipeline CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

# ── Find OpenCV ───────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ── Vitis AI Runtime libraries ────────────────────────────────────────────────
# These are installed in the Vitis AI Docker image on the KV260
set(VART_LIBS
    vart-runner
    xir
    vitis_ai_library-dpu_task
)

# ── Build target ──────────────────────────────────────────────────────────────
add_executable(edge_pipeline edge_pipeline.cpp)

target_link_libraries(edge_pipeline
    ${OpenCV_LIBS}
    ${VART_LIBS}
    pthread
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS edge_pipeline DESTINATION /home/ubuntu/posedetect)
