"""
tcp_receiver.py
---------------
Host PC receiver for the KV260 DeepLabCut edge pipeline.

Receives per-frame TCP packets from the KV260 containing:
  - INT8 ResNet-101 backbone feature tensor [1x23x23x2048]
  - JPEG-compressed preview frame (for visualization)

Then:
  1. Dequantizes the INT8 tensor to FP32 (scale = 0.125, fix_point_out = 3)
  2. Injects the tensor into the original DLC TensorFlow graph at the cut point
  3. Runs the FP32 deconvolution head (50 graph nodes)
  4. Extracts 14 joint (x,y) coordinates and confidence scores
  5. Draws the skeleton on the preview frame

Packet structure (matches edge_pipeline.cpp):
  [4B] total_size   (uint32, little-endian)
  [4B] jpeg_size    (uint32, little-endian)
  [jpeg_size B] JPEG frame
  [1,083,392 B] INT8 feature tensor

Usage:
    conda activate dlc
    python host/tcp_receiver.py \
        --model path/to/snapshot-103000.pb \
        --port  5000

Requirements:
    tensorflow==1.15
    opencv-python
    numpy
"""

import socket
import struct
import argparse
import numpy as np
import cv2
import tensorflow as tf

# ── Tensor and model constants ────────────────────────────────────────────────
TENSOR_SIZE     = 23 * 23 * 2048       # 1,083,392 bytes (INT8)
TENSOR_SHAPE    = (1, 23, 23, 2048)
OUTPUT_SCALE    = 0.125                 # 2^(-fix_point_out) = 2^(-3)
NUM_JOINTS      = 14
CONF_THRESHOLD  = 0.3
HEATMAP_SIZE    = 46                    # heatmap is 46x46

# ── TensorFlow graph node names ───────────────────────────────────────────────
CUT_NODE   = "resnet_v1_101/block4/unit_3/bottleneck_v1/Relu:0"
PRED_NODE  = "pose/part_pred/block4/BiasAdd:0"

# ── Skeleton connections (pairs of joint indices) ─────────────────────────────
# Joint order: ankle1, knee1, hip1, hip2, knee2, ankle2,
#              wrist1, elbow1, shoulder1, shoulder2, elbow2, wrist2,
#              chin, forehead
SKELETON = [
    (0, 1), (1, 2),          # left leg
    (3, 4), (4, 5),          # right leg
    (2, 3),                  # hips
    (6, 7), (7, 8),          # left arm
    (9, 10), (10, 11),       # right arm
    (8, 9),                  # shoulders
    (8, 12), (9, 12),        # shoulders to chin
    (12, 13),                # chin to forehead
]

JOINT_COLORS = [
    (0, 255, 0),    # ankle1
    (0, 255, 0),    # knee1
    (0, 200, 0),    # hip1
    (0, 200, 0),    # hip2
    (0, 255, 0),    # knee2
    (0, 255, 0),    # ankle2
    (255, 100, 0),  # wrist1
    (255, 150, 0),  # elbow1
    (255, 200, 0),  # shoulder1
    (255, 200, 0),  # shoulder2
    (255, 150, 0),  # elbow2
    (255, 100, 0),  # wrist2
    (0, 100, 255),  # chin
    (0, 150, 255),  # forehead
]


def recv_all(sock, n):
    """Receive exactly n bytes from socket."""
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed by remote host")
        data += chunk
    return data


def load_model(model_path):
    """Load the original DLC frozen graph for FP32 head execution."""
    print(f"Loading model: {model_path}")
    with tf.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")

    session = tf.Session(graph=graph)
    print("Model loaded.")
    return session, graph


def run_head(session, graph, int8_tensor):
    """
    Dequantize INT8 tensor and run the FP32 deconvolution head.

    Parameters
    ----------
    session : tf.Session
    graph   : tf.Graph
    int8_tensor : np.ndarray, shape (1, 23, 23, 2048), dtype int8

    Returns
    -------
    heatmaps : np.ndarray, shape (1, 46, 46, 14), dtype float32
    """
    # ── Dequantize: INT8 -> FP32 ──────────────────────────────────────────────
    # x_float = x_int8 * 2^(-fix_point_out) = x_int8 * 0.125
    fp32_tensor = int8_tensor.astype(np.float32) * OUTPUT_SCALE

    # ── Inject at cut point and run FP32 head ────────────────────────────────
    cut_tensor  = graph.get_tensor_by_name(CUT_NODE)
    pred_tensor = graph.get_tensor_by_name(PRED_NODE)

    heatmaps = session.run(
        pred_tensor,
        feed_dict={cut_tensor: fp32_tensor}
    )
    return heatmaps   # shape: (1, 46, 46, 14)


def extract_joints(heatmaps, display_w, display_h):
    """
    Extract joint (x, y) pixel coordinates and confidence scores.

    Parameters
    ----------
    heatmaps  : np.ndarray, shape (1, 46, 46, 14)
    display_w : int   width of the display image
    display_h : int   height of the display image

    Returns
    -------
    joints : list of (x, y, confidence) tuples, length NUM_JOINTS
             x, y are in display image coordinates
             confidence is sigmoid(max_logit)
    """
    hm = heatmaps[0]   # shape: (46, 46, 14)

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

    joints = []
    for j in range(NUM_JOINTS):
        hm_j = hm[:, :, j]

        # Confidence = sigmoid of maximum logit value
        max_val = hm_j.max()
        confidence = sigmoid(max_val)

        if confidence < CONF_THRESHOLD:
            joints.append(None)
            continue

        # Joint location = argmax of sigmoid-transformed heatmap
        # (equivalent to argmax of raw heatmap since sigmoid is monotonic)
        iy, ix = np.unravel_index(np.argmax(hm_j), hm_j.shape)

        # Scale from heatmap space (46x46) to display image space
        x = int(ix * display_w / HEATMAP_SIZE)
        y = int(iy * display_h / HEATMAP_SIZE)

        joints.append((x, y, float(confidence)))

    return joints


def draw_skeleton(frame, joints):
    """Draw joints and skeleton connections on the frame."""
    h, w = frame.shape[:2]

    # Draw skeleton lines
    for (j1, j2) in SKELETON:
        if joints[j1] is not None and joints[j2] is not None:
            x1, y1, _ = joints[j1]
            x2, y2, _ = joints[j2]
            cv2.line(frame, (x1, y1), (x2, y2), (180, 180, 180), 2)

    # Draw joint circles
    for j, joint in enumerate(joints):
        if joint is None:
            continue
        x, y, conf = joint
        color = JOINT_COLORS[j]
        radius = 5
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, (255, 255, 255), 1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="KV260 DLC Host Receiver")
    parser.add_argument("--model", type=str,
                        default="./models/snapshot-103000.pb",
                        help="Path to original DLC frozen graph")
    parser.add_argument("--port", type=int, default=5000,
                        help="TCP port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind to")
    args = parser.parse_args()

    # Load TF model
    session, graph = load_model(args.model)

    # Start TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print(f"Listening on {args.host}:{args.port} ...")

    conn, addr = server.accept()
    print(f"Connected from {addr}")

    # FPS counter
    import time
    fps_timer = time.time()
    fps_count = 0

    try:
        while True:
            # ── Receive packet ────────────────────────────────────────────────
            # [4B total_size][4B jpeg_size][jpeg_bytes][tensor_bytes]
            header = recv_all(conn, 8)
            total_size = struct.unpack("<I", header[0:4])[0]
            jpeg_size  = struct.unpack("<I", header[4:8])[0]

            jpeg_bytes   = recv_all(conn, jpeg_size)
            tensor_bytes = recv_all(conn, TENSOR_SIZE)

            # ── Decode JPEG preview ───────────────────────────────────────────
            jpeg_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
            frame    = cv2.imdecode(jpeg_arr, cv2.IMREAD_COLOR)
            h, w     = frame.shape[:2]

            # ── Reshape INT8 tensor ───────────────────────────────────────────
            int8_tensor = np.frombuffer(tensor_bytes, dtype=np.int8)
            int8_tensor = int8_tensor.reshape(TENSOR_SHAPE)

            # ── Run FP32 head ─────────────────────────────────────────────────
            heatmaps = run_head(session, graph, int8_tensor)

            # ── Extract joints ────────────────────────────────────────────────
            joints = extract_joints(heatmaps, w, h)

            # ── Draw skeleton ─────────────────────────────────────────────────
            frame = draw_skeleton(frame, joints)

            # ── FPS overlay ───────────────────────────────────────────────────
            fps_count += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 2.0:
                fps = fps_count / elapsed
                fps_timer = time.time()
                fps_count = 0

            fps_text = f"FPS: {fps:.1f}" if 'fps' in dir() else "FPS: --"
            cv2.putText(frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # ── Count visible joints ──────────────────────────────────────────
            visible = sum(1 for j in joints if j is not None)
            max_conf = max((j[2] for j in joints if j is not None), default=0.0)
            info_text = f"Joints: {visible}/14  Conf: {max_conf:.2f}"
            cv2.putText(frame, info_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            cv2.imshow("DeepLabCut FPGA", frame)
            if cv2.waitKey(1) == 27:   # ESC to quit
                break

    except ConnectionError as e:
        print(f"Connection closed: {e}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        conn.close()
        server.close()
        session.close()
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
