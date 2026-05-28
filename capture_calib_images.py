"""
capture_calib_images.py
-----------------------
Run this ON THE KV260 BOARD (inside the Docker container) to capture
100 calibration frames from the AR1335 camera and save them to disk.

These images are then transferred to the host PC and used by calib_script.py
during the vai_q_tensorflow quantization step.

Usage (on KV260, inside Docker container):
    python capture_calib_images.py --output ./calib_images --count 100

Requirements:
    - kv260-smartcam overlay loaded:
        sudo xmutil unloadapp
        sudo xmutil loadapp kv260-smartcam
    - Docker launched with /dev mounted (see edge/README.md)
"""

import cv2
import os
import argparse

# ── GStreamer pipeline string for AR1335 via AP1302 ISP ──────────────────────
GSTREAMER_PIPELINE = (
    "mediasrcbin media-device=/dev/media0 "
    "v4l2src0::io-mode=dmabuf v4l2src0::stride-align=256 "
    "! video/x-raw,width=960,height=540,format=NV12,framerate=30/1 "
    "! appsink drop=true max-buffers=1 sync=false"
)


def main():
    parser = argparse.ArgumentParser(description="Capture calibration frames from AR1335")
    parser.add_argument("--output", type=str, default="./calib_images",
                        help="Directory to save captured frames")
    parser.add_argument("--count", type=int, default=100,
                        help="Number of frames to capture")
    parser.add_argument("--interval", type=int, default=5,
                        help="Capture every N-th frame (avoids near-duplicate frames)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera. Make sure:\n"
            "  1. kv260-smartcam overlay is loaded\n"
            "  2. Docker was launched with --privileged and /dev mounted\n"
            "  3. /dev/media0 exists"
        )

    print(f"Capturing {args.count} frames to {args.output} ...")
    print("Move the camera around slightly to vary viewpoints.")

    saved   = 0
    frame_n = 0

    while saved < args.count:
        ret, frame_nv12 = cap.read()
        if not ret:
            print("Warning: failed to read frame, retrying ...")
            continue

        frame_n += 1
        if frame_n % args.interval != 0:
            continue

        # NV12 -> BGR (960x540)
        frame_bgr = cv2.cvtColor(frame_nv12, cv2.COLOR_YUV2BGR_NV12)

        filename = os.path.join(args.output, f"calib_{saved:04d}.jpg")
        cv2.imwrite(filename, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved += 1
        if saved % 10 == 0:
            print(f"  Saved {saved}/{args.count}")

    cap.release()
    print(f"Done. {saved} calibration images saved to {args.output}")
    print("Transfer these to your host PC before running quantization.")


if __name__ == "__main__":
    main()
