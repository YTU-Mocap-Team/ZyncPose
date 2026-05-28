"""
calib_script.py
---------------
Calibration image feeder for the Vitis AI vai_q_tensorflow quantizer.

CRITICAL: Images MUST be fed as raw uint8 values in [0, 255].
Do NOT normalize to [0, 1].

The DeepLabCut frozen graph contains an internal mean-subtraction node
(sub_node) that expects raw uint8 input. If you feed normalized [0, 1]
values, the quantizer observes activation distributions 256x smaller
than at runtime, producing wrong fix_point assignments throughout the
network. This reduces maximum joint confidence from 0.92 to below 0.02,
effectively disabling all joint detection.

Usage (called automatically by vai_q_tensorflow via --input_fn flag):
    vai_q_tensorflow quantize ... --input_fn calib_script.calib_input ...
"""

import os
import glob
import numpy as np
import cv2

# ── Configuration ─────────────────────────────────────────────────────────────
CALIB_IMAGE_DIR = "./calib_images"   # folder of calibration images (JPG or PNG)
INPUT_HEIGHT    = 368                # DPU input height (must match model)
INPUT_WIDTH     = 368                # DPU input width  (must match model)
INPUT_CHANNELS  = 3                  # RGB


def calib_input(iter):
    """
    Called by vai_q_tensorflow once per calibration iteration.
    Returns a dict mapping input tensor name to a numpy array.

    Parameters
    ----------
    iter : int
        Current calibration iteration index (0-based).

    Returns
    -------
    dict
        {'Placeholder': np.ndarray of shape [1, 368, 368, 3], dtype=np.float32}
        Values are in range [0, 255] — NOT normalized.
    """
    image_paths = sorted(glob.glob(os.path.join(CALIB_IMAGE_DIR, "*.jpg")) +
                         glob.glob(os.path.join(CALIB_IMAGE_DIR, "*.png")))

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {CALIB_IMAGE_DIR}. "
            "Capture frames from the AR1335 camera first."
        )

    # Cycle through images if iter exceeds the number of images
    path = image_paths[iter % len(image_paths)]

    # Read image in BGR, convert to RGB (DLC model expects RGB)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise IOError(f"Could not read image: {path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img_resized = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT),
                             interpolation=cv2.INTER_LINEAR)

    # ── CRITICAL ──────────────────────────────────────────────────────────────
    # Convert to float32 but DO NOT divide by 255.
    # Values stay in [0, 255]. The model's internal sub node handles
    # mean subtraction (mu = [123.68, 116.78, 103.94]).
    # ──────────────────────────────────────────────────────────────────────────
    img_float = img_resized.astype(np.float32)   # range: [0.0, 255.0]

    # Add batch dimension: [H, W, C] -> [1, H, W, C]
    img_batch = np.expand_dims(img_float, axis=0)

    return {"Placeholder": img_batch}


if __name__ == "__main__":
    # Quick sanity check — run standalone to verify images load correctly
    sample = calib_input(0)
    arr = sample["Placeholder"]
    print(f"Shape : {arr.shape}")          # should be (1, 368, 368, 3)
    print(f"dtype : {arr.dtype}")          # should be float32
    print(f"min   : {arr.min():.1f}")      # should be >= 0.0
    print(f"max   : {arr.max():.1f}")      # should be <= 255.0
    print(f"mean  : {arr.mean():.1f}")     # should be ~100-150 for typical images
    print("Calibration feeder OK.")
