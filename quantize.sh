#!/bin/bash
# quantize.sh
# -----------
# Runs the Vitis AI post-training INT8 quantization on the DeepLabCut
# full_human ResNet-101 frozen graph (snapshot-103000.pb).
#
# Run this INSIDE the Vitis AI Docker container:
#   conda activate vitis-ai-tensorflow
#   bash quantization/quantize.sh
#
# Prerequisites:
#   - snapshot-103000.pb in ./models/
#   - calibration images in ./calibration/calib_images/  (100+ JPG/PNG frames)
#   - calib_script.py in ./calibration/
#
# Output:
#   ./quantized_v2/quantize_eval_model.pb   (quantized frozen graph)
#   ./quantized_v2/deploy_model.pb          (deployment graph, not used)

set -e  # exit on first error

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_INPUT="./models/snapshot-103000.pb"
CALIB_SCRIPT="calibration/calib_script"      # Python module path (no .py)
OUTPUT_DIR="./quantized_v2"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [ ! -f "$MODEL_INPUT" ]; then
    echo "ERROR: Model not found at $MODEL_INPUT"
    echo "Download with: python -c \"from dlclibrary import download_huggingface_model; download_huggingface_model('full_human')\""
    exit 1
fi

if [ ! -d "./calibration/calib_images" ]; then
    echo "ERROR: Calibration images not found at ./calibration/calib_images/"
    echo "Run capture_calib_images.py on the KV260 first, then transfer images here."
    exit 1
fi

IMG_COUNT=$(ls calibration/calib_images/*.jpg calibration/calib_images/*.png 2>/dev/null | wc -l)
echo "Found $IMG_COUNT calibration images."
if [ "$IMG_COUNT" -lt 50 ]; then
    echo "WARNING: Fewer than 50 calibration images. Recommend at least 100."
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting quantization..."
echo "CRITICAL: calib_script.py feeds raw uint8 [0,255] values (NOT normalized)."
echo ""

# ── Run quantization ─────────────────────────────────────────────────────────
vai_q_tensorflow quantize \
    --input_frozen_graph "$MODEL_INPUT" \
    --input_nodes        Placeholder \
    --output_nodes       pose/part_pred/block4/BiasAdd,pose/locref_pred/block4/BiasAdd \
    --input_shapes       1,368,368,3 \
    --input_fn           ${CALIB_SCRIPT}.calib_input \
    --calib_iter         100 \
    --output_dir         "$OUTPUT_DIR" \
    --skip_check         1

echo ""
echo "Quantization complete. Output: $OUTPUT_DIR/"
echo ""
echo "Next step: python quantization/graph_surgery.py"
