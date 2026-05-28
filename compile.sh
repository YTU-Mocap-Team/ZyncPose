#!/bin/bash
# compile.sh
# ----------
# Compiles the stripped DeepLabCut ResNet-101 backbone graph for the
# Xilinx DPUCZDX8G B3136 DPU on the AMD Kria KV260.
#
# Run this INSIDE the Vitis AI Docker container:
#   conda activate vitis-ai-tensorflow
#   bash compilation/compile.sh
#
# Prerequisites:
#   - stripped_model.pb produced by quantization/graph_surgery.py
#   - compilation/arch_b3136.json (DPU architecture descriptor)
#
# Output:
#   ./compiled_v2/dlc_human_fpga_v2.xmodel   (~46 MB, deploy to KV260)
#
# Expected compiler output:
#   [UNILOG][INFO] Total device subgraph number 4, DPU subgraph number 1
#   [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B3136

set -e

# ── Paths ────────────────────────────────────────────────────────────────────
STRIPPED_MODEL="./stripped_model.pb"
ARCH_FILE="./compilation/arch_b3136.json"
OUTPUT_DIR="./compiled_v2"
NET_NAME="dlc_human_fpga_v2"

# ── Sanity checks ────────────────────────────────────────────────────────────
if [ ! -f "$STRIPPED_MODEL" ]; then
    echo "ERROR: Stripped model not found at $STRIPPED_MODEL"
    echo "Run: python quantization/graph_surgery.py"
    exit 1
fi

if [ ! -f "$ARCH_FILE" ]; then
    echo "ERROR: Architecture file not found at $ARCH_FILE"
    echo "Check that compilation/arch_b3136.json exists."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Compiling for DPUCZDX8G_ISA1_B3136 ..."
echo ""

# ── Run compilation ──────────────────────────────────────────────────────────
vai_c_tensorflow \
    --frozen_pb "$STRIPPED_MODEL" \
    --arch      "$ARCH_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --net_name  "$NET_NAME"

echo ""
echo "Compilation complete."
echo "Output: $OUTPUT_DIR/${NET_NAME}.xmodel"
echo ""
echo "Verify the result contains exactly 1 DPU subgraph:"
echo "  xdputil xmodel $OUTPUT_DIR/${NET_NAME}.xmodel -l"
echo ""
echo "Next step: copy .xmodel to the KV260"
echo "  scp $OUTPUT_DIR/${NET_NAME}.xmodel ubuntu@<board_ip>:/home/ubuntu/posedetect/"
