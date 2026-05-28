"""
graph_surgery.py
----------------
Performs two operations on the quantized DeepLabCut frozen graph:

  1. Fix the dynamic input shape (-1, 368, 368, 3) -> (1, 368, 368, 3)
     The Vitis AI compiler rejects graphs with dynamic batch dimensions.

  2. Strip the unsupported prediction head nodes (transposed convolutions).
     The Xilinx DPUCZDX8G DPU does not support Conv2DBackpropInput
     (transposed convolution). Attempting to compile a graph with these
     nodes causes the Vitis AI compiler to abort late in the mapping stage
     with a KeyError: 'shape' exception.

     The fix is to cut the graph at the last backbone activation tensor
     and discard everything after it. The discarded prediction head nodes
     are then executed separately on the host PC via feed_dict injection.

Cut node:
    resnet_v1_101/block4/unit_3/bottleneck_v1/Relu/aquant
    (the quantized version of the final backbone ReLU activation,
     fix_point=3, dequantization scale=0.125)

Output shape at cut node: [1, 23, 23, 2048]

Usage:
    python graph_surgery.py \
        --input  ./quantized_v2/quantize_eval_model.pb \
        --output ./stripped_model.pb

Run inside the Vitis AI Docker container (vitis-ai-tensorflow conda env).
"""

import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


# ── Cut point: the last tensor produced by the ResNet-101 backbone ─────────
CUT_NODE = "resnet_v1_101/block4/unit_3/bottleneck_v1/Relu/aquant"

# ── Input tensor name ────────────────────────────────────────────────────────
INPUT_NODE = "Placeholder"

# ── Fixed input shape (batch=1 hardcoded for DPU compilation) ───────────────
FIXED_SHAPE = [1, 368, 368, 3]


def fix_input_shape(graph_def):
    """
    Replace the dynamic input shape (-1, 368, 368, 3) with a fixed
    shape (1, 368, 368, 3). The Vitis AI compiler requires a static
    batch dimension.
    """
    for node in graph_def.node:
        if node.name == INPUT_NODE:
            # Clear existing shape attribute and set fixed shape
            node.attr["shape"].shape.CopyFrom(
                tf.TensorShape(FIXED_SHAPE).as_proto()
            )
            print(f"  Fixed shape of '{INPUT_NODE}' to {FIXED_SHAPE}")
            break
    return graph_def


def strip_prediction_head(graph_def):
    """
    Extract only the subgraph up to and including the cut node.
    This removes all 50 transposed convolution nodes in the prediction head.
    """
    stripped = graph_util.extract_sub_graph(graph_def, [CUT_NODE])
    original_count = len(graph_def.node)
    stripped_count  = len(stripped.node)
    removed_count   = original_count - stripped_count
    print(f"  Original graph : {original_count} nodes")
    print(f"  Stripped graph : {stripped_count} nodes")
    print(f"  Removed        : {removed_count} nodes (prediction head)")
    return stripped


def main():
    parser = argparse.ArgumentParser(description="Graph surgery for DLC -> DPU")
    parser.add_argument("--input",  type=str,
                        default="./quantized_v2/quantize_eval_model.pb",
                        help="Path to quantized frozen graph")
    parser.add_argument("--output", type=str,
                        default="./stripped_model.pb",
                        help="Path for the stripped output graph")
    args = parser.parse_args()

    print(f"Loading graph from: {args.input}")
    with tf.gfile.GFile(args.input, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    print(f"  Loaded {len(graph_def.node)} nodes.")

    print("\nStep 1: Fixing dynamic input shape ...")
    graph_def = fix_input_shape(graph_def)

    print("\nStep 2: Stripping prediction head ...")
    graph_def = strip_prediction_head(graph_def)

    print(f"\nSaving stripped graph to: {args.output}")
    with tf.gfile.GFile(args.output, "wb") as f:
        f.write(graph_def.SerializeToString())

    print("\nGraph surgery complete.")
    print(f"  Cut node   : {CUT_NODE}")
    print(f"  Output shape at cut: [1, 23, 23, 2048]  (fix_point=3, scale=0.125)")
    print(f"  Next step  : run compilation/compile.sh")


if __name__ == "__main__":
    main()
