# **Edge-FPGA Deployment of DeepLabCut for Real-Time Human Pose Estimation**

Implementation code for the paper:

> **Edge-FPGA Deployment of DeepLabCut for Real-Time Human Pose Estimation: A Distributed INT8/FP32 Hybrid Inference Architecture**  
> Mohamed Elmahlavy, Basim Elmashharavi, Bahadır ÇATALBAŞ 
> CODIT 2026

---

## What This Is

This repository contains the full pipeline for deploying the DeepLabCut ResNet-101 human pose estimation backbone on an AMD Kria KV260 FPGA board, with the deconvolution prediction head running on a host PC in FP32. The system runs at **13.1 FPS** on the KV260 with joint detection confidence up to **0.92** on visible keypoints.

---

## Hardware Required

* **AMD Kria KV260** Vision AI Starter Kit
* **AR1335** camera module *(comes with KV260)*
* **Host PC** (Windows or Linux) connected via Gigabit Ethernet
* **Vitis AI 2.5** Docker image *(for quantization and compilation, runs on host PC)*

---

## System Overview

graph TD
    %% -- Subgraph: KV260 (Edge) --
    subgraph KV260 ["<br/><b>KV260 (Edge)</b><br/>───────"]
        direction TB
        Camera[AR1335 Camera<br/>960×540 NV12]:::hardware
        
        AP1302_ISP[AP1302 ISP<br/>hardware RAW→NV12]:::fpga_accel
        
        OpenCV[OpenCV<br/>NV12→BGR, resize 368×368]:::cpu_sw
        
        Scale[Scale ×0.5<br/>uint8 → INT8]:::cpu_sw
        
        DPU[<b>DPU B3136: ResNet-101 INT8</b><br/>FPGA Acceleration]:::fpga_accel
        
        Edge_Out[Output INT8 Tensor<br/>1×23×23×2048]:::tensor

        Camera --> AP1302_ISP
        AP1302_ISP --> OpenCV
        OpenCV --> Scale
        Scale --> DPU
        DPU --> Edge_Out
    end

    %% -- Connection --
    Edge_Out -- "TCP<br/>(Dequantized tensor)" --> Host_In:::tcp_comm

    %% -- Subgraph: Host PC --
    subgraph HostPC ["<br/><b>Host PC</b><br/>───────"]
        direction TB
        Host_In[Receive Tensor<br/>INT8 1×23×23×2048]:::tensor
        
        Dequantize[Dequantize ×0.125<br/>INT8 → FP32]:::cpu_sw
        
        TensorFlow[TensorFlow: deconv heads<br/>(FP32)]:::cpu_sw
        
        Heatmaps[Heatmaps<br/>1×46×46×14]:::tensor
        
        PostProcess[Sigmoid + argmax<br/>→ x,y per joint]:::cpu_sw
        
        Visualization[Confidence scores +<br/>visualization]:::cpu_sw

        Host_In --> Dequantize
        Dequantize --> TensorFlow
        TensorFlow --> Heatmaps
        Heatmaps --> PostProcess
        PostProcess --> Visualization
    end

    %% -- Style Definitions --
    classDef hardware fill:#e1f5fe,stroke:#01579b,stroke-width:2px,rx:8,ry:8,color:#000;
    classDef fpga_accel fill:#ffe0b2,stroke:#ef6c00,stroke-width:2px,rx:8,ry:8,color:#000;
    classDef cpu_sw fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:8,ry:8,color:#000;
    classDef tensor fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,stroke-dasharray: 5 5,rx:5,ry:5,color:#000;
    classDef tcp_comm stroke:#b71c1c,stroke-width:2px,color:#b71c1c,font-weight:bold;

    %% Add legend (Optional but helpful)
    subgraph Legend ["Legend"]
        direction LR
        LegHW[Hardware]:::hardware
        LegFPGA[FPGA Accel]:::fpga_accel
        LegCPU[CPU/Software]:::cpu_sw
        LegTensor[Tensor]:::tensor
    end

---

## Step-by-Step Setup

### Step 1: Download the DeepLabCut Model

```bash
pip install dlclibrary
python -c "from dlclibrary import download_huggingface_model; \
           download_huggingface_model('full_human')"

```

This downloads `snapshot-103000.pb` (323 MB).

---

### Step 2: Run Quantization (on Host PC, inside Vitis AI Docker)

Start the Vitis AI Docker container:

```bash
docker run --rm -it \
  -v /path/to/this/repo:/workspace \
  xilinx/vitis-ai-cpu:latest bash

```

Inside the container:

```bash
conda activate vitis-ai-tensorflow
cd /workspace
bash quantization/quantize.sh

```

**Critical note:** The calibration images MUST be fed as raw uint8 values in `[0, 255]`. Do NOT normalize to `[0, 1]`. Feeding normalized images reduces maximum joint confidence from 0.92 to below 0.02.
See `calibration/calib_script.py` for the correct implementation.

---

### Step 3: Run Graph Surgery

After quantization, strip the unsupported deconv nodes:

```bash
python quantization/graph_surgery.py

```

This uses `graph_util.extract_sub_graph` to cut the graph at `resnet_v1_101/block4/unit_3/bottleneck_v1/Relu/aquant`, removing the 50 transposed convolution nodes that the DPU cannot handle.

---

### Step 4: Compile for the DPU (still inside Docker)

```bash
bash compilation/compile.sh

```

Expected output:

```text
[UNILOG][INFO] Total device subgraph number 4, DPU subgraph number 1
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B3136

```

This produces `dlc_human_fpga_v2.xmodel` (46 MB).

---

### Step 5: Deploy on KV260

Copy the xmodel to the board:

```bash
scp compiled_v2/dlc_human_fpga_v2.xmodel ubuntu@<board_ip>:/home/ubuntu/posedetect/

```

Load the smartcam overlay:

```bash
sudo xmutil unloadapp
sudo xmutil loadapp kv260-smartcam

```

Build and run the C++ inference pipeline:

```bash
cd /home/ubuntu/posedetect
g++ -std=c++17 -O2 -o edge_pipeline edge_pipeline.cpp \
  $(pkg-config --cflags --libs opencv4) \
  -lvart-runner -lxir -lvitis_ai_library-dpu_task -lpthread
./edge_pipeline --host <laptop_ip> --port 5000

```

---

### Step 6: Run the Host PC Receiver

```bash
conda activate dlc
python host/tcp_receiver.py \
  --model path/to/snapshot-103000.pb \
  --port 5000

```

You should see the live skeleton visualization at 13.1 FPS.

---

## Quantization Details

| Tensor | Name | fix_point | Scale |
| --- | --- | --- | --- |
| Input | `sub_inserted_fix_0` | -1 | 0.5 |
| Backbone output | `resnet_v1_101/.../Relu/aquant` | 3 | 0.125 |

Input preprocessing on the KV260:

```c
in_ptr[i] = (int8_t)(src[i] * 0.5);

```

Dequantization on the host PC:

```python
x_float = x_int8 * 0.125

```

---

## Results

### Throughput

| Configuration | FPS |
| --- | --- |
| 1 runner, Python | 8.4 |
| 1 runner, Python + heatmap | 6.4 |
| 3 runners, Python TCP | 13.1 |
| 3 runners, DPU benchmark only | 13.2 |

### Comparison with Published Embedded Benchmarks

| Platform | Price | Model | FPS |
| --- | --- | --- | --- |
| KV260 (this work) | $200 | ResNet-101 | 13.1 |
| NVIDIA Jetson Xavier | $699 | ResNet-50 | 19 |
| NVIDIA Jetson TX2 | $400 | ResNet-50 | 6 |

*Source for Jetson numbers: Kane et al. 2020*

---

## Known Issues and Gotchas

1. **Calibration input distribution** — The biggest trap. Normalizing to `[0,1]` completely breaks the quantized model. Always feed raw uint8 `[0,255]` values to the calibrator.
2. **Dynamic shape** — The quantized graph has a dynamic batch dimension (`-1`). Hardcode it to `[1, 368, 368, 3]` before compilation or the Vitis AI compiler will reject the model.
3. **Joint count** — The `full_human` model predicts 14 joints, not 17. Verify from `pose_cfg.yaml` before deploying.
4. **Autofocus on KV260 Rev B** — The DW9790 VCM autofocus driver is not activated in the `kv260-smartcam` overlay for Revision B boards. The camera runs fixed-focus only.
5. **TCP_NODELAY** — Must be enabled on the socket or Nagle's algorithm will batch small packets and add ~40ms of latency.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{elmahlavy2026edge,
  title={Edge-{FPGA} Deployment of {DeepLabCut} for Real-Time Human Pose Estimation: A Distributed {INT8/FP32} Hybrid Inference Architecture},
  author={Elmahlavy, Mohamed and Elmashharavi, Basim},
  booktitle={2026 12th International Conference on Control, Decision and Information Technologies (CODIT)},
  year={2026}
}

```

---

## Acknowledgment
We would like to express our deepest gratitude to our professor, Bahadır ÇATALBAŞ, for his invaluable guidance, continuous support, and for generously providing the necessary hardware and resources to make this project possible. 

Additionally, this work was supported in part by the AMD University Program, and the KV260 boards were provided through the AMD Academic Program.
```


```
