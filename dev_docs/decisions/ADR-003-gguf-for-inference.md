# ADR-003: GGUF Quantization for Edge & CPU Execution

**Date:** 2026-09-25  
**Status:** Accepted  
**Deciders:** Nabin  

---

## 1. Context & Problem Statement

While PyTorch (`.pt` checkpoints) is ideal for pretraining on NVIDIA GPUs, deploying PyTorch for fast interactive inference on local CPU laptops or lightweight web servers carries heavy overhead:
- PyTorch wheel size (>2GB).
- High memory usage for full-precision (FP32/FP16) weight tensors.
- Lack of SIMD/AVX-512 optimized quantized integer kernels out-of-the-box.

---

## 2. Considered Options

1. **ONNX Runtime (Open Neural Network Exchange):** Broad support, but tensor shape dynamics and custom RoPE kernels can be brittle.
2. **TorchScript / TorchDynamo:** Tied to Python and the PyTorch C++ runtime.
3. **GGUF (llama.cpp binary format):** Zero-dependency C/C++ runtime, memory-mapped I/O (`mmap`), 4-bit/8-bit quantization support, cross-platform.

---

## 3. Decision Outcome

**Chosen Solution:** **Option 3 (GGUF Export Pipeline in `src/export/gguf.py`)**

We wrote a custom Python converter mapping Nebium's RoPE, RMSNorm, and SwiGLU weights directly into GGUF v3 format.

### Key Results:
- `nebium_base` exports to ~12MB `.gguf` file.
- `nebium_762m` quantized to Q4_K_M runs at >35 tokens/sec on modern laptop CPUs without GPU acceleration.
- Published to Hugging Face Hub as `nabin2004/nebium-large-gguf`.
