# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
lem_native -- C++ extension for GIL contention reduction in async training.

JIT-compiled via torch.utils.cpp_extension.load_inline().  Pure C++ (no CUDA
kernels) -- all tensor ops go through ATen.  The key win is releasing the GIL
during blocking operations (cudaStreamSynchronize in clip_grad_norm) and
eliminating Python loop overhead in gather/context/imagination helpers.

Falls back to pure-Python implementations if compilation fails.
"""

import logging
import os
import time

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MSVC setup -- reuse the same auto-detect from vision_kernels.py
# ---------------------------------------------------------------------------


def _setup_msvc_env():
    """Auto-detect MSVC via vswhere.exe and configure PATH/INCLUDE/LIB."""
    import subprocess
    try:
        subprocess.run(['where', 'cl'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    vswhere = os.path.join(
        os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
        'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
    if not os.path.isfile(vswhere):
        return False

    try:
        result = subprocess.run(
            [vswhere, '-latest', '-property', 'installationPath',
             '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'],
            capture_output=True, text=True, check=True)
        vs_path = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

    if not vs_path or not os.path.isdir(vs_path):
        return False

    vcvarsall = os.path.join(vs_path, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat')
    if not os.path.isfile(vcvarsall):
        return False

    try:
        result = subprocess.run(
            f'"{vcvarsall}" x64 && set',
            capture_output=True, text=True, shell=True, check=True)
    except subprocess.CalledProcessError:
        return False

    env_vars = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, _, value = line.partition('=')
            env_vars[key.upper()] = value

    for key in ('PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'UCRTVERSION',
                'UNIVERSALCRTSDKDIR', 'VCTOOLSINSTALLDIR', 'WINDOWSSDKDIR',
                'WINDOWSSDKLIBVERSION', 'WINDOWSSDKVERSION'):
        if key in env_vars:
            os.environ[key] = env_vars[key]

    log.info("MSVC environment configured from %s", vcvarsall)
    return True


# ---------------------------------------------------------------------------
# C++ source
# ---------------------------------------------------------------------------

CPP_SOURCE = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/core/InferenceMode.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>

// =========================================================================
// Cross-stream replay buffer synchronization
//
// The native inference loop writes to GPU ring buffer tensors on the
// inference CUDA stream (non-blocking copies).  The training thread reads
// them via gather_batch_native on the training stream.  Without cross-
// stream synchronization, training can read partially-written GPU data,
// causing gradient explosions.
//
// Solution: record a CUDA event after each frame's RB writes on the
// inference stream.  gather_batch_native waits on this event on the
// training stream before reading.  Cost: ~2us per event op.
// =========================================================================

static cudaEvent_t g_rb_write_event = nullptr;

// =========================================================================
// GIL-free scalar extraction
//
// .item() on a CUDA tensor triggers cudaStreamSynchronize which blocks the
// calling thread.  In async training, this holds the GIL and starves the
// inference thread.  These wrappers release the GIL during the sync.
// =========================================================================

double scalar_item_gil_free(torch::Tensor t) {
    double val;
    {
        py::gil_scoped_release release;
        val = t.item<double>();
    }
    return val;
}

bool bool_item_gil_free(torch::Tensor t) {
    bool val;
    {
        py::gil_scoped_release release;
        val = t.item<bool>();
    }
    return val;
}

std::vector<double> tensor_to_list_gil_free(torch::Tensor t) {
    // Copy CUDA tensor to CPU with GIL released (the .cpu() triggers sync)
    torch::Tensor cpu_t;
    {
        py::gil_scoped_release release;
        cpu_t = t.to(torch::kCPU).to(torch::kDouble).contiguous();
    }
    auto accessor = cpu_t.accessor<double, 1>();
    std::vector<double> result(accessor.size(0));
    for (int64_t i = 0; i < accessor.size(0); i++) {
        result[i] = accessor[i];
    }
    return result;
}


// =========================================================================
// Phase 1: clip_grad_norm_gil_free
//
// Drop-in replacement for torch.nn.utils.clip_grad_norm_ that releases the
// GIL during the cudaStreamSynchronize triggered by .item().  With
// error_if_nonfinite=False, does NO sync at all (all GPU-side).
// =========================================================================

torch::Tensor clip_grad_norm_gil_free(
    std::vector<torch::Tensor> parameters,
    double max_norm,
    double norm_type,
    bool error_if_nonfinite
) {
    // 1. Collect grads (skip params without gradients)
    std::vector<torch::Tensor> grads;
    grads.reserve(parameters.size());
    for (auto& p : parameters) {
        if (p.grad().defined()) {
            grads.push_back(p.grad().detach());
        }
    }

    if (grads.empty()) {
        return torch::tensor(0.0);
    }

    torch::Tensor total_norm;

    if (norm_type == std::numeric_limits<double>::infinity()) {
        // Inf norm: max of abs values across all grads
        std::vector<torch::Tensor> norms;
        norms.reserve(grads.size());
        for (auto& g : grads) {
            norms.push_back(g.abs().max());
        }
        total_norm = torch::max(torch::stack(norms));
    } else if (norm_type == 0.0) {
        // L0 "norm": count of nonzero elements
        double count = 0.0;
        for (auto& g : grads) {
            count += g.nonzero().size(0);
        }
        total_norm = torch::tensor(count);
    } else {
        // General Lp norm: foreach-style for efficiency
        // Compute per-tensor norms, then combine
        std::vector<torch::Tensor> norms;
        norms.reserve(grads.size());
        for (auto& g : grads) {
            norms.push_back(torch::norm(g, norm_type));
        }
        // Stack and compute combined norm
        auto stacked = torch::stack(norms);
        total_norm = torch::norm(stacked, norm_type);
    }

    // 2. Check for non-finite values (only if requested -- requires sync)
    if (error_if_nonfinite) {
        double total_norm_val;
        {
            py::gil_scoped_release release;
            total_norm_val = total_norm.item<double>();
        }
        if (!std::isfinite(total_norm_val)) {
            throw std::runtime_error(
                "The total norm of order " + std::to_string(norm_type) +
                " for gradients from `parameters` is non-finite (val=" +
                std::to_string(total_norm_val) +
                "), so it cannot be clipped. To disable this error, set "
                "`error_if_nonfinite=False`.");
        }
    }

    // 3. Clip entirely on GPU -- no .item(), no sync.
    //    clip_coef = max_norm / (total_norm + 1e-6), clamped to max 1.0
    auto clip_coef = torch::clamp(
        torch::tensor(max_norm, total_norm.options()) / (total_norm + 1e-6),
        /*min=*/{}, /*max=*/1.0);
    // In-place scale all grads (async GPU, no sync)
    for (auto& g : grads) {
        g.mul_(clip_coef);
    }

    return total_norm;
}


// =========================================================================
// Phase 2: gather_batch_native
//
// Replaces GPUReplayBuffer.gather_batch().  Eliminates Python loop + dict
// construction overhead.  All index_select ops happen in C++ with a single
// GIL release/reacquire cycle.
// =========================================================================

py::dict gather_batch_native(
    // Buffer tensors (pre-extracted from Python GPUReplayBuffer)
    torch::Tensor vision_buf,           // [capacity, max_patches, d_internal]
    torch::Tensor audio_spectral_buf,   // [capacity, 256, 2]
    torch::Tensor audio_spatial_buf,    // [capacity, 2]
    torch::Tensor proprio_buf,          // [capacity, proprio_dim]
    torch::Tensor motor_buf,            // [capacity, n_handles, features_per_handle]
    torch::Tensor tactile_buf,          // [capacity, 256, 8]
    torch::Tensor voice_buf,            // [capacity, voice_dim]
    torch::Tensor motor_capability_buf, // [capacity, cap_dim]
    torch::Tensor gaze_buf,             // [capacity, 2]
    torch::Tensor indices,              // [B] int64
    // Modality presence per slot
    torch::Tensor modality_present,     // [capacity] int32 (CPU)
    bool skip_vision
) {
    // Gather modality presence flags for requested indices (CPU tensor)
    auto idx_cpu = indices.to(torch::kCPU);
    auto idx_accessor = idx_cpu.accessor<int64_t, 1>();
    int64_t B = indices.size(0);

    // Modality bitflags (must match replay_buffer.py MOD_* constants)
    constexpr int MOD_VISION = 1 << 0;
    constexpr int MOD_AUDIO = 1 << 1;
    constexpr int MOD_PROPRIO = 1 << 2;
    constexpr int MOD_MOTOR = 1 << 3;
    constexpr int MOD_TACTILE = 1 << 4;
    constexpr int MOD_VOICE = 1 << 5;

    // Check which modalities are present in ANY sample
    int any_present = 0;
    auto mp_accessor = modality_present.accessor<int32_t, 1>();
    for (int64_t i = 0; i < B; i++) {
        any_present |= mp_accessor[idx_accessor[i]];
    }

    // Wait for any pending inference-stream RB writes to complete on this
    // (training) stream before reading.  Without this, index_select can
    // read partially-written GPU data, causing gradient explosions.
    if (g_rb_write_event != nullptr) {
        cudaStream_t cur_stream = at::cuda::getCurrentCUDAStream().stream();
        cudaStreamWaitEvent(cur_stream, g_rb_write_event, 0);
    }

    // Perform all index_select ops (async GPU, no GIL needed for ATen ops)
    // Note: py::dict construction requires GIL, so we build tensors first,
    // then construct the dict at the end.

    auto idx_dev = indices.to(vision_buf.device());

    torch::Tensor vision_out, audio_spectral_out, audio_spatial_out;
    torch::Tensor proprio_out, motor_out, tactile_out, voice_out;
    torch::Tensor motor_cap_out, gaze_out;

    bool has_vision = !skip_vision && (any_present & MOD_VISION);
    bool has_audio = (any_present & MOD_AUDIO) != 0;
    bool has_proprio = (any_present & MOD_PROPRIO) != 0;
    bool has_motor = (any_present & MOD_MOTOR) != 0;
    bool has_tactile = (any_present & MOD_TACTILE) != 0;
    bool has_voice = (any_present & MOD_VOICE) != 0;

    if (has_vision) {
        vision_out = torch::index_select(vision_buf, 0, idx_dev);
    }
    if (has_audio) {
        audio_spectral_out = torch::index_select(audio_spectral_buf, 0, idx_dev);
        audio_spatial_out = torch::index_select(audio_spatial_buf, 0, idx_dev);
    }
    if (has_proprio) {
        proprio_out = torch::index_select(proprio_buf, 0, idx_dev);
    }
    if (has_motor) {
        motor_out = torch::index_select(motor_buf, 0, idx_dev);
    }
    if (has_tactile) {
        tactile_out = torch::index_select(tactile_buf, 0, idx_dev);
    }
    if (has_voice) {
        voice_out = torch::index_select(voice_buf, 0, idx_dev);
    }
    motor_cap_out = torch::index_select(motor_capability_buf, 0, idx_dev);
    gaze_out = torch::index_select(gaze_buf, 0, idx_dev);

    // Build output dict (requires GIL -- kept brief)
    py::dict result;
    if (has_vision) {
        result["vision_backbone"] = vision_out;
    }
    if (has_audio) {
        py::dict audio_dict;
        audio_dict["spectral"] = audio_spectral_out;
        audio_dict["spatial"] = audio_spatial_out;
        result["audio"] = audio_dict;
    }
    if (has_proprio) {
        result["proprioception"] = proprio_out;
    }
    if (has_motor) {
        result["motor"] = motor_out;
    }
    if (has_tactile) {
        result["tactile"] = tactile_out;
    }
    if (has_voice) {
        result["voice"] = voice_out;
    }
    result["motor_capability"] = motor_cap_out;
    result["gaze_target"] = gaze_out;

    return result;
}


// =========================================================================
// Phase 3a: build_context_index_plan
//
// Deduplicate context window indices and compute remap positions.
// Pure integer arithmetic -- no GPU ops, no Python objects.
// =========================================================================

std::tuple<torch::Tensor, torch::Tensor> build_context_index_plan(
    torch::Tensor context_windows,  // [B, ctx_len-1] int64 (CPU)
    int64_t capacity
) {
    TORCH_CHECK(context_windows.is_cpu(), "context_windows must be CPU tensor");
    TORCH_CHECK(context_windows.dtype() == torch::kInt64, "context_windows must be int64");

    auto accessor = context_windows.accessor<int64_t, 2>();
    int64_t B = context_windows.size(0);
    int64_t ctx_count = context_windows.size(1);
    int64_t total = B * ctx_count;

    // Flatten and deduplicate
    std::vector<int64_t> all_indices(total);
    for (int64_t b = 0; b < B; b++) {
        for (int64_t c = 0; c < ctx_count; c++) {
            all_indices[b * ctx_count + c] = accessor[b][c];
        }
    }

    // Unique sorted indices
    std::unordered_set<int64_t> seen(all_indices.begin(), all_indices.end());
    std::vector<int64_t> unique_sorted(seen.begin(), seen.end());
    std::sort(unique_sorted.begin(), unique_sorted.end());

    // Build reverse map: index -> position in unique_sorted
    std::unordered_map<int64_t, int64_t> idx_to_pos;
    for (size_t i = 0; i < unique_sorted.size(); i++) {
        idx_to_pos[unique_sorted[i]] = static_cast<int64_t>(i);
    }

    // Build remap positions tensor
    auto positions = torch::empty({total}, torch::kInt64);
    auto pos_accessor = positions.accessor<int64_t, 1>();
    for (int64_t i = 0; i < total; i++) {
        pos_accessor[i] = idx_to_pos[all_indices[i]];
    }

    // Build unique indices tensor
    auto unique_tensor = torch::from_blob(
        unique_sorted.data(), {static_cast<int64_t>(unique_sorted.size())},
        torch::kInt64).clone();

    return std::make_tuple(unique_tensor, positions);
}


// =========================================================================
// Phase 3b: assemble_context_tokens
//
// Reassemble deduplicated tokens back to [B, ctx_count * n_mod, d_model].
// Replaces Python loop + index_select + stack per modality.
// =========================================================================

torch::Tensor assemble_context_tokens(
    std::vector<torch::Tensor> modality_tokens,  // list of [n_unique, d_model] tensors
    torch::Tensor positions,                      // [total] int64 (on target device)
    int64_t B,
    int64_t ctx_count,
    int64_t n_mod,
    int64_t d_model
) {
    int64_t total = B * ctx_count;
    TORCH_CHECK(positions.size(0) == total,
                "positions size mismatch: expected ", total, " got ", positions.size(0));
    TORCH_CHECK(static_cast<int64_t>(modality_tokens.size()) == n_mod,
                "modality_tokens count mismatch: expected ", n_mod,
                " got ", modality_tokens.size());

    auto device = positions.device();
    auto dtype = modality_tokens[0].dtype();

    // Output: [total, n_mod, d_model]
    auto output = torch::empty({total, n_mod, d_model},
                               torch::dtype(dtype).device(device));

    for (int64_t m = 0; m < n_mod; m++) {
        auto& tokens = modality_tokens[m];
        // tokens: [n_unique, d_model], positions: [total]
        // Gather: output[:, m, :] = tokens[positions]
        auto gathered = torch::index_select(tokens, 0, positions);
        output.select(1, m).copy_(gathered);
    }

    // Reshape to [B, ctx_count * n_mod, d_model]
    return output.reshape({B, ctx_count * n_mod, d_model});
}


// =========================================================================
// Phase 4: imagination_index_plan
//
// Compute successor indices for imagination steps.  Pure arithmetic.
// =========================================================================

std::tuple<torch::Tensor, torch::Tensor> imagination_index_plan(
    torch::Tensor primary_indices,  // [B] int64
    int64_t imagination_steps,
    int64_t capacity
) {
    TORCH_CHECK(primary_indices.dtype() == torch::kInt64,
                "primary_indices must be int64");

    int64_t B = primary_indices.size(0);
    auto idx_cpu = primary_indices.to(torch::kCPU);
    auto accessor = idx_cpu.accessor<int64_t, 1>();

    // motor_indices[s][b] = (primary[b] + s) % capacity
    // target_indices[s][b] = (primary[b] + s + 1) % capacity
    auto motor_out = torch::empty({imagination_steps, B}, torch::kInt64);
    auto target_out = torch::empty({imagination_steps, B}, torch::kInt64);
    auto motor_acc = motor_out.accessor<int64_t, 2>();
    auto target_acc = target_out.accessor<int64_t, 2>();

    for (int64_t s = 0; s < imagination_steps; s++) {
        for (int64_t b = 0; b < B; b++) {
            int64_t base = accessor[b];
            motor_acc[s][b] = (base + s) % capacity;
            target_acc[s][b] = (base + s + 1) % capacity;
        }
    }

    return std::make_tuple(motor_out, target_out);
}


// =========================================================================
// GIL-free backward pass
//
// loss.backward() dispatches to torch::autograd::backward() which is pure
// C++ (especially with compiled models where backward nodes are Inductor
// kernels).  Releasing the GIL lets the inference thread run Python code
// (frame building, kernel launches) during the 170-750ms backward pass.
// =========================================================================

void backward_gil_free(torch::Tensor loss, bool retain_graph, bool create_graph) {
    py::gil_scoped_release release;
    torch::autograd::backward({loss}, {}, retain_graph, create_graph);
}


// =========================================================================
// GIL-free tensor pair copy (weight staging / sync)
//
// _stage_weights() and maybe_sync_weights() iterate 133 pre-built
// (src, dst) tensor pairs calling dst.copy_(src, non_blocking=True).
// Each Python-level copy_() call goes through pybind11 dispatch and
// holds the GIL.  With 133 pairs, Python overhead accumulates to
// ~40ms of GIL-held time -- the #1 source of inference stalls.
//
// This function performs the entire copy loop in C++ with the GIL
// released, reducing GIL hold to <1ms (single function call overhead).
// =========================================================================

void copy_pairs_gil_free(
    std::vector<std::tuple<torch::Tensor, torch::Tensor>> pairs,
    bool non_blocking
) {
    py::gil_scoped_release release;
    for (auto& [src, dst] : pairs) {
        dst.copy_(src, /*non_blocking=*/ non_blocking);
    }
}


// =========================================================================
// Native C++ inference loop -- GIL-free with explicit CUDA graph replay.
//
// The Python-side NativeInferenceBundle provides all CUDA graphs and their
// static I/O buffers (captured during warmup by native_inference_graphs.py).
// This C++ function opens SHM, reads sensor data, fills graph input buffers,
// replays graphs (with GIL held briefly), writes actions, and stores to the
// replay buffer -- all in a tight loop with the GIL released for SHM I/O,
// D3D12 vision, and replay buffer writes.
// =========================================================================

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cstring>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <string>
#include <map>

// --- CUDA driver API types (matching cuda_driver_api.h) ----------------
// Only define if not already provided by cuda.h (pulled in by ATen headers)
#ifndef __cuda_cuda_h__
typedef int CUresult;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;
typedef struct CUextSemaphore_st* CUexternalSemaphore;
typedef struct CUarray_st* CUarray;
#define CU_SUCCESS 0
#define CU_MEMORYTYPE_DEVICE 2
#define CU_MEMORYTYPE_ARRAY  3
#endif

struct NI_CUDA_MEMCPY2D {
    size_t srcXInBytes;  size_t srcY;
    unsigned int srcMemoryType;
    const void* srcHost;  CUdeviceptr srcDevice;  CUarray srcArray;  size_t srcPitch;
    size_t dstXInBytes;  size_t dstY;
    unsigned int dstMemoryType;
    void* dstHost;  CUdeviceptr dstDevice;  CUarray dstArray;  size_t dstPitch;
    size_t WidthInBytes;  size_t Height;
};

struct NI_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
    struct { struct { unsigned long long value; } fence;
             union  { void* fence; unsigned long long reserved; } nvSciSync;
             struct { unsigned long long key; unsigned int timeoutMs; } keyedMutex;
             unsigned int reserved[12];
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};

// Function pointer typedefs -- only what we need
typedef CUresult (*PFN_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*PFN_cuWaitExternalSemaphoresAsync)(
    const CUexternalSemaphore*, const NI_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS*,
    unsigned int, CUstream);
typedef CUresult (*PFN_cuMemcpy2DAsync_v2)(const NI_CUDA_MEMCPY2D*, CUstream);

// --- [NI-DBG] Debug instrumentation macros (no-ops unless NI_DBG enabled) --
#ifdef NI_DBG
#define NI_DBG_SHOULD_LOG(fc) ((fc) <= 20 || ((fc) % 100 == 0))
#define NI_DBG_LOG(fc, fmt, ...) \
    do { if (NI_DBG_SHOULD_LOG(fc)) { \
        fprintf(stderr, "[NI-DBG] f%lld: " fmt "\\n", (long long)(fc), ##__VA_ARGS__); \
        fflush(stderr); \
    } } while(0)
#define NI_DBG_BEFORE(fc, op) NI_DBG_LOG(fc, ">> %s", op)
#define NI_DBG_AFTER(fc, op)  NI_DBG_LOG(fc, "<< %s OK", op)
#define NI_DBG_CHECK_CUDA(fc, op) \
    do { if (NI_DBG_SHOULD_LOG(fc)) { \
        cudaError_t _e = cudaGetLastError(); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "[NI-DBG] f%lld: CUDA ERROR after %s: %s\\n", \
                    (long long)(fc), op, cudaGetErrorString(_e)); \
            fflush(stderr); \
        } \
    } } while(0)
#define NI_DBG_STREAM(fc, label) \
    NI_DBG_LOG(fc, "%s stream=%p", label, \
               (void*)at::cuda::getCurrentCUDAStream().stream())
#else
#define NI_DBG_SHOULD_LOG(fc) (false)
#define NI_DBG_LOG(fc, fmt, ...) ((void)0)
#define NI_DBG_BEFORE(fc, op) ((void)0)
#define NI_DBG_AFTER(fc, op)  ((void)0)
#define NI_DBG_CHECK_CUDA(fc, op) ((void)0)
#define NI_DBG_STREAM(fc, label) ((void)0)
#endif

// --- SHM layout constants (mirror tokenizer/config.py) -----------------
constexpr int SHM_HEADER_SIZE   = 256;
constexpr int NUM_PARITY_SLOTS  = 2;
constexpr int FRAME_BUFFER_SIZE_ALIGNED = 31232;

// Frame header offsets (within slot)
constexpr int HDR_FRAME_NUMBER   = 0;   // int64
constexpr int HDR_TIMESTAMP      = 8;   // double
constexpr int HDR_MODALITY_MASK  = 16;  // uint32
constexpr int HDR_DELTA_TIME     = 20;  // float
constexpr int HDR_GAZE_AZIMUTH   = 32;  // float (degrees)
constexpr int HDR_GAZE_ELEVATION = 36;  // float (degrees)

// Slice offsets within frame
constexpr int AUDIO_OFFSET       = 64;
constexpr int PROPRIO_OFFSET     = 8256;
constexpr int MOTOR_OFFSET       = 12352;
constexpr int TOUCH_OFFSET       = 14400;
constexpr int VOICE_OFFSET       = 30784;

// Audio field offsets
constexpr int AUD_NUM_SAMPLES    = 0;
constexpr int AUD_ILD_DB         = 20;
constexpr int AUD_ITD_MS         = 24;
constexpr int AUD_SPECTRAL       = 40;
constexpr int AUDIO_SPECTRAL_BINS = 256;

// Proprio field offsets
constexpr int PRO_NUM_BONES      = 0;
constexpr int PRO_BONE_DATA      = 8;
constexpr int PROPRIO_BONE_STRIDE = 160;

// Motor field offsets
constexpr int MOTOR_MAX_HANDLES   = 24;
constexpr int MOTOR_HANDLE_STRIDE = 64;
constexpr int MOT_ENERGY          = MOTOR_MAX_HANDLES * MOTOR_HANDLE_STRIDE; // 1536
constexpr int NUM_BODY_SETS       = 10;
constexpr int BODY_SET_STATE_STRIDE = 8;
constexpr int MOT_BODY_SETS       = MOT_ENERGY + 32; // 1568

// Touch field offsets
constexpr int TCH_NUM_IMPULSES   = 0;
constexpr int TCH_IMPULSE_DATA   = 16;
constexpr int TOUCH_MAX_IMPULSES = 256;
constexpr int TOUCH_IMPULSE_STRIDE = 40;

// Voice field offsets
constexpr int VOICE_SIZE         = 256;

// Modality bitmasks
constexpr uint32_t MOD_VISION         = 1 << 0;
constexpr uint32_t MOD_AUDIO          = 1 << 1;
constexpr uint32_t MOD_PROPRIOCEPTION = 1 << 2;
constexpr uint32_t MOD_MOTOR          = 1 << 3;
constexpr uint32_t MOD_TOUCH          = 1 << 4;
constexpr uint32_t MOD_VOICE          = 1 << 5;

// SHM header offsets
constexpr int SHM_OFF_LATEST_FRAME   = 8;
constexpr int SHM_OFF_NUM_SLOTS      = 16;
constexpr int SHM_OFF_SLOT_SIZE      = 20;
constexpr int SHM_OFF_CPU_DATA_OFFSET = 24;

// Action SHM layout
constexpr int ACTION_SHM_HEADER_SIZE = 64;
constexpr int ACT_OFF_FRAME_ID       = 0;
constexpr int ACT_OFF_TIMESTAMP_US   = 8;
constexpr int ACT_OFF_FLAGS          = 16;
constexpr int ACT_OFF_MOTOR          = 20;
constexpr int ACT_OFF_GAZE           = 848;
constexpr int ACT_OFF_VOICE          = 856;
constexpr int ACT_FLAG_MOTOR         = 0x1;
constexpr int ACT_FLAG_GAZE          = 0x2;
constexpr int ACT_FLAG_VOICE         = 0x4;

// BEGIN action_frame_rows (lem/tests/test_native_action_frame_rows.py compiles this
// block on its own and runs frames through it)
// The UE action frame (FActionFrame, ModelBridgeComponent.h NUM_HANDLES 23) holds 23
// limb rows of 9 floats, then its own gaze field (2 floats) and voice field (80
// floats). The model's body has more rows than the frame: 24 since the eyes became
// handle 23 (2026-08-24), 26 in the MIMo trunk_v1 layout. So every per-row loop over
// the frame stops at the FRAME's row count, never at the model's n_handles. A loop
// bounded by n_handles = 24 (2026-08-24 to 2026-09-23) read the gaze field and the
// first 7 voice numbers as a 24th limb row: it multiplied the gaze angles and voice[0]
// by rom_limit x position_range_cm (up to 200) and sent them to UE.
constexpr int ACTION_FRAME_LIMB_ROWS  = 23;
constexpr int ACTION_FRAME_ROW_FLOATS = 9;

inline int action_frame_limb_rows(int64_t n_handles) {
    return (int)std::max<int64_t>(0, std::min<int64_t>(n_handles, ACTION_FRAME_LIMB_ROWS));
}

// Developmental limits on the frame's limb rows: position clamped to [-1, 1] then
// times rom_limit x position_range_cm (UE centimetres) and strength clamped to
// [0, strength_limit], the same rule as Python apply_developmental_constraints();
// radius clamped to [0, 1], which the Python rule does not do; the Python rule's
// curriculum handle_mask is not applied here. The gaze and voice fields after the limb
// rows are never touched. Returns the largest strength left on a limb row.
inline float apply_action_frame_limits(float* frame, int64_t n_handles,
                                       float rom_limit, float strength_limit,
                                       float position_range_cm) {
    const int rows = action_frame_limb_rows(n_handles);
    float max_str = 0.0f;
    for (int h = 0; h < rows; h++) {
        float* base = frame + h * ACTION_FRAME_ROW_FLOATS;
        for (int i = 0; i < 3; i++) {
            float p = std::max(-1.0f, std::min(1.0f, base[i]));
            base[i] = p * rom_limit * position_range_cm;
        }
        base[7] = std::max(0.0f, std::min(1.0f, base[7]));
        base[8] = std::max(0.0f, std::min(strength_limit, base[8]));
        if (base[8] > max_str) max_str = base[8];
    }
    return max_str;
}
// END action_frame_rows
static_assert(ACT_OFF_GAZE - ACT_OFF_MOTOR ==
              ACTION_FRAME_LIMB_ROWS * ACTION_FRAME_ROW_FLOATS * (int)sizeof(float),
              "the gaze field sits right after the frame's 23 limb rows");

// Vision constants
constexpr int PANO_WIDTH  = 2520;
constexpr int PANO_HEIGHT = 1560;
constexpr int FOVEA_WIDTH = 512;
constexpr int FOVEA_HEIGHT = 512;
constexpr int PATCH_SIZE  = 16;

// --- Texture handle info (from cuda_d3d12_interop get_texture_handles) --
struct TexHandle {
    CUarray    cuArray;
    CUdeviceptr linearPtr;
    int64_t    linearSize;
    int        width;
    int        height;
};

// EXP-21: Raw cudaMemcpyAsync -- bypasses ATen dispatch + CUDA allocator mutex.
// .copy_() goes through PyTorch's caching allocator which has a global mutex.
// During training backward, the mutex is heavily contended (hundreds of
// alloc/free per step), causing inference .copy_() calls to block 20-40ms.
// Raw cudaMemcpyAsync has zero allocator interaction.
static inline void raw_copy_async(
    void* dst, const void* src, size_t bytes,
    cudaMemcpyKind kind, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, kind, stream);
}

// Convenience: copy entire contiguous tensor (src->dst, same size)
static inline void raw_tensor_copy_async(
    torch::Tensor& dst, const torch::Tensor& src,
    cudaMemcpyKind kind, cudaStream_t stream) {
    raw_copy_async(dst.data_ptr(), src.data_ptr(),
                   src.nbytes(), kind, stream);
}

// Patchification kernel -- loaded at runtime via CUBIN compiled by NVRTC.
// This avoids nvcc entirely (CUDA 13.1 + MSVC 19.50 incompatibility).
static CUmodule  _patchify_module = nullptr;
static CUfunction _patchify_kernel = nullptr;

void init_patchify_kernel(py::bytes cubin_data) {
    std::string data = cubin_data;  // pybind11 bytes -> std::string (binary-safe)
    CUresult rc = cuModuleLoadData(&_patchify_module, data.data());
    if (rc != CUDA_SUCCESS) {
        throw std::runtime_error(
            "cuModuleLoadData failed for patchify kernel: " + std::to_string(rc));
    }
    rc = cuModuleGetFunction(
        &_patchify_kernel, _patchify_module, "ni_bgra_to_rgb_patches_kernel");
    if (rc != CUDA_SUCCESS) {
        throw std::runtime_error(
            "cuModuleGetFunction failed: " + std::to_string(rc));
    }
}

static void launch_bgra_to_rgb_patches(
    const unsigned char* input, float* output,
    int input_w, int patch_cols, int n_patches,
    cudaStream_t stream) {
    if (!_patchify_kernel) return;
    int total_threads = n_patches * 256;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    void* args[] = {
        (void*)&input, (void*)&output,
        (void*)&input_w, (void*)&patch_cols, (void*)&n_patches
    };
    cuLaunchKernel(_patchify_kernel,
                   grid_size, 1, 1,
                   block_size, 1, 1,
                   0, (CUstream)stream, args, nullptr);
}

// =========================================================================
// Phase 2: Fused elementwise kernels (NVRTC-compiled, sm_120 native)
// =========================================================================

static CUmodule  _fused_gelu_bias_module = nullptr;
static CUfunction _fused_gelu_bias_kernel = nullptr;
static CUmodule  _fused_ln_res_module = nullptr;
static CUfunction _fused_ln_res_kernel = nullptr;
static CUmodule  _fused_res_add_module = nullptr;
static CUfunction _fused_res_add_kernel = nullptr;

void init_fused_kernel(py::bytes cubin_data, const std::string& kernel_name) {
    std::string data = cubin_data;
    CUmodule mod = nullptr;
    CUresult rc = cuModuleLoadData(&mod, data.data());
    if (rc != CUDA_SUCCESS) {
        throw std::runtime_error(
            "cuModuleLoadData failed for " + kernel_name + ": " + std::to_string(rc));
    }
    CUfunction func = nullptr;
    rc = cuModuleGetFunction(&func, mod, kernel_name.c_str());
    if (rc != CUDA_SUCCESS) {
        throw std::runtime_error(
            "cuModuleGetFunction failed for " + kernel_name + ": " + std::to_string(rc));
    }
    if (kernel_name == "fused_gelu_bias") {
        _fused_gelu_bias_module = mod;
        _fused_gelu_bias_kernel = func;
    } else if (kernel_name == "fused_layernorm_residual") {
        _fused_ln_res_module = mod;
        _fused_ln_res_kernel = func;
    } else if (kernel_name == "fused_residual_add") {
        _fused_res_add_module = mod;
        _fused_res_add_kernel = func;
    }
}

// Fused GELU + bias: out = GELU(x + bias)
// x: [rows*D] bf16, bias: [D] f32, out: [rows*D] bf16
torch::Tensor launch_fused_gelu_bias(torch::Tensor x, torch::Tensor bias) {
    if (!_fused_gelu_bias_kernel) {
        throw std::runtime_error("fused_gelu_bias kernel not loaded");
    }
    auto x_flat = x.reshape({-1, x.size(-1)}).contiguous();
    int rows = x_flat.size(0);
    int D = x_flat.size(1);
    auto out = torch::empty_like(x_flat);

    const void* x_ptr = x_flat.data_ptr();
    const void* bias_ptr = bias.data_ptr();
    void* out_ptr = out.data_ptr();

    int total = rows * D;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    void* args[] = {
        (void*)&x_ptr, (void*)&bias_ptr, (void*)&out_ptr,
        (void*)&rows, (void*)&D
    };
    cuLaunchKernel(_fused_gelu_bias_kernel,
                   grid_size, 1, 1,
                   block_size, 1, 1,
                   0, (CUstream)stream, args, nullptr);

    auto orig_sizes = x.sizes().vec();
    return out.view(orig_sizes);
}

// Fused LayerNorm + Residual: out = LayerNorm(x) + residual
// x: [rows*D] bf16, residual: [rows*D] bf16, gamma: [D] f32, beta: [D] f32
torch::Tensor launch_fused_layernorm_residual(
    torch::Tensor x, torch::Tensor residual,
    torch::Tensor gamma, torch::Tensor beta, float eps) {
    if (!_fused_ln_res_kernel) {
        throw std::runtime_error("fused_layernorm_residual kernel not loaded");
    }
    auto x_c = x.contiguous();
    auto res_c = residual.contiguous();
    int D = x_c.size(-1);
    int rows = x_c.numel() / D;
    auto out = torch::empty_like(x_c);

    const void* x_ptr = x_c.data_ptr();
    const void* res_ptr = res_c.data_ptr();
    const void* gamma_ptr = gamma.data_ptr();
    const void* beta_ptr = beta.data_ptr();
    void* out_ptr = out.data_ptr();

    int block_size = 256;
    if (D < 256) block_size = D;
    int shared_mem = block_size * 2 * sizeof(float);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    void* args[] = {
        (void*)&x_ptr, (void*)&res_ptr,
        (void*)&gamma_ptr, (void*)&beta_ptr,
        (void*)&out_ptr,
        (void*)&rows, (void*)&D, (void*)&eps
    };
    cuLaunchKernel(_fused_ln_res_kernel,
                   rows, 1, 1,
                   block_size, 1, 1,
                   shared_mem, (CUstream)stream, args, nullptr);
    return out;
}

// --- Helper: open Windows named shared memory --------------------------
static void* open_shm(const std::string& name, size_t expected_size,
                       HANDLE& out_handle) {
    // Convert to wide string
    std::wstring wname(name.begin(), name.end());
    out_handle = OpenFileMappingW(FILE_MAP_ALL_ACCESS, FALSE, wname.c_str());
    if (!out_handle) {
        // Try creating (for action SHM that Python normally creates)
        out_handle = CreateFileMappingW(
            INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
            0, (DWORD)expected_size, wname.c_str());
        if (!out_handle) return nullptr;
    }
    void* ptr = MapViewOfFile(out_handle, FILE_MAP_ALL_ACCESS, 0, 0, 0);
    return ptr;
}

// --- Helper: parse texture handles from Python dict ---------------------
static std::map<std::string, TexHandle> parse_tex_handles(py::dict tex_dict) {
    std::map<std::string, TexHandle> result;
    for (auto& [key, val] : tex_dict) {
        std::string id = py::str(key);
        py::tuple t = val.cast<py::tuple>();
        TexHandle th;
        th.cuArray    = reinterpret_cast<CUarray>(static_cast<uintptr_t>(t[0].cast<int64_t>()));
        th.linearPtr  = static_cast<CUdeviceptr>(static_cast<uint64_t>(t[1].cast<int64_t>()));
        th.linearSize = t[2].cast<int64_t>();
        th.width      = t[3].cast<int>();
        th.height     = t[4].cast<int>();
        result[id] = th;
    }
    return result;
}

// --- Copy CUarray -> linear device memory via CUDA driver API -----------
static CUresult copy_texture_to_linear(
    PFN_cuMemcpy2DAsync_v2 cuMemcpy2DAsync,
    CUarray srcArray, CUdeviceptr dstLinear,
    int width, int height, CUstream stream)
{
    NI_CUDA_MEMCPY2D desc = {};
    desc.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    desc.srcArray      = srcArray;
    desc.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    desc.dstDevice     = dstLinear;
    desc.dstPitch      = (size_t)width * 4;  // BGRA8 = 4 bytes/pixel
    desc.WidthInBytes  = (size_t)width * 4;
    desc.Height        = (size_t)height;
    return cuMemcpy2DAsync(&desc, stream);
}


// =========================================================================
// run_native_inference_loop
// =========================================================================

// Extract at::cuda::CUDAGraph* from a Python torch.cuda.CUDAGraph object.
// torch._C._CUDAGraph is a pybind11 type (MRO: CUDAGraph -> _CUDAGraph -> pybind11_object).
// pybind11 stores C++ objects via pointer indirection in its instance struct,
// NOT inline -- so a reinterpret_cast to a fixed struct layout is WRONG.
// py::cast<T*>() uses pybind11's process-wide type registry to extract the
// pointer correctly, regardless of instance struct layout details.
at::cuda::CUDAGraph* extract_cuda_graph(py::handle obj) {
    try {
        auto* ptr = py::cast<at::cuda::CUDAGraph*>(obj);
        TORCH_CHECK(ptr != nullptr, "py::cast returned nullptr for CUDAGraph");
        return ptr;
    } catch (const py::cast_error& e) {
        // If pybind11 cast fails, the type may not be registered cross-module.
        // Fall back to probing the pybind11 instance struct directly.
        // For simple-layout pybind11 types, the C++ value pointer is at
        // offset sizeof(PyObject) (= 16 bytes on 64-bit) in the instance.
        fprintf(stderr, "[NI] WARNING: py::cast<CUDAGraph*> failed, using struct probe fallback\\n");
        fflush(stderr);
        void** value_slot = reinterpret_cast<void**>(
            reinterpret_cast<char*>(obj.ptr()) + sizeof(PyObject));
        auto* ptr = static_cast<at::cuda::CUDAGraph*>(*value_slot);
        return ptr;
    }
}

py::dict run_native_inference_loop(
    // -- CUDA graph Python objects (extract C++ CUDAGraph* during setup) --
    py::object vision_graph_obj,         // CUDAGraph for vision encoder
    py::object fwd_graph_obj,            // CUDAGraph for forward model
    py::object motor_gaze_graph_obj,     // CUDAGraph for motor+gaze
    // Encoder graphs: list of (name, CUDAGraph_obj, {key:tensor}, output)
    py::list encoder_graph_list,
    // -- Static I/O buffers (from NativeInferenceBundle) --
    // Vision
    torch::Tensor vision_patches_buf,    // [1, max_patches, 768] static input
    torch::Tensor vision_angular_buf,    // [1, max_patches, 2] static input
    torch::Tensor vision_token_buf,      // [1, 1, d_model] static output
    torch::Tensor vision_backbone_buf,   // [1, max_patches, d_internal] static output
    // Forward model
    torch::Tensor fwd_flat_tokens_buf,   // [1, n_mod, d_model]
    torch::Tensor fwd_context_buf,       // [1, (ctx-1)*n_mod, d_model]
    torch::Tensor fwd_latents_buf,       // [1, seq_len, d_model]
    torch::Tensor fwd_last_step_buf,     // [1, n_mod, d_model]
    torch::Tensor fwd_ctx_update_buf,    // [1, n_mod, d_model]
    // Motor + gaze
    torch::Tensor mg_latents_buf,        // [1, seq_len, d_model]
    torch::Tensor mg_handle_state_buf,   // [1, n_handles, fph]
    torch::Tensor mg_last_step_buf,      // [1, n_mod, d_model]
    torch::Tensor motor_out_buf,         // [1, n_handles, 9]
    torch::Tensor motor_confidence_buf,  // [1, 1]
    torch::Tensor gaze_out_buf,          // [1, 2]
    torch::Tensor voice_out_buf,         // [1, 80]
    // Context buffer
    torch::Tensor context_buffer,        // [1, (ctx-1)*n_mod, d_model]

    // -- D3D12 vision handles --
    py::dict tex_handles_dict,           // {id: (cuArray, linearPtr, size, w, h)}
    int64_t fence_semaphore_ptr,         // CUexternalSemaphore as int64
    int64_t cuda_context_ptr,            // CUcontext as int64

    // -- Vision precomputes --
    torch::Tensor pano_angular_lut,      // [n_pano_patches, 2] static angular positions
    torch::Tensor fovea_local_az,        // [n_fovea_patches] local azimuth grid
    torch::Tensor fovea_local_el,        // [n_fovea_patches] local elevation grid
    // Double-buffer tensors
    torch::Tensor vis_dblbuf_0,          // [1, 33000, 768] patches buffer 0
    torch::Tensor vis_dblbuf_1,          // [1, 33000, 768] patches buffer 1
    torch::Tensor vis_ang_dblbuf_0,      // [1, 33000, 2] angular buffer 0
    torch::Tensor vis_ang_dblbuf_1,      // [1, 33000, 2] angular buffer 1

    // -- SHM names --
    std::string cpu_shm_name,            // "LEM_Pipeline"
    std::string action_shm_name,         // action buffer SHM name

    // -- Replay buffer tensors (shared with training thread) --
    torch::Tensor rb_vision_buf,         // [capacity, max_patches, d_internal]
    torch::Tensor rb_vision_n_patches,   // [capacity] int64 (CPU)
    torch::Tensor rb_audio_spectral_buf, // [capacity, 256, 2]
    torch::Tensor rb_audio_spatial_buf,  // [capacity, 2]
    torch::Tensor rb_proprio_buf,        // [capacity, proprio_dim]
    torch::Tensor rb_motor_buf,          // [capacity, n_handles, fph]
    torch::Tensor rb_tactile_buf,        // [capacity, 256, 8]
    torch::Tensor rb_voice_buf,          // [capacity, voice_dim]
    torch::Tensor rb_motor_capability_buf, // [capacity, cap_dim]
    torch::Tensor rb_gaze_buf,           // [capacity, 2]
    torch::Tensor rb_modality_present,   // [capacity] int32 (CPU)
    torch::Tensor rb_frame_numbers,      // [capacity] int64 (CPU)
    torch::Tensor rb_valid,              // [capacity] bool (CPU)
    torch::Tensor rb_sequence_ids,       // [capacity] int64 (CPU)
    // Token cache buffers: {mod_name: [capacity, 1, d_model]}
    py::dict rb_token_bufs_dict,
    torch::Tensor rb_token_valid,        // [capacity] bool (CPU)

    // Ring state (shared int64 tensors for atomic-ish access)
    torch::Tensor rb_write_head,         // [1] int64 (CPU)
    torch::Tensor rb_size,               // [1] int64 (CPU)
    torch::Tensor rb_total_written,      // [1] int64 (CPU)
    int64_t rb_capacity,

    // -- Config scalars --
    py::list active_modalities,          // ["vision", "audio", ...]
    int64_t n_mod,
    int64_t d_model,
    int64_t ctx_len,
    int64_t n_handles,
    int64_t features_per_handle,
    int64_t max_patches,
    int64_t warmup_frames,

    // -- Control --
    torch::Tensor shutdown_flag,         // [1] int64 CPU tensor -- set to 1 to stop
    bool has_vision,                     // whether D3D12 vision is available
    int64_t phase_rebuild_flag_ptr,      // pointer to int64 for phase rebuild check (0=unused)
    // -- Inference stream (must be the SAME stream graphs were captured on) --
    int64_t inference_stream_id,         // stream_id from torch.cuda.Stream
    int64_t inference_stream_device_index, // device_index
    int64_t inference_stream_device_type,  // device_type (1 = CUDA)
    // -- Weight sync callback (called every N frames with GIL held) --
    py::object weight_sync_callback,     // Python callable() -> None
    // -- Stats callback (called every stats_interval frames with GIL held) --
    py::object stats_callback,           // Python callable(dict) -> None
    int64_t stats_interval,              // frames between stats callbacks (0=disabled)
    // -- Timestamp reference (perf_counter us at Python call time) --
    int64_t perf_counter_ref_us,         // time.perf_counter_ns()//1000 at launch
    // -- Vision stream (for cross-device: runs D3D12 ops + texture staging on cuda:1) --
    int64_t vision_stream_id,
    int64_t vision_stream_device_index,
    int64_t vision_stream_device_type,   // 0 = not provided, 1 = CUDA
    // -- Developmental constraints (shared with Python training thread) --
    // [3] float32 CPU tensor: [rom_limit, strength_limit, position_range_cm]
    // Training thread updates these atomically on phase transitions;
    // writer thread reads them each frame (relaxed: float load is atomic on x86).
    torch::Tensor dev_limits_buf,
    // -- Combined graph (EXP-20): replaces encoder+forward+motor/gaze graphs --
    py::object combined_graph_obj = py::none()  // None = legacy separate graphs
) {
    // --- Use the CAPTURE stream for graph replay ----------------------------
    // CUDA graphs bind their memory pool to the capture stream. Replaying on
    // a different stream causes cudaErrorIllegalAddress because pool-allocated
    // device pointers are not visible to the new stream's allocator.
    auto inference_stream = c10::cuda::CUDAStream::unpack3(
        inference_stream_id,
        static_cast<c10::DeviceIndex>(inference_stream_device_index),
        static_cast<c10::DeviceType>(inference_stream_device_type));
    c10::cuda::CUDAStreamGuard stream_guard(inference_stream);

    // Unpack vision stream (for cross-device vision pipeline on cuda:1)
    bool has_vision_stream = (vision_stream_device_type > 0);
    c10::cuda::CUDAStream vis_stream = inference_stream;  // fallback
    if (has_vision_stream) {
        vis_stream = c10::cuda::CUDAStream::unpack3(
            vision_stream_id,
            static_cast<c10::DeviceIndex>(vision_stream_device_index),
            static_cast<c10::DeviceType>(vision_stream_device_type));
    }
    bool vision_on_capture = has_vision_stream &&
        (vis_stream.device_index() != inference_stream.device_index());

    // RB store stream: lower priority than inference, allows RB copies to
    // overlap with next frame's SHM poll + D3D12 copy + graph replay.
    auto rb_store_stream = c10::cuda::getStreamFromPool(
        /*isHighPriority=*/false, inference_stream.device_index());
    cudaEvent_t rb_dep_event = nullptr;  // signals when graph outputs are ready
    cudaEventCreateWithFlags(&rb_dep_event, cudaEventDisableTiming);

    // --- Extract C++ CUDAGraph pointers from Python objects ---------------
    // py::cast<> uses pybind11's process-wide type registry to resolve the
    // pointer from the pybind11 instance struct. Works cross-module because
    // torch._C registered at::cuda::CUDAGraph globally.
    // Done with GIL held; stored pointers used GIL-free in the loop.
    auto* vis_graph = extract_cuda_graph(vision_graph_obj);
    auto* fwd_graph = extract_cuda_graph(fwd_graph_obj);
    auto* mg_graph  = extract_cuda_graph(motor_gaze_graph_obj);

    // EXP-20: combined graph (encoders+forward+motor/gaze in one replay)
    bool use_combined = !combined_graph_obj.is_none();
    at::cuda::CUDAGraph* combined_graph = nullptr;
    if (use_combined) {
        combined_graph = extract_cuda_graph(combined_graph_obj);
        fprintf(stderr, "[NI] Combined graph mode active (EXP-20)\\n");
        fflush(stderr);
    }

    // Validate: replay each graph once with GIL held before entering the loop.
    {
        c10::cuda::CUDAStreamGuard tmp_guard(inference_stream);
        cudaError_t err;

        fwd_graph->replay();
        err = cudaStreamSynchronize(inference_stream.stream());
        if (err != cudaSuccess) {
            py::dict result;
            result["error"] = std::string("fwd graph validation failed: ") + cudaGetErrorString(err);
            return result;
        }

        {
            // Vision graph always on inference device (cuda:0)
            vis_graph->replay();
            err = cudaStreamSynchronize(inference_stream.stream());
            if (err != cudaSuccess) {
                py::dict result;
                result["error"] = std::string("vis graph validation failed: ") + cudaGetErrorString(err);
                return result;
            }
        }

        mg_graph->replay();
        err = cudaStreamSynchronize(inference_stream.stream());
        if (err != cudaSuccess) {
            py::dict result;
            result["error"] = std::string("mg graph validation failed: ") + cudaGetErrorString(err);
            return result;
        }
    }

    // --- Shutdown flag (GIL-free) ----------------------------------------
    volatile int64_t* shutdown_ptr = shutdown_flag.data_ptr<int64_t>();

    // --- Parse modality list --------------------------------------------
    std::vector<std::string> active_mods;
    std::map<std::string, int> mod_index;  // modality name -> index in active list
    for (int i = 0; i < py::len(active_modalities); i++) {
        std::string name = active_modalities[i].cast<std::string>();
        mod_index[name] = i;
        active_mods.push_back(name);
    }

    // --- Parse encoder graph list ---------------------------------------
    struct EncoderGraphInfo {
        at::cuda::CUDAGraph* graph;  // C++ graph ptr (via pybind11 cast)
        std::map<std::string, torch::Tensor> inputs;
        torch::Tensor output;
        std::string name;
    };
    std::vector<EncoderGraphInfo> encoder_graphs;
    for (auto item : encoder_graph_list) {
        py::tuple t = item.cast<py::tuple>();
        EncoderGraphInfo info;
        info.name = t[0].cast<std::string>();
        py::object graph_obj = t[1];
        info.graph = extract_cuda_graph(graph_obj);
        py::dict input_dict = t[2].cast<py::dict>();
        for (auto& [k, v] : input_dict) {
            info.inputs[py::str(k).cast<std::string>()] = v.cast<torch::Tensor>();
        }
        info.output = t[3].cast<torch::Tensor>();
        encoder_graphs.push_back(std::move(info));
    }
    // Validate encoder graphs on inference stream
    {
        c10::cuda::CUDAStreamGuard tmp_guard(inference_stream);
        for (auto& eg : encoder_graphs) {
            eg.graph->replay();
            cudaError_t err = cudaStreamSynchronize(inference_stream.stream());
            if (err != cudaSuccess) {
                py::dict result;
                result["error"] = std::string("encoder graph '") + eg.name +
                    "' validation failed: " + cudaGetErrorString(err);
                return result;
            }
        }
    }

    // --- Parse token cache buffers --------------------------------------
    std::map<std::string, torch::Tensor> token_bufs;
    for (auto& [k, v] : rb_token_bufs_dict) {
        token_bufs[py::str(k).cast<std::string>()] = v.cast<torch::Tensor>();
    }

    // --- Load nvcuda.dll for D3D12 interop ------------------------------
    HMODULE nvcuda = LoadLibraryW(L"nvcuda.dll");
    PFN_cuCtxSetCurrent fn_cuCtxSetCurrent = nullptr;
    PFN_cuWaitExternalSemaphoresAsync fn_cuWaitExtSem = nullptr;
    PFN_cuMemcpy2DAsync_v2 fn_cuMemcpy2DAsync = nullptr;

    if (has_vision && nvcuda) {
        fn_cuCtxSetCurrent = reinterpret_cast<PFN_cuCtxSetCurrent>(
            GetProcAddress(nvcuda, "cuCtxSetCurrent"));
        fn_cuWaitExtSem = reinterpret_cast<PFN_cuWaitExternalSemaphoresAsync>(
            GetProcAddress(nvcuda, "cuWaitExternalSemaphoresAsync"));
        fn_cuMemcpy2DAsync = reinterpret_cast<PFN_cuMemcpy2DAsync_v2>(
            GetProcAddress(nvcuda, "cuMemcpy2DAsync_v2"));
    }

    CUexternalSemaphore fence_sem = reinterpret_cast<CUexternalSemaphore>(
        static_cast<uintptr_t>(fence_semaphore_ptr));
    CUcontext cuda_ctx = reinterpret_cast<CUcontext>(
        static_cast<uintptr_t>(cuda_context_ptr));

    // Parse texture handles
    auto tex_handles = parse_tex_handles(tex_handles_dict);

    // --- Open CPU SHM ---------------------------------------------------
    HANDLE shm_handle = nullptr;
    volatile unsigned char* shm_ptr = nullptr;
    {
        py::gil_scoped_release release;
        void* p = open_shm(cpu_shm_name, SHM_HEADER_SIZE + NUM_PARITY_SLOTS * FRAME_BUFFER_SIZE_ALIGNED, shm_handle);
        shm_ptr = static_cast<volatile unsigned char*>(p);
    }
    if (!shm_ptr) {
        py::dict result;
        result["error"] = "Failed to open CPU SHM";
        return result;
    }

    // Read SHM layout from header
    int32_t num_slots = *reinterpret_cast<volatile int32_t*>(
        const_cast<unsigned char*>(shm_ptr) + SHM_OFF_NUM_SLOTS);
    int32_t slot_size = *reinterpret_cast<volatile int32_t*>(
        const_cast<unsigned char*>(shm_ptr) + SHM_OFF_SLOT_SIZE);
    int32_t data_offset = *reinterpret_cast<volatile int32_t*>(
        const_cast<unsigned char*>(shm_ptr) + SHM_OFF_CPU_DATA_OFFSET);

    // --- Open action SHM ------------------------------------------------
    HANDLE action_handle = nullptr;
    volatile unsigned char* action_ptr = nullptr;
    if (!action_shm_name.empty()) {
        py::gil_scoped_release release;
        void* p = open_shm(action_shm_name,
                            ACTION_SHM_HEADER_SIZE + 1176, action_handle);
        action_ptr = static_cast<volatile unsigned char*>(p);
    }

    // --- Pinned staging buffers for CPU<->GPU transfers -------------------
    // Allocate pinned host memory for SHM->GPU async copies
    constexpr size_t SLOT_STAGING_SIZE = FRAME_BUFFER_SIZE_ALIGNED;
    unsigned char* pinned_slot = nullptr;
    {
        auto t = torch::empty({SLOT_STAGING_SIZE}, torch::dtype(torch::kUInt8).pinned_memory(true));
        pinned_slot = t.data_ptr<unsigned char>();
    }
    // Keep the tensor alive (prevent dealloc)
    auto pinned_slot_keeper = torch::empty({SLOT_STAGING_SIZE}, torch::dtype(torch::kUInt8).pinned_memory(true));
    pinned_slot = pinned_slot_keeper.data_ptr<unsigned char>();

    // Pinned buffer for action D->H (motor [23,9]=828B + gaze [2]=8B)
    // Double-buffered: inference writes to [write_idx], writer thread reads from [1-write_idx]
    auto pinned_action_keeper_0 = torch::empty({1280}, torch::dtype(torch::kUInt8).pinned_memory(true));
    auto pinned_action_keeper_1 = torch::empty({1280}, torch::dtype(torch::kUInt8).pinned_memory(true));
    float* pinned_action_bufs[2] = {
        reinterpret_cast<float*>(pinned_action_keeper_0.data_ptr<unsigned char>()),
        reinterpret_cast<float*>(pinned_action_keeper_1.data_ptr<unsigned char>())
    };
    // Legacy pointer for stats readback (points to last completed buffer)
    float* pinned_action = pinned_action_bufs[0];
    std::atomic<int> pinned_write_idx{0};  // inference writes to this index

    // D2H event per buffer slot for the writer thread to sync on
    cudaEvent_t d2h_events[2] = {nullptr, nullptr};
    cudaEventCreateWithFlags(&d2h_events[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&d2h_events[1], cudaEventDisableTiming);
    // Legacy event alias for backward compat with gpu timing
    cudaEvent_t& d2h_event = d2h_events[0];
    bool d2h_pending = false;
    double prev_gpu_exec_us = 0.0;

    // Confidence staging (double-buffered)
    auto confidence_staging_0 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    auto confidence_staging_1 = torch::zeros({1}, torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    float* confidence_bufs[2] = {
        confidence_staging_0.data_ptr<float>(),
        confidence_staging_1.data_ptr<float>()
    };
    float* cached_confidence_d2h_ptr = confidence_bufs[0];
    float prev_confidence = 0.0f;

    // Forward-declare timing vars used by writer thread (assigned later)
    auto start_time_fwd = std::chrono::steady_clock::now();
    int64_t ts_offset_us_fwd = 0;

    // Forward-declare D2H staging pointer arrays (assigned after tensor creation).
    // Double-buffered: writer thread reads staging[slot] while inference writes
    // staging[1-slot]. Pointers assigned after tensor creation below.
    float* cached_motor_d2h_ptr_wr[2] = {nullptr, nullptr};
    float* cached_gaze_d2h_ptr_wr[2] = {nullptr, nullptr};
    float* cached_voice_d2h_ptr_wr[2] = {nullptr, nullptr};

    // -- Action writer thread --
    // Runs independently of inference loop. Waits for D2H event on each
    // buffer slot, reads the pinned staging, writes to action SHM.
    // Inference loop never blocks on D2H.
    std::atomic<bool> writer_running{true};
    std::atomic<int64_t> writer_frame_counter{0};
    std::atomic<int> writer_pending_slot{-1};  // -1 = nothing pending
    std::atomic<float> writer_confidence{0.0f};

    // Motor snapshot for stats (written by writer thread, read by stats callback)
    float _last_motor[7] = {0};
    std::mutex motor_snapshot_mtx;

    auto writer_fn = [&]() {
        int last_processed_slot = -1;
        while (writer_running.load(std::memory_order_relaxed)) {
            int slot = writer_pending_slot.load(std::memory_order_acquire);
            if (slot < 0 || slot == last_processed_slot) {
                // Nothing new -- yield briefly
                SwitchToThread();
                continue;
            }

            // Wait for D2H to complete on this slot
            cudaEventSynchronize(d2h_events[slot]);
            last_processed_slot = slot;

            // GPU->CPU DMA is complete. Copy from staging tensors to
            // pinned_action_bufs (CPU memcpy, not on CUDA stream).
            if (cached_motor_d2h_ptr_wr[slot]) {
                memcpy(pinned_action_bufs[slot], cached_motor_d2h_ptr_wr[slot],
                       (size_t)std::min(n_handles * 9, (int64_t)207) * sizeof(float));
            }
            if (cached_gaze_d2h_ptr_wr[slot]) {
                memcpy(pinned_action_bufs[slot] + 207, cached_gaze_d2h_ptr_wr[slot],
                       2 * sizeof(float));
            }
            if (cached_voice_d2h_ptr_wr[slot]) {
                memcpy(pinned_action_bufs[slot] + 207 + 2, cached_voice_d2h_ptr_wr[slot],
                       80 * sizeof(float));
            }

            float* src = pinned_action_bufs[slot];
            int64_t fc = writer_frame_counter.fetch_add(1) + 1;

            // Developmental constraints: match Python apply_developmental_constraints()
            // dev_limits_buf = [rom_limit, strength_limit, position_range_cm]
            float* dl = dev_limits_buf.data_ptr<float>();
            float rom_limit = dl[0];
            float strength_limit = dl[1];
            float position_range_cm = dl[2];

            // Limb rows only (see action_frame_rows): the gaze field at +207 and the
            // voice field at +209 go to UE as the model produced them.
            const float max_str = apply_action_frame_limits(
                src, n_handles, rom_limit, strength_limit, position_range_cm);

            // Write to action SHM
            if (action_ptr && fc > warmup_frames) {
                unsigned char* action_base = const_cast<unsigned char*>(action_ptr) +
                    ACTION_SHM_HEADER_SIZE;
                uint64_t ts_us = (uint64_t)(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start_time_fwd).count() + ts_offset_us_fwd);
                memcpy(action_base + ACT_OFF_FRAME_ID, &fc, 8);
                memcpy(action_base + ACT_OFF_TIMESTAMP_US, &ts_us, 8);
                uint32_t flags = ACT_FLAG_MOTOR | ACT_FLAG_GAZE | ACT_FLAG_VOICE;
                memcpy(action_base + ACT_OFF_FLAGS, &flags, 4);
                memcpy(action_base + ACT_OFF_MOTOR, src, 207 * sizeof(float));
                memcpy(action_base + ACT_OFF_GAZE, src + 207, 2 * sizeof(float));
                memcpy(action_base + ACT_OFF_VOICE, src + 207 + 2, 80 * sizeof(float));
            }

            // Update confidence
            writer_confidence.store(confidence_bufs[slot][0], std::memory_order_relaxed);

            // Motor snapshot for stats
            {
                std::lock_guard<std::mutex> lock(motor_snapshot_mtx);
                _last_motor[0] = src[0]; _last_motor[1] = src[1]; _last_motor[2] = src[2];
                _last_motor[3] = src[8]; _last_motor[4] = max_str;
                _last_motor[5] = src[207]; _last_motor[6] = src[208];
            }

            // Don't mark -1 -- just track last_processed_slot.
            // Inference loop alternates 0/1 so next slot will differ.
        }
    };
    std::thread action_writer_thread(writer_fn);

    // --- CPU (pinned) staging tensors for encoder inputs -----------------
    // Graph input buffers live on GPU; we can't write to them from CPU.
    // Instead: fill pinned CPU staging -> .copy_() to GPU.
    struct EncoderStaging {
        std::map<std::string, torch::Tensor> cpu_bufs;
    };
    std::vector<EncoderStaging> encoder_staging(encoder_graphs.size());
    for (size_t i = 0; i < encoder_graphs.size(); i++) {
        for (auto& [key, gpu_tensor] : encoder_graphs[i].inputs) {
            encoder_staging[i].cpu_bufs[key] = torch::zeros(
                gpu_tensor.sizes().vec(),
                torch::dtype(gpu_tensor.dtype()).device(torch::kCPU).pinned_memory(true));
        }
    }

    // CPU staging for motor+gaze graph handle_state input
    auto mg_handle_state_staging = torch::zeros(
        mg_handle_state_buf.sizes().vec(),
        torch::dtype(mg_handle_state_buf.dtype()).device(torch::kCPU).pinned_memory(true));

    // CPU staging for replay buffer motor_capability + gaze
    auto rb_motor_cap_staging = torch::zeros(
        {rb_motor_capability_buf.size(1)},
        torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    auto rb_gaze_staging = torch::zeros(
        {2}, torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));

    // CPU (pinned, float32) staging for action D2H (graph outputs are bf16).
    // Double-buffered to match pinned_action_bufs -- inference loop writes to
    // staging[widx] while writer thread reads from staging[1-widx].
    auto motor_d2h_staging_0 = torch::zeros(
        {n_handles * 9},
        torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    auto motor_d2h_staging_1 = torch::zeros(
        {n_handles * 9},
        torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    torch::Tensor motor_d2h_staging[2] = {motor_d2h_staging_0, motor_d2h_staging_1};
    auto gaze_d2h_staging_0 = torch::zeros(
        {2},
        torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    auto gaze_d2h_staging_1 = torch::zeros(
        {2},
        torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    torch::Tensor gaze_d2h_staging[2] = {gaze_d2h_staging_0, gaze_d2h_staging_1};
    auto voice_d2h_staging_0 = torch::zeros(
        {80},
        torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    auto voice_d2h_staging_1 = torch::zeros(
        {80},
        torch::dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
    torch::Tensor voice_d2h_staging[2] = {voice_d2h_staging_0, voice_d2h_staging_1};

    // --- Vision patchification output buffers (pre-allocated on GPU) ----
    // Determine actual texture dimensions from handles (may differ from constants)
    int actual_pano_w = PANO_WIDTH;
    int actual_fovea_w = FOVEA_WIDTH;
    for (auto& [id, th] : tex_handles) {
        if (id.find("pano") != std::string::npos) actual_pano_w = th.width;
        if (id.find("fovea") != std::string::npos) actual_fovea_w = th.width;
    }
    int infer_device = static_cast<int>(inference_stream.device_index());
    if (vision_on_capture) {
        int vis_device = static_cast<int>(vis_stream.device_index());
        fprintf(stderr, "[NI] Cross-device vision: D3D12 on cuda:%d, patchify+encoder on cuda:%d\\n",
                vis_device, infer_device);
        fflush(stderr);
    }

    // Patch dimensions use crop widths (patch-aligned)
    int pano_crop_w = (actual_pano_w / PATCH_SIZE) * PATCH_SIZE;
    int pano_crop_h = (PANO_HEIGHT / PATCH_SIZE) * PATCH_SIZE;
    int pano_patch_cols = pano_crop_w / PATCH_SIZE;
    int pano_patch_rows = pano_crop_h / PATCH_SIZE;
    int pano_n_patches  = pano_patch_rows * pano_patch_cols;
    int fovea_crop_w = (actual_fovea_w / PATCH_SIZE) * PATCH_SIZE;
    int fovea_crop_h = (FOVEA_HEIGHT / PATCH_SIZE) * PATCH_SIZE;
    int fovea_patch_cols = fovea_crop_w / PATCH_SIZE;
    int fovea_patch_rows = fovea_crop_h / PATCH_SIZE;
    int fovea_n_patches  = fovea_patch_rows * fovea_patch_cols;

    auto patch_opts = torch::dtype(torch::kFloat32).device(vision_patches_buf.device());
    torch::Tensor pano_L_patches  = torch::empty({pano_n_patches, PATCH_SIZE * PATCH_SIZE * 3}, patch_opts);
    torch::Tensor pano_R_patches  = torch::empty({pano_n_patches, PATCH_SIZE * PATCH_SIZE * 3}, patch_opts);
    torch::Tensor fovea_L_patches = torch::empty({fovea_n_patches, PATCH_SIZE * PATCH_SIZE * 3}, patch_opts);
    torch::Tensor fovea_R_patches = torch::empty({fovea_n_patches, PATCH_SIZE * PATCH_SIZE * 3}, patch_opts);

    // Double-buffered cross-device texture staging: two sets of raw BGRA
    // buffers on cuda:0.  Frame N writes to staging[stg_write] via
    // cudaMemcpyPeerAsync on vis_stream while inference_stream reads from
    // staging[1-stg_write] (previous frame's data, already complete).
    // Eliminates the ~15ms GPU stall from cudaStreamWaitEvent.
    struct TextureStaging { unsigned char* ptr; size_t size; };
    std::map<std::string, TextureStaging> staging_bufs[2];
    std::vector<torch::Tensor> staging_keepers[2];
    if (vision_on_capture) {
        for (int db = 0; db < 2; db++) {
            for (auto& [id, th] : tex_handles) {
                size_t sz = (size_t)th.width * th.height * 4;
                auto t = torch::empty({(int64_t)sz},
                    torch::dtype(torch::kUInt8).device(torch::kCUDA, infer_device));
                staging_bufs[db][id] = {t.data_ptr<unsigned char>(), sz};
                staging_keepers[db].push_back(std::move(t));
            }
        }
        fprintf(stderr, "[NI] Allocated 2x%zu texture staging buffers on cuda:%d\\n",
                staging_bufs[0].size(), infer_device);
        fflush(stderr);
    }

    // Double-buffered staging events + state.
    // stg_events[i] is recorded on vis_stream after peer copy into staging[i].
    // Created on cuda:1 (cudaEventRecord requires event+stream same device).
    // cudaStreamWaitEvent works cross-device (cuda:0 waiting on cuda:1 event).
    cudaEvent_t stg_events[2] = {nullptr, nullptr};
    int  stg_write = 0;                    // which staging set receives current frame
    bool stg_has_data[2] = {false, false}; // true once a copy has been recorded
    int  stg_slot_idx[2] = {0, 0};         // SHM slot_idx used when writing each set
    if (vision_on_capture) {
        int vis_dev = static_cast<int>(vis_stream.device_index());
        cudaSetDevice(vis_dev);
        cudaEventCreateWithFlags(&stg_events[0], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&stg_events[1], cudaEventDisableTiming);
        cudaSetDevice(infer_device);  // restore
    }

    // Total patches per frame: 2*pano + 2*fovea = 32506
    int total_patches = 2 * pano_n_patches + 2 * fovea_n_patches;

    // --- Double-buffer state (unused -- kept for parameter compatibility) --
    (void)vis_dblbuf_0; (void)vis_dblbuf_1;
    (void)vis_ang_dblbuf_0; (void)vis_ang_dblbuf_1;

    // --- Fovea angular position computation buffers ---------------------
    // fovea_local_az/el are [n_fovea_patches] -- add gaze offset per frame
    // We need 4 sets: L pano(static), R pano(static), L fovea(gaze-relative), R fovea
    torch::Tensor fovea_ang_buf = torch::empty({fovea_n_patches, 2},
        torch::dtype(torch::kFloat32).device(vision_patches_buf.device()));

    // --- Cached data_ptr for GIL-released memcpy operations -------------
    // Pinned CPU staging buffers are allocated once and never reallocated,
    // so their data_ptr values are stable for the loop's lifetime.
    struct CachedEncPtrs {
        std::string name;
        std::map<std::string, float*> ptrs;
        std::map<std::string, std::vector<int64_t>> shapes;
    };
    std::vector<CachedEncPtrs> cached_enc(encoder_graphs.size());
    for (size_t i = 0; i < encoder_graphs.size(); i++) {
        cached_enc[i].name = encoder_graphs[i].name;
        for (auto& [key, buf] : encoder_staging[i].cpu_bufs) {
            cached_enc[i].ptrs[key] = buf.data_ptr<float>();
            cached_enc[i].shapes[key] = buf.sizes().vec();
        }
    }
    float* cached_mg_hs_ptr = mg_handle_state_staging.data_ptr<float>();
    int cached_mg_hs_nh = (int)mg_handle_state_staging.size(1);
    int cached_mg_hs_fph = (int)mg_handle_state_staging.size(2);
    float* cached_motor_d2h_ptrs[2] = {
        motor_d2h_staging[0].data_ptr<float>(),
        motor_d2h_staging[1].data_ptr<float>()};
    float* cached_gaze_d2h_ptrs[2] = {
        gaze_d2h_staging[0].data_ptr<float>(),
        gaze_d2h_staging[1].data_ptr<float>()};
    float* cached_voice_d2h_ptrs[2] = {
        voice_d2h_staging[0].data_ptr<float>(),
        voice_d2h_staging[1].data_ptr<float>()};
    // Assign writer-thread pointer arrays (forward-declared before writer lambda)
    cached_motor_d2h_ptr_wr[0] = cached_motor_d2h_ptrs[0];
    cached_motor_d2h_ptr_wr[1] = cached_motor_d2h_ptrs[1];
    cached_gaze_d2h_ptr_wr[0] = cached_gaze_d2h_ptrs[0];
    cached_gaze_d2h_ptr_wr[1] = cached_gaze_d2h_ptrs[1];
    cached_voice_d2h_ptr_wr[0] = cached_voice_d2h_ptrs[0];
    cached_voice_d2h_ptr_wr[1] = cached_voice_d2h_ptrs[1];
    float* cached_rb_cap_ptr = rb_motor_cap_staging.data_ptr<float>();
    int cached_rb_cap_dim = (int)rb_motor_cap_staging.size(0);
    float* cached_rb_gaze_ptr = rb_gaze_staging.data_ptr<float>();

    // CPU replay buffer metadata ptrs (CPU tensors, stable addresses)
    int64_t* cached_rb_vis_np = rb_vision_n_patches.data_ptr<int64_t>();
    int32_t* cached_rb_mod_present = rb_modality_present.data_ptr<int32_t>();
    int64_t* cached_rb_frame_nums = rb_frame_numbers.data_ptr<int64_t>();
    bool* cached_rb_valid = rb_valid.data_ptr<bool>();
    int64_t* cached_rb_total_wr = rb_total_written.data_ptr<int64_t>();
    int64_t* cached_rb_seq_ids = rb_sequence_ids.data_ptr<int64_t>();
    bool* cached_rb_token_valid = rb_token_valid.data_ptr<bool>();
    int64_t* cached_rb_wh = rb_write_head.data_ptr<int64_t>();
    int64_t* cached_rb_sz = rb_size.data_ptr<int64_t>();

    // --- Loop state -----------------------------------------------------
    int64_t last_frame = -1;
    int64_t frame_count = 0;
    int64_t action_frame_counter = 0;
    auto start_time = std::chrono::steady_clock::now();
    // Offset so that (steady_clock - start_time + offset) ~= perf_counter_ns//1000
    // This lets UE compute correct latency vs its own FPlatformTime::Seconds() epoch.
    int64_t ts_offset_us = perf_counter_ref_us;  // add to (now - start_time) for absolute us
    // Update forward-declared copies for writer thread
    start_time_fwd = start_time;
    ts_offset_us_fwd = ts_offset_us;
    double fps_ema = 0.0;
    constexpr double EMA_ALPHA = 0.02;

    // Previous frame's action (for pipelined write)
    bool has_pending_action = false;
    int64_t poll_count = 0;
    int64_t poll_miss_accum = 0;    // SHM poll misses since last stats report
    int64_t slow_frame_accum = 0;   // frames > 50ms since last stats report
    int64_t fresh_vision_accum = 0; // frames with fresh vision since last report
    int64_t d2h_skip_accum = 0;     // frames where D2H wasn't ready (reused prev motor)

    // --- Per-frame timing instrumentation --------------------------------
    // Measures wall-clock us per phase.  Accumulators reset every stats_interval.
    using ni_clock = std::chrono::steady_clock;
    ni_clock::time_point _tp[12];  // timing checkpoints within a frame
    struct { double sum; double peak; } _tm[10] = {};  // 10 phases
    // Phase indices (enum-like):
    //  0=phase1  1=h2d  2=d3d12  3=graphs  4=ctx_shift
    //  5=gpu_sync  6=wsync  7=stats_cb  8=d2h  9=rb_store
    int64_t _tm_count = 0;

    // CUDA event pair for device-side GPU execution timing.
    // Measures actual GPU work time between start-of-Phase2a and
    // end-of-context-shift (all enqueued work).  Compare with wall-clock
    // gpu_sync to quantify training SM contention.
    cudaEvent_t gpu_t0 = nullptr, gpu_t1 = nullptr;
    cudaEventCreateWithFlags(&gpu_t0, 0);  // timing-enabled (no DisableTiming)
    cudaEventCreateWithFlags(&gpu_t1, 0);
    double gpu_exec_sum = 0.0;
    double gpu_exec_peak = 0.0;

    // Phase1 sub-timer accumulators (memcpy, d2h_sync, action)
    struct { double sum; double peak; } _p1_memcpy = {}, _p1_d2h = {}, _p1_action = {};

    // Motor/gaze snapshot -- defined above with writer thread (line ~1199)

    // --- Main inference loop --------------------------------------------
    fprintf(stderr, "[NI] loop start: shm=%p action=%p has_vision=%d\\n",
            (void*)shm_ptr, (void*)action_ptr, (int)has_vision);
    fflush(stderr);

    // *** GIL MANAGEMENT ***
    // The entire loop runs GIL-free.  All GPU work (graph replays, tensor
    // copies, CUDA events) uses libtorch C++ API which doesn't need the GIL.
    // GIL is acquired ONLY for the two Python callbacks (stats, weight_sync).
    // This lets the training thread hold the GIL freely for build_context
    // and weight staging without blocking inference.

    bool running = true;
    // Release GIL ONCE for the entire loop -- not per-iteration.
    // Per-iteration release causes GIL reacquire at the loop boundary
    // (destructor of loop_release), blocking for 10-80ms when the
    // training thread holds the GIL during build_context.  This was
    // the source of the 17ms unaccounted gap per frame.
    // Callbacks use py::gil_scoped_acquire to temporarily reacquire.
    py::gil_scoped_release loop_release;
    while (running) {

        // ===== SHM Poll + Parse ==========================================
        int64_t current_frame = -1;
        int slot_idx = 0;
        uint32_t modality_mask = 0;
        float gaze_az_deg = 0, gaze_el_deg = 0, dt = 0;
        float gaze_az_rad = 0, gaze_el_rad = 0;
        bool have_new_frame = false;
        ni_clock::time_point _p1a, _p1b, _p1c, _p1d;  // phase1 sub-timers
        {

            // -- 1. Poll CPU SHM for new frame ---------------------------
            // Inner poll loop: stays GIL-free while spinning on SHM.
            // Without this, `continue` on the outer loop would trigger
            // loop_release destructor -> GIL reacquire+release per poll miss
            // (thousands per frame).
            while (true) {
                current_frame = *reinterpret_cast<volatile int64_t*>(
                    const_cast<unsigned char*>(shm_ptr) + SHM_OFF_LATEST_FRAME);

                if (current_frame > last_frame) break;  // new frame available

                // No new frame -- yield CPU then re-poll immediately.
                // sleep_for(100us) is WRONG on Windows: default 15.6ms timer
                // resolution makes it sleep 1-15ms, adding ~7ms avg latency
                // per poll miss and dropping FPS from 35 to 28.
#ifdef _WIN32
                SwitchToThread();
#else
                std::this_thread::yield();
#endif
                poll_count++;
                poll_miss_accum++;

                // Periodic shutdown check (every ~1M polls).
                // No heartbeat fprintf -- status logged via Python stats callback.
                if (poll_count % 1000000 == 0) {
                    if (*shutdown_ptr != 0) {
                        running = false;
                        break;
                    }
                }
            }

            if (!running) break;  // exit outer loop on shutdown

            have_new_frame = true;
            _tp[0] = ni_clock::now();  // TP0: frame processing start
            last_frame = current_frame;
            frame_count++;

            // Compute parity slot
            slot_idx = (int)(current_frame % num_slots);
            const unsigned char* slot_base = const_cast<unsigned char*>(shm_ptr) +
                data_offset + slot_idx * slot_size;

            // -- 2. Parse frame header (pure memcpy + pointer reads) -----
            _p1a = ni_clock::now();  // after slot computation, before memcpy
            memcpy(pinned_slot, slot_base, slot_size);
            modality_mask = *reinterpret_cast<uint32_t*>(pinned_slot + HDR_MODALITY_MASK);
            gaze_az_deg = *reinterpret_cast<float*>(pinned_slot + HDR_GAZE_AZIMUTH);
            gaze_el_deg = *reinterpret_cast<float*>(pinned_slot + HDR_GAZE_ELEVATION);
            dt = *reinterpret_cast<float*>(pinned_slot + HDR_DELTA_TIME);

            gaze_az_rad = gaze_az_deg * (float)(M_PI / 180.0);
            gaze_el_rad = gaze_el_deg * (float)(M_PI / 180.0);

            _p1b = ni_clock::now();  // after header parse, before D2H handoff

            // ---- D2H: hand off to writer thread (non-blocking) ----
            // The writer thread waits for the D2H event and writes to action SHM.
            // Inference loop never blocks. GPU exec timing uses query to check
            // if previous frame's events completed (for stats only, not blocking).
            if (d2h_pending) {
                // Non-blocking GPU exec timing check
                cudaError_t q = cudaEventQuery(gpu_t1);
                if (q == cudaSuccess) {
                    float gpu_ms = 0.0f;
                    cudaEventElapsedTime(&gpu_ms, gpu_t0, gpu_t1);
                    prev_gpu_exec_us = (double)gpu_ms * 1000.0;
                }
                // Confidence from writer thread
                prev_confidence = writer_confidence.load(std::memory_order_relaxed);
                d2h_pending = false;
            }

            _p1c = ni_clock::now();  // after D2H handoff
        }

        _p1d = ni_clock::now();  // end of phase1 (before FPS tracking)
        // Phase1 sub-timer accumulation
        {
            double mc = std::chrono::duration<double, std::micro>(_p1b - _p1a).count();
            double d2h = std::chrono::duration<double, std::micro>(_p1c - _p1b).count();
            double act = std::chrono::duration<double, std::micro>(_p1d - _p1c).count();
            _p1_memcpy.sum += mc;  if (mc > _p1_memcpy.peak) _p1_memcpy.peak = mc;
            _p1_d2h.sum += d2h;    if (d2h > _p1_d2h.peak) _p1_d2h.peak = d2h;
            _p1_action.sum += act;  if (act > _p1_action.peak) _p1_action.peak = act;
        }

        // FPS tracking
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - start_time).count();
        double inst_fps = (elapsed_s > 0) ? frame_count / elapsed_s : 0;

        if (NI_DBG_SHOULD_LOG(frame_count)) { fprintf(stderr, "[NI] phase1 done\\n"); fflush(stderr); }

        // ===== GPU Work (GIL-FREE) ======================================
        // All libtorch C++ ops: .copy_(), graph.replay(), CUDA events.
        // No GIL needed -- training thread can run freely.

        _tp[1] = ni_clock::now();  // TP1: Phase 2a start (GIL held)
        cudaEventRecord(gpu_t0, inference_stream.stream());
        if (NI_DBG_SHOULD_LOG(frame_count)) { fprintf(stderr, "[NI] step3 start\\n"); fflush(stderr); }
        // -- 3. Fill encoder input buffers from SHM data ------------------
        // Graph input buffers are on GPU -- fill CPU staging first, then .copy_().
        // We use pinned_slot (already copied from SHM) as source.
        {
            // Audio: spectral [1, 256, 2] + spatial [1, 2]
            if (modality_mask & MOD_AUDIO) {
                const unsigned char* audio_base = pinned_slot + AUDIO_OFFSET;
                const float* spectral_src = reinterpret_cast<const float*>(
                    audio_base + AUD_SPECTRAL);
                for (size_t ei = 0; ei < encoder_graphs.size(); ei++) {
                    if (encoder_graphs[ei].name == "audio") {
                        auto spec_ptr = encoder_staging[ei].cpu_bufs["spectral"].data_ptr<float>();
                        for (int i = 0; i < 256; i++) {
                            spec_ptr[i * 2 + 0] = spectral_src[i];         // Left
                            spec_ptr[i * 2 + 1] = spectral_src[256 + i];   // Right
                        }
                        auto spat_ptr = encoder_staging[ei].cpu_bufs["spatial"].data_ptr<float>();
                        spat_ptr[0] = *reinterpret_cast<const float*>(audio_base + AUD_ILD_DB);
                        spat_ptr[1] = *reinterpret_cast<const float*>(audio_base + AUD_ITD_MS);
                        // H2D transfer (raw cudaMemcpyAsync -- EXP-21)
                        NI_DBG_BEFORE(frame_count, "H2D audio");
                        raw_tensor_copy_async(encoder_graphs[ei].inputs["spectral"],
                            encoder_staging[ei].cpu_bufs["spectral"],
                            cudaMemcpyHostToDevice, inference_stream.stream());
                        raw_tensor_copy_async(encoder_graphs[ei].inputs["spatial"],
                            encoder_staging[ei].cpu_bufs["spatial"],
                            cudaMemcpyHostToDevice, inference_stream.stream());
                        NI_DBG_AFTER(frame_count, "H2D audio");
                        break;
                    }
                }
            }

            // Proprioception: [1, proprio_dim] (flattened bone data)
            if (modality_mask & MOD_PROPRIOCEPTION) {
                const unsigned char* proprio_base = pinned_slot + PROPRIO_OFFSET;
                int32_t n_bones = *reinterpret_cast<const int32_t*>(proprio_base + PRO_NUM_BONES);
                for (size_t ei = 0; ei < encoder_graphs.size(); ei++) {
                    if (encoder_graphs[ei].name == "proprioception") {
                        auto& cpu_buf = encoder_staging[ei].cpu_bufs["data"];
                        auto dst = cpu_buf.data_ptr<float>();
                        int proprio_dim = (int)cpu_buf.size(1);
                        memset(dst, 0, proprio_dim * sizeof(float));
                        const unsigned char* bone_data = proprio_base + PRO_BONE_DATA;
                        int out_idx = 0;
                        for (int b = 0; b < n_bones && out_idx < proprio_dim; b++) {
                            const float* bone_f = reinterpret_cast<const float*>(
                                bone_data + b * PROPRIO_BONE_STRIDE + 4);  // skip hash
                            int n_copy = std::min(12, proprio_dim - out_idx);
                            memcpy(dst + out_idx, bone_f, n_copy * sizeof(float));
                            out_idx += n_copy;
                        }
                        NI_DBG_BEFORE(frame_count, "H2D proprio");
                        raw_tensor_copy_async(encoder_graphs[ei].inputs["data"],
                            cpu_buf, cudaMemcpyHostToDevice, inference_stream.stream());
                        NI_DBG_AFTER(frame_count, "H2D proprio");
                        break;
                    }
                }
            }

            // Motor: [1, n_handles, features_per_handle]
            if (modality_mask & MOD_MOTOR) {
                const unsigned char* motor_base = pinned_slot + MOTOR_OFFSET;
                for (size_t ei = 0; ei < encoder_graphs.size(); ei++) {
                    if (encoder_graphs[ei].name == "motor") {
                        auto& cpu_buf = encoder_staging[ei].cpu_bufs["data"];
                        auto dst = cpu_buf.data_ptr<float>();
                        int nh = (int)cpu_buf.size(1);
                        int fph = (int)cpu_buf.size(2);
                        memset(dst, 0, nh * fph * sizeof(float));
                        for (int h = 0; h < std::min(nh, (int)MOTOR_MAX_HANDLES); h++) {
                            const float* src = reinterpret_cast<const float*>(
                                motor_base + h * MOTOR_HANDLE_STRIDE);
                            int n_copy = std::min(fph, (int)(MOTOR_HANDLE_STRIDE / 4));
                            memcpy(dst + h * fph, src, n_copy * sizeof(float));
                        }
                        NI_DBG_BEFORE(frame_count, "H2D motor");
                        raw_tensor_copy_async(encoder_graphs[ei].inputs["data"],
                            cpu_buf, cudaMemcpyHostToDevice, inference_stream.stream());
                        NI_DBG_AFTER(frame_count, "H2D motor");
                        break;
                    }
                }
            }

            // Tactile: [1, 256, 8]
            if (modality_mask & MOD_TOUCH) {
                const unsigned char* touch_base = pinned_slot + TOUCH_OFFSET;
                int32_t n_impulses = *reinterpret_cast<const int32_t*>(touch_base + TCH_NUM_IMPULSES);
                for (size_t ei = 0; ei < encoder_graphs.size(); ei++) {
                    if (encoder_graphs[ei].name == "tactile") {
                        auto& cpu_buf = encoder_staging[ei].cpu_bufs["data"];
                        auto dst = cpu_buf.data_ptr<float>();
                        int np = (int)cpu_buf.size(1);
                        int ppd = (int)cpu_buf.size(2);
                        memset(dst, 0, np * ppd * sizeof(float));
                        const float* imp_data = reinterpret_cast<const float*>(
                            touch_base + TCH_IMPULSE_DATA);
                        int n_copy = std::min(n_impulses, np);
                        for (int i = 0; i < n_copy; i++) {
                            int src_floats = TOUCH_IMPULSE_STRIDE / 4;  // 10
                            int n_f = std::min(ppd, src_floats);
                            memcpy(dst + i * ppd, imp_data + i * src_floats, n_f * sizeof(float));
                        }
                        NI_DBG_BEFORE(frame_count, "H2D tactile");
                        raw_tensor_copy_async(encoder_graphs[ei].inputs["data"],
                            cpu_buf, cudaMemcpyHostToDevice, inference_stream.stream());
                        NI_DBG_AFTER(frame_count, "H2D tactile");
                        break;
                    }
                }
            }

            // Voice: [1, voice_dim]
            if (modality_mask & MOD_VOICE) {
                const unsigned char* voice_base = pinned_slot + VOICE_OFFSET;
                for (size_t ei = 0; ei < encoder_graphs.size(); ei++) {
                    if (encoder_graphs[ei].name == "voice") {
                        auto& cpu_buf = encoder_staging[ei].cpu_bufs["data"];
                        auto dst = cpu_buf.data_ptr<float>();
                        int vdim = (int)cpu_buf.size(1);
                        int n_copy = std::min(vdim, (int)(VOICE_SIZE / 4));
                        memcpy(dst, voice_base, n_copy * sizeof(float));
                        if (n_copy < vdim) memset(dst + n_copy, 0, (vdim - n_copy) * sizeof(float));
                        NI_DBG_BEFORE(frame_count, "H2D voice");
                        raw_tensor_copy_async(encoder_graphs[ei].inputs["data"],
                            cpu_buf, cudaMemcpyHostToDevice, inference_stream.stream());
                        NI_DBG_AFTER(frame_count, "H2D voice");
                        break;
                    }
                }
            }

            // Also fill motor handle_state for motor+gaze graph
            if (modality_mask & MOD_MOTOR) {
                const unsigned char* motor_base = pinned_slot + MOTOR_OFFSET;
                auto dst = mg_handle_state_staging.data_ptr<float>();
                int nh = (int)mg_handle_state_staging.size(1);
                int fph_mg = (int)mg_handle_state_staging.size(2);
                memset(dst, 0, nh * fph_mg * sizeof(float));
                for (int h = 0; h < std::min(nh, (int)MOTOR_MAX_HANDLES); h++) {
                    const float* src = reinterpret_cast<const float*>(
                        motor_base + h * MOTOR_HANDLE_STRIDE);
                    int n_copy = std::min(fph_mg, (int)(MOTOR_HANDLE_STRIDE / 4));
                    memcpy(dst + h * fph_mg, src, n_copy * sizeof(float));
                }
                NI_DBG_BEFORE(frame_count, "H2D mg_handle_state");
                raw_tensor_copy_async(mg_handle_state_buf, mg_handle_state_staging,
                    cudaMemcpyHostToDevice, inference_stream.stream());
                NI_DBG_AFTER(frame_count, "H2D mg_handle_state");
            }
        }

        _tp[2] = ni_clock::now();  // TP2: H2D done
        if (NI_DBG_SHOULD_LOG(frame_count)) { fprintf(stderr, "[NI] step4 start\\n"); fflush(stderr); }
        // -- 4. D3D12 Vision ------------------------------------------------
        bool has_fresh_vision_this_frame = false;
        bool d3d12_textures_copied = false;
        // Vision diagnostic: first 3 frames only (confirmed working, no ongoing overhead)
        if (frame_count <= 3) {
            fprintf(stderr, "[NI-VIS-DIAG] f%lld: has_vision=%d modality_mask=0x%x (vis_bit=%d) "
                    "fn_ctx=%d fn_sem=%d fn_cpy=%d tex_handles=%zu\\n",
                    (long long)frame_count, (int)has_vision, modality_mask,
                    (int)((modality_mask & MOD_VISION) != 0),
                    (int)(fn_cuCtxSetCurrent != nullptr),
                    (int)(fn_cuWaitExtSem != nullptr),
                    (int)(fn_cuMemcpy2DAsync != nullptr),
                    tex_handles.size());
            fflush(stderr);
        }
        if (has_vision && (modality_mask & MOD_VISION) &&
            fn_cuCtxSetCurrent && fn_cuWaitExtSem && fn_cuMemcpy2DAsync)
        {
            bool _vis_diag = (frame_count <= 3);

            // Stream for D3D12 driver API calls (fence wait + cuMemcpy2D):
            // Cross-device: vis_stream (cuda:1) for D3D12 ops + texture staging
            // Same-device: inference_stream
            cudaStream_t raw_vis_stream = vision_on_capture ?
                vis_stream.stream() : inference_stream.stream();
            CUstream cu_vis_stream = reinterpret_cast<CUstream>(raw_vis_stream);

            // Set interop device CUDA context for driver API calls
            CUresult _cr = fn_cuCtxSetCurrent(cuda_ctx);
            if (_vis_diag) {
                fprintf(stderr, "[NI-VIS] f%lld: cuCtxSetCurrent ctx=%p rc=%d vision_on_capture=%d\\n",
                        (long long)frame_count, (void*)(uintptr_t)cuda_ctx,
                        (int)_cr, (int)vision_on_capture);
                fflush(stderr);
            }

            // Wait for D3D12 fence (GPU-side wait on vision stream)
            NI_CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait_params = {};
            wait_params.params.fence.value = (unsigned long long)(current_frame);
            if (_vis_diag) {
                fprintf(stderr, "[NI-VIS] f%lld: cuWaitExtSem fence_value=%lld sem=%p stream=%p\\n",
                        (long long)frame_count, (long long)current_frame,
                        (void*)(uintptr_t)fence_sem, (void*)cu_vis_stream);
                fflush(stderr);
            }
            _cr = fn_cuWaitExtSem(&fence_sem, &wait_params, 1, cu_vis_stream);
            if (_vis_diag || _cr != 0) {
                fprintf(stderr, "[NI-VIS] f%lld: cuWaitExtSem rc=%d\\n",
                        (long long)frame_count, (int)_cr);
                fflush(stderr);
            }

            // Copy 4 textures: CUarray -> linear device memory (on vision stream)
            std::string slot_prefix = "slot_" + std::to_string(slot_idx) + "_";
            const char* tex_keys[] = {"pano_left", "pano_right", "fovea_left", "fovea_right"};
            bool _d3d12_ok = true;
            for (int t = 0; t < 4; t++) {
                std::string tex_id = slot_prefix + tex_keys[t];
                auto it = tex_handles.find(tex_id);
                if (it != tex_handles.end()) {
                    auto& th = it->second;
                    if (_vis_diag) {
                        fprintf(stderr, "[NI-VIS] f%lld: cuMemcpy2D %s cuArray=%p linear=%llu size=%lld %dx%d\\n",
                                (long long)frame_count, tex_id.c_str(),
                                (void*)th.cuArray, (unsigned long long)th.linearPtr,
                                (long long)th.linearSize, th.width, th.height);
                        fflush(stderr);
                    }
                    _cr = copy_texture_to_linear(fn_cuMemcpy2DAsync, th.cuArray, th.linearPtr,
                                           th.width, th.height, cu_vis_stream);
                    if (_vis_diag || _cr != 0) {
                        fprintf(stderr, "[NI-VIS] f%lld: cuMemcpy2D %s rc=%d\\n",
                                (long long)frame_count, tex_id.c_str(), (int)_cr);
                        fflush(stderr);
                    }
                    if (_cr != 0) _d3d12_ok = false;
                } else if (_vis_diag) {
                    fprintf(stderr, "[NI-VIS] f%lld: tex %s NOT FOUND in handles\\n",
                            (long long)frame_count, tex_id.c_str());
                    fflush(stderr);
                }
            }

          if (_d3d12_ok) {
            if (vision_on_capture) {
                // Cross-device double-buffered: copy raw BGRA textures from
                // cuda:1 to staging_bufs[stg_write] on cuda:0.  Patchification
                // reads from staging_bufs[1-stg_write] (previous frame's data).
                int vis_dev_idx = static_cast<int>(vis_stream.device_index());
                for (int t = 0; t < 4; t++) {
                    std::string tex_id = slot_prefix + tex_keys[t];
                    auto th_it = tex_handles.find(tex_id);
                    auto st_it = staging_bufs[stg_write].find(tex_id);
                    if (th_it != tex_handles.end() && st_it != staging_bufs[stg_write].end()) {
                        void* src = reinterpret_cast<void*>(
                            static_cast<uintptr_t>(th_it->second.linearPtr));
                        cudaError_t cpy_err = cudaMemcpyPeerAsync(
                            st_it->second.ptr, infer_device,
                            src, vis_dev_idx,
                            st_it->second.size, raw_vis_stream);
                        if (_vis_diag || cpy_err != cudaSuccess) {
                            fprintf(stderr, "[NI-VIS] f%lld: cudaMemcpyPeerAsync %s -> stg[%d] %zuB rc=%d\\n",
                                    (long long)frame_count, tex_id.c_str(), stg_write,
                                    st_it->second.size, (int)cpy_err);
                            fflush(stderr);
                        }
                    }
                }
                // Event: signals when staging_bufs[stg_write] copies are done
                cudaEventRecord(stg_events[stg_write], raw_vis_stream);
                stg_has_data[stg_write] = true;
                stg_slot_idx[stg_write] = slot_idx;
                d3d12_textures_copied = true;
            } else {
                // Same-device: patchify + vision graph inline on inference_stream
                auto get_src_ptr = [&](const std::string& key) -> unsigned char* {
                    std::string tex_id = slot_prefix + key;
                    auto it = tex_handles.find(tex_id);
                    if (it != tex_handles.end())
                        return reinterpret_cast<unsigned char*>(static_cast<uintptr_t>(it->second.linearPtr));
                    return nullptr;
                };
                auto* pano_L_src = get_src_ptr("pano_left");
                auto* pano_R_src = get_src_ptr("pano_right");
                auto* fovea_L_src = get_src_ptr("fovea_left");
                auto* fovea_R_src = get_src_ptr("fovea_right");

                if (_vis_diag) {
                    fprintf(stderr, "[NI-VIS] f%lld: patchify pano=%dx%d(%d patches) fovea=%dx%d(%d patches)\\n",
                            (long long)frame_count, actual_pano_w, PANO_HEIGHT, pano_n_patches,
                            actual_fovea_w, FOVEA_HEIGHT, fovea_n_patches);
                    fflush(stderr);
                }
                cudaStream_t inf_raw = inference_stream.stream();
                if (pano_L_src)
                    launch_bgra_to_rgb_patches(pano_L_src, pano_L_patches.data_ptr<float>(),
                                                actual_pano_w, pano_patch_cols, pano_n_patches, inf_raw);
                if (pano_R_src)
                    launch_bgra_to_rgb_patches(pano_R_src, pano_R_patches.data_ptr<float>(),
                                                actual_pano_w, pano_patch_cols, pano_n_patches, inf_raw);
                if (fovea_L_src)
                    launch_bgra_to_rgb_patches(fovea_L_src, fovea_L_patches.data_ptr<float>(),
                                                actual_fovea_w, fovea_patch_cols, fovea_n_patches, inf_raw);
                if (fovea_R_src)
                    launch_bgra_to_rgb_patches(fovea_R_src, fovea_R_patches.data_ptr<float>(),
                                                actual_fovea_w, fovea_patch_cols, fovea_n_patches, inf_raw);

                // Patch concat + angular + vision graph on inference_stream
                {
                    // EXP-21: raw cudaMemcpyAsync for vision concat (bypass allocator)
                    NI_DBG_BEFORE(frame_count, "vis patches concat");
                    int patch_dim = (int)pano_L_patches.size(1);  // 768
                    size_t patch_bytes = patch_dim * sizeof(float);
                    float* vp_base = vision_patches_buf.data_ptr<float>();
                    int off = 0;
                    cudaMemcpyAsync(vp_base + off * patch_dim,
                        pano_L_patches.data_ptr<float>(),
                        pano_n_patches * patch_bytes,
                        cudaMemcpyDeviceToDevice, inf_raw);
                    off += pano_n_patches;
                    cudaMemcpyAsync(vp_base + off * patch_dim,
                        pano_R_patches.data_ptr<float>(),
                        pano_n_patches * patch_bytes,
                        cudaMemcpyDeviceToDevice, inf_raw);
                    off += pano_n_patches;
                    cudaMemcpyAsync(vp_base + off * patch_dim,
                        fovea_L_patches.data_ptr<float>(),
                        fovea_n_patches * patch_bytes,
                        cudaMemcpyDeviceToDevice, inf_raw);
                    off += fovea_n_patches;
                    cudaMemcpyAsync(vp_base + off * patch_dim,
                        fovea_R_patches.data_ptr<float>(),
                        fovea_n_patches * patch_bytes,
                        cudaMemcpyDeviceToDevice, inf_raw);
                    NI_DBG_AFTER(frame_count, "vis patches concat");

                    NI_DBG_BEFORE(frame_count, "vis angular");
                    int pano_total = 2 * pano_n_patches;
                    float* va_base = vision_angular_buf.data_ptr<float>();
                    // Pano angular LUT (D2D)
                    cudaMemcpyAsync(va_base, pano_angular_lut.data_ptr<float>(),
                        pano_total * 2 * sizeof(float),
                        cudaMemcpyDeviceToDevice, inf_raw);
                    // Fovea angular: need .add_() for gaze offset -- keep ATen for these
                    // (small tensors, minimal allocator impact)
                    fovea_ang_buf.select(1, 0).copy_(fovea_local_az);
                    fovea_ang_buf.select(1, 0).add_(gaze_az_rad);
                    fovea_ang_buf.select(1, 1).copy_(fovea_local_el);
                    fovea_ang_buf.select(1, 1).add_(gaze_el_rad);
                    // Copy fovea angular to vision angular buf (D2D)
                    size_t fovea_ang_bytes = fovea_n_patches * 2 * sizeof(float);
                    cudaMemcpyAsync(va_base + pano_total * 2,
                        fovea_ang_buf.data_ptr<float>(),
                        fovea_ang_bytes, cudaMemcpyDeviceToDevice, inf_raw);
                    cudaMemcpyAsync(va_base + (pano_total + fovea_n_patches) * 2,
                        fovea_ang_buf.data_ptr<float>(),
                        fovea_ang_bytes, cudaMemcpyDeviceToDevice, inf_raw);
                    NI_DBG_AFTER(frame_count, "vis angular");

                    vis_graph->replay();
                }
                has_fresh_vision_this_frame = true;
                fresh_vision_accum++;
            }
          } // _d3d12_ok
        }

        _tp[3] = ni_clock::now();  // TP3: D3D12 vision done
        if (frame_count <= 3) {
            fprintf(stderr, "[NI-VIS-DIAG] f%lld: d3d12_copied=%d has_fresh_vision=%d stg_has[0]=%d stg_has[1]=%d stg_write=%d\\n",
                    (long long)frame_count, (int)d3d12_textures_copied,
                    (int)has_fresh_vision_this_frame,
                    (int)stg_has_data[0], (int)stg_has_data[1], stg_write);
            fflush(stderr);
        }
        if (NI_DBG_SHOULD_LOG(frame_count)) { fprintf(stderr, "[NI] step5 start\\n"); fflush(stderr); }
        // -- 5. CUDA Graph Replays + token assembly -----------------------
        // EXP-20: Combined mode replays ONE graph for encoders+forward+motor/gaze.
        // Legacy mode replays separate graphs with CPU .copy_() between them.
        // Vision pipeline (D3D12 double-buffer + patchify + vision graph) runs
        // in BOTH modes since vision has a separate D3D12 dependency.
        {
            // 5a. Double-buffered deferred vision pipeline (cross-device only)
            // Read from the PREVIOUS frame's staging set while current frame
            // copies into stg_write on vis_stream.  The event wait is a no-op
            // since the copy finished ~33ms ago (previous frame).
            {
                int stg_read = 1 - stg_write;
                if (stg_has_data[stg_read]) {
                    bool _vis_diag2 = (frame_count <= 3);
                    cudaStreamWaitEvent(inference_stream.stream(), stg_events[stg_read], 0);

                    int read_slot = stg_slot_idx[stg_read];
                    auto get_staging_ptr = [&](const std::string& key) -> unsigned char* {
                        std::string tex_id = "slot_" + std::to_string(read_slot) + "_" + key;
                        auto it = staging_bufs[stg_read].find(tex_id);
                        if (it != staging_bufs[stg_read].end()) return it->second.ptr;
                        return nullptr;
                    };
                    auto* pano_L_src = get_staging_ptr("pano_left");
                    auto* pano_R_src = get_staging_ptr("pano_right");
                    auto* fovea_L_src = get_staging_ptr("fovea_left");
                    auto* fovea_R_src = get_staging_ptr("fovea_right");

                    if (_vis_diag2) {
                        fprintf(stderr, "[NI-VIS] f%lld: dblbuf patchify stg[%d] (slot %d) pano=%dx%d(%d) fovea=%dx%d(%d)\\n",
                                (long long)frame_count, stg_read, read_slot,
                                actual_pano_w, PANO_HEIGHT, pano_n_patches,
                                actual_fovea_w, FOVEA_HEIGHT, fovea_n_patches);
                        fflush(stderr);
                    }
                    cudaStream_t inf_raw = inference_stream.stream();
                    if (pano_L_src)
                        launch_bgra_to_rgb_patches(pano_L_src, pano_L_patches.data_ptr<float>(),
                                                    actual_pano_w, pano_patch_cols, pano_n_patches, inf_raw);
                    if (pano_R_src)
                        launch_bgra_to_rgb_patches(pano_R_src, pano_R_patches.data_ptr<float>(),
                                                    actual_pano_w, pano_patch_cols, pano_n_patches, inf_raw);
                    if (fovea_L_src)
                        launch_bgra_to_rgb_patches(fovea_L_src, fovea_L_patches.data_ptr<float>(),
                                                    actual_fovea_w, fovea_patch_cols, fovea_n_patches, inf_raw);
                    if (fovea_R_src)
                        launch_bgra_to_rgb_patches(fovea_R_src, fovea_R_patches.data_ptr<float>(),
                                                    actual_fovea_w, fovea_patch_cols, fovea_n_patches, inf_raw);

                    // Patch concat + angular + vision graph on inference_stream
                    {
                        // EXP-21: raw cudaMemcpyAsync (bypass allocator)
                        NI_DBG_BEFORE(frame_count, "vis patches concat");
                        int patch_dim = (int)pano_L_patches.size(1);
                        size_t patch_bytes = patch_dim * sizeof(float);
                        float* vp_base = vision_patches_buf.data_ptr<float>();
                        int off = 0;
                        cudaMemcpyAsync(vp_base + off * patch_dim,
                            pano_L_patches.data_ptr<float>(),
                            pano_n_patches * patch_bytes,
                            cudaMemcpyDeviceToDevice, inf_raw);
                        off += pano_n_patches;
                        cudaMemcpyAsync(vp_base + off * patch_dim,
                            pano_R_patches.data_ptr<float>(),
                            pano_n_patches * patch_bytes,
                            cudaMemcpyDeviceToDevice, inf_raw);
                        off += pano_n_patches;
                        cudaMemcpyAsync(vp_base + off * patch_dim,
                            fovea_L_patches.data_ptr<float>(),
                            fovea_n_patches * patch_bytes,
                            cudaMemcpyDeviceToDevice, inf_raw);
                        off += fovea_n_patches;
                        cudaMemcpyAsync(vp_base + off * patch_dim,
                            fovea_R_patches.data_ptr<float>(),
                            fovea_n_patches * patch_bytes,
                            cudaMemcpyDeviceToDevice, inf_raw);
                        NI_DBG_AFTER(frame_count, "vis patches concat");

                        NI_DBG_BEFORE(frame_count, "vis angular");
                        int pano_total = 2 * pano_n_patches;
                        float* va_base = vision_angular_buf.data_ptr<float>();
                        cudaMemcpyAsync(va_base, pano_angular_lut.data_ptr<float>(),
                            pano_total * 2 * sizeof(float),
                            cudaMemcpyDeviceToDevice, inf_raw);
                        fovea_ang_buf.select(1, 0).copy_(fovea_local_az);
                        fovea_ang_buf.select(1, 0).add_(gaze_az_rad);
                        fovea_ang_buf.select(1, 1).copy_(fovea_local_el);
                        fovea_ang_buf.select(1, 1).add_(gaze_el_rad);
                        size_t fovea_ang_bytes = fovea_n_patches * 2 * sizeof(float);
                        cudaMemcpyAsync(va_base + pano_total * 2,
                            fovea_ang_buf.data_ptr<float>(),
                            fovea_ang_bytes, cudaMemcpyDeviceToDevice, inf_raw);
                        cudaMemcpyAsync(va_base + (pano_total + fovea_n_patches) * 2,
                            fovea_ang_buf.data_ptr<float>(),
                            fovea_ang_bytes, cudaMemcpyDeviceToDevice, inf_raw);
                        NI_DBG_AFTER(frame_count, "vis angular");

                        vis_graph->replay();
                    }
                    has_fresh_vision_this_frame = true;
                    fresh_vision_accum++;
                } else if (frame_count <= 3) {
                    fprintf(stderr, "[NI-VIS] f%lld: dblbuf stg[%d] not ready yet (warmup)\\n",
                            (long long)frame_count, stg_read);
                    fflush(stderr);
                }
            }

            // Flip double-buffer write index for next frame
            if (d3d12_textures_copied) {
                stg_write = 1 - stg_write;
            }

            // 5b. Non-vision encoders + forward + motor/gaze
            if (!use_combined) {
            // Legacy: separate graph replays with CPU .copy_() between them
            for (auto& eg : encoder_graphs) {
                eg.graph->replay();
            }

            // Assemble flat tokens for forward model
            if (mod_index.count("vision") && has_fresh_vision_this_frame) {
                fwd_flat_tokens_buf.select(1, mod_index["vision"]).copy_(
                    vision_token_buf.squeeze(0).squeeze(0), /*non_blocking=*/true);
            }
            for (auto& eg : encoder_graphs) {
                if (mod_index.count(eg.name)) {
                    auto out_squeezed = eg.output.squeeze(0);
                    if (out_squeezed.dim() == 2) out_squeezed = out_squeezed.squeeze(0);
                    fwd_flat_tokens_buf.select(1, mod_index[eg.name]).copy_(out_squeezed);
                }
            }

            // Copy context into forward model input
            raw_tensor_copy_async(fwd_context_buf, context_buffer,
                cudaMemcpyDeviceToDevice, inference_stream.stream());

            fwd_graph->replay();

            mg_latents_buf.copy_(fwd_latents_buf);
            mg_last_step_buf.copy_(fwd_last_step_buf);

            mg_graph->replay();

            } else {
                // ========== EXP-20: Combined graph path ==========
                // Vision graph already replayed above (separate D3D12 dependency).
                // The combined graph runs: non-vision encoders -> token assembly ->
                // forward model -> motor+gaze in ONE replay.
                // Vision token flows via shared buffer (vision_graph output feeds
                // into combined_graph's vision token input).
                raw_tensor_copy_async(fwd_context_buf, context_buffer,
                cudaMemcpyDeviceToDevice, inference_stream.stream());
                combined_graph->replay();
            }
        }

        _tp[4] = ni_clock::now();  // TP4: graph replays done
        // -- 6. Update context buffer (shift left, append ctx_update) ----
        {
            // EXP-21: context shift with raw cudaMemcpyAsync (no .clone() allocation)
            int ctx_tokens = (int)((ctx_len - 1) * n_mod);
            if (ctx_tokens > n_mod) {
                NI_DBG_BEFORE(frame_count, "ctx_shift_copy");
                // Shift left: memmove semantics (overlapping regions)
                // Use cudaMemcpyAsync D2D -- safe for same-device overlapping
                // regions when copy direction is left-to-right (dst < src).
                float* ctx_base = context_buffer.data_ptr<float>();
                int d = (int)context_buffer.size(2);  // d_model
                cudaMemcpyAsync(ctx_base,
                    ctx_base + n_mod * d,
                    (ctx_tokens - n_mod) * d * sizeof(float),
                    cudaMemcpyDeviceToDevice, inference_stream.stream());
                NI_DBG_AFTER(frame_count, "ctx_shift_copy");
            }
            NI_DBG_BEFORE(frame_count, "ctx_append");
            {
                float* ctx_base = context_buffer.data_ptr<float>();
                int d = (int)context_buffer.size(2);
                cudaMemcpyAsync(ctx_base + (ctx_tokens - n_mod) * d,
                    fwd_ctx_update_buf.data_ptr<float>(),
                    n_mod * d * sizeof(float),
                    cudaMemcpyDeviceToDevice, inference_stream.stream());
            }
            NI_DBG_AFTER(frame_count, "ctx_append");
        }
        _tp[5] = ni_clock::now();  // TP5: context shift done
        cudaEventRecord(gpu_t1, inference_stream.stream());
        if (NI_DBG_SHOULD_LOG(frame_count)) {
            cudaError_t err = cudaStreamSynchronize(inference_stream.stream());
            NI_DBG_LOG(frame_count, "ctx_update sync: %s",
                    err == cudaSuccess ? "OK" : cudaGetErrorString(err));
        }

        // (Action SHM write moved to Phase 1 -- pure memcpy, runs GIL-free)

        // ===== Yield (already GIL-free) ==================================
        NI_DBG_BEFORE(frame_count, "yield");
        _tp[6] = ni_clock::now();  // TP6: before yield
#ifdef _WIN32
        SwitchToThread();
#else
        std::this_thread::yield();
#endif
        _tp[7] = ni_clock::now();  // TP7: after yield

        // GPU exec timing: use value computed in Phase 1 from PREVIOUS
        // frame's gpu_t0/t1 (events are complete by then).
        gpu_exec_sum += prev_gpu_exec_us;
        if (prev_gpu_exec_us > gpu_exec_peak) gpu_exec_peak = prev_gpu_exec_us;

        // ===== Weight sync (acquire GIL for Python callback) =============
        if (!weight_sync_callback.is_none() && frame_count % 30 == 0) {
            py::gil_scoped_acquire gil;
            try {
                weight_sync_callback();
            } catch (py::error_already_set& e) {
                if (e.matches(PyExc_KeyboardInterrupt)) {
                    fprintf(stderr, "[NI] KeyboardInterrupt in weight_sync -- shutting down\\n");
                    fflush(stderr);
                    *shutdown_ptr = 1;
                    running = false;
                    e.restore();
                    PyErr_Clear();
                    continue;
                }
                fprintf(stderr, "[NI] weight_sync_callback error: %s\\n", e.what());
                fflush(stderr);
                e.restore();
                PyErr_Clear();
            }
        }

        _tp[8] = ni_clock::now();  // TP8: after weight sync

        // ===== Stats callback (acquire GIL for Python) =================
        if (!stats_callback.is_none() && stats_interval > 0
            && frame_count > 0 && frame_count % stats_interval == 0) {
            py::gil_scoped_acquire gil;
            try {
                py::dict info;
                info["frame_count"] = frame_count;
                info["fps"] = inst_fps;
                info["stall_count"] = slow_frame_accum;
                info["poll_misses"] = poll_miss_accum;
                info["d2h_skips"] = d2h_skip_accum;
                info["has_fresh_vision"] = (fresh_vision_accum > 0);
                info["motor_confidence"] = prev_confidence;
                slow_frame_accum = 0;
                poll_miss_accum = 0;
                d2h_skip_accum = 0;
                fresh_vision_accum = 0;
                // Motor/gaze snapshot for Python logging (from writer thread)
                {
                    py::dict motor;
                    std::lock_guard<std::mutex> lock(motor_snapshot_mtx);
                    motor["pos_x"] = _last_motor[0];
                    motor["pos_y"] = _last_motor[1];
                    motor["pos_z"] = _last_motor[2];
                    motor["str0"]  = _last_motor[3];
                    motor["max_str"] = _last_motor[4];
                    motor["gaze_az"] = _last_motor[5];
                    motor["gaze_el"] = _last_motor[6];
                    info["motor_summary"] = motor;
                }
                // Per-phase timing (mean + peak us over stats_interval)
                if (_tm_count > 0) {
                    static const char* _phase_names[] = {
                        "phase1", "h2d", "d3d12", "graphs", "ctx_shift",
                        "gpu_sync", "wsync", "stats_cb", "d2h", "rb_store"
                    };
                    py::dict timing;
                    double n = (double)_tm_count;
                    double total_mean = 0;
                    for (int i = 0; i < 10; i++) {
                        double mean = _tm[i].sum / n;
                        timing[py::cast(_phase_names[i])] = mean;
                        timing[py::cast(std::string(_phase_names[i]) + "_peak")] = _tm[i].peak;
                        total_mean += mean;
                    }
                    timing["total"] = total_mean;
                    timing["n_frames"] = _tm_count;
                    // Device-side GPU execution time (h2d through ctx_shift)
                    timing["gpu_exec"] = gpu_exec_sum / n;
                    timing["gpu_exec_peak"] = gpu_exec_peak;
                    // Phase1 sub-breakdown
                    timing["p1_memcpy"] = _p1_memcpy.sum / n;
                    timing["p1_memcpy_peak"] = _p1_memcpy.peak;
                    timing["p1_d2h_sync"] = _p1_d2h.sum / n;
                    timing["p1_d2h_sync_peak"] = _p1_d2h.peak;
                    timing["p1_action"] = _p1_action.sum / n;
                    timing["p1_action_peak"] = _p1_action.peak;
                    info["frame_timing"] = timing;
                    // Reset accumulators
                    for (int i = 0; i < 10; i++) { _tm[i].sum = 0; _tm[i].peak = 0; }
                    gpu_exec_sum = 0.0;
                    gpu_exec_peak = 0.0;
                    _p1_memcpy = {}; _p1_d2h = {}; _p1_action = {};
                    _tm_count = 0;
                }
                stats_callback(info);
            } catch (py::error_already_set& e) {
                if (e.matches(PyExc_KeyboardInterrupt)) {
                    fprintf(stderr, "[NI] KeyboardInterrupt in stats_callback -- shutting down\\n");
                    fflush(stderr);
                    *shutdown_ptr = 1;
                    running = false;
                    e.restore();
                    PyErr_Clear();
                    continue;
                }
                fprintf(stderr, "[NI] stats_callback error: %s\\n", e.what());
                fflush(stderr);
                e.restore();
                PyErr_Clear();
            }
        }

        _tp[9] = ni_clock::now();  // TP9: before D2H (after stats cb)
        // ===== Non-blocking D2H + Writer Thread ==========================
        // D2H copies go to double-buffered pinned staging. Writer thread
        // waits for the event and writes to action SHM independently.
        {
            c10::cuda::CUDAStreamGuard d2h_guard(inference_stream);
            c10::InferenceMode infer_guard;

            int widx = pinned_write_idx.load(std::memory_order_relaxed);

            // Non-blocking D2H: GPU graph outputs -> pinned staging
            // Writer thread copies staging -> pinned_action_bufs after event sync.
            // Previously used cudaMemcpyAsync(HostToHost) here which is synchronous
            // per CUDA spec, causing a 21ms stall waiting for graph replay.
            NI_DBG_BEFORE(frame_count, "D2H motor (async)");
            motor_d2h_staging[widx].copy_(motor_out_buf.reshape({-1}), /*non_blocking=*/true);
            NI_DBG_AFTER(frame_count, "D2H motor (async)");

            NI_DBG_BEFORE(frame_count, "D2H gaze (async)");
            gaze_d2h_staging[widx].copy_(gaze_out_buf.reshape({-1}), /*non_blocking=*/true);
            NI_DBG_AFTER(frame_count, "D2H gaze (async)");

            NI_DBG_BEFORE(frame_count, "D2H voice (async)");
            voice_d2h_staging[widx].copy_(voice_out_buf.reshape({-1}), /*non_blocking=*/true);
            NI_DBG_AFTER(frame_count, "D2H voice (async)");

            // Confidence: GPU -> pinned confidence staging[widx]
            torch::Tensor& conf_stg = (widx == 0) ? confidence_staging_0 : confidence_staging_1;
            conf_stg.copy_(
                motor_confidence_buf.reshape({1}), /*non_blocking=*/true);

            cudaEventRecord(d2h_events[widx], inference_stream.stream());

            // Signal writer thread: "slot widx has new data"
            writer_pending_slot.store(widx, std::memory_order_release);

            // Flip to other buffer for next frame
            pinned_write_idx.store(1 - widx, std::memory_order_relaxed);
        }
        d2h_pending = true;

        _tp[10] = ni_clock::now();  // TP10: D2H done, before RB store
        NI_DBG_BEFORE(frame_count, "replay_buffer_store");
        // -- 8. Store to replay buffer -----------------------------------
        // OPT-2: RB stores run on a SEPARATE low-priority stream so they
        // overlap with the next frame's SHM poll + D3D12 copy + graph replay.
        // rb_dep_event ensures RB copies start after graph outputs are ready.
        {
            // Signal that graph outputs are ready on inference_stream
            cudaEventRecord(rb_dep_event, inference_stream.stream());
            // RB copies wait for graph outputs, then run on rb_store_stream
            cudaStreamWaitEvent(rb_store_stream.stream(), rb_dep_event, 0);
            c10::cuda::CUDAStreamGuard rb_stream_guard(rb_store_stream);
            c10::InferenceMode rb_infer_guard;
            int64_t write_head = cached_rb_wh[0];
            int64_t rb_slot = write_head;
            int32_t presence = 0;

            // Vision backbone: [1, total_patches, d_internal] -> fp16 -> rb
            // Vision graph runs on inference_stream (cuda:0), same device as rb.
            if (has_fresh_vision_this_frame) {
                NI_DBG_BEFORE(frame_count, "RB vision");
                auto vbb = vision_backbone_buf.squeeze(0);
                auto vbb_narrow = vbb.narrow(0, 0, total_patches);
                auto vbb_half = vbb_narrow.to(torch::kFloat16);
                rb_vision_buf[rb_slot].narrow(0, 0, total_patches).copy_(vbb_half, true);
                cached_rb_vis_np[rb_slot] = total_patches;
                presence |= (int32_t)MOD_VISION;
                NI_DBG_AFTER(frame_count, "RB vision");
            }

            // Audio
            if (modality_mask & MOD_AUDIO) {
                NI_DBG_BEFORE(frame_count, "RB audio");
                for (auto& eg : encoder_graphs) {
                    if (eg.name == "audio") {
                        rb_audio_spectral_buf[rb_slot].copy_(
                            eg.inputs["spectral"].squeeze(0), true);
                        rb_audio_spatial_buf[rb_slot].copy_(
                            eg.inputs["spatial"].squeeze(0), true);
                        break;
                    }
                }
                presence |= (int32_t)MOD_AUDIO;
                NI_DBG_AFTER(frame_count, "RB audio");
            }

            // Proprioception
            if (modality_mask & MOD_PROPRIOCEPTION) {
                NI_DBG_BEFORE(frame_count, "RB proprio");
                for (auto& eg : encoder_graphs) {
                    if (eg.name == "proprioception") {
                        rb_proprio_buf[rb_slot].copy_(
                            eg.inputs["data"].squeeze(0), true);
                        break;
                    }
                }
                presence |= (int32_t)MOD_PROPRIOCEPTION;
                NI_DBG_AFTER(frame_count, "RB proprio");
            }

            // Motor
            if (modality_mask & MOD_MOTOR) {
                NI_DBG_BEFORE(frame_count, "RB motor");
                for (auto& eg : encoder_graphs) {
                    if (eg.name == "motor") {
                        rb_motor_buf[rb_slot].copy_(
                            eg.inputs["data"].squeeze(0), true);
                        break;
                    }
                }
                presence |= (int32_t)MOD_MOTOR;
                NI_DBG_AFTER(frame_count, "RB motor");
            }

            // Tactile
            if (modality_mask & MOD_TOUCH) {
                NI_DBG_BEFORE(frame_count, "RB tactile");
                for (auto& eg : encoder_graphs) {
                    if (eg.name == "tactile") {
                        rb_tactile_buf[rb_slot].copy_(
                            eg.inputs["data"].squeeze(0), true);
                        break;
                    }
                }
                presence |= (int32_t)MOD_TOUCH;
                NI_DBG_AFTER(frame_count, "RB tactile");
            }

            // Voice
            if (modality_mask & MOD_VOICE) {
                NI_DBG_BEFORE(frame_count, "RB voice");
                for (auto& eg : encoder_graphs) {
                    if (eg.name == "voice") {
                        rb_voice_buf[rb_slot].copy_(
                            eg.inputs["data"].squeeze(0), true);
                        break;
                    }
                }
                presence |= (int32_t)MOD_VOICE;
                NI_DBG_AFTER(frame_count, "RB voice");
            }

            // Motor capability: extract from SHM motor slice -> CPU staging -> GPU
            {
                const unsigned char* motor_base = pinned_slot + MOTOR_OFFSET;
                int cap_dim = cached_rb_cap_dim;
                float* cap_ptr = cached_rb_cap_ptr;
                memset(cap_ptr, 0, cap_dim * sizeof(float));
                const unsigned char* bs_data = motor_base + MOT_BODY_SETS;
                for (int s = 0; s < NUM_BODY_SETS && (s * 2 + 1) < cap_dim; s++) {
                    const float* bs = reinterpret_cast<const float*>(
                        bs_data + s * BODY_SET_STATE_STRIDE);
                    cap_ptr[s] = bs[0];               // capacity
                    cap_ptr[NUM_BODY_SETS + s] = bs[1]; // accumulated energy
                }
                const float* nrg = reinterpret_cast<const float*>(motor_base + MOT_ENERGY);
                if (cap_dim > NUM_BODY_SETS * 2) {
                    cap_ptr[NUM_BODY_SETS * 2] = nrg[2];  // NRG_TOTAL at offset 8
                }
                NI_DBG_BEFORE(frame_count, "RB motor_cap");
                rb_motor_capability_buf[rb_slot].copy_(rb_motor_cap_staging, true);
                NI_DBG_AFTER(frame_count, "RB motor_cap");
            }

            // Gaze -> CPU staging -> GPU
            cached_rb_gaze_ptr[0] = gaze_az_rad;
            cached_rb_gaze_ptr[1] = gaze_el_rad;
            NI_DBG_BEFORE(frame_count, "RB gaze");
            rb_gaze_buf[rb_slot].copy_(rb_gaze_staging, true);
            NI_DBG_AFTER(frame_count, "RB gaze");

            // -- 8b. Cached encoder tokens (EXP-8 token cache) ----------
            // Tokens MUST be written before metadata (valid=true) so
            // training never sees a "valid" slot with stale tokens.
            NI_DBG_BEFORE(frame_count, "RB tok_vision");
            if (has_fresh_vision_this_frame && token_bufs.count("vision")) {
                token_bufs["vision"][rb_slot].copy_(vision_token_buf.squeeze(0), true);
            }
            NI_DBG_AFTER(frame_count, "RB tok_vision");
            // Non-vision tokens
            for (auto& eg : encoder_graphs) {
                if (token_bufs.count(eg.name)) {
                    NI_DBG_LOG(frame_count, ">> RB tok_%s", eg.name.c_str());
                    auto out = eg.output;
                    if (out.dim() == 3) out = out.squeeze(0);  // [1, d_model]
                    token_bufs[eg.name][rb_slot].copy_(out, true);
                    NI_DBG_LOG(frame_count, "<< RB tok_%s OK", eg.name.c_str());
                }
            }

            // Record CUDA event AFTER all GPU copies for this slot.
            // gather_batch_native waits on this event on the training
            // stream, ensuring all non-blocking copies are visible.
            // Event is recorded on rb_store_stream (where the copies run).
            {
                cudaStream_t rb_raw = rb_store_stream.stream();
                if (g_rb_write_event == nullptr) {
                    cudaEventCreateWithFlags(&g_rb_write_event,
                                             cudaEventDisableTiming);
                }
                cudaEventRecord(g_rb_write_event, rb_raw);
            }

            // Metadata (CPU tensors -- use cached ptrs)
            // Written AFTER all GPU copies + event record so training
            // never sees valid=true for a slot with incomplete GPU data.
            cached_rb_mod_present[rb_slot] = presence;
            cached_rb_frame_nums[rb_slot] = current_frame;
            cached_rb_token_valid[rb_slot] = true;
            int64_t total_wr = cached_rb_total_wr[0] + 1;
            cached_rb_total_wr[0] = total_wr;
            cached_rb_seq_ids[rb_slot] = total_wr;
            cached_rb_valid[rb_slot] = true;  // LAST -- gates visibility

            // Advance ring
            int64_t new_head = (write_head + 1) % rb_capacity;
            cached_rb_wh[0] = new_head;
            int64_t cur_size = cached_rb_sz[0];
            if (cur_size < rb_capacity) {
                cached_rb_sz[0] = cur_size + 1;
            }
        }
        NI_DBG_AFTER(frame_count, "replay_buffer_store");
        _tp[11] = ni_clock::now();  // TP11: RB store done = frame end

        // -- 9a. Accumulate per-frame timing -----------------------------
        {
            auto _us = [&](int a, int b) -> double {
                return std::chrono::duration_cast<std::chrono::microseconds>(
                    _tp[b] - _tp[a]).count();
            };
            double vals[10] = {
                _us(0, 1),    // 0: phase1 (parse + GIL reacquire)
                _us(1, 2),    // 1: h2d
                _us(2, 3),    // 2: d3d12 vision
                _us(3, 4),    // 3: graph replays
                _us(4, 5),    // 4: ctx_shift
                _us(6, 7),    // 5: gpu_sync (just the sync, inside GIL-free)
                _us(7, 8),    // 6: weight_sync (includes GIL reacquire)
                _us(8, 9),    // 7: stats_cb
                _us(9, 10),   // 8: d2h
                _us(10, 11),  // 9: rb_store
            };
            double total_us = 0;
            for (int i = 0; i < 10; i++) {
                _tm[i].sum += vals[i];
                if (vals[i] > _tm[i].peak) _tm[i].peak = vals[i];
                total_us += vals[i];
            }
            if (total_us > 50000.0) slow_frame_accum++;  // > 50ms
            _tm_count++;
        }

        // -- 9b. FPS tracking --------------------------------------------
        {
            auto now = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(now - start_time).count();
            if (elapsed_s > 0 && frame_count > 1) {
                double instant_fps = 1.0 / (elapsed_s / frame_count);
                if (fps_ema == 0.0) fps_ema = instant_fps;
                else fps_ema = EMA_ALPHA * instant_fps + (1.0 - EMA_ALPHA) * fps_ema;
            }
        }

        // -- 10. Periodic shutdown check (reads CPU tensor via cached ptr) --
        if (frame_count % 100 == 0) {
            if (*shutdown_ptr != 0) {
                running = false;
            }
        }
    }

    // --- Cleanup --------------------------------------------------------
    if (gpu_t0) { cudaEventDestroy(gpu_t0); gpu_t0 = nullptr; }
    if (gpu_t1) { cudaEventDestroy(gpu_t1); gpu_t1 = nullptr; }
    if (d2h_event) { cudaEventDestroy(d2h_event); d2h_event = nullptr; }
    {
        int vis_dev = static_cast<int>(vis_stream.device_index());
        cudaSetDevice(vis_dev);
        for (int db = 0; db < 2; db++) {
            if (stg_events[db]) {
                cudaEventDestroy(stg_events[db]);
                stg_events[db] = nullptr;
            }
        }
        cudaSetDevice(infer_device);
    }

    if (shm_ptr) {
        UnmapViewOfFile(const_cast<unsigned char*>(shm_ptr));
        CloseHandle(shm_handle);
    }
    if (action_ptr) {
        UnmapViewOfFile(const_cast<unsigned char*>(action_ptr));
        CloseHandle(action_handle);
    }
    if (nvcuda) FreeLibrary(nvcuda);

    // --- Return stats ---------------------------------------------------
    auto total_time = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();

    // Shutdown writer thread
    writer_running.store(false, std::memory_order_relaxed);
    if (action_writer_thread.joinable()) {
        action_writer_thread.join();
    }
    cudaEventDestroy(d2h_events[0]);
    cudaEventDestroy(d2h_events[1]);
    if (rb_dep_event) cudaEventDestroy(rb_dep_event);

    py::dict result;
    result["frames"] = frame_count;
    result["total_time_s"] = total_time;
    result["avg_fps"] = frame_count > 0 ? frame_count / total_time : 0.0;
    result["ema_fps"] = fps_ema;
    return result;
}


// =========================================================================
// Module registration
// =========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_patchify_kernel", &init_patchify_kernel,
          "Load patchify CUDA kernel from CUBIN/PTX (compiled by NVRTC)",
          py::arg("cubin_data"));

    m.def("init_fused_kernel", &init_fused_kernel,
          "Load a fused NVRTC kernel from CUBIN data",
          py::arg("cubin_data"), py::arg("kernel_name"));

    m.def("launch_fused_gelu_bias", &launch_fused_gelu_bias,
          "Fused GELU + bias: out = GELU(x + bias). x=bf16, bias=f32",
          py::arg("x"), py::arg("bias"));

    m.def("launch_fused_layernorm_residual", &launch_fused_layernorm_residual,
          "Fused LayerNorm + residual: out = LN(x) + res. x,res=bf16, gamma,beta=f32",
          py::arg("x"), py::arg("residual"),
          py::arg("gamma"), py::arg("beta"), py::arg("eps"));

    m.def("scalar_item_gil_free", &scalar_item_gil_free,
          "Extract scalar from tensor with GIL released during sync",
          py::arg("t"));

    m.def("bool_item_gil_free", &bool_item_gil_free,
          "Extract bool from tensor with GIL released during sync",
          py::arg("t"));

    m.def("tensor_to_list_gil_free", &tensor_to_list_gil_free,
          "Copy 1D CUDA tensor to CPU list with GIL released during sync",
          py::arg("t"));

    m.def("clip_grad_norm_gil_free", &clip_grad_norm_gil_free,
          "clip_grad_norm_ with GIL released during cudaStreamSynchronize",
          py::arg("parameters"), py::arg("max_norm"),
          py::arg("norm_type") = 2.0, py::arg("error_if_nonfinite") = false);

    m.def("gather_batch_native", &gather_batch_native,
          "GPU replay buffer gather with minimal Python overhead",
          py::arg("vision_buf"), py::arg("audio_spectral_buf"),
          py::arg("audio_spatial_buf"), py::arg("proprio_buf"),
          py::arg("motor_buf"), py::arg("tactile_buf"),
          py::arg("voice_buf"), py::arg("motor_capability_buf"),
          py::arg("gaze_buf"), py::arg("indices"),
          py::arg("modality_present"), py::arg("skip_vision") = false);

    m.def("build_context_index_plan", &build_context_index_plan,
          "Deduplicate context indices and compute remap positions",
          py::arg("context_windows"), py::arg("capacity"));

    m.def("assemble_context_tokens", &assemble_context_tokens,
          "Reassemble deduplicated tokens to [B, ctx*n_mod, d_model]",
          py::arg("modality_tokens"), py::arg("positions"),
          py::arg("B"), py::arg("ctx_count"),
          py::arg("n_mod"), py::arg("d_model"));

    m.def("imagination_index_plan", &imagination_index_plan,
          "Compute successor indices for imagination steps",
          py::arg("primary_indices"), py::arg("imagination_steps"),
          py::arg("capacity"));

    m.def("backward_gil_free", &backward_gil_free,
          "Run loss.backward() with GIL released during C++ autograd",
          py::arg("loss"), py::arg("retain_graph") = false,
          py::arg("create_graph") = false);

    m.def("copy_pairs_gil_free", &copy_pairs_gil_free,
          "Copy list of (src, dst) tensor pairs with GIL released",
          py::arg("pairs"), py::arg("non_blocking") = true);

    m.def("run_native_inference_loop", &run_native_inference_loop,
          "C++ inference loop with GIL-free at::cuda::CUDAGraph::replay()",
          py::arg("vision_graph_obj"), py::arg("fwd_graph_obj"),
          py::arg("motor_gaze_graph_obj"),
          py::arg("encoder_graph_list"),
          py::arg("vision_patches_buf"),
          py::arg("vision_angular_buf"), py::arg("vision_token_buf"),
          py::arg("vision_backbone_buf"),
          py::arg("fwd_flat_tokens_buf"),
          py::arg("fwd_context_buf"), py::arg("fwd_latents_buf"),
          py::arg("fwd_last_step_buf"), py::arg("fwd_ctx_update_buf"),
          py::arg("mg_latents_buf"),
          py::arg("mg_handle_state_buf"), py::arg("mg_last_step_buf"),
          py::arg("motor_out_buf"), py::arg("motor_confidence_buf"),
          py::arg("gaze_out_buf"), py::arg("voice_out_buf"),
          py::arg("context_buffer"),
          py::arg("tex_handles_dict"), py::arg("fence_semaphore_ptr"),
          py::arg("cuda_context_ptr"),
          py::arg("pano_angular_lut"), py::arg("fovea_local_az"),
          py::arg("fovea_local_el"),
          py::arg("vis_dblbuf_0"), py::arg("vis_dblbuf_1"),
          py::arg("vis_ang_dblbuf_0"), py::arg("vis_ang_dblbuf_1"),
          py::arg("cpu_shm_name"), py::arg("action_shm_name"),
          py::arg("rb_vision_buf"), py::arg("rb_vision_n_patches"),
          py::arg("rb_audio_spectral_buf"), py::arg("rb_audio_spatial_buf"),
          py::arg("rb_proprio_buf"), py::arg("rb_motor_buf"),
          py::arg("rb_tactile_buf"), py::arg("rb_voice_buf"),
          py::arg("rb_motor_capability_buf"), py::arg("rb_gaze_buf"),
          py::arg("rb_modality_present"), py::arg("rb_frame_numbers"),
          py::arg("rb_valid"), py::arg("rb_sequence_ids"),
          py::arg("rb_token_bufs_dict"), py::arg("rb_token_valid"),
          py::arg("rb_write_head"), py::arg("rb_size"),
          py::arg("rb_total_written"), py::arg("rb_capacity"),
          py::arg("active_modalities"), py::arg("n_mod"),
          py::arg("d_model"), py::arg("ctx_len"),
          py::arg("n_handles"), py::arg("features_per_handle"),
          py::arg("max_patches"), py::arg("warmup_frames"),
          py::arg("shutdown_flag"), py::arg("has_vision"),
          py::arg("phase_rebuild_flag_ptr"),
          py::arg("inference_stream_id"),
          py::arg("inference_stream_device_index"),
          py::arg("inference_stream_device_type"),
          py::arg("weight_sync_callback") = py::none(),
          py::arg("stats_callback") = py::none(),
          py::arg("stats_interval") = 0,
          py::arg("perf_counter_ref_us") = 0,
          py::arg("vision_stream_id") = (int64_t)0,
          py::arg("vision_stream_device_index") = (int64_t)0,
          py::arg("vision_stream_device_type") = (int64_t)0,
          py::arg("dev_limits_buf") = torch::tensor({1.0f, 1.0f, 200.0f}),
          py::arg("combined_graph_obj") = py::none());
}
"""

# ---------------------------------------------------------------------------
# CUDA source -- bgra_to_rgb_patches kernel for vision patchification
# ---------------------------------------------------------------------------

CUDA_SOURCE_NATIVE = r"""
#include <cuda_runtime.h>

// Fused: crop + BGRA->RGB channel swap + uint8->float32 + /255 + patch reshape
// Input:  linear device memory [H, W, 4] uint8 BGRA
// Output: [n_patches, 16*16*3] float32 RGB  (flat patch_dim=768)
// Thread mapping: one thread per output pixel (n_patches * 256 threads total).
extern "C" __global__ void ni_bgra_to_rgb_patches_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int input_w,
    int patch_cols,
    int n_patches
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_patches * 256) return;

    int pixel_idx = tid % 256;
    int patch_idx = tid / 256;
    int py = pixel_idx / 16;
    int px = pixel_idx % 16;
    int patch_row = patch_idx / patch_cols;
    int patch_col = patch_idx % patch_cols;

    int src_y = patch_row * 16 + py;
    int src_x = patch_col * 16 + px;
    int src_off = (src_y * input_w + src_x) * 4;

    float r = (float)input[src_off + 2] * (1.0f / 255.0f);
    float g = (float)input[src_off + 1] * (1.0f / 255.0f);
    float b = (float)input[src_off + 0] * (1.0f / 255.0f);

    int dst_off = (patch_idx * 256 + py * 16 + px) * 3;
    output[dst_off + 0] = r;
    output[dst_off + 1] = g;
    output[dst_off + 2] = b;
}

// C-linkage launch wrapper (takes stream to avoid default-stream races)
extern "C" void launch_bgra_to_rgb_patches(
    const unsigned char* input, float* output,
    int input_w, int patch_cols, int n_patches,
    cudaStream_t stream
) {
    int total_threads = n_patches * 256;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    ni_bgra_to_rgb_patches_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, input_w, patch_cols, n_patches);
}
"""

# ---------------------------------------------------------------------------
# NVRTC kernel compilation (bypasses nvcc + MSVC incompatibility)
# ---------------------------------------------------------------------------

def _compile_patchify_ptx(build_dir, force=False):
    """Compile patchify CUDA kernel to PTX using NVRTC. No MSVC needed."""
    ptx_file = os.path.join(build_dir, 'patchify.ptx')
    if not force and os.path.exists(ptx_file):
        with open(ptx_file, 'rb') as f:
            return f.read()

    import ctypes
    # Locate nvrtc DLL
    _cuda_home = os.environ.get(
        'CUDA_HOME',
        os.environ.get('CUDA_PATH',
                       r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'))
    _nvrtc_paths = [
        os.path.join(_cuda_home, 'bin', 'x64', 'nvrtc64_130_0.dll'),
        os.path.join(_cuda_home, 'bin', 'nvrtc64_130_0.dll'),
    ]
    nvrtc = None
    for p in _nvrtc_paths:
        if os.path.exists(p):
            try:
                nvrtc = ctypes.CDLL(p)
                break
            except OSError:
                continue
    if nvrtc is None:
        # Try system PATH
        try:
            nvrtc = ctypes.CDLL('nvrtc64_130_0')
        except OSError:
            log.warning("NVRTC not found -- patchify kernel unavailable")
            return None

    # Extract just the kernel (no launch wrapper -- that uses <<<>>> syntax)
    # Strip #include <cuda_runtime.h> -- NVRTC has built-in CUDA types.
    kernel_src = CUDA_SOURCE_NATIVE.split(
        '// C-linkage launch wrapper')[0]
    kernel_src = kernel_src.replace('#include <cuda_runtime.h>', '')
    kernel_src = kernel_src.encode('utf-8')

    prog = ctypes.c_void_p()
    rc = nvrtc.nvrtcCreateProgram(
        ctypes.byref(prog), kernel_src, b'patchify.cu', 0, None, None)
    if rc != 0:
        log.error("nvrtcCreateProgram failed: %d", rc)
        return None

    # Compile to CUBIN (native machine code) with sm_120, not PTX.
    # CUDA 13.1 NVRTC always emits PTX 9.1 which requires driver r590+.
    # Driver 581.x (CUDA 13.0) only supports PTX 9.0.  CUBIN bypasses
    # the driver's PTX JIT entirely -- no version check.
    import torch
    cap = torch.cuda.get_device_capability()
    arch_flag = f'--gpu-architecture=sm_{cap[0]}{cap[1]}'.encode('utf-8')
    opts = (ctypes.c_char_p * 1)(arch_flag)
    rc = nvrtc.nvrtcCompileProgram(prog, 1, opts)
    if rc != 0:
        # Get compile log
        log_size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_size))
        compile_log = ctypes.create_string_buffer(log_size.value)
        nvrtc.nvrtcGetProgramLog(prog, compile_log)
        log.error("NVRTC compile failed (rc=%d):\n%s", rc,
                  compile_log.value.decode('utf-8', errors='replace'))
        nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        return None

    # Get CUBIN (not PTX) -- native code that loads without JIT
    cubin_size = ctypes.c_size_t()
    nvrtc.nvrtcGetCUBINSize(prog, ctypes.byref(cubin_size))
    cubin_buf = ctypes.create_string_buffer(cubin_size.value)
    nvrtc.nvrtcGetCUBIN(prog, cubin_buf)
    nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    cubin = cubin_buf.raw[:cubin_size.value]
    # Cache to disk
    with open(ptx_file, 'wb') as f:
        f.write(cubin)
    log.info("NVRTC compiled patchify kernel: %d bytes CUBIN (sm_%d%d)",
             len(cubin), cap[0], cap[1])
    return cubin


# The UE action frame's limb-row count; the C++ constant of the same name in CPP_SOURCE
# (lem/tests/test_native_action_frame_rows.py checks that the two agree).
ACTION_FRAME_LIMB_ROWS = 23


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------

_MODULE = None
_LOAD_ATTEMPTED = False
_DISABLED = False


def disable_native_ops():
    """Disable the C++ extension (use Python fallbacks)."""
    global _DISABLED, _LOAD_ATTEMPTED, _MODULE
    _DISABLED = True
    _LOAD_ATTEMPTED = True
    _MODULE = None
    log.info("lem_native C++ extension disabled by --no-native flag")


def load_native_ops():
    """
    JIT-compile the lem_native C++ extension.  Returns the compiled module
    or None if compilation fails.  Cached after first successful load.
    """
    global _MODULE, _LOAD_ATTEMPTED

    if _LOAD_ATTEMPTED:
        return _MODULE

    _LOAD_ATTEMPTED = True

    try:
        if not _setup_msvc_env():
            log.warning("MSVC environment not available -- lem_native disabled")
            return None

        log.info("Compiling lem_native C++ extension (first run only, ~30-60s)...")

        from torch.utils.cpp_extension import load_inline

        build_dir = os.path.join(os.path.dirname(__file__), '..', '.cuda_build')
        os.makedirs(build_dir, exist_ok=True)

        # Invalidate stale build cache per-source: CPP and CUDA are hashed
        # separately so a CPP-only change doesn't force CUDA recompilation
        # (nvcc + MSVC 19.50 has PyTorch header incompatibility).
        import hashlib
        _cpp_hash = hashlib.md5(CPP_SOURCE.encode('utf-8')).hexdigest()
        _cuda_hash = hashlib.md5(CUDA_SOURCE_NATIVE.encode('utf-8')).hexdigest()
        _src_hash = f"{_cpp_hash}:{_cuda_hash}"
        _hash_file = os.path.join(build_dir, '.source_hash')
        _old_hash = ''
        _cuda_changed = False
        if os.path.exists(_hash_file):
            with open(_hash_file, 'r') as f:
                _old_hash = f.read().strip()
        if _old_hash != _src_hash:
            # Determine what changed
            if _old_hash and ':' in _old_hash:
                _old_cpp, _, _old_cuda = _old_hash.partition(':')
                _cpp_changed = (_old_cpp != _cpp_hash)
                _cuda_changed = (_old_cuda != _cuda_hash)
            elif _old_hash:
                # Old format (no colon): migrating -- preserve CUDA objects
                _cpp_changed = True
                _cuda_changed = False
                log.info("Migrating build cache to split hash format "
                         "(preserving CUDA objects)")
            else:
                # No hash file at all (fresh build dir)
                _cpp_changed = True
                _cuda_changed = True
            # Only remove the object files for the source that changed
            for _f in os.listdir(build_dir):
                _p = os.path.join(build_dir, _f)
                if os.path.isfile(_p):
                    _is_cuda_obj = _f.startswith('cuda.')
                    _is_cpp_obj = _f.startswith('main.')
                    if (_cpp_changed and _is_cpp_obj) or \
                       (_cuda_changed and _is_cuda_obj) or \
                       _f.endswith('.so') or _f.endswith('.pyd') or \
                       _f == 'lock':
                        os.remove(_p)
            what = []
            if _cpp_changed: what.append("CPP")
            if _cuda_changed: what.append("CUDA")
            if what:
                log.info("Cleared stale %s build cache (source hash changed)",
                         "+".join(what))

        import torch

        # PyTorch builds the linker command as "{dirname(where cl)}/link.exe"
        # in the ninja file.  On some Windows setups (GCP VMs), this resolves
        # to "/link.exe" which fails.  Fix: after load_inline writes the
        # ninja file (but before running it), patch the linker path.  We do
        # this via a two-phase approach: first write sources, then fix + build.
        if os.name == 'nt':
            import subprocess as _sp
            import shutil as _sh
            _link_exe = _sh.which('link')
            if _link_exe:
                _link_exe = os.path.abspath(_link_exe)
                log.info("Found link.exe at: %s", _link_exe)

        try:
            _MODULE = load_inline(
                name='lem_native',
                cpp_sources=[CPP_SOURCE],
                cuda_sources=[],
                build_directory=build_dir,
                verbose=False,
                with_cuda=True,
                extra_ldflags=['cuda.lib'],  # CUDA Driver API (cuModuleLoad, cuLaunchKernel)
            )
        except RuntimeError as _e:
            if os.name == 'nt' and 'link.exe' in str(_e) and _link_exe:
                # Patch the ninja build file and retry
                _ninja_file = os.path.join(build_dir, 'build.ninja')
                if os.path.exists(_ninja_file):
                    with open(_ninja_file, 'r') as f:
                        _ninja_content = f.read()
                    # Replace bad linker path with full path (ninja needs $: for colon)
                    _ninja_link = _link_exe.replace('\\', '/').replace(':', '$:')
                    _ninja_content = _ninja_content.replace(
                        '"/link.exe"', f'"{_ninja_link}"')
                    with open(_ninja_file, 'w') as f:
                        f.write(_ninja_content)
                    log.info("Patched build.ninja linker path, retrying...")
                    # Re-run ninja manually
                    _result = _sp.run(['ninja', '-v'],
                                      cwd=build_dir,
                                      capture_output=True, text=True)
                    if _result.returncode != 0:
                        raise RuntimeError(
                            f"Ninja retry failed:\n{_result.stdout}\n{_result.stderr}")
                    # Load the built module
                    import importlib
                    import importlib.util
                    _pyd = os.path.join(build_dir, 'lem_native.pyd')
                    _spec = importlib.util.spec_from_file_location('lem_native', _pyd)
                    _MODULE = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_MODULE)
                else:
                    raise
            else:
                raise

        # Compile patchify kernel via NVRTC (no MSVC needed) and load PTX.
        # cuModuleLoadData requires an active CUDA context -- force init.
        # Non-fatal: if PTX version exceeds driver capability (e.g. CUDA
        # toolkit 13.1 vs driver 13.0), vision patchify falls back to no-op.
        if torch.cuda.is_available():
            torch.cuda.init()
            torch.zeros(1, device='cuda')  # ensure primary context is active
            ptx = _compile_patchify_ptx(build_dir, _cuda_changed)
            if ptx:
                try:
                    _MODULE.init_patchify_kernel(ptx)
                    log.info("Patchify CUDA kernel loaded via NVRTC")
                except RuntimeError as _e:
                    log.warning("Patchify kernel load failed (driver may not "
                                "support PTX from this CUDA toolkit): %s", _e)
        # Write source hash so next run skips rebuild
        with open(_hash_file, 'w') as f:
            f.write(_src_hash)
        log.info("lem_native C++ extension compiled and loaded")
    except Exception as e:
        import traceback
        log.error("lem_native compilation failed:\n%s", traceback.format_exc())
        _MODULE = None

    return _MODULE


# ---------------------------------------------------------------------------
# Python fallback wrappers
# ---------------------------------------------------------------------------

def scalar_item_gil_free(tensor):
    """Extract scalar float from tensor, releasing GIL during sync.

    This is the core GIL-reduction primitive.  Every .item() on a CUDA tensor
    triggers cudaStreamSynchronize which blocks until ALL pending GPU work on
    that stream completes.  Normally the GIL is held during this wait, starving
    the inference thread.  The C++ wrapper releases the GIL so inference can
    run concurrently.
    """
    mod = load_native_ops()
    if mod is not None:
        return mod.scalar_item_gil_free(tensor)
    return tensor.item()


def bool_item_gil_free(tensor):
    """Extract bool from tensor, releasing GIL during sync."""
    mod = load_native_ops()
    if mod is not None:
        return mod.bool_item_gil_free(tensor)
    return bool(tensor.item())


def tensor_to_list_gil_free(tensor):
    """Copy 1D CUDA tensor to Python list, releasing GIL during the sync.

    Replaces `tensor.cpu().tolist()` pattern which holds GIL during the
    device-to-host copy.
    """
    mod = load_native_ops()
    if mod is not None:
        return mod.tensor_to_list_gil_free(tensor)
    return tensor.cpu().tolist()


def clip_grad_norm_gil_free(parameters, max_norm, norm_type=2.0,
                            error_if_nonfinite=False):
    """clip_grad_norm_ with GIL release.  Falls back to PyTorch if C++ unavailable."""
    mod = load_native_ops()
    if mod is not None:
        return mod.clip_grad_norm_gil_free(
            parameters, max_norm, norm_type, error_if_nonfinite)
    else:
        import torch
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite, foreach=True)


def backward_gil_free(loss, retain_graph=False, create_graph=False):
    """Run loss.backward() with GIL released during C++ autograd pass."""
    mod = load_native_ops()
    if mod is not None:
        mod.backward_gil_free(loss, retain_graph, create_graph)
    else:
        loss.backward(retain_graph=retain_graph, create_graph=create_graph)


def copy_pairs_gil_free(pairs, non_blocking=True):
    """Copy list of (src, dst) tensor pairs with GIL released.

    Replaces the Python for-loop in _stage_weights() and maybe_sync_weights()
    that iterates 133 pre-built (train_param, staged_buf) pairs.  Each
    Python-level copy_() holds the GIL during pybind11 dispatch; with 133
    pairs that accumulates to ~40ms.  The C++ version releases the GIL for
    the entire batch, reducing hold to <1ms.
    """
    mod = load_native_ops()
    if mod is not None:
        mod.copy_pairs_gil_free(pairs, non_blocking)
    else:
        for src, dst in pairs:
            dst.copy_(src, non_blocking=non_blocking)


def run_native_inference_loop(bundle, tex_handles, fence_sem, cuda_ctx,
                               pano_angular_lut, fovea_local_az, fovea_local_el,
                               vis_dblbuf, vis_ang_dblbuf,
                               cpu_shm_name, action_shm_name,
                               replay_buffer, config,
                               shutdown_flag, has_vision=False,
                               inference_stream=None,
                               vision_stream=None,
                               weight_sync_callback=None,
                               stats_callback=None,
                               stats_interval=0,
                               perf_counter_ref_us=None,
                               dev_limits_buf=None):
    """Launch the C++ inference loop.

    The C++ loop releases the GIL at entry and runs GIL-free.
    Graph replays use at::cuda::CUDAGraph::replay() via pybind11 cast
    access -- pure C++, no GIL needed (pool mgmt + cudaGraphLaunch).
    Training keeps the GIL permanently on its own thread.

    Args:
        bundle: NativeInferenceBundle from capture_all_inference_graphs().
        tex_handles: Dict from cuda_d3d12_interop.get_texture_handles().
        fence_sem: int64 from cuda_d3d12_interop.get_fence_semaphore().
        cuda_ctx: (context_ptr, device_ordinal) from get_cuda_context().
        pano_angular_lut: [n_pano_total, 2] precomputed panorama angular positions.
        fovea_local_az: [n_fovea_patches] local fovea azimuth grid.
        fovea_local_el: [n_fovea_patches] local fovea elevation grid.
        vis_dblbuf: (buf0, buf1) each [1, 33000, 768].
        vis_ang_dblbuf: (buf0, buf1) each [1, 33000, 2].
        cpu_shm_name: CPU pipeline SHM name string.
        action_shm_name: Action buffer SHM name string.
        replay_buffer: GPUReplayBuffer instance (for tensor refs).
        config: Resolved master config dict.
        shutdown_flag: torch.Tensor [1] int64 CPU -- set [0]=1 to stop loop.
        has_vision: Whether D3D12 vision interop is available.

    Returns:
        Dict with {frames, total_time_s, avg_fps, ema_fps} or None on failure.
    """
    # The UE action frame has 23 limb rows (ACTION_FRAME_LIMB_ROWS in the C++). The
    # writer copies min(n_handles, 23) rows into a pinned buffer that is never cleared
    # and sends all 23, so a body with fewer rows would send uninitialised memory to UE
    # as the missing rows.
    if int(bundle.n_handles) < ACTION_FRAME_LIMB_ROWS:
        raise ValueError(
            f"the native inference loop needs at least {ACTION_FRAME_LIMB_ROWS} body "
            f"handles to fill the UE action frame's {ACTION_FRAME_LIMB_ROWS} limb rows; "
            f"this bundle has {int(bundle.n_handles)} (config body.n_handles)")
    mod = load_native_ops()
    if mod is None:
        raise RuntimeError("lem_native C++ extension not available -- "
                           "cannot run native inference loop")

    import torch

    # Pass Python CUDAGraph objects -- C++ extracts at::cuda::CUDAGraph*
    # via pybind11 cast for GIL-free replay()
    vision_graph_obj = bundle.vision_graph
    use_combined = bundle.combined_graph is not None

    if use_combined:
        # EXP-20: combined graph replaces encoder+forward+motor/gaze
        # Pass combined buffers via legacy parameter slots so C++ reads
        # motor/gaze outputs and ctx_update from the right memory.
        fwd_graph_obj = bundle.vision_graph  # dummy, not replayed in combined mode
        mg_graph_obj = bundle.vision_graph  # dummy, not replayed in combined mode
        combined_graph_obj = bundle.combined_graph

        # Build encoder_graph_list with combined graph's encoder input buffers
        # so C++ H2D section can fill them. The graph objects are dummies
        # (vision_graph) since they won't be replayed in combined mode.
        encoder_graph_list = []
        for mod_name in bundle.active_modalities:
            if mod_name == "vision":
                continue
            if mod_name in bundle.combined_encoder_input_bufs:
                enc_bufs = bundle.combined_encoder_input_bufs[mod_name]
                # Dummy output tensor (not used in combined mode)
                dummy_out = torch.zeros(1, 1, bundle.d_model,
                                        device=bundle.vision_token_buf.device)
                encoder_graph_list.append(
                    (mod_name, bundle.vision_graph, enc_bufs, dummy_out))

        # Map combined buffers to legacy parameter names
        _fwd_flat_tokens = bundle.combined_flat_tokens_buf
        _fwd_context = bundle.combined_context_buf
        _fwd_latents = bundle.combined_latents_buf
        _fwd_last_step = bundle.combined_last_step_buf
        _fwd_ctx_update = bundle.combined_ctx_update_buf
        _mg_latents = bundle.combined_latents_buf  # same buffer
        _mg_handles = bundle.combined_handle_state_buf
        _mg_last_step = bundle.combined_last_step_buf
        _motor_out = bundle.combined_motor_out_buf
        _motor_conf = bundle.combined_confidence_buf
        _gaze_out = bundle.combined_gaze_out_buf
        _voice_out = bundle.combined_voice_out_buf
    else:
        # Legacy separate graphs
        fwd_graph_obj = bundle.fwd_graph
        mg_graph_obj = bundle.motor_gaze_graph
        combined_graph_obj = None

        # Build encoder graph list with CUDAGraph objects:
        # [(name, CUDAGraph_obj, {key: tensor}, output), ...]
        encoder_graph_list = []
        for mod_name, (graph, input_bufs, output) in bundle.encoder_graphs.items():
            encoder_graph_list.append((mod_name, graph, input_bufs, output))

        _fwd_flat_tokens = bundle.fwd_flat_tokens_buf
        _fwd_context = bundle.fwd_context_buf
        _fwd_latents = bundle.fwd_latents_buf
        _fwd_last_step = bundle.fwd_last_step_buf
        _fwd_ctx_update = bundle.fwd_ctx_update_buf
        _mg_latents = bundle.mg_latents_buf
        _mg_handles = bundle.mg_handle_state_buf
        _mg_last_step = bundle.mg_last_step_buf
        _motor_out = bundle.motor_out_buf
        _motor_conf = bundle.motor_confidence_buf
        _gaze_out = bundle.gaze_out_buf
        _voice_out = bundle.voice_out_buf

    # Bind shared ring state so training thread sees C++ updates
    rb = replay_buffer
    rb_write_head, rb_size, rb_total_written = rb.bind_shared_ring_state()

    # Token cache buffers
    token_bufs_dict = dict(rb._token_bufs)

    ctx_ptr = cuda_ctx[0] if isinstance(cuda_ctx, tuple) else cuda_ctx

    # Pack the capture stream for C++ -- must replay graphs on the same stream
    # they were captured on (memory pool is bound to the capture stream).
    cap_stream = inference_stream or bundle.capture_stream
    if cap_stream is None:
        raise RuntimeError("No inference stream provided and bundle has no capture_stream -- "
                           "cannot replay CUDA graphs safely")
    stream_id = cap_stream.stream_id
    stream_device_index = cap_stream.device_index
    stream_device_type = cap_stream.device_type

    return mod.run_native_inference_loop(
        # CUDA graph Python objects (C++ extracts at::cuda::CUDAGraph*)
        vision_graph_obj,
        fwd_graph_obj,
        mg_graph_obj,
        # Encoder graphs (with CUDAGraph objects)
        encoder_graph_list,
        # Vision static I/O buffers
        bundle.vision_patches_buf,
        bundle.vision_angular_buf,
        bundle.vision_token_buf,
        bundle.vision_backbone_buf,
        # Forward model static I/O buffers (mapped to combined in EXP-20)
        _fwd_flat_tokens,
        _fwd_context,
        _fwd_latents,
        _fwd_last_step,
        _fwd_ctx_update,
        # Motor + gaze static I/O buffers (mapped to combined in EXP-20)
        _mg_latents,
        _mg_handles,
        _mg_last_step,
        _motor_out,
        _motor_conf,
        _gaze_out,
        _voice_out,
        # Context buffer
        bundle.context_buffer,
        # D3D12 vision handles
        tex_handles if tex_handles else {},
        fence_sem if fence_sem else 0,
        ctx_ptr if ctx_ptr else 0,
        # Vision precomputes
        pano_angular_lut,
        fovea_local_az,
        fovea_local_el,
        vis_dblbuf[0], vis_dblbuf[1],
        vis_ang_dblbuf[0], vis_ang_dblbuf[1],
        # SHM names
        cpu_shm_name,
        action_shm_name,
        # Replay buffer tensors
        rb._vision_buf if rb._vision_buf is not None else torch.empty(0),
        rb._vision_n_patches if rb._vision_n_patches is not None else torch.empty(0, dtype=torch.int64),
        rb._audio_spectral_buf if rb._audio_spectral_buf is not None else torch.empty(0),
        rb._audio_spatial_buf if rb._audio_spatial_buf is not None else torch.empty(0),
        rb._proprio_buf if rb._proprio_buf is not None else torch.empty(0),
        rb._motor_buf if rb._motor_buf is not None else torch.empty(0),
        rb._tactile_buf if rb._tactile_buf is not None else torch.empty(0),
        rb._voice_buf if rb._voice_buf is not None else torch.empty(0),
        rb._motor_capability_buf,
        rb._gaze_buf,
        rb._modality_present,
        rb._frame_numbers,
        torch.from_numpy(rb._valid) if hasattr(rb._valid, '__array__') else rb._valid,
        rb._sequence_ids,
        token_bufs_dict,
        rb._token_valid,
        rb_write_head,
        rb_size,
        rb_total_written,
        rb._capacity,
        # Config
        list(config["_active_modalities"]),
        bundle.n_mod,
        bundle.d_model,
        bundle.ctx_len,
        bundle.n_handles,
        bundle.features_per_handle,
        bundle.max_patches,
        config.get("warmup_frames", 100),
        # Control
        shutdown_flag,
        has_vision,
        0,  # phase_rebuild_flag_ptr (unused for now)
        # Inference stream (must match capture stream for graph pool bindings)
        stream_id,
        stream_device_index,
        stream_device_type,
        # Weight sync callback
        weight_sync_callback,
        # Stats callback (for JSONL dashboard emission)
        stats_callback,
        stats_interval,
        # Timestamp reference so C++ timestamps align with perf_counter
        perf_counter_ref_us if perf_counter_ref_us is not None
            else time.perf_counter_ns() // 1000,
        # Vision stream (for cross-device vision pipeline on cuda:1)
        vision_stream.stream_id if vision_stream else 0,
        vision_stream.device_index if vision_stream else 0,
        vision_stream.device_type if vision_stream else 0,
        # Developmental constraints [rom_limit, strength_limit, position_range_cm]
        dev_limits_buf if dev_limits_buf is not None
            else torch.tensor([1.0, 1.0, 200.0], dtype=torch.float32),
        # EXP-20: combined graph (None = legacy separate graphs)
        combined_graph_obj if combined_graph_obj is not None else None,
    )
