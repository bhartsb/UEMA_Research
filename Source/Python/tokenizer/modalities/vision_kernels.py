# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Vision CUDA kernels — fused crop + BGRA->RGB + uint8->float32 + patch reshape.

JIT-compiled via torch.utils.cpp_extension.load_inline(). Replaces the
PyTorch-op sequence (index select + float() + /255 + reshape+permute+reshape)
with a single kernel pass: one read (4 bytes BGRA), one write (12 bytes RGB float32).

Falls back to None if nvcc is unavailable, letting vision.py use the original path.
"""

import logging
import os
import subprocess

log = logging.getLogger(__name__)


def _setup_msvc_env():
    """
    Auto-detect MSVC and set up PATH/INCLUDE/LIB for cl.exe.
    Uses vswhere.exe to find VS installation, then runs vcvarsall.bat x64
    and captures the resulting environment variables.
    """
    # Skip if cl.exe is already findable
    try:
        subprocess.run(['where', 'cl'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Find vswhere.exe (shipped with VS 2017+)
    vswhere = os.path.join(
        os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'),
        'Microsoft Visual Studio', 'Installer', 'vswhere.exe'
    )
    if not os.path.isfile(vswhere):
        log.warning("vswhere.exe not found at %s", vswhere)
        return False

    # Query VS installation path
    try:
        result = subprocess.run(
            [vswhere, '-latest', '-property', 'installationPath',
             '-requires', 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'],
            capture_output=True, text=True, check=True
        )
        vs_path = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning("vswhere failed to find VS installation with C++ tools")
        return False

    if not vs_path or not os.path.isdir(vs_path):
        log.warning("VS installation path invalid: %s", vs_path)
        return False

    vcvarsall = os.path.join(vs_path, 'VC', 'Auxiliary', 'Build', 'vcvarsall.bat')
    if not os.path.isfile(vcvarsall):
        log.warning("vcvarsall.bat not found at %s", vcvarsall)
        return False

    # Run vcvarsall.bat and capture resulting environment
    try:
        result = subprocess.run(
            f'"{vcvarsall}" x64 && set',
            capture_output=True, text=True, shell=True, check=True
        )
    except subprocess.CalledProcessError as e:
        log.warning("vcvarsall.bat failed: %s", e)
        return False

    # Parse environment variables from output (case-insensitive keys)
    env_vars = {}
    for line in result.stdout.splitlines():
        if '=' in line:
            key, _, value = line.partition('=')
            env_vars[key.upper()] = value

    # Apply the key variables that cl.exe / link.exe / nvcc need
    for key in ('PATH', 'INCLUDE', 'LIB', 'LIBPATH', 'UCRTVERSION',
                'UNIVERSALCRTSDKDIR', 'VCTOOLSINSTALLDIR', 'WINDOWSSDKDIR',
                'WINDOWSSDKLIBVERSION', 'WINDOWSSDKVERSION'):
        if key in env_vars:
            os.environ[key] = env_vars[key]

    log.info("MSVC environment configured from %s", vcvarsall)
    return True

_KERNEL_MODULE = None
_LOAD_ATTEMPTED = False

CUDA_SOURCE = r"""
#include <cuda_runtime.h>

// Fused: crop + BGRA->RGB channel swap + uint8->float32 + /255 + patch reshape
// Input:  [H, W, 4] uint8 BGRA (contiguous, original dimensions)
// Output: [n_patches, 16, 16, 3] float32 RGB
// Crop is implicit: only the first crop_h x crop_w pixels are addressed.
// Thread mapping: one thread per output pixel (n_patches * 256 threads total).

__global__ void bgra_to_rgb_patches_kernel(
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

// C-linkage launch wrapper callable from the C++ source
extern "C" void launch_bgra_to_rgb_patches(
    const unsigned char* input, float* output,
    int input_w, int patch_cols, int n_patches
) {
    int total_threads = n_patches * 256;
    int block_size = 256;
    int grid_size = (total_threads + block_size - 1) / block_size;
    bgra_to_rgb_patches_kernel<<<grid_size, block_size>>>(
        input, output, input_w, patch_cols, n_patches);
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

extern "C" void launch_bgra_to_rgb_patches(
    const unsigned char* input, float* output,
    int input_w, int patch_cols, int n_patches);

torch::Tensor bgra_to_rgb_patches(
    torch::Tensor input,
    int crop_h,
    int crop_w,
    int patch_cols,
    int n_patches
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kUInt8, "input must be uint8");
    TORCH_CHECK(input.dim() == 3 && input.size(2) == 4, "input must be [H, W, 4]");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    int input_w = input.size(1);

    auto output = torch::empty(
        {n_patches, 16, 16, 3},
        torch::dtype(torch::kFloat32).device(input.device())
    );

    launch_bgra_to_rgb_patches(
        input.data_ptr<unsigned char>(),
        output.data_ptr<float>(),
        input_w, patch_cols, n_patches);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bgra_to_rgb_patches", &bgra_to_rgb_patches,
          "Fused BGRA->RGB patches (crop + channel swap + float32 + /255 + reshape)");
}
"""


def load_vision_kernels():
    """
    JIT compile CUDA vision kernels. Returns compiled module or None.
    Cached after first successful load.
    """
    global _KERNEL_MODULE, _LOAD_ATTEMPTED

    if _LOAD_ATTEMPTED:
        return _KERNEL_MODULE

    _LOAD_ATTEMPTED = True

    try:
        # Ensure CUDA_HOME is set for nvcc discovery
        if not os.environ.get('CUDA_HOME') and not os.environ.get('CUDA_PATH'):
            cuda_default = r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
            if os.path.isdir(cuda_default):
                versions = sorted(os.listdir(cuda_default), reverse=True)
                if versions:
                    cuda_home = os.path.join(cuda_default, versions[0])
                    os.environ['CUDA_HOME'] = cuda_home
                    log.info("Auto-detected CUDA_HOME: %s", cuda_home)

        # Ensure MSVC (cl.exe) is discoverable
        if not _setup_msvc_env():
            log.warning("MSVC environment not available — cannot compile CUDA kernels")
            return _KERNEL_MODULE

        from torch.utils.cpp_extension import load_inline

        # Cache directory for compiled kernels
        build_dir = os.path.join(os.path.dirname(__file__), '..', '..', '.cuda_build')
        os.makedirs(build_dir, exist_ok=True)

        # Cache link.exe path for ninja workaround on GCP VMs
        _link_exe = None
        if os.name == 'nt':
            import shutil as _sh
            _link_exe = _sh.which('link')
            if _link_exe:
                _link_exe = os.path.abspath(_link_exe)

        try:
            _KERNEL_MODULE = load_inline(
                name='vision_kernels',
                cpp_sources=[CPP_SOURCE],
                cuda_sources=[CUDA_SOURCE],
                build_directory=build_dir,
                verbose=False,
                extra_cuda_cflags=['-allow-unsupported-compiler'],
            )
        except RuntimeError as _e:
            # Workaround: ninja generates "/link.exe" (relative) on GCP VMs.
            # Patch build.ninja with absolute path and retry.
            if os.name == 'nt' and 'link.exe' in str(_e) and _link_exe:
                import subprocess as _sp
                _ninja_file = os.path.join(build_dir, 'build.ninja')
                if os.path.exists(_ninja_file):
                    with open(_ninja_file, 'r') as f:
                        _ninja_content = f.read()
                    _ninja_link = _link_exe.replace('\\', '/').replace(':', '$:')
                    _ninja_content = _ninja_content.replace(
                        '"/link.exe"', f'"{_ninja_link}"')
                    with open(_ninja_file, 'w') as f:
                        f.write(_ninja_content)
                    _result = _sp.run(['ninja', '-v'], cwd=build_dir,
                                      capture_output=True, text=True)
                    if _result.returncode != 0:
                        raise RuntimeError(
                            f"Ninja retry failed:\n{_result.stdout}\n{_result.stderr}"
                        ) from _e
                    import importlib.util
                    _pyd = os.path.join(build_dir, 'vision_kernels.pyd')
                    _spec = importlib.util.spec_from_file_location('vision_kernels', _pyd)
                    _KERNEL_MODULE = importlib.util.module_from_spec(_spec)
                    _spec.loader.exec_module(_KERNEL_MODULE)
                else:
                    raise
            else:
                raise
        log.info("CUDA vision kernels compiled and loaded")
    except Exception as e:
        log.warning("CUDA vision kernels unavailable (nvcc not found?): %s", e)
        _KERNEL_MODULE = None

    return _KERNEL_MODULE
