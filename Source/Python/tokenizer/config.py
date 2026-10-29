# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
LEM Tokenizer Configuration — buffer layout constants, dataclasses, config.

All constants mirror Source/UEAvatar/Pipeline/MultiModalTypes.h and
CrossProcessTransport.h exactly. Do NOT change offsets here without
updating the C++ side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING
import math

if TYPE_CHECKING:
    import torch


# =============================================================================
# BUFFER LAYOUT CONSTANTS (from MultiModalTypes.h)
# =============================================================================

# Parity ring
NUM_PARITY_SLOTS = 2

# Frame header
HEADER_OFFSET = 0
HEADER_SIZE   = 64

# Slice offsets and sizes
AUDIO_OFFSET       = 64
AUDIO_META_SIZE    = 40
AUDIO_SPECTRAL_BINS = 256   # Per-channel FFT bins
AUDIO_SPECTRAL_SIZE = AUDIO_SPECTRAL_BINS * 2 * 4  # 2048 bytes (L[256] + R[256])
AUDIO_MAX_SAMPLES  = 1526   # Raw PCM after spectral region
AUDIO_SIZE         = 8192

PROPRIO_OFFSET     = 8256
PROPRIO_MAX_BONES  = 24
PROPRIO_BONE_STRIDE = 160
PROPRIO_SIZE       = 4096

MOTOR_OFFSET       = 12352
MOTOR_MAX_HANDLES  = 24
MOTOR_HANDLE_STRIDE = 64
MOTOR_SIZE         = 2048

TOUCH_OFFSET       = 14400
TOUCH_MAX_IMPULSES = 256   # Max sensor patches (one FTMCDerivedPatch per sensor)
TOUCH_MAX_COLLISIONS = 16  # Match C++ MultiModalTypes.h
TOUCH_IMPULSE_STRIDE = 40  # 10 × float32 per FTMCDerivedPatch (8 base + 2 stretch)
TOUCH_COLLISION_STRIDE = 64
TOUCH_SIZE         = 16384  # 256×40 + 16×64 + 16 header = 11,280 used, 16 KB total

VOICE_OFFSET       = 30784  # TOUCH_OFFSET + TOUCH_SIZE
VOICE_SIZE         = 256

FRAME_BUFFER_SIZE  = 31040   # VOICE_OFFSET + VOICE_SIZE
FRAME_BUFFER_SIZE_ALIGNED = 31232  # (31040 + 255) & ~255

# --- Header field offsets (within frame header) ---
HDR_FRAME_NUMBER   = 0   # int64
HDR_TIMESTAMP      = 8   # double
HDR_MODALITY_MASK  = 16  # uint32
HDR_DELTA_TIME     = 20  # float
HDR_WALL_CLOCK     = 24  # double
HDR_GAZE_AZIMUTH   = 32  # float
HDR_GAZE_ELEVATION = 36  # float

# --- Audio field offsets (within audio slice) ---
AUD_NUM_SAMPLES    = 0
AUD_SAMPLE_RATE    = 4
AUD_NUM_CHANNELS   = 8
AUD_LEFT_RMS       = 12
AUD_RIGHT_RMS      = 16
AUD_ILD_DB         = 20
AUD_ITD_MS         = 24
AUD_AZIMUTH        = 28
AUD_FRONT_BACK     = 32
AUD_SPECTRAL_NBINS = 36    # int32: number of per-channel bins (256)
AUD_SPECTRAL       = 40    # float[512]: L[256] then R[256], normalized [0,1]
AUD_RAW_SAMPLES    = AUD_SPECTRAL + AUDIO_SPECTRAL_SIZE  # 2088

# --- Proprioception field offsets (within proprio slice) ---
PRO_NUM_BONES      = 0
PRO_BONE_DATA      = 8

# --- Bone entry field offsets (within each PROPRIO_BONE_STRIDE block) ---
BONE_NAME_HASH     = 0
BONE_ROTATION      = 4   # 4 x float (XYZW)
BONE_TRANSLATION   = 20  # 3 x float (cm)
BONE_SCALE         = 32  # float
BONE_VELOCITY      = 36  # 3 x float (cm/s)
BONE_FATIGUE       = 48  # float

# --- Motor slice field offsets ---
MOT_HANDLES        = 0
MOT_ENERGY         = MOTOR_MAX_HANDLES * MOTOR_HANDLE_STRIDE  # 1536

# --- Handle entry field offsets ---
HND_TARGET_POS     = 0   # 3 x float
HND_TARGET_ROT     = 12  # 4 x float (quat XYZW)
HND_RADIUS         = 28
HND_STRENGTH       = 32
HND_IS_ACTIVE      = 36

# --- Energy entry field offsets ---
NRG_PH_WORK        = 0
NRG_PCC_HOLDING    = 4
NRG_TOTAL          = 8

# --- Body set state offsets (within motor slice) ---
NUM_BODY_SETS = 10
BODY_SET_STATE_STRIDE = 8
MOT_BODY_SETS = MOT_ENERGY + 32  # 1568

# Body set state field offsets
BST_CAPACITY = 0
BST_ACCUMULATED_ENERGY = 4

# --- Touch slice field offsets ---
TCH_NUM_IMPULSES   = 0   # int32: number of derived patches written
TCH_NUM_COLLISIONS = 4
TCH_IMPULSE_DATA   = 16  # Start of patch array
TCH_COLLISION_DATA = TCH_IMPULSE_DATA + TOUCH_MAX_IMPULSES * TOUCH_IMPULSE_STRIDE  # 10256

# --- Derived tactile patch field offsets (within each 40-byte stride) ---
# FTMCDerivedPatch: 10 × float32, computed by ProcessDerivedFields()
IMP_NORMAL_PRESSURE  = 0   # float — Pa (GPU Neo-Hookean, max impulse)
IMP_SHEAR_MAGNITUDE  = 4   # float — Pa
IMP_SLIP_RATE        = 8   # float — mm/s
IMP_CONTACT_AREA     = 12  # float — mm² (max single-impulse spherical cap)
IMP_VIB_POWER_50HZ   = 16  # float — N²/bin (Goertzel PSD at 5 Hz; name historical)
IMP_VIB_POWER_100HZ  = 20  # float — N²/bin (Goertzel PSD at 10 Hz)
IMP_VIB_POWER_200HZ  = 24  # float — N²/bin (Goertzel PSD at 15 Hz)
IMP_VIB_POWER_400HZ  = 28  # float — N²/bin (Goertzel PSD at 20 Hz)
IMP_PRINCIPAL_STRETCH_1 = 32  # float — major axis stretch ratio (1.0 = rest)
IMP_PRINCIPAL_STRETCH_2 = 36  # float — minor axis stretch ratio (1.0 = rest)

# --- Collision entry field offsets ---
COL_BONE_HASH      = 0
COL_CONTACT_POINT  = 4
COL_CONTACT_NORMAL = 16
COL_PENETRATION_DEPTH = 28
COL_IMPACT_SPEED   = 32
COL_IMPACT_IMPULSE = 36
COL_CONTACT_DURATION = 48
COL_FRAME_COUNT    = 52

# --- Voice field offsets (within voice slice) ---
VOC_TARGET_FREQ    = 0
VOC_TARGET_TENS    = 4
VOC_TONGUE_INDEX   = 8
VOC_TONGUE_DIAMETER = 12
VOC_VIBRATO_AMOUNT = 16
VOC_VIBRATO_FREQ   = 20
VOC_INTENSITY      = 24
VOC_LOUDNESS       = 28
VOC_VELUM_TARGET   = 32
VOC_AUTO_WOBBLE    = 36
VOC_ALWAYS_VOICE   = 40
VOC_SYNTH_TIME     = 44  # double
VOC_DIAMETERS      = 52  # 44 x float
VOC_NUM_DIAMETERS  = 44

# --- Vision constants (GPU-resident, not in CPU buffer) ---
PANO_WIDTH         = 2520
PANO_HEIGHT        = 1560
FOVEA_WIDTH        = 512
FOVEA_HEIGHT       = 512
FOVEA_FOV_DEGREES  = 5.0
PANO_FOV_H_DEGREES = 210.0
PANO_FOV_V_DEGREES = 130.0

# =============================================================================
# SHARED MEMORY CONSTANTS (from CrossProcessTransport.h)
# =============================================================================

# CPU shared memory
SHM_MAGIC          = 0x4C454D50  # "LEMP"
SHM_VERSION        = 1
SHM_HEADER_SIZE    = 256
SHM_OFF_MAGIC      = 0
SHM_OFF_VERSION    = 4
SHM_OFF_LATEST_FRAME = 8
SHM_OFF_NUM_SLOTS  = 16
SHM_OFF_SLOT_SIZE  = 20
SHM_OFF_CPU_DATA_OFFSET = 24
SHM_OFF_TOTAL_SIZE = 28
SHM_OFF_MANIFEST_PATH = 32
SHM_MANIFEST_PATH_MAX = 200
SHM_CPU_DATA_OFFSET = SHM_HEADER_SIZE  # 256
SHM_TOTAL_SIZE     = SHM_HEADER_SIZE + NUM_PARITY_SLOTS * FRAME_BUFFER_SIZE_ALIGNED  # 62720

# Vision shared memory
VISION_MAGIC       = 0x4C454D56  # "LEMV"
VISION_VERSION     = 1
VISION_HEADER_SIZE = 64
VISION_OFF_MAGIC   = 0
VISION_OFF_VERSION = 4
VISION_OFF_LATEST_FRAME = 8
VISION_OFF_PANO_WIDTH = 16
VISION_OFF_PANO_HEIGHT = 20
VISION_OFF_FOVEA_WIDTH = 24
VISION_OFF_FOVEA_HEIGHT = 28
VISION_OFF_BPP     = 32
VISION_OFF_PIXEL_OFFSET = 36
VISION_OFF_WRITING = 40

VISION_PANO_BYTES  = PANO_WIDTH * PANO_HEIGHT * 4   # 15,724,800
VISION_FOVEA_BYTES = FOVEA_WIDTH * FOVEA_HEIGHT * 4  # 1,048,576
VISION_PIXEL_OFFSET = VISION_HEADER_SIZE  # 64
VISION_OFF_LEFT_PANO  = VISION_PIXEL_OFFSET
VISION_OFF_RIGHT_PANO = VISION_OFF_LEFT_PANO + VISION_PANO_BYTES
VISION_OFF_LEFT_FOVEA = VISION_OFF_RIGHT_PANO + VISION_PANO_BYTES
VISION_OFF_RIGHT_FOVEA = VISION_OFF_LEFT_FOVEA + VISION_FOVEA_BYTES
VISION_TOTAL_SIZE  = VISION_OFF_RIGHT_FOVEA + VISION_FOVEA_BYTES  # ~33.5 MB

# =============================================================================
# MODALITY BITMASK (matches EModality enum)
# =============================================================================

MOD_VISION         = 1 << 0
MOD_AUDIO          = 1 << 1
MOD_PROPRIOCEPTION = 1 << 2
MOD_MOTOR          = 1 << 3
MOD_TOUCH          = 1 << 4
MOD_VOICE          = 1 << 5


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class TokenizerConfig:
    """Configuration for the LEM tokenizer. All values overridable from pipeline_manifest.json."""

    # Pipeline connection
    cpu_shared_memory_name: str = "LEM_Pipeline"
    vision_shared_memory_name: str = "LEM_Vision"
    pipeline_manifest_path: str = "Saved/Pipeline/pipeline_manifest.json"

    # Vision
    panorama_width: int = PANO_WIDTH
    panorama_height: int = PANO_HEIGHT
    fovea_width: int = FOVEA_WIDTH
    fovea_height: int = FOVEA_HEIGHT
    patch_size: int = 16
    panorama_fov_h_deg: float = PANO_FOV_H_DEGREES
    panorama_fov_v_deg: float = PANO_FOV_V_DEGREES
    fovea_fov_deg: float = FOVEA_FOV_DEGREES

    # Audio
    fft_window_size: int = 1024
    fft_n_bins: int = 256
    fft_window_type: str = "hann"
    audio_sample_rate: int = 48000

    # Normalization ranges
    position_range_cm: float = 200.0
    velocity_range_cms: float = 500.0
    # The row index a proprioception row carries, divided by this, so it lands in 0..1
    # like every other number in the row. A FIXED constant, not len(bones)-1: measured,
    # len-1 makes row 12 read 0.5217 on a 24-row body and 0.4800 on a 26-row body,
    # which would make the same bone mean two different things on the UE and bridge
    # paths. Raw, the 0..25 ramp carried 99.16% of the 312-number row's squared
    # energy on a 26-row body; divided by 25 it carries 15.83%.
    proprio_index_range: float = 25.0
    force_range_n: float = 100.0
    energy_max: float = 1000.0

    # GPU Interop (populated from pipeline_manifest.json gpu_interop section)
    gpu_interop_available: bool = False
    adapter_luid_low: int = 0
    adapter_luid_high: int = 0
    parity_handle_names: dict = field(default_factory=dict)
    fence_handle_name: str = ""
    texture_alloc_sizes: dict = field(default_factory=dict)

    # Output
    output_manifest_path: str = "Saved/Pipeline/tokenizer_manifest.json"
    device: str = "cuda:0"


@dataclass
class FrameMetadata:
    """Per-frame metadata extracted from the pipeline frame header."""
    frame_number: int = 0
    timestamp: float = 0.0
    wall_clock: float = 0.0
    dt: float = 0.0
    gaze_azimuth_deg: float = 0.0
    gaze_elevation_deg: float = 0.0
    modality_valid_mask: int = 0


@dataclass
class TokenizedFrame:
    """Complete tokenized output for one pipeline frame. All tensors on cuda:0."""
    metadata: FrameMetadata

    # Vision (GPU-resident, None if vision SHM unavailable)
    vision_patches: Optional[torch.Tensor] = None       # [n_patches, 16, 16, 3] float32
    vision_angular_pos: Optional[torch.Tensor] = None    # [n_patches, 2] float32
    vision_eye_id: Optional[torch.Tensor] = None         # [n_patches] int32
    vision_source_id: Optional[torch.Tensor] = None      # [n_patches] int32

    # Audio (spectral pre-computed in UE C++ FFT, read from SHM)
    audio_spectral: Optional[torch.Tensor] = None        # [n_bins, 2] float32
    audio_metadata: Optional[torch.Tensor] = None        # [5] float32

    # Proprioception
    proprio: Optional[torch.Tensor] = None               # [n_bones, 12] float32

    # Motor
    motor_handles: Optional[torch.Tensor] = None         # [n_handles, 10] float32
    motor_energy: Optional[torch.Tensor] = None          # [3] float32
    motor_capability: Optional[torch.Tensor] = None      # [21] float32: 10 capacity + 10 accum + 1 total

    # Touch
    touch_impulses: Optional[torch.Tensor] = None        # [256, 10] float32 — derived tactile patches (incl. principal stretches)
    touch_collisions: Optional[torch.Tensor] = None      # [24, 8] float32 — bone collision events
    touch_valid_counts: Tuple[int, int] = (0, 0)

    # Voice
    voice: Optional[torch.Tensor] = None                 # [56] float32

    # InterMind (stub)
    intermind: Optional[torch.Tensor] = None             # [3] float32
