# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Vision adapter: MIMo's two eye cameras -> the model's vision input.

Mode A (in-process): each eye image (H x W x 3 uint8, RGB) becomes 16x16 patches,
flattened the same way the model's tokenizer does ([N,16,16,3] -> [N,768], RGB / 255),
plus one (azimuth, elevation) angle in radians per patch. The angle is the pinhole
direction of the patch centre (MIMo eye cameras have fovy = 60 degrees, square images)
plus the eye's own yaw/pitch from the eye joints, so the model gets "where in the world
is this patch looking", like the UE panorama/fovea angles. Both eyes are concatenated
(left then right), like the tokenizer's [pano_L, pano_R, ...] order.

Mode B (shared memory): each eye is letterboxed into the LEM_Vision layout as BGRA8
(left eye -> left panorama + left fovea, right eye -> right ones), see shm_writer.py.

Rendering itself happens in env.py with mujoco.Renderer by camera NAME, because MIMo's
SimpleVision under gymnasium 1.2.3 renders the free camera for both eyes (Phase 0 bug).
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

PATCH = 16   # tokenizer/config.py TokenizerConfig.patch_size


def patch_grid_angles(width: int, height: int, fovy_deg: float, patch: int = PATCH) -> np.ndarray:
    """(rows*cols, 2) azimuth/elevation in radians of each patch centre for a pinhole camera.

    Convention: azimuth is in gaze-command space - positive = eye-LEFT (image content
    left of center carries positive azimuth), so base + eye_yaw is the world direction
    of the patch in the same units the eye is commanded in. Elevation grows bottom ->
    top (top row of the image has the largest elevation), which already matched.
    For a square 256x256 image with fovy 60: the centre patch is at (0, 0); the patch at the
    far left middle row is at az = +atan(tan(30deg) * (1 - 1/16)) ~ +28.4 deg, and the
    top-left patch's elevation is asin(y/|ray|) = 25.455 deg (not atan(y) = 28.425: the
    ray's elevation shrinks off the centre column).
    """
    rows, cols = height // patch, width // patch
    fovy = math.radians(fovy_deg)
    ty = math.tan(fovy / 2.0)
    tx = ty * (width / height)
    out = np.zeros((rows * cols, 2), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            cx = (c * patch + patch / 2.0) / width          # 0..1 left->right
            cy = (r * patch + patch / 2.0) / height         # 0..1 top->bottom
            x = (2.0 * cx - 1.0) * tx
            y = (1.0 - 2.0 * cy) * ty
            # review 2026-09-06: azimuth is emitted in GAZE-COMMAND convention
            # (positive = eye-left, the eye joint's positive direction), i.e. the
            # NEGATED image-x angle - -atan(x) is the exact spherical azimuth of the ray.
            # Elevation is the exact ray elevation asin(y/|ray|) = atan2(y, sqrt(1+x^2))
            # (audit P026, 2026-09-18). Until then it was atan(y), exact only on the
            # vertical centre column and 2.970 deg out at the wide camera's corners, 0.79
            # of one 3.750 deg patch (fovea 288 px / 24 deg: 0.216 deg, 0.16 patch); the
            # two forms agree to 6.4e-15 deg on the centre column. The gaze is COMPOSED
            # onto these directions by compose_gaze below, not added: adding the eye's
            # (yaw, pitch) was exact only with a level eye and 10.194 deg out in azimuth
            # (2.72 wide patches) at 25 deg of pitch.
            out[r * cols + c, 0] = -math.atan(x)
            out[r * cols + c, 1] = math.atan2(y, math.sqrt(1.0 + x * x))
    return out


def patch_grid_rays(width: int, height: int, fovy_deg: float, patch: int = PATCH) -> np.ndarray:
    """(rows*cols, 3) unit ray per patch centre in the eye's rest frame, columns
    (forward, left, up). The exact geometry patch_grid_angles reports as angles:
    azimuth = atan2(left, forward), elevation = asin(up)."""
    rows, cols = height // patch, width // patch
    fovy = math.radians(fovy_deg)
    ty = math.tan(fovy / 2.0)
    tx = ty * (width / height)
    out = np.zeros((rows * cols, 3), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            x = (2.0 * ((c * patch + patch / 2.0) / width) - 1.0) * tx
            y = (1.0 - 2.0 * ((r * patch + patch / 2.0) / height)) * ty
            n = math.sqrt(1.0 + x * x + y * y)
            out[r * cols + c] = (1.0 / n, -x / n, y / n)
    return out


def compose_gaze(rays: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    """(N,3) eye-rest-frame unit rays plus the eye's two gimbal joints -> (N,2) (azimuth,
    elevation) in radians, referenced to the HEAD (the neck and trunk are not folded in).
    The eye is a yaw-then-pitch gimbal: robot:*_eye_horizontal (axis 0 0 1, the head's up
    axis) is the outer joint and robot:*_eye_vertical (axis 0 -1 0, the left axis) the
    inner one, so the head-referenced direction of a patch is R_up(yaw) @ R_left(pitch) @ ray.
    Audit P026: this replaces `base + (yaw, pitch)`, which was exact only with a level eye."""
    cf, sf = math.cos(pitch), math.sin(pitch)
    f = rays[:, 0] * cf - rays[:, 2] * sf
    l = rays[:, 1]
    u = rays[:, 0] * sf + rays[:, 2] * cf
    cy, sy = math.cos(yaw), math.sin(yaw)
    out = np.empty((rays.shape[0], 2), dtype=np.float32)
    out[:, 0] = np.arctan2(f * sy + l * cy, f * cy - l * sy)
    out[:, 1] = np.arcsin(np.clip(u, -1.0, 1.0))
    return out


def image_to_patches(img: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """(H, W, 3) uint8 RGB -> (rows*cols, patch*patch*3) float32 in [0, 1].

    Same reshape/permutation as tokenizer/modalities/vision.py _extract_patches followed by
    the model's reshape(N, -1), so a patch is laid out row-major (y, x, rgb).
    """
    h, w, _ = img.shape
    rows, cols = h // patch, w // patch
    x = img[:rows * patch, :cols * patch, :].astype(np.float32) / 255.0
    x = x.reshape(rows, patch, cols, patch, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, patch * patch * 3)
    return np.ascontiguousarray(x)


class EyePatcher:
    """Precomputes the angle table once; turns two eye images into model tensors."""

    def __init__(self, width: int, height: int, fovy_deg: float = 60.0, patch: int = PATCH):
        self.width, self.height, self.patch = width, height, patch
        self.base_angles = patch_grid_angles(width, height, fovy_deg, patch)   # (N_eye, 2)
        self.base_rays = patch_grid_rays(width, height, fovy_deg, patch)       # (N_eye, 3)
        self.n_per_eye = self.base_angles.shape[0]

    def __call__(self, left: np.ndarray, right: np.ndarray,
                 left_yaw_pitch: Tuple[float, float] = (0.0, 0.0),
                 right_yaw_pitch: Tuple[float, float] = (0.0, 0.0)) -> Tuple[np.ndarray, np.ndarray]:
        """Returns patches (2N, 768) float32 and angles (2N, 2) float32 (left eye first)."""
        p = np.concatenate([image_to_patches(left, self.patch), image_to_patches(right, self.patch)], axis=0)
        a_l = compose_gaze(self.base_rays, float(left_yaw_pitch[0]), float(left_yaw_pitch[1]))
        a_r = compose_gaze(self.base_rays, float(right_yaw_pitch[0]), float(right_yaw_pitch[1]))
        return p, np.concatenate([a_l, a_r], axis=0)


class BinocularPatcher:
    """Turns one or more camera PAIRS into the model's vision input, with stereo intact.

    Each source is a pair of same-size cameras (left, right) with its own field of view:
    a wide 60-degree pair for the periphery and, when a fovea is configured, a narrow pair
    boresighted with it. A narrow camera spends the same pixels over less world, so it is
    sharper: 192 px over 16 degrees is 12 px/degree against the wide view's 4.27, which is
    a 2.8x gain in acuity for 56% more patches and 0.7 ms more render time.

    Four positional numbers per patch instead of two:
        azimuth, elevation  where in the WORLD this patch looks (gaze-referenced: the patch
                            direction plus that eye's own yaw and pitch)
        eye                 -1 left, +1 right
        span                how much of the world one patch covers, radians

    The eye number is what makes stereo REPRESENTABLE. Everything upstream of the encoder's
    pool is computed per patch, and the pool is one learned query over the whole set, so
    with only (azimuth, elevation) the pooled token comes out the same whether an object
    sits in the left eye or the right one - and disparity, the only depth cue two eyes give
    you, cannot be read at all. The span number separates a coarse peripheral patch from a
    sharp foveal one pointing the same way.

    Left and right are INTERLEAVED (L0, R0, L1, R1, ...) rather than concatenated, so the
    encoder's 7-patch attention window straddles both eyes and can compare them directly.
    Concatenated, a window only ever held one eye. Sources stay in separate blocks so a
    window compares like with like.
    """

    def __init__(self, sources, patch: int = PATCH):
        """sources: sequence of (name, size_px, fovy_deg), coarsest first."""
        self.patch = patch
        self.sources = []
        for name, size, fovy in sources:
            base = patch_grid_angles(int(size), int(size), float(fovy), patch)
            rays = patch_grid_rays(int(size), int(size), float(fovy), patch)
            span = math.radians(float(fovy)) * patch / float(size)
            self.sources.append((str(name), int(size), float(fovy), base, float(span), rays))
        self.n_per_eye = sum(s[3].shape[0] for s in self.sources)
        self.n_total = 2 * self.n_per_eye

    @property
    def names(self):
        return [s[0] for s in self.sources]

    def describe(self) -> str:
        bits = []
        for name, size, fovy, base, span, _rays in self.sources:
            bits.append(f"{name} {size}px/{fovy:g}deg = {size / fovy:.1f} px/deg, "
                        f"{base.shape[0]} patches/eye")
        return "; ".join(bits) + f" -> {self.n_total} patches"

    def __call__(self, images: dict,
                 left_yaw_pitch: Tuple[float, float] = (0.0, 0.0),
                 right_yaw_pitch: Tuple[float, float] = (0.0, 0.0)):
        """images: {source_name: (left_rgb, right_rgb)}. Returns (N, 768) and (N, 4)."""
        lyp = np.asarray(left_yaw_pitch, dtype=np.float32)
        ryp = np.asarray(right_yaw_pitch, dtype=np.float32)
        p_out, a_out = [], []
        for name, size, fovy, base, span, rays in self.sources:
            left, right = images[name]
            pl = image_to_patches(left, self.patch)
            pr = image_to_patches(right, self.patch)
            n = base.shape[0]
            if pl.shape[0] != n or pr.shape[0] != n:
                raise ValueError(f"source {name}: expected {n} patches per eye, "
                                 f"got {pl.shape[0]}/{pr.shape[0]} (image size mismatch)")
            p = np.empty((2 * n, pl.shape[1]), dtype=np.float32)
            p[0::2] = pl
            p[1::2] = pr
            a = np.empty((2 * n, 4), dtype=np.float32)
            a[0::2, :2] = compose_gaze(rays, float(lyp[0]), float(lyp[1]))
            a[1::2, :2] = compose_gaze(rays, float(ryp[0]), float(ryp[1]))
            a[0::2, 2] = -1.0
            a[1::2, 2] = +1.0
            a[:, 3] = span
            p_out.append(p)
            a_out.append(a)
        return (np.ascontiguousarray(np.concatenate(p_out, axis=0)),
                np.ascontiguousarray(np.concatenate(a_out, axis=0)))


def acuity(size_px: int, fovy_deg: float) -> dict:
    """What a square camera of this size and field of view can actually resolve.

    px_per_deg -> Nyquist in cycles/degree -> the arcmin of detail that survives -> the
    Snellen line that matches it (20/20 resolves 1 arcmin). Reference points: a newborn is
    near 20/400, six months near 20/100, an adult 20/20.
    """
    ppd = size_px / float(fovy_deg)
    cyc_per_deg = ppd / 2.0
    arcmin = 60.0 / cyc_per_deg / 2.0        # half a cycle = the finest resolvable feature
    return {"px_per_deg": ppd, "cycles_per_deg": cyc_per_deg,
            "arcmin": arcmin, "snellen": 20.0 * arcmin}


def letterbox_to_bgra(img_rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Place an RGB image centred inside an out_h x out_w BGRA8 canvas (no scaling). Black border.
    Used by Mode B to fill the panorama / fovea regions with an eye image."""
    canvas = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    canvas[..., 3] = 255
    h, w = img_rgb.shape[:2]
    h2, w2 = min(h, out_h), min(w, out_w)
    y0 = (out_h - h2) // 2
    x0 = (out_w - w2) // 2
    src = img_rgb[:h2, :w2, :]
    canvas[y0:y0 + h2, x0:x0 + w2, 0] = src[..., 2]   # B
    canvas[y0:y0 + h2, x0:x0 + w2, 1] = src[..., 1]   # G
    canvas[y0:y0 + h2, x0:x0 + w2, 2] = src[..., 0]   # R
    return canvas
