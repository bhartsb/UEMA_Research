# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Paint the playpen walls: letters, numbers, shapes, animals, plants.

The crib had four blank translucent panels, which gives the eyes almost nothing to
predict - a wall that looks the same everywhere is perfectly predictable and teaches
nothing. These textures give each wall its own bold, high-contrast content, drawn large
because the eye cameras are low resolution (16x16-pixel patches): fine detail would be
lost, so everything is sized to survive heavy downsampling.

    north  A B C D E F G H (alphabet, primary colours)
    east   1 2 3 4 5 6 7 8 (numbers)
    south  circle, square, triangle, star, diamond, heart (shapes, distinct colours)
    west   cat, fish, bird, flower, tree, sun (simple living things)

One PNG per wall, generated once per run directory and referenced by absolute path.
"""
from __future__ import annotations

import os
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont

W, H = 2048, 342          # 6:1, matching the 1.44 m x 0.24 m wall panels
BG = (250, 248, 240)
INK = (30, 30, 35)
RED, BLUE, GREEN = (214, 64, 54), (46, 96, 190), (48, 150, 82)
YELLOW, PURPLE, ORANGE = (238, 190, 40), (140, 74, 178), (232, 130, 40)


def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _cells(n: int) -> list:
    """n evenly spaced cell centres across the panel."""
    step = W / n
    return [(int(step * (i + 0.5)), H // 2) for i in range(n)]


def _glyph_wall(chars: str, colours) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    f = _font(int(H * 0.72))
    for (cx, cy), ch, col in zip(_cells(len(chars)), chars, colours):
        box = d.textbbox((0, 0), ch, font=f)
        d.text((cx - (box[2] - box[0]) // 2, cy - (box[3] - box[1]) // 2 - box[1]), ch, font=f, fill=col)
    d.rectangle([0, 0, W - 1, H - 1], outline=INK, width=6)
    return img


def _shapes_wall() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    r = int(H * 0.34)
    pts = _cells(6)
    d.ellipse([pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r], fill=RED)
    d.rectangle([pts[1][0] - r, pts[1][1] - r, pts[1][0] + r, pts[1][1] + r], fill=BLUE)
    d.polygon([(pts[2][0], pts[2][1] - r), (pts[2][0] - r, pts[2][1] + r), (pts[2][0] + r, pts[2][1] + r)], fill=GREEN)
    # star
    import math
    cx, cy = pts[3]
    star = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        a = -math.pi / 2 + i * math.pi / 5
        star.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
    d.polygon(star, fill=YELLOW)
    cx, cy = pts[4]
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=PURPLE)
    # heart
    cx, cy = pts[5]
    d.polygon([(cx - r, cy - r * 0.45), (cx + r, cy - r * 0.45), (cx, cy + r)], fill=ORANGE)
    d.ellipse([cx - r, cy - r, cx, cy - r * 0.05], fill=ORANGE)
    d.ellipse([cx, cy - r, cx + r, cy - r * 0.05], fill=ORANGE)
    d.rectangle([0, 0, W - 1, H - 1], outline=INK, width=6)
    return img


def _living_wall() -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    r = int(H * 0.30)
    p = _cells(6)

    cx, cy = p[0]                                  # cat
    GREY = (120, 120, 128)
    d.polygon([(cx - r * 0.85, cy - r * 0.2), (cx - r * 0.62, cy - r * 1.25), (cx - r * 0.12, cy - r * 0.42)], fill=GREY)
    d.polygon([(cx + r * 0.85, cy - r * 0.2), (cx + r * 0.62, cy - r * 1.25), (cx + r * 0.12, cy - r * 0.42)], fill=GREY)
    d.ellipse([cx - r, cy - r * 0.8, cx + r, cy + r * 0.9], fill=GREY)
    for s in (-1, 1):
        d.ellipse([cx + s * r * 0.42 - r * 0.16, cy - r * 0.2, cx + s * r * 0.42 + r * 0.16, cy + r * 0.12], fill=(250, 250, 90))

    cx, cy = p[1]                                  # fish
    d.ellipse([cx - r, cy - r * 0.62, cx + r * 0.6, cy + r * 0.62], fill=ORANGE)
    d.polygon([(cx + r * 0.55, cy), (cx + r * 1.05, cy - r * 0.6), (cx + r * 1.05, cy + r * 0.6)], fill=ORANGE)
    d.ellipse([cx - r * 0.62, cy - r * 0.2, cx - r * 0.34, cy + r * 0.08], fill=(20, 20, 20))

    cx, cy = p[2]                                  # bird
    d.ellipse([cx - r * 0.75, cy - r * 0.45, cx + r * 0.45, cy + r * 0.55], fill=BLUE)
    d.ellipse([cx + r * 0.25, cy - r * 0.85, cx + r * 0.9, cy - r * 0.2], fill=BLUE)
    d.polygon([(cx + r * 0.85, cy - r * 0.6), (cx + r * 1.25, cy - r * 0.45), (cx + r * 0.85, cy - r * 0.32)], fill=YELLOW)
    d.ellipse([cx + r * 0.55, cy - r * 0.68, cx + r * 0.7, cy - r * 0.52], fill=(20, 20, 20))

    cx, cy = p[3]                                  # flower
    for i in range(6):
        import math
        a = i * math.pi / 3
        px, py = cx + r * 0.62 * math.cos(a), cy - r * 0.15 + r * 0.62 * math.sin(a)
        d.ellipse([px - r * 0.34, py - r * 0.34, px + r * 0.34, py + r * 0.34], fill=(232, 96, 150))
    d.ellipse([cx - r * 0.3, cy - r * 0.45, cx + r * 0.3, cy + r * 0.15], fill=YELLOW)
    d.line([(cx, cy + r * 0.1), (cx, cy + r * 1.15)], fill=GREEN, width=int(r * 0.18))

    cx, cy = p[4]                                  # tree
    d.rectangle([cx - r * 0.16, cy, cx + r * 0.16, cy + r * 1.1], fill=(120, 80, 45))
    d.ellipse([cx - r * 0.85, cy - r * 1.0, cx + r * 0.85, cy + r * 0.25], fill=GREEN)

    cx, cy = p[5]                                  # sun
    d.ellipse([cx - r * 0.6, cy - r * 0.6, cx + r * 0.6, cy + r * 0.6], fill=(250, 200, 40))
    for i in range(8):
        import math
        a = i * math.pi / 4
        d.line([(cx + r * 0.7 * math.cos(a), cy + r * 0.7 * math.sin(a)),
                (cx + r * 1.15 * math.cos(a), cy + r * 1.15 * math.sin(a))],
               fill=(250, 200, 40), width=int(r * 0.16))

    d.rectangle([0, 0, W - 1, H - 1], outline=INK, width=6)
    return img


def _floor_texture(seed: int = 7) -> Image.Image:
    """A patterned play mat: swirls, arcs and lines on a warm ground.

    The floor is the largest thing in the baby's view and it is what SLIDES when the baby
    moves - a blank floor shows no motion at all, so the eyes can learn nothing about
    self-movement from it. Pattern turns every roll and kick into predictable optical
    flow, which is exactly the signal the next-moment vision lesson feeds on. Pattern is
    varied but bold: fine detail would vanish at the eye's resolution.
    """
    import math
    import random as _r
    _r.seed(seed)
    S = 1024
    img = Image.new("RGB", (S, S), (236, 226, 205))
    d = ImageDraw.Draw(img)
    pal = [RED, BLUE, GREEN, YELLOW, PURPLE, ORANGE, (232, 96, 150), (60, 170, 175)]

    for _ in range(9):                                    # long sweeping lines
        x0, y0 = _r.randrange(S), _r.randrange(S)
        x1, y1 = _r.randrange(S), _r.randrange(S)
        d.line([(x0, y0), (x1, y1)], fill=_r.choice(pal), width=_r.randint(10, 22))

    for _ in range(7):                                    # spirals
        cx, cy = _r.randrange(120, S - 120), _r.randrange(120, S - 120)
        col = _r.choice(pal)
        wid = _r.randint(7, 13)
        pts, turns = [], _r.uniform(2.0, 3.5)
        steps = 220
        rmax = _r.uniform(70, 130)
        for i in range(steps):
            t = i / steps
            a = t * turns * 2 * math.pi
            rr = rmax * t
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.line(pts, fill=col, width=wid, joint="curve")

    for _ in range(10):                                   # arcs and rings
        cx, cy = _r.randrange(60, S - 60), _r.randrange(60, S - 60)
        rr = _r.randint(45, 130)
        a0 = _r.randrange(360)
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], a0, a0 + _r.randint(120, 300),
              fill=_r.choice(pal), width=_r.randint(8, 16))

    for _ in range(14):                                   # dots
        cx, cy = _r.randrange(S), _r.randrange(S)
        rr = _r.randint(12, 30)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=_r.choice(pal))
    return img


def _ceiling_texture(seed: int = 11) -> Image.Image:
    """A nursery ceiling: stars, clouds, moons, balloons.

    Discovered by looking through the baby's own eye (2026-08-23): lying on its back it
    sees almost nothing but blank sky - the wall art is out of view and the floor is a
    sliver. The ceiling is the supine infant's actual visual world, so it carries the
    boldest, largest content of any surface here.
    """
    import math, random as _r
    _r.seed(seed)
    S = 1024
    img = Image.new("RGB", (S, S), (208, 226, 240))
    d = ImageDraw.Draw(img)
    for _ in range(11):                                   # clouds
        cx, cy = _r.randrange(S), _r.randrange(S)
        for k in range(4):
            rr = _r.randint(38, 74)
            d.ellipse([cx + k * 42 - rr, cy - rr, cx + k * 42 + rr, cy + rr], fill=(252, 252, 250))
    for _ in range(16):                                   # stars
        cx, cy = _r.randrange(S), _r.randrange(S)
        rr = _r.randint(26, 52)
        pts = []
        for i in range(10):
            rad = rr if i % 2 == 0 else rr * 0.42
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + rad * math.cos(a), cy + rad * math.sin(a)))
        d.polygon(pts, fill=_r.choice([YELLOW, (255, 214, 90), (250, 190, 40)]))
    for _ in range(5):                                    # moons
        cx, cy = _r.randrange(80, S - 80), _r.randrange(80, S - 80)
        rr = _r.randint(42, 66)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(248, 236, 170))
        d.ellipse([cx - rr * 0.45, cy - rr * 1.05, cx + rr * 1.35, cy + rr * 0.85],
                  fill=(208, 226, 240))
    for _ in range(6):                                    # balloons on strings
        cx, cy = _r.randrange(60, S - 60), _r.randrange(60, S - 220)
        rr = _r.randint(30, 46)
        col = _r.choice([RED, BLUE, GREEN, PURPLE, ORANGE, (232, 96, 150)])
        d.ellipse([cx - rr, cy - rr * 1.2, cx + rr, cy + rr], fill=col)
        d.line([(cx, cy + rr), (cx + _r.randint(-20, 20), cy + rr + _r.randint(70, 150))],
               fill=(90, 90, 95), width=5)
    return img


def make_ceiling_texture(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.abspath(os.path.join(out_dir, "ceiling.png"))
    if not os.path.exists(p):
        _ceiling_texture().save(p)
    return p


def make_floor_texture(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.abspath(os.path.join(out_dir, "play_mat.png"))
    if not os.path.exists(p):
        _floor_texture().save(p)
    return p


def make_pen_textures(out_dir: str) -> dict:
    """Write the four wall textures; return {wall: absolute png path}."""
    os.makedirs(out_dir, exist_ok=True)
    walls = {
        "n": _glyph_wall("ABCDEFGH", [RED, BLUE, GREEN, PURPLE, ORANGE, RED, BLUE, GREEN]),
        "e": _glyph_wall("12345678", [BLUE, GREEN, RED, ORANGE, PURPLE, BLUE, GREEN, RED]),
        "s": _shapes_wall(),
        "w": _living_wall(),
    }
    paths = {}
    for k, img in walls.items():
        # the east and west panels sit on planes whose in-plane axes MuJoCo chooses for us,
        # and it picks the one that stands our wide image on end - so those two are rotated
        # in the source to compensate (verified by rendering, not by reasoning)
        if k == "w":
            img = img.rotate(90, expand=True)
        elif k == "e":
            img = img.rotate(-90, expand=True)   # opposite normal -> opposite in-plane spin

        p = os.path.abspath(os.path.join(out_dir, f"pen_wall_{k}.png"))
        if not os.path.exists(p):
            img.save(p)
        paths[k] = p
    return paths


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    for k, v in make_pen_textures(d).items():
        print(k, v)
