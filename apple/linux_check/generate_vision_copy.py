# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Write Generated/VisionPatcherLinux: VisionPatcherTests.swift with its ImageIO PNG loader
replaced by a reader of the same PNGs' RGB bytes as Python's PIL decodes them (written beside
it as .rgb files), so the WI-30 fixture test runs on Linux. Run by run.sh."""
import os
import sys

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
UEMAKIT = os.path.normpath(os.path.join(HERE, "..", "UEMAKit"))
OUT = os.path.join(HERE, "Generated", "VisionPatcherLinux")
EYES = os.path.join(UEMAKIT, "Tests", "UEMASensesTests", "Fixtures", "eyes")


def main():
    os.makedirs(os.path.join(OUT, "rgb"), exist_ok=True)
    for state in sorted(os.listdir(EYES)):
        folder = os.path.join(EYES, state)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.endswith(".png"):
                image = Image.open(os.path.join(folder, name))
                if image.mode != "RGB":
                    sys.exit(f"{state}/{name}: mode {image.mode}, the loader reads RGB")
                with open(os.path.join(OUT, "rgb", f"{state}_{name[:-4]}.rgb"), "wb") as f:
                    f.write(image.tobytes())
    src = open(os.path.join(UEMAKIT, "Tests", "UEMATokenizerTests", "VisionPatcherTests.swift"), encoding="utf-8").read()
    src = src.replace("import ImageIO\n", "")
    start = src.index("    private static func loadPNG(")
    end_marker = "        return try EyeImage(camera: camera, width: width, height: height, rgb: rgb)\n    }\n"
    end = src.index(end_marker, start) + len(end_marker)
    rgb_dir = os.path.join(OUT, "rgb").replace("\\", "/")
    loader = (
        "    private static func loadPNG(_ url: URL, camera: EyeCamera) throws -> EyeImage {\n"
        "        // Linux check: the PNG's RGB bytes as PIL decoded them (generate_vision_copy.py).\n"
        "        let state = url.deletingLastPathComponent().lastPathComponent\n"
        f"        let raw = URL(fileURLWithPath: \"{rgb_dir}/\\(state)_\\(url.deletingPathExtension().lastPathComponent).rgb\")\n"
        "        let bytes = [UInt8](try Data(contentsOf: raw))\n"
        "        return try EyeImage(camera: camera, width: camera.pixelSize, height: camera.pixelSize, rgb: bytes)\n"
        "    }\n")
    src = src[:start] + loader + src[end:]
    tests = os.path.join(UEMAKIT, "Tests", "UEMATokenizerTests").replace("\\", "/")
    src = src.replace("URL(fileURLWithPath: #filePath)\n        .deletingLastPathComponent()\n        .deletingLastPathComponent()",
                      f"URL(fileURLWithPath: \"{tests}/VisionPatcherTests.swift\")\n        .deletingLastPathComponent()\n        .deletingLastPathComponent()", 1)
    with open(os.path.join(OUT, "VisionPatcherTests.swift"), "w", encoding="utf-8", newline="\n") as f:
        f.write(src)


if __name__ == "__main__":
    main()
