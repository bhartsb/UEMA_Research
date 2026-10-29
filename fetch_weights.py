# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Download the saved training state (model step 494,025) from the private Hugging Face repository into weights/.

    python fetch_weights.py

Needs the huggingface_hub package (`pip install huggingface_hub`, pinned in requirements.txt). Access: the token in
hf_reviewer_token.txt beside this script, which can only read this one repository (Mind Machine Learning revokes it
on huggingface.co to end access); without that file, a login of your own (`hf auth login`) as a member of the
mml-user organization. Downloads the six large files of weights/b1_continue_B1K_skip13_492800 (the model's weights, the
optimizer state, the trainer's runtime state, the eight worlds' episode companion, the latent-body state and its
trajectory) into this repository at the same paths, then checks each against the SHA-256 that
weights/b1_continue_B1K_skip13_492800/MANIFEST.json lists. A file already here with the right SHA-256 is kept.
Exit code 0 when all six match.
"""
import hashlib
import json
import sys
from pathlib import Path

REPO_ID = "mml-user/uema-research-weights"
REPO = Path(__file__).resolve().parent
TOKEN_FILE = REPO / "hf_reviewer_token.txt"     # read-only, this repository only
BUNDLE = "weights/b1_continue_B1K_skip13_492800"
FILES = ["latent_body/latest.pt", "latent_body/trajectory.jsonl",
         "run/checkpoints/ckpt_step_494025/episode_state.pt.gz", "run/checkpoints/ckpt_step_494025/model.safetensors",
         "run/checkpoints/ckpt_step_494025/optimizer.pt", "run/checkpoints/ckpt_step_494025/runtime_state.pt"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("install the Hugging Face library first: pip install huggingface_hub")
        return 1
    manifest = {f["path"].replace("\\", "/"): f for f in
                json.loads((REPO / BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))["files"]}
    token = None                                  # None: the login of `hf auth login` (or HF_TOKEN)
    if TOKEN_FILE.is_file():
        lines = [line.strip() for line in TOKEN_FILE.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        token = lines[0] if lines else None
    bad = []
    for rel in FILES:
        want = manifest[rel]["sha256"]
        local = REPO / BUNDLE / rel
        if local.is_file() and sha256(local) == want:
            print("present   %s" % rel)
            continue
        print("download  %s (%.0f MB)" % (rel, manifest[rel].get("bytes", 0) / 1e6), flush=True)
        try:
            hf_hub_download(REPO_ID, "%s/%s" % (BUNDLE, rel), repo_type="model", local_dir=str(REPO), token=token)
        except Exception as exc:  # noqa: BLE001 - name the cause the reviewer can act on
            print("could not download it: %s: %s\n(%s)" % (
                type(exc).__name__, str(exc)[:300],
                "the token in hf_reviewer_token.txt was refused: it may have been revoked; ask Mind Machine Learning"
                if token else "are you a member of mml-user, and logged in with `hf auth login`?"))
            return 1
        if sha256(local) != want:
            bad.append(rel)
    if bad:
        print("SHA-256 differs from MANIFEST.json for: %s" % ", ".join(bad))
        return 1
    print("all six files present and matching MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
