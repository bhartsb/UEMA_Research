# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Download the saved training state (model step 494,025) from the Hugging Face repository into weights/.

    python fetch_weights.py --accept-license

The flag is required: the saved training state is licensed under the Mind Machine Learning Noncommercial
License 1.0 (the file LICENSE beside this script), whose terms also cover every model, set of weights or dataset
trained or generated with it (its section Trained Models and Other Outputs). Running this script with the flag
states your agreement to those terms, for yourself or for the organization you act for; without the flag it
prints this notice and downloads nothing (exit code 2).
Needs the huggingface_hub package (`pip install huggingface_hub`, pinned in requirements.txt). The repository is
public: no account and no token is needed (a saved `hf auth login` or HF_TOKEN is used when present). Downloads the six large files of weights/b1_continue_B1K_skip13_492800 (the model's weights, the
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


LICENSE_NOTICE = """The saved training state is licensed under the Mind Machine Learning Noncommercial License 1.0:
the file LICENSE beside this script. Its terms cover these weights and every model, set of weights or dataset
trained or generated, in whole or in part, with them (the section Trained Models and Other Outputs):
noncommercial purposes only, and no sale or commercial use of what is made with them.
Read LICENSE, then run:

    python fetch_weights.py --accept-license

Running it with that flag states your agreement to those terms, for yourself or for the organization you act for.
Nothing was downloaded."""


def main() -> int:
    if "--accept-license" not in sys.argv[1:]:
        print(LICENSE_NOTICE)
        return 2
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("install the Hugging Face library first: pip install huggingface_hub")
        return 1
    manifest = {f["path"].replace("\\", "/"): f for f in
                json.loads((REPO / BUNDLE / "MANIFEST.json").read_text(encoding="utf-8"))["files"]}
    token = None                                  # the repository is public; a saved login or HF_TOKEN is used when present
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
                "the repository is public and needs no login: check the network, then retry"))
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
