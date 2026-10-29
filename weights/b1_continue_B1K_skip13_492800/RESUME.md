# Resuming training from model step 494,025

This folder is a copy of the preserved complete end state of run `B1K_skip13_492800` at model step
494,025, the checkpoint production training is held at. `MANIFEST.json` lists every file with its
sha256; `MANIFEST_original.json` is the preserved bundle's own record.

| Path | What it is |
| --- | --- |
| `run/checkpoints/ckpt_step_494025/model.safetensors` | the model's learned numbers |
| `run/checkpoints/ckpt_step_494025/optimizer.pt` | the optimizer state |
| `run/checkpoints/ckpt_step_494025/runtime_state.pt` | trainer state (random generators, memories, counters) |
| `run/checkpoints/ckpt_step_494025/episode_state.pt.gz` | the eight worlds' physical state (episode companion) |
| `run/checkpoints/ckpt_step_494025/episode_state.json` | the companion's receipt; the loader checks its sha256 |
| `latent_body/latest.pt` | the latent-body state |
| `run/args.json`, `run/command.json` | the flags and the exact command of the run that produced this state |

The code this state was trained with is commit `641c931a6`; between it and the release's source commit
only explainer, dashboard and tool files changed, so the trainer files are the same.

To continue from it, run from `Source/Python` the command in `run/command.json` with these six values
changed (the others stay as they are):

- `--resume <this folder>/run/checkpoints/ckpt_step_494025`
- `--out <a new run folder>`
- `--checkpoint-storage <the new run folder>/checkpoints`
- `--latent-body-resume <this folder>/latent_body/latest.pt`
- `--latent-body-output <a new folder>`
- `--caregiver-record-dir <a new folder>`

and the interpreter path replaced by your own. The command also names `--max-steps`; set it to the
model step to stop at.
