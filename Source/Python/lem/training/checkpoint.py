# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""
Checkpoint system — version-aware save/load with selective encoder reinitialization.

Features:
  - Format: safetensors + model_receipt snapshot
  - Save every save_every_steps, keep keep_last_n checkpoints
  - Load: compare checkpoint's receipt against current config
  - Shape mismatches trigger selective encoder reinit (not a coherence failure)
  - Progressive resize: Net2Net-style d_model scaling
  - GazeHead, VoiceHead, and GradScaler state saved/restored alongside model components
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def require_save_space(directory,state):
    """Reject obvious full-drive saves before writing large tensor histories.

    Shared tensor storage is counted once, with serialization overhead and a
    64 MiB working margin. This estimate supplements atomic publication; it
    cannot reserve space against another process writing concurrently.
    """
    pending=[state];seen=set();storages=set();size=0
    while pending:
        item=pending.pop()
        if id(item) in seen:continue
        seen.add(id(item));size+=256
        if torch.is_tensor(item):
            storage=item.untyped_storage();key=(str(item.device),storage._cdata)
            if key not in storages:storages.add(key);size+=storage.nbytes()
        elif isinstance(item,dict):pending.extend(item.values());pending.extend(item.keys())
        elif isinstance(item,(tuple,list,set)):pending.extend(item)
        elif isinstance(item,(str,bytes,bytearray)):size+=len(item)*2
        elif hasattr(item,'nbytes'):size+=int(item.nbytes)
    del pending,seen,storages
    return require_save_bytes(directory,size)


def require_save_bytes(directory,size):
    """require_save_space for a part whose size is already known. The episode companion's worlds arrive as
    pickle bytes; walking their objects instead costs about 78 bytes of memory per object (measured 2026-10-01)
    and the eight worlds held 86.5 million objects at step 447,350."""
    import errno
    import shutil
    required=int(size*1.25)+64*1024**2
    from .retention import ensure_registered_space
    ensure_registered_space(directory,required)
    free=shutil.disk_usage(directory).free
    if free<required:
        raise OSError(errno.ENOSPC,f'checkpoint storage needs approximately {required} free bytes; {free} available',str(directory))
    return dict(estimated_required_bytes=required,free_bytes=free)


def _count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _unwrap_compiled(module):
    """Unwrap torch.compile's OptimizedModule to get the original nn.Module."""
    if hasattr(module, "_orig_mod"):
        return module._orig_mod
    return module


def _save_receipt(receipt, components, optimizer, step, states=None):
    """Describe the modules actually saved, including later learning extensions.

    The construction receipt predates side modules and its old parameter sum
    even omits prediction heads. Keep its configuration/contract fields, but
    measure checkpoint architecture at publication. Buffers are not parameters;
    shared Parameter objects count once. Presence does not imply use or learning.
    """
    from copy import deepcopy
    from datetime import datetime, timezone
    saved = deepcopy(receipt)
    parameters = {}
    records = {}
    for name, module in components.items():
        module = _unwrap_compiled(module)
        named = dict(module.named_parameters(remove_duplicate=False))
        buffers = dict(module.named_buffers(remove_duplicate=False))
        local = {id(p): p for p in named.values()}
        parameters.update(local)
        tensors = {}
        recorded = module.state_dict() if states is None else states[name]
        for key, tensor in recorded.items():
            if not torch.is_tensor(tensor):
                continue
            tensors[key] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                kind=('parameter' if key in named else
                                      'buffer' if key in buffers else 'state_tensor'),
                                elements=tensor.numel())
        records[name] = dict(
            module_type=type(module).__module__ + '.' + type(module).__qualname__,
            parameters=sum(p.numel() for p in local.values()),
            parameter_tensors=len(local),
            trainable_parameters=sum(p.numel() for p in local.values() if p.requires_grad),
            saved_tensor_elements=sum(t['elements'] for t in tensors.values()),
            saved_buffer_elements=sum(t['elements'] for t in tensors.values() if t['kind']=='buffer'),
            tensors=tensors)
    optimized = {id(p): p for group in optimizer.param_groups for p in group['params']}
    saved['total_params'] = sum(p.numel() for p in parameters.values())
    saved['checkpoint_inventory'] = dict(
        schema='saved_module_inventory_v1', step=int(step),
        generated_at=datetime.now(timezone.utc).isoformat(),
        scope='Saved native modules only; runtime/physical histories and frozen actors are separate. '
              'Registered parameters include unused or frozen heads; presence is not evidence of learning.',
        unique_parameter_tensors=len(parameters),
        optimizer_parameters=sum(p.numel() for p in optimized.values()),
        optimizer_parameter_tensors=len(optimized),
        components=records)
    return saved


def _ckpt_step_key(path: Path) -> int:
    """Extract numeric step from checkpoint dir name for sorting."""
    try:
        return int(path.name.split("_")[-1])
    except (ValueError, IndexError):
        return -1


def _validate_saved_inventory(state: dict, receipt: dict, want_optimizer: bool) -> None:
    """Check a modern snapshot against its OWN save inventory before copying weights.

    This is payload integrity, not current-architecture compatibility. Legacy
    receipts without an inventory and named migrations retain their load rules.
    A component with no tensor state need not appear in the safetensors payload.
    """
    inventory = receipt.get("checkpoint_inventory")
    if inventory is None:
        return
    if not isinstance(inventory, dict) or inventory.get("schema") != "saved_module_inventory_v1":
        raise RuntimeError("checkpoint saved inventory has an unsupported schema")
    components = inventory.get("components")
    if not isinstance(components, dict):
        raise RuntimeError("checkpoint saved inventory has no component map")
    expected = {}
    for component, record in components.items():
        if not isinstance(record, dict) or not isinstance(record.get("tensors"), dict):
            raise RuntimeError(f"checkpoint saved inventory is malformed for {component}")
        for key, meta in record["tensors"].items():
            if not isinstance(meta, dict) or not isinstance(meta.get("shape"), list):
                raise RuntimeError(f"checkpoint saved inventory has no shape for {component}.{key}")
            expected[f"{component}.{key}"] = (tuple(meta["shape"]), meta.get("dtype"))
    actual = {
        f"{component}.{key}": (tuple(value.shape), str(value.dtype))
        for component, values in state.items()
        if component not in ("optimizer", "scaler", "curriculum_state") and isinstance(values, dict)
        for key, value in values.items() if torch.is_tensor(value)
    }
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    mismatched = sorted(key for key in expected.keys() & actual.keys()
                        if expected[key] != actual[key])
    if missing or unexpected or mismatched:
        raise RuntimeError(
            "checkpoint tensor payload disagrees with its saved inventory: "
            f"missing={missing}, unexpected={unexpected}, wrong_shape_or_dtype={mismatched}; "
            "refusing partial restoration")
    if state.get("step") != inventory.get("step"):
        raise RuntimeError("checkpoint step disagrees with its saved inventory")
    if want_optimizer:
        saved_optimizer = state.get("optimizer")
        if (not isinstance(saved_optimizer, dict)
                or not isinstance(saved_optimizer.get("state"), dict)
                or not isinstance(saved_optimizer.get("param_groups"), list)):
            raise RuntimeError("checkpoint saved inventory requires optimizer state for resume; "
                               "optimizer payload is missing or malformed")
        parameter_ids = []
        for group in saved_optimizer["param_groups"]:
            if (not isinstance(group, dict) or not isinstance(group.get("params"), list)
                    or any(type(index) is not int for index in group["params"])):
                raise RuntimeError("checkpoint saved inventory requires valid optimizer parameter groups")
            parameter_ids.extend(group["params"])
        saved_count = inventory.get("optimizer_parameter_tensors")
        unique_ids = set(parameter_ids)
        if len(unique_ids) != len(parameter_ids):
            raise RuntimeError("checkpoint saved inventory has duplicate optimizer parameter IDs")
        if any(type(index) is not int or index not in unique_ids for index in saved_optimizer["state"]):
            raise RuntimeError("checkpoint saved inventory has optimizer state outside its parameter groups")
        if type(saved_count) is not int or saved_count < 0 or len(unique_ids) != saved_count:
            raise RuntimeError("checkpoint optimizer parameter count disagrees with its saved inventory")


def _migrate_optimizer_state(optimizer, saved_opt: dict, id2name: dict,
                             saved_names: set):
    """Remap a saved optimizer state onto an optimizer whose parameter groups
    GREW by named new parameters (a zero-init side path such as the audio
    frequency bands). 09-09 follow-up review: without this, enabling the new
    path dropped ALL saved Adam moments (torch refuses to load across a group
    size change), so 'only the new weights start fresh' was false.

    Matching is by QUALIFIED PARAMETER NAME (module key + named_parameters
    path), never by tensor shape: a parameter is 'fresh' exactly when its name
    is absent from the saved model weights. Within each group, the saved index
    list is paired IN ORDER with the name-matched surviving parameters - valid
    because adding a module attribute never reorders the surviving parameters.
    Raises ValueError when that accounting does not work out exactly; the
    caller falls back to the old fresh-moments warning."""
    cur_sd = optimizer.state_dict()
    cur_groups = cur_sd["param_groups"]
    old_groups = saved_opt.get("param_groups") or []
    if len(cur_groups) != len(old_groups):
        raise ValueError(f"group count differs: saved {len(old_groups)} vs "
                         f"current {len(cur_groups)}")
    remap = {}
    fresh = []
    for gi, (og, cg, gobj) in enumerate(zip(old_groups, cur_groups,
                                            optimizer.param_groups)):
        old_idx = list(og["params"])
        new_idx = list(cg["params"])
        names = [id2name.get(id(p)) for p in gobj["params"]]
        if any(n is None for n in names):
            raise ValueError(f"group {gi} holds a parameter with no qualified "
                             "name - cannot migrate by name")
        survivors = [(ni, n) for ni, n in zip(new_idx, names)
                     if n in saved_names]
        fresh += [n for n in names if n not in saved_names]
        if len(survivors) != len(old_idx):
            raise ValueError(
                f"group {gi}: saved {len(old_idx)} params but "
                f"{len(survivors)} name-matched in the current optimizer")
        for oi, (ni, _n) in zip(old_idx, survivors):
            remap[oi] = ni
    saved_state = saved_opt.get("state") or {}
    new_state = {remap[oi]: sv for oi, sv in saved_state.items() if oi in remap}
    if len(new_state) != len(saved_state):
        raise ValueError(f"only {len(new_state)} of {len(saved_state)} saved "
                         "optimizer states could be remapped")
    groups = []
    for og, cg in zip(old_groups, cur_groups):
        g = dict(og)                       # saved hyperparameters (lr, betas...)
        g["params"] = list(cg["params"])   # current membership
        groups.append(g)
    return {"state": new_state, "param_groups": groups}, fresh


class CheckpointManager:
    """
    Manages checkpoint save/load with version awareness.

    Compares checkpoint's model_receipt against current resolved config
    to determine which components can be loaded vs need reinitialization.
    """

    def __init__(self, config: dict, checkpoint_dir: Optional[str] = None):
        ckpt_cfg = config["checkpoint"]
        self._format = ckpt_cfg["format"]
        self._keep_last_n = ckpt_cfg["keep_last_n"]
        self._progressive_resize = ckpt_cfg["progressive_resize"]

        self._checkpoint_dir = checkpoint_dir or ckpt_cfg.get("checkpoint_dir", "checkpoints")
        Path(self._checkpoint_dir).mkdir(parents=True, exist_ok=True)

        self._config = config

    def save(self, step: int,
             encoder_registry: nn.Module,
             forward_model: nn.Module,
             prediction_heads: nn.Module,
             motor_controller: nn.Module,
             wellness: nn.Module,
             optimizer: torch.optim.Optimizer,
             model_receipt: dict,
             gaze_head: Optional[nn.Module] = None,
             voice_head: Optional[nn.Module] = None,
             scaler: Optional[torch.amp.GradScaler] = None,
             body_schema: Optional[nn.Module] = None,
             curriculum_state: Optional[dict] = None,
             side_modules: Optional[dict] = None,
             companion_writer=None) -> str:
        """
        Save checkpoint.

        Args:
            step: Current training step.
            All model components + optimizer + receipt.
            gaze_head: Optional GazeHead module.
            voice_head: Optional VoiceHead module.
            scaler: Optional GradScaler (AMP).
            body_schema: Optional DifferentiableBodySchema (EMA buffers).

        Returns:
            Path to saved checkpoint.
        """
        # review T6: a crash mid-save used to leave a partial ckpt_step_N that
        # resumed silently wrong (missing meta.json -> step 0 and start_phase;
        # missing optimizer.pt -> fresh Adam with no log line). The directory is
        # built under a dot-prefix (invisible to the ckpt_step_* globs) and renamed
        # into place only when every file is written.
        import shutil as _shutil
        import uuid
        final_path = Path(self._checkpoint_dir) / f"ckpt_step_{step}"
        checkpoint_path = Path(self._checkpoint_dir) / f".tmp_ckpt_step_{step}_{uuid.uuid4().hex}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Unwrap torch.compile wrappers to get clean state_dict keys
        state = {
            "step": step,
            "timestamp": time.time(),
            "encoder_registry": _unwrap_compiled(encoder_registry).state_dict(),
            "forward_model": _unwrap_compiled(forward_model).state_dict(),
            "prediction_heads": _unwrap_compiled(prediction_heads).state_dict(),
            "motor_controller": _unwrap_compiled(motor_controller).state_dict(),
            "wellness": _unwrap_compiled(wellness).state_dict(),
            # SIDE MODULES. recon_decoders and td_gates are trained parameters that
            # were never handed to save(), so they were absent from every checkpoint
            # (0 of 274 tensors) and re-randomised at every resume. Measured cost:
            # the reconstruction loss restarts at 0.87-0.98 and takes 16,600 steps
            # (H31) and 64,300 steps (H28) to come back within 10% of its floor.
            **{k: _unwrap_compiled(m).state_dict()
               for k, m in (side_modules or {}).items() if m is not None},
            "optimizer": optimizer.state_dict(),
        }
        if gaze_head is not None:
            state["gaze_head"] = _unwrap_compiled(gaze_head).state_dict()
        if voice_head is not None:
            state["voice_head"] = _unwrap_compiled(voice_head).state_dict()
        if body_schema is not None:
            state["body_schema"] = _unwrap_compiled(body_schema).state_dict()
        if scaler is not None:
            state["scaler"] = scaler.state_dict()

        require_save_space(checkpoint_path,state)

        if self._format == "safetensors":
            try:
                from safetensors.torch import save_file
                # safetensors requires flat dict of tensors
                flat = {}
                for component, sd in state.items():
                    if component in ("step", "timestamp", "optimizer", "scaler"):
                        continue
                    if isinstance(sd, dict):
                        for k, v in sd.items():
                            if isinstance(v, torch.Tensor):
                                flat[f"{component}.{k}"] = v
                save_file(flat, str(checkpoint_path / "model.safetensors"))
                # Save step, timestamp, curriculum state, and non-tensor optimizer state
                meta = {"step": step, "timestamp": time.time()}
                if curriculum_state is not None:
                    meta["curriculum_state"] = curriculum_state
                with open(checkpoint_path / "meta.json", "w") as f:
                    json.dump(meta, f)
                # Save full optimizer + scaler state as pytorch (has non-tensor data)
                opt_state = {"optimizer": optimizer.state_dict()}
                if scaler is not None:
                    opt_state["scaler"] = scaler.state_dict()
                torch.save(opt_state, checkpoint_path / "optimizer.pt")
            except ImportError:
                logger.warning("safetensors not installed, falling back to pytorch format")
                torch.save(state, checkpoint_path / "checkpoint.pt")
        else:
            torch.save(state, checkpoint_path / "checkpoint.pt")

        # The early construction receipt does not include subsequently appended
        # heads, memory or self-state modules. Measure the same modules saved above.
        components = dict(encoder_registry=encoder_registry, forward_model=forward_model,
                          prediction_heads=prediction_heads, motor_controller=motor_controller,
                          wellness=wellness)
        components.update({k: m for k, m in (side_modules or {}).items() if m is not None})
        components.update({k: m for k, m in dict(gaze_head=gaze_head, voice_head=voice_head,
                                                body_schema=body_schema).items() if m is not None})
        receipt_snapshot = _save_receipt(model_receipt, components, optimizer, step, states=state)
        with open(checkpoint_path / "model_receipt.json", "w") as f:
            json.dump(receipt_snapshot, f, indent=2)

        # Runtime and physical continuation are part of the same transaction.
        # Disk-full here must leave the preceding complete checkpoint visible.
        if companion_writer is not None:
            companion_writer(checkpoint_path,final_path)

        # everything written: move into place (a same-step re-save replaces its
        # predecessor - the weights are the same step's)
        previous=None
        if final_path.exists():
            previous=final_path.with_name('.replaced_'+final_path.name+'_'+uuid.uuid4().hex)
            os.replace(str(final_path),str(previous))
        try:
            os.replace(str(checkpoint_path), str(final_path))
        except BaseException:
            if previous is not None and not final_path.exists():os.replace(str(previous),str(final_path))
            raise
        if previous is not None:
            if previous.resolve().parent!=Path(self._checkpoint_dir).resolve():raise RuntimeError('checkpoint cleanup escaped storage directory')
            from .retention import retire_replaced
            try:
                if not retire_replaced(previous,final_path):_shutil.rmtree(previous)
            except (OSError,ValueError) as error:
                # The new complete checkpoint is already published. A busy
                # retention lock must not crash continuing training; registered
                # maintenance can retire this exact backup on its next pass.
                logger.warning('Published replacement retained; old backup cleanup deferred: %s',error)
        checkpoint_path = final_path

        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Cleanup old checkpoints
        self._cleanup_old(step)

        return str(checkpoint_path)

    def load(self, checkpoint_path: str,
             encoder_registry: nn.Module,
             forward_model: nn.Module,
             prediction_heads: nn.Module,
             motor_controller: nn.Module,
             wellness: nn.Module,
             optimizer: Optional[torch.optim.Optimizer] = None,
             gaze_head: Optional[nn.Module] = None,
             voice_head: Optional[nn.Module] = None,
             body_schema: Optional[nn.Module] = None,
             map_location=None,
             side_modules: Optional[dict] = None,
             skip_encoders: Optional[Sequence[str]] = None,
             allow_retrained_gaze_head: bool = False,
             ) -> dict:
        """
        Load checkpoint with selective reinitialization on shape mismatch.

        Args:
            checkpoint_path: Path to checkpoint directory.
            All model components to load into.
            optimizer: If provided, load optimizer state.
            gaze_head: If provided, load gaze head state.
            body_schema: If provided, load body schema EMA buffers.

        Returns:
            Dict with "step", "loaded_components", "reinitialized_components",
            and optionally "scaler_state" if scaler was saved.
        """
        ckpt_dir = Path(checkpoint_path)

        # Load receipt to compare
        receipt_path = ckpt_dir / "model_receipt.json"
        if receipt_path.exists():
            with open(receipt_path) as f:
                ckpt_receipt = json.load(f)
        else:
            ckpt_receipt = {}
        # A GAZEHEAD THE EFFERENCE HEADS RETRAINED (review of b1, MINOR 5, 2026-09-23). The first
        # efference version (efference_v1, mimo_bridge LemLite) trained the vestigial GazeHead's
        # layers to predict how far the eyes TURN in 50 ms. Every lem/* caller that loads a
        # gaze_head (live_test, replay_setup, replay_train, training_process, and through them
        # inference_process, training_step, phase_rebuild, benchmark_dispatch) reads its output as
        # the ABSOLUTE direction the eyes should point, so such a checkpoint's GazeHead would hand
        # them a scaled turn as a direction. They are refused by name; the bridge model, which
        # never reads the GazeHead, passes allow_retrained_gaze_head=True.
        if (gaze_head is not None and not allow_retrained_gaze_head
                and (ckpt_receipt.get("efference_heads") or {}).get("gaze_head_repurposed")):
            raise ValueError(
                f"{checkpoint_path}: this checkpoint's GazeHead was retrained by the efference "
                "heads (efference_v1) to predict the eyes' turn over 50 ms, not the direction they "
                "should point; loading it as a gaze direction would steer the eyes by a turn. "
                "Build without the gaze head, or load it in the bridge model that trained it.")

        # Load state (map_location ensures tensors land on correct device)
        state = self._load_state(ckpt_dir, map_location=map_location, want_optimizer=optimizer is not None)
        _validate_saved_inventory(state, ckpt_receipt, want_optimizer=optimizer is not None)
        # A named encoder can be started fresh on purpose: its keys are dropped here so the
        # registry loader reports it reinitialised, exactly as it would for a shape clash.
        # Used when what an encoder learned is known to be wrong (2026-08-27: the motor
        # encoder had learned the centimetre block, see command_for_encoder).
        for _enc in (skip_encoders or ()):
            _pref = f"encoders.{_enc}."
            _reg = state.get("encoder_registry", {})
            for _k in [k for k in _reg if k.startswith(_pref)]:
                del _reg[_k]

        loaded = []
        reinitialized = []

        # Compare and load each component
        components = {
            "encoder_registry": encoder_registry,
            "forward_model": forward_model,
            "prediction_heads": prediction_heads,
            "motor_controller": motor_controller,
            "wellness": wellness,
        }

        for name, module in components.items():
            if name in state:
                try:
                    if name == "encoder_registry":
                        # Selective encoder loading
                        self._load_encoder_registry(
                            module, state[name], ckpt_receipt,
                            loaded, reinitialized,
                        )
                    else:
                        _r = module.load_state_dict(state[name], strict=False)
                        # review T15: strict=False reported "loaded" while leaving
                        # missing keys at random init (a pose-only tenure_heads next
                        # to trained encoders, silently)
                        if _r.missing_keys or _r.unexpected_keys:
                            logger.warning("%s loaded with %d missing / %d unexpected keys",
                                           name, len(_r.missing_keys), len(_r.unexpected_keys))
                        loaded.append(name)
                except (RuntimeError, KeyError) as e:
                    logger.warning(f"Cannot load {name}: {e}. Reinitializing.")
                    reinitialized.append(name)
            else:
                reinitialized.append(name)

        # SIDE MODULES, restored by the same names they were saved under. Absent from
        # older checkpoints, which is why a miss is silent rather than an error.
        for _name, _mod in (side_modules or {}).items():
            if _mod is None:
                continue
            if _name in state:
                try:
                    _r = _mod.load_state_dict(state[_name], strict=False)
                    if _r.missing_keys or _r.unexpected_keys:
                        logger.warning("%s loaded with %d missing / %d unexpected keys",
                                       _name, len(_r.missing_keys), len(_r.unexpected_keys))
                    loaded.append(_name)
                except (RuntimeError, KeyError) as e:
                    logger.warning("Cannot load %s: %s. Reinitializing.", _name, e)
                    reinitialized.append(_name)
            else:
                reinitialized.append(_name)

        # Load gaze head
        if gaze_head is not None and "gaze_head" in state:
            try:
                gaze_head.load_state_dict(state["gaze_head"], strict=False)
                loaded.append("gaze_head")
            except (RuntimeError, KeyError) as e:
                logger.warning(f"Cannot load gaze_head: {e}. Reinitializing.")
                reinitialized.append("gaze_head")
        elif gaze_head is not None:
            reinitialized.append("gaze_head")

        # Load voice head
        if voice_head is not None and "voice_head" in state:
            try:
                voice_head.load_state_dict(state["voice_head"], strict=False)
                loaded.append("voice_head")
            except (RuntimeError, KeyError) as e:
                logger.warning(f"Cannot load voice_head: {e}. Reinitializing.")
                reinitialized.append("voice_head")
        elif voice_head is not None:
            reinitialized.append("voice_head")

        # Load body schema (EMA buffers)
        if body_schema is not None and "body_schema" in state:
            try:
                body_schema.load_state_dict(state["body_schema"], strict=False)
                loaded.append("body_schema")
            except (RuntimeError, KeyError) as e:
                logger.warning(f"Cannot load body_schema: {e}. Using fresh EMA buffers.")
                reinitialized.append("body_schema")
        elif body_schema is not None:
            reinitialized.append("body_schema")

        # review T6: this else-path used to be silent - fresh Adam moments with
        # no line in the log
        if optimizer is not None and "optimizer" not in state:
            logger.warning("optimizer state ABSENT from checkpoint - Adam moments "
                           "start fresh (partial save, or saved without an optimizer)")
        # Load optimizer
        if optimizer is not None and "optimizer" in state:
            try:
                optimizer.load_state_dict(state["optimizer"])
            except (ValueError, KeyError) as e:
                # A GROWN model (named zero-init additions, e.g. the audio
                # frequency bands) changes group sizes and torch refuses the
                # load. Migrate by qualified parameter name so every saved Adam
                # moment and step counter survives and ONLY the new parameters
                # start fresh (09-09 follow-up review: this path used to drop
                # all saved states while the report claimed otherwise).
                try:
                    _mods = dict(components)
                    _mods.update({k: v for k, v in (side_modules or {}).items()
                                  if v is not None})
                    if gaze_head is not None:
                        _mods["gaze_head"] = gaze_head
                    if voice_head is not None:
                        _mods["voice_head"] = voice_head
                    id2name = {}
                    for _mk, _m in _mods.items():
                        if _m is None:
                            continue
                        for _pn, _p in _unwrap_compiled(_m).named_parameters():
                            id2name[id(_p)] = f"{_mk}.{_pn}"
                    saved_names = {f"{_mk}.{_k}" for _mk in _mods
                                   for _k in (state.get(_mk) or {})}
                    _sd, _fresh = _migrate_optimizer_state(
                        optimizer, state["optimizer"], id2name, saved_names)
                    optimizer.load_state_dict(_sd)
                    logger.warning(
                        "optimizer state MIGRATED across a parameter-set change: "
                        "%d saved states preserved (step counters included); "
                        "fresh (zero-history) parameters: %s",
                        len(_sd["state"]), ", ".join(_fresh) or "(none)")
                except (ValueError, KeyError) as e2:
                    # Say WHICH failure this is. Checkpoints written before the
                    # no-decay split hold one parameter group; the optimizer now
                    # has two, and torch refuses to load across that. Adam
                    # moments and the bias-correction counter are lost, which is
                    # survivable but must not look like a corrupt file.
                    n_have = len(optimizer.state_dict().get("param_groups", []))
                    n_want = len((state.get("optimizer") or {}).get("param_groups", []))
                    if n_have != n_want:
                        logger.warning(
                            "optimizer state NOT restored: checkpoint has %d parameter "
                            "group(s), this optimizer has %d (migration also failed: %s). "
                            "Pre-dates the weight-decay split; Adam moments start fresh. "
                            "Training continues.", n_want, n_have, e2)
                    else:
                        logger.warning("Cannot load optimizer state: %s "
                                       "(name migration also failed: %s)", e, e2)

        step = state.get("step", 0)
        timestamp = state.get("timestamp", None)
        ts_str = f", saved at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}" if timestamp else ""
        logger.info(
            f"Loaded checkpoint step {step}{ts_str}. "
            f"Loaded: {loaded}, Reinitialized: {reinitialized}"
        )

        result = {
            "step": step,
            "timestamp": timestamp,
            "loaded_components": loaded,
            "reinitialized_components": reinitialized,
        }
        # Pass scaler state through if present (caller loads it)
        if "scaler" in state:
            result["scaler_state"] = state["scaler"]
        # Pass curriculum state through if present (caller loads it)
        if "curriculum_state" in state:
            result["curriculum_state"] = state["curriculum_state"]
        return result

    def _load_encoder_registry(self, registry: nn.Module,
                                ckpt_state: dict, ckpt_receipt: dict,
                                loaded: list, reinitialized: list) -> None:
        """Load encoder registry with selective reinit on shape mismatch."""
        current_encoders = set(registry.encoders.keys())
        ckpt_encoders = set()
        for key in ckpt_state:
            # Keys are like "encoders.vision.patch_embed.weight"
            parts = key.split(".")
            if len(parts) >= 2 and parts[0] == "encoders":
                ckpt_encoders.add(parts[1])

        # Load matching encoders
        for enc_name in current_encoders:
            if enc_name in ckpt_encoders:
                # Check if d_model matches
                ckpt_d = ckpt_receipt.get("encoders_built", {}).get(enc_name, {}).get("d_output")
                current_d = self._config["model"]["d_model"]

                if ckpt_d is not None and ckpt_d != current_d and self._progressive_resize:
                    logger.info(f"Encoder '{enc_name}': d_model changed {ckpt_d}->{current_d}, "
                                f"attempting progressive resize")
                    reinitialized.append(f"encoder:{enc_name}")
                else:
                    # Try loading
                    prefix = f"encoders.{enc_name}."
                    enc_state = {
                        k[len(prefix):]: v for k, v in ckpt_state.items()
                        if k.startswith(prefix)
                    }
                    try:
                        registry.encoders[enc_name].load_state_dict(enc_state, strict=False)
                        loaded.append(f"encoder:{enc_name}")
                    except RuntimeError as e:
                        logger.warning(f"Shape mismatch for encoder '{enc_name}': {e}")
                        reinitialized.append(f"encoder:{enc_name}")
            else:
                reinitialized.append(f"encoder:{enc_name}")

        # THE REGISTRY IS MORE THAN ITS ENCODERS. Its state_dict also carries the shared
        # OUTPUT LAYERNORM that every token passes through - output_norm.weight and
        # output_norm.bias. The loop above only ever matched keys under "encoders.", so
        # those two were written to every checkpoint and silently dropped on every load:
        # each resume restarted the shared normaliser from a fresh LayerNorm while the
        # encoders around it stayed adapted to the old one.
        #
        # Measured on disk rather than argued: H26_tc resumed from a parent whose
        # ||output_norm.bias|| was 1.5042 and its own first checkpoint, 983 steps later,
        # reads 0.0997 - 6.6% of the parent. AdamW decoupled decay at lr 3e-4, wd 0.01 is
        # a factor 0.999997 per step, so 983 steps of pure decay with no gradient at all
        # could only reach 1.4998. The observed value is 15x below that floor, so it is a
        # reset and not decay. A control tensor in the same files,
        # encoders.vision.patch_embed.weight, round-trips parent to child at cosine
        # 0.959-0.999, so this was not a general load failure - only these two tensors.
        extra = {k: v for k, v in ckpt_state.items() if not k.startswith("encoders.")}
        if extra:
            try:
                missing, unexpected = registry.load_state_dict(
                    {**{f"encoders.{n}.{k}": v                        for n in current_encoders                        for k, v in registry.encoders[n].state_dict().items()},
                     **extra}, strict=False)
                loaded.append("encoder_registry:shared(" + ",".join(sorted(extra)) + ")")
            except RuntimeError as e:
                logger.warning("shared registry state failed to load: %s", e)
                reinitialized.append("encoder_registry:shared")

    def _load_state(self, ckpt_dir: Path, map_location=None,
                    want_optimizer: bool = True) -> dict:
        """Load state from checkpoint directory.

        Args:
            map_location: Device to map tensors to (e.g. 'cuda:1' for multi-GPU).
                          Passed to torch.load() and safetensors load_file().
        """
        # Try safetensors first
        sf_path = ckpt_dir / "model.safetensors"
        if sf_path.exists():
            try:
                from safetensors.torch import load_file
                device_str = str(map_location) if map_location else "cpu"
                flat = load_file(str(sf_path), device=device_str)
                # Unflatten
                state = {}
                for key, tensor in flat.items():
                    parts = key.split(".", 1)
                    component = parts[0]
                    param_key = parts[1] if len(parts) > 1 else key
                    if component not in state:
                        state[component] = {}
                    state[component][param_key] = tensor

                # Load meta. review T6: absence used to mean "step 0, start
                # phase" silently - the signature of a crash mid-save, not a
                # loadable checkpoint.
                meta_path = ckpt_dir / "meta.json"
                if not meta_path.exists():
                    raise RuntimeError(
                        f"{ckpt_dir} has model.safetensors but no meta.json - a "
                        f"partial save (crash or disk-full mid-write). Resume from "
                        f"the previous checkpoint instead.")
                with open(meta_path) as f:
                    meta = json.load(f)
                state["step"] = meta.get("step", 0)
                if "timestamp" in meta:
                    state["timestamp"] = meta["timestamp"]
                if "curriculum_state" in meta:
                    state["curriculum_state"] = meta["curriculum_state"]

                # Load optimizer + scaler (review T16: skipped when the caller
                # passes optimizer=None - tens of MB it would discard)
                opt_path = ckpt_dir / "optimizer.pt"
                if want_optimizer and opt_path.exists():
                    opt_state = torch.load(opt_path, weights_only=False,
                                           map_location=map_location)
                    state["optimizer"] = opt_state.get("optimizer", {})
                    if "scaler" in opt_state:
                        state["scaler"] = opt_state["scaler"]

                return state
            except ImportError:
                pass

        # Fall back to pytorch
        pt_path = ckpt_dir / "checkpoint.pt"
        if pt_path.exists():
            return torch.load(pt_path, weights_only=False,
                              map_location=map_location)

        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    def _cleanup_old(self, current_step: int) -> None:
        """Remove old checkpoints, keeping only keep_last_n."""
        from .retention import find_policy,policy,owned_save_directory,ensure_registered_space
        ckpt_dir = Path(self._checkpoint_dir)
        registered=find_policy(ckpt_dir)
        if registered is not None:
            try:
                if owned_save_directory(policy(registered),ckpt_dir):
                    ensure_registered_space(ckpt_dir)
                    return
            except (OSError,ValueError) as error:
                logger.warning('Published checkpoint retained; registered cleanup deferred: %s',error)
                return
        checkpoints = sorted(ckpt_dir.glob("ckpt_step_*"), key=_ckpt_step_key)

        if len(checkpoints) > self._keep_last_n:
            to_remove = checkpoints[:len(checkpoints) - self._keep_last_n]
            for old_ckpt in to_remove:
                import shutil
                shutil.rmtree(old_ckpt, ignore_errors=True)
                logger.info(f"Removed old checkpoint: {old_ckpt}")

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        ckpt_dir = Path(self._checkpoint_dir)
        results = []
        for d in sorted(ckpt_dir.glob("ckpt_step_*"), key=_ckpt_step_key):
            step_str = d.name.split("_")[-1]
            try:
                step = int(step_str)
            except ValueError:
                continue

            receipt_path = d / "model_receipt.json"
            receipt = None
            if receipt_path.exists():
                with open(receipt_path) as f:
                    receipt = json.load(f)

            timestamp = None
            meta_path = d / "meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                timestamp = meta.get("timestamp")

            results.append({
                "path": str(d),
                "step": step,
                "timestamp": timestamp,
                "receipt": receipt,
            })

        return results
