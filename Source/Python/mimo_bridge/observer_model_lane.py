# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The developmental observer's model lane: what the learner predicted, per step, for the observer.

Addendum v4 sections 24 and 34 ("intent/prediction join"): after an external episode is found,
the observer attaches the prediction, candidate-action and wellness-selection records that
preceded it. The causal-chain record (<run>/causal_chain.jsonl, --causal-chain on) already carries
every latch's plan, its candidates and its predicted wellness. This lane adds, per model step,
what the causal-chain file does not:

  author / latch / order / rest / bout   who wrote the order in force (the same provenance the
                                          world recorder stamps on each world record)
  next_pe[S]                              the next-moment prediction error per sense, the batch
                                          mean the lesson reads (LemLite.last_next_pe)
  next_pe_body[B, S]                      the same error per body (LemLite.observer_moment): one
                                          body's 50 ms failure must not vanish into the mean
  wellness[C, B]                          each grounded wellness channel per body this moment
                                          (LemLite.last_wellness_grounded), the internal evidence
                                          shown under a behaviour, never in place of it
  contact_q[B, P, 3]                      the contact head's probabilities for THIS moment
                                          (formed from the moment before and the order in
                                          force), bits any/self/world, and each patch's base rate
                                          of starting a touch [3, P] (LemLite.observer_contact)
  efference tenure (every order's end)    each handle's expected movement (cm) and turn (degrees)
                                          formed where the order began, what the body achieved,
                                          and the same expectation with another body's order
                                          (LemLite.last_efference_tenure)
  plan licence                            the planner's per-depth win rates when they change

Written by the trainer's process into <run>/observer/model/ as compressed chunks; the learner
never reads them. Only runs under --observer record (train_mimo.py). Holds to the same disk
budget as the world recorder (observer_record.py): nothing is written while the disk is below its
reserve or the folder is at its budget, and every dropped step is listed in gaps.jsonl.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional

import numpy as np

from .observer_record import AUTHOR_CODES, DEFAULT_RESERVE_BYTES, _atomic_json, _lp

OBSERVER_MODEL_VERSION = "observer_model_v1"
SENSES = ("vision", "tactile", "audio", "proprioception", "vestibular")
DEFAULT_MODEL_BUDGET_BYTES = 2 * 2 ** 30


def _np(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().float().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _records_failures(lane) -> dict:
    """The comparison records' failures counted on this lane (comparison_records.failures_of), {} when unreadable."""
    try:
        from .comparison_records import failures_of
        return failures_of(lane)
    except Exception:
        return {}


def read_model_chunk(path: str) -> dict:
    """One model-lane chunk back as arrays (float16 columns widened to float32)."""
    with np.load(path, allow_pickle=False) as z:
        out = {k: np.asarray(z[k]) for k in z.files}
    for k, v in list(out.items()):
        if v.dtype == np.float16:
            out[k] = v.astype(np.float32)
    return out


class ModelLane:
    def __init__(self, out_dir: str, run: str, n_envs: int, handle_names, chunk_steps: int = 100,
                 budget_bytes: int = DEFAULT_MODEL_BUDGET_BYTES,
                 reserve_bytes: int = DEFAULT_RESERVE_BYTES):
        self.dir = out_dir
        os.makedirs(self.dir, exist_ok=True)
        self.run = str(run)
        self.n_envs = int(n_envs)
        self.chunk_steps = max(1, int(chunk_steps))
        self.budget_bytes = int(budget_bytes)
        self.reserve_bytes = int(reserve_bytes)
        self.pid = os.getpid()
        self._rows = []
        self._eff = []
        self._lic = []
        self._seq = 0
        self._last_eff_step = None
        self._last_lic = None
        self.enabled = True
        self.error = None
        self.records = 0
        self.dropped = 0
        self.written_bytes = 0
        self.cost_s = 0.0
        from .comparison_records import DEFINITIONS as _CRD, RECORDS_VERSION as _CRV
        header = dict(version=OBSERVER_MODEL_VERSION, run=self.run, n_envs=self.n_envs,
                      senses=list(SENSES), handle_names=list(handle_names), author_codes=AUTHOR_CODES,
                      contact_bits=["any", "self", "world"], created=time.time(), pid=self.pid,
                      budget_bytes=self.budget_bytes, reserve_bytes=self.reserve_bytes,
                      meaning="what the learner predicted; read by the developmental observer only",
                      # the columns the comparison records add (mimo_bridge/comparison_records.py, records a and b)
                      comparison_records=dict(version=_CRV, persist_pe_body=_CRD["lane_persist_pe_body"],
                                              vision_no_copy_body=_CRD["lane_vision_no_copy_body"],
                                              gaze_copy_applied="1 the gaze copy added its term, 0 it added "
                                                                "nothing, -1 not read"))
        # named like the world recorder's headers (pid, then milliseconds), so a reader orders
        # the trainer's relaunches by time, not by process id
        _atomic_json(os.path.join(self.dir, "header_%d_%d.json" % (self.pid, int(header["created"] * 1000))),
                     header)

    def add(self, step: int, model, *, author: str, latched: bool, order_seq: int, rest: bool,
            bout: bool) -> None:
        if not self.enabled:
            return
        t0 = time.perf_counter()
        try:
            self._add(step, model, author=author, latched=latched, order_seq=order_seq, rest=rest,
                      bout=bout)
        except Exception as ex:               # a report must never stop the learner
            self.enabled = False
            self.error = repr(ex)
            try:
                _atomic_json(os.path.join(self.dir, "lane_error_%d.json" % self.pid),
                             dict(time=time.time(), step=int(step), error=repr(ex)))
            except Exception:
                pass
        self.cost_s += time.perf_counter() - t0

    def _add(self, step, model, *, author, latched, order_seq, rest, bout):
        B = self.n_envs
        pe = getattr(model, "last_next_pe", {}) or {}
        next_pe = np.asarray([float(pe.get(s, np.nan)) for s in SENSES], dtype=np.float32)
        per = np.full((B, len(SENSES)), np.nan, dtype=np.float32)
        # the comparison records (mimo_bridge/comparison_records.py): record a, the persistence error of the same
        # transitions; record b, vision's error without the gaze copy's term and whether the copy added anything
        pers = np.full((B, len(SENSES)), np.nan, dtype=np.float32)
        no_copy = np.full(B, np.nan, dtype=np.float32)
        applied = -1
        mom = getattr(model, "observer_moment", None)
        if mom is not None and int(mom.get("step", -1)) == int(step):
            arr = np.asarray(mom["next_pe_per_body"], dtype=np.float32)
            for i, s in enumerate(mom["senses"]):
                if s in SENSES and arr.shape[1] == B:
                    per[:, SENSES.index(s)] = arr[i]
            # the comparison records' columns, behind their own guard: a failure leaves them NaN / -1 and is counted
            # (comparison_records.record_failed, read into this lane's status file); it never turns the lane off
            try:
                parr = mom.get("persistence_per_body")
                parr = None if parr is None else np.asarray(parr, dtype=np.float32)
                if parr is not None and parr.shape == arr.shape and arr.shape[1] == B:
                    for i, s in enumerate(mom["senses"]):
                        if s in SENSES:
                            pers[:, SENSES.index(s)] = parr[i]
                nc = mom.get("vision_without_copy_per_body")
                if nc is not None and np.asarray(nc).size == B:
                    no_copy[:] = np.asarray(nc, dtype=np.float32).reshape(B)
                if mom.get("gaze_copy_applied") is not None:
                    applied = int(bool(mom["gaze_copy_applied"]))
            except Exception as ex:
                pers[:] = np.nan
                no_copy[:] = np.nan
                applied = -1
                from .comparison_records import record_failed
                record_failed(self, "lane_columns", ex)
        well = {}
        for k, v in (getattr(model, "last_wellness_grounded", None) or {}).items():
            a = _np(v)
            if a is not None and a.size == B:
                well[str(k)] = a.reshape(B).astype(np.float32)
        cq = getattr(model, "observer_contact", None)
        if cq is not None and int(cq.get("step", -1)) == int(step):
            q = np.asarray(cq["q"], dtype=np.float16)
            base = np.asarray(cq["base_rate_start"], dtype=np.float16)
            valid = np.asarray(cq["valid"], dtype=bool)
        else:
            q = base = valid = None
        self._rows.append(dict(step=int(step), author=int(AUTHOR_CODES.get(str(author or "unknown"), 0)),
                               latched=bool(latched), order_seq=int(order_seq), rest=bool(rest),
                               bout=bool(bout), next_pe=next_pe, per=per, well=well,
                               q=q, base=base, valid=valid, pers=pers, no_copy=no_copy, applied=applied))
        self.records += 1
        rec = getattr(model, "last_efference_tenure", None)
        if rec is not None and rec.get("step") != self._last_eff_step:
            self._last_eff_step = rec.get("step")

            def f16(key):
                v = _np(rec.get(key))
                return None if v is None else v.astype(np.float16)
            self._eff.append(dict(step=int(rec.get("step", -1)), began=int(rec.get("order_began", -1)),
                                  exp_cm=f16("expected_cm"), exp_deg=f16("expected_deg"),
                                  ach_cm=f16("achieved_cm"), ach_deg=f16("achieved_deg"),
                                  sh_cm=f16("expected_shuffled_cm"), sh_deg=f16("expected_shuffled_deg")))
        lic = getattr(model, "last_plan_license", None)
        if lic:
            key = json.dumps(lic, sort_keys=True, default=str)
            if key != self._last_lic:
                self._last_lic = key
                self._lic.append(dict(step=int(step), license=lic))
        if len(self._rows) >= self.chunk_steps:
            self.flush()

    def _refusal(self, need: int) -> Optional[str]:
        try:
            free = shutil.disk_usage(self.dir).free
        except OSError:
            free = None
        if free is not None and free - need < self.reserve_bytes:
            return ("disk has %.1f GiB free; the observer never takes the last %.1f GiB"
                    % (free / 2 ** 30, self.reserve_bytes / 2 ** 30))
        used = 0
        try:
            with os.scandir(self.dir) as it:
                for e in it:
                    if e.is_file():
                        used += e.stat().st_size
        except OSError:
            pass
        if used + need > self.budget_bytes:
            return ("the model lane's folder holds %.2f GiB of its %.2f GiB budget; the analysis "
                    "has not digested it" % (used / 2 ** 30, self.budget_bytes / 2 ** 30))
        return None

    def flush(self) -> Optional[str]:
        if not self._rows:
            return None
        rows, eff, lic = self._rows, self._eff, self._lic
        self._rows, self._eff, self._lic = [], [], []
        refusal = self._refusal(16384 * len(rows))
        if refusal:
            self.dropped += len(rows)
            try:
                with open(os.path.join(self.dir, "gaps.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(time=time.time(), pid=self.pid, first_step=rows[0]["step"],
                                            last_step=rows[-1]["step"], rows=len(rows),
                                            reason=refusal)) + "\n")
            except OSError:
                pass
            self._status(refusal)
            return None
        a = {k: np.asarray([r[k] for r in rows]) for k in ("step", "author", "latched", "order_seq", "rest", "bout")}
        a["next_pe"] = np.stack([r["next_pe"] for r in rows])
        a["next_pe_body"] = np.stack([r["per"] for r in rows])
        # the comparison records' columns (comparison_records.DEFINITIONS lane_*): [rows, bodies, senses] persistence
        # error, [rows, bodies] vision's error without the gaze copy, [rows] whether the copy added anything (-1:
        # not read); NaN where not graded
        a["persist_pe_body"] = np.stack([r["pers"] for r in rows])
        a["vision_no_copy_body"] = np.stack([r["no_copy"] for r in rows])
        a["gaze_copy_applied"] = np.asarray([r["applied"] for r in rows], dtype=np.int8)
        names = sorted({k for r in rows for k in r["well"]})
        if names:
            B = self.n_envs
            w = np.full((len(rows), len(names), B), np.nan, dtype=np.float32)
            for i, r in enumerate(rows):
                for j, k in enumerate(names):
                    if k in r["well"]:
                        w[i, j] = r["well"][k]
            a["wellness_names"] = np.asarray(names)
            a["wellness"] = w
        have_q = [r for r in rows if r["q"] is not None]
        if have_q:
            a["contact_step"] = np.asarray([r["step"] for r in have_q], dtype=np.int64)
            a["contact_q"] = np.stack([r["q"] for r in have_q])
            a["contact_base"] = np.stack([r["base"] for r in have_q])
            a["contact_valid"] = np.stack([r["valid"] for r in have_q])
        good = [e for e in eff if e["exp_cm"] is not None and e["ach_cm"] is not None]
        if good:
            a["eff_step"] = np.asarray([e["step"] for e in good], dtype=np.int64)
            a["eff_began"] = np.asarray([e["began"] for e in good], dtype=np.int64)
            for key in ("exp_cm", "exp_deg", "ach_cm", "ach_deg"):
                a["eff_" + key] = np.stack([e[key] for e in good])
            if all(e["sh_cm"] is not None for e in good):
                a["eff_sh_cm"] = np.stack([e["sh_cm"] for e in good])
                a["eff_sh_deg"] = np.stack([e["sh_deg"] for e in good])
        if lic:
            a["license_json"] = np.asarray([json.dumps(x, default=str) for x in lic])
        name = "chunk_%09d_%d_%05d.npz" % (int(a["step"][0]), self.pid, self._seq)
        self._seq += 1
        path = _lp(os.path.join(self.dir, name))
        tmp = path[:-4] + ".part.npz"
        np.savez_compressed(tmp, **a)
        os.replace(tmp, path)
        try:
            self.written_bytes += os.path.getsize(path)
        except OSError:
            pass
        self._status(None)
        return path

    def _status(self, refusal: Optional[str]) -> None:
        try:
            _atomic_json(os.path.join(self.dir, "status_%d.json" % self.pid), dict(
                time=time.time(), run=self.run, pid=self.pid, records=self.records,
                dropped=self.dropped, written_bytes=self.written_bytes,
                ms_per_record=(1000.0 * self.cost_s / self.records) if self.records else None,
                enabled=self.enabled, error=self.error, refusal=refusal,
                comparison_records_failures=_records_failures(self)))
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.flush()
        finally:
            self.enabled = False
            self._status(None)
