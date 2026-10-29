# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Resource-aware explainer refresh. Imports no model or numerical libraries."""
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

LOG = logging.getLogger("mimo_bridge.explainer.refresh")
TOOLS = Path(__file__).resolve().parents[1] / "tools"


@contextmanager
def output_lock(path):
    """One refresher per output; the operating system releases a crashed owner."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another explainer refresher holds " + str(path)) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _resolved(path):
    """One spelling for a path: absolute, links and ".." resolved, letter case and separators normalised."""
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def output_inside_run(output, run):
    """True when the output path is the run folder or anywhere inside it, under any spelling of either (relative or
    absolute, "..", letter case, a link, the \\\\?\\ extended-length prefix, an administrative or loopback share such as
    \\\\localhost\\C$), and True when that cannot be told. --no-run-copy refuses such an output (the
    output-destination decision of Oct 15, 2026, arm1-failure-reconciled-output-fence-20261015: the input run is only
    read).

    Two tests. The string test resolves each path to one spelling and compares the two. The identity test then
    compares the file identity (os.path.samefile: the volume and the file index) of the output and of each of its
    existing parent folders with the run folder's, which catches the spellings realpath does not rewrite (the
    independent recheck of Oct 15, 2026, finding A-M1: build() wrote into a fixture run under the \\\\?\\ and the
    \\\\localhost\\C$ spellings). An identity that cannot be read refuses."""
    out, folder = _resolved(output), _resolved(run)
    try:
        if os.path.commonpath([out, folder]) == folder:
            return True
    except ValueError:                       # another drive, or a \\?\ or UNC spelling of either path
        pass
    try:
        target = Path(os.path.abspath(str(output)))
        return any(p.exists() and os.path.samefile(p, run) for p in (target, *target.parents))
    except OSError:
        return True                          # identity cannot be read: refuse


def inputs_holding(output, run, stats_from=None):
    """The input run folders (the --run folder and, when given, the --stats-from folder) that hold the output, under
    output_inside_run's rule; [] when the output is outside both (the independent recheck's finding A-m2: the chart
    run of --stats-from is an input the fence covers too)."""
    return [r for r in (run, stats_from) if r and output_inside_run(output, r)]


def destinations(args, run, target):
    """The files one rebuild writes: the run's own copy unless --no-run-copy, then the requested output."""
    own = [] if getattr(args, "no_run_copy", False) else [str(Path(run) / "explainer.html")]
    return own + [target]


def build_commands(args, run, target):
    guard = [sys.executable, "-B", str(TOOLS / "commit_guard.py"),
             "--need-gb", str(args.need_gb), "--json"]
    child = [sys.executable, "-B", str(TOOLS / "build_explainer.py"),
             "--run", str(run), "--out", target, "--max-snapshots", str(args.max_snapshots),
             "--every", str(args.every), "--watch-child"]
    if args.stats_from:
        child.extend(["--stats-from", args.stats_from])
    # the fresh child is the process that writes: the option must reach it, or the run copy is written anyway
    if getattr(args, "no_run_copy", False):
        child.append("--no-run-copy")
    return guard, child


def write_receipt(path, record):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def refresh_loop(args, choose_run, source_stamp, target, *, run_command=subprocess.run,
                 sleep=time.sleep, max_cycles=None):
    """Retry failed input unchanged; acknowledge an input only after child success.

    choose_run is invoked on every cycle so an unpinned watcher follows production
    resumes. Each build imports current explanation source in a fresh process.
    """
    previous = None
    receipt = {"pid": os.getpid(), "started_at": time.time(), "output": target,
               "run_copy": not getattr(args, "no_run_copy", False),
               "interval_seconds": max(2.0, args.every), "last_success_at": None}
    cycles = 0
    while max_cycles is None or cycles < max_cycles:
        cycles += 1
        receipt.update(checked_at=time.time(), error=None)
        fenced = False          # an output inside the input run: nothing is written, not even this receipt
        try:
            run = choose_run()
            receipt["run"] = run
            if not run or not (Path(run) / "args.json").is_file():
                raise RuntimeError("No selected run with saved settings; retaining previous page")
            receipt["destinations"] = destinations(args, run, target)
            held = inputs_holding(target, run, args.stats_from) if getattr(args, "no_run_copy", False) else []
            if held:
                fenced = True
                raise RuntimeError("Output %s is inside the input run %s; --no-run-copy keeps the input runs "
                                   "read-only, so nothing is built" % (target, held[0]))
            signature = (os.path.normcase(os.path.abspath(run)), source_stamp(run, args.stats_from))
            if signature == previous:
                receipt["state"] = "unchanged"
            else:
                guard, child = build_commands(args, run, target)
                admission = run_command(guard, capture_output=True, text=True, timeout=60)
                receipt["guard"] = admission.stdout.strip()
                if admission.returncode:
                    receipt["state"] = "waiting_for_memory_or_disk"
                    LOG.warning("Explainer build waits: %s", admission.stdout.strip() or admission.stderr.strip())
                else:
                    receipt["state"] = "building"
                    write_receipt(target + ".refresh.json", receipt)
                    result = run_command(child, timeout=180)
                    if result.returncode:
                        raise RuntimeError("Fresh builder exited " + str(result.returncode))
                    # An input changed during the build triggers another build next
                    # cycle. Never mark that newer, unread input as acknowledged.
                    previous = signature
                    receipt.update(state="built", last_success_at=time.time())
        except Exception as exc:
            receipt.update(state="retry_pending", error=str(exc))
            LOG.exception("Explainer refresh failed; retrying next cycle")
        if not fenced:
            write_receipt(target + ".refresh.json", receipt)
        if max_cycles is None or cycles < max_cycles:
            sleep(max(2.0, args.every))
