# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""THE EXPERIMENT MIRROR (Oct 13, 2026): copies the display files of a live experiment training run that trains outside
the production supervisor (an owner such as D:/UEMA_Research/external/D/UEMA_scratch/b1_recover_489625/launch_recover.py) into a folder inside
the project, so the development dashboard can show that run while its file serving stays confined to the project
(development_dashboard.confined is unchanged; nothing outside the project is served).

THE POINTER, Saved/mimo_bridge/experiment_live.json, names the live run. The run's owner or the coordinator writes it
(`point`):
    {"schema": 1, "run": <run folder>, "owner_state": <the owner's owner_state.json>, "owner": <the owner's script>,
     "started_at": <the owner's start, AM/PM PDT>, "started_s": <its epoch seconds>, "by": <writer>,
     "written_at": <AM/PM PDT>}

THE MIRROR, Saved/mimo_bridge/experiment_mirror/<run folder name>/, holds copies of the run's
  args.json and checkpoints.json (each read as complete JSON);
  replay_training_stats.jsonl: the run file's bytes from its start to its last complete line (each pass appends the
    lines the run added since; the run's own modification time of this file is not used: measured on Oct 13, 2026,
    os.stat gave 2:11:09 PM PDT while the file held records up to 3:30 PM PDT, its size being current);
  live/frame.jpg (a complete JPEG: starts FF D8 and ends FF D9), every TICK_S;
  videos/*.mp4 (complete MP4 containers only: ftyp, moov and mdat boxes whose sizes add up to the file),
    videos/*_control.json (complete JSON) and videos/*_heard.wav (a RIFF size equal to the file);
  owner_state.json (the owner's record, read by the dashboard to check the owner process);
and this program's own record, mirror_status.json (each pass's time, the owner's phase, what was copied, the run's
nearest ancestor checkpoint inside the project).

EVERY COPY is written to a temporary file beside its target in the mirror folder and moved into place with os.replace,
so a reader never sees half a file; a copied picture, video or record keeps the run file's modification time (the
dashboard reads a clip's recording time and the picture's age from it). NOTHING IS WRITTEN INTO THE RUN FOLDER: run files
are opened only to read them.

A VIDEO IS NOT OPENED WHILE THE TRAINER MAY STILL REPLACE IT. env.py writes a clip's .mp4, _eye.mp4, _control.json and
_heard.wav, then mux_wav_into_mp4 replaces the .mp4 and the _eye.mp4 with sounding copies (os.replace of a
<clip>.mp4.mux.mp4, which stays in the run folder when the merge fails). On Windows a replace onto a file another process
holds open fails (the trainer catches it; that clip would keep no sound). So the files of one clip (its group) are read
only after every file of the group has kept the same size and the same modification second for SETTLE_S seconds over at
least two passes. Whole seconds: the D: disk's driver later reports a file's sub-second time as zero (measured Oct 13,
2026: step_491771_153337.mp4 read 3:57:20.69 PM PDT at first and 3:57:20.00 later, bytes unchanged), which would make a
settled clip look changed. Measured the same day: each clip's four files were written within 2 s, its .mp4 and _eye.mp4
carry their sound track (mp4a) after the mirror ran beside them.

THE LIVE PICTURE is read in one open-read-close, only when its size or time changed (a stat each TICK_S), so the read
follows the run's newest write by at most one TICK_S while the run writes the next picture a model step later (2.4 s on
average at 0.41 model steps per second). If env.live_frame's replace still meets the read, the replace fails and
datasource.py's one try around live_frame and live_state skips that model step's picture, its published body state and
any frame being kept for a recording; training goes on. The production dashboard serves the same file the same way.
Not measured: whether os.stat on this disk's driver can hold a replace back.

THE PROGRAM STOPS BY ITSELF when the owner state shows the run ended - its phase reads arm_end, arm_failed, completion
or done, it names another run, the file is gone, or neither the owner's nor the trainer's recorded process (pid and
creation time) is alive - on every pass for END_CONFIRM_S seconds (one failed look at the D: disk does not end it). It
then copies what the end leaves for up to SETTLE_S + 3 x MAIN_EVERY_S seconds more and exits. It also exits when no
readable pointer exists for END_CONFIRM_S seconds. An error inside one tick (a full disk, an unreadable file) is written
into the service record and the loop goes on.

A MIRROR FOLDER BELONGS TO ONE RUN: when its record names another run folder than the pointer (a later run with the same
folder name), the folder is moved to experiment_mirror/_replaced/<name>_<epoch seconds> before anything is copied.

    python -m mimo_bridge.tools.experiment_mirror point --run DIR --owner-state FILE --owner SCRIPT --by TEXT
    python -m mimo_bridge.tools.experiment_mirror copy-once --run DIR --by TEXT   an ended run's clips, once
    python -m mimo_bridge.tools.experiment_mirror start      start the loop detached; print its process identity
    python -m mimo_bridge.tools.experiment_mirror run        the loop itself (what `start` starts)
    python -m mimo_bridge.tools.experiment_mirror status     the service record and the mirror's record
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

REPO = Path(__file__).resolve().parents[4]
POINTER = Path('Saved/mimo_bridge/experiment_live.json')
MIRRORS = Path('Saved/mimo_bridge/experiment_mirror')
SERVICE = 'mirror_service.json'          # in MIRRORS: the running loop's process identity
STATUS = 'mirror_status.json'            # in each run's mirror folder
LOG = 'mirror.log'                       # in MIRRORS: the detached loop's output
# the live picture's copy period. The run publishes one every model step (2.4 s on average at the recovery run's 0.41
# model steps per second, longer on planning steps), and the dashboard calls the picture current while it is younger
# than three live-view periods (7.2 s at that rate); the copy adds up to one TICK_S to the picture's age, which the
# dashboard adds to its line (development_dashboard.live_texts). Measured Oct 13, 2026 at 3:59 PM PDT with 2 s: a copy
# 6.4 s old against the 7.2 s line.
TICK_S = 1.0
MAIN_EVERY_S = 15.0                      # records, videos and the owner state
SETTLE_S = 90.0                          # a clip's files unchanged this long, over two passes, before they are read
END_CONFIRM_S = 300.0                    # an end (or a missing pointer) read on every pass this long before the exit
END_PHASES = ('arm_end', 'arm_failed', 'completion', 'done')
STATS = 'replay_training_stats.jsonl'
SMALL_JSON = ('args.json', 'checkpoints.json')
RUN_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,119}')
VIDEO_PART = re.compile(r'^(?P<group>.+?)(?P<kind>_eye\.mp4|_control\.json|_heard\.wav|\.mp4)$')
PREFIX_CHECK = 65_536                    # bytes of the stats file compared with the mirror's on a loop's first pass


# --------------------------------------------------------------------------------------------- small helpers
def clock(t=None) -> str:
    lt = time.localtime(time.time() if t is None else t)
    return "%s %d, %d, %d:%02d:%02d %s %s" % (time.strftime('%b', lt), lt.tm_mday, lt.tm_year, lt.tm_hour % 12 or 12,
                                              lt.tm_min, lt.tm_sec, 'AM' if lt.tm_hour < 12 else 'PM',
                                              'PDT' if lt.tm_isdst else 'PST')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, TypeError):
        return default


def norm(path) -> str:
    return os.path.normcase(os.path.normpath(str(path)))


def created_of(pid):
    try:
        import psutil
        return psutil.Process(int(pid)).create_time()
    except Exception:                                                          # noqa: BLE001 - gone or no access
        return None


def same_process(pid, created) -> bool:
    """True while process `pid` runs and is the recorded one (creation time within 1 s; Windows reuses ids)."""
    if pid is None or created is None:
        return False
    now = created_of(pid)
    try:
        return now is not None and abs(now - float(created)) < 1.0
    except (TypeError, ValueError):
        return False


def replace_into(tmp: Path, target: Path, attempts=50, wait_s=0.1) -> bool:
    """os.replace(tmp, target), tried again while a reader (the dashboard serving the file) holds the target open;
    False and the temporary file removed when every attempt was refused."""
    for i in range(attempts):
        try:
            os.replace(tmp, target)
            return True
        except PermissionError:
            if i + 1 < attempts:
                time.sleep(wait_s)
    try:
        tmp.unlink()
    except OSError:
        pass
    return False


def write_atomic(target: Path, data: bytes, mtime=None, attempts=50) -> bool:
    """`data` into `target` through a temporary file beside it and os.replace; the copy keeps `mtime` when given."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name('%s.%d.tmp' % (target.name, os.getpid()))
    tmp.write_bytes(data)
    if mtime is not None:
        os.utime(tmp, (mtime, mtime))
    return replace_into(tmp, target, attempts=attempts)


def sig_of(st, coarse=False):
    """[size, modification time]: in ns, or in whole seconds with coarse (clips; see the module text)."""
    return [st.st_size, int(st.st_mtime) if coarse else st.st_mtime_ns]


def signature(path, coarse=False):
    """sig_of a file, or None when it cannot be read."""
    try:
        return sig_of(os.stat(path), coarse)
    except OSError:
        return None


def read_stable(path):
    """(bytes, stat) of a file when its size and modification time are the same before and after the read; else
    (None, None)."""
    try:
        before = os.stat(path)
        with open(path, 'rb') as stream:
            data = stream.read()
        after = os.stat(path)
    except OSError:
        return None, None
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(data) != after.st_size:
        return None, None
    return data, after


def mp4_complete(data: bytes) -> bool:
    """The MP4 container is finished: top-level boxes whose sizes add up to the file, holding ftyp, moov and mdat (the
    rule of development_dashboard.complete_mp4, on bytes already read)."""
    offset, kinds, size_all = 0, set(), len(data)
    while offset < size_all:
        if offset + 8 > size_all:
            return False
        size = int.from_bytes(data[offset:offset + 4], 'big')
        kind = data[offset + 4:offset + 8]
        header = 8
        if size == 1:
            if offset + 16 > size_all:
                return False
            size, header = int.from_bytes(data[offset + 8:offset + 16], 'big'), 16
        if size == 0:
            size = size_all - offset
        if size < header or offset + size > size_all:
            return False
        kinds.add(kind)
        offset += size
    return {b'ftyp', b'moov', b'mdat'}.issubset(kinds)


def jpeg_complete(data: bytes) -> bool:
    return len(data) > 4 and data[:2] == b'\xff\xd8' and data[-2:] == b'\xff\xd9'


def wav_complete(data: bytes) -> bool:
    return (len(data) >= 12 and data[:4] == b'RIFF' and data[8:12] == b'WAVE'
            and int.from_bytes(data[4:8], 'little') + 8 == len(data))


def json_complete(data: bytes) -> bool:
    try:
        json.loads(data.decode('utf-8-sig'))
        return True
    except (ValueError, UnicodeDecodeError):
        return False


CHECKS = {'.mp4': mp4_complete, '.json': json_complete, '.wav': wav_complete, '.jpg': jpeg_complete}


# --------------------------------------------------------------------------------------------- the pointer
def pointer_path(root=REPO) -> Path:
    return Path(root) / POINTER


def read_pointer(root=REPO):
    """The pointer as a dict, or None: no file, unreadable, no run or owner_state, or a run folder name that is not
    plain (letters, digits, _ . -), since the name becomes a folder inside the project."""
    p = read_json(pointer_path(root))
    if not isinstance(p, dict) or not p.get('run') or not p.get('owner_state'):
        return None
    if not RUN_NAME.fullmatch(Path(str(p['run'])).name):
        return None
    return p


def mirror_dir(root, pointer) -> Path:
    return Path(root) / MIRRORS / Path(str(pointer['run'])).name


def write_pointer(root, run, owner_state, owner, by) -> dict:
    """Write the pointer (atomic) for a run whose folder and owner state exist; the start comes from the owner state."""
    run, owner_state = Path(run), Path(owner_state)
    if not run.is_dir():
        raise SystemExit('refused: the run folder %s does not exist' % run)
    if not RUN_NAME.fullmatch(run.name):
        raise SystemExit('refused: the run folder name %r is not plain (letters, digits, _ . -)' % run.name)
    state = read_json(owner_state)
    if not isinstance(state, dict):
        raise SystemExit('refused: the owner state %s cannot be read' % owner_state)
    names = {norm(state.get('run') or ''), norm((state.get('trainer') or {}).get('run') or '')}
    if norm(run) not in names:
        raise SystemExit('refused: the owner state names run %s, not %s' % (state.get('run'), run))
    value = dict(schema=1, run=str(run), owner_state=str(owner_state), owner=str(owner),
                 started_at=state.get('started_at'), started_s=state.get('owner_created'), by=str(by),
                 written_at=clock())
    write_atomic(pointer_path(root), (json.dumps(value, indent=1) + '\n').encode('utf-8'))
    return value


# --------------------------------------------------------------------------------------------- the run's state
def run_ended(state, run, exists=True):
    """(ended, why) from the owner's record. A record file that is gone is an end; one that exists but does not read
    as JSON (caught mid-write) is (False, None): an unknown is not an end."""
    if not exists:
        return True, 'the owner state file is gone'
    if not isinstance(state, dict):
        return False, None
    phase = state.get('phase')
    if phase in END_PHASES:
        return True, 'the owner state reads phase "%s"' % phase
    if state.get('run') and norm(state['run']) != norm(run):
        return True, 'the owner state names another run, %s' % state['run']
    trainer = state.get('trainer') or {}
    owner_alive = same_process(state.get('owner_pid'), state.get('owner_created'))
    trainer_alive = same_process(trainer.get('pid'), trainer.get('created')) if isinstance(trainer, dict) else False
    if not owner_alive and not trainer_alive:
        return True, ('neither the owner process %s nor the trainer process %s is alive as recorded'
                      % (state.get('owner_pid'), trainer.get('pid') if isinstance(trainer, dict) else None))
    return False, None


def project_parent(root, run, hops=50):
    """(the nearest ancestor checkpoint inside the project as {run, step}, or None; the resume chain followed to it).
    Follows each run's args.json "resume" (a checkpoints/ckpt_step_N folder) to the run folder holding it; a run
    saved with --checkpoint-storage counts as its "out" folder, as development_dashboard.ancestry_bounds reads it."""
    root_r = Path(root).resolve()
    chain, current = [], Path(run)
    for _ in range(hops):
        args = read_json(current / 'args.json')
        if not isinstance(args, dict) or not args.get('resume'):
            return None, chain
        checkpoint = Path(str(args['resume']))
        match = re.fullmatch(r'ckpt_step_(\d+)(?:_.*)?', checkpoint.name)
        if not match or checkpoint.parent.name != 'checkpoints':
            return None, chain
        holder = checkpoint.parent.parent
        chain.append(dict(run=str(current), resume=str(checkpoint), step=int(match.group(1))))
        saved = read_json(holder / 'args.json') or {}
        if isinstance(saved, dict) and saved.get('checkpoint_storage') and saved.get('out'):
            holder = Path(str(saved['out']))
        try:
            inside = holder.resolve().is_relative_to(root_r)
        except OSError:
            inside = False
        if inside:
            return dict(run=holder.resolve().relative_to(root_r).as_posix(), step=int(match.group(1))), chain
        current = holder
    return None, chain


# --------------------------------------------------------------------------------------------- the copies
def copy_checked(src: Path, dst: Path, copied: dict, rel: str, notes: list, coarse=False):
    """Copy one file when its signature (sig_of, coarse for clips) differs from the copy's record; the bytes must pass
    the check of the file's suffix (CHECKS) and stay the same during the read. Returns the bytes when a copy was
    written, else None."""
    sig = signature(src, coarse)
    if sig is None:
        return None
    if copied.get(rel) == sig and dst.is_file():
        return None
    data, st = read_stable(src)
    if data is None:
        notes.append('%s changed during its read; read again next pass' % rel)
        return None
    check = CHECKS.get(src.suffix.lower())
    if check is not None and not check(data):
        notes.append('%s is not complete (%s check); not copied' % (rel, src.suffix.lower()))
        return None
    if not write_atomic(dst, data, mtime=st.st_mtime):
        notes.append('%s: the copy could not replace the mirror file (held open); tried again next pass' % rel)
        return None
    copied[rel] = sig_of(st, coarse)
    return data


def copy_frame(run: Path, mirror: Path, copied: dict, notes: list) -> bool:
    return copy_checked(run / 'live' / 'frame.jpg', mirror / 'live' / 'frame.jpg', copied, 'live/frame.jpg',
                        notes) is not None


def copy_stats(src: Path, dst: Path, state: dict) -> int:
    """Bring the mirror's stats file to the run file's bytes up to its last complete line; returns the bytes added.
    The mirror stays an exact prefix of the run file (checked on PREFIX_CHECK bytes once per loop, and whenever the run
    file is shorter than the copy), so a reader that remembers an offset reads on correctly."""
    try:
        size = os.stat(src).st_size
    except OSError:
        return 0
    have = dst.stat().st_size if dst.is_file() else 0
    if have and (size < have or not state.get('prefix_checked')):
        n = min(PREFIX_CHECK, have)
        try:
            with open(src, 'rb') as a, open(dst, 'rb') as b:
                same = a.read(n) == b.read(n)
        except OSError:
            return 0
        state['prefix_checked'] = True
        if not same or size < have:
            have = 0
            state['recopies'] = int(state.get('recopies') or 0) + 1
    if size <= have:
        return 0
    try:
        with open(src, 'rb') as stream:
            stream.seek(have)
            new = stream.read(size - have)
    except OSError:
        return 0
    cut = new.rfind(b'\n')
    if cut < 0:
        return 0
    new = new[:cut + 1]
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name('%s.%d.tmp' % (dst.name, os.getpid()))
    if have:
        shutil.copyfile(dst, tmp)
        with open(tmp, 'ab') as stream:
            stream.write(new)
    else:
        tmp.write_bytes(new)
    if not replace_into(tmp, dst):
        return 0
    return len(new)


def video_groups(vdir: Path) -> dict:
    """{group: {file name: signature}} of the clips' files (.mp4, _eye.mp4, _control.json, _heard.wav); a writer's
    temporary file (.part, .mux., .tmp) is left out."""
    groups = {}
    try:
        names = sorted(os.listdir(vdir))
    except OSError:
        return groups
    for name in names:
        if '.part' in name or '.mux.' in name or name.endswith('.tmp'):
            continue
        match = VIDEO_PART.match(name)
        if not match:
            continue
        sig = signature(vdir / name, coarse=True)
        if sig is not None:
            groups.setdefault(match.group('group'), {})[name] = sig
    return groups


def copy_videos(run: Path, mirror: Path, state: dict, copied: dict, now: float, settle_s: float, notes: list) -> dict:
    """Copy each clip whose files have settled (module text). Returns {copied: [...], waiting: {group: seconds}}."""
    seen = state.setdefault('groups', {})
    groups = video_groups(run / 'videos')
    out = dict(copied=[], waiting={})
    for group, files in groups.items():
        sig = sorted([name, s] for name, s in files.items())
        record = seen.get(group)
        if not record or record['sig'] != sig:
            seen[group] = record = dict(sig=sig, since=now, passes=1)
        else:
            record['passes'] += 1
        newest = max(s[1] for s in files.values())
        if record['passes'] < 2 or now - record['since'] < settle_s or now - newest < settle_s:
            if not all(copied.get('videos/' + name) == s and (mirror / 'videos' / name).is_file()
                       for name, s in files.items()):
                out['waiting'][group] = round(now - record['since'], 1)
            continue
        for name in files:
            if copy_checked(run / 'videos' / name, mirror / 'videos' / name, copied, 'videos/' + name, notes,
                            coarse=True) is not None:
                out['copied'].append(name)
    for group in [g for g in seen if g not in groups]:
        del seen[group]
    return out


def claim_mirror(root, pointer, now):
    """A mirror folder whose record names another run folder than the pointer is moved to _replaced/<name>_<epoch s>
    (module text); returns the folder it was moved to, or None."""
    mirror = mirror_dir(root, pointer)
    status = read_json(mirror / STATUS)
    if not isinstance(status, dict) or not status.get('run') or norm(status['run']) == norm(pointer['run']):
        return None
    aside = Path(root) / MIRRORS / '_replaced' / ('%s_%d' % (mirror.name, int(now)))
    aside.parent.mkdir(parents=True, exist_ok=True)
    os.replace(mirror, aside)
    return aside


def main_pass(root, pointer, state: dict, now: float, settle_s: float = SETTLE_S) -> dict:
    """One pass: the owner state, the small records, the stats file and the settled clips into the mirror; then the
    mirror's record. Returns that record. The end is judged on the same bytes of the owner state that the pass copies
    (one read), so the copy never lags the judgement."""
    run = Path(str(pointer['run']))
    mirror = mirror_dir(root, pointer)
    moved = claim_mirror(root, pointer, now)
    if moved:
        state.clear()
    mirror.mkdir(parents=True, exist_ok=True)
    copied = state.setdefault('copied', {})
    notes = ['the folder held another run\'s copies and was moved to %s' % moved] if moved else []
    owner_path = Path(str(pointer['owner_state']))
    data, st = read_stable(owner_path)
    owner = None
    if data is not None and json_complete(data):
        owner = json.loads(data.decode('utf-8-sig'))
        if copied.get('owner_state.json') != sig_of(st) or not (mirror / 'owner_state.json').is_file():
            if write_atomic(mirror / 'owner_state.json', data, mtime=st.st_mtime):
                copied['owner_state.json'] = sig_of(st)
    elif owner_path.is_file():
        notes.append('owner_state.json could not be read as complete JSON this pass')
    for name in SMALL_JSON:
        copy_checked(run / name, mirror / name, copied, name, notes)
    added = copy_stats(run / STATS, mirror / STATS, state)
    if 'parent' not in state:
        state['parent'], state['chain'] = project_parent(root, run)
    videos = copy_videos(run, mirror, state, copied, now, settle_s, notes)
    ended, why = run_ended(owner, run, exists=data is not None or owner_path.is_file())
    trainer = (owner or {}).get('trainer') if isinstance(owner, dict) else None
    trainer = trainer if isinstance(trainer, dict) else {}
    record = dict(
        schema=1, about='experiment_mirror.py: copies of a live experiment run kept for the development dashboard',
        pid=os.getpid(), pid_created=created_of(os.getpid()), pass_at=now, pass_at_text=clock(now),
        run=str(run), run_name=run.name, mirror=mirror.relative_to(Path(root)).as_posix(),
        pointer=str(pointer_path(root)), owner=pointer.get('owner'), owner_state=str(pointer['owner_state']),
        owner_phase=(owner or {}).get('phase') if isinstance(owner, dict) else None,
        owner_pid=(owner or {}).get('owner_pid') if isinstance(owner, dict) else None,
        owner_alive=same_process((owner or {}).get('owner_pid'), (owner or {}).get('owner_created'))
        if isinstance(owner, dict) else False,
        trainer_pid=trainer.get('pid'), trainer_alive=same_process(trainer.get('pid'), trainer.get('created')),
        ended=ended, end_reason=why,
        stats=dict(added_bytes=added, mirrored_bytes=(mirror / STATS).stat().st_size if (mirror / STATS).is_file() else 0,
                   recopies=int(state.get('recopies') or 0)),
        videos_copied_this_pass=videos['copied'], videos_waiting_s=videos['waiting'],
        project_parent=state.get('parent'), resume_chain=state.get('chain'),
        copied=copied, notes=notes[-20:])
    write_atomic(mirror / STATUS, (json.dumps(record, indent=1) + '\n').encode('utf-8'))
    return record


# --------------------------------------------------------------------------------------------- the loop
def opt_out_of_power_throttling() -> str:
    """Opt this process out of Windows power throttling (the setting no_power_throttling_watch.py applies to the
    project's hidden jobs, measured Oct 6, 2026: throttled hidden jobs ran 4x slower once the user was idle); returns
    what was done, checked by reading the setting back."""
    if os.name != 'nt':
        return 'not applicable (not Windows)'
    try:
        import ctypes
        from ctypes import wintypes

        class PPTS(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG), ('StateMask', wintypes.ULONG)]
        k32 = ctypes.WinDLL('kernel32', use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.GetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        me = k32.GetCurrentProcess()
        want = PPTS(1, 0x1, 0)
        if not k32.SetProcessInformation(me, 4, ctypes.byref(want), ctypes.sizeof(want)):
            return 'not opted out: SetProcessInformation error %d' % ctypes.get_last_error()
        got = PPTS(1, 0, 0)
        if k32.GetProcessInformation(me, 4, ctypes.byref(got), ctypes.sizeof(got)) and got.ControlMask & 0x1 \
                and not got.StateMask & 0x1:
            return 'opted out (read back: execution-speed throttling controlled and off)'
        return 'set; the read-back did not confirm it'
    except Exception as exc:                                                   # noqa: BLE001 - reported, not fatal
        return 'not opted out: %s: %s' % (type(exc).__name__, exc)


def tick(root, states, memo, t, main_every_s, settle_s, end_confirm_s):
    """One tick of the loop (serve). Returns None to go on, or the service fields of the loop's end."""
    pointer = read_pointer(root)
    if pointer is None:
        memo['no_pointer_since'] = memo.get('no_pointer_since') or t
        if t - memo['no_pointer_since'] >= end_confirm_s:
            return dict(state='ended', ended_at=clock(t), why='no readable pointer at %s for %d s'
                        % (pointer_path(root), t - memo['no_pointer_since']))
        return None
    memo['no_pointer_since'] = None
    name = Path(str(pointer['run'])).name
    if name not in states:
        status = read_json(mirror_dir(root, pointer) / STATUS) or {}
        same = isinstance(status, dict) and norm(status.get('run') or '') == norm(pointer['run'])
        states[name] = dict(copied=dict(status.get('copied') or {}) if same else {})
    state = states[name]
    if memo.get('ended_at') is None:
        copy_frame(Path(str(pointer['run'])), mirror_dir(root, pointer), state['copied'], [])
    if memo.get('last_main') is None or t - memo['last_main'] >= main_every_s:
        memo['last_main'] = t
        record = main_pass(root, pointer, state, t, settle_s)
        if not record['ended']:
            memo['ended_at'] = None
            return None
        memo['ended_at'] = memo.get('ended_at') or t
        since = t - memo['ended_at']
        if since >= end_confirm_s and (not record['videos_waiting_s']
                                       or since >= end_confirm_s + settle_s + 3 * main_every_s):
            return dict(state='ended', ended_at=clock(t), why='%s (read on every pass for %d s)'
                        % (record['end_reason'], since), left_waiting=record['videos_waiting_s'])
    return None


def serve(root=REPO, tick_s=TICK_S, main_every_s=MAIN_EVERY_S, settle_s=SETTLE_S, now=time.time, sleep=time.sleep,
          max_ticks=None, end_confirm_s=END_CONFIRM_S) -> int:
    """The loop: the live picture every tick_s, a main pass every main_every_s; ends after the run ended, or after the
    pointer went missing, for end_confirm_s (module text). An error inside a tick is recorded in the service record
    (errors, last_error) and the loop goes on. One loop at a time (a lock file in the mirrors folder)."""
    from mimo_bridge.tools.training_continuation import OwnerLock
    base = Path(root) / MIRRORS
    base.mkdir(parents=True, exist_ok=True)
    try:
        lock = OwnerLock(base / 'mirror.lock')
    except OSError:
        print('refused: another experiment mirror holds %s' % (base / 'mirror.lock'), flush=True)
        return 3
    try:
        throttle = opt_out_of_power_throttling()
        service = dict(pid=os.getpid(), pid_created=created_of(os.getpid()), started_at=clock(), started_s=now(),
                       command=[sys.executable] + sys.argv, power_throttling=throttle, state='running')
        write_atomic(base / SERVICE, (json.dumps(service, indent=1) + '\n').encode('utf-8'))
        print(json.dumps(service), flush=True)
        states, memo, ticks = {}, {}, 0
        while max_ticks is None or ticks < max_ticks:
            ticks += 1
            t = now()
            try:
                end = tick(root, states, memo, t, main_every_s, settle_s, end_confirm_s)
            except Exception as exc:                                           # noqa: BLE001 - recorded; the loop goes on
                import traceback
                service.update(errors=int(service.get('errors') or 0) + 1, last_error=dict(
                    at=clock(t), error='%s: %s' % (type(exc).__name__, exc), traceback=traceback.format_exc()[-1500:]))
                memo['last_main'] = t                                          # the next main pass waits its period
                try:
                    write_atomic(base / SERVICE, (json.dumps(service, indent=1) + '\n').encode('utf-8'))
                except Exception:                                              # noqa: BLE001 - a full disk: go on
                    pass
                end = None
            if end:
                service.update(end)
                break
            sleep(tick_s)
        if service['state'] == 'running':
            service.update(state='stopped', ended_at=clock(), why='tick limit of this call reached')
        write_atomic(base / SERVICE, (json.dumps(service, indent=1) + '\n').encode('utf-8'))
        print(json.dumps(service), flush=True)
        return 0
    finally:
        lock.close()


def start(root=REPO, wait_s=30.0) -> int:
    """Start `run` detached (no window, its own process group, output to the mirrors folder's mirror.log) from this
    checkout's Source/Python; return once the service record names a process of the started family that is alive."""
    import subprocess
    import psutil
    base = Path(root) / MIRRORS
    base.mkdir(parents=True, exist_ok=True)
    old = read_json(base / SERVICE) or {}
    if same_process(old.get('pid'), old.get('pid_created')):
        print('refused: the mirror process %s (created %s) is alive' % (old.get('pid'), clock(old.get('pid_created'))))
        return 3
    if read_pointer(root) is None:
        print('refused: no readable pointer at %s' % pointer_path(root))
        return 2
    py = Path(root) / '.venv_mimo' / 'Scripts' / 'python.exe'
    py = py if py.is_file() else Path(sys.executable)
    src = Path(root) / 'Source' / 'Python'
    env = dict(os.environ, PYTHONPATH=str(src), PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='-1')
    flags = 0x08000000 | 0x00000008 | 0x00000200      # CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    began = time.time()
    with open(base / LOG, 'ab') as log:
        p = subprocess.Popen([str(py), '-B', '-u', '-m', 'mimo_bridge.tools.experiment_mirror', 'run'], cwd=str(src),
                             env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=flags if os.name == 'nt' else 0)
    while time.time() - began < wait_s:
        time.sleep(0.5)
        rec = read_json(base / SERVICE) or {}
        try:
            family = {q.pid for q in [psutil.Process(p.pid)] + psutil.Process(p.pid).children(recursive=True)}
        except psutil.Error:
            family = set()
        if rec.get('pid') in family and same_process(rec.get('pid'), rec.get('pid_created')) \
                and (rec.get('pid_created') or 0) >= began - 1:
            print(json.dumps(dict(started=True, launcher_pid=p.pid, pid=rec['pid'],
                                  pid_created=rec['pid_created'], created_at=clock(rec['pid_created']),
                                  power_throttling=rec.get('power_throttling'), log=str(base / LOG)), indent=1))
            return 0
    print(json.dumps(dict(started=False, launcher_pid=p.pid, alive=p.poll() is None, log=str(base / LOG),
                          why='no service record naming a live process of the started family within %d s' % wait_s)))
    return 4


LIVE_RECORD_S = 600.0        # a run whose newest training record is younger than this may still train (the dashboard's line)


def newest_record_time(stats):
    """The 'timestamp' of the last complete line of a stats file that has one (its last 4 MB read), or None."""
    try:
        with open(stats, 'rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4_000_000))
            lines = stream.read().split(b'\n')
    except OSError:
        return None
    for raw in reversed(lines[:-1] if lines and lines[-1] else lines):
        try:
            stamp = json.loads(raw).get('timestamp')
        except (ValueError, AttributeError):
            continue
        if isinstance(stamp, (int, float)):
            return float(stamp)
    return None


def copy_once(root, run, by, now=None, settle_s=SETTLE_S) -> dict:
    """COPY AN ENDED EXPERIMENT RUN'S CLIPS ONCE (Oct 13, 2026: runs that ended before the mirror existed, such as B1C
    and B1E): its args.json and every clip file (videos/*.mp4, _eye.mp4, _control.json, _heard.wav) whose group has not
    changed for settle_s, into Saved/mimo_bridge/experiment_mirror/<run name>/, each checked and written as the loop
    writes it; no stats file and no owner state. Refused for a run whose folder name is the pointer's run's (the loop
    owns that mirror folder), for a run whose newest training record is under LIVE_RECORD_S old (it may still train),
    for a run name that is not plain and for a missing run folder. Writes the folder's record with copied_once true and
    returns it."""
    now = time.time() if now is None else now
    run = Path(run)
    if not run.is_dir() or not RUN_NAME.fullmatch(run.name):
        raise SystemExit('refused: %s is not a run folder with a plain name' % run)
    live = read_pointer(root)
    if live and (norm(live['run']) == norm(run) or Path(str(live['run'])).name == run.name):
        raise SystemExit('refused: the pointer names %s, whose mirror folder has this name; the mirror loop copies it'
                         % live['run'])
    newest = newest_record_time(run / STATS)
    if newest is not None and now - newest < LIVE_RECORD_S:
        raise SystemExit('refused: %s wrote a training record %d s ago (under %d s): it may still be training'
                         % (run, now - newest, LIVE_RECORD_S))
    pointer = dict(run=str(run), owner_state=None)
    moved = claim_mirror(root, pointer, now)
    mirror = mirror_dir(root, pointer)
    copied, notes, done, waiting = {}, ['moved another run\'s copies to %s' % moved] if moved else [], [], []
    old = read_json(mirror / STATUS) or {}
    if isinstance(old, dict) and norm(old.get('run') or '') == norm(run):
        copied = dict(old.get('copied') or {})
    copy_checked(run / 'args.json', mirror / 'args.json', copied, 'args.json', notes)
    for group, files in video_groups(run / 'videos').items():
        if now - max(s[1] for s in files.values()) < settle_s:
            waiting.append(group)
            continue
        for name in files:
            if copy_checked(run / 'videos' / name, mirror / 'videos' / name, copied, 'videos/' + name, notes,
                            coarse=True) is not None:
                done.append(name)
    record = dict(schema=1, about='experiment_mirror.py copy-once: clips of an ended experiment run kept for the '
                  'development dashboard', copied_once=True, ended=True, end_reason='copied once after the run ended',
                  run=str(run), run_name=run.name, mirror=mirror.relative_to(Path(root)).as_posix(), by=str(by),
                  pass_at=now, pass_at_text=clock(now), copied_this_call=done, waiting=waiting, copied=copied,
                  notes=notes[-20:])
    write_atomic(mirror / STATUS, (json.dumps(record, indent=1) + '\n').encode('utf-8'))
    return record


def live_view(root=REPO):
    """The experiment run the dashboard may show, or None. Read only inside the project: the pointer, the mirror's
    record, its service record and its copy of the owner state; the mirror's, the owner's and the trainer's recorded
    processes are checked alive (pid and creation time). None when the pointer, the mirror's copies or the owner state
    copy is missing, the mirror's record names another run folder, the mirror process is not alive as recorded (its
    copy of the owner state would no longer follow the owner), the owner state names another run or an end phase, or
    the owner process is not alive as recorded."""
    pointer = read_pointer(root)
    if pointer is None:
        return None
    mirror = mirror_dir(root, pointer)
    owner = read_json(mirror / 'owner_state.json')
    status = read_json(mirror / STATUS) or {}
    service = read_json(Path(root) / MIRRORS / SERVICE) or {}
    if not isinstance(owner, dict) or not isinstance(status, dict) or not (mirror / 'args.json').is_file():
        return None
    if norm(status.get('run') or '') != norm(pointer['run']) or not isinstance(service, dict):
        return None
    if not same_process(service.get('pid'), service.get('pid_created')):
        return None
    if owner.get('phase') in END_PHASES or norm(owner.get('run') or '') != norm(pointer['run']):
        return None
    if not same_process(owner.get('owner_pid'), owner.get('owner_created')):
        return None
    trainer = owner.get('trainer') if isinstance(owner.get('trainer'), dict) else {}
    return dict(mirror=mirror, run_name=mirror.name, source_run=str(pointer['run']), owner=pointer.get('owner'),
                owner_pid=owner.get('owner_pid'), owner_alive=True, phase=owner.get('phase'),
                trainer_pid=trainer.get('pid'), trainer_alive=same_process(trainer.get('pid'), trainer.get('created')),
                started_at=pointer.get('started_at'), pointer_by=pointer.get('by'),
                mirror_pass_at=status.get('pass_at') if isinstance(status.get('pass_at'), (int, float)) else None,
                mirror_pid=service.get('pid'), mirror_alive=True,
                project_parent=status.get('project_parent') if isinstance(status.get('project_parent'), dict) else None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    pt = sub.add_parser('point')
    pt.add_argument('--run', required=True)
    pt.add_argument('--owner-state', required=True)
    pt.add_argument('--owner', required=True)
    pt.add_argument('--by', required=True)
    co = sub.add_parser('copy-once')
    co.add_argument('--run', required=True)
    co.add_argument('--by', required=True)
    sub.add_parser('start')
    sub.add_parser('run')
    sub.add_parser('status')
    a = ap.parse_args(argv)
    if a.cmd == 'point':
        print(json.dumps(write_pointer(REPO, a.run, a.owner_state, a.owner, a.by), indent=1))
        return 0
    if a.cmd == 'copy-once':
        rec = copy_once(REPO, a.run, a.by)
        print(json.dumps({k: rec[k] for k in ('run', 'mirror', 'copied_this_call', 'waiting', 'notes')}, indent=1))
        return 0
    if a.cmd == 'start':
        return start(REPO)
    if a.cmd == 'run':
        return serve(REPO)
    pointer = read_pointer(REPO)
    print(json.dumps(dict(pointer=pointer, service=read_json(REPO / MIRRORS / SERVICE),
                          mirror=read_json(mirror_dir(REPO, pointer) / STATUS) if pointer else None,
                          view=(lambda v: v and dict(v, mirror=str(v['mirror'])))(live_view(REPO))), indent=1,
                     default=str))
    return 0


if __name__ == '__main__':
    sys.exit(main())
