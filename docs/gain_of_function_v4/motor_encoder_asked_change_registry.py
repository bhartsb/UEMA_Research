"""The test registry rows of the motor encoder's asked-change work, and the two writers that put them into production's
registry (docs/gain_of_function_v4/test_registry.json of D:/UEMA_Research).

    PY motor_encoder_asked_change_registry.py check            every row of ROWS: complete, and its wording passes the hooks
    PY motor_encoder_asked_change_registry.py add ID [ID ...]  append those rows (refused when a row with the id exists)
    PY motor_encoder_asked_change_registry.py set ID FILE      write the fields of JSON file FILE into row ID
    PY motor_encoder_asked_change_registry.py run ID LOG -- CMD...   run CMD as row ID's job: refused without a complete
                                                               row; writes status running, then done or failed, and last_run
    PY motor_encoder_asked_change_registry.py capped -- CMD... a development run under the same limits, without a row
    PY motor_encoder_asked_change_registry.py room             the commit room now

`run` and `capped` work beside production training: the command runs CPU only inside a Windows job object that holds
each process at 1.5 GB of RAM and 2.5 GB of committed memory, and is refused with under 8 GB of commit room (WORK
BESIDE PRODUCTION TRAINING below).

Both writers follow mimo_bridge/tools/test_registry.py: its lock file, a fresh read of the file under the lock, a write
in the file's own layout (dumps reproduces the file byte for byte) and a replace only when the file's size and
modification time still equal those of that read. Before any write the new text is checked with the two hooks' own lists
(.agent/hooks/worklog_time_check.hiding_lines, reply_lint.BANNED), because the hooks themselves run only on the Write
and Edit tools.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

MAIN = Path("D:/UEMA_Research")
sys.path.insert(0, str(MAIN / "Source" / "Python"))
sys.path.insert(0, str(MAIN / ".agent" / "hooks"))
from mimo_bridge.tools import test_registry as tr          # noqa: E402

HERE = Path(__file__).resolve().parent
WT = "C:/Users/bhart/Documents/UEMA_wt/motor_encoder/"
REGISTRY = MAIN / "docs" / "gain_of_function_v4" / "test_registry.json"
ROWS_FILE = HERE / "motor_encoder_asked_change_registry_rows.json"


def rows() -> list:
    return json.loads(ROWS_FILE.read_text(encoding="utf-8"))


def _texts(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _texts(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _texts(v)


def wording_problems(row: dict) -> list:
    """The hooks' findings on every text of the row except its code list (the only field where code names belong)."""
    from worklog_time_check import hiding_lines
    from reply_lint import BANNED
    try:
        # since Oct 5, 2026: a forbidden alias of the component manifest, and a status whose state is not one of the
        # eight repair states (the hooks of a checkout older than that do not have the check)
        from reply_lint import name_problems
    except ImportError:
        name_problems = None
    out = []
    for key, value in row.items():
        if key == "code":
            continue
        for text in _texts(value):
            for line in hiding_lines(text):
                out.append("%s: %s" % (key, line))
            for line in (name_problems(text) if name_problems else []):
                out.append("%s: %s" % (key, line))
            low = re.sub(r"`[^`\n]*`", " ", text).lower()
            for pat, why in BANNED:
                m = re.search(pat, low)
                if m:
                    out.append("%s: %s: '...%s...'" % (key, why, low[max(0, m.start() - 40):m.end() + 40]))
    return out


def problems(row: dict, required) -> list:
    return ["%s (%s)" % (text, key) for key, text in tr.row_problems(row, required)] + wording_problems(row)


def check(ids=None) -> int:
    data = tr.load(REGISTRY)
    required = tr.required_fields(data)
    bad = 0
    for row in rows():
        if ids and row["id"] not in ids:
            continue
        found = problems(row, required)
        print(row["id"], "OK" if not found else "PROBLEMS")
        for f in found:
            print("   ", f)
        bad += bool(found)
    return 1 if bad else 0


def add(ids) -> int:
    wanted = [r for r in rows() if r["id"] in ids]
    if len(wanted) != len(set(ids)):
        raise SystemExit("unknown row id among %s" % (ids,))
    with tr._Lock(REGISTRY):
        for _ in range(5):
            before = tr._stamp(REGISTRY)
            data = tr.load(REGISTRY)                      # the file as it is now, read under the lock
            required = tr.required_fields(data)
            for row in wanted:
                if tr.rows_with_id(data, row["id"]):
                    raise SystemExit("a row with id %s is already in the registry" % row["id"])
                found = problems(row, required)
                if found:
                    raise SystemExit("row %s is not written: %s" % (row["id"], "; ".join(found)))
                row = dict(row)
                row["written_at"] = tr.clock()            # the machine's clock at the write
                data["rows"].append(row)
            text = tr.dumps(data) + "\n"
            if tr._stamp(REGISTRY) != before:
                continue                                  # another writer replaced the file: read it again
            tr._replace(REGISTRY, text)
            for row in wanted:
                print("added", row["id"], "at", tr.clock())
            return 0
    raise SystemExit("the registry changed during 5 attempts")


def set_fields(row_id: str, fields: dict) -> int:
    if isinstance(fields.get("result"), dict) and fields["result"].get("read_at") == "NOW":
        fields["result"]["read_at"] = tr.clock()          # the machine's clock at the write
    found = wording_problems(fields)
    if found:
        raise SystemExit("not written: %s" % "; ".join(found))
    try:
        written = tr.update_row(row_id, fields, path=REGISTRY)
    except tr.RegistryError as exc:
        # since Oct 5, 2026 the registry refuses a hand write of the runner's fields (status running, done or failed;
        # outcome, last_run, next_action): those are written by `run`
        raise SystemExit("not written: %s" % exc)
    if not written:
        raise SystemExit("no single row has id %s" % row_id)
    print("set", row_id, sorted(fields), "at", tr.clock())
    return 0


# WORK BESIDE PRODUCTION TRAINING (the coordinator's limits of Oct 3, 2026 1:39 AM PDT, when the 8-world trainer
# restarted at about 37 GB): one process at a time, each under 1.5 GB of RAM, CPU only, and no run started with under
# 8 GB of commit room. capped() makes the limits facts instead of estimates, for the command and every process it
# starts:
#   * RAM: each process gets a hard working-set maximum of RAM_GB (SetProcessWorkingSetSizeEx with
#     QUOTA_LIMITS_HARDWS_MAX_ENABLE): pages above it go to the page file instead of taking RAM from the trainer.
#     Set on the command at its start and on each process it starts when the twice-a-second sampling first sees it.
#   * committed memory: the processes run in a Windows job object whose per-process limit is COMMIT_GB: an
#     allocation above it fails inside that process (a MemoryError) instead of taking commit room from the trainer.
# WHY TWO NUMBERS. MEASURED Oct 3, 2026 (this machine, the trainer's own interpreter): `import torch` alone commits
# 1.35 GB (the graphics libraries' data, committed and never touched with the card hidden) while holding 0.38 GB of
# RAM; a small test learner built on top holds 0.47 GB of RAM at 1.47 GB committed. A commit limit of 1.5 GB therefore
# refuses every process that imports torch. The RAM limit is the coordinator's 1.5 GB; the commit limit is 2.5 GB
# (torch's 1.35 GB and 1.15 GB of work); a run starts only with ROOM_GB of commit room, of which one process can
# take at most COMMIT_GB. MEASURED the same day: a process that touched 2.0 GB under the hard maximum held 1.50 GB of
# RAM; a job-level working-set limit needs a privilege this account does not hold (error 1314), so the maximum is
# set per process.
# THE RAM A RUN MAY TAKE IS WHAT IS FREE. MEASURED 1:46 AM PDT Oct 3, four minutes after the trainer started: 0.76 to
# 0.93 GB of RAM available (the trainer and its 8 world processes held 19.6 GB of the machine's 29.7), not the few GB
# the limits assumed. So a run's hard maximum is min(RAM_GB, the RAM available at its start - RAM_MARGIN_GB): it can
# never take RAM the trainer holds. A run needs RAM_NEED_GB (the small learner's measured 0.53 GB); with less
# available, or with under ROOM_GB of commit room, capped() waits, asking every 10 s for up to WAIT_S, then refuses. The command runs at below-normal
# priority (its children inherit it), so the trainer's processes get the processor first.
RAM_GB = 1.5
RAM_MARGIN_GB = 0.15
RAM_NEED_GB = 0.55
WAIT_S = 600
COMMIT_GB = 2.5
ROOM_GB = 8.0
_JOB = None


def available_ram_gb() -> float:
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def commit_room_gb() -> float:
    """How much more memory Windows can still commit (the commit limit minus what is committed), GB."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPageFile / 2 ** 30


def _win():
    import ctypes
    from ctypes import wintypes

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in ("r", "w", "o", "rt", "wt", "ot")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    class LUID(ctypes.Structure):
        _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    class TP(ctypes.Structure):
        _fields_ = [("PrivilegeCount", wintypes.DWORD), ("Luid", LUID), ("Attributes", wintypes.DWORD)]
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                              ctypes.c_void_p]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.SetProcessWorkingSetSizeEx.argtypes = [wintypes.HANDLE, ctypes.c_size_t, ctypes.c_size_t, wintypes.DWORD]
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    adv.LookupPrivilegeValueW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, ctypes.POINTER(LUID)]
    adv.AdjustTokenPrivileges.argtypes = [wintypes.HANDLE, wintypes.BOOL, ctypes.c_void_p, wintypes.DWORD,
                                          ctypes.c_void_p, ctypes.c_void_p]
    return ctypes, wintypes, k32, adv, EXT, LUID, TP


def cap_memory(commit_gb: float = COMMIT_GB) -> None:
    """Put this process and every child it starts into a job object with the per-process commit limit, and enable
    the privilege the hard working-set maximum needs (SeIncreaseWorkingSetPrivilege, held by every user account)."""
    global _JOB
    if _JOB is not None:
        return
    ctypes, wintypes, k32, adv, EXT, LUID, TP = _win()
    job = k32.CreateJobObjectW(None, None)
    if not job:
        raise OSError("CreateJobObjectW failed: %d" % ctypes.get_last_error())
    info = EXT()
    info.BasicLimitInformation.LimitFlags = 0x100          # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = int(commit_gb * 2 ** 30)
    if not k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        raise OSError("SetInformationJobObject failed: %d" % ctypes.get_last_error())
    if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
        raise OSError("AssignProcessToJobObject failed: %d" % ctypes.get_last_error())
    tok = wintypes.HANDLE()
    luid = LUID()
    tp = TP()
    if not (adv.OpenProcessToken(k32.GetCurrentProcess(), 0x20 | 0x8, ctypes.byref(tok))
            and adv.LookupPrivilegeValueW(None, "SeIncreaseWorkingSetPrivilege", ctypes.byref(luid))):
        raise OSError("the working-set privilege could not be looked up: %d" % ctypes.get_last_error())
    tp.PrivilegeCount, tp.Luid, tp.Attributes = 1, luid, 0x2
    ctypes.set_last_error(0)
    if not adv.AdjustTokenPrivileges(tok, False, ctypes.byref(tp), 0, None, None) or ctypes.get_last_error():
        raise OSError("the working-set privilege could not be enabled: %d" % ctypes.get_last_error())
    _JOB = job


def hold_ram(pid: int, ram_gb: float = RAM_GB) -> bool:
    """The hard working-set maximum on process `pid`; False when the process is gone or refuses."""
    ctypes, wintypes, k32, adv, EXT, LUID, TP = _win()
    h = k32.OpenProcess(0x0100 | 0x1000, False, int(pid))          # PROCESS_SET_QUOTA, QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        # QUOTA_LIMITS_HARDWS_MAX_ENABLE (0x4), QUOTA_LIMITS_HARDWS_MIN_DISABLE (0x2)
        return bool(k32.SetProcessWorkingSetSizeEx(h, 1 << 20, int(ram_gb * 2 ** 30), 0x4 | 0x2))
    finally:
        k32.CloseHandle(h)


def peak_commit_gb():
    """The most memory any one process of the job has committed, GB (None before cap_memory)."""
    if _JOB is None:
        return None
    ctypes, wintypes, k32, adv, EXT, LUID, TP = _win()
    info = EXT()
    if not k32.QueryInformationJobObject(_JOB, 9, ctypes.byref(info), ctypes.sizeof(info), None):
        return None
    return info.PeakProcessMemoryUsed / 2 ** 30


def capped(cmd: list, stream=None, timeout_s=None, room_gb: float = ROOM_GB, on_start=None) -> int:
    """Run cmd under the limits above: refused with under room_gb of commit room; CPU only (the card hidden). Prints
    (and keeps in capped.last) the largest RAM one process of the run held, sampled twice a second, and the most one
    process committed. on_start(pid), when given, is called once with the command's process id right after its
    start (run() writes it into the registry row)."""
    import os
    import psutil
    waited = time.time()
    ram = min(RAM_GB, available_ram_gb() - RAM_MARGIN_GB)
    room = commit_room_gb()
    while (ram < RAM_NEED_GB or room < room_gb) and time.time() - waited < WAIT_S:
        time.sleep(10)
        ram = min(RAM_GB, available_ram_gb() - RAM_MARGIN_GB)
        room = commit_room_gb()
    if ram < RAM_NEED_GB:
        raise SystemExit("refused: %.2f GB of RAM available after %.0f s of waiting; a run needs %.2f GB and leaves "
                         "%.2f GB" % (available_ram_gb(), time.time() - waited, RAM_NEED_GB, RAM_MARGIN_GB))
    if room < room_gb:
        raise SystemExit("refused: %.2f GB of commit room after %.0f s of waiting, under the %.0f GB a run beside "
                         "training needs" % (room, time.time() - waited, room_gb))
    cap_memory()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", UEMA_TESTS_BESIDE_TRAINING="1")
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT if stream is not None else None, env=env,
                            creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    if on_start is not None:
        on_start(proc.pid)
    held = {proc.pid: hold_ram(proc.pid, ram)}
    peak_ram, most_processes = 0.0, 0
    code = None
    while code is None:
        try:
            code = proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        try:
            family = [psutil.Process(proc.pid)] + psutil.Process(proc.pid).children(recursive=True)
            rss = []
            for p in family:
                if p.pid not in held:
                    held[p.pid] = hold_ram(p.pid, ram)
                rss.append(p.memory_info().rss)
            peak_ram = max(peak_ram, max(rss) / 2 ** 30)
            most_processes = max(most_processes, sum(1 for r in rss if r > 200 * 2 ** 20))
        except (psutil.Error, ValueError):
            pass
        if code is None and timeout_s and time.time() - t0 > timeout_s:
            try:
                for p in psutil.Process(proc.pid).children(recursive=True):
                    p.kill()
            except psutil.Error:
                pass
            proc.kill()
            code = 124
    capped.last = dict(exit_code=code, seconds=round(time.time() - t0, 1), peak_ram_gb=round(peak_ram, 2),
                       peak_commit_gb=round(peak_commit_gb() or float("nan"), 2), commit_room_at_start_gb=round(room, 1),
                       ram_limit_gb=round(ram, 2), commit_limit_gb=COMMIT_GB, processes=len(held),
                       processes_held_to_the_ram_limit=sum(1 for v in held.values() if v),
                       most_processes_above_200_mb_at_once=most_processes)
    print("exit %s after %.0f s; largest RAM of one process %.2f GB (limit %.2f); most committed by one process %.2f GB "
          "(limit %.1f); commit room at the start %.1f GB; %d of %d processes held to the RAM limit; at most %d above "
          "200 MB at once" % (code, time.time() - t0, peak_ram, ram, peak_commit_gb() or float("nan"), COMMIT_GB,
                              room, capped.last["processes_held_to_the_ram_limit"], len(held), most_processes),
          flush=True)
    return code


def run(row_id: str, log: str, cmd: list, timeout_s=None) -> int:
    """Run `cmd` as the job of row `row_id`, as the pause runners do: refused unless the row is complete; the row's
    status and last_run are written before and after. The job runs under capped()'s limits."""
    why = tr.refusal(row_id, path=REGISTRY)
    if why:
        raise SystemExit("refused: " + why)
    row = tr.rows_with_id(tr.load(REGISTRY), row_id)[0]
    limit = timeout_s or (row.get("cost") or {}).get("time_limit_s")
    waited = time.time()
    while commit_room_gb() < ROOM_GB and time.time() - waited < WAIT_S:
        time.sleep(10)                                    # before the row says "running": the wait for commit room
    if commit_room_gb() < ROOM_GB:
        raise SystemExit("refused: %.2f GB of commit room after %.0f s of waiting, under %.0f GB"
                         % (commit_room_gb(), time.time() - waited, ROOM_GB))
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    name = "motor_encoder_asked_change_registry.py run"
    if not hasattr(tr, "finish_job"):
        # production's registry module as it was before Oct 5, 2026: status and last_run only
        tr.job_started(row_id, started=t0, log=str(log), runner=name, path=REGISTRY)
        with open(log, "w", encoding="utf-8") as stream:
            code = capped(cmd, stream=stream, timeout_s=limit)
            stream.write("\n" + json.dumps(capped.last) + "\n")
        tr.job_finished(row_id, exit_code=code, started=t0, ended=time.time(), log=str(log), runner=name,
                        path=REGISTRY)
        print("job %s exit %s after %.0f s; log %s" % (row_id, code, time.time() - t0, log))
        return code
    # THE RUNNER'S RECORD OF OCT 5, 2026 (the recovery directive's section 8; mimo_bridge/tools/test_registry.py): the
    # start is written before the command exists, with the commit of this file's repository and this runner's
    # process; the command's processes carry the row's two marks in their environment (capped() builds the command's
    # environment from this process's); the end is PASS, FAIL, INVALID or STOPPED with the next action.
    import os
    run_id = tr.new_run_id(row_id, t0)
    why = tr.job_started(row_id, started=t0, log=str(log), runner=name, path=REGISTRY, run_id=run_id,
                         command=cmd, cwd=HERE, checkpoint=next((cmd[i + 1] for i, x in enumerate(cmd[:-1])
                                                                 if x in ("--checkpoint", "--resume")), None))
    if why:
        raise SystemExit("refused: the start could not be written into the row, so the job was not started: %s" % why)
    os.environ.update(UEMA_TEST_ROW=row_id, UEMA_TEST_RUN=run_id)
    try:
        with open(log, "w", encoding="utf-8") as stream:
            code = capped(cmd, stream=stream, timeout_s=limit,
                          on_start=lambda pid: tr.job_process(row_id, pid, path=REGISTRY, run_id=run_id))
            stream.write("\n" + json.dumps(capped.last) + "\n")
    except SystemExit as exc:
        tr.finish_job(row_id, started=t0, log=str(log), runner=name, path=REGISTRY, run_id=run_id,
                      not_started="not started: %s" % exc)
        raise
    finally:
        os.environ.pop("UEMA_TEST_ROW", None)
        os.environ.pop("UEMA_TEST_RUN", None)
    stopped = "timed out after %s s (the runner ended the job's processes)" % limit if code == 124 else None
    why, state = tr.finish_job(row_id, exit_code=code, started=t0, ended=time.time(), log=str(log), runner=name,
                               path=REGISTRY, run_id=run_id, stopped=stopped)
    print("job %s: %s (exit %s) after %.0f s; log %s%s" % (row_id, state, code, time.time() - t0, log,
                                                         "" if why is None else "; THE END WAS NOT WRITTEN: %s" % why))
    return code


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit(__doc__)
    if argv[0] == "check":
        return check(argv[1:])
    if argv[0] == "add":
        return add(argv[1:])
    if argv[0] == "set":
        return set_fields(argv[1], json.loads(Path(argv[2]).read_text(encoding="utf-8")))
    if argv[0] == "run":
        split = argv.index("--")
        return run(argv[1], argv[2], argv[split + 1:])
    if argv[0] == "capped":
        # a development run under the same limits, without a row: PY ... capped -- CMD...
        return capped(argv[argv.index("--") + 1:])
    if argv[0] == "room":
        print("commit room %.1f GB" % commit_room_gb())
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
