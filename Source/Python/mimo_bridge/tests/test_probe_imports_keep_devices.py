# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Importing a module never changes which graphics cards the process can see (lane D, 2026-09-26).

Eight probe modules ran os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1") when imported. A test process
that collected any probe test lost the card from then on: production's whole-folder runs skipped the card
continuation test (test_self_continuation[cuda:0]) and the card liveness test, so a card defect stayed
hidden. The hide now runs only when a probe runs as a program, before torch loads, as it always did then.
Record: docs/gain_of_function_v4/agents/probe_gpu_hiding_2026-09-26.md
"""
import ast
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

SRC_PY = Path(__file__).resolve().parents[2]
REPO = SRC_PY.parents[1]
KEY = "CUDA_VISIBLE_DEVICES"
# the harness first, so each module's own top-level code runs at its own import
PROBES = ["mimo_bridge.tools.gof_probe_harness", "mimo_bridge.tools.gof_probe_binding",
          "mimo_bridge.tools.gof_probe_body", "mimo_bridge.tools.gof_probe_gaze",
          "mimo_bridge.tools.gof_probe_posture", "mimo_bridge.tools.gof_probe_suite",
          "mimo_bridge.tools.gof_probe_watch", "mimo_bridge.tools.liveness_probe"]
SCANNED = [SRC_PY / "mimo_bridge", SRC_PY / "lem", SRC_PY / "tokenizer", REPO / "docs"]


def _scanned_files():
    """Every .py file under SCANNED, except inside a pytest temporary root (a folder named pytest_temp_* or pytest-of-*):
    such a root holds copies of the repository a test made, not programs (Oct 11, 2026: the explainer completion run of
    Oct 9 left docs/explainer_completion_2026-10-09/pytest_temp_61d11a25, whose copies failed both scans)."""
    return [p for root in SCANNED if root.is_dir() for p in sorted(root.rglob("*.py"))
            if not any(part.startswith(("pytest_temp_", "pytest-of-")) for part in p.relative_to(root).parts[:-1])]


def _is_main_block(node):
    t = node.test
    return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in t.comparators))


def _dotted(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _names_key(node):
    return (any(isinstance(n, ast.Constant) and n.value == KEY for n in ast.walk(node))
            or any(isinstance(n, ast.keyword) and n.arg == KEY for n in ast.walk(node)))


def import_time_device_changes(source):
    """[(line, what)] for each statement that runs at import - outside every function and outside the
    `if __name__ == "__main__":` block - and changes the cards the process sees or uses: a write of
    CUDA_VISIBLE_DEVICES into os.environ (assignment, del, setdefault, pop, update, putenv, |=) or a call of
    torch.cuda.set_device / torch.set_default_device. Building a child's environment
    (dict(os.environ, CUDA_VISIBLE_DEVICES=...)) changes nothing in this process and is not counted."""
    found = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.If) and _is_main_block(node):
            for n in node.orelse:
                visit(n)
            return
        if isinstance(node, (ast.Assign, ast.Delete)):
            for t in node.targets:
                if isinstance(t, ast.Subscript) and _dotted(t.value).endswith("environ") and _names_key(t.slice):
                    found.append((node.lineno, "os.environ[%s] written" % KEY))
        elif isinstance(node, ast.AugAssign) and _dotted(node.target).endswith("environ") and _names_key(node.value):
            found.append((node.lineno, "os.environ |= %s" % KEY))
        elif isinstance(node, ast.Call):
            name = _dotted(node.func)
            last = name.rsplit(".", 1)[-1]
            writes_environ = "environ" in name.split(".") and last in (
                "setdefault", "pop", "update", "__setitem__", "__delitem__")
            if (writes_environ or last == "putenv") and _names_key(node):
                found.append((node.lineno, name))
            elif name in ("torch.cuda.set_device", "torch.set_default_device"):
                found.append((node.lineno, name))
        for child in ast.iter_child_nodes(node):
            visit(child)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # old scripts' invalid escape sequences
        visit(ast.parse(source))
    return found


def test_the_detector_finds_an_import_time_hide_and_passes_a_program_s_own():
    planted = {
        'import os\nos.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")\n': 1,       # the old probe header
        'import os\nos.environ["CUDA_VISIBLE_DEVICES"] = "-1"\n': 1,
        'import os\nos.environ.pop("CUDA_VISIBLE_DEVICES", None)\n': 1,
        'import os\nos.environ.update(CUDA_VISIBLE_DEVICES="-1")\n': 1,
        'import os\ntry:\n    del os.environ["CUDA_VISIBLE_DEVICES"]\nexcept KeyError:\n    pass\n': 1,
        'import torch\ntorch.cuda.set_device(0)\n': 1,
        'class A:\n    import os\n    os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n': 1,   # a class body runs at import
        'import os\nif __name__ == "__main__":\n    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")\n': 0,
        'import os\ndef main():\n    os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n': 0,
        'import os\nenv = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")\n': 0,
        'import os\nos.environ.setdefault("PYTEST_DEBUG_TEMPROOT", "D:/x")\n': 0,
    }
    for source, expected in planted.items():
        assert len(import_time_device_changes(source)) == expected, source


def test_no_module_changes_the_visible_cards_when_imported():
    offenders = []
    for path in _scanned_files():                 # a sparse worktree may hold no docs
        try:
            found = import_time_device_changes(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        offenders += ["%s:%d %s" % (path.relative_to(REPO).as_posix(), line, what) for line, what in found]
    assert not offenders, ("importing these changes the process's cards; move each into its program's "
                           "`if __name__ == \"__main__\":` block:\n" + "\n".join(offenders))


# A PROGRAM THAT REACHED A PROBE GOT ITS HIDE FROM THAT IMPORT (the fix's review, 2026-09-26). Before this fix, a
# script that loaded a probe - itself, through another module, or by loading another script by its file path, as
# persistent_memory_repair_measurements.py loads vision_carrier_measurements.py - was kept off the card by the probe's
# import, and six such scripts lost the hide unseen. Each program that can load a probe now hides the card itself, in
# a module-level `if __name__ == "__main__":` block placed before its first import of torch or of our code, or it is
# named here with the reason it keeps the card visible. A name whose file this tree lacks (another branch's) is
# skipped. The check follows imports inside functions too, so a program whose probe code never runs still needs one.
_LAUNCHED = ("resumes training: its trainer and supervisor start through deploy_qualified_bootstrap.launch, which "
             "removes the variable from each child's environment")
_PROVES_ON_CARD = "sets the variable to 0 itself and proves a trainer on cuda:0"
KEEPS_THE_CARD = {
    "docs/gain_of_function_v3/deploy_qualified_bootstrap.py":
        "starts trainers: launch() removes the variable from each child's environment",
    "docs/gain_of_function_v3/prove_takeover_restart.py":
        "starts a trainer whose variable it sets from --device (removed for cuda, -1 for the CPU)",
    "docs/gain_of_function_v4/prove_observer.py":
        "starts trainers whose variable it sets from their device (removed for cuda, -1 for the CPU)",
    "docs/gain_of_function_v3/check_liveness_roundtrip.py": _PROVES_ON_CARD,
    "docs/gain_of_function_v3/prove_liveness_retention.py": _PROVES_ON_CARD + ", and checks its model is on the card",
    "docs/gain_of_function_v3/prove_motor_repair.py": _PROVES_ON_CARD,
    "docs/gain_of_function_v3/prove_planner_repair.py": _PROVES_ON_CARD + ", and checks its model is on the card",
    "docs/gain_of_function_v3/qualify_control_repairs_2026_09_24.py":
        "proves trainers on cuda:0 and removes the variable from each child's environment",
    "docs/gain_of_function_v3/resume_after_health_repair.py": _LAUNCHED,
    "docs/gain_of_function_v3/resume_after_motor_repair.py": _LAUNCHED,
    "docs/gain_of_function_v3/resume_after_planner_repair.py": _LAUNCHED,
    "docs/gain_of_function_v3/resume_control_repairs_2026_09_24.py": _LAUNCHED,
    "docs/gain_of_function_v3/refresh_takeover_readers.py":
        "starts its readers through deploy_qualified_bootstrap.launch, which removes the variable from each child's "
        "environment; it never imports torch",
    "docs/gain_of_function_v3/verify_visual_activation_precision.py":
        "clears the variable at its first line: it checks the card",
    "docs/gain_of_function_v3/verify_visual_activation_roundtrip.py":
        "checks the card when one is visible, as its code intends",
    "docs/gain_of_function_v4/persistent_memory_card_proof.py":
        "reads the memory on the card on purpose (it starts CUDA in main() before its first harness import)",
    "docs/gain_of_function_v4/remembered_hand_learning_2026-10-11/common_starts_arm.py":
        "runs the common-start comparison's arm on the card on purpose (rows rhl_common_starts_*, --device gpu); its "
        "known saved result at 481,800 was reproduced on the card",
    "docs/gain_of_function_v4/remembered_hand_learning_2026-10-11/common_starts_trial_trainer.py":
        "the arm's trial trainer: trains on the card on purpose, as production training does (rows rhl_common_starts_*)",
    "docs/gain_of_function_v4/vestibular_encoder_measurements.py":
        "its --device cuda reading is the one its deploy decision rests on (fix/vestibular-encoder-2026-09-25); a "
        "hide would turn --device cuda into a CPU run with no error, because the model falls back to the CPU",
}
LOADERS = ("spec_from_file_location", "run_path", "SourceFileLoader", "load_source", "exec_module")
IMPORTERS = ("import_module", "run_module", "__import__")


def _parse(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # old scripts' invalid escape sequences
        return ast.parse(path.read_text(encoding="utf-8-sig"))


def _import_time_nodes(tree):
    """Every node that runs when the file runs, outside functions and outside `if __name__ == "__main__":`."""
    out = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            return
        if isinstance(node, ast.If) and _is_main_block(node):
            for n in node.orelse:
                visit(n)
            return
        out.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out


def _module_files(dotted, here, src):
    """The files an import of `dotted` runs: its packages and module under `src`, or a script beside `here`."""
    parts = [p for p in dotted.split(".") if p]
    found = []
    for i in range(1, len(parts) + 1):
        package, module = src.joinpath(*parts[:i], "__init__.py"), src.joinpath(*parts[:i - 1], parts[i - 1] + ".py")
        found += [f for f in (package, module) if f.is_file()]
    if len(parts) == 1 and (here / (parts[0] + ".py")).is_file():
        found.append(here / (parts[0] + ".py"))
    return found


def _node_loads(node, path, src):
    """The files one import statement, or one import_module / run_module call of a constant name, runs."""
    here = path.parent
    if isinstance(node, ast.Import):
        return [f for a in node.names for f in _module_files(a.name, here, src)]
    if isinstance(node, ast.ImportFrom):
        base = node.module or ""
        if node.level:
            try:
                package = list(path.relative_to(src).parent.parts)
            except ValueError:
                return []
            package = package[:len(package) - (node.level - 1)]
            base = ".".join(package + ([base] if base else []))
        names = [base + "." + a.name if base else a.name for a in node.names]
        return _module_files(base, here, src) + [f for name in names for f in _module_files(name, here, src)]
    if (isinstance(node, ast.Call) and _dotted(node.func).rsplit(".", 1)[-1] in IMPORTERS and node.args
            and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
        return _module_files(node.args[0].value, here, src)
    return []


def _names_torch(node):
    names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
        [node.module or ""] if isinstance(node, ast.ImportFrom) and not node.level else []
    return any(n.split(".")[0] == "torch" for n in names)


def _loads(path, tree, src, by_name):
    """The files `path` can load when it runs: its imports anywhere in the file and, when it loads scripts by file
    path, every string naming a script beside it (a bare name or a name.py) or a .py file found once in the tree."""
    here = path.parent
    out = set()
    for node in ast.walk(tree):
        out.update(_node_loads(node, path, src))
    if any(isinstance(n, ast.Call) and _dotted(n.func).rsplit(".", 1)[-1] in LOADERS for n in ast.walk(tree)):
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Constant) and isinstance(n.value, str)) or "\n" in n.value:
                continue
            if n.value.endswith(".py"):
                name = n.value.replace("\\", "/").rsplit("/", 1)[-1]
                if name and (here / name).is_file():
                    out.add(here / name)
                elif len(by_name.get(name, ())) == 1:
                    out.add(by_name[name][0])
            elif n.value.isidentifier() and (here / (n.value + ".py")).is_file():
                out.add(here / (n.value + ".py"))
    out.discard(path)
    return out


def _sets_minus_one(node):
    if isinstance(node, ast.Call) and _dotted(node.func).endswith("environ.setdefault") and len(node.args) == 2:
        return [a.value if isinstance(a, ast.Constant) else None for a in node.args] == [KEY, "-1"]
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value == "-1":
        return any(isinstance(t, ast.Subscript) and _dotted(t.value).endswith("environ") and _names_key(t.slice)
                   for t in node.targets)
    return False


def _hides_before_loading(path, tree, src):
    """A module-level `if __name__ == "__main__":` block sets the variable to -1 before the file's first import of
    torch or of our code (the hide must run before torch loads)."""
    hide = min((n.lineno for n in tree.body if isinstance(n, ast.If) and _is_main_block(n)
                and any(_sets_minus_one(m) for m in ast.walk(n))), default=None)
    first = min((n.lineno for n in _import_time_nodes(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                 and (_names_torch(n) or _node_loads(n, path, src))), default=None)
    return hide is not None and (first is None or hide < first)


def programs_relying_on_a_probe_hide(files, probes, src, repo, card_users):
    """[(program, probe it reaches)] for each program that can load a probe and neither hides the card itself before
    loading torch or our code nor is named in `card_users`. A program is any file under repo/docs (a script: run as
    a program, even one whose work sits at its top level with no `__main__` block, or loaded by one) and any other
    file with a module-level `if __name__ == "__main__":` block; test files are left out."""
    trees = {}
    for f in files:
        try:
            trees[f] = _parse(f)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
    by_name = {}
    for f in trees:
        by_name.setdefault(f.name, []).append(f)
    loaded_by = {}
    for f, tree in trees.items():
        for g in _loads(f, tree, src, by_name):
            loaded_by.setdefault(g, set()).add(f)
    reach = {p: p for p in probes if p in trees}
    queue = list(reach)
    while queue:
        g = queue.pop()
        for f in loaded_by.get(g, ()):
            if f not in reach:
                reach[f] = reach[g]
                queue.append(f)
    found = []
    for f in sorted(reach):
        rel = f.relative_to(repo).as_posix()
        if f.name.startswith("test_") or f.name == "conftest.py" or rel in card_users:
            continue
        if not (f.is_relative_to(repo / "docs") or any(isinstance(n, ast.If) and _is_main_block(n)
                                                       for n in trees[f].body)):
            continue                                   # a module others import, not a program
        if not _hides_before_loading(f, trees[f], src):
            found.append((rel, reach[f].relative_to(repo).as_posix()))
    return found


def test_the_reach_check_finds_a_program_that_relied_on_a_probe_s_hide(tmp_path):
    src, docs = tmp_path / "src", tmp_path / "docs"
    (src / "pkg").mkdir(parents=True)
    docs.mkdir()
    (src / "pkg" / "__init__.py").write_text("")
    probe = src / "pkg" / "probe.py"
    hide = 'import os\nif __name__ == "__main__":\n    os.environ.setdefault("%s", "-1")\n' % KEY
    probe.write_text(hide + "import torch\n")
    main = "if __name__ == '__main__':\n    pass\n"
    load = ("import importlib.util\ndef load(p):\n    s = importlib.util.spec_from_file_location('m', p)\n"
            "    s.loader.exec_module(importlib.util.module_from_spec(s))\n")
    cases = {                                           # file: (source, found)
        "docs/imports.py": ("from pkg import probe\n" + main, True),         # imports the probe, no hide
        "docs/carrier.py": ("import pkg.probe\n", True),                     # a script with its work at top level
        "docs/by_path.py": (load + "if __name__ == '__main__':\n    load('x/carrier.py')\n", True),   # by path
        "docs/by_name.py": (load + "if __name__ == '__main__':\n    load('carrier' + '.py')\n", True),  # bare name
        "docs/own_hide.py": (hide + "from pkg import probe\n" + main, False),        # hides first: passes
        "docs/late_hide.py": ("import torch\n" + hide + "from pkg import probe\n", True),  # after torch loads
        "docs/ours_first.py": ("import pkg\n" + hide + "from pkg import probe\n", True),   # after our code loads
        "docs/card_user.py": ("from pkg import probe\n" + main, False),      # named with its reason
        "docs/unrelated.py": ("import json\n" + main, False),                # never reaches a probe
        "docs/test_probe_use.py": ("from pkg import probe\n" + main, False),  # a test: its session's rules apply
        "src/pkg/helper.py": ("from pkg import probe\n", False),             # a module others import
        "src/pkg/tool.py": ("from pkg import probe\n" + main, True),         # a program in the source tree
    }
    for name, (source, _) in cases.items():
        (tmp_path / name).write_text(source)
    files = sorted(src.rglob("*.py")) + sorted(docs.rglob("*.py"))
    found = programs_relying_on_a_probe_hide(files, [probe], src, tmp_path, {"docs/card_user.py": "a stand-in"})
    assert sorted(program for program, _ in found) == sorted(n for n, (_, bad) in cases.items() if bad)
    assert {reached for _, reached in found} == {"src/pkg/probe.py"}


def test_every_program_that_reaches_a_probe_hides_the_card_itself():
    files = _scanned_files()
    probes = [SRC_PY.joinpath(*name.split(".")).with_suffix(".py") for name in PROBES]
    found = programs_relying_on_a_probe_hide(files, probes, SRC_PY, REPO, KEEPS_THE_CARD)
    assert not found, ("these programs load a probe, whose import no longer hides the card; hide it in each "
                       "program's own `if __name__ == \"__main__\":` block before torch loads, or name the program in "
                       "KEEPS_THE_CARD with the reason it keeps the card:\n"
                       + "\n".join("%s (reaches %s)" % row for row in found))


def _child(code, *args, timeout=600):
    env = {k: v for k, v in os.environ.items() if k != KEY}
    env["PYTHONPATH"] = os.pathsep.join([str(SRC_PY)] + [p for p in [env.get("PYTHONPATH")] if p])
    r = subprocess.run([sys.executable, "-c", code, *args], cwd=str(SRC_PY), env=env, capture_output=True,
                       text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


IMPORT_EACH = r'''
import importlib, json, os, sys
import torch
key = "CUDA_VISIBLE_DEVICES"
rows = []
for name in json.loads(sys.argv[1]):
    before = [os.environ.get(key), torch.cuda.device_count()]
    module = importlib.import_module(name)
    after = [os.environ.get(key), torch.cuda.device_count()]
    rows.append(dict(module=name, file=module.__file__, before=before, after=after))
    if os.environ.get(key) != before[0]:           # undo, so the next module is judged on its own
        if before[0] is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = before[0]
print("ROWS=" + json.dumps(rows))
'''


def test_importing_each_probe_leaves_the_visible_cards_unchanged():
    """A fresh process with the card variable unset imports each probe module; after each import the
    variable and torch's count of visible cards (read through NVML, which follows the variable) are
    what they were before it."""
    code, out, err = _child(IMPORT_EACH, json.dumps(PROBES))
    assert code == 0, err[-3000:]
    rows = json.loads(out.split("ROWS=", 1)[1])
    assert [r["module"] for r in rows] == PROBES
    for r in rows:
        assert Path(r["file"]).resolve().is_relative_to(SRC_PY), r     # this tree's code, not another's
    changed = ["%s: variable %r -> %r, cards %d -> %d" % (r["module"], r["before"][0], r["after"][0],
                                                          r["before"][1], r["after"][1])
               for r in rows if r["after"] != r["before"]]
    assert not changed, "importing a probe changed the process's cards:\n" + "\n".join(changed)


RUN_AS_PROGRAM = r'''
import os, runpy, sys
key = "CUDA_VISIBLE_DEVICES"
def hook(event, args):
    if event == "import" and args and args[0] == "torch":
        sys.stdout.write("AT_TORCH=%r\n" % (os.environ.get(key),))
        sys.stdout.flush()
        os._exit(0)                                   # stop before torch loads: nothing of the probe runs
sys.addaudithook(hook)
name = sys.argv[1]
sys.argv = [name, "--help"]
try:
    runpy.run_module(name, run_name="__main__", alter_sys=True)     # what python -m does
finally:
    sys.stdout.write("NO_TORCH=%r\n" % (os.environ.get(key),))
'''


def test_each_probe_run_as_a_program_still_hides_the_card_before_torch_loads():
    """Run as a program (python -m, as the scheduler, the probe watcher and the pause jobs start them),
    each probe still hides the card before torch is first imported, as before 2026-09-26. A probe that
    imports torch only inside its functions (liveness_probe) prints its --help first: the variable then
    reads -1 at exit, which only the hide at its top can have written."""
    for name in PROBES:
        code, out, err = _child(RUN_AS_PROGRAM, name, timeout=300)
        assert "AT_TORCH='-1'" in out or "NO_TORCH='-1'" in out, (name, code, out[-500:], err[-2000:])
