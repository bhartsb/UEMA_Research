# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The component manifest (tools/component_manifest.py): the check that fails on unmapped model code and on mapped
code changed after its entry was confirmed, last code change read from git at the level of one function, QA STALE, the
switch state, and the three places the check is wired into (this file, the relaunch script, the API).

The operator's recovery directive of Oct 5, 2026, sections 7C and 9. Every repository here is a temporary git
repository of a few small files, except the two tests at the end that read this checkout's own manifest: the first of
them is THE WIRING OF THE CHECK INTO THE UNIT TESTS (a model code file mapped to no component, or mapped code changed
without its entry being confirmed again, fails this test run).
"""
import ast
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

from mimo_bridge.tools import component_manifest as cm

REPO = Path(__file__).resolve().parents[4]
SRC = 'Source/Python/'
TRAINER = SRC + 'mimo_bridge/train_mimo.py'
MODEL = SRC + 'mimo_bridge/lem_model.py'
HEADS = SRC + 'lem/heads.py'

MODEL_TEXT = '''"""A small model file with several parts in one class."""
from lem.heads import Head


class Lite:
    def __init__(self, width):
        self.head = Head(width)

    def step(
        self,
        frames,
    ):
        text = """
def step(self): this line sits at column 0 inside a string
"""
        return self.gate(frames), text

    @staticmethod
    def gate(tokens):
        return [0.5 * t for t in tokens]

    def plan(self, goal):
        def inner(x):
            return x - goal
        return inner
'''


def git(root, *args, when=None):
    env = None
    if when is not None:                  # the commit's own time: what `git log --format=%ct` prints
        env = dict(os.environ, GIT_COMMITTER_DATE='@%d +0000' % when, GIT_AUTHOR_DATE='@%d +0000' % when)
    done = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True, env=env)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def write(root, path, text):
    target = Path(root)/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8', newline='\n')


def commit(root, message, when=None):
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', message, when=when)
    return git(root, 'rev-parse', 'HEAD')


def component(cid, paths, kind='model part', **more):
    entry = dict(id=cid, kind=kind, name=cid.lower().replace('_', ' '), does='Takes x and outputs y.', paths=paths,
                 blocks=[], flags=[], aliases=[],
                 confirmed=dict(at='never', by='nobody', commit='none', content='none', basis='not yet'))
    entry.update(more)
    return entry


def make_repo(tmp_path):
    """A temporary git repository: a trainer that imports a model file with four parts in one class and a heads
    file, and a manifest of four components whose entries are confirmed against that code."""
    cm.forget()
    root = tmp_path/'repo'
    root.mkdir()
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'test@example.com')
    git(root, 'config', 'user.name', 'A Test')
    git(root, 'config', 'core.autocrlf', 'false')
    write(root, SRC + 'mimo_bridge/__init__.py', '')
    write(root, SRC + 'lem/__init__.py', '')
    write(root, TRAINER, 'from mimo_bridge.lem_model import Lite\n\n\ndef main():\n    return Lite(4)\n')
    write(root, MODEL, MODEL_TEXT)
    write(root, HEADS, 'class Head:\n    def __init__(self, width):\n        self.width = width\n')
    manifest = dict(about='a test manifest', schema=1, scope=dict(entry='mimo_bridge.train_mimo', also=[], outside=[]),
                    components=[
                        component('TRAINER', [dict(file=TRAINER), dict(file=SRC + 'mimo_bridge/__init__.py'),
                                              dict(file=SRC + 'lem/__init__.py'),
                                              dict(file=MODEL, functions=['Lite.__init__', 'Lite.step', cm.REST])], kind='tool'),
                        component('TOP_DOWN_GATE', [dict(file=MODEL, functions=['Lite.gate'])]),
                        component('PLANNER_SEARCH', [dict(file=MODEL, functions=['Lite.plan'])]),
                        component('PREDICTION_HEADS', [dict(file=HEADS)])])
    cm.save(manifest, root)
    commit(root, 'the model and its manifest', when=1790000000)
    cm.confirm(['TRAINER', 'TOP_DOWN_GATE', 'PLANNER_SEARCH', 'PREDICTION_HEADS'], 'a test', 'checked against the code',
               root)
    commit(root, 'the entries confirmed', when=1790000100)
    cm.forget()
    return root


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path)


def edit(root, path, old, new):
    target = Path(root)/path
    text = target.read_text(encoding='utf-8')
    assert old in text
    target.write_text(text.replace(old, new), encoding='utf-8', newline='\n')
    cm.forget()


# ------------------------------------------------------------------------------------------ the mapped code's text
def test_a_definitions_text_is_found_as_the_parser_finds_it():
    lines = MODEL_TEXT.split('\n')
    tree = ast.parse(MODEL_TEXT)
    lite = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    for node in lite.body:
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        assert cm.blocks_named(MODEL_TEXT, 'Lite.' + node.name) == ['\n'.join(lines[first - 1:node.end_lineno])], node.name
    assert cm.blocks_named(MODEL_TEXT, 'Lite')[0].startswith('class Lite:') and cm.blocks_named(MODEL_TEXT, 'inner') == []
    assert cm.blocks_named(MODEL_TEXT, 'Lite.inner') == [], 'a function nested inside a method is not the class\'s own'
    assert cm.blocks_named(MODEL_TEXT, 'Lite.step')[0].count('\n') == 7, 'the signature\'s dedented ) does not end it'
    twice = 'class A:\n    @property\n    def w(self):\n        return 1\n\n    @w.setter\n    def w(self, v):\n        pass\n'
    assert len(cm.blocks_named(twice, 'A.w')) == 2 and cm.part_text(twice, 'A.w').count('def w') == 2


def test_every_function_the_real_manifest_maps_is_found_as_the_parser_finds_it():
    """The extractor reads by indentation; Python's own parser is the reference. Trailing comment lines after a
    definition may differ (the extractor keeps a comment that follows the body at a deeper indentation)."""
    data = cm.load(REPO)
    checked = 0
    for entry in data['components']:
        if entry.get('ref'):
            continue
        for path in entry['paths']:
            if not path.get('functions'):
                continue
            text = cm.file_now(REPO, path['file'])
            tree = ast.parse(text)
            lines = text.split('\n')
            truth = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    truth.setdefault(node.name, []).append(node)
                    if isinstance(node, ast.ClassDef):
                        for sub in node.body:
                            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                truth.setdefault(node.name + '.' + sub.name, []).append(sub)
            for name in [n for n in path['functions'] if n != cm.REST]:     # REST is no definition
                want = ['\n'.join(lines[min([n.lineno] + [d.lineno for d in n.decorator_list]) - 1:n.end_lineno])
                        for n in truth.get(name, [])]
                got = cm.blocks_named(text, name)
                assert len(got) == len(want) and want, (entry['id'], path['file'], name)
                for g, w in zip(got, want):
                    assert g.startswith(w), (entry['id'], name)
                    assert all(not line.strip() or line.lstrip().startswith('#') for line in g[len(w):].split('\n'))
                checked += 1
    assert checked > 150, 'the manifest maps %d functions' % checked


# ------------------------------------------------------------------------------------------ the check
def test_a_fresh_manifest_passes_and_an_edit_of_a_mapped_function_fails_only_its_component(repo):
    assert cm.check(repo)['ok'] and cm.refusal(repo) is None
    edit(repo, MODEL, 'return [0.5 * t for t in tokens]', 'return [0.4 * t for t in tokens]')
    found = cm.check(repo)
    assert found['stale'] == ['TOP_DOWN_GATE'] and not found['ok'] and not found['unmapped']
    assert 'TOP_DOWN_GATE: its mapped code changed after its entry was confirmed' in found['problems'][0]
    assert MODEL + '::Lite.gate' in found['problems'][0], 'the finding names the function that changed'
    assert 'component_manifest confirm TOP_DOWN_GATE' in found['problems'][0]
    assert cm.refusal(repo).startswith('the component manifest is not fresh (1 finding): TOP_DOWN_GATE')
    # confirming the entry against the changed code clears the finding, whether or not the edit is committed yet
    assert cm.confirm(['TOP_DOWN_GATE'], 'a reader', 'checked against the code', repo) == ['TOP_DOWN_GATE']
    cm.forget()
    assert cm.check(repo)['ok']
    stamped = cm.components(cm.load(repo))['TOP_DOWN_GATE']['confirmed']
    assert stamped['by'] == 'a reader' and stamped['basis'] == 'checked against the code' and len(stamped['content']) == 64


def test_a_whole_file_mapping_fails_on_any_edit_of_the_file(repo):
    edit(repo, HEADS, 'self.width = width', 'self.width = 2 * width')
    assert cm.check(repo)['stale'] == ['PREDICTION_HEADS']


def test_a_model_code_file_mapped_to_no_component_fails_the_check(repo):
    write(repo, SRC + 'lem/new_path.py', 'def extra(x):\n    return x\n')
    edit(repo, HEADS, 'class Head:', 'from lem.new_path import extra\n\n\nclass Head:')
    cm.confirm(['PREDICTION_HEADS'], 'a reader', 'checked against the code', repo)
    cm.forget()
    found = cm.check(repo)
    assert found['unmapped'] == [SRC + 'lem/new_path.py'] and not found['stale']
    assert found['problems'] == ['model code file %slem/new_path.py is mapped to no component' % SRC]
    # a file no trainer import reaches is not model code
    write(repo, SRC + 'lem/not_imported.py', 'x = 1\n')
    cm.forget()
    assert cm.check(repo)['unmapped'] == [SRC + 'lem/new_path.py']
    # scope.also names files outside the import closure; scope.outside exempts a file with its reason
    data = cm.load(repo)
    data['scope']['also'] = ['docs/tool.py']
    data['scope']['outside'] = [dict(path=SRC + 'lem/new_path.py', why='a test exemption')]
    cm.save(data, repo)
    cm.forget()
    assert cm.check(repo)['unmapped'] == ['docs/tool.py']


def test_a_malformed_manifest_is_named(repo):
    data = cm.load(repo)
    data['components'][1]['paths'] = [dict(file=MODEL, functions=['Lite.no_such_method'])]
    data['components'][2]['kind'] = 'thing'
    data['components'][3]['aliases'] = ['top down gate']
    data['components'].append(component('planner_extra', [dict(file=SRC + 'lem/missing.py')]))
    data['components'].append(component('HEAD_TILT_REPAIR', [], kind='repair'))
    found = cm.structure_problems(repo, data)
    assert '%s holds no definition named Lite.no_such_method' % MODEL in ' '.join(found)
    assert 'PLANNER_SEARCH: kind is not one of model part, world, repair, tool' in found
    assert "PREDICTION_HEADS: its alias 'top down gate' is a canonical name" in found
    assert 'planner_extra: the id is not upper case words joined by underscores' in found
    assert 'planner_extra: the mapped file %slem/missing.py does not exist' % SRC in found
    assert 'HEAD_TILT_REPAIR: a repair names its switch {flag, on}' in found
    cm.manifest_path(repo).write_text('{"components": ', encoding='utf-8')
    assert 'cannot be read' in cm.check(repo)['problems'][0] and cm.refusal(repo) is not None
    assert cm.aliases(repo) == []


# ------------------------------------------------------------------------------------------ last code change
def test_the_last_code_change_is_read_per_function(repo, tmp_path, monkeypatch):
    monkeypatch.setenv('UEMA_COMPONENT_CACHE', str(tmp_path/'cache'))
    edit(repo, MODEL, 'return [0.5 * t for t in tokens]', 'return [0.4 * t for t in tokens]')
    gate_commit = commit(repo, 'the gate mixes 0.4', when=1790001000)
    edit(repo, MODEL, 'return x - goal', 'return goal - x')
    plan_commit = commit(repo, 'the plan flips its sign', when=1790002000)
    data = cm.components(cm.load(repo))
    gate = cm.last_changes(repo, data['TOP_DOWN_GATE'])
    assert gate['commit'] == gate_commit[:9] and gate['time'] == 1790001000 and gate['part'] == MODEL + '::Lite.gate'
    assert gate['time_text'] == cm.tr.clock(1790001000) and gate['uncommitted'] is False
    plan = cm.last_changes(repo, data['PLANNER_SEARCH'])
    assert plan['commit'] == plan_commit[:9] and plan['subject'] == 'the plan flips its sign'
    heads = cm.last_changes(repo, data['PREDICTION_HEADS'])
    assert heads['time'] == 1790000000 and heads['part'] == HEADS, 'a whole-file part: the file\'s newest commit'
    # an edit not yet committed is reported beside the last commit
    edit(repo, MODEL, 'return goal - x', 'return 2 * (goal - x)')
    cm.forget()
    assert cm.last_changes(repo, data['PLANNER_SEARCH'])['uncommitted'] is True
    assert cm.last_changes(repo, data['TOP_DOWN_GATE'])['uncommitted'] is False
    assert cm.last_changes(repo, component('PLANNER_SEARCH_REPAIR', [], kind='repair'))['searched'] == \
        'the entry maps no code yet'


def test_a_merge_that_changes_a_function_itself_is_its_last_change(repo, tmp_path, monkeypatch):
    monkeypatch.setenv('UEMA_COMPONENT_CACHE', str(tmp_path/'cache'))
    git(repo, 'checkout', '-q', '-b', 'side')
    edit(repo, MODEL, 'return [0.5 * t for t in tokens]', 'return [0.3 * t for t in tokens]')
    side = commit(repo, 'side: 0.3', when=1790003000)
    git(repo, 'checkout', '-q', '-')
    edit(repo, MODEL, 'self.head = Head(width)', 'self.head = Head(width + 1)')
    commit(repo, 'main: a wider head', when=1790004000)
    git(repo, 'merge', '-q', '--no-ff', '-m', 'merge side', 'side', when=1790005000)
    cm.forget()
    gate = cm.last_changes(repo, cm.components(cm.load(repo))['TOP_DOWN_GATE'])
    assert gate['commit'] == side[:9], 'the commit that made the edit, not the merge that brought it in'


# ------------------------------------------------------------------------------------------ independent QA
def test_qa_is_recorded_only_with_its_commit_and_goes_stale_when_the_reviewed_code_changes(repo):
    reviewed = git(repo, 'rev-parse', 'HEAD')
    entry = cm.components(cm.load(repo))['TOP_DOWN_GATE']
    assert cm.last_qa(repo, entry) is None, 'no independent QA on record'
    git(repo, 'checkout', '-q', '-b', 'review/gate')
    write(repo, 'docs/review_gate.md', 'Reviewed: the gate at commit %s. Verdict: sound.\n' % reviewed[:9])
    commit(repo, 'the review of the gate', when=1790005000)
    git(repo, 'checkout', '-q', '-')
    with pytest.raises(cm.ManifestError, match='is not in this repository'):
        cm.record_qa('TOP_DOWN_GATE', 'ffffffffffffffffffffffffffffffffffffffff', 'review/gate', 'docs/review_gate.md',
                     'sound', 'a test', repo)
    with pytest.raises(cm.ManifestError, match='is not a file at the tip of review/gate'):
        cm.record_qa('TOP_DOWN_GATE', reviewed, 'review/gate', 'docs/no_such_review.md', 'sound', 'a test', repo)
    older = git(repo, 'rev-parse', 'HEAD~1')
    with pytest.raises(cm.ManifestError, match='does not name the reviewed commit'):
        cm.record_qa('TOP_DOWN_GATE', older, 'review/gate', 'docs/review_gate.md', 'sound', 'a test', repo)
    with pytest.raises(cm.ManifestError, match='no component has id'):
        cm.record_qa('GATE', reviewed, 'review/gate', 'docs/review_gate.md', 'sound', 'a test', repo)
    written = cm.record_qa('TOP_DOWN_GATE', reviewed, 'review/gate', 'docs/review_gate.md', 'sound', 'a test', repo)
    assert written['reviewed_commit'] == reviewed[:9] and written['reviewed_at'] == cm.tr.clock(1790005000)
    qa = cm.last_qa(repo, entry)
    assert qa['stale'] is False and qa['stale_text'] is None and qa['verdict'] == 'sound'
    # another function of the same file changes: this component's reviewed code is the same
    edit(repo, MODEL, 'return x - goal', 'return goal - x')
    commit(repo, 'the plan flips its sign', when=1790006000)
    cm.forget()
    assert cm.last_qa(repo, entry)['stale'] is False
    # the reviewed function changes: QA STALE, by itself
    edit(repo, MODEL, 'return [0.5 * t for t in tokens]', 'return [0.4 * t for t in tokens]')
    cm.forget()
    qa = cm.last_qa(repo, entry)
    assert qa['stale'] is True and qa['changed'] == [MODEL + '::Lite.gate']
    assert qa['stale_text'] == 'QA STALE: %s::Lite.gate changed after the reviewed commit %s.' % (MODEL, reviewed[:9])
    # a review of the newer commit is recorded: fresh again
    changed = commit(repo, 'the gate mixes 0.4', when=1790007000)
    git(repo, 'checkout', '-q', 'review/gate')
    write(repo, 'docs/review_gate.md', 'Reviewed again: the gate at commit %s. Verdict: sound.\n' % changed[:9])
    commit(repo, 'the second review of the gate', when=1790008000)
    git(repo, 'checkout', '-q', '-')
    cm.record_qa('TOP_DOWN_GATE', changed, 'review/gate', 'docs/review_gate.md', 'sound', 'a test', repo)
    cm.forget()
    qa = cm.last_qa(repo, entry)
    assert qa['stale'] is False and qa['reviewed_commit'] == changed[:9]
    assert len(cm.qa_entries(repo)) == 2, 'the log is appended to, never rewritten'


def test_a_repair_on_its_own_branch_is_read_at_that_branchs_tip(repo):
    git(repo, 'checkout', '-q', '-b', 'fix/tilt')
    write(repo, SRC + 'lem/tilt_path.py', 'def path(x):\n    return x\n')
    built = commit(repo, 'the tilt path', when=1790009000)
    git(repo, 'checkout', '-q', '-')
    data = cm.load(repo)
    data['components'].append(component('HEAD_TILT_REPAIR', [dict(file=SRC + 'lem/tilt_path.py')], kind='repair',
                                        ref='fix/tilt', switch=dict(flag='--vestibular-input-path', on=['on'])))
    cm.save(data, repo)
    cm.confirm(['HEAD_TILT_REPAIR'], 'a test', 'checked against the code', repo)
    commit(repo, 'the entry of the repair', when=1790009500)
    cm.forget()
    assert cm.check(repo)['ok'], 'the file is not in this branch\'s tree and the entry reads it on fix/tilt'
    entry = cm.components(cm.load(repo))['HEAD_TILT_REPAIR']
    assert entry['confirmed']['commit'] == built[:9]
    git(repo, 'checkout', '-q', 'fix/tilt')
    edit(repo, SRC + 'lem/tilt_path.py', 'return x', 'return 2 * x')
    commit(repo, 'the tilt path, version 2', when=1790010000)
    git(repo, 'checkout', '-q', '-')
    cm.forget()
    assert cm.check(repo)['stale'] == ['HEAD_TILT_REPAIR'], 'the branch moved on after the entry was confirmed'
    missing = dict(entry, ref='fix/gone')
    assert 'its branch fix/gone is not in this repository' in ' '.join(cm.structure_problems(repo, dict(components=[missing])))


# ------------------------------------------------------------------------------------------ the switch state
def test_the_switch_state_is_read_from_the_active_runs_launch_settings(tmp_path):
    run = tmp_path/'Saved/mimo_bridge/experiments/H84_x_seg41o_1431'
    run.mkdir(parents=True)
    (run/'args.json').write_text(json.dumps(dict(heads_all_slot_path='on', goal_locality='off', top_down_gate=True)),
                                 encoding='utf-8')
    (tmp_path/'Saved/mimo_bridge/training_continuation.json').write_text(json.dumps(dict(active_run=str(run))),
                                                                       encoding='utf-8')
    name, args = cm.active_args(tmp_path)
    assert name == 'H84_x_seg41o_1431'
    on = cm.switch_state(dict(flags=['--heads-all-slot-path'], switch=dict(flag='--heads-all-slot-path', on=['on'])),
                         name, args)
    assert on['repair_on'] is True and on['sentence'] == \
        '--heads-all-slot-path (on) is ON in the active run H84_x_seg41o_1431.'
    other_value = cm.switch_state(dict(flags=['--heads-all-slot-path'],
                                       switch=dict(flag='--heads-all-slot-path', on=['imagined_offset'])), name, args)
    assert other_value['repair_on'] is False
    absent = cm.switch_state(dict(flags=['--vestibular-input-path'],
                                  switch=dict(flag='--vestibular-input-path', on=['on'])), name, args)
    assert absent['repair_on'] is False and absent['flags'][0]['present'] is False
    part = cm.switch_state(dict(flags=['--top-down-gate', '--goal-locality']), name, args)
    assert part['repair_on'] is None and [f['text'] for f in part['flags']] == [
        'on in H84_x_seg41o_1431', 'off in H84_x_seg41o_1431']
    assert cm.switch_state(dict(flags=['--x'], switch=dict(flag='--x', on=['on'])), None, None)['repair_on'] is None


# ------------------------------------------------------------------------------------------ the three wirings
def relaunch_under_test(repo, tmp_path, monkeypatch, preflight_answer=None):
    """The relaunch script loaded from this checkout, with a temporary Saved/ folder whose supervisor record holds a
    recorded trainer command (the training preflight reads it first since production's commit 32607d9c5) and whose
    control file is a pause, no trainer process, the active run's newest complete checkpoint ckpt_step_200 (the
    recorded command resumed ckpt_step_100), and every step that would end or start a process made a test failure.
    The training preflight's launch_refusal is a stub that keeps the command it was asked about and answers
    `preflight_answer` (None: the configuration may launch); its launch_plan answers normal training."""
    spec = importlib.util.spec_from_file_location(
        'relaunch_under_test', REPO/'docs/gain_of_function_v4/coordinator/relaunch_h84_seg41_2026-09-29.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved = tmp_path/'Saved/mimo_bridge'
    saved.mkdir(parents=True)
    run = saved/'experiments/run'
    recorded = ['python.exe', '-u', '-m', 'mimo_bridge.train_mimo', '--out', str(run), '--resume',
                str(run/'checkpoints/ckpt_step_100'), '--heads-all-slot-path', 'on']
    (saved/'training_continuation.json').write_text(json.dumps(dict(active_run=str(run), command=recorded)))
    (saved/'supervisor_control.json').write_text(json.dumps(dict(cmd='pause', reason='a test hold')))
    monkeypatch.setattr(module.L, 'SAVED', saved)
    monkeypatch.setattr(module.L, 'EXP', saved/'experiments')
    monkeypatch.setattr(module.L, 'ROOT', repo)
    monkeypatch.setattr(module.L, 'trainers', lambda: [])
    monkeypatch.setattr(module, 'newest_complete_checkpoint', lambda parent: parent/'checkpoints/ckpt_step_200')
    import psutil
    monkeypatch.setattr(psutil, 'process_iter', lambda *a, **k: pytest.fail('a process was looked for to be ended'))
    monkeypatch.setattr(module.L, 'guard', lambda: pytest.fail('the memory guard was asked'))
    from mimo_bridge.tools import training_preflight
    asked = []
    monkeypatch.setattr(training_preflight, 'launch_refusal',
                        lambda cmd, *a, **k: asked.append(list(cmd)) or preflight_answer)
    monkeypatch.setattr(training_preflight, 'launch_plan', lambda cmd, *a, **k: dict(
        kind='normal', fingerprint='0123456789abcdef', max_steps=None, stop_criteria=None))
    return module, recorded, asked


def asked_flags(asked, saved_run):
    """The flags of the one command the preflight was asked about, and checks that it is the command the relaunch
    starts: it resumes the newest complete checkpoint, writes a new sibling run folder and has no step bound."""
    from mimo_bridge.tools import training_preflight
    assert len(asked) == 1
    flags = training_preflight.flags_of(asked[0])
    assert flags['--resume'] == str(saved_run/'checkpoints/ckpt_step_200')
    out = Path(flags['--out'])
    assert out.parent == saved_run.parent and out.name.startswith('H84_fresh_drivers_20260918_seg41x_')
    assert flags['--checkpoint-storage'] == str(out/'checkpoints') and flags['--max-steps'] == '0'
    assert not out.exists(), 'the run folder was created before the checks'
    return flags


def test_the_relaunch_script_refuses_on_a_stale_manifest_before_it_ends_or_starts_anything(repo, tmp_path, monkeypatch):
    module, recorded, asked = relaunch_under_test(repo, tmp_path, monkeypatch)
    monkeypatch.setattr('sys.argv', ['relaunch.py', '--tag', 'x', '--reason', 'a test', '--set', 'goal-locality', 'on'])
    assert module.manifest_refusal(repo) is None
    edit(repo, MODEL, 'return [0.5 * t for t in tokens]', 'return [0.4 * t for t in tokens]')
    with pytest.raises(SystemExit, match='refused: the component manifest is not fresh .1 finding.: TOP_DOWN_GATE'):
        module.main()
    # the preflight was asked first, about the command the relaunch starts, with the --set change applied
    flags = asked_flags(asked, tmp_path/'Saved/mimo_bridge/experiments/run')
    assert flags['--goal-locality'] == 'on' and flags['--heads-all-slot-path'] == 'on'
    # with the control removed (the refusal answers None) the script goes on to its next step, which this test stops
    monkeypatch.setattr(module, 'manifest_refusal', lambda root=None: None)
    with pytest.raises(pytest.fail.Exception, match='a process was looked for to be ended'):
        module.main()


def test_the_relaunch_script_asks_the_training_preflight_before_the_manifest(repo, tmp_path, monkeypatch):
    """The operator's rule of Oct 5, 2026: no new or changed configuration launches until its preflight's seven gates
    read PASS or N/A. After the merge of production's branch both refusals stand in the script, the preflight's first:
    a preflight refusal ends the script before the manifest is read and before any process is looked for."""
    module, recorded, asked = relaunch_under_test(repo, tmp_path, monkeypatch,
                                                  preflight_answer='BLOCKED: 1 of 7 gates block launch')
    monkeypatch.setattr(module, 'manifest_refusal', lambda root=None: pytest.fail('the manifest was read first'))
    monkeypatch.setattr('sys.argv', ['relaunch.py', '--tag', 'x', '--reason', 'a test'])
    with pytest.raises(SystemExit, match='refused by the training preflight:\nBLOCKED: 1 of 7 gates block launch'):
        module.main()
    asked_flags(asked, tmp_path/'Saved/mimo_bridge/experiments/run')


def test_the_preflight_is_asked_about_the_checkpoint_the_relaunch_resumes(repo, tmp_path, monkeypatch):
    """Before Oct 5, 2026 (evening) the script asked the preflight about the recorded command, whose --resume is the
    checkpoint the stopped trainer had resumed, and then launched from the run's newest complete checkpoint: a record
    that holds for one model state could have passed a launch from another. The checkpoint is chosen first now."""
    module, recorded, asked = relaunch_under_test(repo, tmp_path, monkeypatch, preflight_answer='BLOCKED: stop here')
    monkeypatch.setattr('sys.argv', ['relaunch.py', '--tag', 'x', '--reason', 'a test'])
    with pytest.raises(SystemExit, match='refused by the training preflight'):
        module.main()
    assert asked_flags(asked, tmp_path/'Saved/mimo_bridge/experiments/run')['--resume'] != recorded[recorded.index(
        '--resume') + 1]


def test_a_run_folder_without_a_complete_checkpoint_is_refused_before_the_preflight(repo, tmp_path, monkeypatch):
    module, recorded, asked = relaunch_under_test(repo, tmp_path, monkeypatch)
    spec = importlib.util.spec_from_file_location(
        'relaunch_fresh', REPO/'docs/gain_of_function_v4/coordinator/relaunch_h84_seg41_2026-09-29.py')
    fresh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh)
    empty = tmp_path/'empty_run'
    (empty/'checkpoints').mkdir(parents=True)
    with pytest.raises(SystemExit, match='has no complete eight-world checkpoint'):
        fresh.newest_complete_checkpoint(empty)
    assert asked == []


def test_the_api_view_carries_the_check_and_every_generated_field(repo, tmp_path, monkeypatch):
    monkeypatch.setenv('UEMA_COMPONENT_CACHE', str(tmp_path/'cache'))
    view = cm.status(repo, repo)
    assert view['available'] and view['check']['ok'] and [c['id'] for c in view['components']] == [
        'TRAINER', 'TOP_DOWN_GATE', 'PLANNER_SEARCH', 'PREDICTION_HEADS']
    gate = view['components'][1]
    assert gate['last_qa'] is None and gate['last_qa_text'] == 'no independent QA on record' and gate['qa_stale'] is False
    assert gate['entry_fresh'] and gate['last_code_change']['commit'] and gate['open_defects'] == []
    assert gate['last_code_change_text'] == '%s, commit %s' % (cm.tr.clock(1790000000),
                                                               gate['last_code_change']['commit'])
    edit(repo, MODEL, 'return [0.5 * t for t in tokens]', 'return [0.4 * t for t in tokens]')
    view = cm.status(repo, repo, only='TOP_DOWN_GATE')
    assert not view['check']['ok'] and view['check']['stale'] == ['TOP_DOWN_GATE']
    assert [c['id'] for c in view['components']] == ['TOP_DOWN_GATE'] and not view['components'][0]['entry_fresh']
    assert view['components'][0]['last_code_change']['uncommitted'] is True


def test_the_view_carries_each_open_defects_closing_test(repo, tmp_path, monkeypatch):
    """The decided design of the Architecture tab, Oct 7, 2026, D-E2: the explainer's list of the parts being worked on
    prints, beside each repair's state, what ends it - the closing test the defect register types - and the register's
    note on that repair (HEAD_TILT_REPAIR_1's "Purpose not met."). status() copies the closing test into every open
    defect as {row, text}: the row None while it is not written, both None for an entry without the field; a closed
    entry (VERIFIED) is not an open defect; repair_note is the note of this component's own repair entry."""
    monkeypatch.setenv('UEMA_COMPONENT_CACHE', str(tmp_path/'cache'))

    def defect(did, **more):
        entry = dict(id=did, title='A measured fault of %s.' % did, component='TOP_DOWN_GATE', repairs=[],
                     found_at='Oct 4, 2026 2:45 PM PDT', severity='GOF-BLOCKER', severity_basis='a test',
                     owner='the coordinator', backup_owner='an agent', cause_status='FOUND: not named',
                     next_action='find the cause', waits_on='nothing', state='FOUND', tests=[])
        entry.update(more)
        return entry
    register = dict(about='a test register', schema=1, defects=[
        defect('GATE_STUCK', closing_test=dict(row='gate_probe', text='the gate opens on a surprise')),
        defect('GATE_SLOW', closing_test=dict(row=None, text='the gate opens within one model step')),
        defect('GATE_UNTYPED'),
        defect('GATE_DONE', state='VERIFIED', closing_test=dict(row='done_probe', text='the gate opens')),
        # a defect of another part that lists a repair of this one: its note on the repair comes with it
        defect('HEADS_SLOW', component='PREDICTION_HEADS', closing_test=dict(row='heads_probe', text='t'),
               repairs=[dict(component='TOP_DOWN_GATE', state='IN TRAINING', note='Switched on Oct 3. Purpose not met.'),
                        dict(component='PLANNER_SEARCH', state='BUILT/OFF', note='another repair\'s note')])])
    write(repo, 'docs/gain_of_function_v4/defect_register.json', json.dumps(register))
    view = cm.status(repo, repo, only='TOP_DOWN_GATE', with_history=False)
    opened = view['components'][0]['open_defects']
    got = {d['id']: d['closing_test'] for d in opened}
    assert got == {'GATE_STUCK': dict(row='gate_probe', text='the gate opens on a surprise'),
                   'GATE_SLOW': dict(row=None, text='the gate opens within one model step'),
                   'GATE_UNTYPED': dict(row=None, text=None),
                   'HEADS_SLOW': dict(row='heads_probe', text='t')}, got
    notes = {d['id']: (d['repair_state'], d['repair_note']) for d in opened}
    assert notes['HEADS_SLOW'] == ('IN TRAINING', 'Switched on Oct 3. Purpose not met.'), notes
    assert notes['GATE_STUCK'] == (None, None), notes


# ------------------------------------------------------------------------------------------ this checkout's manifest
def test_this_checkouts_manifest_is_fresh():
    """THE CHECK, WIRED INTO THE UNIT TESTS. It fails when a file the trainer imports is mapped to no component, when
    a component's mapped code differs from the code its entry was confirmed against, or when the manifest is
    malformed. To clear a finding: read the changed code, correct the entry in
    docs/gain_of_function_v4/component_manifest.json, and run
    `python -m mimo_bridge.tools.component_manifest confirm ID --by NAME --basis "..."`."""
    found = cm.check(REPO)
    assert found['ok'], '\n'.join(found['problems'])


def test_this_checkouts_manifest_holds_the_contracts_names_and_the_forbidden_alias():
    data = cm.load(REPO)
    known = cm.components(data)
    contract = ['PLANNER_SEARCH', 'PLANNER_GOAL_SELECTION', 'PLANNER_GOAL_DISTANCE', 'PLANNER_PROSPECTIVE_WELLNESS',
                'PLANNER_RECORD', 'PREDICTOR_REPAIR', 'HEAD_TILT_REPAIR', 'HEAD_TILT_REPAIR_1', 'MOTOR_ENCODER_REPAIR',
                'TACTILE_OFFSET_FIX', 'PLANNER_SENSE_WEIGHTS', 'PLANNER_GOAL_LOCALITY', 'PLANNER_PROSPECTIVE_TERM',
                'PLANNER_SEARCH_REPAIR']
    parts = ['ENCODER_VISION', 'ENCODER_TACTILE', 'ENCODER_AUDIO', 'ENCODER_PROPRIOCEPTION', 'ENCODER_VESTIBULAR',
             'TRANSFORMER', 'PREDICTION_HEADS', 'MOTOR_ENCODER', 'MOTOR_CONTROLLER', 'PERSISTENT_MEMORY',
             'WELLNESS_EVALUATORS', 'WELLNESS_PREDICTOR', 'GOAL_RING', 'IMAGINATION_CHAIN', 'REPLAY', 'TOP_DOWN_GATE',
             'LIVENESS_REPAIR', 'HAND_STAGE', 'COMMAND_EXECUTION_PATH', 'TEST_REGISTRY', 'DASHBOARD', 'EXPLAINER',
             'WATCH_PROGRAMS']
    assert [name for name in contract + parts if name not in known] == []
    owners = dict(cm.aliases(REPO))
    assert sorted(owners['head repair']) == ['HEAD_TILT_REPAIR', 'PREDICTOR_REPAIR'], 'it named two different repairs'
    assert all(len(c['does'].split()) >= 8 and c['does'].rstrip().endswith('.') for c in data['components'])
    assert all(c['confirmed']['basis'].startswith(('checked against the code', 'carried over from the explainer'))
               for c in data['components'])
    blocks = {b for c in data['components'] for b in c['blocks']}
    from mimo_bridge.explainer import schema
    boxes = {k for k in schema.DOC if k.startswith('node.')}
    assert boxes - blocks == set(), 'every Architecture box has a component'
