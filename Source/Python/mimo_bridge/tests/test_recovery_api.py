# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""The dashboard's routes of Oct 5, 2026 (tools/recovery_api.py, served by tools/development_dashboard.py), and the
second dashboard that reads another checkout's data without writing there.

Every route is answered here from a temporary tree: a small registry, a small manifest in a small git repository, a
small defect register and a run folder. The server is the dashboard's own Handler on a free loopback port.
"""
import json
import subprocess
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from mimo_bridge.tools import component_manifest as cm
from mimo_bridge.tools import defect_register as dr
from mimo_bridge.tools import recovery_api as api
from mimo_bridge.tools import test_registry as tr
from test_run_in_training_pause import complete_row, write_registry

SRC = 'Source/Python/'


def git(root, *args):
    done = subprocess.run(['git', '-C', str(root), *args], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def put(root, path, text):
    target = Path(root)/path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding='utf-8', newline='\n')


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """`code`: a checkout with a manifest, a QA log and a defect register; `data`: a checkout with the registry and
    Saved/. The two are separate folders, as for a second dashboard beside production's."""
    monkeypatch.setenv('UEMA_COMPONENT_CACHE', str(tmp_path/'cache'))
    monkeypatch.setattr(api, 'KEEP_S', 0.0)
    cm.forget()
    code, data = tmp_path/'code', tmp_path/'data'
    code.mkdir()
    git(code, 'init', '-q')
    git(code, 'config', 'user.email', 'test@example.com')
    git(code, 'config', 'user.name', 'A Test')
    git(code, 'config', 'core.autocrlf', 'false')
    put(code, SRC + 'mimo_bridge/__init__.py', '')
    put(code, SRC + 'mimo_bridge/train_mimo.py', 'def main():\n    return 1\n')
    put(code, SRC + 'mimo_bridge/search.py', 'def search(x):\n    return x\n')
    entry = dict(kind='model part', does='Takes x and outputs y.', blocks=['node.intention'], flags=['--intention'],
                 confirmed=dict(at='never', by='nobody', commit='none', content='none', basis='not yet'))
    cm.save(dict(about='t', schema=1, scope=dict(entry='mimo_bridge.train_mimo', also=[SRC + 'mimo_bridge/search.py'],
                                                 outside=[]),
                 components=[
                     dict(entry, id='TRAINER', name='trainer', kind='tool', blocks=[], flags=[], aliases=[],
                          paths=[dict(file=SRC + 'mimo_bridge/train_mimo.py'), dict(file=SRC + 'mimo_bridge/__init__.py')]),
                     dict(entry, id='PLANNER_SEARCH', name='planner search', aliases=['intention generator'],
                          paths=[dict(file=SRC + 'mimo_bridge/search.py', functions=['search'])]),
                     dict(entry, id='HEAD_TILT_REPAIR', name='head-tilt repair', kind='repair', aliases=['head repair'],
                          flags=['--vestibular-input-path'], paths=[dict(file=SRC + 'mimo_bridge/search.py')],
                          switch=dict(flag='--vestibular-input-path', on=['on']))]), code)
    git(code, 'add', '-A')
    git(code, 'commit', '-q', '-m', 'the code and its manifest')
    cm.confirm(['TRAINER', 'PLANNER_SEARCH', 'HEAD_TILT_REPAIR'], 'a test', 'checked against the code', code)
    dr.save(dict(about='t', schema=1, defects=[dict(
        id='PLANNER_SEARCH_UNSTABLE', title='The search is unstable.', component='PLANNER_SEARCH', repairs=[],
        stage='PLANNER', found_at='Oct 4, 2026 2:45 PM PDT', severity='GOF-BLOCKER', owner='an agent',
        backup_owner='the coordinator', cause_status='FOUND: not named', next_action='find the cause',
        waits_on='nothing', state='FOUND', closing_test=dict(row='search_probe', text='the same command twice'),
        tests=['search_probe'])]), code)
    run = data/'Saved/mimo_bridge/experiments/H84_x_seg41o_1431'
    run.mkdir(parents=True)
    (run/'args.json').write_text(json.dumps(dict(intention='on')), encoding='utf-8')
    (data/'Saved/mimo_bridge/training_continuation.json').write_text(json.dumps(dict(active_run=str(run))),
                                                                   encoding='utf-8')
    (data/'Saved/mimo_bridge/supervisor_control.json').write_text(json.dumps(dict(cmd='pause', reason='a test hold')),
                                                                 encoding='utf-8')
    write_registry(data, [complete_row('search_probe', component='PLANNER_SEARCH'),
                          complete_row('old_done', status='done')])
    cm.forget()
    return dict(code=code, data=data)


def test_every_route_answers_with_json_from_the_two_roots(tree):
    for route in api.ROUTES:
        out = api.respond(route, {}, tree['data'], tree['code'])
        json.dumps(out, allow_nan=False)
    with pytest.raises(KeyError):
        api.respond('/api/nothing', {}, tree['data'], tree['code'])


def test_the_components_route_carries_generated_fields_defects_and_linked_tests(tree):
    view = api.respond('/api/components', {}, tree['data'], tree['code'])
    assert view['check']['ok'] and view['active_run'] == 'H84_x_seg41o_1431'
    search = next(c for c in view['components'] if c['id'] == 'PLANNER_SEARCH')
    assert search['last_code_change']['commit'] and search['last_qa_text'] == 'no independent QA on record'
    assert [d['id'] for d in search['open_defects']] == ['PLANNER_SEARCH_UNSTABLE']
    assert [(t['id'], t['section']) for t in search['linked_tests']] == [('search_probe', 'QUEUED')]
    assert search['switch']['flags'][0]['text'] == 'on in H84_x_seg41o_1431'
    assert view['by_block'] == {'node.intention': ['PLANNER_SEARCH', 'HEAD_TILT_REPAIR']}
    one = api.respond('/api/components', {'id': ['HEAD_TILT_REPAIR'], 'history': ['0']}, tree['data'], tree['code'])
    assert [c['id'] for c in one['components']] == ['HEAD_TILT_REPAIR'] and one['components'][0]['last_code_change'] is None
    assert one['components'][0]['switch']['repair_on'] is False


def test_the_names_route_lists_every_vocabulary_once(tree):
    names = api.respond('/api/names', {}, tree['data'], tree['code'])
    assert [c['id'] for c in names['components']] == ['TRAINER', 'PLANNER_SEARCH', 'HEAD_TILT_REPAIR']
    assert names['forbidden_aliases'] == [dict(alias='intention generator', canonical=['PLANNER_SEARCH']),
                                          dict(alias='head repair', canonical=['HEAD_TILT_REPAIR'])]
    assert [s['name'] for s in names['repair_states']] == list(dr.STATES)
    assert [s['name'] for s in names['test_sections']] == list(tr.SECTIONS)
    assert [s['name'] for s in names['test_outcomes']] == list(tr.OUTCOMES)
    assert [s['name'] for s in names['machine_modes']] == ['TRAINING', 'TEST WINDOW', 'PAUSED FOR OPERATOR']
    assert len(names['scoreboard_stages']) == 5 and names['fixed_rule'] == 'Only VERIFIED may be called fixed.'
    assert [s['name'] for s in names['process_states']] == ['RUNNING', 'HELD', 'LOST']
    assert all(s['definition'] for key in ('repair_states', 'test_sections', 'test_outcomes', 'machine_modes',
                                           'scoreboard_stages', 'resource_classes', 'process_states')
               for s in names[key])
    # directive 7B: the Terms tab loads its definitions expanded and one button toggles both ways
    assert names['expanded_by_default'] is True
    assert names['toggle_labels'] == dict(when_expanded='Collapse all', when_collapsed='Expand all')


def test_the_overview_collects_every_error_the_pages_must_show(tree, monkeypatch):
    from mimo_bridge.tools import test_runner
    monkeypatch.setattr(test_runner, 'scan_processes', lambda: [])
    ok = api.respond('/api/overview', {}, tree['data'], tree['code'])
    assert ok['machine_mode']['mode'] == 'PAUSED FOR OPERATOR' and ok['machine_mode']['reason'] == 'a test hold'
    assert ok['scoreboard']['headline'] == 'No gain of function demonstrated.' and len(ok['scoreboard']['stages']) == 5
    assert ok['top_blockers'][0]['id'] == 'PLANNER_SEARCH_UNSTABLE' and ok['open_blockers'] == 1
    assert ok['status_lines'] == ['PLANNER_SEARCH_UNSTABLE: FOUND'] and ok['manifest']['ok']
    assert [e for e in ok['errors'] if e['source'] != 'test registry audit'] == []
    # a stale manifest, a running row without a process and a register problem each become an error
    put(tree['code'], SRC + 'mimo_bridge/search.py', 'def search(x):\n    return 2 * x\n')
    cm.forget()
    assert tr.job_started('search_probe', path=tr.registry_path(tree['data']), pid=999999, pid_created=1.0,
                          runner_pid=999998) is None
    data = dr.load(tree['code'])
    data['defects'][0]['state'] = 'FIXED'
    dr.save(data, tree['code'])
    bad = api.respond('/api/overview', {}, tree['data'], tree['code'])
    by_source = {}
    for e in bad['errors']:
        by_source.setdefault(e['source'], []).append(e['text'])
    assert any('PLANNER_SEARCH: its mapped code changed' in t for t in by_source['component manifest'])
    assert any('Row search_probe says running' in t and 'is gone' in t for t in by_source['test registry audit'])
    assert any('its state is not one of' in t for t in by_source['defect register'])
    assert bad['error_count'] == len(bad['errors']) >= 3 and bad['manifest']['ok'] is False


@pytest.fixture
def server(tree):
    from mimo_bridge.tools.development_dashboard import Handler
    httpd = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    httpd.root, httpd.code_root = tree['data'], tree['code']
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield 'http://127.0.0.1:%d' % httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def fetch(url):
    return json.loads(urllib.request.urlopen(url, timeout=30).read())


def test_the_dashboard_serves_the_routes_and_the_code_roots_own_files(tree, server):
    assert fetch(server + '/api/defects')['defects'][0]['id'] == 'PLANNER_SEARCH_UNSTABLE'
    assert fetch(server + '/api/scoreboard')['headline'] == 'No gain of function demonstrated.'
    assert fetch(server + '/api/machine_mode')['now']['mode'] in ('TRAINING', 'TEST WINDOW', 'PAUSED FOR OPERATOR')
    assert fetch(server + '/api/components?id=PLANNER_SEARCH&history=0')['components'][0]['name'] == 'planner search'
    assert 'mismatches' in fetch(server + '/api/registry_audit')
    tests = fetch(server + '/api/tests')
    assert tests['schema'] == 2 and 'groups' in tests and [s['key'] for s in tests['sections']] == list(tr.SECTIONS)
    assert 'rows' not in tests['sections'][0] and tests['sections'][0]['positions'] == [0]
    by_section = fetch(server + '/api/tests?view=sections')
    assert 'groups' not in by_section and by_section['sections'][0]['rows'][0]['id'] == 'search_probe'
    assert by_section['sections'][0]['open_by_default'] is True
    assert [s['open_by_default'] for s in by_section['sections'][1:]] == [False] * 6
    # the manifest lives in the code's checkout, the registry in the data's: both are served under /files/
    manifest = fetch(server + '/files/docs/gain_of_function_v4/component_manifest.json')
    assert manifest['components'][0]['id'] == 'TRAINER'
    assert fetch(server + '/files/docs/gain_of_function_v4/test_registry.json')['rows'][0]['id'] == 'search_probe'


def test_the_second_dashboard_writes_nothing_into_the_data_checkout(tree, server):
    def listing():
        return sorted((str(p.relative_to(tree['data'])), p.stat().st_mtime_ns, p.stat().st_size)
                      for p in tree['data'].rglob('*') if p.is_file())
    before = listing()
    for route in api.ROUTES + ('/api/tests', '/api/tests?view=sections'):
        fetch(server + route)
    assert listing() == before, 'no file of the data checkout was written, added or removed'


WATCH_PROGRAM = '''
import os, sys
sys.path.insert(0, sys.argv[4])
from mimo_bridge.tools import write_watch
watched, outside, log = sys.argv[1], sys.argv[2], sys.argv[3]
write_watch.watch_writes(watched, log)
open(os.path.join(watched, "Saved", "old.txt")).read()
open(os.path.join(outside, "cache.json"), "w").write("{}")
open(os.path.join(watched, "Saved", "new.json"), "w").write("{}")
os.replace(os.path.join(watched, "Saved", "new.json"), os.path.join(watched, "Saved", "moved.json"))
os.remove(os.path.join(watched, "Saved", "moved.json"))
os.mkdir(os.path.join(watched, "made"))
fd = os.open(os.path.join(watched, "Saved", "flags.bin"), os.O_WRONLY | os.O_CREAT)
os.close(fd)
'''


def test_the_write_watch_records_every_write_inside_its_folder_and_nothing_outside(tmp_path):
    """The second dashboard's evidence (tools/write_watch.py), run in a child process: an audit hook cannot be removed
    from the process that installs it."""
    import sys
    from mimo_bridge.tools import write_watch
    watched, outside, log = tmp_path/'production', tmp_path/'scratch', tmp_path/'logs/writes.jsonl'
    (watched/'Saved').mkdir(parents=True)
    (watched/'Saved/old.txt').write_text('x')
    outside.mkdir()
    source = Path(__file__).resolve().parents[2]
    done = subprocess.run([sys.executable, '-c', WATCH_PROGRAM, str(watched), str(outside), str(log), str(source)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    events = write_watch.read_log(log)
    assert [e['event'] for e in events] == ['open', 'os.rename', 'os.remove', 'os.mkdir', 'open']
    names = [Path(p).name for e in events for p in e['paths']]
    assert names == ['new.json', 'new.json', 'moved.json', 'moved.json', 'made', 'flags.bin']
    assert not any('cache.json' in p or 'old.txt' in p for e in events for p in e['paths']), \
        'a read and a write outside the watched folder are not recorded'
    with pytest.raises(ValueError, match='lies inside the watched folder'):
        write_watch.watch_writes(watched, watched/'log.jsonl')
