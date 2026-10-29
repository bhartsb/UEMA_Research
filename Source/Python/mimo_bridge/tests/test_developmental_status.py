# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json
import hashlib
import pytest
from mimo_bridge.tools import developmental_status as status
from mimo_bridge.tools import development_dashboard as dashboard


def put(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def test_finished_experiment_is_visible_but_never_an_acquired_ability(tmp_path):
    definition=dict(id='roll_over',tier=2,name='Roll',minimum_environment={'description':'Body and floor'},
        natural_driver='Explore another view',human_comparison={'text':'Infant comparison'},
        tests=[{'question':'Can it finish a real turn?'}],implementation_gap='Physical reachability')
    put(tmp_path/'docs/gain_function_2026-09-13/requirements.json',dict(tier_names={'2':'Posture'},abilities=[definition]))
    work=dict(jobs=[dict(id='test',title='Roll test',abilities=['roll_over'],state='complete',completed_at=3)])
    ability=status.build(tmp_path,work)['stages'][0]['abilities'][0]
    # 2026-09-22: "Results available" was status shorthand (every result belonged to an earlier
    # lineage whose files are gone); the status now says what was measured on which model.
    assert ability['status']=='Tested (1 finished test); the tested model is not recorded, so it is not linked to the running model'
    assert ability['acquired'] is False
    assert ability['experiments'][0]['id']=='test'


@pytest.mark.parametrize('jobs,expected',[
    ([dict(state='complete',completed_at=3,lineage='ancestor_checkpoint')],
     "Measured on this model's ancestry; acquisition, retention and transfer not established"),
    ([dict(state='complete',completed_at=3,lineage='not_linked_to_current_model')],
     'Tested on an earlier lineage, not on the running model (1 finished test)'),
    ([dict(state='files_missing',lineage='unknown'),dict(state='files_missing',lineage='unknown')],
     'The result files of 2 tests are missing (their registered folders no longer exist)'),
    ([dict(state='queued')], 'Planned; never run on the running model'),
    ([dict(state='running')], 'A measurement is running now; no result yet'),
    ([], 'No test has run on any model; the test is defined in the requirements')])
def test_status_names_what_was_measured_and_on_which_model(jobs,expected):
    assert status.ability_status(jobs)==expected
    for banned in ('Test queued','Results available','Test definition ready'):
        assert banned not in status.ability_status(jobs)


def test_external_checkpoint_ancestry_does_not_expand_public_file_access(tmp_path):
    root=tmp_path/'repo';outside=tmp_path/'authorized';run=root/'Saved/mimo_bridge/experiments/new'
    earlier=root/'Saved/mimo_bridge/experiments/old';checkpoint=outside/'run/checkpoints/ckpt_step_100'
    put(root/'Saved/mimo_bridge/checkpoint_storage_roots.json',dict(roots=[str(outside)]))
    put(run/'args.json',dict(resume=str(checkpoint)))
    put(checkpoint.parent.parent/'args.json',dict(out=str(earlier),checkpoint_storage=str(checkpoint.parent)))
    put(earlier/'args.json',{})
    bounds=dashboard.ancestry_bounds(root,dict(run=str(run.relative_to(root)),step=150))
    assert bounds[str(earlier)]==100
    assert dashboard.lineage_relation(root,str(checkpoint),bounds)=='ancestor_checkpoint'
    import pytest
    with pytest.raises(ValueError):dashboard.confined(root,checkpoint/'model.safetensors')
    with pytest.raises(ValueError):dashboard.checkpoint_reference(root,tmp_path/'unregistered/checkpoints/ckpt_step_10')


def bundle(tmp_path):
    from mimo_bridge.tests.test_developmental_evidence import full_declaration,pairs
    folder=tmp_path/'run';folder.mkdir()
    def artifact(name,value):
        path=folder/name;put(path,value)
        return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    d=full_declaration();traces=pairs(phases=d['required_phases'],conditions=d['required_conditions'])
    trace_refs=[artifact(f'trace_{i}.json',trace) for i,trace in enumerate(traces)]
    receipts={}
    for kind in ('learning_comparison','retention_without_training','held_out_transfer','physical_calibration'):
        record=dict(contract='developmental_scope_v1',ability_id='gaze_control',kind=kind,verified=True,
            artifacts=[trace_refs[0]],learning_updates=10,before_checkpoint_sha256='a'*64,after_checkpoint_sha256='b'*64,
            optimizer_updates=0,after_intervening_experience=True,training_worlds=[99],held_out_worlds=list(range(8)),
            split_declared_before_test=True,reachability_verified=True,same_action_replay_exact=True)
        receipts[kind]=artifact(kind+'.json',record)
        if kind=='physical_calibration':d['calibration']['receipt']=kind+'.json'
        else:d['evidence_scope'][kind]['receipt']=kind+'.json'
    report=dict(developmental_evidence=dict(contract='developmental_comparison_v1',ability_id='gaze_control',
        declaration=artifact('declaration.json',d),trials=trace_refs,scope_receipts=receipts))
    put(folder/'report.json',report)
    return folder,report


def test_complete_located_comparison_is_recomputed_and_can_earn_status(tmp_path):
    folder,_=bundle(tmp_path)
    result=status.ingest_comparison(tmp_path,folder,'gaze_control')
    assert result['capability_pass'] is True
    assert result['artifacts_verified'] is True
    # Green booleans in an ordinary report have no developmental effect.
    put(folder/'report.json',dict(capability_pass=True))
    assert status.ingest_comparison(tmp_path,folder,'gaze_control')['capability_pass'] is False


@pytest.mark.parametrize('corruption',['missing_scope','trace_changed','other_ability','absent_artifact','outside_root','missing_transfer'])
def test_incomplete_or_changed_evidence_never_earns_an_ability(tmp_path,corruption):
    folder,report=bundle(tmp_path);b=report['developmental_evidence']
    if corruption=='missing_scope':del b['scope_receipts']['learning_comparison']
    elif corruption=='trace_changed':put(folder/'trace_0.json',{})
    elif corruption=='other_ability':b['ability_id']='roll_over'
    elif corruption=='absent_artifact':(folder/'retention_without_training.json').unlink()
    elif corruption=='outside_root':b['declaration']['path']='../../outside.json'
    else:
        d=json.loads((folder/'declaration.json').read_text());d['required_phases']=['acquisition','retention']
        put(folder/'declaration.json',d)
        b['declaration']['sha256']=hashlib.sha256((folder/'declaration.json').read_bytes()).hexdigest()
    put(folder/'report.json',report)
    assert status.ingest_comparison(tmp_path,folder,'gaze_control')['capability_pass'] is False
