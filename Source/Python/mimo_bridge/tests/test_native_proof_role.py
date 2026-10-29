# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import json

import pytest

from mimo_bridge.tools import native_proof_role as role
from mimo_bridge.tools import training_continuation as tc


@pytest.fixture
def proof(tmp_path):
    root=tmp_path/'private_trial';root.mkdir()
    source=tmp_path/'cpu_copy'/'ckpt_step_100';source.mkdir(parents=True)
    receipt=dict(cpu_probe_only=True,world_count=8,production_choice_license=False,
        old_episode_exact_except_declared_boundary=True,parent=str(tmp_path/'production'/'ckpt_step_100'),step=100,
        changes={'self_choice_receipt':str(tmp_path/'calibration.json')})
    (source/'continuity_migration.json').write_text(json.dumps(receipt))
    (source/'meta.json').write_text(json.dumps({'step':100}))
    declaration=root/'declaration.json'
    declaration.write_text(json.dumps(dict(schema=role.CONTRACT,out=str(root),source=str(source),
        updates_per_pass=2,passes=2,body_count=8,source_preservation=receipt,production_controls_used=False)))
    folder=root/'pass_1'
    command=['python','-m','mimo_bridge.train_mimo',role.FLAG,str(declaration),
        '--resume',str(source),'--device','cpu','--n-envs','8','--out',str(folder),
        '--checkpoint-storage',str(folder/'checkpoints'),'--caregiver-record-dir',str(folder/'recordings'),
        '--self-choice','shadow','--self-choice-receipt',str(tmp_path/'calibration.json'),
        '--contact-profile','mechanical_v2_local','--max-steps','102','--minutes','0']
    return command


def test_only_declared_bounded_command_is_isolated(proof):
    assert role.is_isolated(proof)
    command=proof.copy();index=command.index(role.FLAG);del command[index:index+2]
    assert not role.is_isolated(command)


@pytest.mark.parametrize('flag,value',[
    ('--device','cuda:0'),('--max-steps','0'),('--max-steps','103'),('--minutes','60'),
    ('--self-choice','on'),('--n-envs','4'),('--out','C:/a_live_training_run'),
    ('--checkpoint-storage','C:/production/checkpoints'),('--caregiver-record-dir','C:/production/recordings'),
    ('--resume','C:/production/ckpt_step_100'),('--self-choice-receipt','C:/different_receipt.json')])
def test_flag_alone_cannot_hide_an_ordinary_or_modified_learner(proof,flag,value):
    command=tc.replace_argument(proof,flag,value)
    assert not role.is_isolated(command)
    with pytest.raises(ValueError):role.validate(command)


def test_duplicate_override_cannot_hide_its_actual_argparse_value(proof):
    assert not role.is_isolated(proof+['--max-steps','9000000'])


def test_second_pass_requires_the_first_pass_actual_snapshot(proof):
    declaration=role.validate(proof);root=role._path(declaration['out']);folder=root/'pass_2'
    source=root/'pass_1/checkpoints/ckpt_step_102';source.mkdir(parents=True)
    (source/'meta.json').write_text(json.dumps({'step':102}))
    command=proof
    for flag,value in {'--resume':source,'--out':folder,'--checkpoint-storage':folder/'checkpoints',
                       '--caregiver-record-dir':folder/'recordings','--max-steps':104}.items():
        command=tc.replace_argument(command,flag,value)
    assert role.is_isolated(command)
    (source/'meta.json').write_text(json.dumps({'step':101}))
    assert not role.is_isolated(command)


def test_missing_or_changed_cpu_copy_receipt_remains_production(proof):
    source=role._path(tc.argument(proof,'--resume'));path=source/'continuity_migration.json'
    receipt=json.loads(path.read_text());receipt['cpu_probe_only']=False;path.write_text(json.dumps(receipt))
    assert not role.is_isolated(proof)
    path.unlink();assert not role.is_isolated(proof)


def test_process_inventory_excludes_only_validated_proof(proof,monkeypatch):
    class Process:
        def __init__(self,pid,command):self.pid=pid;self.info={'pid':pid,'ppid':0,'name':'python.exe','cmdline':command}
    live=Process(10,['python','-m','mimo_bridge.train_mimo','--out','production'])
    valid=Process(20,proof)
    bad=Process(30,tc.replace_argument(proof,'--max-steps',0))
    monkeypatch.setattr(tc.psutil,'process_iter',lambda _:iter([live,valid,bad]))
    assert [process.pid for process in tc.processes('trainer')]==[10,30]


def test_proof_marker_does_not_change_the_acting_configuration(proof):
    from mimo_bridge.train_mimo import acting_configuration,parse_args
    ordinary=parse_args(['--out','ordinary'])
    copied=parse_args(['--out','private','--isolated-native-proof',tc.argument(proof,role.FLAG)])
    assert acting_configuration(ordinary)==acting_configuration(copied)


@pytest.fixture
def extension(proof,tmp_path):
    import hashlib
    original=role.validate(proof);root=role._path(original['out'])
    source=root/'pass_2/checkpoints/ckpt_step_104';source.mkdir(parents=True)
    (source/'meta.json').write_text(json.dumps({'step':104}))
    report=root/'report.json'
    report.write_text(json.dumps(dict(complete=True,production_choice_license=False,
        original_production_trainer_controlled=False,passes=[
            dict(world_count=8,qualification_only=True,applied_to_body=False),
            dict(world_count=8,qualification_only=True,applied_to_body=False,checkpoint=str(source),step=104)])))
    out=tmp_path/'audit_extension';out.mkdir();folder=out/'pass_1';declaration=out/'declaration.json'
    declaration.write_text(json.dumps(dict(schema=role.EXTENSION_CONTRACT,out=str(out),source=str(source),
        previous_report=str(report),previous_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        updates_per_pass=2,passes=1,body_count=8,production_controls_used=False)))
    command=proof
    for flag,value in {role.FLAG:declaration,'--resume':source,'--out':folder,
                       '--checkpoint-storage':folder/'checkpoints','--caregiver-record-dir':folder/'recordings',
                       '--max-steps':106}.items():command=tc.replace_argument(command,flag,value)
    return command


def test_single_audit_extension_is_recognized_only_from_completed_cpu_proof(extension):
    assert role.is_isolated(extension)
    proof=role.validate(extension);report=role._path(proof['previous_report'])
    report.write_text(report.read_text()+' ')
    assert not role.is_isolated(extension)


@pytest.mark.parametrize('flag,value',[
    ('--device','cuda:0'),('--max-steps',107),('--max-steps',0),('--n-envs',4),
    ('--self-choice','on'),('--resume','C:/production/checkpoint'),('--out','C:/production')])
def test_extension_cannot_hide_a_changed_or_unbounded_trainer(extension,flag,value):
    assert not role.is_isolated(tc.replace_argument(extension,flag,value))
