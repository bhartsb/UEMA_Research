# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Recognize only the explicitly declared, bounded isolated native proof.

A role flag alone never exempts a trainer from production ownership checks.
Malformed or missing declarations remain production processes.
"""
import json
import hashlib
from pathlib import Path

FLAG='--isolated-native-proof'
CONTRACT='isolated_native_proof_v1'
EXTENSION_CONTRACT='isolated_native_audit_extension_v1'


def _arg(command,flag):
    if command.count(flag)!=1:raise ValueError('proof requires exactly one '+flag)
    index=command.index(flag)
    if index+1>=len(command):raise ValueError('proof flag lacks a value: '+flag)
    return command[index+1]


def _path(value):
    path=Path(value)
    if not path.is_absolute():raise ValueError('proof paths must be absolute')
    return path.resolve()


def validate(command):
    """Return its declaration only if this precise command is an isolated proof."""
    declaration=_path(_arg(command,FLAG))
    proof=json.loads(declaration.read_text(encoding='utf-8'))
    if proof.get('schema')==EXTENSION_CONTRACT:
        return _validate_extension(command,declaration,proof)
    count=proof.get('updates_per_pass')
    if (proof.get('schema')!=CONTRACT or type(count) is not int or not 1<=count<=5
            or proof.get('passes')!=2 or proof.get('body_count')!=8
            or proof.get('production_controls_used') is not False):
        raise ValueError('not a bounded two-pass eight-body proof')
    root=_path(proof['out'])
    if declaration!=root/'declaration.json':raise ValueError('declaration does not own this proof directory')
    original=_path(proof['source'])
    receipt=json.loads((original/'continuity_migration.json').read_text(encoding='utf-8'))
    if (receipt!=proof['source_preservation'] or receipt.get('cpu_probe_only') is not True
            or receipt.get('world_count')!=8 or receipt.get('production_choice_license') is not False
            or receipt.get('old_episode_exact_except_declared_boundary') is not True
            or original==_path(receipt['parent'])):
        raise ValueError('proof must derive from a separately preserved CPU copy')
    folder=_path(_arg(command,'--out'))
    if folder not in (root/'pass_1',root/'pass_2'):raise ValueError('not a declared isolated pass')
    index=1 if folder.name=='pass_1' else 2
    initial=int(receipt['step'])
    source=original if index==1 else root/'pass_1/checkpoints'/f'ckpt_step_{initial+count}'
    expected={'--resume':source,'--checkpoint-storage':folder/'checkpoints',
              '--caregiver-record-dir':folder/'recordings',
              '--self-choice-receipt':_path(receipt['changes']['self_choice_receipt'])}
    for flag,path in expected.items():
        if _path(_arg(command,flag))!=path:raise ValueError('proof changed its isolated '+flag)
    exact={'--device':'cpu','--self-choice':'shadow','--contact-profile':'mechanical_v2_local'}
    if any(_arg(command,flag)!=value for flag,value in exact.items()):
        raise ValueError('proof changed its CPU shadow-only regime')
    if (int(_arg(command,'--n-envs'))!=8 or float(_arg(command,'--minutes'))!=0
            or int(_arg(command,'--max-steps'))!=initial+index*count):
        raise ValueError('proof changed its physical or learning budget')
    meta=json.loads((source/'meta.json').read_text(encoding='utf-8'))
    if meta['step']!=initial+(index-1)*count:raise ValueError('proof source step differs')
    return proof


def _validate_extension(command,declaration,proof):
    """Exactly two more updates of a completed CPU proof, for prepare+outcome."""
    root=_path(proof['out']);report_path=_path(proof['previous_report'])
    if (declaration!=root/'declaration.json' or proof.get('updates_per_pass')!=2
            or proof.get('passes')!=1 or proof.get('body_count')!=8
            or proof.get('production_controls_used') is not False):
        raise ValueError('not a bounded two-update audit extension')
    raw=report_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=proof['previous_report_sha256']:
        raise ValueError('completed native report changed')
    report=json.loads(raw);rows=report.get('passes',[])
    if (report.get('complete') is not True or report.get('production_choice_license') is not False
            or report.get('original_production_trainer_controlled') is not False or len(rows)!=2
            or any(r.get('world_count')!=8 or r.get('qualification_only') is not True
                   or r.get('applied_to_body') is not False for r in rows)):
        raise ValueError('extension requires the completed unqualified eight-body native proof')
    original_root=report_path.parent
    if root==original_root or root in original_root.parents or original_root in root.parents:
        raise ValueError('extension requires a separate output tree')
    original_declaration=json.loads((original_root/'declaration.json').read_text(encoding='utf-8'))
    original=_path(original_declaration['source'])
    receipt=json.loads((original/'continuity_migration.json').read_text(encoding='utf-8'))
    if (receipt!=original_declaration['source_preservation'] or receipt.get('cpu_probe_only') is not True
            or receipt.get('world_count')!=8 or receipt.get('production_choice_license') is not False
            or receipt.get('old_episode_exact_except_declared_boundary') is not True
            or original==_path(receipt['parent'])):
        raise ValueError('extension must retain original CPU-copy provenance')
    source=_path(rows[-1]['checkpoint']);initial=int(rows[-1]['step'])
    expected_source=original_root/'pass_2/checkpoints'/f'ckpt_step_{initial}'
    if source!=expected_source or _path(proof['source'])!=source:
        raise ValueError('extension source is not the completed copied checkpoint')
    folder=root/'pass_1'
    expected={'--out':folder,'--resume':source,'--checkpoint-storage':folder/'checkpoints',
              '--caregiver-record-dir':folder/'recordings',
              '--self-choice-receipt':_path(receipt['changes']['self_choice_receipt'])}
    for flag,path in expected.items():
        if _path(_arg(command,flag))!=path:raise ValueError('extension changed its isolated '+flag)
    exact={'--device':'cpu','--self-choice':'shadow','--contact-profile':'mechanical_v2_local'}
    if any(_arg(command,flag)!=value for flag,value in exact.items()):
        raise ValueError('extension changed its CPU shadow-only regime')
    if (int(_arg(command,'--n-envs'))!=8 or float(_arg(command,'--minutes'))!=0
            or int(_arg(command,'--max-steps'))!=initial+2):
        raise ValueError('extension changed its physical or learning budget')
    meta=json.loads((source/'meta.json').read_text(encoding='utf-8'))
    if meta['step']!=initial:raise ValueError('extension source step differs')
    return proof


def is_isolated(command):
    if FLAG not in command:return False
    try:validate(command)
    except (OSError,ValueError,TypeError,KeyError,IndexError):return False
    return True
