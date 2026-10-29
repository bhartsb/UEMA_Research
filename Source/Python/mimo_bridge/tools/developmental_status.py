# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Connect the complete ability definitions to actual scheduled experiments."""
from pathlib import Path
import hashlib
from .experiment_board import read
from mimo_bridge.developmental_evidence import assess_comparison


def checked_artifact(root,folder,reference):
    if not isinstance(reference,dict) or not {'path','sha256'}.issubset(reference):
        raise ValueError('Evidence needs a located artifact and its SHA-256')
    path=Path(reference['path']);path=(path if path.is_absolute() else folder/path).resolve()
    roots=(Path(root).resolve(),folder.resolve())
    if not any(path.is_relative_to(bound) for bound in roots):
        raise ValueError('Evidence artifact leaves the registered experiment/repository')
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']:
        raise ValueError('Evidence artifact is missing or changed')
    return path


def scope_receipt(root,folder,reference,kind,ability):
    path=checked_artifact(root,folder,reference);record=read(path)
    if record.get('contract')!='developmental_scope_v1' or record.get('kind')!=kind or record.get('ability_id')!=ability:
        raise ValueError('Evidence scope receipt belongs to a different comparison')
    if record.get('verified') is not True or not record.get('artifacts'):
        raise ValueError('A bare verification flag is not a located execution receipt')
    for artifact in record['artifacts']:checked_artifact(root,path.parent,artifact)
    if kind=='learning_comparison':
        updates=record.get('learning_updates')
        if isinstance(updates,bool) or not isinstance(updates,int) or updates<=0:
            raise ValueError('Acquisition needs actual intervening learning')
        before,after=record.get('before_checkpoint_sha256'),record.get('after_checkpoint_sha256')
        if any(not isinstance(v,str) or len(v)!=64 for v in (before,after)) or before==after:
            raise ValueError('Acquisition needs two identified, different learned checkpoints')
    elif kind=='retention_without_training':
        if record.get('optimizer_updates')!=0 or record.get('after_intervening_experience') is not True:
            raise ValueError('Retention needs a later no-fitting test')
    elif kind=='held_out_transfer':
        training,held=record.get('training_worlds'),record.get('held_out_worlds')
        if not isinstance(training,list) or not isinstance(held,list) or not held or set(training)&set(held):
            raise ValueError('Transfer worlds must be held out from learning')
        if record.get('split_declared_before_test') is not True:
            raise ValueError('Transfer split was not declared before testing')
    elif kind=='physical_calibration':
        if record.get('reachability_verified') is not True or record.get('same_action_replay_exact') is not True:
            raise ValueError('No executed physical reachability/replay reference')
    return path


def ingest_comparison(root,folder,ability):
    """Recompute the strict evaluator from hash-bound inputs, never trust a green flag."""
    folder=Path(folder)
    try:
        report=read(folder/'report.json');bundle=report.get('developmental_evidence')
        if bundle is None:return dict(state='unassessable',capability_pass=False,reason='Completed report has no complete developmental comparison bundle.')
        if bundle.get('contract')!='developmental_comparison_v1' or bundle.get('ability_id')!=ability:
            raise ValueError('Comparison bundle names another ability or contract')
        declaration=read(checked_artifact(root,folder,bundle['declaration']))
        if declaration.get('ability_id')!=ability:raise ValueError('Declaration names another ability')
        refs=bundle.get('trials')
        if not isinstance(refs,list) or not 1<=len(refs)<=4096:raise ValueError('Missing or excessive comparison traces')
        trials=[read(checked_artifact(root,folder,ref)) for ref in refs]
        receipts=bundle.get('scope_receipts',{})
        for kind in ('learning_comparison','retention_without_training','held_out_transfer','physical_calibration'):
            path=scope_receipt(root,folder,receipts[kind],kind,ability)
            claimed=(declaration.get('calibration',{}).get('receipt') if kind=='physical_calibration'
                     else declaration.get('evidence_scope',{}).get(kind,{}).get('receipt'))
            if not claimed:raise ValueError('Declaration lacks a required scope receipt')
            claimed=Path(claimed);claimed=(claimed if claimed.is_absolute() else folder/claimed).resolve()
            if claimed!=path:raise ValueError('Declaration receipt differs from its verified artifact')
        result=assess_comparison(trials,declaration)
        result['artifacts_verified']=True
        result['report_sha256']=hashlib.sha256((folder/'report.json').read_bytes()).hexdigest()
        return result
    except (OSError,ValueError,TypeError,KeyError) as error:
        return dict(state='unassessable',capability_pass=False,artifacts_verified=False,reason=str(error))


def ability_status(relevant):
    """One plain sentence: what was measured, on which lineage, and what that leaves open.

    Replaces "Test queued" / "Results available" / "Test definition ready" (status shorthand:
    every queued job and every result belonged to the H69-H80 lineages, whose result folders
    on D: are gone, while the running H84 model had none of these tests)."""
    running=[j for j in relevant if j.get('state')=='running']
    complete=[j for j in relevant if j.get('completed_at')]
    here=[j for j in complete if j.get('lineage')=='ancestor_checkpoint']
    other=[j for j in complete if j.get('lineage')=='not_linked_to_current_model']
    missing=[j for j in relevant if j.get('state')=='files_missing']
    tests=lambda n:f'{n} finished test{"s" if n!=1 else ""}'
    if running:
        return 'A measurement is running now; no result yet'
    if here:
        return 'Measured on this model\'s ancestry; acquisition, retention and transfer not established'
    if other:
        return f'Tested on an earlier lineage, not on the running model ({tests(len(complete))})'
    if complete:
        return f'Tested ({tests(len(complete))}); the tested model is not recorded, so it is not linked to the running model'
    if missing:
        return (f'The result files of {len(missing)} test{"s" if len(missing)!=1 else ""} are missing '
                f'(their registered folders no longer exist)')
    if relevant:
        return 'Planned; never run on the running model'
    return 'No test has run on any model; the test is defined in the requirements'


def build(root, work, observations=None):
    path=Path(root)/'docs/gain_function_2026-09-13/requirements.json'
    if not path.is_file():return dict(stages=[],error='Ability definitions are unavailable.')
    spec=read(path);jobs=work.get('jobs',[]);stages=[]
    from .experiment_board import registry,directory
    registered={job['id']:job for job in registry(root).get('jobs',[])}
    observed={row['id']:row for row in (observations or []) if row.get('kind')=='live_behavior'}
    for number,title in spec['tier_names'].items():
        abilities=[]
        for definition in spec['abilities']:
            if definition['tier']!=int(number):continue
            relevant=[j for j in jobs if definition['id'] in j.get('abilities',[])]
            complete=[j for j in relevant if j.get('completed_at')]
            status=ability_status(relevant)
            outcomes=[]
            for job in complete:
                source=registered.get(job['id'],{})
                if source.get('run_dir') and job.get('state') not in ('cancelled','record_error','result_missing'):
                    result=ingest_comparison(root,directory(root,source),definition['id'])
                    outcomes.append(dict(experiment_id=job['id'],**result))
            acquired=any(row.get('capability_pass') is True and row.get('artifacts_verified') is True for row in outcomes)
            measured=[dict(experiment_id=job['id'],summary=observed[job['id']]['summary'],
                limitation=observed[job['id']]['missing'],model_identity=observed[job['id']]['model_identity'],
                sources=observed[job['id']]['sources']) for job in complete if job['id'] in observed and
                observed[job['id']].get('valid') and observed[job['id']].get('lineage')=='ancestor_checkpoint']
            if measured:status='Observed on this model\'s ancestry; acquisition, retention and transfer not established'
            if acquired:status='Acquisition, retention and transfer passed their declared comparisons'
            # Completion alone never earns a gain; complete physical comparisons can.
            abilities.append(dict(id=definition['id'],name=definition['name'],status=status,
                acquired=acquired,outcomes=outcomes,observations=measured,minimum_environment=definition['minimum_environment']['description'],
                driver=definition['natural_driver'],human_comparison=definition['human_comparison']['text'],
                question=definition['tests'][0]['question'],
                prerequisite=definition['implementation_gap'],
                experiments=[dict(id=j['id'],title=j['title'],state=j.get('state'),lineage=j.get('lineage'),
                                  failure=j.get('failure'),links=j.get('links',[])) for j in relevant]))
        stages.append(dict(number=int(number),title=title,abilities=abilities))
    # The 25 abilities are the 2026-09-13 registry; its definitions page is the September 13 revision
    # (MIMo_Gain_of_Function_Tree_2026-09-06.html, "revised September 13"). The 2026-09-17 tree is an
    # old plan whose dates run September 16 to 19 (dashboard staleness check, 2026-09-24).
    return dict(stages=stages,total=len(spec['abilities']),
        policy='A completed measurement is not a retained ability. Acquisition, retention and transfer require the defined controlled comparisons.',
        definitions='/files/docs/MIMo_Gain_of_Function_Tree_2026-09-06.html',
        definitions_label='Definitions and tests (revised 2026-09-13, kept as history)')
