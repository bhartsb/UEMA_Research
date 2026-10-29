# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Exact, future-only continuation for one declared copied caregiver fit.

The physical episode, Adam, model execution, speech accounting and order RNG
share one atomic checkpoint. Old partials lack this contract and are retained.
Resuming uses the same attempt directory so recording paths are never rebased.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from mimo_bridge import continuation_state, copied_execution

CONTRACT = 'caregiver_fit_continuation_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def execution_source_digest(extra=()):
    """A changed implementation cannot silently continue an old physical fit."""
    root=Path(__file__).resolve().parents[2]
    paths=list((root/'mimo_bridge').glob('*.py'))
    paths += [p for p in (root/'lem').rglob('*.py') if 'tests' not in p.parts]
    paths += [Path(__file__),Path(__file__).with_name('caregiver_learning_trial.py'),*extra]
    return digest({str(path.relative_to(root)):continuation_state.file_sha256(path) for path in sorted(paths)})


def binding(args,parent,out,seed,steps,limits,tape):
    parent=Path(parent).resolve()
    return dict(parent=str(parent),parent_sha256=continuation_state.file_sha256(parent/'model.safetensors'),
        out=str(Path(out).resolve()),seed=int(seed),steps=int(steps),device=str(args.device),
        config_sha256=digest(vars(args)),execution_source_sha256=execution_source_digest(),limits=deepcopy(limits),
        tape=None if tape is None else dict(path=str(Path(tape).resolve()),
            sha256=continuation_state.file_sha256(tape)))


def save(staging,final,source,model,trainer,config):
    """Capture execution once; do not duplicate the large replay in a supplement."""
    receipt=copied_execution.save_episode(staging,final,source,model,trainer,config)
    metadata=dict(contract=CONTRACT,checkpoint=str(Path(final).resolve()),
        episode_sha256=receipt['sha256'],step=trainer['step'],trial_step=trainer['trial_step'],
        binding=trainer['fit_resume']['binding'])
    (Path(staging)/'fit_continuation.json').write_text(json.dumps(metadata,indent=2,allow_nan=False),encoding='utf-8')


def load(checkpoint,expected):
    checkpoint=Path(checkpoint)
    metadata=json.loads((checkpoint/'fit_continuation.json').read_text(encoding='utf-8'))
    if metadata.get('contract')!=CONTRACT or metadata.get('binding')!=expected:
        raise ValueError('Partial caregiver fit differs from its original declared execution')
    receipt=json.loads((checkpoint/'episode_state.json').read_text(encoding='utf-8'))
    if metadata.get('episode_sha256')!=receipt.get('sha256'):
        raise ValueError('Partial caregiver fit is not bound to its physical episode')
    state=continuation_state.load(checkpoint)
    trainer=state['trainer'];resume=trainer.get('fit_resume',{})
    if (resume.get('contract')!=CONTRACT or resume.get('binding')!=expected or
            trainer['trial_step']!=metadata['trial_step'] or trainer['step']!=metadata['step'] or
            trainer['step']!=resume['parent_step']+trainer['trial_step'] or
            not 0<=trainer['trial_step']<=expected['steps'] or
            not isinstance(resume.get('utterances'),list) or 'last_utterance' not in resume or
            not resume.get('initial_encoder_sha256') or resume.get('active_wall_seconds',-1)<0 or
            not trainer.get('fit_execution')):
        raise ValueError('Partial caregiver fit lacks exact learning/order/speech continuation')
    return state


def restore(model,source,orders,state):
    trainer=state['trainer']
    for name in ('held','executed','order','bout','until'):
        setattr(orders,name,continuation_state.tree(trainer[name]))
    orders.rng.bit_generator.state=deepcopy(trainer['bout_rng'])
    # Last: constructors/restored physics must not consume the saved next random draw.
    return copied_execution.restore_episode(model,source,state)


def latest(base,make_expected):
    """Inspect tiny manifests; fully validate the selected state when loading it.

    A manifest from another protocol is an error, not permission to restart and
    discard its evidence. Legacy checkpoints without manifests remain untouched.
    Multiple eligible attempt lineages require review; never select a winner.
    """
    base=Path(base);found=[]
    for folder in ([base,*sorted(base.glob('attempt_*'))] if base.exists() else []):
        candidates=[];expected=None
        for path in folder.glob('checkpoints/*/fit_continuation.json'):
            metadata=json.loads(path.read_text(encoding='utf-8'))
            if expected is None:expected=make_expected(folder)
            if metadata.get('contract')!=CONTRACT or metadata.get('binding')!=expected:
                raise ValueError('Preserved partial fit has a different execution contract')
            checkpoint=path.parent
            receipt_path=checkpoint/'episode_state.json'
            if not receipt_path.is_file():raise ValueError('Preserved partial fit lost its episode receipt')
            receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
            if metadata['episode_sha256']!=receipt.get('sha256'):
                raise ValueError('Preserved partial fit episode binding changed')
            candidates.append((int(metadata['trial_step']),checkpoint))
        if candidates:
            _,checkpoint=max(candidates,key=lambda row:row[0]);found.append((folder,checkpoint))
    if len(found)>1:raise ValueError('Multiple resumable caregiver fit lineages; do not select a winner')
    return found[0] if found else None


def segment(folder):
    """Append process/control receipts; old pause and stop requests stay intact."""
    root=Path(folder)/'segments';root.mkdir(exist_ok=True)
    index=1
    while (root/f'segment_{index:04d}').exists():index+=1
    path=root/f'segment_{index:04d}';path.mkdir()
    return path
