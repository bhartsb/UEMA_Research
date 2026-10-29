# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Audited compatibility for one unused auxiliary file in a copied study.

This does not migrate checkpoints. It proves the entire original source
inventory, documents one independent auxiliary edit and forbids that auxiliary
module from executing in the copied process. Any other difference is fatal.
"""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys

EXCLUDED = 'lem/model/causal_readout_v2.py'
MODULE = 'lem.model.causal_readout_v2'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def inventory(root):
    root=Path(root)
    paths=list((root/'mimo_bridge').glob('*.py'))
    paths += [p for p in (root/'lem').rglob('*.py') if 'tests' not in p.parts]
    paths += [root/'mimo_bridge/tools/caregiver_fit_continuation.py',
              root/'mimo_bridge/tools/caregiver_learning_trial.py']
    return {str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def verify(pin,current):
    original=pin['original_source_inventory']
    if digest(original)!=pin['saved_sha256']:
        raise ValueError('Original copied source inventory does not reproduce its saved digest')
    changed={key for key in set(original)|set(current) if original.get(key)!=current.get(key)}
    allowed=pin['changed_module']
    if allowed.replace('\\','/')!=EXCLUDED or changed!={allowed}:
        raise ValueError('A source outside the one unused auxiliary module changed')
    if current[allowed]!=pin['current_auxiliary_sha256']:
        raise ValueError('The reviewed auxiliary edit changed again')
    return pin['saved_sha256']


class BlockAuxiliary(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname==MODULE or fullname.startswith(MODULE+'.'):
            raise ImportError('This copied caregiver study forbids the independently edited auxiliary module')
        return None


def verify_actual_root(root,implementation):
    if Path(root).resolve()!=Path(implementation).resolve().parents[2]:
        raise ValueError('Source inventory root differs from the actual imported caregiver implementation')


def reject_loaded_auxiliary(modules,excluded):
    for name,module in list(modules.items()):
        filename=getattr(module,'__file__',None)
        if name==MODULE or (filename and Path(filename).resolve()==excluded):
            raise RuntimeError('Auxiliary was already loaded; source compatibility is invalid')


def install(pin_path):
    pin_path=Path(pin_path).resolve();pin=json.loads(pin_path.read_text())
    if pin.get('contract')!='caregiver_unused_auxiliary_source_pin_v1':
        raise ValueError('Unrecognized copied source compatibility contract')
    root=Path(pin['source_root']).resolve()
    verify_actual_root(root,__file__)
    saved=json.loads((Path(pin['checkpoint'])/'fit_continuation.json').read_text())
    if saved['binding']['execution_source_sha256']!=pin['saved_sha256']:
        raise ValueError('Source pin does not match the unchanged saved checkpoint binding')
    old=Path(pin['original_auxiliary_source'])
    if hashlib.sha256(old.read_bytes()).hexdigest()!=pin['original_source_inventory'][pin['changed_module']]:
        raise ValueError('Recovered original auxiliary bytes no longer match')
    verify(pin,inventory(root))
    excluded=(root/EXCLUDED).resolve()
    reject_loaded_auxiliary(sys.modules,excluded)
    sys.meta_path.insert(0,BlockAuxiliary())
    def audit(event,args):
        if event=='exec' and Path(args[0].co_filename).resolve()==excluded:
            raise RuntimeError('Edited auxiliary execution is forbidden in this copied study')
    sys.addaudithook(audit)
    from mimo_bridge.tools import caregiver_fit_continuation as continuation
    verify_actual_root(root,continuation.__file__)
    def checked_digest(extra=()):
        if extra:
            raise ValueError('Source pin only applies to the declared caregiver execution')
        reject_loaded_auxiliary(sys.modules,excluded)
        return verify(pin,inventory(root))
    continuation.execution_source_digest=checked_digest
    receipt=dict(pid=os.getpid(),pin=str(pin_path),saved_sha256=checked_digest(),
        original_source_inventory_verified=True,forbidden_module=MODULE,
        scope='Only an unused auxiliary source edit; original saved execution binding retained')
    receipts=pin_path.parent/'process_receipts';receipts.mkdir(exist_ok=True)
    (receipts/f'{os.getpid()}.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
