# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Implementation identity survives Python bookkeeping but rejects real changes."""
import hashlib
import json
import marshal
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from mimo_bridge.execution_contract import _bytes,_code_digest,_function_record,_module_record


def test_interning_cannot_change_the_identity_of_unchanged_code():
    code=compile("def f():\n return 'an uninterned constant with spaces 983517'\n",'<probe>','exec').co_consts[0]
    old=marshal.dumps(code)
    expected=_code_digest((('f',code),))
    sys.intern(code.co_consts[1])
    # The former serializer really has this defect on our native Python.
    assert marshal.dumps(code)!=old
    assert _code_digest((('f',code),))==expected
    altered=code.replace(co_consts=(None,'a different predicted consequence'))
    assert _code_digest((('f',altered),))!=expected


def test_equal_code_with_different_constant_aliases_has_one_identity():
    first=''.join(['repeated ','constant'])
    other=''.join(['repeated ','constant'])
    assert first==other and first is not other
    code=compile("def f():\n return None\n",'<probe>','exec').co_consts[0]
    a=code.replace(co_consts=(None,first,first))
    b=code.replace(co_consts=(None,first,other))
    assert _code_digest((('f',a),))==_code_digest((('f',b),))


def test_defaults_and_motor_tables_are_bound_without_weight_changes():
    module=ModuleType('test_motor_contract')
    exec("HANDLE_CHAINS={'hand':('shoulder','elbow')}\ndef move(scale=2):\n return HANDLE_CHAINS,scale\n",module.__dict__)
    expected=_module_record(module)
    module.HANDLE_CHAINS['hand']=('shoulder','wrist')
    assert _module_record(module)!=expected
    module.HANDLE_CHAINS['hand']=('shoulder','elbow')
    assert _module_record(module)==expected
    module.move.__defaults__=(3,)
    assert _module_record(module)!=expected


def test_code_identity_is_stable_across_fresh_hash_seeds_and_interning(tmp_path):
    script=tmp_path/'probe.py'
    script.write_text("""import sys
from mimo_bridge.execution_contract import _code_digest
code=compile("def f():\\n return {'left limb','right limb','caregiver'}\\n",'<probe>','exec').co_consts[0]
if sys.argv[1]=='intern':
 for value in code.co_consts:
  if isinstance(value,frozenset):
   for item in value: sys.intern(item)
print(_code_digest((('f',code),)))
""")
    values=[]
    for seed,mode in [('1','cold'),('2','intern')]:
        result=subprocess.run([sys.executable,str(script),mode],env=dict(os.environ,PYTHONHASHSEED=seed),
                              text=True,capture_output=True,check=True)
        values.append(result.stdout.strip())
    assert values[0]==values[1]
