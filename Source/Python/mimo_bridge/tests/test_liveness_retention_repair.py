# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""A repair must survive a second collapse and preserve the contact residual path."""
from types import SimpleNamespace
from pathlib import Path
import json
import runpy
import copy

import pytest
import torch
from lem.model import liveness as LV
from lem.model.contact_head import ContactHead
from test_liveness import Toy


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_reset_rows_match_fresh_adam_and_preserve_the_other_rows(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.manual_seed(11)
    row=LV.RevivedRows(3,4).to(device)
    opt=torch.optim.AdamW(row.lin.parameters(),lr=.002,weight_decay=.03,fused=device=='cuda')
    for p in row.lin.parameters():p.grad=torch.ones_like(p)*.2
    opt.step()
    # A mature tensor, with only its first two rows freshly restarted.
    for p,birth in ((row.lin.weight,row.adam_birth_weight),(row.lin.bias,row.adam_birth_bias)):
        opt.state[p]['step'].fill_(10000)
        opt.state[p]['exp_avg'][:2]=0
        opt.state[p]['exp_avg_sq'][:2]=0
        birth[:2]=10000
    row.adam_correction_py=True
    untouched=copy.deepcopy(row.lin)
    control=torch.optim.AdamW(untouched.parameters(),lr=.002,weight_decay=.03,fused=device=='cuda')
    control.load_state_dict(copy.deepcopy(opt.state_dict()))
    fresh=[torch.nn.Parameter(p[:2].detach().clone()) for p in row.lin.parameters()]
    fresh_opt=torch.optim.AdamW(fresh,lr=.002,weight_decay=.03,fused=device=='cuda')
    remove=LV.install_row_adam(opt,[row])
    # Execution snapshots replace buffers, so callbacks must resolve the current ones.
    for key in ('adam_birth_weight','adam_birth_bias'):
        stale=getattr(row,key)
        row._buffers[key]=stale.clone()
        stale.zero_()
    for step in range(8):
        for p,q,f in zip(row.lin.parameters(),untouched.parameters(),fresh):
            grad=torch.sin(torch.arange(p.numel(),device=device).reshape_as(p)+step+.3)
            if step==0:grad=grad*1e-8  # epsilon matters; a scalar update multiplier is insufficient
            p.grad=grad;q.grad=grad.clone();f.grad=grad[:2].clone()
        opt.step();control.step();fresh_opt.step()
        for p,q,f in zip(row.lin.parameters(),untouched.parameters(),fresh):
            torch.testing.assert_close(p[:2],f,atol=1e-7,rtol=1e-6)
            assert torch.equal(p[2:],q[2:]),'an untouched row changed'
        if step==3:
            # A saved repair retains its per-row ages across checkpoint reload.
            remove()
            state,history=copy.deepcopy(row.state_dict()),copy.deepcopy(opt.state_dict())
            row=LV.RevivedRows(3,4).to(device);row.load_state_dict(state);row.adam_correction_py=True
            opt=torch.optim.AdamW(row.lin.parameters(),lr=.002,weight_decay=.03,fused=device=='cuda')
            opt.load_state_dict(history)
            remove=LV.install_row_adam(opt,[row])
    if device=='cuda':
        before=[p.detach().clone() for p in row.lin.parameters()]
        opt.grad_scale=torch.tensor(1.,device=device)
        opt.found_inf=torch.tensor(1.,device=device)
        opt.step()
        assert all(torch.equal(p,b) for p,b in zip(row.lin.parameters(),before))
        del opt.grad_scale,opt.found_inf
    remove()


def test_a_weight_only_load_can_start_a_fresh_optimizer():
    row=LV.RevivedRows(3,4)
    row.adam_birth_weight.fill_(10000);row.adam_birth_bias.fill_(10000)
    row.adam_correction_py=True
    opt=torch.optim.AdamW(row.lin.parameters(),lr=.001)
    remove=LV.install_row_adam(opt,[row])
    for p in row.lin.parameters():p.grad=torch.ones_like(p)
    opt.step()
    assert not row.adam_birth_weight.any() and not row.adam_birth_bias.any()
    assert all(torch.isfinite(p).all() for p in row.lin.parameters())
    remove()


def test_optimizer_registration_is_idempotent_and_detaches():
    model=Toy(dead=12)
    rep=LV.LivenessRepair(model,'on',layers=('head.0',),pools=())
    model.optimizer=torch.optim.AdamW(list(model.head.parameters())+list(rep.parameters()))
    for _ in range(3):rep.attach_optimizer(model)
    assert len(model.optimizer._optimizer_step_pre_hooks)==1
    assert len(model.optimizer._optimizer_step_post_hooks)==1
    rep.detach()
    assert not model.optimizer._optimizer_step_pre_hooks
    assert not model.optimizer._optimizer_step_post_hooks
    rep.attach_optimizer(model)
    assert len(model.optimizer._optimizer_step_pre_hooks)==1
    rep.detach()


def test_previously_repaired_silent_units_are_detected_and_repaired_again():
    model = Toy(dead=12)
    rep = LV.LivenessRepair(model, 'on', layers=('head.0',), pools=(), revive_min_samples=64)
    rep.attach_repair(model)
    x = model.inputs(64, 0)
    model.head(x)
    rep.end_step(model, 0, 64, train=True)
    rows = rep.rows['head__0']
    repaired = rows.mask.nonzero().flatten()
    assert len(repaired) >= 12
    with torch.no_grad():
        rows.lin.weight[repaired] = 0
        rows.lin.bias[repaired] = -20
        model.head[2].weight[:, repaired] = .3
    before = model.head(x).detach()
    rep.end_step(model, 1, 64, train=True)
    rec = rep.revival_record['head.0']
    assert rec['repaired_again'] == len(repaired)
    after = model.head(x).detach()
    assert torch.allclose(before, after, atol=1e-6)
    assert rec['revived_live_share_on_probe'] > .5
    assert rep.completed_windows['head.0']==2
    rep.after_load(model)
    assert not rep.window_summary(model)['watching']['head.0']['ready']
    model.head(x)
    rep.end_step(model,2,64,train=True)
    assert rep.window_summary(model)['watching']['head.0']['ready']
    rep.detach()


def test_contact_repair_preserves_logits_and_the_original_skip_path():
    torch.manual_seed(5)
    head = ContactHead(8, 2, 3, n_patches=4, hidden=16, patch_hidden=8, patch_embed=4)
    model = SimpleNamespace(contact_head=head, receipt={})
    with torch.no_grad():
        head.ctx_lat.weight[:12] = 0
        head.ctx_lat.bias[:12] = -20
        head.ctx_ord.weight[:12] = 0
        head.ctx_ord.bias[:12] = 0
    latent, order, touch = torch.randn(64, 2, 8), torch.randn(64, 3, 7), torch.rand(64, 4, 2)
    original = {k:v.clone() for k,v in head.state_dict().items()}
    before = head(latent, order, touch).detach()
    rep = LV.LivenessRepair(model, 'on', layers=('contact_head.ctx1',), pools=(), revive_min_samples=64)
    rep.attach_repair(model)
    assert torch.equal(before, head(latent, order, touch).detach())
    rep.end_step(model, 0, 64, train=True)
    after, pre = head(latent, order, touch, return_pre=True)
    assert torch.allclose(before, after, atol=1e-6)
    assert (pre['ctx1'] > LV.LIVE_LINE).any(0).float().mean() > .5
    for key in ('ctx_lat.weight','ctx_lat.bias','ctx_ord.weight','ctx_ord.bias','p_ctx.weight'):
        assert torch.equal(head.state_dict()[key], original[key]), key
    # The repaired input receives learning through ctx2 once its zeroed columns learn.
    model.optimizer = torch.optim.AdamW(list(head.parameters()) + list(rep.parameters()), lr=1e-3)
    for _ in range(3):
        model.optimizer.zero_grad()
        head(latent, order, touch).square().mean().backward()
        model.optimizer.step()
    assert rep.rows['contact_head__ctx1'].lin.weight.grad.abs().sum() > 0
    rep.detach()


def _report(step, share, ready=True, samples=112):
    return dict(step=step,liveness=dict(mode='on', samples=samples,window_steps=100,
        layers={'tenure_heads.pose.0':share}, watching={'tenure_heads.pose.0':{'ready':ready}}))


def test_watcher_stops_persistent_failure_but_not_startup_or_one_bad_reading():
    path = Path(__file__).resolve().parents[4]/'docs/h84_launch_2026-09-18/hourly_watch.py'
    check = runpy.run_path(str(path))['liveness_breach']
    assert check([_report(100,0),_report(200,0)])
    assert check([_report(100,0),_report(100,0)]) is None
    assert check([_report(100,0,False),_report(200,0)]) is None
    assert check([_report(100,0),_report(200,1)]) is None
    assert check([_report(100,0,samples=0),_report(200,0)]) is None
    assert check([_report(100,float('nan')),_report(200,0)]) is None
    reports=[_report(100,1),_report(200,1)]
    for r in reports:r['synergy']={'liveness':{'S1':0}}
    assert 'synergy.S1' in check(reports)


def test_watcher_preserves_live_lesson_coverage_beside_generic_hooks():
    path=Path(__file__).resolve().parents[4]/'docs/h84_launch_2026-09-18/hourly_watch.py'
    check=runpy.run_path(str(path))['liveness_breach']
    def reports():
        result=[_report(100,1),_report(200,1)]
        for row in result:
            row['liveness']['layers']['efference.body_mlp.0']=.9
            row['efference']={'liveness':{'body_mlp.0':.45,'vision_in':.95}}
        return result
    assert 'efference.lesson.body_mlp.0' in check(reports())
    partial=reports();partial[0]['liveness']['window_steps']=99
    assert check(partial) is None
    no_input=reports();no_input[0]['efference']['liveness']={}
    assert check(no_input) is None
    startup=reports()
    startup[0]['liveness']['watching']['efference.body_mlp.0']={'ready':False}
    assert check(startup) is None
    recovered=reports();recovered[-1]['efference']['liveness']['body_mlp.0']=.6
    assert check(recovered) is None
    # A generic failure remains visible even if lesson inputs were responsive.
    other=reports()
    for row in other:
        row['liveness']['layers']['efference.body_mlp.0']=.4
        row['efference']['liveness']['body_mlp.0']=.9
    assert 'efference.body_mlp.0:' in check(other)


def test_repair_stop_pauses_only_its_run_and_keeps_existing_controls(tmp_path):
    path=Path(__file__).resolve().parents[4]/'docs/h84_launch_2026-09-18/hourly_watch.py'
    stop=runpy.run_path(str(path))['stop_run']
    run=tmp_path/'run';run.mkdir()
    stop.__globals__.update(RUN=str(run),SAVED=tmp_path)
    state=tmp_path/'training_continuation.json'
    state.write_text(json.dumps({'active_run':str(run)}))
    control=tmp_path/'supervisor_control.json'
    control.write_text(json.dumps({'cmd':'run','archiver':'off'}))
    stop('test persistent failure')
    assert json.loads(control.read_text())['cmd']=='pause'
    assert json.loads(control.read_text())['archiver']=='off'
    assert json.loads((run/'control.json').read_text())['cmd']=='snapshot_and_stop'
    assert json.loads((run/'stop_reason.json').read_text())['resume_policy']=='repair_required'
    state.write_text(json.dumps({'active_run':str(tmp_path/'another_run')}))
    control.write_text(json.dumps({'cmd':'run'}))
    stop('stale watcher')
    assert json.loads(control.read_text())['cmd']=='run'
