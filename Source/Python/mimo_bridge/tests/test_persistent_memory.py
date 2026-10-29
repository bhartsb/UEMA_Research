# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
import copy
import math
from types import SimpleNamespace
import torch
import pytest
from lem.model.persistent_memory import PersistentMemory, MemoryHistory
from mimo_bridge.episode_continuity import physical_fault, scheduled_reset


def model(): return PersistentMemory(8,2,4,3,hidden_size=12)


def test_new_memory_preserves_parent_and_has_a_trainable_write_path():
    torch.manual_seed(3);m=model()
    x=torch.randn(2,30,8,requires_grad=True);t=torch.arange(30.).expand(2,-1)
    state,_=m.encode(x,t,t[:,-1])
    assert m.bias(state).count_nonzero() == 0
    m.predict_needs(state,torch.ones(2,3)).square().mean().backward()
    assert m.recurrent.weight_ih_l0.grad.abs().sum() > 0
    assert m.input_projection[0].weight.grad.abs().sum() > 0
    assert x.grad[:,0].abs().sum() > 0  # learning reaches the earlier event


def test_memory_body_independence_and_no_future_access():
    m=model();x=torch.randn(2,20,8);t=torch.arange(20.).expand(2,-1)
    a,_=m.encode(x,t,t[:,-1]);y=x.clone();y[1]=100
    b,_=m.encode(y,t,t[:,-1]);torch.testing.assert_close(a[0],b[0])
    with pytest.raises(ValueError): m.encode(x,t,t[:,-1]-1)
    with pytest.raises(ValueError): m.encode(x,t.flip(1),t[:,-1])


def test_history_is_bounded_keeps_short_events_and_roundtrips():
    m=model();h=MemoryHistory(capacity=4,interval_s=.5)
    for i in range(100):
        x=torch.zeros(2,8)
        if i==92: x[0,0]=1
        h.observe(x,torch.full((2,),i*.05),m)
    assert len(h.events) == 4
    assert h.carry is not None
    snap=h.snapshot()
    assert snap['events'][0,:,0].max() == 1
    restored=MemoryHistory(4,.5);restored.load_state_dict(h.state_dict(),'cpu')
    for k in ('events','times','now','carry','carry_time'):
        torch.testing.assert_close(snap[k],restored.snapshot()[k])
    h.reset();assert h.snapshot() is None and h.carry is None


def test_elapsed_time_is_an_input_and_cannot_read_a_future_outcome():
    m=model();x=torch.zeros(1,4,8);x[0,0,0]=1
    t=torch.tensor([[0.,1.,2.,3.]])
    a,_=m.encode(x,t,t[:,-1]);b,_=m.encode(x,2*t,2*t[:,-1])
    assert not torch.allclose(a,b)
    h=MemoryHistory(4,.5);h.observe(x[:,0],t[:,0],m)
    snap=h.snapshot();before=snap['events'].clone()
    h.observe(torch.full_like(x[:,0],99.),t[:,1],m)
    torch.testing.assert_close(snap['events'],before)


def test_prefix_elapsed_time_is_preserved_after_history_eviction():
    m=model();x=torch.zeros(1,1,8);times=torch.tensor([[10.]])
    carry=torch.ones(1,1,m.hidden_size)
    a,_=m.encode(x,times,times[:,0],carry,torch.tensor([9.]))
    b,_=m.encode(x,times,times[:,0],carry,torch.tensor([1.]))
    assert not torch.allclose(a,b)
    with pytest.raises(ValueError): m.encode(x,times,times[:,0],carry,torch.tensor([11.]))


def test_episode_zero_really_disables_scheduled_resets_but_not_fault_detection():
    assert not any(scheduled_reset(i,0) for i in (2000,4000,6000))
    assert scheduled_reset(2000,2000)
    f=[SimpleNamespace(sim_time=101.,meta={'physical_state_finite':True})]
    assert physical_fault(f,[100.95]) is None
    assert physical_fault(f,[200.]) == 'physics clock moved backwards'
    f[0].meta['physical_state_finite']=False
    assert 'nonfinite' in physical_fault(f)


def test_memory_is_registered_and_reaches_the_real_forward_prediction(tmp_path):
    from mimo_bridge.lem_model import LemLite
    m=LemLite(device='cpu',seed=4,imagination=False,next_frame_loss=True,
              persistent_memory='on',checkpoint_dir=str(tmp_path))
    mem=m.persistent_memory;opt_ids={id(p) for g in m.optimizer.param_groups for p in g['params']}
    assert all(id(p) in opt_ids for p in mem.parameters())
    m.forward_model.eval()
    tokens={k:torch.randn(2,1,m.d_model) for k in m.active}
    events=torch.randn(2,30,mem.input_size)
    times=torch.arange(30.).expand(2,-1)
    state,_=mem.encode(events,times,times[:,-1])
    ordinary=m.forward_model(tokens)[1]
    installed=m.forward_model(tokens,memory_bias=mem.bias(state))[1]
    torch.testing.assert_close(ordinary,installed,rtol=0,atol=0)
    with torch.no_grad(): mem.readout.weight.fill_(.01)
    state,_=mem.encode(events,times,times[:,-1])
    changed=m.forward_model(tokens,memory_bias=mem.bias(state))[1]
    assert not torch.allclose(ordinary,changed)
    changed.square().mean().backward()
    assert mem.recurrent.weight_ih_l0.grad.abs().sum()>0
    assert mem.input_projection[0].weight.grad.abs().sum()>0
    path=m.save_checkpoint(0)
    restored=LemLite(device='cpu',seed=9,imagination=False,next_frame_loss=True,persistent_memory='on')
    loaded=restored.load_checkpoint(path)
    assert not loaded['reinitialized_components']
    for k,v in mem.state_dict().items(): torch.testing.assert_close(v,restored.persistent_memory.state_dict()[k],rtol=0,atol=0)
    without=LemLite(device='cpu',seed=9,imagination=False,next_frame_loss=True)
    with pytest.raises(ValueError,match='memory'): without.load_checkpoint(path)


def test_saved_plan_owns_its_memory_for_all_alternative_actions():
    from mimo_bridge.lem_model import LemLite
    # The existing contract test covers context, achieved state and execution
    # smoothing. Exercise the newly added input independently of those fields.
    m=LemLite(device='cpu',imagination=False,intention=True)
    m._memory_bias_current=torch.ones(1,m.n_mod,m.d_model)
    decision=m._capture_plan_state()
    m._memory_bias_current.fill_(7)
    assert decision['memory_bias'].eq(1).all()


def test_intermittent_physical_failures_cannot_restart_forever():
    from mimo_bridge.episode_continuity import PhysicalFaultBudget
    budget=PhysicalFaultBudget()
    assert not budget.record(10)
    assert not budget.record(12)
    assert budget.record(14)
    assert not budget.record(2015)
    assert physical_fault([])=='empty observation batch'
    f=[SimpleNamespace(sim_time=10.,meta={})]
    assert physical_fault(f,[10.])=='physics clock did not advance'


def test_the_memory_bias_lands_only_on_the_newest_moment_of_the_core_output():
    """It is added AFTER the final LayerNorm, to the newest moment's output tokens.
    Every earlier position, and the context handed to the next step, are untouched -
    so the transformer never attends to the recall on the step it is made. Without
    this, a bias applied to the INPUT tokens instead would pass every other test."""
    from lem.model.forward_model import ForwardModel
    mods=['vision','tactile','audio','proprioception','motor','intermind']
    fm=ForwardModel({"model":{"d_model":16,"context_length":4,
            "forward_model":{"_n_modalities":6,"_seq_length":24,"n_heads":2,
                             "ffn_expansion":2,"n_layers":1,
                             "stages":{"cross_modal":True,"causal_temporal":True,
                                       "cross_modal_bridge":True}}},
        "_active_modalities":mods}).eval()
    torch.manual_seed(0)
    tokens=torch.randn(2,6,16);ctx=torch.randn(2,3*6,16);bias=torch.randn(2,6,16)*.2
    with torch.no_grad():
        plain,_,plain_ctx=fm(tokens,ctx)
        biased,biased_last,biased_ctx=fm(tokens,ctx,memory_bias=bias)
    torch.testing.assert_close(biased[:,:-6],plain[:,:-6],rtol=0,atol=0)
    torch.testing.assert_close(biased[:,-6:]-plain[:,-6:],bias,rtol=0,atol=1e-6)
    assert torch.equal(biased_last,biased[:,-6:])
    assert torch.equal(plain_ctx,biased_ctx)     # the next step's context never sees it
    with torch.no_grad(): _,input_last,input_ctx=fm(tokens+bias,ctx)
    assert not torch.allclose(input_last,biased_last)
    assert not torch.equal(plain_ctx,input_ctx)


def test_the_production_ring_keeps_128_events_one_per_half_second():
    m=model();h=MemoryHistory()
    assert (h.capacity,h.interval_s) == (128,.5)
    for i in range(3000): h.observe(torch.randn(1,8)*.01,torch.full((1,),i*.05),m)
    assert len(h.events) == h.capacity
    snap=h.snapshot()
    assert snap['events'].shape[1] == h.capacity+2   # ring + the open interval + now
    age=float((snap['now'][:,None]-snap['times']).max())
    assert h.capacity*h.interval_s-.5 <= age <= h.capacity*h.interval_s+.5
    assert h.carry is not None and h.carry.requires_grad is False


def test_the_lookup_is_the_present_querying_every_retained_summary():
    m=model();x=torch.randn(1,30,8);t=(torch.arange(30.)*.5).expand(1,-1)
    state,states=m.encode(x,t,t[:,-1])
    w=torch.softmax((m.query(x[:,-1]).unsqueeze(1)*m.key(states)).sum(-1)
                    /math.sqrt(m.hidden_size),dim=1)
    assert w.shape[1] == x.shape[1]          # every retained position...
    assert float(w[0,-1]) > 0                # ...including the present observation
    torch.testing.assert_close(w.sum(1),torch.ones(1))
    torch.testing.assert_close(state,torch.cat([states[:,-1],
                                                (w.unsqueeze(-1)*states).sum(1)],-1))
    with torch.no_grad(): m.key.weight.mul_(5.)
    assert not torch.allclose(state,m.encode(x,t,t[:,-1])[0])   # not a fixed average


def test_a_trained_readout_is_never_silent_even_on_a_perfectly_still_body():
    """'zero when it has nothing to say' is true only of a freshly built memory."""
    m=model();h=MemoryHistory(capacity=8,interval_s=.5);still=torch.zeros(1,8)
    for i in range(200): h.observe(still,torch.full((1,),i*.05),m)
    assert m.bias(m.encode(**h.snapshot())[0]).count_nonzero() == 0
    with torch.no_grad(): m.readout.weight.normal_(0,.05);m.readout.bias.normal_(0,.05)
    assert float(m.bias(m.encode(**h.snapshot())[0]).square().mean().sqrt()) > 1e-3


def test_the_first_entry_of_an_episode_is_the_first_frame_not_a_biggest_change_pick():
    m=model();h=MemoryHistory(capacity=8,interval_s=.5)
    for i in range(12):
        x=torch.zeros(1,8)
        if i == 7: x=x+5.
        h.observe(x,torch.full((1,),i*.05),m)
    assert float(h.times[0]) == 0. and float(h.events[0].abs().max()) == 0.
    assert float(h.times[1]) == pytest.approx(.35,abs=1e-5)   # the spike at step 7
    assert float(h.events[1].max()) == 5.


def test_the_memorys_pressure_and_shear_channels_are_the_normalised_skin_units():
    """The memory's five bodily-need channels are pain, pleasantness, capacity, pressure
    and shear. The last two are NOT Pascals: skin feature channel 0 arrives divided by
    100,000 Pa and channel 1 by 50,000 Pa (tokenizer/modalities/touch.py _FIELD_NORMS,
    mimo_bridge/skin_contract.py), and lem/model/contact_wellness.py multiplies channel 0
    back by 100000 to recover Pascals. The x/(1+x) squash applied here was written for a
    quantity of order 1 and is the identity to within 1.37% at the pressures this body
    meets, so these two channels occupy about 1.4% of the 0-to-1 range the contract
    states. Anything that changes that scale changes what the needs-forecast head is
    taught, and the persistent memory does not raise on a silent change of a channel's
    scale the way it raises on a layout change - so it is pinned here.
    """
    from tokenizer.modalities.touch import _FIELD_NORMS
    from mimo_bridge.memory_integration import measured_needs

    assert float(_FIELD_NORMS[0]) == 100000.0 and float(_FIELD_NORMS[1]) == 50000.0

    pressure_pa, shear_pa = 1000.0, 1000.0
    features = torch.zeros(1, 4, 10)
    features[0, 0, 0] = pressure_pa / float(_FIELD_NORMS[0])      # 0.01
    features[0, 0, 1] = shear_pa / float(_FIELD_NORMS[1])         # 0.02
    extras = {"raw_tactile": {"features": features, "sensitivity": None,
                              "valid": None, "profile": "legacy", "history": None},
              "capability": torch.ones(1, 12)}
    values, valid = measured_needs(extras, 1, torch.device("cpu"))

    assert values.shape == (1, 5) and bool(valid.all())
    assert float(values[0, 3]) == pytest.approx(pressure_pa / float(_FIELD_NORMS[0]),
                                                rel=0.02), (
        f"the memory's pressure channel reads {float(values[0, 3]):.5f} for {pressure_pa} "
        f"Pa; the contract is Pa/{float(_FIELD_NORMS[0]):.0f} with a near-identity squash")
    assert float(values[0, 4]) == pytest.approx(shear_pa / float(_FIELD_NORMS[1]),
                                                rel=0.02), (
        f"the memory's shear channel reads {float(values[0, 4]):.5f} for {shear_pa} Pa; "
        f"the contract is Pa/{float(_FIELD_NORMS[1]):.0f}")
    # H83 measured 921.5 to 1389.5 Pa of mean active-patch pressure over 137 records,
    # which is 0.0092 to 0.0139 here - the bottom 1.4% of a 0-to-1 channel
    assert 0.005 < float(values[0, 3]) < 0.05
