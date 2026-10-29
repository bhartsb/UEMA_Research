# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Value further learnable distinctions while bodily function remains viable.

This observer receives completed prediction comparisons in a fixed encoder's
coordinates. It changes no reward labels, model weights or physical sensations.
Fresh action controls and improvement beyond retained-anchor historical best
are both required. Raw surprise and recovered old competence do not suffice.
"""
from collections import deque
import copy
import math
import numpy as np

CONTRACT='controlled_learning_opportunity_v1'


class LearningOpportunity:
    def __init__(self,window,effect_floors,calibration,groups=None,require_retention=False):
        if int(window)!=window or window<2 or not str(calibration).strip():
            raise ValueError('learning opportunity needs declared evidence window and calibration')
        if not effect_floors or any(not math.isfinite(v) or v<=0 for v in effect_floors.values()):
            raise ValueError('each sensed effect requires its measured positive floor')
        self.window=int(window);self.floors=dict(effect_floors);self.calibration=str(calibration)
        self.groups=dict(groups or {});self.fresh={};self.best={};self.progress={};self.times={}
        self.archive=[]
        self.require_retention=bool(require_retention);self.retention={}

    def reset_episode(self):
        self.fresh.clear();self.progress.clear();self.times.clear()

    def observe_fresh(self,body,modality,*,prediction,shuffled,persistence,time,healthy):
        key=(str(body),str(modality));values=(prediction,shuffled,persistence,time)
        if not all(math.isfinite(float(v)) for v in values) or min(values[:3])<0:
            raise ValueError('nonfinite or negative completed prediction evidence')
        if modality not in self.floors:return
        if time<=self.times.get(key,-float('inf')):raise ValueError('prediction evidence was replayed or moved backwards')
        self.times[key]=float(time)
        if not healthy:
            self.fresh.pop(key,None);self.progress.pop(key,None)
            return
        ring=self.fresh.setdefault(key,deque(maxlen=self.window))
        ring.append(dict(prediction=float(prediction),shuffled=float(shuffled),persistence=float(persistence)))

    def observe_anchor(self,body,modality,anchor_id,*,reference,prediction,healthy):
        """The same stored target/input pair is reused; improvement cannot be earned twice."""
        key=(str(body),str(modality));anchor=key+(str(anchor_id),)
        if not all(math.isfinite(float(v)) and v>=0 for v in (reference,prediction)):
            raise ValueError('invalid fixed-anchor prediction error')
        previous=self.best.setdefault(anchor,float(reference))
        self.best[anchor]=min(previous,float(prediction))
        floor=self.floors.get(modality,float('inf'))
        gain=max(0.,previous-float(prediction))/max(previous,floor)
        self.progress.setdefault(key,deque(maxlen=self.window)).append(gain if healthy else 0.)

    def observe_retention(self,body,modality,panel_id,*,prediction,reference):
        """An immutable panel prevents a new window from rewarding old recovery."""
        if not all(math.isfinite(float(v)) and v>=0 for v in (prediction,reference)):
            raise ValueError('invalid fixed-panel retention error')
        key=(str(body),str(modality));previous=self.retention.get(key)
        if previous is not None and previous['panel']!=panel_id:
            raise ValueError('retained calibration panel changed inside one encoder version')
        best=float(reference) if previous is None else previous['best']
        self.retention[key]=dict(panel=panel_id,error=float(prediction),best=min(best,float(prediction)),
            preserved=bool(float(prediction)<=best+self.floors[modality]),
            improved=bool(float(prediction)<best-self.floors[modality]))

    def available(self,body,modality):
        key=(str(body),str(modality));fresh=self.fresh.get(key,());gains=self.progress.get(key,())
        if self.require_retention and not self.retention.get(key,{}).get('preserved',False):return 0.,None
        if len(fresh)<self.window or not gains:return 0.,None
        floor=self.floors[modality]
        # All retained fresh comparisons must favor the real command and beat
        # persistence. Missing controls, stillness and irreducible noise remain
        # ineligible. This is an operating condition, not a skill pass/p-value.
        if not all(min(v['shuffled'],v['persistence'])>floor and
                   v['prediction']<min(v['shuffled'],v['persistence']) for v in fresh):return 0.,None
        controllable=min(1-v['prediction']/min(v['shuffled'],v['persistence']) for v in fresh)
        progress=float(np.mean(gains))
        return max(0.,min(1.,controllable*progress)),float(np.mean([v['persistence'] for v in fresh]))

    def explain(self,body,modality):
        key=(str(body),str(modality));fresh=self.fresh.get(key,());gains=self.progress.get(key,())
        amount,_=self.available(body,modality)
        if self.require_retention and key not in self.retention:reason='fixed retention panel not yet measured'
        elif self.require_retention and not self.retention[key]['preserved']:reason='fixed retention panel worse than earned best'
        elif len(fresh)<self.window:reason='insufficient eligible fresh comparisons'
        elif not gains:reason='no fixed-anchor progress measured'
        elif any(min(v['shuffled'],v['persistence'])<=self.floors[modality] for v in fresh):reason='action distinction below measured floor'
        elif any(v['prediction']>=min(v['shuffled'],v['persistence']) for v in fresh):reason='prediction does not beat action control and persistence'
        elif amount<=0:reason='no improvement beyond prior best; mastered or recovering old accuracy'
        else:reason='available'
        return dict(available=amount,reason=reason,fresh_comparisons=len(fresh),required_comparisons=self.window,
                    anchor_comparisons=len(gains),retention=copy.deepcopy(self.retention.get(key)))

    def score(self,body,predictions,hold_predictions,*,eligible,healthy):
        """One score per candidate; sensors/coordinate duplication cannot multiply it."""
        allowed=np.asarray(eligible,dtype=bool)
        if not healthy:return np.zeros(allowed.shape,np.float64)
        grouped={}
        for modality,values in predictions.items():
            if modality not in hold_predictions:continue
            opportunity,scale=self.available(body,modality)
            if opportunity<=0 or scale is None:continue
            values=np.asarray(values,np.float64);baseline=np.asarray(hold_predictions[modality],np.float64)
            if values.shape[0]!=len(allowed) or values.shape[1:]!=baseline.shape:
                raise ValueError('candidate sensory layout differs from its hold reference')
            if not np.isfinite(values).all() or not np.isfinite(baseline).all():continue
            delta=np.mean((values-baseline)**2,axis=tuple(range(1,values.ndim)))
            effect=np.minimum(delta/scale,1.)
            group=self.groups.get(modality,modality)
            grouped[group]=np.maximum(grouped.get(group,np.zeros_like(effect)),opportunity*effect)
        if not grouped:return np.zeros(allowed.shape,np.float64)
        score=np.mean(list(grouped.values()),axis=0)
        return np.where(allowed,score,0.)

    def state_dict(self):
        return copy.deepcopy(dict(contract=CONTRACT,window=self.window,effect_floors=self.floors,
            calibration=self.calibration,groups=self.groups,fresh=self.fresh,best=self.best,
            progress=self.progress,times=self.times,archive=self.archive,
            require_retention=self.require_retention,retention=self.retention))

    def load_state_dict(self,state):
        if (state['contract']!=CONTRACT or state['window']!=self.window or state['effect_floors']!=self.floors
                or state['groups']!=self.groups or state['calibration']!=self.calibration):
            raise ValueError('learning opportunity calibration changed without migration')
        if state.get('require_retention',False)!=self.require_retention:
            raise ValueError('learning opportunity retention contract changed without migration')
        for key in ('fresh','best','progress','times','archive'):setattr(self,key,copy.deepcopy(state[key]))
        self.retention=copy.deepcopy(state.get('retention',{}))

    def archive_previous(self,state,encoder_version):
        # Evidence from different token coordinates stays explicitly separate.
        self.archive.append(dict(encoder_version=encoder_version,evidence=copy.deepcopy(state)))
