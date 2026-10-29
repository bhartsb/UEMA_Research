# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Compare a predicted movement with an observed physical goal.

Goals are measured handle poses kept beside the sensory memory that selected
them. No sensory embedding is treated as a world position or joint angle.
"""
from __future__ import annotations

import math
import torch

from .efference import EFFERENCE_SCALES, EfferenceHeads, rotation_change_deg
from .motor_synergy import quat_mul, rotvec_to_quat


def pose_error_per_body(position_cm, quaternion, target, indices):
    """One normalized squared physical error per independent body."""
    position = position_cm[:, indices] - target[:, indices, :3] * 200.
    rotation = rotation_change_deg(quaternion[:, indices], target[:, indices, 3:7])
    return ((position / EFFERENCE_SCALES['pos_tenure_cm']).square().mean((1, 2))
            + (rotation / EFFERENCE_SCALES['rot_tenure_deg']).square().mean((1, 2)))


class PhysicalGoalCost:
    """A 250 ms physical goal, with sensory fallback for legacy memory entries."""
    def __init__(self, efference, target, valid, handle_names):
        self.efference = efference
        self.target = target.detach().clone()
        self.valid = valid.detach().bool().clone()
        self.indices = [i for i, name in enumerate(handle_names) if name != 'gaze']

    def frozen_modules(self):
        return [self.efference]

    def __call__(self, latent, order, achieved, sensory_error):
        effects = EfferenceHeads.to_lived(body=self.efference.body(latent, order, achieved))
        position = achieved[..., :3] * 200. + effects['body_tenure_cm']
        quaternion = quat_mul(rotvec_to_quat(effects['body_tenure_deg'] * math.pi / 180.),
                              achieved[..., 3:7])
        error = pose_error_per_body(position, quaternion, self.target, self.indices)
        return torch.where(self.valid.to(error.device), error, sensory_error)


def align_pose_memory(memory, count):
    """Preserve old sensory goals and explicitly mark their missing poses."""
    while len(memory) > count:
        memory.popleft()
    while len(memory) < count:
        memory.appendleft(None)


def selected_pose(memory, indices, achieved):
    """Gather each body's measured pose from its selected memory row."""
    target = achieved.detach().clone()
    valid = torch.zeros(achieved.shape[0], dtype=torch.bool, device=achieved.device)
    for body, index in enumerate(indices):
        if 0 <= int(index) < len(memory):
            pose = memory[int(index)]
            if pose is not None and pose.shape == achieved.shape:
                target[body] = pose[body].to(target)
                valid[body] = True
    return target, valid


class GazeOnlyActionSpace:
    """Choose the eye orientation while retaining the complete body proposal."""
    def __init__(self, baseline, gaze_idx):
        self.baseline = baseline.detach().clone()
        self.gaze_idx = int(gaze_idx)

    def init(self):
        self.rotation = self.baseline[:, self.gaze_idx, 3:7].clone().requires_grad_(True)
        return [self.rotation]

    def assemble(self):
        command = self.baseline.clone()
        command[:, self.gaze_idx, 3:7] = PhysicalTrustRegion.unit_quaternion(self.rotation)
        return command, command[..., 8:9], None

    def frozen_modules(self):
        return []

    def summary(self):
        return {'action_space': 'gaze_only'}


class PhysicalTrustRegion:
    """Bound a forecast-driven search using observed movement scales.

    Positions move at most one measured tenure scale per world coordinate;
    rotations and hand joints change at most one measured tenure rotation scale.
    Hand feasibility is enforced along a segment from its feasible starting pose.
    These are achieved-motion scales, not a measured region of forecast reliability.
    """
    def __init__(self, base_space, baseline, handle_names):
        self.base_space=base_space
        self.baseline=baseline.detach().clone()
        self.body=[i for i,n in enumerate(handle_names) if n!='gaze']
        self.gaze=[i for i,n in enumerate(handle_names) if n=='gaze']
        self.position_cm=float(EFFERENCE_SCALES['pos_tenure_cm'])
        self.rotation_deg=float(EFFERENCE_SCALES['rot_tenure_deg'])
        self.hand=(base_space is not None and hasattr(base_space,'kinematics') and hasattr(base_space,'initial'))
        self.fingers=set() if not self.hand else {i for h in base_space.kinematics.geometry['hands'] for i in h['finger_indices']}
        self.free_body=[i for i in self.body if i not in self.fingers]

    def init(self):
        if self.base_space is not None:return self.base_space.init()
        b=self.baseline
        self.position=b[...,:3].clone().requires_grad_(True)
        self.rotation=b[...,3:7].clone().requires_grad_(True)
        def logit(x):
            x=x.clamp(1e-4,1-1e-4)
            return torch.log(x/(1-x)).clone().requires_grad_(True)
        self.radius=logit(b[...,7:8]);self.strength=logit(b[...,8:9])
        return [self.position,self.rotation,self.radius,self.strength]

    @staticmethod
    def unit_quaternion(q):
        norm=q.norm(dim=-1,keepdim=True)
        identity=torch.zeros_like(q);identity[...,3]=1
        return torch.where(norm>1e-12,q/norm.clamp_min(1e-12),identity)

    def goal_joints(self):
        if not self.hand:raise ValueError('This action space has no hand coordinates')
        kin=self.base_space.kinematics
        initial=self.base_space.initial
        bound=math.radians(self.rotation_deg)
        limits=initial.new_tensor([j['limits'] for j in kin.geometry['joints']])
        proposed=initial+(self.base_space.joints-initial).clamp(-bound,bound)
        proposed=torch.maximum(torch.minimum(proposed,limits[:,1]),limits[:,0])
        out=proposed
        for group in kin.groups:
            delta=out-initial
            factors=[torch.ones_like(out[...,0])]
            for constraint in kin.geometry['constraints']:
                if not any(i in group and c for i,c in constraint['terms']):continue
                base=sum(initial[...,i]*c for i,c in constraint['terms'])
                change=sum(delta[...,i]*c for i,c in constraint['terms'])
                lo,hi=constraint['limits']
                factor=torch.where(change>=0,(hi-base)/change.clamp_min(1e-12),
                                   (base-lo)/(-change).clamp_min(1e-12))
                factors.append(factor.clamp(0,1))
            scale=torch.stack(factors,-1).amin(-1)
            out=out.clone();out[...,group]=initial[...,group]+delta[...,group]*scale[...,None]
        return out

    def assemble(self):
        if self.base_space is None:
            strength=self.strength.sigmoid()
            command=torch.cat((self.position,self.unit_quaternion(self.rotation),self.radius.sigmoid(),strength),-1)
            penalty=None
        else:
            command,strength,penalty=self.base_space.assemble()
        result=command.clone();indices=self.free_body
        delta=command[:,indices,:3]-self.baseline[:,indices,:3]
        result[:,indices,:3]=self.baseline[:,indices,:3]+delta.clamp(-self.position_cm/200.,self.position_cm/200.)
        baseline_q=self.unit_quaternion(self.baseline[:,indices,3:7])
        candidate_q=self.unit_quaternion(command[:,indices,3:7])
        rotation=rotation_change_deg(baseline_q,candidate_q)
        rotation=rotation/(rotation.norm(dim=-1,keepdim=True)/self.rotation_deg).clamp_min(1.)
        result[:,indices,3:7]=quat_mul(rotvec_to_quat(rotation*math.pi/180.),baseline_q)
        if self.hand:
            result=self.base_space.kinematics.assemble_world(result,self.goal_joints())
        # Eye aiming has its own visual objective. A body forecast must not
        # recruit eye movements to exploit an accidental learned correlation.
        result[:,self.gaze]=self.baseline[:,self.gaze]
        return result,result[...,8:9],penalty

    def frozen_modules(self):
        return [] if self.base_space is None else self.base_space.frozen_modules()

    def penalty_per_body(self):
        return self.baseline.new_zeros(self.baseline.shape[0]) if self.base_space is None else self.base_space.penalty_per_body()

    def summary(self):
        return dict({} if self.base_space is None else self.base_space.summary(),
                    physical_trust_region=dict(position_cm_per_coordinate=self.position_cm,
                        rotation_deg=self.rotation_deg,hand_joint_deg=self.rotation_deg,
                        gaze='held_at_visual_choice'))
