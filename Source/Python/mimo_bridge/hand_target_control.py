# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Versioned, feasible hand goals shared by planning and physical execution.

Public motor poses remain world centimetres/xyzw. A hand goal is a bounded
joint vector. Its canonical world poses use the requested wrist; execution
regenerates the same goal around the actual wrist before every physics step.
Torch FK uses the compiled MuJoCo hinge frames, with no surrogate gradients.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
import numpy as np

CONTRACT = 'wrist_relative_joint_v1'
DIGITS = ('thumb', 'index', 'middle', 'ring', 'pinky')


def _rotation_xyzw(quaternion):
    """SciPy rotation with the native decoder's zero-quaternion identity rule."""
    from scipy.spatial.transform import Rotation
    q=np.array(quaternion,dtype=float,copy=True)
    norm=np.linalg.norm(q,axis=-1,keepdims=True)
    identity=np.zeros_like(q);identity[...,3]=1.
    # SciPy already normalizes nonzero inputs. Leave that established path
    # unchanged instead of introducing a second rounding through normalization.
    q=np.where(norm<1e-12,identity,q)
    return Rotation.from_quat(q)


def _native_inverse_objective(model, data, joint_ids, bodies, desired_p, desired_R, weight):
    """Exact position/rotation error and hinge derivatives for constrained IK."""
    import mujoco
    from scipy.spatial.transform import Rotation
    joint_ids=np.asarray(joint_ids,dtype=int)
    addresses=model.jnt_qposadr[joint_ids]
    ancestors=[]
    for body in bodies:
        chain=set();cursor=int(body)
        while cursor:
            chain.add(cursor);cursor=int(model.body_parentid[cursor])
        ancestors.append([i for i,joint in enumerate(joint_ids) if int(model.jnt_bodyid[joint]) in chain])
    def objective(q):
        data.qpos[addresses]=q
        mujoco.mj_kinematics(model,data)
        position=data.xpos[bodies]
        rotation=data.xmat[bodies].reshape(-1,3,3)
        errors=Rotation.from_matrix(desired_R@rotation.transpose(0,2,1)).as_rotvec()
        delta=position-desired_p
        value=float(np.sum((weight*delta)**2)+np.sum(errors**2))
        gradient=np.zeros(len(joint_ids))
        for finger,active in enumerate(ancestors):
            selected=joint_ids[active]
            axis=data.xaxis[selected]
            jacobian=np.cross(axis,position[finger]-data.xanchor[selected])
            # Squared rotation-log error has derivative -2*axis dot error in
            # the same world frame. The exact pi branch cut is nonsmooth.
            gradient[active]+=2*weight**2*(jacobian@delta[finger])-2*(axis@errors[finger])
        return value,gradient
    return objective


def geometry_from_model(model, decoder):
    """Export only immutable hand geometry; no simulator state or learned data."""
    import mujoco
    from .action_decoder import POSITION_WEIGHT
    names = list(decoder.handle_names)
    bodies = []
    joints = []
    hands = []
    constraints = []
    for side, prefix in (('L', 'left'), ('R', 'right')):
        wrist_index = names.index('wrist_' + side)
        wrist = decoder.handles[wrist_index].body_id
        finger_indices = [names.index(digit + '_' + side) for digit in DIGITS]
        needed = set()
        for index in finger_indices:
            body = decoder.handles[index].body_id
            while body != wrist:
                if body == 0:
                    raise ValueError('Finger ancestry does not reach its wrist')
                needed.add(body)
                body = int(model.body_parentid[body])
        body_map = {wrist: -1}
        for body in sorted(needed):
            joint_indices = []
            for joint in range(int(model.body_jntadr[body]),
                               int(model.body_jntadr[body] + model.body_jntnum[body])):
                if int(model.jnt_type[joint]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                    raise ValueError('Hand contract supports compiled hinge joints only')
                if not model.jnt_limited[joint]:
                    raise ValueError('Hand contract requires finite native joint limits')
                joint_indices.append(len(joints))
                address = int(model.jnt_qposadr[joint])
                joints.append(dict(name=model.joint(joint).name, source_id=joint,
                    qpos_address=address, axis=model.jnt_axis[joint].tolist(),
                    anchor=model.jnt_pos[joint].tolist(), limits=model.jnt_range[joint].tolist(),
                    reference=float(model.qpos0[address])))
            body_map[body] = len(bodies)
            bodies.append(dict(name=model.body(body).name,
                parent=body_map[int(model.body_parentid[body])], side=side,
                position=model.body_pos[body].tolist(), quaternion=model.body_quat[body].tolist(),
                joints=joint_indices))
        hands.append(dict(side=side, wrist_index=wrist_index, wrist_body=wrist,
            finger_indices=finger_indices,
            finger_bodies=[body_map[decoder.handles[i].body_id] for i in finger_indices]))
        source_to_index = {j['source_id']: i for i, j in enumerate(joints)}
        for tendon in range(model.ntendon):
            if not (model.tendon(tendon).name or '').startswith('finger_couple_' + prefix):
                continue
            if not model.tendon_limited[tendon]:
                continue
            terms = []
            start = int(model.tendon_adr[tendon])
            for wi in range(start, start + int(model.tendon_num[tendon])):
                if int(model.wrap_type[wi]) != int(mujoco.mjtWrap.mjWRAP_JOINT):
                    raise ValueError('Hand contract requires linear fixed coupling tendons')
                source = int(model.wrap_objid[wi])
                if source not in source_to_index:
                    raise ValueError('Coupling tendon leaves the projected hand')
                terms.append([source_to_index[source], float(model.wrap_prm[wi])])
            constraints.append(dict(name=model.tendon(tendon).name, terms=terms,
                                    limits=model.tendon_range[tendon].tolist()))
    result = dict(contract=CONTRACT, handle_names=names, bodies=bodies, joints=joints,
                  hands=hands, constraints=constraints, position_weight_rad_per_m=float(POSITION_WEIGHT))
    # Neutral must lie in every half-space used by the radial projection.
    neutral = np.clip(np.zeros(len(joints)), *np.asarray([j['limits'] for j in joints]).T)
    for c in constraints:
        value = sum(neutral[i] * coefficient for i, coefficient in c['terms'])
        if not c['limits'][0] <= value <= c['limits'][1]:
            raise ValueError('Native tendon limits do not admit the neutral hand reference')
    return result


def _qm(a, b):
    """Torch quaternion product, wxyz, supporting arbitrary leading dimensions."""
    import torch
    aw, av = a[..., :1], a[..., 1:]
    bw, bv = b[..., :1], b[..., 1:]
    return torch.cat((aw*bw - (av*bv).sum(-1, keepdim=True),
                      aw*bv + bw*av + torch.linalg.cross(av, bv)), -1)


def _qr(q, v):
    import torch
    vector = q[..., 1:]
    vector, v = torch.broadcast_tensors(vector, v)
    cross = 2 * torch.linalg.cross(vector, v)
    return v + q[..., :1] * cross + torch.linalg.cross(vector, cross)


def _qnormal(q):
    import torch
    norm = q.norm(dim=-1, keepdim=True)
    identity = torch.zeros_like(q)
    identity[..., 0] = 1
    return torch.where(norm > 1e-12, q / norm.clamp_min(1e-12), identity)


class HandKinematics:
    def __init__(self, geometry):
        if geometry.get('contract') != CONTRACT:
            raise ValueError('Unknown hand target contract')
        self.geometry = copy.deepcopy(geometry)
        self._projection_model = None
        self._projection_data = None
        # Connected components of joints sharing a coupling tendon. Uncoupled
        # joints retain their requested angle even when another finger conflicts.
        components = []
        for constraint in geometry['constraints']:
            group = {int(i) for i, coefficient in constraint['terms'] if coefficient}
            touching = [old for old in components if old & group]
            for old in touching:
                group |= old
                components.remove(old)
            components.append(group)
        self.groups = [sorted(group) for group in components]

    @property
    def n_joints(self):
        return len(self.geometry['joints'])

    def project_joints(self, value):
        """Differentiable, exact feasibility map, identity on feasible inputs.

        Clamp native angle limits, then radially project each connected tendon
        group toward its feasible zero/clipped-zero reference. This is not a
        closest-point projection; gradients are ordinary piecewise derivatives.
        """
        import torch
        limits = value.new_tensor([j['limits'] for j in self.geometry['joints']])
        out = torch.maximum(torch.minimum(value, limits[:, 1]), limits[:, 0])
        neutral = torch.maximum(torch.minimum(torch.zeros_like(limits[:, 0]),
                                              limits[:, 1]), limits[:, 0])
        for group in self.groups:
            delta = out - neutral
            factors = [torch.ones_like(out[..., 0])]
            for c in self.geometry['constraints']:
                if not any(i in group and coefficient for i, coefficient in c['terms']):
                    continue
                base = sum(neutral[i] * coefficient for i, coefficient in c['terms'])
                change = sum(delta[..., i] * coefficient for i, coefficient in c['terms'])
                lo, hi = c['limits']
                # Denominators stay finite even in the inactive torch.where arm.
                factor = torch.where(change >= 0,
                    (hi-base) / change.clamp_min(1e-12),
                    (base-lo) / (-change).clamp_min(1e-12))
                factors.append(factor.clamp(0, 1))
            scale = torch.stack(factors, -1).amin(-1)
            replacement = neutral[group] + delta[..., group] * scale[..., None]
            out = out.clone()
            out[..., group] = replacement
        return out

    def local_poses(self, joint_angles):
        """Exact compiled hinge FK, returning finger positions in metres/wxyz."""
        import torch
        positions, quaternions = [], []
        shape = joint_angles.shape[:-1]
        for body in self.geometry['bodies']:
            p = joint_angles.new_tensor(body['position']).expand(*shape, 3)
            q = joint_angles.new_tensor(body['quaternion']).expand(*shape, 4)
            for index in body['joints']:
                joint = self.geometry['joints'][index]
                angle = (joint_angles[..., index] - joint['reference']) * .5
                axis = joint_angles.new_tensor(joint['axis'])
                rotation = torch.cat((angle.cos()[..., None],
                                      angle.sin()[..., None] * axis), -1)
                anchor = joint_angles.new_tensor(joint['anchor'])
                after = _qm(q, rotation)
                p = p + _qr(q, anchor) - _qr(after, anchor)
                q = after
            if body['parent'] >= 0:
                parent = body['parent']
                p = positions[parent] + _qr(quaternions[parent], p)
                q = _qm(quaternions[parent], q)
            positions.append(p)
            quaternions.append(q)
        return torch.stack(positions, -2), torch.stack(quaternions, -2)

    def assemble_world(self, template, joint_angles, position_scale=200.):
        """Replace only finger poses; preserve every radius/strength and other row.

        position_scale=200 for normalized model motor positions, 1 for execution
        centimetres. Joint goals are projected by the caller (and validated by env).
        """
        import torch
        local_p, local_q = self.local_poses(joint_angles)
        result = template.clone()
        for hand in self.geometry['hands']:
            wrist = template[..., hand['wrist_index'], :]
            wq = _qnormal(wrist[..., [6, 3, 4, 5]])
            p = local_p[..., hand['finger_bodies'], :]
            q = local_q[..., hand['finger_bodies'], :]
            result[..., hand['finger_indices'], :3] = (wrist[..., None, :3] +
                _qr(wq[..., None, :].expand_as(q), p) * (100./position_scale))
            world_q = _qm(wq[..., None, :].expand_as(q), q)
            result[..., hand['finger_indices'], 3:7] = world_q[..., [1, 2, 3, 0]]
        return result

    def local_encoding(self, canonical_world, position_scale=200.):
        """Invariant predictor encoding; only fingers become wrist-relative poses.

        Positions keep the caller's scale. Radius, strength and non-finger world
        rows retain their public meaning. The contract version accompanies data.
        """
        result = canonical_world.clone()
        for hand in self.geometry['hands']:
            wrist = canonical_world[..., hand['wrist_index'], :]
            wq = _qnormal(wrist[..., [6, 3, 4, 5]])
            inverse = wq.clone(); inverse[..., 1:] = -inverse[..., 1:]
            fingers = canonical_world[..., hand['finger_indices'], :]
            expanded = inverse[..., None, :].expand(*fingers.shape[:-1], 4)
            result[..., hand['finger_indices'], :3] = _qr(expanded,
                fingers[..., :3] - wrist[..., None, :3])
            fq = _qnormal(fingers[..., [6, 3, 4, 5]])
            result[..., hand['finger_indices'], 3:7] = _qm(expanded, fq)[..., [1,2,3,0]]
        return result

    def validate_joints(self, angles, tolerance=1e-7):
        angles = np.asarray(angles, dtype=float)
        if angles.shape[-1:] != (self.n_joints,) or not np.isfinite(angles).all():
            raise ValueError('Wrong shape or non-finite hand joint goal')
        limits = np.asarray([j['limits'] for j in self.geometry['joints']])
        if np.any(angles < limits[:, 0]-tolerance) or np.any(angles > limits[:, 1]+tolerance):
            raise ValueError('Hand goal exceeds a native joint limit')
        for c in self.geometry['constraints']:
            value = sum(angles[..., i]*coefficient for i, coefficient in c['terms'])
            if np.any(value < c['limits'][0]-tolerance) or np.any(value > c['limits'][1]+tolerance):
                raise ValueError('Hand goal exceeds a native finger coupling limit')

    def _mini_model(self):
        """Private kinematics only; compiled C FK keeps inverse solves inexpensive."""
        import mujoco
        if self._projection_model is not None:
            return self._projection_model, self._projection_data
        root = ET.Element('mujoco')
        ET.SubElement(root, 'compiler', angle='radian')
        world = ET.SubElement(root, 'worldbody')
        nodes = []
        text = lambda value: ' '.join(format(float(v), '.17g') for v in value)
        for body in self.geometry['bodies']:
            parent = world if body['parent'] < 0 else nodes[body['parent']]
            node = ET.SubElement(parent, 'body', name=body['name'],
                pos=text(body['position']), quat=text(body['quaternion']))
            ET.SubElement(node, 'inertial', pos='0 0 0', mass='.001', diaginertia='.00001 .00001 .00001')
            for index in body['joints']:
                joint = self.geometry['joints'][index]
                ET.SubElement(node, 'joint', name=joint['name'], type='hinge',
                    pos=text(joint['anchor']), axis=text(joint['axis']),
                    range=text(joint['limits']), ref=str(joint['reference']))
            nodes.append(node)
        self._projection_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self._projection_data = mujoco.MjData(self._projection_model)
        return self._projection_model, self._projection_data

    def project_world_orders(self, orders, position_scale=200.):
        """Once-per-choice native constrained IK; detached initialization only.

        Returns (canonical world orders, joint vectors, per-order diagnostics).
        Every canonical finger pose comes from one feasible hand configuration.
        """
        import mujoco
        from scipy.optimize import minimize, LinearConstraint
        from scipy.spatial.transform import Rotation
        model, data = self._mini_model()
        command = np.asarray(orders, dtype=float)
        if command.shape[-2:] != (len(self.geometry['handle_names']), 9) or not np.isfinite(command).all():
            raise ValueError('Expected finite (..., handles, 9) world motor orders')
        shape = command.shape[:-2]
        flat = command.reshape(-1, *command.shape[-2:]).copy()
        goals, diagnostics = [], []
        for order in flat:
            angles = np.zeros(self.n_joints)
            rows = []
            for hand in self.geometry['hands']:
                indices = sorted({i for body in self.geometry['bodies'] if body['side'] == hand['side']
                                  for i in body['joints']})
                joint_index = {index: j for j, index in enumerate(indices)}
                limits = np.asarray([self.geometry['joints'][i]['limits'] for i in indices])
                constraints = []
                for c in self.geometry['constraints']:
                    if not any(i in joint_index for i, coefficient in c['terms']): continue
                    row = np.zeros(len(indices))
                    for i, coefficient in c['terms']: row[joint_index[i]] += coefficient
                    constraints.append(LinearConstraint(row, *c['limits']))
                wrist = order[hand['wrist_index']]
                wrist_R = _rotation_xyzw(wrist[3:7]).as_matrix()
                desired = order[hand['finger_indices']]
                desired_p = (desired[:, :3]-wrist[:3]) @ wrist_R * (position_scale/100.)
                desired_R = wrist_R.T @ _rotation_xyzw(desired[:, 3:7]).as_matrix()
                bodies = [model.body(self.geometry['bodies'][b]['name']).id for b in hand['finger_bodies']]
                joint_ids = [model.joint(self.geometry['joints'][i]['name']).id for i in indices]
                objective = _native_inverse_objective(model,data,joint_ids,bodies,desired_p,desired_R,
                    self.geometry['position_weight_rad_per_m'])
                initial = np.clip(np.zeros(len(indices)), limits[:, 0], limits[:, 1])
                solved = minimize(objective, initial, jac=True, method='SLSQP', bounds=limits,
                    constraints=constraints, options=dict(maxiter=150, ftol=1e-9))
                if not solved.success:
                    raise RuntimeError('Hand inverse projection failed: '+str(solved.message))
                angles[indices] = solved.x
                objective(solved.x)
                order[hand['finger_indices'], :3] = wrist[:3] + data.xpos[bodies] @ wrist_R.T * (100./position_scale)
                order[hand['finger_indices'], 3:7] = Rotation.from_matrix(
                    wrist_R @ data.xmat[bodies].reshape(-1,3,3)).as_quat()
                rows.append(dict(side=hand['side'], iterations=int(solved.nit), objective=float(solved.fun)))
            self.validate_joints(angles)
            goals.append(angles)
            diagnostics.append(rows)
        return flat.reshape(command.shape), np.asarray(goals).reshape(*shape, self.n_joints), diagnostics

    def canonical_from_joints(self, requested, angles, position_scale=1.):
        """NumPy canonical world poses for explicit goals; no inverse solve."""
        import mujoco
        from scipy.spatial.transform import Rotation
        model, data = self._mini_model()
        output = np.array(requested, dtype=float, copy=True)
        for joint, angle in zip(self.geometry['joints'], angles):
            data.qpos[int(model.jnt_qposadr[model.joint(joint['name']).id])] = angle
        mujoco.mj_kinematics(model, data)
        for hand in self.geometry['hands']:
            wrist = output[hand['wrist_index']].copy()
            rotation = _rotation_xyzw(wrist[3:7]).as_matrix()
            bodies = [model.body(self.geometry['bodies'][b]['name']).id for b in hand['finger_bodies']]
            output[hand['finger_indices'], :3] = wrist[:3] + data.xpos[bodies] @ rotation.T * (100./position_scale)
            output[hand['finger_indices'], 3:7] = Rotation.from_matrix(
                rotation @ data.xmat[bodies].reshape(-1,3,3)).as_quat()
        return output


class HandActionSpace:
    """Planner variables are ordinary world commands plus feasible hand angles."""
    def __init__(self, geometry, current_motor, initial_joints, base_space=None):
        self.kinematics = HandKinematics(geometry)
        self.template = current_motor.detach().float()
        self.initial = initial_joints.detach().to(self.template)
        self.base_space = base_space
        self.position = self.rotation = self.radius = self.strength = self.joints = None

    def frozen_modules(self):
        return [] if self.base_space is None else self.base_space.frozen_modules()

    def init(self):
        import torch
        self.joints = self.initial.clone().requires_grad_(True)
        if self.base_space is not None:
            return self.base_space.init() + [self.joints]
        self.position = self.template[..., :3].clone().requires_grad_(True)
        self.rotation = self.template[..., 3:7].clone().requires_grad_(True)
        self.radius = self.template[..., 7:8].clone().requires_grad_(True)
        value = self.template[..., 8:9].clamp(1e-4, 1-1e-4)
        self.strength = torch.log(value/(1-value)).clone().requires_grad_(True)
        return [self.position, self.rotation, self.radius, self.strength, self.joints]

    def assemble(self):
        import torch
        if self.base_space is not None:
            template, strength, penalty = self.base_space.assemble()
        else:
            rotation = _qnormal(self.rotation[..., [3,0,1,2]])[..., [1,2,3,0]]
            strength = self.strength.sigmoid()
            template = torch.cat((self.position, rotation, self.radius.clamp(0,1), strength), -1)
            penalty = None
        joints = self.goal_joints()
        return self.kinematics.assemble_world(template, joints), strength, penalty

    def goal_joints(self):
        return self.kinematics.project_joints(self.joints)

    def summary(self):
        summary = {} if self.base_space is None else dict(self.base_space.summary())
        summary.update(hand_target_contract=CONTRACT, feasible_hand_goals=True,
                       hand_base_space='world' if self.base_space is None else type(self.base_space).__name__)
        return summary

    def penalty_per_body(self):
        return (self.template.new_zeros(self.template.shape[0]) if self.base_space is None
                else self.base_space.penalty_per_body())


class HandTargetControl:
    """Environment-side projection cache and exact executed-world readback."""
    def __init__(self, model, decoder):
        import mujoco
        self.geometry = geometry_from_model(model, decoder)
        self.kinematics = HandKinematics(self.geometry)
        self.model = model
        self.scratch = mujoco.MjData(model)
        self._joint_addresses = [j['qpos_address'] for j in self.geometry['joints']]
        self._joint_dof_addresses = [int(model.jnt_dofadr[j['source_id']]) for j in self.geometry['joints']]
        self._finger_bodies = {i: decoder.handles[i].body_id for hand in self.geometry['hands']
                               for i in hand['finger_indices']}
        self._pose_indices = sorted({i for hand in self.geometry['hands']
            for i in [hand['wrist_index']] + hand['finger_indices']})

    def goals_by_joint(self, joint_goal):
        """{MuJoCo joint id: goal angle (rad)} for the hand hinges of a joint goal in geometry order: the decoder's
        finger_goals under --finger-joint-servo on (action_decoder.ActionDecoder.decode)."""
        goal = np.asarray(joint_goal, dtype=float).reshape(-1)
        if goal.shape != (self.kinematics.n_joints,) or not np.isfinite(goal).all():
            raise ValueError('Expected one finite angle per hand hinge')
        return {int(j['source_id']): float(v) for j, v in zip(self.geometry['joints'], goal)}

    def execute(self, data, requested, cache=None, joint_goal=None):
        import mujoco
        request = np.array(requested, dtype=float, copy=True)
        expected = (len(self.geometry['handle_names']), 9)
        if request.shape != expected or not np.isfinite(request).all():
            raise ValueError('Expected finite world-centimetre motor targets')
        supplied = None if joint_goal is None else np.asarray(joint_goal, dtype=float)
        if supplied is not None and supplied.shape != (self.kinematics.n_joints,):
            raise ValueError('One environment requires a single hand joint-goal vector')
        reuse = (cache is not None and cache.get('contract') == CONTRACT and
                 np.array_equal(request[self._pose_indices, :7], cache['requested'][self._pose_indices, :7]) and
                 ((supplied is None and cache['supplied'] is None) or
                  (supplied is not None and cache['supplied'] is not None and
                   np.array_equal(supplied, cache['supplied']))))
        if not reuse:
            if supplied is None:
                canonical, angles, diagnostics = self.kinematics.project_world_orders(request, position_scale=1.)
            else:
                self.kinematics.validate_joints(supplied, tolerance=2e-6)
                angles = supplied.copy()
                canonical = self.kinematics.canonical_from_joints(request, angles)
                diagnostics = []
            cache = dict(contract=CONTRACT, requested=request.copy(), supplied=copy.deepcopy(supplied),
                         canonical=canonical, joint_goal=angles.copy(), projection=diagnostics)
        else:
            # Other commands and strength/radius may change during a held hand
            # goal. Preserve them without re-solving unchanged hand geometry.
            canonical = request.copy()
            fingers = list(self._finger_bodies)
            canonical[fingers, :7] = cache['canonical'][fingers, :7]
            cache = dict(cache, requested=request.copy(), canonical=canonical)
        self.scratch.qpos[:] = data.qpos
        self.scratch.qpos[self._joint_addresses] = cache['joint_goal']
        mujoco.mj_kinematics(self.model, self.scratch)
        executed = request.copy()
        for index, body in self._finger_bodies.items():
            executed[index, :3] = self.scratch.xpos[body]*100.
            executed[index, 3:7] = self.scratch.xquat[body][[1,2,3,0]]
        return executed, cache
