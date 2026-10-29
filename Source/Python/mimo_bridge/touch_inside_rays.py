# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Which directions MIMo's skin-point inside test uses: the switch that makes a world's skin layout repeatable.

THE DEFECT (docs/gain_of_function_v4/agents/training_repeatability_2026-09-26.md). MIMo's TrimeshTouch decides which
skin points of a body exist: a geom's sensor points that lie inside another geom of the same body are dropped
(MIMo_private/mimo_infant/touch/touch.py:918-939). Its inside test casts one ray from the point in each of 10
directions and calls the point inside when any ray crosses the other shape's surface an odd number of times
(touch.py:1941-1970). The 10 directions are drawn ONCE PER PROCESS from an unseeded generator (touch.py:1937-1938:
np.random.default_rng() takes its seed from the operating system), so every process draws other directions.
A skin point lying exactly on a neighbouring box's face (32 such points on the hands and feet at age 24 with
trunk_v1, each 4e-17 m or less from that face) is crossed just ahead of each ray's start, because MIMo starts every ray
1e-10 m behind the point (touch.py:1960): a ray leaving the box there crosses once (odd: inside), a ray entering it
crosses twice (even: outside). Such a point counts as outside only when all 10 rays point into the box, 1 draw in
2^10 = 1,024. Measured with MIMo's own recipe over 20,000 draws: 74 give a different set of live skin points (0.37 in
100 world builds), each exactly the draws whose 10 directions all point the same way along one of 4 axis directions.
The trainer builds every world in its own spawned process (datasource.py:463 and 479), so every world at every launch
is one draw. One such world, 8 extra live points on the hands, made the skin proof's second run diverge at step 9.

THE SWITCH. 'process' (the default, today's behaviour): the directions this process drew. 'fixed_v1': while the MIMo
env (and in it TrimeshTouch) is built, MIMo's inside test is replaced by one that runs MIMo's own test with the 10
directions below, whatever directions a caller passes, and counts its calls; MIMo's own function is back afterwards,
error or not. The directions are MIMo's recipe with generator seed 0, written out so a numpy release cannot move them.
They give the layout 19,926 of the 20,000 measured draws give: at age 24 with trunk_v1, 4,683 skin points with digest
0581069917fae467 (skin_layout below), the layout 11 of the skin proof's 12 worlds were built with. Under 'fixed_v1' the
world's layout no longer depends on the process, and a world that would have drawn the usual layout trains exactly as
before. A build in which fewer inside tests ran through the switch than the touch has shape pairs is refused
(check_reached), so a MIMo change cannot bypass the switch for some or all pairs unseen.
"""
from contextlib import contextmanager
import hashlib
import inspect

import numpy as np

CHOICES = ('process', 'fixed_v1')

# normalize_vectors(np.random.default_rng(0).normal(size=(10, 3))): MIMo's recipe (touch.py:1937-1938) with seed 0,
# computed 2026-09-26 on numpy 2.2.6; the sha256 of these float64 values is FIXED_V1_SHA256
FIXED_V1 = np.array([
    (0.18881711923692265, -0.19839032737660414, 0.9617636786063786),
    (0.16021416297716448, -0.818128926665578, 0.5522648652001644),
    (0.7415052042025201, 0.5385471155343273, -0.4001712589507583),
    (-0.8967022761251738, -0.4416644114201207, 0.029284393059288184),
    (-0.8784069177470692, -0.08266045905288898, -0.4707106705432322),
    (-0.7583158229918717, -0.5636195521419911, -0.32755169522293953),
    (0.3648617673568587, 0.9240647543268903, -0.11393077078653183),
    (0.8759982916180951, -0.4264361324882868, 0.22534244604736822),
    (0.7696741376445092, 0.0800898974604638, -0.6333935034131261),
    (-0.8758196390296379, -0.43492918578181555, 0.20922849529918608),
], dtype=np.float64)
FIXED_V1.setflags(write=False)
FIXED_V1_SHA256 = 'a112217f34c9b46bc1e38b9aeadf8d11b6a3441d2a4b59a05ce68a0bba8a6951'


def mimo_inside_test():
    """MIMo's inside test (touch._contains), checked to still be (points, mesh, directions, tol) with a default tol;
    anything else refuses rather than guess how to call it."""
    import mimo_infant.touch.touch as mt
    fn = mt._contains
    params = list(inspect.signature(fn).parameters)
    if params[:4] != ['points', 'mesh', 'directions', 'tol'] or fn.__defaults__ is None or len(fn.__defaults__) != 2:
        raise RuntimeError("MIMo's skin inside test changed shape (touch._contains%s); touch_inside_rays cannot set "
                           "its directions" % (inspect.signature(fn),))
    return fn


class InsideRayUse:
    """What the switch did during one build: how many inside tests ran with FIXED_V1."""

    def __init__(self, choice):
        self.choice = choice
        self.calls = 0

    def report(self):
        return dict(setting=self.choice, inside_tests=self.calls,
                    directions_sha256=FIXED_V1_SHA256 if self.choice != 'process' else None)


@contextmanager
def inside_rays(choice):
    """Build MIMo's touch inside this block; yields an InsideRayUse. 'fixed_v1': every call of MIMo's inside test in
    the block runs MIMo's own test with FIXED_V1 and MIMo's default tolerance (or the caller's), and MIMo's own
    function is back afterwards, error or not. 'process': nothing is touched."""
    if choice not in CHOICES:
        raise ValueError('touch_inside_rays must be one of %s, not %r' % (', '.join(CHOICES), choice))
    use = InsideRayUse(choice)
    if choice == 'process':
        yield use
        return
    import mimo_infant.touch.touch as mt
    own = mimo_inside_test()
    default_tol = own.__defaults__[1]

    def fixed_inside_test(points, mesh, directions=None, tol=None):
        use.calls += 1
        return own(points, mesh, FIXED_V1.copy(), default_tol if tol is None else tol)

    mt._contains = fixed_inside_test
    try:
        yield use
    finally:
        mt._contains = own


def shapes_to_test(touch):
    """How many shape-against-shape inside tests MIMo's TrimeshTouch.add_body makes on this touch
    (touch.py:919-926: each shape of a part against each other shape that is not a point cloud)."""
    from trimesh import PointCloud
    return sum(1 for meshes in touch._submeshes.values() for m in meshes for o in meshes
               if o is not m and not isinstance(o, PointCloud))


def check_reached(use, touch):
    """Refuse a 'fixed_v1' build in which fewer inside tests ran through the switch than MIMo's touch has shape pairs
    to test: the pairs it missed would have left their points to the process's draw."""
    if use.choice == 'process':
        return
    pairs = shapes_to_test(touch)
    if use.calls < pairs:
        raise RuntimeError("touch_inside_rays %s ran %d inside tests while MIMo's touch has %d shape pairs to test: MIMo "
                           "no longer calls touch._contains for every pair; the switch cannot fix the layout"
                           % (use.choice, use.calls, pairs))


def skin_layout(touch, body_ids):
    """(live points, digest) of MIMo's live skin points on body_ids: the digest is the first 16 hex characters of
    the sha256 of their float64 body-frame positions, bodies in id order."""
    pos = [np.ascontiguousarray(np.asarray(touch.sensor_positions[b], dtype=np.float64))
           for b in sorted(int(b) for b in body_ids)]
    return int(sum(len(p) for p in pos)), hashlib.sha256(b''.join(p.tobytes() for p in pos)).hexdigest()[:16]
