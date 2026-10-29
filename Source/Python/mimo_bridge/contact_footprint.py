# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Geometry and explicit virtual material assumptions, independent of taxel spacing.

This is a pressure transducer for rigid-body simulation, not an identified human
tissue law. MuJoCo penetration is deliberately absent from its inputs. Its solved
normal force is conserved while a declared elastic coating estimates load area.
A FootprintGuard (off unless named at launch) floors the area of a degenerate
flat-face sliver and caps every footprint's mean pressure; see FootprintGuard.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np

CONTRACT='virtual_elastic_skin_v1_2026_09_13'


@dataclass(frozen=True)
class VirtualSkinProfile:
    effective_modulus_pa: float=1e6
    soft_modulus_pa: float=2e5
    stiff_modulus_pa: float=2e6
    # These are properties of a virtual coating. They have not been measured
    # from infant skin, and its variation interval is not a confidence interval.
    def __post_init__(self):
        if any(not math.isfinite(value) or value<=0 for value in asdict(self).values()):
            raise ValueError('virtual skin modulus must be finite and positive')
        if not self.soft_modulus_pa<=self.effective_modulus_pa<=self.stiff_modulus_pa:
            raise ValueError('nominal virtual modulus outside declared bounds')


@dataclass(frozen=True)
class FootprintGuard:
    """Declared bounds that keep a degenerate footprint from reading as a crushing press.

    Why (docs/gain_of_function_v4/agents/body2_pain_2026-09-25.md): two aligned flat box faces
    are scored by the exact overlap of their rectangles. When they meet only at an edge or a
    corner that overlap is a sliver; H84 body 2's palm side face on the heavy box's top edge
    read 0.0165 mm2 at a sampled tick (1.2e-5 mm2 on the recorded pose one tick later), so an
    ordinary 4.8 N read 3.17e11 Pa and stored rubbing load 12,941 (harm line 11.5).

    edge_rounding_m      every box edge and corner is taken as rounded to this radius. Two aligned
                         flat faces always share at least the contact their rounded edge makes
                         along the overlap (its Hertz line contact, rounded_edge_contact), so that
                         contact is a floor on the overlap's area: a sliver reads as the rounded
                         edge, a wide overlap as itself, with no jump between (review 2026-09-25,
                         finding 4: a first rule that floored only overlaps too narrow to hold a
                         disk of the radius moved the pressure 3 to 8 times at 2 mm).
                         Declared, not measured tissue.
    pressure_ceiling_pa  no footprint's mean pressure exceeds this: area >= force / ceiling.
                         10 MPa is 2.1 times the largest of the 72 measured v4 falls (4.79 MPa,
                         90 cm) and 2.1 times the largest v3 fall (4.84 MPa).
    Off is FOOTPRINT_GUARDS['off'] = None: footprint() then returns the v1 records exactly.
    """
    name: str = 'rounded_edge_v1'
    edge_rounding_m: float = 1e-3
    pressure_ceiling_pa: float = 1e7

    def __post_init__(self):
        for value in (self.edge_rounding_m, self.pressure_ceiling_pa):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('footprint guard settings must be finite and positive')


# The launch flag's values (train_mimo --contact-footprint-guard, BridgeEnv contact_footprint_guard).
FOOTPRINT_GUARDS = {'off': None, 'rounded_edge_v1': FootprintGuard()}
RESCORED_METHOD = 'flat_overlap_floored_at_rounded_edge'


@dataclass(frozen=True)
class Geometry:
    kind: int  # MuJoCo plane=0,sphere=2,capsule=3,ellipsoid=4,cylinder=5,box=6
    size: np.ndarray
    position: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        for name, shape in [('size',(3,)),('position',(3,)),('rotation',(3,3))]:
            value=np.asarray(getattr(self,name),dtype=float)
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('contact geometry must have finite physical coordinates')
            object.__setattr__(self,name,value.copy())
        if not np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-6) or np.linalg.det(self.rotation)<0:
            raise ValueError('contact geometry requires a proper rotation')
        needed={2:1,3:2,4:3,5:2,6:3}.get(self.kind,0)
        if needed and (self.size[:needed]<=0).any():
            raise ValueError('contact geometry requires positive physical dimensions')


def tangent_basis(normal):
    normal=np.asarray(normal,dtype=float)
    if normal.shape!=(3,) or not np.isfinite(normal).all() or np.linalg.norm(normal)<1e-10:
        raise ValueError('contact requires a finite nonzero normal')
    normal=normal/np.linalg.norm(normal)
    axis=np.eye(3)[np.argmin(abs(normal))]
    first=np.cross(normal,axis);first/=np.linalg.norm(first)
    return np.stack([first,np.cross(normal,first)],axis=1)


def projected_area(geometry,normal):
    """Finite silhouette area bounds the projected loaded area of convex shapes."""
    n=abs(geometry.rotation.T@normal);s=geometry.size
    if geometry.kind==0:return math.inf
    if geometry.kind==2:return math.pi*s[0]**2
    if geometry.kind in (3,5):
        side=4*s[0]*s[1]*math.sqrt(max(0.,1-n[2]**2))
        return side+math.pi*s[0]**2*(1. if geometry.kind==3 else n[2])
    if geometry.kind==4:return math.pi*np.prod(s)*np.sqrt(np.sum((n/s)**2))
    if geometry.kind==6:return 4*(s[1]*s[2]*n[0]+s[0]*s[2]*n[1]+s[0]*s[1]*n[2])
    return None


def curvature(geometry,point,normal,basis):
    """Principal surface curvature projected into the contact tangent plane."""
    s=geometry.size;local=geometry.rotation.T@(point-geometry.position)
    if geometry.kind==0:return np.zeros((2,2)),None,'plane'
    if geometry.kind==2:return np.eye(2)/s[0],None,'sphere'
    if geometry.kind==3:
        if abs(local[2])>s[1]:return np.eye(2)/s[0],None,'capsule_end'
        axis=geometry.rotation[:,2]
        around=np.cross(axis,normal)
        if np.linalg.norm(around)<1e-8:return None,None,'capsule_transition_unresolved'
        around/=np.linalg.norm(around);v=basis.T@around
        return np.outer(v,v)/s[0],2*s[1],'capsule_side'
    if geometry.kind==4:
        h=np.diag(1/s**2);gradient=h@local
        if np.linalg.norm(gradient)<1e-10:return None,None,'ellipsoid_center_unresolved'
        world=geometry.rotation@h@geometry.rotation.T/np.linalg.norm(gradient)
        return basis.T@world@basis,None,'ellipsoid_quadratic'
    if geometry.kind==6:
        n=abs(geometry.rotation.T@normal)
        if n.max()<.995:return None,None,'sharp_edge_unresolved'
        return np.zeros((2,2)),None,'flat_box_face'
    return None,None,'unsupported_geometry'


def polygon_area(points):
    if len(points)<3:return 0.
    p=np.asarray(points);return abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))))/2


def face_polygon(geometry,point,normal,basis):
    if geometry.kind==0:return None
    if geometry.kind!=6:return []
    n=geometry.rotation.T@normal;axis=int(np.argmax(abs(n)))
    if abs(n[axis])<.995:return []
    axes=[i for i in range(3) if i!=axis]
    center=geometry.position+geometry.rotation[:,axis]*geometry.size[axis]*np.sign(
        (geometry.rotation.T@(point-geometry.position))[axis])
    corners=[]
    for x,y in [(-1,-1),(1,-1),(1,1),(-1,1)]:
        world=center+x*geometry.rotation[:,axes[0]]*geometry.size[axes[0]]+y*geometry.rotation[:,axes[1]]*geometry.size[axes[1]]
        corners.append(basis.T@(world-point))
    p=np.asarray(corners)
    signed=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
    return p if signed>0 else p[::-1]


def clip_polygon(subject,clip):
    """Convex face intersection in physical coordinates; contact count is irrelevant."""
    result=[np.asarray(p) for p in subject]
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        incoming=result;result=[]
        if not incoming:break
        edge=b-a
        def side(p):return edge[0]*(p-a)[1]-edge[1]*(p-a)[0]
        previous=incoming[-1];ps=side(previous)
        for current in incoming:
            cs=side(current)
            if (cs>=0)!=(ps>=0):result.append(previous+(current-previous)*ps/(ps-cs))
            if cs>=0:result.append(current)
            previous=current;ps=cs
    return np.asarray(result)


# Edges shorter than this are clipping noise (a double at 0.03 m resolves 7e-18 m), never geometry:
# the thinnest sliver H84 recorded is 3e-6 m across.
_EDGE_NOISE_M=1e-12


def convex_hull(points):
    """Counter-clockwise convex hull (Andrew's monotone chain), without repeated or collinear points.
    The overlap of two faces is convex, but clipping can leave duplicate vertices and backwards
    spikes of 1e-17 m; an edge test on those would read a random direction."""
    p=sorted({(float(x),float(y)) for x,y in np.asarray(points,dtype=float).reshape(-1,2)})
    if len(p)<3:return np.asarray(p,dtype=float).reshape(-1,2)
    def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
    lower,upper=[],[]
    for q in p:
        while len(lower)>=2 and cross(lower[-2],lower[-1],q)<=0:lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper)>=2 and cross(upper[-2],upper[-1],q)<=0:upper.pop()
        upper.append(q)
    return np.asarray(lower[:-1]+upper[:-1],dtype=float)


def overlap_strip(polygon):
    """(width, length, direction) of a convex polygon in the tangent plane: its narrowest width,
    its extent along the direction that gives that width, and that direction as a unit 2-vector.
    A convex polygon's narrowest enclosing strip has one side along a polygon edge (rotating
    calipers), so the hull's edges are the only candidates. None when it has no edge."""
    p=convex_hull(polygon);best=None
    if len(p)<2:return None
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        edge=b-a;size=float(np.linalg.norm(edge))
        if not size>_EDGE_NOISE_M:continue
        along=edge/size;across=np.array([-along[1],along[0]])
        a_=p@across;l_=p@along
        width=float(a_.max()-a_.min());length=float(l_.max()-l_.min())
        if best is None or width<best[0]:best=(width,length,along)
    return best


def rounded_edge_contact(force,length,radius,modulus):
    """Area, length and half-width of an edge rounded to `radius` pressed with `force` along
    `length` into the coating: the Hertz line formula of finite_hertz_line_ignores_end_effects,
    half-width b = sqrt(4 F R / (pi L E)), area 2 L b. A contact is never shorter than it is
    wide: when length < 2b the edge's end acts as a rounded corner, the length becomes 2b and
    b = (2 F R / (pi E))^(1/3), continuous at length = 2b. That corner area, 4 b^2, is 1.14 times
    the Hertz circle of a sphere of the same radius on a flat, pi (3 F R / 4 E)^(2/3)."""
    halfwidth=math.sqrt(4*force*radius/(math.pi*length*modulus)) if length>0 else math.inf
    if length<2*halfwidth:
        halfwidth=(2*force*radius/(math.pi*modulus))**(1/3);length=2*halfwidth
    return 2*length*halfwidth,length,halfwidth


def holds_disk(polygon,radius):
    """Whether a convex polygon contains a disk of `radius`: its inset by `radius` (every edge moved
    inward by it) keeps a positive area. For two rounded faces this is whether their flat parts
    overlap, because insetting distributes over intersection: (A inset) and (B inset) overlap
    exactly where the overlap of A and B, inset, is not empty. Tested on the convex hull, whose
    edges run counter-clockwise."""
    p=convex_hull(polygon)
    if len(p)<3 or polygon_area(p)<=0:return False
    result=list(p)
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        edge=b-a;size=float(np.linalg.norm(edge))
        if not size>_EDGE_NOISE_M:continue
        incoming=result;result=[]
        if not incoming:break
        def side(q):return (edge[0]*(q-a)[1]-edge[1]*(q-a)[0])/size-radius
        previous=incoming[-1];ps=side(previous)
        for current in incoming:
            cs=side(current)
            if (cs>=0)!=(ps>=0):result.append(previous+(current-previous)*ps/(ps-cs))
            if cs>=0:result.append(current)
            previous=current;ps=cs
    return len(result)>=3 and polygon_area(result)>0


def rounded_edge_rescore(polygon,area,force,guard,modulus):
    """The guard's decision on one flat-face overlap (polygon in the tangent plane, its area):
    (edge area, edge length, half-width, (width, length, direction)) when the rounded edge's contact
    along the overlap is larger than the overlap, else None: the overlap's area is floored at that
    contact, and an overlap already larger is never changed. Continuous in the overlap's size."""
    strip=overlap_strip(polygon)
    if strip is None:return None
    edge_area,edge_length,halfwidth=rounded_edge_contact(force,strip[1],guard.edge_rounding_m,modulus)
    if not edge_area>area:return None
    return edge_area,edge_length,halfwidth,strip


def check_guard_note(record,guard):
    """Why a footprint the guard changed breaks the guard's rule, or None when it keeps it: the guard's stop rule
    on its own record (review 2026-09-25, finding 1). A floored flat overlap must carry the larger of its overlap
    and its rounded edge's contact along the overlap (then the silhouette cap, then the ceiling's floor), so never
    less than the overlap; a ceiling case must have read over the ceiling before and exactly the ceiling after;
    no other kind of change exists."""
    note=record.get('footprint_guard') or {}
    force=float(record['normal_force_n']);area=float(record['area_m2']);floor=force/guard.pressure_ceiling_pa
    if 'rescored' in note:
        if record.get('method')!=RESCORED_METHOD:return 'floored but method %r'%record.get('method')
        overlap=float(note['flat_overlap_area_m2'])
        if area<overlap*(1-1e-9):return 'the floored area %.6g m2 is under its flat overlap %.6g m2'%(area,overlap)
        modulus=float((record.get('material_profile') or {}).get('effective_modulus_pa',1e6))
        edge=rounded_edge_contact(force,float(note['flat_overlap_length_m']),guard.edge_rounding_m,modulus)[0]
        cap=float(record.get('projected_geometry_cap_m2') or math.inf)
        expected=max(min(max(overlap,edge),cap),floor)
        if abs(area-expected)>1e-9*expected:return 'area %.6g m2 where the rule gives %.6g m2'%(area,expected)
    if 'area_before_ceiling_m2' in note:
        before=float(note['area_before_ceiling_m2'])
        if not before<floor*(1+1e-12) or abs(area-floor)>1e-9*floor:
            return 'a ceiling case with %.6g m2 before and %.6g m2 after (floor %.6g m2)'%(before,area,floor)
    if not ('rescored' in note or 'area_before_ceiling_m2' in note):
        return 'a guard note of no known kind: %s'%sorted(note)
    return None


def bound_mean_pressure(record,guard):
    """The guard's pressure ceiling on a record made outside footprint(): the env's sensor-support
    fallback for shapes the geometry cannot resolve. Unchanged when the guard is off or the
    record's mean pressure is at or under the ceiling."""
    if guard is None or not record.get('supported',False):return record
    force=float(record['normal_force_n']);area=float(record['area_m2']);floor=force/guard.pressure_ceiling_pa
    if not area<floor:return record
    note=dict(record.get('footprint_guard') or {},setting=guard.name,pressure_ceiling_pa=guard.pressure_ceiling_pa,
              area_before_ceiling_m2=area)
    record.update(area_m2=floor,mean_pressure_pa=force/floor,tangent_axes_m=[math.sqrt(floor/math.pi)]*2,
                  footprint_guard=note)
    return record


def footprint(geometries,point,normal,normal_force,profile=VirtualSkinProfile(),guard=None):
    force=float(normal_force);point=np.asarray(point,dtype=float);normal=np.asarray(normal,dtype=float)
    if force<=0 or not math.isfinite(force):raise ValueError('footprint needs positive resolved force')
    if len(geometries)!=2 or point.shape!=(3,) or not np.isfinite(point).all():
        raise ValueError('contact footprint requires two shapes and a finite location')
    basis=tangent_basis(normal);normal=normal/np.linalg.norm(normal)
    kinds=[];matrix=np.zeros((2,2));lengths=[];caps=[]
    for geometry in geometries:
        k,length,kind=curvature(geometry,point,normal,basis);kinds.append(kind)
        cap=projected_area(geometry,normal)
        if k is None or cap is None:return dict(supported=False,reason=kind,contract=CONTRACT)
        matrix+=k
        if length is not None:lengths.append(length)
        caps.append(cap)
    values,vectors=np.linalg.eigh(matrix);values=np.maximum(values,0)
    cap=min(caps)
    exact=False;polygon=None;deformation_ratio=None
    guard_note=None
    if values[-1]<1e-8:
        polygons=[face_polygon(g,point,normal,basis) for g in geometries]
        finite=[p for p in polygons if p is not None]
        if not finite or any(len(p)<3 for p in finite):return dict(supported=False,reason='flat_patch_unresolved',contract=CONTRACT)
        polygon=finite[0]
        for other in finite[1:]:polygon=clip_polygon(polygon,other)
        area=polygon_area(polygon)
        if area<=0:return dict(supported=False,reason='no_finite_face_overlap',contract=CONTRACT)
        estimate=lambda modulus:area
        axes=np.array([np.sqrt(area/math.pi)]*2);method='aligned_flat_face_overlap';exact=True
        rescore=None if guard is None else rounded_edge_rescore(polygon,area,force,guard,profile.effective_modulus_pa)
        if rescore is not None:
            # The rounded edge's contact along the overlap is larger than the overlap: the load
            # spreads at least that far, so its contact is the area. A larger overlap stays as is.
            _,edge_length,halfwidth,(width,span,along)=rescore;overlap=area
            estimate=lambda modulus:max(overlap,rounded_edge_contact(force,span,guard.edge_rounding_m,modulus)[0])
            axes=np.array([edge_length/2,halfwidth]);vectors=np.stack([along,[-along[1],along[0]]],axis=1)
            method=RESCORED_METHOD;exact=False;deformation_ratio=float(halfwidth/guard.edge_rounding_m)
            guard_note=dict(setting=guard.name,rescored='the rounded edge contact is larger than the flat overlap',
                edge_rounding_m=guard.edge_rounding_m,flat_overlap_area_m2=overlap,
                flat_overlap_width_m=width,flat_overlap_length_m=span,edge_contact_length_m=float(edge_length),
                flat_parts_touch=bool(holds_disk(polygon,guard.edge_rounding_m)))
    elif values[0]<1e-8:
        if not lengths:return dict(supported=False,reason='line_length_unresolved',contract=CONTRACT)
        length=min(lengths);radius=1/values[-1]
        estimate=lambda modulus:2*length*np.sqrt(4*force*radius/(math.pi*length*modulus))
        halfwidth=np.sqrt(4*force*radius/(math.pi*length*profile.effective_modulus_pa))
        axes=np.array([length/2,halfwidth]);method='finite_hertz_line_ignores_end_effects'
        deformation_ratio=float(halfwidth/radius)
    else:
        radius=1/np.sqrt(np.prod(values))
        estimate=lambda modulus:math.pi*(3*force*radius/(4*modulus))**(2/3)
        a=np.sqrt(estimate(profile.effective_modulus_pa)/math.pi)
        aspect=(values[1]/values[0])**.25;axes=np.array([a*aspect,a/aspect])
        exact=abs(values[1]/values[0]-1)<1e-5
        method='hertz_circular' if exact else 'equivalent_curvature_hertz_ellipse_approximation'
        deformation_ratio=float(a/radius)
    nominal=float(estimate(profile.effective_modulus_pa))
    soft=float(estimate(profile.soft_modulus_pa));stiff=float(estimate(profile.stiff_modulus_pa))
    # A finite shape cannot expose more projected contact than its silhouette.
    # Reaching this bound marks departure from the small-deformation model.
    bounded=nominal>cap or soft>cap
    area=min(nominal,cap);soft=min(soft,cap);stiff=min(stiff,cap)
    if area<=0 or not math.isfinite(area):raise ValueError('virtual footprint has no finite area')
    if guard is not None:
        # The backstop for every shape: the ceiling outranks the silhouette cap above.
        floor=force/guard.pressure_ceiling_pa
        if area<floor:
            guard_note=dict(guard_note or {},setting=guard.name,pressure_ceiling_pa=guard.pressure_ceiling_pa,
                            area_before_ceiling_m2=area)
            area=floor
        soft=max(soft,floor);stiff=max(stiff,floor)
    record=dict(supported=True,contract=CONTRACT,method=method,shape_classes=kinds,
        area_m2=area,area_bounds_m2=[stiff,soft],mean_pressure_pa=force/area,
        pressure_bounds_pa=[force/soft,force/stiff],normal_force_n=force,
        tangent_axes_m=(axes*np.sqrt(area/nominal)).tolist(),tangent_basis=(basis@vectors).tolist(),
        polygon_tangent=None if polygon is None else np.asarray(polygon).tolist(),
        projected_geometry_cap_m2=float(cap),geometry_cap_reached=bool(bounded),
        circular_or_flat_formula=bool(exact),material_profile=asdict(profile),
        characteristic_patch_to_curvature_ratio=deformation_ratio,
        small_deformation_validity='Hertz assumes this ratio is much smaller than one; no tissue validity is certified.',
        scope='Declared virtual elastic coating; material bounds are sensitivity settings, not biological confidence bounds.')
    # Only a record the guard changed carries this key; with the guard off no record does.
    if guard_note is not None:record['footprint_guard']=guard_note
    return record


def aggregate_contact_footprints(model,data,contacts,profile=VirtualSkinProfile(),guard=None):
    """One footprint per body/convex geom pair, independent of solver point count."""
    groups={}
    for contact in contacts:
        groups.setdefault((contact.body_id,contact.pair_key),[]).append(contact)
    records=[]
    for (body_id,pair),group in groups.items():
        force=sum(c.normal_force for c in group)
        if force<=0:continue
        point=sum(c.normal_force*np.asarray(c.pos_world) for c in group)/force
        normal=sum(c.normal_force*np.asarray(c.normal_world) for c in group)/force
        normal/=np.linalg.norm(normal)
        geoms=[Geometry(int(model.geom_type[g]),np.asarray(model.geom_size[g]),
                        np.asarray(data.geom_xpos[g]),np.asarray(data.geom_xmat[g]).reshape(3,3)) for g in pair]
        record=footprint(geoms,point,normal,force,profile,guard)
        rotation=np.asarray(data.xmat[body_id]).reshape(3,3)
        record.update(body_id=body_id,pair=list(pair),position_world=point.tolist(),
            normal_world=normal.tolist(),
            position_body=(sum(c.normal_force*np.asarray(c.pos_body) for c in group)/force).tolist(),
            normal_force_n=force,tangential_force_n=float(np.linalg.norm(sum(np.asarray(c.tangential_body) for c in group))),
            friction_power_w=sum(c.friction_power_w or 0. for c in group),
            slip_mm_s=1000*max(c.rel_tangent_speed or 0. for c in group),solver_points=len(group))
        if record['supported']:record['tangent_basis_body']=(rotation.T@np.asarray(record['tangent_basis'])).tolist()
        records.append(record)
    return records
