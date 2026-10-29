# Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
# Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
"""Local virtual-skin precision modulation before body-level touch aggregation.

Taxel spacing determines spatial sampling, never the contact area or the material
law. Unloaded taxels retain their original precision. Existing body rows are a
summary of the experienced local signals, with locality reported explicitly.
"""
from dataclasses import asdict
import numpy as np

from lem.body_function import BodyFunction,physical_drives
from lem.skin_mechanics import MechanicalExposure
from .contact_footprint import CONTRACT,VirtualSkinProfile


class VirtualTaxelSkin:
    def __init__(self,specs,positions,sensitivity,profile=VirtualSkinProfile()):
        self.profile=profile;self.specs=list(specs);self.slices={};self.positions={};self.spacing={};areas=[];ids=[]
        gains=[];offset=0
        for index,spec in enumerate(self.specs):
            point=np.asarray(positions[spec.body_id],dtype=float).copy()
            if point.shape!=(spec.n_points,3) or not len(point) or not np.isfinite(point).all():
                raise ValueError('virtual skin requires finite existing taxel positions')
            _,inverse,count=np.unique(point,axis=0,return_inverse=True,return_counts=True)
            area=np.full(len(point),spec.point_area_m2)/count[inverse]
            self.positions[spec.body_id]=point
            self.spacing[spec.body_id]=spec.scale_m
            self.slices[spec.body_id]=slice(offset,offset+len(point));offset+=len(point)
            areas.extend(area);ids.extend(f'{spec.body_name}:{i}' for i in range(len(point)))
            gains.extend([sensitivity[index]]*len(point))
        self.sample_area=np.asarray(areas);self.sensitivity=np.asarray(gains,dtype=float)
        self.function=BodyFunction(ids);self.exposure=MechanicalExposure(offset,pain_sensitivity=self.sensitivity[:,0])
        self.physical=np.zeros((offset,10),dtype=np.float64);self.physical[:,8:]=1
        self.loaded_area=np.zeros(offset);self.normal_force=np.zeros(offset);self.shear_force=np.zeros(offset)
        self.friction_power=np.zeros(offset);self.last_time=None;self.contact_records=[]
        self.experienced=self.physical.copy()
        # A second per-taxel memory under a recalibrated rubbing rule (skin_mechanics.RUBBING_CALIBRATIONS),
        # advanced with exactly the same inputs; None when no calibration is set. self.exposure stays
        # today's memory and alone drives the local function (tissue impairment) below.
        self.pain_exposure=None

    def set_pain_profile(self,profile):
        """Advance a second memory under `profile` beside the existing one, or stop (None). A new one
        starts as a copy of the existing memory, so loads already stored keep fading instead of vanishing;
        one restored from a saved state keeps its own values under the profile given now."""
        if profile is None:
            self.pain_exposure=None
            return
        pain=getattr(self,'pain_exposure',None)
        if pain is None or pain.values.shape!=self.exposure.values.shape:
            pain=MechanicalExposure(len(self.exposure.values),profile,pain_sensitivity=self.exposure.pain_sensitivity)
            pain.values[:]=self.exposure.values
        pain.profile=profile
        self.pain_exposure=pain

    def inherit_body_function(self,body_function):
        """Explicit boundary migration preserves existing local impairment evidence."""
        if tuple(body_function.site_ids)!=tuple(spec.body_name for spec in self.specs):
            raise ValueError('cannot migrate mismatched skin identities')
        for i,spec in enumerate(self.specs):
            segment=self.slices[spec.body_id]
            self.function.quality[segment]=body_function.quality[i]
            self.function.last_load[segment]=body_function.last_load[i]
            self.function.last_stimulation[segment]=body_function.last_stimulation[i]
        self.function.last_time=body_function.last_time

    def _weights(self,body_id,record):
        segment=self.slices[body_id];points=self.positions[body_id]
        center=np.asarray(record['position_body']);basis=np.asarray(record['tangent_basis_body'])
        axes=np.asarray(record['tangent_axes_m']);delta=points-center
        projected=delta@basis;normalized=projected/axes
        radial=np.sum(normalized**2,axis=1)
        normal=np.cross(basis[:,0],basis[:,1]);normal_distance=abs(delta@normal)
        # Taxels on the back of a thin body cannot feel a front-side contact.
        # A long strip's LENGTH is not permission to sample through the body.
        # Use its narrow width plus the existing sampling tolerance.
        near_surface=normal_distance<=max(self.spacing[body_id],axes.min())
        if record['method']=='finite_hertz_line_ignores_end_effects':
            eligible=(abs(normalized[:,0])<=1)&(abs(normalized[:,1])<=1)&near_surface
            shape=np.sqrt(np.maximum(1-normalized[:,1]**2,0))
        elif record['method']=='aligned_flat_face_overlap' and record.get('polygon_tangent') is not None:
            polygon=np.asarray(record['polygon_tangent']);eligible=near_surface.copy()
            for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
                edge=b-a;offset=projected-a
                eligible&=(edge[0]*offset[:,1]-edge[1]*offset[:,0])>=-1e-12
            shape=np.ones(len(points))
        else:
            eligible=(radial<=1)&near_surface
            shape=np.ones(len(points)) if record['method']=='aligned_flat_face_overlap' else np.sqrt(np.maximum(1-radial,0))
        if not eligible.any() or not np.any(shape[eligible]>0):
            # A physical patch smaller than a taxel's spacing is observed by its
            # nearest sample. Its pressure still uses physical area, not spacing².
            distance=np.linalg.norm(delta,axis=1);eligible=np.isclose(distance,distance.min(),rtol=0,atol=1e-12)
            shape=np.ones(len(points))
        area_weights=self.sample_area[segment]*eligible
        area_weights/=area_weights.sum()
        force_weights=area_weights*shape
        force_weights/=force_weights.sum()
        return segment,area_weights,force_weights

    def advance(self,records,now,dt,*,aggregate=True):
        if not np.isfinite(now) or not np.isfinite(dt) or dt<=0:raise ValueError('invalid virtual skin time')
        if self.last_time is not None and now<self.last_time:raise ValueError('virtual skin time reversed')
        self.physical.fill(0);self.physical[:,8:]=1
        self.loaded_area.fill(0);self.normal_force.fill(0);self.shear_force.fill(0);self.friction_power.fill(0)
        self.contact_records=[];unsupported=[]
        for record in records:
            if not record.get('supported',False):
                raise ValueError('unresolved contact load requires explicit fallback; it cannot disappear from sensing')
            if record.get('geometry_supported') is False:unsupported.append(record)
            segment,area_weights,force_weights=self._weights(record['body_id'],record)
            self.loaded_area[segment]+=record['area_m2']*area_weights
            self.normal_force[segment]+=record['normal_force_n']*force_weights
            self.shear_force[segment]+=record.get('tangential_force_n',0.)*force_weights
            self.friction_power[segment]+=record.get('friction_power_w',0.)*force_weights
            self.physical[segment,2]=np.maximum(self.physical[segment,2],record.get('slip_mm_s',0.)*(force_weights>0))
            self.contact_records.append(record)
        area=self.loaded_area
        self.physical[:,0]=np.divide(self.normal_force,area,out=np.zeros_like(area),where=area>0)
        self.physical[:,1]=np.divide(self.shear_force,area,out=np.zeros_like(area),where=area>0)
        self.physical[:,3]=area*1e6
        self.unsupported=unsupported
        return self._advance_function(now,dt,aggregate=aggregate)

    def advance_support(self,fields,point_forces,power_density,now,dt,*,aggregate=True):
        """Localize the existing v2 pressure proxy without changing its area law.

        These remain sensor-support pressures, not resolved tissue pressures.
        All active samples of a body see that body's existing pressure/shear;
        inactive samples keep their own sensing and exposure recovery.
        """
        from .touch_adapter import ACTIVE_FORCE_EPS
        fields=np.asarray(fields);power_density=np.asarray(power_density)
        if fields.shape!=(len(self.specs),10) or power_density.shape!=(len(self.specs),) or not np.isfinite(fields).all():
            raise ValueError('local v2 skin requires complete native body pressure fields')
        self.physical.fill(0);self.physical[:,8:]=1
        self.loaded_area.fill(0);self.normal_force.fill(0);self.shear_force.fill(0);self.friction_power.fill(0)
        self.contact_records=[];self.unsupported=[]
        for i,spec in enumerate(self.specs):
            segment=self.slices[spec.body_id]
            forces=np.asarray(point_forces.get(spec.body_id,np.zeros((spec.n_points,3))))
            if forces.shape!=(spec.n_points,3) or not np.isfinite(forces).all():raise ValueError('local v2 taxel identity changed')
            active=np.linalg.norm(forces,axis=-1)>ACTIVE_FORCE_EPS
            area=float(fields[i,3])*1e-6
            if area<=0:continue
            if not active.any():raise ValueError('native pressure area has no corresponding active skin sample')
            weights=self.sample_area[segment]*active;weights/=weights.sum()
            self.loaded_area[segment]=area*weights
            self.physical[segment,:3]=active[:,None]*fields[i,:3]
            self.physical[segment,3]=self.loaded_area[segment]*1e6
            self.normal_force[segment]=fields[i,0]*self.loaded_area[segment]
            self.shear_force[segment]=fields[i,1]*self.loaded_area[segment]
            self.friction_power[segment]=power_density[i]*self.loaded_area[segment]
        return self._advance_function(now,dt,aggregate=aggregate)

    def _advance_function(self,now,dt,*,aggregate):
        if not np.isfinite(now) or not np.isfinite(dt) or dt<=0:raise ValueError('invalid local skin time')
        if self.last_time is not None and now<self.last_time:raise ValueError('local skin time reversed')
        area=self.loaded_area
        power=np.divide(self.friction_power,area,out=np.zeros_like(area),where=area>0)
        self.exposure.update(self.physical,dt,power)
        if getattr(self,'pain_exposure',None) is not None:self.pain_exposure.update(self.physical,dt,power)
        load,stimulation=physical_drives(self.physical[:,0],self.exposure.values,self.sensitivity)
        self.function.advance(load,stimulation,float(now))
        # Same operator as BodyFunction.observe, vectorized for thousands of
        # taxels at physical substeps. Never quantize intact observations.
        self.experienced=self.physical.copy();p=self.function.profile
        mask=self.function.quality<1-1e-12
        bits=np.clip(np.rint(p.minimum_bits+(p.maximum_bits-p.minimum_bits)*self.function.quality[mask]),p.minimum_bits,p.maximum_bits)
        steps=p.pressure_range_pa/(2.**bits-1)
        self.experienced[mask,:2]=np.rint(self.physical[mask,:2]/steps[:,None])*steps[:,None]
        self.last_time=float(now)
        return self.aggregate() if aggregate else None

    def aggregate(self):
        physical=np.zeros((len(self.specs),10));experienced=physical.copy()
        physical[:,8:]=1;experienced[:,8:]=1
        function=[];affected=[];mean_quality=[];surface_moments=[]
        values=self.function.features();p=self.function.profile
        bits=np.rint(p.minimum_bits+(p.maximum_bits-p.minimum_bits)*self.function.quality)
        for i,spec in enumerate(self.specs):
            segment=self.slices[spec.body_id];area=self.loaded_area[segment];total=area.sum()
            for output,source in [(physical,self.physical),(experienced,self.experienced)]:
                if total>0:
                    output[i,:2]=(source[segment,:2]*area[:,None]).sum(0)/total
                    output[i,2]=source[segment,2].max();output[i,3]=total*1e6
            # This is explicitly the worst local site, not uniform impairment of
            # every taxel on that body. The affected area is reported alongside it.
            function.append([values[segment,0].min(),values[segment,1].max(),values[segment,2].max()])
            weights=self.sample_area[segment];quality=self.function.quality[segment]
            affected.append(float(np.sum(weights*(bits[segment]<p.maximum_bits))/weights.sum()))
            mean_quality.append(float(np.sum(weights*quality)/weights.sum()))
            # Use sensed pressure, not the hidden unmodulated force, for the
            # observed contact centre. Function location is an internal sense.
            sensed=np.maximum(self.experienced[segment,0],0)*area
            impaired=weights*(1-quality)
            moments=[]
            for field in (sensed,impaired):
                if field.sum()>0:
                    centre=(self.positions[spec.body_id]*field[:,None]).sum(0)/field.sum()
                    delta=self.positions[spec.body_id]-centre
                    covariance=(delta.T*field)@delta/field.sum()
                    moments.extend(centre.tolist()+covariance[np.triu_indices(3)].tolist())
                else:moments.extend([0.]*9)
            moments.extend([float(sensed.sum()>0),float(impaired.sum()>0)])
            surface_moments.append(moments)
        return dict(physical=physical.astype(np.float32),experienced=experienced.astype(np.float32),
            values=np.asarray(function,dtype=np.float32),area_weighted_quality=mean_quality,surface_moments_body=surface_moments,
            affected_sensor_area_fraction=affected,contract=CONTRACT,profile=asdict(self.profile),
            taxels=len(self.sample_area),supported_contacts=len(self.contact_records),unsupported_contacts=len(self.unsupported),
            scope='Physical footprint is a virtual material estimate. Function is per taxel; existing body rows report its worst local precision and affected sensor area.')
