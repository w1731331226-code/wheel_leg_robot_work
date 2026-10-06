"""Shadow query copy:only motion tilt/integral/velocity coordinates differ."""
from native.controller import *
wp.set_module_options({"enable_backward": False})

@wp.func
def control_step(w:int,qpos:wp.array2d[float],qvel:wp.array2d[float],sensor:wp.array2d[float],
            targets:wp.array2d[float],command:wp.array[D],active:wp.array[int],state:wp.array2d[D],
            ids:wp.array[int],heights:wp.array[D],gains:wp.array3d[D],feed:wp.array2d[D],angles:wp.array[D],
            reference:wp.array2d[D],yaw_cfg:wp.array[D],ctrl:wp.array2d[float],diagnostic:wp.array2d[D],project_clipped_base:int,grouped_residual:int,actuator_gains:V6,nominal_correction:V6,nominal_correction_enabled:int,raw_motor:int):
    if active[w]==0:return
    dt=D(.0005);boot=state[w,0]+dt;state[w,0]=boot
    qw=D(qpos[w,3]);qx=D(qpos[w,4]);qy=D(qpos[w,5]);qz=D(qpos[w,6])
    roll=wp.atan2(D(2)*(qw*qx+qy*qz),D(1)-D(2)*(qx*qx+qy*qy))
    pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    qa=D(qpos[w,ids[0]]);qb=D(qpos[w,ids[1]]);qc=D(qpos[w,ids[2]]);qd=D(qpos[w,ids[3]])
    va=D(qvel[w,ids[4]]);vb=D(qvel[w,ids[5]]);vc=D(qvel[w,ids[6]]);vd=D(qvel[w,ids[7]])
    speeds=V6(va,vb,vc,vd,D(qvel[w,ids[8]]),D(qvel[w,ids[9]]))
    bounds=command_bounds(speeds,actuator_gains)
    left=fk(qa,qb);right=fk(qc,qd);length=(left[3]+right[3])/D(2)
    th=(left[2]+right[2])/D(2)+D(PI)/D(2)-pitch
    state[w,3]=state[w,3]+((length-state[w,1])/dt-state[w,3])*D(.05);state[w,1]=length
    state[w,4]=state[w,4]+((th-state[w,2])/dt-state[w,4])*D(.05);state[w,2]=th
    gyro=ids[10]
    state[w,5]=state[w,5]+(D(sensor[w,gyro])-state[w,5])*D(.05)
    state[w,6]=state[w,6]+(D(sensor[w,gyro+1])-state[w,6])*D(.05)
    vx=wp.cos(yaw)*D(qvel[w,0])+wp.sin(yaw)*D(qvel[w,1])
    state[w,7]=state[w,7]+(vx-state[w,7])*D(.025)
    state[w,8]=state[w,8]+(D(sensor[w,gyro+2])-state[w,8])*D(.025)
    cmd=D(command[w]);motion_cmd=cmd
    if reference.shape[1]>16:motion_cmd=reference[w,16]
    error=wp.atan2(wp.sin(yaw-state[w,9]),wp.cos(yaw-state[w,9]))
    parking_guard=reference.shape[1]>12 and reference[w,10]==D(2)
    parking_index=20+targets.shape[1]
    if parking_guard:
        if wp.abs(cmd)>=D(.01):state[w,parking_index+2]=wp.sign(cmd)
        elif state[w,12]<D(0) and state[w,parking_index+2]!=D(0):
            state[w,parking_index]=D(qpos[w,0]);state[w,parking_index+1]=D(qpos[w,1])
    if wp.abs(cmd)>=D(.01):state[w,12]=D(-1)
    elif state[w,12]<D(0):state[w,12]=boot
    if wp.abs(cmd)<D(.01) and wp.abs(state[w,7])<D(.05) and state[w,10]<D(.1):state[w,9]=yaw
    if wp.abs(cmd)<D(.01):state[w,10]=state[w,10]+dt
    else:state[w,10]=D(0)
    if wp.abs(cmd)<D(.01):state[w,11]=D(0)
    elif wp.abs(state[w,7]-cmd)>D(.08):state[w,11]=wp.clamp(state[w,11]+(state[w,7]-motion_cmd)*dt,D(-.3),D(.3))
    ktha=D(-.3)
    if wp.abs(cmd)<D(.01):ktha=D(-.6)
    vmax=D(0)
    if wp.abs(state[w,7])>D(1):vmax=wp.sign(state[w,7])*(wp.abs(state[w,7])-D(1))
    thcmd=wp.clamp((ktha*(motion_cmd-state[w,7])+D(.08)*state[w,11]+D(.5)*pitch)*wp.min(D(1),boot/D(.5))+vmax,D(-.2),D(.2))
    if boot<D(1):thcmd=D(0)
    braking=wp.abs(cmd)<D(.01) and wp.abs(state[w,7])>D(.03)
    if braking:
        thcmd=wp.sign(state[w,7])*wp.min(D(sim.STOP_LEAN_MAX),D(sim.STOP_LEAN_BASE)+D(sim.STOP_LEAN_GAIN)*wp.abs(state[w,7]))*wp.clamp((boot-state[w,12])/D(.1),D(0),D(1))
    kd=D(.5)
    if boot<D(.6):kd=D(1.2)
    hub_old=D(4)*(thcmd-th)-kd*state[w,4]
    if braking:hub_old=-(D(4)*(th-thcmd)+D(.5)*state[w,4]-D(2)*pitch)*D(1.5)-D(4)*pitch-D(.5)*state[w,6]
    offset=wp.clamp(D(.30)*roll+D(.12)*state[w,5],D(-.035),D(.035))
    if reference.shape[1]>6 and reference[w,4]>D(0):
        room=wp.max(D(0),wp.min(reference[w,2]-reference[w,3],reference[w,6]-reference[w,2]))
        offset=wp.clamp(offset,-room,room)
    left_target=reference[w,2]+offset;right_target=reference[w,2]-offset
    if reference.shape[1]>3:
        left_target=wp.max(left_target,reference[w,3])
        right_target=wp.max(right_target,reference[w,3])
    height=length*wp.cos(th);gravity=D(4)*D(MASS)*wp.min(D(1),boot/D(.15))
    fl=wp.clamp(D(MASS)*(D(500)*(left_target-height)-D(25)*state[w,3]*wp.cos(th))+gravity,-D(40)*D(MASS),D(40)*D(MASS))
    fr=wp.clamp(D(MASS)*(D(500)*(right_target-height)-D(25)*state[w,3]*wp.cos(th))+gravity,-D(40)*D(MASS),D(40)*D(MASS))
    kp=D(.8)*D(MASS);damping=D(.08)*D(MASS)
    if braking:kp=kp*wp.clamp((D(.3)-wp.abs(state[w,7]))/D(.2),D(0),D(1))
    old_l=legacy_vmc(qa,qb,fl,hub_old)+V2(kp*(reference[w,0]-qa)-damping*va,kp*(reference[w,1]-qb)-damping*vb)
    old_r=legacy_vmc(qc,qd,fr,hub_old)+V2(kp*(reference[w,0]-qc)-damping*vc,kp*(reference[w,1]-qd)-damping*vd)
    undamped_l=old_l+V2(damping*va,damping*vb);undamped_r=old_r+V2(damping*vc,damping*vd)
    limit=D(.8)*D(MASS)
    if braking:limit=D(2)*D(MASS)
    old_l=V2(wp.clamp(old_l[0],-wp.min(limit,bounds[0]),wp.min(limit,bounds[0])),wp.clamp(old_l[1],-wp.min(limit,bounds[1]),wp.min(limit,bounds[1])))
    old_r=V2(wp.clamp(old_r[0],-wp.min(limit,bounds[2]),wp.min(limit,bounds[2])),wp.clamp(old_r[1],-wp.min(limit,bounds[3]),wp.min(limit,bounds[3])))
    undamped_l=V2(wp.clamp(undamped_l[0],-wp.min(limit,bounds[0]),wp.min(limit,bounds[0])),wp.clamp(undamped_l[1],-wp.min(limit,bounds[1]),wp.min(limit,bounds[1])))
    undamped_r=V2(wp.clamp(undamped_r[0],-wp.min(limit,bounds[2]),wp.min(limit,bounds[2])),wp.clamp(undamped_r[1],-wp.min(limit,bounds[3]),wp.min(limit,bounds[3])))
    jl=polar_jac(qa,qb);jr=polar_jac(qc,qd)
    rate_l=jl[0,0]*va+jl[1,0]*vb;rate_r=jr[0,0]*vc+jr[1,0]*vd
    arate_l=jl[0,1]*va+jl[1,1]*vb;arate_r=jr[0,1]*vc+jr[1,1]*vd
    index=int(0)
    for knot in range(1,heights.shape[0]-1):
        if length>heights[knot]:index=knot
    ratio=wp.clamp((length-heights[index])/(heights[index+1]-heights[index]),D(0),D(1))
    theta_eq=(D(1)-ratio)*angles[index]+ratio*angles[index+1]
    arrival_hold=reference.shape[1]>10 and reference[w,10]==D(1)
    if wp.abs(cmd)>D(.01) or (not arrival_hold and wp.abs(vx)>D(.03)):state[w,13]=D(0)
    elif state[w,13]==D(0):
        state[w,13]=D(1);state[w,14]=D(qpos[w,0]);state[w,15]=D(qpos[w,1])
    position=D(0)
    if state[w,13]>D(0):position=wp.cos(yaw)*(D(qpos[w,0])-state[w,14])+wp.sin(yaw)*(D(qpos[w,1])-state[w,15])
    pitch_rate=D(sensor[w,gyro+1])
    x=V6(th-theta_eq,(arate_l+arate_r)/D(2)-pitch_rate,position,vx-motion_cmd,pitch,pitch_rate)
    wheel=(D(1)-ratio)*feed[index,0]+ratio*feed[index+1,0]
    hub=(D(1)-ratio)*feed[index,1]+ratio*feed[index+1,1]
    support=(D(1)-ratio)*feed[index,2]+ratio*feed[index+1,2]
    for j in range(6):
        wheel=wheel-((D(1)-ratio)*gains[index,0,j]+ratio*gains[index+1,0,j])*x[j]
        hub=hub-((D(1)-ratio)*gains[index,1,j]+ratio*gains[index+1,1,j])*x[j]
    vl=inverse2(jl)*old_l;vr=inverse2(jr)*old_r;average=(vl[1]+vr[1])/D(2)
    hl=vl[1]+hub-average;hr=vr[1]+hub-average
    original_hub_mean=(hl+hr)/D(2)
    if reference.shape[1]>6 and reference[w,5]>D(0):
        kg=(D(1)-ratio)*gains[index,1,0]+ratio*gains[index+1,1,0]
        if kg<=D(0) or not wp.isfinite(kg):
            for j in range(6):ctrl[w,j]=0.
            diagnostic[w,14]=D(2)
            return
        hd=-((D(1)-ratio)*gains[index,1,1]+ratio*gains[index+1,1,1])*x[1]-((D(1)-ratio)*gains[index,1,5]+ratio*gains[index+1,1,5])*x[5]
        # Preserve the existing differential motor-velocity damping. Static
        # pose projection must not reinterpret it as an infeasible setpoint.
        vl_without_damping=inverse2(jl)*undamped_l;vr_without_damping=inverse2(jr)*undamped_r
        differential_damping=((vl[1]-vl_without_damping[1])-(vr[1]-vr_without_damping[1]))/D(2)
        hdl=hd+differential_damping;hdr=hd-differential_damping
        al=left[2]+D(PI)/D(2);ar=right[2]+D(PI)/D(2)
        requested_l=al+(hl-hdl)/kg;requested_r=ar+(hr-hdr)/kg
        safe_l,valid_l=project_leg_angle(left_target,requested_l,reference[w,5])
        safe_r,valid_r=project_leg_angle(right_target,requested_r,reference[w,5])
        if not valid_l or not valid_r:
            for j in range(6):ctrl[w,j]=0.
            diagnostic[w,14]=D(2)
            return
        if safe_l!=requested_l:hl=kg*(safe_l-al)+hdl
        if safe_r!=requested_r:hr=kg*(safe_r-ar)+hdr
        if reference.shape[1]>7 and reference[w,7]>D(0):
            channel=int(3)
            if reference[w,7]==D(2):channel=0
            kh=(D(1)-ratio)*gains[index,1,channel]+ratio*gains[index+1,1,channel]
            kw=(D(1)-ratio)*gains[index,0,channel]+ratio*gains[index+1,0,channel]
            # Update BOTH inputs in the same reference coordinate. Angle mode
            # changes the angle reference and keeps the speed reference fixed.
            wheel=wheel+(kw/kh)*((hl+hr)/D(2)-original_hub_mean)
        if diagnostic.shape[1]>=31:
            diagnostic[w,21]=left_target;diagnostic[w,22]=right_target
            diagnostic[w,23]=requested_l;diagnostic[w,24]=requested_r
            diagnostic[w,25]=safe_l;diagnostic[w,26]=safe_r;diagnostic[w,27]=hd
            diagnostic[w,28]=D(int(safe_l!=requested_l)+int(safe_r!=requested_r))
            diagnostic[w,29]=hl;diagnostic[w,30]=hr
    fleft=support+D(MASS)*(D(500)*(left_target-left[3])-D(25)*rate_l)
    fright=support+D(MASS)*(D(500)*(right_target-right[3])-D(25)*rate_r)
    tl=jl*V2(fleft,hl);tr=jr*V2(fright,hr)
    if reference.shape[1]>9 and reference[w,8]>D(0):
        k=reference[w,8];il=inverse2(jl);ir=inverse2(jr)
        # Public mass upper bound plus reflected nominal hip-rotor energy.
        ml=reference[w,9]+D(HIP_INERTIA)*(il[0,0]*il[0,0]+il[0,1]*il[0,1])
        mr=reference[w,9]+D(HIP_INERTIA)*(ir[0,0]*ir[0,0]+ir[0,1]*ir[0,1])
        # Preserve validated low-height support. Above the original nominal
        # design's lower node, do not turn every tracking target into a barrier.
        radial_reference=wp.min(reference[w,2],D(sim.L_SQUAT_MIN))
        if reference.shape[1]>15 and reference[w,15]>D(0):
            radial_reference=reference[w,15]
        requested=wp.max(D(0),wp.max(ml*(-D(2)*k*rate_l-k*k*(left[3]-radial_reference)),
                                   mr*(-D(2)*k*rate_r-k*k*(right[3]-radial_reference))))
        room=wp.min(outward_force_headroom(V2(jl[0,0],jl[1,0]),tl,V2(bounds[0],bounds[1])),
                    outward_force_headroom(V2(jr[0,0],jr[1,0]),tr,V2(bounds[2],bounds[3])))
        applied=wp.min(requested,room)
        if state.shape[1]>=20+targets.shape[1]:
            base_index=16+targets.shape[1]
            state[w,base_index]=wp.max(state[w,base_index],requested)
            state[w,base_index+1]=wp.max(state[w,base_index+1],applied)
            state[w,base_index+2]+=D(int(applied<requested))
            state[w,base_index+3]+=D(int(applied>D(0)))
        fleft+=applied;fright+=applied
        tl=jl*V2(fleft,hl);tr=jr*V2(fright,hr)
        if diagnostic.shape[1]>=38:
            diagnostic[w,31]=requested;diagnostic[w,32]=applied;diagnostic[w,33]=ml;diagnostic[w,34]=mr
            diagnostic[w,35]=fleft;diagnostic[w,36]=fright;diagnostic[w,37]=D(int(applied<requested))
    yaw_torque=D(0)
    if parking_guard and wp.abs(cmd)<D(.01) and state[w,parking_index+2]!=D(0):
        dx=D(qpos[w,0])-state[w,parking_index];dy=D(qpos[w,1])-state[w,parking_index+1]
        remaining=reference[w,11]-wp.sqrt(dx*dx+dy*dy)
        braking_torque=wp.min(bounds[4],bounds[5])
        if remaining>D(0):braking_torque=D(sim.hw.WHEEL_RADIUS)*reference[w,12]*vx*vx/(D(4)*remaining)
        wheel-=wp.sign(vx)*braking_torque
        state[w,parking_index+3]=wp.max(state[w,parking_index+3],braking_torque)
        state[w,parking_index+4]+=D(int(braking_torque>D(0)))
    if wp.abs(error)>=D(PI)/D(360) or wp.abs(state[w,8])>=D(.05):yaw_torque=wp.clamp(-yaw_cfg[0]*error-yaw_cfg[1]*state[w,8],-yaw_cfg[2]*D(MASS),yaw_cfg[2]*D(MASS))
    base=V6(tl[0],tl[1],tr[0],tr[1],wheel+yaw_torque,wheel-yaw_torque)
    invalid_base=int(0);invalid_leg_base=int(0);invalid_wheel_base=int(0);bad_control=bool(False)
    for j in range(6):
        if diagnostic.shape[1]>=21:diagnostic[w,15+j]=base[j]
        if not wp.isfinite(base[j]):bad_control=True
        maximum=D(4.5)
        if j<4:maximum=D(40)
        base[j]=wp.clamp(base[j],-maximum,maximum)
        bound=bounds[j]
        if base[j]<-bound-D(1.e-9) or base[j]>bound+D(1.e-9):
            invalid_base=1
            if j<4:invalid_leg_base=1
            else:invalid_wheel_base=1
        base[j]=wp.clamp(base[j],-bound,bound)
    projection_mode=project_clipped_base
    if nominal_correction_enabled:
        projection_mode=1
        for j in range(6):
            if not wp.isfinite(nominal_correction[j]) or wp.abs(nominal_correction[j])>D(1):bad_control=True
            # Match the formerly external zero-Actor path's float32 accepted Nom.
            base[j]=wp.clamp(D(float(base[j]))+nominal_correction[j],-bounds[j],bounds[j])
    for j in range(targets.shape[1]):state[w,16+j]=state[w,16+j]+wp.clamp(D(targets[w,j])-state[w,16+j],D(-.01),D(.01))
    rr_l=jl*V2(state[w,16]*D(.1)*D(7)*D(9.81)/D(2),state[w,17])
    rr_r=jr*V2(-state[w,16]*D(.1)*D(7)*D(9.81)/D(2),-state[w,17])
    residual=V6(rr_l[0],rr_l[1],rr_r[0],rr_r[1],state[w,18]*yaw_cfg[3],-state[w,18]*yaw_cfg[3])
    if targets.shape[1]==6:
        rr_l=jl*V2(state[w,16]*D(.1)*D(7)*D(9.81)/D(2),state[w,18])
        rr_r=jr*V2(state[w,17]*D(.1)*D(7)*D(9.81)/D(2),state[w,19])
        residual=V6(rr_l[0],rr_l[1],rr_r[0],rr_r[1],state[w,20]*yaw_cfg[3],state[w,21]*yaw_cfg[3])
    if raw_motor:
        residual=V6(state[w,16],state[w,17],state[w,18],state[w,19],state[w,20]*yaw_cfg[3],state[w,21]*yaw_cfg[3])
    bad_map=bool(False)
    for side in range(2):
        matrix=jl
        if side==1:matrix=jr
        square=matrix[0,0]*matrix[0,0]+matrix[0,1]*matrix[0,1]+matrix[1,0]*matrix[1,0]+matrix[1,1]*matrix[1,1]
        det=wp.abs(matrix[0,0]*matrix[1,1]-matrix[0,1]*matrix[1,0])
        maximum=(square+wp.sqrt(wp.max(D(0),square*square-D(4)*det*det)))/D(2)
        if not raw_motor and (det==D(0) or maximum/det>D(1.e6)):bad_map=True
    nonzero=bool(False);leg_nonzero=bool(False);wheel_nonzero=bool(False)
    for j in range(targets.shape[1]):
        if state[w,16+j]!=D(0):
            nonzero=True
            if j<2:leg_nonzero=True
            else:wheel_nonzero=True
    lam,projection_error=project_bounds(base,residual,speeds,bounds,invalid_base,bad_map,nonzero,bad_control,projection_mode)
    leg_lam=lam;wheel_lam=lam
    if grouped_residual:
        leg_residual=V6(residual[0],residual[1],residual[2],residual[3],D(0),D(0))
        wheel_residual=V6(D(0),D(0),D(0),D(0),residual[4],residual[5])
        leg_lam,leg_error=project_bounds(base,leg_residual,speeds,bounds,invalid_leg_base,bad_map,leg_nonzero,bad_control,0)
        wheel_lam,wheel_error=project_bounds(base,wheel_residual,speeds,bounds,invalid_wheel_base,bad_map,wheel_nonzero,bad_control,0)
        lam=wp.min(leg_lam,wheel_lam);projection_error=leg_error
        if wheel_error>projection_error:projection_error=wheel_error
    diagnostic[w,14]=D(projection_error)
    if (projection_mode and projection_error) or (grouped_residual and projection_error==2):
        # Do not send 0*NaN or a singular mapping to physics; after() terminates it.
        for j in range(6):
            ctrl[w,j]=0.;diagnostic[w,j]=D(0);diagnostic[w,6+j]=D(0)
        diagnostic[w,12]=D(0);diagnostic[w,13]=D(invalid_base)
        return
    for j in range(6):
        group_lam=leg_lam
        if j>=4:group_lam=wheel_lam
        ctrl[w,j]=float(wp.clamp(base[j]+group_lam*residual[j],-bounds[j],bounds[j]))
        diagnostic[w,j]=base[j];diagnostic[w,6+j]=group_lam*residual[j]
    for j in range(6):
        if not wp.isfinite(ctrl[w,j]) or not wp.isfinite(residual[j]):diagnostic[w,14]=D(2)
    diagnostic[w,12]=lam;diagnostic[w,13]=D(invalid_base)

@wp.kernel
def control_physical_nominal(qpos:wp.array2d[float],qvel:wp.array2d[float],sensor:wp.array2d[float],
            targets:wp.array2d[float],command:wp.array[D],active:wp.array[int],state:wp.array2d[D],
            ids:wp.array[int],heights:wp.array[D],gains:wp.array3d[D],feed:wp.array2d[D],angles:wp.array[D],
            reference:wp.array2d[D],yaw_cfg:wp.array[D],ctrl:wp.array2d[float],diagnostic:wp.array2d[D],project_clipped_base:int,grouped_residual:int,
            actuator_gains:wp.array2d[D],nominal_correction:wp.array2d[D]):
    w=wp.tid();scales=V6();correction=V6()
    for j in range(6):scales[j]=actuator_gains[w,j];correction[j]=nominal_correction[w,j]
    control_step(w,qpos,qvel,sensor,targets,command,active,state,ids,heights,gains,feed,angles,reference,yaw_cfg,ctrl,diagnostic,
                 project_clipped_base,grouped_residual,scales,correction,1,0)
