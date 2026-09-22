"""GPU驻留地面M3控制器；不接管跳跃，原CPU控制器保留为逐点参考。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wheelleg_ppo/tools'))
import numpy as np
import warp as wp
import wheelleg_sim as sim
from rm_controller import nominal_design

D=wp.float64
V2=wp.types.vector(length=2,dtype=D)
V3=wp.types.vector(length=3,dtype=D)
V4=wp.types.vector(length=4,dtype=D)
V6=wp.types.vector(length=6,dtype=D)
M2=wp.types.matrix(shape=(2,2),dtype=D)
PI=wp.constant(np.pi)
P1=wp.constant(sim.PHI1_STAND);P4=wp.constant(sim.PHI4_STAND)
MASS=wp.constant(sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS)


@wp.func
def fk(qa:D,qb:D):
    p1=D(P1)-qa;p4=D(P4)-qb
    bx=D(.15)*wp.cos(p1);bz=D(.15)*wp.sin(p1)
    dx=D(.15)+D(.15)*wp.cos(p4);dz=D(.15)*wp.sin(p4)
    bd=wp.sqrt((dx-bx)*(dx-bx)+(dz-bz)*(dz-bz))
    aa=D(.54)*(dx-bx);bb=D(.54)*(dz-bz);cc=D(.27)*D(.27)+bd*bd-D(.27)*D(.27)
    p2=D(2)*wp.atan2(bb-wp.sqrt(wp.max(D(0),aa*aa+bb*bb-cc*cc)),aa+cc)
    cx=bx+D(.27)*wp.cos(p2);cz=bz+D(.27)*wp.sin(p2)
    angle=wp.atan2(cz,cx-D(.075));length=wp.sqrt((cx-D(.075))*(cx-D(.075))+cz*cz)
    return V4(cx,cz,angle,length)


@wp.func
def inverse2(a:M2):
    determinant=a[0,0]*a[1,1]-a[0,1]*a[1,0]
    return M2(a[1,1],-a[0,1],-a[1,0],a[0,0])/determinant


@wp.func
def polar_jac(qa:D,qb:D):
    p1=D(P1)-qa;p4=D(P4)-qb;r=fk(qa,qb)
    cb=V2(r[0]-D(.15)*wp.cos(p1),r[1]-D(.15)*wp.sin(p1))
    cd=V2(r[0]-D(.15)-D(.15)*wp.cos(p4),r[1]-D(.15)*wp.sin(p4))
    db=V2(D(.15)*wp.sin(p1),-D(.15)*wp.cos(p1));dd=V2(D(.15)*wp.sin(p4),-D(.15)*wp.cos(p4))
    j=inverse2(M2(cb[0],cb[1],cd[0],cd[1]))*M2(wp.dot(cb,db),D(0),D(0),wp.dot(cd,dd))
    length=r[3];lx=r[0]-D(.075);lz=r[1]
    return wp.transpose(M2(lx/length,lz/length,-lz/(length*length),lx/(length*length))*j)


@wp.func
def legacy_vmc(qa:D,qb:D,force:D,hub:D):
    e=D(1.e-6);r=fk(qa,qb)
    p=fk(qa+e,qb);m=fk(qa-e,qb);u=fk(qa,qb+e);v=fk(qa,qb-e)
    fx=force*(r[0]-D(.075))/r[3];fz=force*r[1]/r[3]
    return V2(((p[0]-m[0])*fx+(p[1]-m[1])*fz+hub*(p[2]-m[2]))/(D(2)*e),
              ((u[0]-v[0])*fx+(u[1]-v[1])*fz+hub*(u[2]-v[2]))/(D(2)*e))


@wp.func
def allowed(speed:D,hip:bool):
    rpm=wp.abs(speed)*D(60)/(D(2)*D(PI));rated=D(490);no_load=D(710);peak=D(4.5)
    if hip:rated=D(175);no_load=D(280);peak=D(40)
    if rpm>rated:peak=peak*wp.max(D(0),(no_load-rpm)/(no_load-rated))
    return peak


@wp.kernel
def control(qpos:wp.array2d[float],qvel:wp.array2d[float],sensor:wp.array2d[float],
            targets:wp.array2d[float],command:wp.array[D],active:wp.array[int],state:wp.array2d[D],
            ids:wp.array[int],heights:wp.array[D],gains:wp.array3d[D],feed:wp.array2d[D],angles:wp.array[D],
            reference:wp.array[D],yaw_cfg:wp.array[D],ctrl:wp.array2d[float],diagnostic:wp.array2d[D]):
    w=wp.tid()
    if active[w]==0:return
    dt=D(.0005);boot=state[w,0]+dt;state[w,0]=boot
    qw=D(qpos[w,3]);qx=D(qpos[w,4]);qy=D(qpos[w,5]);qz=D(qpos[w,6])
    roll=wp.atan2(D(2)*(qw*qx+qy*qz),D(1)-D(2)*(qx*qx+qy*qy))
    pitch=wp.asin(wp.clamp(D(2)*(qw*qy-qz*qx),D(-1),D(1)))
    yaw=wp.atan2(D(2)*(qw*qz+qx*qy),D(1)-D(2)*(qy*qy+qz*qz))
    qa=D(qpos[w,ids[0]]);qb=D(qpos[w,ids[1]]);qc=D(qpos[w,ids[2]]);qd=D(qpos[w,ids[3]])
    va=D(qvel[w,ids[4]]);vb=D(qvel[w,ids[5]]);vc=D(qvel[w,ids[6]]);vd=D(qvel[w,ids[7]])
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
    cmd=D(command[w]);error=wp.atan2(wp.sin(yaw-state[w,9]),wp.cos(yaw-state[w,9]))
    if wp.abs(cmd)>=D(.01):state[w,12]=D(-1)
    elif state[w,12]<D(0):state[w,12]=boot
    if wp.abs(cmd)<D(.01) and wp.abs(state[w,7])<D(.05) and state[w,10]<D(.1):state[w,9]=yaw
    if wp.abs(cmd)<D(.01):state[w,10]=state[w,10]+dt
    else:state[w,10]=D(0)
    if wp.abs(cmd)<D(.01):state[w,11]=D(0)
    elif wp.abs(state[w,7]-cmd)>D(.08):state[w,11]=wp.clamp(state[w,11]+(state[w,7]-cmd)*dt,D(-.3),D(.3))
    ktha=D(-.3)
    if wp.abs(cmd)<D(.01):ktha=D(-.6)
    vmax=D(0)
    if wp.abs(state[w,7])>D(1):vmax=wp.sign(state[w,7])*(wp.abs(state[w,7])-D(1))
    thcmd=wp.clamp((ktha*(cmd-state[w,7])+D(.08)*state[w,11]+D(.5)*pitch)*wp.min(D(1),boot/D(.5))+vmax,D(-.2),D(.2))
    if boot<D(1):thcmd=D(0)
    braking=wp.abs(cmd)<D(.01) and wp.abs(state[w,7])>D(.03)
    if braking:
        thcmd=wp.sign(state[w,7])*wp.min(D(sim.STOP_LEAN_MAX),D(sim.STOP_LEAN_BASE)+D(sim.STOP_LEAN_GAIN)*wp.abs(state[w,7]))*wp.clamp((boot-state[w,12])/D(.1),D(0),D(1))
    kd=D(.5)
    if boot<D(.6):kd=D(1.2)
    hub_old=D(4)*(thcmd-th)-kd*state[w,4]
    if braking:hub_old=-(D(4)*(th-thcmd)+D(.5)*state[w,4]-D(2)*pitch)*D(1.5)-D(4)*pitch-D(.5)*state[w,6]
    offset=wp.clamp(D(.30)*roll+D(.12)*state[w,5],D(-.035),D(.035))
    height=length*wp.cos(th);gravity=D(4)*D(MASS)*wp.min(D(1),boot/D(.15))
    fl=wp.clamp(D(MASS)*(D(500)*(D(.3)+offset-height)-D(25)*state[w,3]*wp.cos(th))+gravity,-D(40)*D(MASS),D(40)*D(MASS))
    fr=wp.clamp(D(MASS)*(D(500)*(D(.3)-offset-height)-D(25)*state[w,3]*wp.cos(th))+gravity,-D(40)*D(MASS),D(40)*D(MASS))
    kp=D(.8)*D(MASS);damping=D(.08)*D(MASS)
    if braking:kp=kp*wp.clamp((D(.3)-wp.abs(state[w,7]))/D(.2),D(0),D(1))
    old_l=legacy_vmc(qa,qb,fl,hub_old)+V2(kp*(reference[0]-qa)-damping*va,kp*(reference[1]-qb)-damping*vb)
    old_r=legacy_vmc(qc,qd,fr,hub_old)+V2(kp*(reference[0]-qc)-damping*vc,kp*(reference[1]-qd)-damping*vd)
    limit=D(.8)*D(MASS)
    if braking:limit=D(2)*D(MASS)
    old_l=V2(wp.clamp(old_l[0],-wp.min(limit,allowed(va,True)),wp.min(limit,allowed(va,True))),wp.clamp(old_l[1],-wp.min(limit,allowed(vb,True)),wp.min(limit,allowed(vb,True))))
    old_r=V2(wp.clamp(old_r[0],-wp.min(limit,allowed(vc,True)),wp.min(limit,allowed(vc,True))),wp.clamp(old_r[1],-wp.min(limit,allowed(vd,True)),wp.min(limit,allowed(vd,True))))
    jl=polar_jac(qa,qb);jr=polar_jac(qc,qd)
    rate_l=jl[0,0]*va+jl[1,0]*vb;rate_r=jr[0,0]*vc+jr[1,0]*vd
    arate_l=jl[0,1]*va+jl[1,1]*vb;arate_r=jr[0,1]*vc+jr[1,1]*vd
    index=int(0)
    if length>heights[1]:index=1
    if length>heights[2]:index=2
    ratio=wp.clamp((length-heights[index])/(heights[index+1]-heights[index]),D(0),D(1))
    theta_eq=(D(1)-ratio)*angles[index]+ratio*angles[index+1]
    if wp.abs(cmd)>D(.01) or wp.abs(vx)>D(.03):state[w,13]=D(0)
    elif state[w,13]==D(0):
        state[w,13]=D(1);state[w,14]=D(qpos[w,0]);state[w,15]=D(qpos[w,1])
    position=D(0)
    if state[w,13]>D(0):position=wp.cos(yaw)*(D(qpos[w,0])-state[w,14])+wp.sin(yaw)*(D(qpos[w,1])-state[w,15])
    pitch_rate=D(sensor[w,gyro+1])
    x=V6(th-theta_eq,(arate_l+arate_r)/D(2)-pitch_rate,position,vx-cmd,pitch,pitch_rate)
    wheel=(D(1)-ratio)*feed[index,0]+ratio*feed[index+1,0]
    hub=(D(1)-ratio)*feed[index,1]+ratio*feed[index+1,1]
    support=(D(1)-ratio)*feed[index,2]+ratio*feed[index+1,2]
    for j in range(6):
        wheel=wheel-((D(1)-ratio)*gains[index,0,j]+ratio*gains[index+1,0,j])*x[j]
        hub=hub-((D(1)-ratio)*gains[index,1,j]+ratio*gains[index+1,1,j])*x[j]
    vl=inverse2(jl)*old_l;vr=inverse2(jr)*old_r;average=(vl[1]+vr[1])/D(2)
    tl=jl*V2(support+D(MASS)*(D(500)*(D(.3)+offset-left[3])-D(25)*rate_l),vl[1]+hub-average)
    tr=jr*V2(support+D(MASS)*(D(500)*(D(.3)-offset-right[3])-D(25)*rate_r),vr[1]+hub-average)
    yaw_torque=D(0)
    if wp.abs(error)>=D(PI)/D(360) or wp.abs(state[w,8])>=D(.05):yaw_torque=wp.clamp(-yaw_cfg[0]*error-yaw_cfg[1]*state[w,8],-yaw_cfg[2]*D(MASS),yaw_cfg[2]*D(MASS))
    base=V6(tl[0],tl[1],tr[0],tr[1],wheel+yaw_torque,wheel-yaw_torque)
    speeds=V6(va,vb,vc,vd,D(qvel[w,ids[8]]),D(qvel[w,ids[9]]))
    invalid_base=int(0);bad_control=bool(False)
    for j in range(6):
        if not wp.isfinite(base[j]):bad_control=True
        maximum=D(4.5)
        if j<4:maximum=D(40)
        base[j]=wp.clamp(base[j],-maximum,maximum)
        bound=allowed(speeds[j],j<4)
        if base[j]<-bound-D(1.e-9) or base[j]>bound+D(1.e-9):invalid_base=1
        base[j]=wp.clamp(base[j],-bound,bound)
    for j in range(targets.shape[1]):state[w,16+j]=state[w,16+j]+wp.clamp(D(targets[w,j])-state[w,16+j],D(-.01),D(.01))
    rr_l=jl*V2(state[w,16]*D(.1)*D(7)*D(9.81)/D(2),state[w,17])
    rr_r=jr*V2(-state[w,16]*D(.1)*D(7)*D(9.81)/D(2),-state[w,17])
    residual=V6(rr_l[0],rr_l[1],rr_r[0],rr_r[1],state[w,18]*yaw_cfg[3],-state[w,18]*yaw_cfg[3])
    if targets.shape[1]==6:
        rr_l=jl*V2(state[w,16]*D(.1)*D(7)*D(9.81)/D(2),state[w,18])
        rr_r=jr*V2(state[w,17]*D(.1)*D(7)*D(9.81)/D(2),state[w,19])
        residual=V6(rr_l[0],rr_l[1],rr_r[0],rr_r[1],state[w,20]*yaw_cfg[3],state[w,21]*yaw_cfg[3])
    diagnostic[w,14]=D(0)
    if bad_control:diagnostic[w,14]=D(2)
    bad_map=bool(False)
    for side in range(2):
        matrix=jl
        if side==1:matrix=jr
        square=matrix[0,0]*matrix[0,0]+matrix[0,1]*matrix[0,1]+matrix[1,0]*matrix[1,0]+matrix[1,1]*matrix[1,1]
        det=wp.abs(matrix[0,0]*matrix[1,1]-matrix[0,1]*matrix[1,0])
        maximum=(square+wp.sqrt(wp.max(D(0),square*square-D(4)*det*det)))/D(2)
        if det==D(0) or maximum/det>D(1.e6):bad_map=True
    lam=D(1)
    nonzero=bool(False)
    for j in range(targets.shape[1]):
        if state[w,16+j]!=D(0):nonzero=True
    if invalid_base==0 and bad_map and nonzero:
        if not bad_control:diagnostic[w,14]=D(1)
        lam=D(0)
    if invalid_base:lam=D(0)
    for j in range(6):
        bound=allowed(speeds[j],j<4)
        if residual[j]>D(0):lam=wp.min(lam,(bound-base[j])/residual[j])
        elif residual[j]<D(0):lam=wp.min(lam,(-bound-base[j])/residual[j])
    lam=wp.clamp(lam,D(0),D(1))
    for j in range(6):
        ctrl[w,j]=float(wp.clamp(base[j]+lam*residual[j],-allowed(speeds[j],j<4),allowed(speeds[j],j<4)))
        diagnostic[w,j]=base[j];diagnostic[w,6+j]=lam*residual[j]
    for j in range(6):
        if not wp.isfinite(ctrl[w,j]) or not wp.isfinite(residual[j]):diagnostic[w,14]=D(2)
    diagnostic[w,12]=lam;diagnostic[w,13]=D(invalid_base)


def constants(model,worlds,yaw_config=(.4,2.,.24,.3),action_dim=3):
    h,t,_,_=nominal_design()
    ids=[int(model.jnt_qposadr[model.joint(n).id]) for n in ('alphaL','betaL','alphaR','betaR')]
    ids += [int(model.jnt_dofadr[model.joint(n).id]) for n in ('alphaL','betaL','alphaR','betaR','wheel1','wheel2')]
    ids += [int(model.sensor('body_gyro').adr[0])]
    state=np.zeros((worlds,16+action_dim));state[:,1]=sim.L_STAND;state[:,12]=-1
    return dict(state=wp.array(state,dtype=D),ids=wp.array(ids,dtype=wp.int32),
        heights=wp.array(h,dtype=D),gains=wp.array(np.stack([r[0] for r in t]),dtype=D),
        feed=wp.array(np.stack([r[1] for r in t]),dtype=D),angles=wp.array([r[2] for r in t],dtype=D),yaw=wp.array(yaw_config,dtype=D),
        reference=wp.array(sim.ik(sim.L_STAND),dtype=D))
