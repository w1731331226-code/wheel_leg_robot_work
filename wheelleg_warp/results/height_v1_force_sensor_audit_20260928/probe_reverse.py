"""Temporary, read-only 2 kHz audit of wheel force sensors against contact truth."""
from dataclasses import asdict
import hashlib,json,sys,argparse
from pathlib import Path
ROOT=Path('/home/wmt/wheel_leg_robot_work')
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
import warp as wp
import mujoco_warp as mjw
from mujoco_warp._src.support import contact_force_fn
from mujoco_warp._src.types import vec5
from native.controller import D,control
from native.environment import NativeEnv,begin,command_step,reduce_contacts,after
from native.terrain import HeightTerrainScenario,bank_height_v3

@wp.kernel
def clear(slot:int,valid:wp.array[int],trace:wp.array3d[D]):
    w=wp.tid()
    for k in range(trace.shape[2]):trace[slot,w,k]=D(0)
    trace[slot,w,0]=D(valid[w])

@wp.kernel
def contact_truth(slot:int,ncon:wp.array[int],world:wp.array[int],geom:wp.array[wp.vec2i],
                  frame:wp.array[wp.mat33],friction:wp.array[vec5],dim:wp.array[int],address:wp.array2d[int],
                  force:wp.array2d[float],njmax:int,cone:int,ids:wp.array[int],kind:wp.array[int],trace:wp.array3d[D]):
    j=wp.tid()
    if j>=ncon[0]:return
    w=world[j]
    if w<0 or w>=trace.shape[1] or trace[slot,w,0]==D(0):return
    a=geom[j][0];b=geom[j][1]
    target=int(0)
    if kind[w]==1:
        if (a==ids[13] and b==ids[11]) or (b==ids[13] and a==ids[11]):target=1
        if (a==ids[14] and b==ids[12]) or (b==ids[14] and a==ids[12]):target=2
    if kind[w]==2:
        terrain=(a>=ids[15] and a<=ids[16]) or (b>=ids[15] and b<=ids[16])
        if terrain and (a==ids[11] or b==ids[11]):target=1
        if terrain and (a==ids[12] or b==ids[12]):target=2
    if target==0:return
    f=contact_force_fn(cone,frame,friction,dim,address,force,njmax,ncon,w,j,False)
    if f[0]>0:wp.atomic_add(trace,slot,w,target+1,D(f[0]))

@wp.kernel
def sample(slot:int,active:wp.array[int],clock:wp.array[float],sensors:wp.array2d[float],
           mat:wp.array2d[wp.mat33f],sid_l:int,sid_r:int,trace:wp.array3d[D]):
    w=wp.tid()
    if active[w]==0:return
    trace[slot,w,1]=D(clock[w])
    l=wp.vec3(sensors[w,22],sensors[w,23],sensors[w,24])
    r=wp.vec3(sensors[w,25],sensors[w,26],sensors[w,27])
    lw=mat[w,sid_l]*l;rw=mat[w,sid_r]*r
    trace[slot,w,4]=D(l[0]);trace[slot,w,5]=D(l[1]);trace[slot,w,6]=D(l[2])
    trace[slot,w,7]=D(r[0]);trace[slot,w,8]=D(r[1]);trace[slot,w,9]=D(r[2])
    trace[slot,w,10]=D(lw[0]);trace[slot,w,11]=D(lw[1]);trace[slot,w,12]=D(lw[2])
    trace[slot,w,13]=D(rw[0]);trace[slot,w,14]=D(rw[1]);trace[slot,w,15]=D(rw[2])


def cases():
    hs=(.16,.20,.25,.30,.35,.38)
    speeds=(.5,1.,-.5,-1.)
    one=[HeightTerrainScenario(speed=v,stand_height_m=h,**{side:.02}) for side in ('height_l','height_r') for v in speeds for h in hs]
    flat=[HeightTerrainScenario(speed=v,stand_height_m=h) for v in speeds for h in hs]
    step=[HeightTerrainScenario(speed=v,stand_height_m=h,terrain='step',step_height_m=.02,relative_attitude=True) for v in speeds for h in hs]
    return one+flat+step, np.array([1]*len(one)+[0]*len(flat)+[2]*len(step),np.int32)


def first_run3(hit,t):
    streak=0
    for i,h in enumerate(hit):
        streak=streak+1 if h else 0
        if streak==3:return float(t[i])
    return None


def analyze(rows,cases,kind):
    out=[]
    for i,(x,s,k) in enumerate(zip(rows,cases,kind)):
        x=np.asarray(x,dtype=float);t=x[:,1];truth=x[:,2:4]
        contact=np.array([t[np.flatnonzero(truth[:,z]>1)[0]] if np.any(truth[:,z]>1) else np.nan for z in range(2)])
        # Fixed threshold, fixed 3 consecutive physics samples. Local x is the actual
        # MJCF sensor channel; world x/world horizontal project by wheel site matrix.
        signal=np.stack([abs(x[:,[4,7]]),abs(x[:,[10,13]]),np.hypot(x[:,[10,13]],x[:,[11,14]])],axis=0)
        first=np.empty((3,2),dtype=object)
        for v in range(3):
            for side in range(2):first[v,side]=first_run3(signal[v,:,side]>=30,t)
        target_side=0 if s.height_l else 1 if s.height_r else None
        r=dict(index=i,kind=['flat','single','symmetric'][k],height=s.stand_height_m,speed=s.speed,
               target_side=['L','R'][target_side] if target_side is not None else None,
               first_contact_s=[None if np.isnan(z) else round(float(z),4) for z in contact],
               first_trigger_s=[[None if z is None else round(z,4) for z in pair] for pair in first],
               max_force_N={
                   'local_abs_x':np.max(signal[0],axis=0).round(2).tolist(),
                   'world_abs_x':np.max(signal[1],axis=0).round(2).tolist(),
                   'world_horizontal':np.max(signal[2],axis=0).round(2).tolist()},
               valid_samples=len(x),end_time_s=round(float(t[-1]),4))
        if target_side is not None:
            ct=contact[target_side]
            pre=t<ct;post=(t>=ct)&(t<ct+.1)
            r['precontact_max_N']={name:np.max(signal[v,pre],axis=0).round(2).tolist() for v,name in enumerate(('local_abs_x','world_abs_x','world_horizontal'))}
            r['post100ms_max_N']={name:np.max(signal[v,post],axis=0).round(2).tolist() for v,name in enumerate(('local_abs_x','world_abs_x','world_horizontal'))}
        out.append(r)
    return out


def summarize(rows):
    out={}
    for v,name in enumerate(('local_abs_x','world_abs_x','world_horizontal')):
        groups={}
        for kind in ('single','flat','symmetric'):
            items=[r for r in rows if r['kind']==kind]
            hits=[];lat=[];early=0;wrong=0;both=0;none=0
            for r in items:
                ts=r['first_trigger_s'][v]
                if kind=='single':
                    side=0 if r['target_side']=='L' else 1;ct=r['first_contact_s'][side]
                    right=ts[side];other=ts[1-side]
                    if right is None:none+=1
                    else:
                        lat.append(round((right-ct)*1000,2))
                        if right<ct:early+=1
                    if other is not None and (right is None or other<=right):wrong+=1
                    hits.append(right is not None and right>=ct and (other is None or right<other))
                else:
                    hits.append(any(z is not None for z in ts))
                    if all(z is not None for z in ts):both+=1
                    if sum(z is not None for z in ts)==1:wrong+=1
            groups[kind]=dict(n=len(items),correct_first_side_after_contact=sum(hits) if kind=='single' else None,
                any_trigger=sum(hits) if kind!='single' else None,
                correct_side_no_trigger=none if kind=='single' else None,
                precontact_correct_side=early if kind=='single' else None,
                wrong_side_first_or_equal=wrong if kind=='single' else None,
                both_sides_trigger=both if kind!='single' else None,
                only_one_side_trigger=wrong if kind!='single' else None,
                latency_ms_min_med_max=[min(lat),round(float(np.median(lat)),2),max(lat)] if lat else None)
        out[name]=groups
    return out


def main(output,small,reverse):
    scenarios,kind=cases()
    if small:scenarios=scenarios[:3]+scenarios[48:51]+scenarios[72:75];kind=np.r_[kind[:3],kind[48:51],kind[72:75]]
    if reverse:scenarios=scenarios[::-1];kind=kind[::-1]
    n=len(scenarios);env=NativeEnv(n=n,scenario=scenarios,bank_factory=bank_height_v3,height_conditioned=True,residual_scale=0)
    env.reset();trace=wp.zeros((40,n,16),dtype=D);k=wp.array(kind,dtype=wp.int32)
    sid_l=env.cpu.site('wheel_contact_L').id;sid_r=env.cpu.site('wheel_contact_R').id
    assert [int(env.cpu.sensor(name).adr[0]) for name in ('wheel1_ground_force','wheel2_ground_force')]==[22,25]
    data=env.data
    with wp.ScopedCapture() as capture:
        wp.launch(begin,n,[env.reward])
        for slot in range(40):
            wp.launch(clear,n,[slot,env.active,trace])
            wp.launch(command_step,n,[env.state,env.param,env.command,env.active,data.qpos,data.qvel,data.qacc_warmstart,
                env.stopped_q,env.stopped_v,env.stopped_w,env.contact_flags])
            wp.launch(control,n,[data.qpos,data.qvel,data.sensordata,env.targets,env.command,env.active,env.k['state'],
                env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],env.k['yaw'],
                data.ctrl,env.diag,int(env.project_clipped_base),int(env.grouped_residual)],block_dim=32)
            mjw.step(env.model,data)
            c=data.contact
            wp.launch(contact_truth,data.naconmax,[slot,data.nacon,c.worldid,c.geom,c.frame,c.friction,c.dim,c.efc_address,
                data.efc.force,data.njmax,env.model.opt.cone,env.ids,k,trace])
            wp.launch(sample,n,[slot,env.active,data.time,data.sensordata,data.site_xmat,sid_l,sid_r,trace])
            wp.launch(reduce_contacts,data.naconmax,[data.nacon,c.worldid,c.geom,env.ids,env.contact_flags])
            wp.launch(after,n,[data.qpos,data.qvel,data.sensordata,data.qacc_warmstart,data.time,env.contact_flags,env.ids,
                env.param,env.command,env.state,env.k['state'],env.diag,env.residual,env.active,env.done,env.reward,env.obs,
                env.history,env.stopped_q,env.stopped_v,env.stopped_w,env.wheel_offsets],block_dim=32)
    env.graph=capture.graph
    zero=np.zeros((n,3),np.float32);rows=[[] for _ in scenarios];finished=np.zeros(n,bool)
    for step in range(700):
        env.step_async(zero)
        chunk=trace.numpy();done=env.done.numpy()
        for i in range(n):
            if not finished[i]:rows[i].append(chunk[chunk[:,i,0]>0,i,:])
        _,_,_,infos=env.step_wait()
        finished|=done!=0
        if step%50==0:print('step',step,'finished',int(finished.sum()),'/',n,flush=True)
        if finished.all():break
    assert finished.all(),('not all complete',np.flatnonzero(~finished))
    rows=[np.concatenate(parts,axis=0) for parts in rows]
    summary_rows=analyze(rows,scenarios,kind)
    payload=dict(n=n,protocol='sensor force 30 N sustained 3 consecutive 2 kHz samples; zero residual; target collision = wheel-target contact normal >1 N',
       event_end_timestamp=True,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
       [ROOT/'wheelleg_ppo/xml/wheelleg.xml',ROOT/'wheelleg_warp/native/environment.py',ROOT/'wheelleg_warp/native/models.py',ROOT/'wheelleg_warp/native/terrain.py']},
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summarize(summary_rows),rows=summary_rows)
    output.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print('SUMMARY',json.dumps(payload['summary'],ensure_ascii=False),flush=True)
    print('OUTPUT',output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--small',action='store_true');p.add_argument('--reverse',action='store_true');args=p.parse_args()
    main(args.output,args.small,args.reverse)
