"""Independent first-episode dense receipt. No controller, simulation or NN calls."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import review_cartesian_light_map as frozen

ROOT=frozen.ROOT
OUT=frozen.BASE/'cartesian_classical_candidate_v1'
RECEIVER_TYPE='independent_cartesian_classical_raw_v1'
SOURCE='wheelleg_warp/review_cartesian_classical_raw.py'
SCOPES=('original_physical_design','original_contact_task','actual_cart_command_history_reset','original_B1_parking')
JOINTS=('alphaL','betaL','passA_L','passC_L','alphaR','betaR','passA_R','passC_R')
MOTORS=('alphaL','betaL','alphaR','betaR','wheel1','wheel2')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError('Nonfinite JSON '+s)))


def file_under(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Artifact outside admitted runtime')
    return path


def arrays(path):
    with np.load(path,allow_pickle=False) as z:
        assert len(z.files)==len(set(z.files)),'Duplicate NPZ array'
        return {n:z[n].copy() for n in z.files}


def finite(a,shape=None):
    a=np.asarray(a)
    assert a.dtype.kind in 'biuf' and np.isfinite(a).all()
    if shape is not None:assert a.shape==shape
    return a


def classic(raw,gamma,yaw):
    """Independent registered378 arithmetic, from delayed public input only."""
    x=finite(raw);assert x.ndim==2 and x.shape[1]==481 and len(x)>0
    assert type(gamma) in (int,float) and gamma in (0,.5,1,2)
    assert yaw['arm'] in (0,1) and type(yaw['arm']) is int
    assert (yaw['kp'],yaw['kd'],yaw['roll_gain'])==(.4,.3,0)
    p=x[:,351:390].astype(float);d=np.clip(.30*p[:,0]+.12*p[:,3],-.035,.035)
    legs=frozen.geometry(p[:,12:16].reshape(-1,2,2))[0]
    delta=gamma*3.4335*d/.035
    force=delta[:,None]*(legs/np.linalg.norm(legs,axis=-1)[...,None]).sum(axis=1)/2
    shrink=1/np.maximum(1,abs(force/3.4335).max(axis=1))
    kp=.4+yaw['kp'];kd=2+yaw['kd'];ref=np.zeros(len(p));moving=abs(p[:,9])>.05
    ref[moving]=-yaw['arm']*np.sign(p[moving,9])*np.arctan2(p[moving,38],abs(p[moving,9])*kd/kp)
    ref=np.clip(ref,-np.pi/60,np.pi/60)
    aw=np.clip((-yaw['kp']*p[:,2]-yaw['kd']*p[:,5]+kp*ref)/.3,-1,1).astype(np.float32)
    latent=np.column_stack((force*shrink[:,None]/3.4335,aw)).astype(np.float32)
    return latent,dict(d=d,DeltaF=delta,geometry=legs,force_raw=force,shrink=shrink,latent3=latent,
        force_sagittal_N=3.4335*latent[:,:2].astype(float))


def world_array(model,name,world,shape):
    a=finite(model['model.'+name])
    if a.shape==shape:return a
    assert a.ndim==len(shape)+1 and a.shape[1:]==shape and a.shape[0]>world
    return a[world]


def model_view(model,meta,world,xml):
    assert meta['model_schema']=='classical_actual_model_v1'
    assert set(model)==set(meta['array_header'])
    for n,a in model.items():
        finite(a);h=meta['array_header'][n]
        assert list(a.shape)==h['shape'] and str(a.dtype)==h['dtype']
        assert hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()==h['sha256']
    t=meta['topology'];d=meta['dimensions'];assert (d['nq'],d['nv'],d['nu'])==(17,16,6)
    tree=ET.parse(xml).getroot();bodies=['world'];parents=[0];joints=[];owners=[];qadr=[];vadr=[];q=v=0;sites=[];siteowners=[]
    def visit(body,parent):
        nonlocal q,v
        i=len(bodies);bodies.append(body.attrib['name']);parents.append(parent)
        for child in body:
            if child.tag in ('joint','freejoint'):
                joints.append(child.get('name'));owners.append(i);qadr.append(q);vadr.append(v)
                free=child.tag=='freejoint';q+=7 if free else 1;v+=6 if free else 1
            if child.tag=='site':sites.append(child.attrib['name']);siteowners.append(i)
        for child in body.findall('body'):visit(child,i)
    for body in tree.find('worldbody').findall('body'):visit(body,0)
    normalize=lambda names:[None if n in (None,'') else n for n in names]
    assert t['body_names']==bodies and t['body_parentid']==parents and normalize(t['joint_names'])==joints
    t={**t,'joint_names':normalize(t['joint_names'])}
    assert t['site_names']==sites and t['site_bodyid']==siteowners
    assert t['jnt_qposadr']==qadr and t['jnt_dofadr']==vadr and t['jnt_bodyid']==owners
    assert t['jnt_type']==[0]+[3]*(len(joints)-1)
    assert q==17 and v==16 and d['njnt']==len(joints) and d['nbody']==len(bodies)
    core={}
    for name,size in (('jnt_qposadr',d['njnt']),('jnt_dofadr',d['njnt']),('jnt_bodyid',d['njnt']),('jnt_type',d['njnt']),('body_parentid',d['nbody']),('body_rootid',d['nbody']),('site_bodyid',d['nsite']),('geom_bodyid',d['ngeom'])):
        original=world_array(model,name,world,(size,));assert original.dtype.kind in 'iu';core[name]=original.astype(int)
        np.testing.assert_array_equal(core[name],t[name])
    core.update(jnt_range=world_array(model,'jnt_range',world,(d['njnt'],2)),body_pos=world_array(model,'body_pos',world,(d['nbody'],3)),site_pos=world_array(model,'site_pos',world,(d['nsite'],3)))
    active=[joints.index(n) for n in MOTORS[:4]];motor=[joints.index(n) for n in MOTORS];eight=[joints.index(n) for n in JOINTS]
    ids=np.asarray(meta['ids']);assert ids.ndim==1 and len(ids)>=19 and np.issubdtype(ids.dtype,np.integer)
    np.testing.assert_array_equal(ids[:4],core['jnt_qposadr'][active]);np.testing.assert_array_equal(ids[4:10],core['jnt_dofadr'][motor])
    p=meta['physical_ids'];assert p['active_qpos_ids']==ids[:4].tolist() and p['motor_qvel_ids']==ids[4:10].tolist()
    assert p['eight_joint_ids']==eight and p['eight_qpos_ids']==core['jnt_qposadr'][eight].tolist()
    assert t['geom_names'][ids[11]]=='wheel_collide_L' and t['geom_names'][ids[12]]=='wheel_collide_R'
    assert t['geom_names'][ids[13]]=='bump_L' and t['geom_names'][ids[14]]=='bump_R'
    assert core['geom_bodyid'][ids[11]]==bodies.index('wheelL') and core['geom_bodyid'][ids[12]]==bodies.index('wheelR')
    assert core['geom_bodyid'][ids[13]]==core['geom_bodyid'][ids[14]]==0
    terrain=[t['geom_names'].index(n) if n in t['geom_names'] else -1 for n in ('terrain_00','terrain_15')]
    np.testing.assert_array_equal(ids[15:17],terrain)
    np.testing.assert_array_equal(ids[17:19],core['jnt_qposadr'][[joints.index('passA_L'),joints.index('passA_R')]])
    assert t['actuator_names']==[e.attrib['name'] for e in tree.find('actuator')]
    np.testing.assert_array_equal(world_array(model,'actuator_trnid',world,(6,2)),t['actuator_trnid'])
    np.testing.assert_array_equal(np.asarray(t['actuator_trnid'])[:,0],motor)
    for s in ('L','R'):
        assert p['chain_body_ids'][s]==[bodies.index(n) for n in ('leg'+s,'kneeA_'+s,'wheel'+s,'leg'+s+'_D','kneeB_'+s)]
        assert p['chain_site_ids'][s]==sites.index('couplerB_'+s+'_end')
    view=SimpleNamespace(**core,joint=lambda n:SimpleNamespace(id=joints.index(n)),body=lambda n:SimpleNamespace(id=bodies.index(n)),site=lambda n:SimpleNamespace(id=sites.index(n)))
    return view,ids


def physical(full,row,view,ids):
    # These366 pure coordinate/RPM functions consume recorded arrays, never a simulator.
    from review_fixed_force_evaluation import chain_minima,physical_metrics
    n=row['physical_steps'];t=finite(full['trace'],(n,112));pre=finite(full['pre'],(n,39));post=finite(full['post'],(n,39))
    np.testing.assert_array_equal(pre[1:,:33],post[:-1,:33]);np.testing.assert_array_equal(t[:,0],1)
    np.testing.assert_array_equal(t[:,1],np.arange(1,n+1))
    np.testing.assert_allclose(t[:,2:4],np.column_stack((np.arange(n),np.arange(1,n+1)))*.0005,rtol=0,atol=1e-9)
    lengths,loop=chain_minima(post[:,:17],view);metrics=physical_metrics(pre,post,view)
    np.testing.assert_allclose([*lengths,loop,*metrics],[row['min_actual_A_leg_m'],row['min_actual_B_leg_m'],row['max_loop_error_m'],row['min_eight_joint_margin_rad'],row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm']],rtol=0,atol=1e-12)
    margin=1.4-abs(post[:,ids[:4]]).max(axis=1);np.testing.assert_array_equal(margin,t[:,24]);assert margin.min()==row['min_active_design_margin_rad']
    assert row['physical_evidence_steps']==n and row['duration_s']==n*.0005
    passed=min(lengths)>=row['geometric_limit_m'] and metrics[0]>=0 and max(metrics[1:])<=1e-6
    assert row['geometric_limit_m']==.1147044660616607 and passed is True and row['physical_safety_passed'] is True
    assert margin.min()>=0 and row['design_joint_passed'] is True
    return pre,post,t


def contact_task(full,row,param,ids,view,world):
    n=row['physical_steps'];t=full['trace'];post=full['post'];pre=full['pre'];meta=finite(full['meta'],(n,6));contacts=finite(full['contacts']);steps=np.asarray(full['contact_steps'])
    assert contacts.ndim==2 and contacts.shape[1]==29 and np.issubdtype(steps.dtype,np.integer) and len(contacts)==len(steps)
    assert not meta[:,2].any() and np.all((steps>=1)&(steps<=n));np.testing.assert_array_equal(np.bincount(steps,minlength=n+1)[1:],meta[:,1])
    np.testing.assert_array_equal(contacts[:,0],1);np.testing.assert_array_equal(contacts[:,1],world)
    geoms=contacts[:,2:4].astype(int);np.testing.assert_array_equal(geoms,contacts[:,2:4]);assert np.all((geoms>=0)&(geoms<len(view.geom_bodyid)))
    assert np.isin(contacts[:,5],[1,3,4,6]).all() and (contacts[:,18:23]>=0).all()
    assert (meta[:,:2]>=0).all() and np.array_equal(meta[:,:2],meta[:,:2].astype(int))
    touch=np.zeros(n,int);nonwheel=np.zeros(n,bool)
    for i,(a,b) in enumerate(geoms):
        k=steps[i]-1;nonwheel[k]|=not any(g in ids[11:13] for g in (a,b))
        touch[k]|=(1 if ids[13] in (a,b) else 0)|(2 if ids[14] in (a,b) else 0)
        terrain=any(ids[15]<=g<=ids[16] for g in (a,b))
        if terrain:touch[k]|=(4 if ids[11] in (a,b) else 0)|(8 if ids[12] in (a,b) else 0)
    # Verify uncensored per-wheel raw-contact counts and normal/vertical load sums.
    for side,wheel in enumerate(ids[11:13]):
        expected=np.zeros((n,6))
        for i,(a,b) in enumerate(geoms):
            if wheel not in (a,b):continue
            k=steps[i]-1;other=b if a==wheel else a;load=max(0.,contacts[i,23]);vertical=max(0.,(-1 if a==wheel else 1)*load*contacts[i,11])
            target=other in ids[13:15] or ids[15]<=other<=ids[16]
            expected[k,[0,2,4]]+=[1,load,vertical]
            if target:expected[k,[1,3,5]]+=[1,load,vertical]
        np.testing.assert_allclose(t[:,60+26*side:66+26*side],expected,rtol=0,atol=1e-6)
    total=int(np.bitwise_or.reduce(touch,initial=0));assert row['touched_contact_mask']==(total&3) and row['touched_terrain_contact_mask']==(total&12)
    assert row['required_contact_mask']==int(param[5]) and row['required_terrain_contact_mask']==int(param[6])
    angles=t[:,7:10];q=post[:,:17];v=post[:,17:33];qw,qx,qy,qz=q[:,3:7].T
    euler=np.column_stack((np.arctan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)),np.arcsin(np.clip(2*(qw*qy-qz*qx),-1,1)),np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))))
    np.testing.assert_allclose(angles,euler,rtol=0,atol=1e-12)
    reference=np.zeros((n,3));relative=param[1]*q[:,0]-param[10];kind=int(param[8]);a=param[9]
    if param[7]:
        if kind==1:reference[:,1]=np.where((relative>=-.65)&(relative<-.15),-param[1]*a,np.where((relative>.15)&(relative<=.65),param[1]*a,0))
        if kind in (2,4):reference[abs(relative)<=.65,0]=a
        if kind==3:
            mask=(relative>=-.64)&(relative<=.64);segment=np.minimum(3,((relative[mask]+.64)/.32).astype(int));reference[mask,1]=np.where(np.isin(segment,[1,3]),param[1]*a,-param[1]*a)
    peak=abs(angles).max(axis=0)*180/np.pi;relpeak=abs(angles-reference).max(axis=0)*180/np.pi;rms=np.sqrt(np.mean(angles*angles,axis=0))*180/np.pi
    for name,values in (('peak_deg',peak),('relative_peak_deg',relpeak),('rms_deg',rms)):np.testing.assert_allclose(row[name],values,rtol=0,atol=1e-10)
    # Actual wheel centres, using the saved passive chain and quaternion transform.
    wheels=[]
    for side,s in enumerate(('L','R')):
        idx=[view.body(n).id for n in ('leg'+s,'kneeA_'+s,'wheel'+s)]
        def rotate(point,angle):
            c=np.cos(angle);sn=np.sin(angle);return np.column_stack((c*point[0]+sn*point[2],np.full(n,point[1]),-sn*point[0]+c*point[2]))
        local=view.body_pos[idx[0]]+rotate(view.body_pos[idx[1]],q[:,ids[2*side]])+rotate(view.body_pos[idx[2]],q[:,ids[2*side]]+q[:,ids[17+side]])
        cross=2*np.cross(q[:,4:7],local);wheels.append(q[:,:3]+qw[:,None]*cross+np.cross(q[:,4:7],cross)+local)
    wheels=np.stack(wheels,axis=1);np.testing.assert_allclose(t[:,16:22],wheels.reshape(n,6),rtol=0,atol=1e-7)
    progress=param[1]*wheels[-1,:,0]-param[10];np.testing.assert_allclose(row['wheel_progress_m'],progress,rtol=0,atol=1e-7)
    exited=(int(param[6])==0)|((param[1]*wheels[:,:,0]-param[10]).min(axis=1)>=param[12])
    at=np.flatnonzero((param[1]*q[:,0]>=param[2])&exited);arrival=float((at[0]+1)*.0005) if len(at) else None
    assert row['arrival_s']==arrival
    command=param[0]*np.clip(np.arange(n)*.0005-1,0,1)
    if len(at):command[at[0]+1:]=0
    np.testing.assert_allclose(t[:,43],command,rtol=0,atol=1e-12)
    vx=np.cos(angles[:,2])*v[:,0]+np.sin(angles[:,2])*v[:,1];np.testing.assert_allclose(t[:,44],vx,rtol=0,atol=1e-12)
    moving=t[:,43]!=0;vr=float(np.sqrt(np.mean((vx[moving]-t[moving,43])**2))) if moving.any() else None
    if vr is None:assert row['velocity_rmse'] is None
    else:np.testing.assert_allclose(row['velocity_rmse'],vr,rtol=0,atol=1e-12)
    fk=frozen.geometry(q[:,ids[:4]].reshape(n,2,2))[0];height=np.linalg.norm(fk,axis=-1).mean(axis=1)
    heightmask=t[:,3]>=1;hr=float(np.sqrt(np.mean((height[heightmask]-param[13])**2))) if heightmask.any() else None
    if hr is None:assert row['height_rmse_m'] is None
    else:np.testing.assert_allclose(row['height_rmse_m'],hr,rtol=0,atol=1e-12)
    np.testing.assert_allclose(row['final_mean_fk_leg_m'],height[-1],rtol=0,atol=1e-7);assert row['target_leg_m']==param[13]
    distance=tail=0.
    if len(at):
        distance=float(np.linalg.norm(q[at[0]:,:2]-q[at[0],:2],axis=1).max());tailmask=t[:,3]-arrival>=1.5
        if tailmask.any():tail=float(np.linalg.norm(v[tailmask,:2],axis=1).max())
    np.testing.assert_allclose([row['stop_distance_m'],row['tail_speed_m_s']],[distance,tail],rtol=0,atol=1e-12)
    reason='nonwheel_contact' if nonwheel.any() else ('completed' if arrival is not None and t[-1,3]-arrival>=2-1e-10 else 'timeout')
    fallen=(abs(angles[:,:2]).max(axis=1)>.6981317007977318)|(q[:,2]<.02)
    bad=np.flatnonzero(fallen|nonwheel)
    if len(bad):reason='fall' if fallen[bad[0]] else 'nonwheel_contact'
    assert row['reason']==reason and (reason!='timeout' or t[-1,3]>=param[3]-1e-10)
    terrain=reason=='completed' and (total&int(param[6]))==int(param[6]);assert row['terrain_passed']==terrain and row['terrain_exit_passed']==bool(exited[-1])
    success=(row['physical_safety_passed'] and row['design_joint_passed'] and terrain and bool(exited[-1]) and (total&int(param[5]))==int(param[5]) and max(relpeak if param[7] else peak)<=5 and peak[2]<=5 and (not param[7] or max(peak[:2])<=10) and vr is not None and vr<=.2*abs(param[0]) and distance<=.6 and tail<=.03 and hr is not None and hr<=.02 and abs(height[-1]-param[13])<=.02)
    assert row['success'] is bool(success)
    return len(contacts)


def parameter_binding(param,case,descriptor,model,world):
    s=case['scenario'];registered=descriptor['cases'][world]
    assert registered['seed']==case['seed'] and param[2]==registered['task_goal_progress_m_archived'] and param[3]==registered['param3_s_arithmetic']
    assert descriptor['solver_iterations']==s['solver_iterations']
    assert int(np.ceil((param[3]+2)/.02))+2==registered['deadline_packets']
    expected_goal=s['center']+abs(s['offset'])/2+.75;end=0.
    if s.get('terrain','legacy')!='legacy':
        # Actual static model poses/sizes recorded beforelaunch, not a new forward call.
        names=descriptor.get('_geom_names')
        assert names is not None
        ids=[names.index('terrain_00'),names.index('terrain_15')]
        position=model['geom_xpos'][world,ids[0]:ids[1]+1];matrix=model['geom_xmat'][world,ids[0]:ids[1]+1]
        size=model['model.geom_size'];size=size[world if len(size)>1 else 0,ids[0]:ids[1]+1]
        present=position[:,2]>-1.;assert present.any();radii=np.sum(abs(matrix[:,0,:])*size,axis=-1)
        end=float(np.max(np.sign(s['speed'])*position[present,0]+radii[present])-s['center'])
        expected_goal=max(expected_goal,s['center']+end+.15)
    assert param[12]==end and param[2]==expected_goal


def cart_case(actor,c,g,park,full,reference,mode,gamma,yaw,diagnostic):
    n=len(c);packets=(n+39)//40;finite(actor,(packets,968));finite(c,(n,38));finite(g,(n,62));finite(park,(n,30))
    np.testing.assert_array_equal(actor[:,:481],actor[:,481:962]);latent,expected=classic(actor[:,:481],gamma,yaw)
    np.testing.assert_allclose(actor[:,962:],frozen.pairs(latent),rtol=0,atol=1e-7)
    assert set(diagnostic)==set(expected)
    for name,value in expected.items():np.testing.assert_allclose(finite(diagnostic[name]),value,rtol=0,atol=1e-7 if name in ('geometry','latent3','force_sagittal_N') else 1e-12)
    np.testing.assert_array_equal(c[:,0],1);np.testing.assert_array_equal(c[:,33],0);np.testing.assert_array_equal(park[:,0],1)
    np.testing.assert_array_equal(park[:,3],np.maximum.accumulate(park[:,2]!=0));np.testing.assert_array_equal(park[:,4],0)
    np.testing.assert_array_equal(park[:,2],full['trace'][:,43])
    np.testing.assert_array_equal(park[:,5:11],actor[np.arange(n)//40,962:].astype(float));np.testing.assert_array_equal(park[:,11:17],park[:,5:11])
    np.testing.assert_array_equal(c[0,1:4],0);np.testing.assert_array_equal(c[1:,1:4],c[:-1,4:7])
    np.testing.assert_array_equal(c[:,4:7],c[:,1:4]+np.clip(park[:,11:17:2]-c[:,1:4],-.01,.01))
    np.testing.assert_array_equal(park[:,17:23],frozen.pairs(c[:,1:4]));np.testing.assert_array_equal(park[:,23:29],frozen.pairs(c[:,4:7]));np.testing.assert_array_equal(park[:,29],0)
    maxima={};frozen.mapping(c,g,np.broadcast_to(reference,(n,4)),mode,maxima)
    speed=full['pre'][:,17+np.asarray(full['ids'][4:10])];np.testing.assert_allclose(c[:,27:33],frozen.limits(speed),rtol=0,atol=1e-12)
    np.testing.assert_array_equal(g[:,20:26],full['pre'][:,33:39]);np.testing.assert_array_equal(full['trace'][:,47:53],c[:,7:13])
    np.testing.assert_allclose(full['trace'][:,53],g[:,26]*g[:,27],rtol=0,atol=1e-12)
    assert np.isin(g[:,28:30],[0,1]).all() and np.all((g[:,27]>=0)&(g[:,27]<=1))
    return maxima


def review(runtime_path,*,out=OUT,completion_path=None,selection_path=None):
    runtime=Path(runtime_path).resolve();out=Path(out).resolve();complete=Path(completion_path) if completion_path else runtime/'completion.json'
    # Refuse absent completion before opening any case/future data.
    done=read(complete);assert done['verified'] is True and done['episodes']==584 and len(done['records'])==68
    p=read(out/'proposal.json');assert done['proposal_sha256']==sha(out/'proposal.json')
    admission=Path(done['source_admission_path']).resolve();assert admission==(out/'source_admission.json').resolve() and sha(admission)==done['source_admission_sha256']
    a=read(admission)
    from review_cartesian_classical_baseline import verify_admission_metadata,verify_execution_budget
    verify_admission_metadata(out,a,runtime);verify_execution_budget(out,p,done,True)
    needed=(SOURCE,'wheelleg_warp/cartesian_classical_history_receipt.py','wheelleg_warp/review_cartesian_light_map.py','wheelleg_warp/review_fixed_force_evaluation.py','wheelleg_ppo/xml/wheelleg.xml')
    assert all(n in a['source_sha256'] for n in needed)
    assert all(sha(ROOT/n)==h for n,h in a['source_sha256'].items()) and a['source_sha256'][SOURCE]==sha(__file__)
    assert np.__version__==a['runtime']['versions']['numpy']
    for mapping in (p['source_sha256'],p['input_sha256']):assert all(sha(ROOT/n)==h for n,h in mapping.items())
    assert Path(done['runtime_path']).resolve()==runtime and done['new_learning_samples']==done['new_model_forward_rows']==done['new_optimizer_updates']==0
    selection=Path(selection_path) if selection_path else runtime/'selection_review.json';chosen=read(selection);assert sha(selection)==done['selection_review_sha256'] and chosen['proposal_sha256']==sha(out/'proposal.json')
    budget=read(out/'job_budget_review.json')
    expected=[(c,j,'selection') for c in p['conditions'] for j in p['selection_jobs']]+[(next(c for c in p['conditions'] if c['label']==s['label']),j,'evaluation') for s in chosen['selected'] for j in p['eval_jobs']]
    assert len(expected)==68;reports=[];steps=contacts=episodes=0
    from cartesian_classical_history_receipt import review_case
    for record,(condition,job,phase) in zip(done['records'],expected):
        assert (record['condition'],record['chart'],record['gamma'],record['panel'],record['batch'],record['phase'])==(condition['label'],condition['chart'],condition['gamma'],job['panel'],job['batch'],phase)
        assert record['cases']==job['cases'] and record['indices']==job['indices']
        if phase=='evaluation':assert record['selection_review_sha256']==sha(selection)
        registered=record['raw_artifacts'];assert registered and all(sha(file_under(runtime,n))==h for n,h in registered.items())
        result_path=file_under(runtime,record['result_path']);assert sha(result_path)==record['result_sha256'];directory=result_path.parent
        def path(name):
            f=file_under(runtime,str((directory/name).relative_to(runtime)));assert str(f.relative_to(runtime)) in registered;return f
        result=read(result_path);rows=result['runs'];assert len(rows)==len(job['cases'])
        model=arrays(path('actual_model_before_launch.npz'));meta=read(path('actual_model_before_launch.json'));initial=arrays(path('initial.npz'))
        descriptor=next(j for j in budget['jobs'] if (j['phase'],j['panel'],j['batch'])==(phase,job['panel'],job['batch']))
        assert meta['descriptor']==descriptor
        contract=read(path('classical_delivery_contract.json'));assert contract['compiled_parking_prepare_enabled']==0 and contract['accepted_nominal_mode']==1 and contract['chart_mode']==(condition['chart']=='CartReset3')
        assert contract['captured_counts']==dict(parking=80,candidate=80,Nom=80) and contract['Nom_target_separate_zero'] is True and contract['Nom_candidate_guard_order'] is True
        assert contract['delivery_schema']=='classical_firstepisode_delivery_v1' and meta['contact_capacity']==64*len(job['cases'])
        assert meta['solver_iterations']==dict(cpu=meta['descriptor']['solver_iterations'],model=meta['descriptor']['solver_iterations'])
        topology=read(path('cartesian_topology.json'));assert topology['accepted_nominal_mode']==1 and topology['candidate_kernel_sha256']==sha(ROOT/'wheelleg_warp/cartesian_pair_kernel_v2.py')
        assert read(path('parking_contract.json'))['condition']=='original'
        np.testing.assert_array_equal(model['q0'],initial['q']);np.testing.assert_array_equal(model['param'],initial['param'])
        details=[]
        for world,(case,row) in enumerate(zip(job['cases'],rows)):
            assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and type(row['physical_steps']) is int and row['physical_steps']>0
            stream={}
            for field in ('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','joint_guard_trace','actor_trace','cartesian_trace','cartesian_terminal'):
                ref=row[field];f=path(ref['path']);assert sha(f)==ref['sha256'];stream[field]=arrays(f)
            full=stream['complete_trace'];view,ids=model_view(model,meta,world,ROOT/'wheelleg_ppo/xml/wheelleg.xml');full['ids']=ids
            param=initial['param'][world];assert param[0]==case['scenario']['speed'] and param[4]==round(case['scenario']['delay_ms']*2) and 0<=param[4]<=20
            scenario=case['scenario'];kind={'ramp':1,'cross_slope':2,'rolling_slope':3,'split_level':4}.get(scenario.get('terrain','legacy'),0)
            required=(1 if scenario['height_l'] else 0)|(2 if scenario['height_r'] else 0);terrain_mask=12 if scenario.get('terrain','legacy')!='legacy' else 0
            if scenario.get('terrain')=='single_side_ramp':terrain_mask=8 if scenario['grade_deg']>0 else 4
            np.testing.assert_array_equal(param[[1,5,6,7,8,10,11,13,14,15,16]],[np.sign(param[0]),required,terrain_mask,bool(scenario.get('relative_attitude',False)),kind,scenario['center'],0,scenario['stand_height_m'],1,.1147044660616607,6])
            assert param[9]==np.deg2rad(scenario.get('grade_deg',0))
            parameter_binding(param,case,{**descriptor,'_geom_names':meta['topology']['geom_names']},model,world)
            assert param[3]==1.5+1.5*param[2]/abs(param[0]) and row['task_goal_progress_m']==param[2]
            pre,post,t=physical(full,row,view,ids);contact_count=contact_task(full,row,param,ids,view,world)
            assert (full['meta'][:,0]<=meta['contact_capacity']).all()
            for field,width in (('gyro_trace',8),('role_trace',11),('phase_trace',6)):
                log=finite(stream[field]['trace'],(len(t),width));np.testing.assert_array_equal(log[:,:2],t[:,:2])
            np.testing.assert_array_equal(stream['phase_trace']['trace'][:,2],t[:,43])
            c=stream['cartesian_trace'];g=stream['joint_guard_trace']['trace'];park=stream['parking_trace']['trace'];actor=stream['actor_trace']['trace']
            np.testing.assert_array_equal(c['clock'],t[:,1:4]);np.testing.assert_array_equal(g[:,50:54],pre[:,ids[:4]]);np.testing.assert_array_equal(g[:,54:58],pre[:,17+ids[4:8]])
            reference=initial['cart_reference'][world];maxima=cart_case(actor,c['trace'],g,park,full,reference,int(condition['chart']=='CartReset3'),condition['gamma'],p['parameters']['yaw_config'],arrays(path(f'classical_request_{world}.npz')))
            cart={**c,'terminal':stream['cartesian_terminal']}
            history=review_case(case=case,row=row,world=world,initial=initial,actor=actor,full=full,cart=cart,parking=park,guard=g,
                actual_terminal=arrays(path(f'actual_terminal_history_{world}.npz'))['actual_raw481'],after_reset=arrays(path(f'actual_after_reset_{world}.npz')),ids=ids,delivery=arrays(path(f'delivery_{world}.npz')))
            assert history['verified'] is True
            details.append(dict(seed=case['seed'],physical_steps=len(t),contacts=contact_count,max_abs_errors=maxima,history=history));steps+=len(t);contacts+=contact_count;episodes+=1
        assert record['first_episode_world_steps']==sum(r['physical_steps'] for r in rows)
        reports.append(dict(condition=record['condition'],panel=record['panel'],batch=record['batch'],independent_raw_received=True,raw_artifacts=registered,details=details))
    assert episodes==584 and steps==done['first_episode_world_steps']
    return dict(verified=True,receiver_type=RECEIVER_TYPE,receiver_source=SOURCE,receiver_sha256=sha(__file__),independent_raw_received=True,
        episodes=episodes,records=reports,first_episode_world_steps=steps,contacts_checked=contacts,scope_checks={n:True for n in SCOPES},
        proposal_sha256=sha(out/'proposal.json'),completion_sha256=sha(complete),selection_review_sha256=sha(selection),source_admission_sha256=sha(admission),
        tolerances=frozen.TOL,limits=['Only received first episodes; all-world graph slots/autoreset surplus are separate.','Recorded public gyro/sensor origin, Nom flag13 generation and solver contact/frame/slip generation remain source-bound, not independently simulated.','Phase role/monitor full-private generation is source-bound; the named phase-reference lifecycle is numerically checked.','Original B1 parking; no inheritance of round377 withdrawal-parking PPO receipt. No learned benefit/formal5 admission.'],
        new_physics_steps=0,new_FD_calls=0,new_model_forward_rows=0,new_optimizer_updates=0,baseline_qualified=False,formal5_admitted=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--save',type=Path)
    args=parser.parse_args()
    try:
        receipt=review(args.runtime,out=args.output)
        if args.save:
            assert not args.save.exists(),'Never overwrite a receipt';args.save.write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(receipt,ensure_ascii=False,allow_nan=False))
    except (AssertionError,ValueError,KeyError,FileNotFoundError,TypeError) as e:
        print(json.dumps(dict(verified=False,status='refused_missing_or_invalid_raw_admission',error=str(e),new_physics_steps=0,new_FD_calls=0,new_model_forward_rows=0)));raise SystemExit(1)
