"""Nearest guide request in the current static geometry and original motor domain."""
import math
import numpy as np
import warp as wp
from scipy.optimize import linprog
from native.controller import control_physical
from probe_height_angle_envelope import feasible
from probe_height_115_radial_authority import torque_box
from state_estimation import leg_kinematics


def angle_bound(length, cap, sign):
    assert feasible(length, 0., cap)
    inside, outside = 0., sign*math.pi/2
    # Existing connected standing branch and original 26-step bisection.
    for _ in range(26):
        mid = (inside+outside)/2
        if feasible(length, mid, cap): inside = mid
        else: outside = mid
    return inside


def allocate(env, previous, wanted, enabled, shadow_state, shadow_ctrl, shadow_diag):
    d = env.data
    wp.copy(shadow_state, env.k['state'])
    wp.launch(control_physical, env.num_envs, [d.qpos,d.qvel,d.sensordata,env.targets,env.command,env.active,
        shadow_state,env.ids,env.k['heights'],env.k['gains'],env.k['feed'],env.k['angles'],env.k['reference'],
        env.k['yaw'],shadow_ctrl,shadow_diag,0,0]+env.control_extra[:1], block_dim=32)
    q, v = d.qpos.numpy(), d.qvel.numpy()
    base, diag = shadow_ctrl.numpy().astype(float), shadow_diag.numpy()
    ids = env.ids.numpy(); heights = env.k['heights'].numpy(); gains = env.k['gains'].numpy()
    selected = wanted.copy(); records = []
    for w in np.flatnonzero(enabled):
        legs = [leg_kinematics(q[w,ids[2*s:2*s+2]], np.zeros(2)) for s in range(2)]
        length = sum(np.linalg.norm(leg[0]) for leg in legs)/2
        kg = float(np.interp(length, heights, gains[:,1,0]))
        assert kg > 0
        A, b = [], []
        for s, leg in enumerate(legs):
            row = np.zeros(13); inverse = np.linalg.inv(leg[3]); row[2*s:2*s+2] = inverse[1]
            hub = float(inverse[1] @ base[w,2*s:2*s+2])
            angle = math.atan2(leg[0][1],leg[0][0])+math.pi/2
            low, high = [kg*(angle_bound(diag[w,21+s],1.4,sign)-angle)+diag[w,27]-hub for sign in (-1,1)]
            A.extend((row,-row)); b.extend((high,-low))
        for j in range(6):
            row=np.zeros(13);row[j]=1;row[6]=-1;A.append(row);b.append(wanted[w,j])
            row=np.zeros(13);row[j]=-1;row[6]=-1;A.append(row);b.append(-wanted[w,j])
            row=np.zeros(13);row[j]=1;row[7+j]=-1;A.append(row);b.append(previous[w,j])
            row=np.zeros(13);row[j]=-1;row[7+j]=-1;A.append(row);b.append(-previous[w,j])
        row=np.zeros(13);row[7:]=1;A.append(row);b.append(.1)
        bounds = torque_box(env.cpu,v[w])/env.actuator_gain_upper
        motor_bounds = [(max(-1.,-bounds[j]-base[w,j]),min(1.,bounds[j]-base[w,j])) for j in range(6)]
        c=np.zeros(13);c[6]=1
        # ponytail: current static target only; full physical rollout decides
        # admission. No acceleration model, recursive certificate or lookahead.
        result=linprog(c,A_ub=A,b_ub=b,bounds=motor_bounds+[(0,None)]*7,method='highs',options={'primal_feasibility_tolerance':1e-10,'dual_feasibility_tolerance':1e-10})
        row=dict(world=int(w),solver_status=int(result.status),message=result.message)
        if not result.success:
            return None, records+[row]
        selected[w]=result.x[:6]
        assert abs(selected[w]).max() <= 1.+1e-12
        assert abs(selected[w]-previous[w]).sum() <= .1+1e-12
        excess=float(np.max(np.array(A)@result.x-b))
        if excess > 1e-9:
            row.update(reason="independent_constraint_residual_failed",maximum_linear_excess=excess)
            return None, records+[row]
        row.update(wanted=wanted[w].tolist(),selected=selected[w].tolist(),maximum_linear_excess=excess)
        records.append(row)
    return selected, records
