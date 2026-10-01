"""Fixed7kg current-J design shared by height-115 candidate evaluation and training preflight."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'wheelleg_ppo/tools'))
import mujoco
import numpy as np
import wheelleg_sim as sim
import model_lqr as ml
from model_lqr import sagittal_basis,vmc_coordinates
from state_estimation import leg_kinematics


def mapped_input(model,reference,delta,virtual):
    """Actual current-J VMC plus radial PD, expressed as three common motor inputs."""
    basis,inputs=sagittal_basis(model);_,g,_=vmc_coordinates(reference)
    feed=np.linalg.solve(g,np.linalg.pinv(inputs)@reference.ctrl)
    full=basis@delta;q=reference.qpos.copy();mujoco.mj_integratePos(model,q,full[:model.nv],1.)
    v=reference.qvel+full[model.nv:]
    joints=[model.joint(name) for name in ('alphaL','betaL')]
    leg,_,_,jac=leg_kinematics(q[[j.qposadr[0] for j in joints]],v[[j.dofadr[0] for j in joints]])
    height=sim.fk_joints(reference.qpos[joints[0].qposadr[0]],reference.qpos[joints[1].qposadr[0]])['leg_len']
    rate=(jac.T@v[[j.dofadr[0] for j in joints]])[0]
    force=feed[2]+sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(sim.KP_LEG*(height-np.linalg.norm(leg))-sim.KD_LEG*rate)
    return np.r_[feed[0]+virtual[0],jac@np.array([force,feed[1]+virtual[1]])]


def nominal_input_jacobian(model,reference,eps=1e-6):
    eye=np.eye(15)*eps;zero=np.zeros(2)
    return np.column_stack([(mapped_input(model,reference,d,zero)-mapped_input(model,reference,-d,zero))/(2*eps) for d in eye])



def current_vmc_table():
    model,_=sim.load_model(ml.XML,True);table=[];reports=[]
    for height in (.115,sim.L_SQUAT_MIN,sim.L_PREP,sim.L_STAND,sim.L_MAX):
        ref,a,b,_=ml.design(model,height,min_height=.115);c,g,_=ml.vmc_coordinates(ref)
        outer=np.zeros((3,15));outer[2]=sim.hw.DESIGN_MASS/sim.hw.BASELINE_MASS*(sim.KP_LEG*c[6]+sim.KD_LEG*c[7])
        actual_input=nominal_input_jacobian(model,ref)
        assert np.allclose(actual_input,nominal_input_jacobian(model,ref,5e-7),rtol=.01,atol=1e-4)
        # Existing reducer subtracts the radial PD. Add geometric feed derivative
        # first, so its full closed-loop check represents the actual current-J map.
        effective_a=a+b@(actual_input+g@outer)
        gain,report=ml.reduced_design(model,ref,effective_a,b,6)
        assert report['linear_pass'];reports.append(dict(height_m=height,model_scope='fixed-node current-J VMC and nominal radial PD; excludes unilateral guard, interpolation derivatives, projection and saturation',**report))
        k=np.linalg.solve(g,gain)[:2]@np.linalg.pinv(c[:6]);mapping=ml.sagittal_basis(model)[1]
        feed=np.linalg.solve(g,np.linalg.pinv(mapping)@ref.ctrl)
        angle=sim.fk_joints(ref.qpos[7],ref.qpos[10])['phi5']+np.pi/2
        table.append((k,feed,angle))
    return table,reports

