import sys
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
import manual_demo as demo
import glfw,numpy as np
from PIL import Image
original_overlay=demo.mujoco.mjr_overlay;original_swap=glfw.swap_buffers;original_callback=glfw.set_key_callback
state=dict(frame=0,callback=None,passed=False,rect=None,ctx=None)
def callback(window,handler):
    state['callback']=handler;return original_callback(window,handler)
def overlay(*args):
    state['rect']=args[2];state['ctx']=args[-1]
    if 'TASK  PASS' in args[3]:state['passed']=True
    return original_overlay(*args)
def swap(window):
    state['frame']+=1
    if state['frame']==2:state['callback'](window,glfw.KEY_T,0,glfw.PRESS,0)
    if state['passed']:
        r=state['rect'];rgb=np.empty((r.height,r.width,3),np.uint8)
        demo.mujoco.mjr_readPixels(rgb,None,r,state['ctx'])
        Image.fromarray(np.flipud(rgb)).save('/home/wmt/wheel_leg_robot_work/wheelleg_warp/results/manual_demo_v2_20261003/preview_task_pass.png')
        glfw.set_window_should_close(window,True)
    elif state['frame']>=800:glfw.set_window_should_close(window,True)
    original_swap(window)
glfw.set_key_callback=callback;glfw.swap_buffers=swap;demo.mujoco.mjr_overlay=overlay
demo.run_window(.115,1.)
assert state['passed'],'T-key task did not show PASS'
print('PASS real T-key callback,115mm full task and PASS overlay; frames',state['frame'])
