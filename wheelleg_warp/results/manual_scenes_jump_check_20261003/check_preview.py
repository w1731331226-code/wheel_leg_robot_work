import sys
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
import manual_demo as demo
import glfw,numpy as np
from PIL import Image
original_overlay=demo.mujoco.mjr_overlay;original_swap=glfw.swap_buffers;original_callback=glfw.set_key_callback
state=dict(frame=0,callback=None,rect=None,ctx=None,text='',scene_saved=False,jump_saved=False,landed=False)
def callback(window,handler):state['callback']=handler;return original_callback(window,handler)
def overlay(*args):
 state['rect']=args[2];state['ctx']=args[-1]
 if args[3].startswith(('MANUAL','GPU JUMP')):state['text']=args[3]
 return original_overlay(*args)
def save(name):
 r=state['rect'];rgb=np.empty((r.height,r.width,3),np.uint8)
 demo.mujoco.mjr_readPixels(rgb,None,r,state['ctx'])
 Image.fromarray(np.flipud(rgb)).save('/home/wmt/wheel_leg_robot_work/wheelleg_warp/results/manual_scenes_jump_check_20261003/'+name)
def swap(window):
 state['frame']+=1;f=state['frame']
 if f==2:state['callback'](window,glfw.KEY_F6,0,glfw.PRESS,0)
 if f==50:
  assert 'Step15mm' in state['text'];save('preview_step.png');state['scene_saved']=True
  state['callback'](window,glfw.KEY_7,0,glfw.PRESS,0)
 if 'GPU JUMP FLY' in state['text'] and not state['jump_saved']:
  save('preview_gpu_jump.png');state['jump_saved']=True
 if state['jump_saved'] and 'GPU JUMP DRIVE' in state['text']:
  state['landed']=True;glfw.set_window_should_close(window,True)
 if f>700:glfw.set_window_should_close(window,True)
 original_swap(window)
glfw.set_key_callback=callback;glfw.swap_buffers=swap;demo.mujoco.mjr_overlay=overlay
demo.run_window(.3,1.)
assert state['scene_saved'] and state['jump_saved'] and state['landed'],state
print('PASS real F6 scene switching and7-key GPU flight/landing; frames',state['frame'])
