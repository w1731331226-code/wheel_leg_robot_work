from pathlib import Path
import sys,json,time
sys.path.insert(0,'wheelleg_warp')
import manual_demo as demo
import glfw,numpy as np
from PIL import Image
swap0=glfw.swap_buffers;overlay0=demo.mujoco.mjr_overlay;cb0=glfw.set_key_callback
s=dict(frame=0,callback=None,text='',ctx=None,rect=None,flights=0,prev=False,restores=0,entry_frames=[],durations=[])
def cb(w,h):s['callback']=h;return cb0(w,h)
def overlay(*args):
 s['ctx']=args[-1];s['rect']=args[2]
 if args[3].startswith(('MANUAL','GPU JUMP')):s['text']=args[3]
 return overlay0(*args)
def swap(w):
 s['frame']+=1;f=s['frame']
 if f==60:s['callback'](w,glfw.KEY_7,0,glfw.PRESS,0);s['entry_frames'].append(f)
 fly='GPU JUMP FLY' in s['text']
 if fly and not s['prev']:s['flights']+=1
 s['prev']=fly
 if s['flights']>s['restores'] and s['text'].startswith('MANUAL') and 'Target: 115.0 mm | Mean FK leg: 115.0 mm' in s['text']:
  s['restores']+=1
  if s['restores']==1:s['callback'](w,glfw.KEY_KP_7,0,glfw.PRESS,0);s['entry_frames'].append(f)
  else:
   r=s['rect'];rgb=np.empty((r.height,r.width,3),np.uint8);demo.mujoco.mjr_readPixels(rgb,None,r,s['ctx'])
   Image.fromarray(np.flipud(rgb)).save('wheelleg_warp/results/manual_jump_latency_20261003/preview_restored.png')
   glfw.set_window_should_close(w,True)
 if f>850:glfw.set_window_should_close(w,True)
 swap0(w)
glfw.set_key_callback=cb;glfw.swap_buffers=swap;demo.mujoco.mjr_overlay=overlay
demo.run_window(.115,1.)
assert s['flights']==s['restores']==2,s
Path('wheelleg_warp/results/manual_jump_latency_20261003/window_check.json').write_text(json.dumps(dict(passed=True,flights=2,automatic_115_restores=2,frames=s['frame'],entry_frames=s['entry_frames']),indent=2)+'\n')
print('PASS actual main7/keypad7 repeated jump; original115mm automatically restored both times')
