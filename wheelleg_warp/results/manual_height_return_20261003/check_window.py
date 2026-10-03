from pathlib import Path
import sys,json
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
import manual_demo as demo
import glfw,numpy as np
from PIL import Image
from training_contract import digest
swap0=glfw.swap_buffers;overlay0=demo.mujoco.mjr_overlay;callback0=glfw.set_key_callback;focus0=glfw.get_window_attrib
s=dict(frame=0,callback=None,flight=False,returned=False,actual115=False,rect=None,ctx=None,text='',down_start=None)
def callback(window,handler):s['callback']=handler;return callback0(window,handler)
def overlay(*args):
 s['rect']=args[2];s['ctx']=args[-1]
 if args[3].startswith(('MANUAL','TASK','GPU JUMP')):s['text']=args[3]
 return overlay0(*args)
def key(window,key):return glfw.PRESS if key==glfw.KEY_DOWN and s['down_start'] is not None else glfw.RELEASE
def focus(window,flag):return 1 if flag==glfw.FOCUSED else focus0(window,flag)
def swap(window):
 s['frame']+=1;f=s['frame']
 if f==50:s['callback'](window,glfw.KEY_7,0,glfw.PRESS,0)
 if 'GPU JUMP FLY' in s['text']:s['flight']=True
 if s['flight'] and s['text'].startswith('MANUAL') and not s['returned']:s['returned']=True;s['down_start']=f
 if s['returned'] and 'Target: 115.0 mm | Mean FK leg: 115.0 mm' in s['text']:
  s['actual115']=True;r=s['rect'];rgb=np.empty((r.height,r.width,3),np.uint8)
  demo.mujoco.mjr_readPixels(rgb,None,r,s['ctx'])
  Image.fromarray(np.flipud(rgb)).save('/home/wmt/wheel_leg_robot_work/wheelleg_warp/results/manual_height_return_20261003/preview_return_115.png')
  glfw.set_window_should_close(window,True)
 if f>1200:glfw.set_window_should_close(window,True)
 swap0(window)
glfw.set_key_callback=callback;glfw.swap_buffers=swap;demo.mujoco.mjr_overlay=overlay;glfw.get_key=key;glfw.get_window_attrib=focus
demo.run_window(.3,1.,'step')
assert s['flight'] and s['returned'] and s['actual115'],s
out=Path('/home/wmt/wheel_leg_robot_work/wheelleg_warp/results/manual_height_return_20261003')
(out/'window_check.json').write_text(json.dumps(dict(passed=True,actual_flight=True,automatic_return_to_ground=True,
 actual_display_115=True,frames=s['frame'],final_overlay=s['text'],source_sha256=digest('/home/wmt/wheel_leg_robot_work/wheelleg_warp/manual_demo.py')),indent=2)+'\n')
print('PASS actual7-key flight->automatic ground->Down->115mm displayed')
