from pathlib import Path
import sys,json
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
import manual_demo as demo
import glfw,numpy as np
from PIL import Image
from training_contract import digest
swap0=glfw.swap_buffers;overlay0=demo.mujoco.mjr_overlay;cb0=glfw.set_key_callback;focus0=glfw.get_window_attrib
s=dict(frame=0,callback=None,text='',ctx=None,rect=None,flight_count=0,previous_fly=False,second_queued=False,queued_visible=False,complete=False)
def callback(w,h):s['callback']=h;return cb0(w,h)
def overlay(*args):
 s['rect']=args[2];s['ctx']=args[-1]
 if args[3].startswith(('MANUAL','GPU JUMP')):s['text']=args[3]
 return overlay0(*args)
def key(w,k):return glfw.PRESS if k==glfw.KEY_A and 100<=s['frame']<200 else glfw.RELEASE
def focus(w,k):return 1 if k==glfw.FOCUSED else focus0(w,k)
def swap(w):
 s['frame']+=1;f=s['frame']
 if f==200:assert 'Turn command: +0.60rad/s' in s['text'],s['text']
 if f==330:s['callback'](w,glfw.KEY_7,0,glfw.PRESS,0)
 if 'GPU JUMP SQUAT' in s['text'] and not s['second_queued']:
  s['callback'](w,glfw.KEY_KP_7,0,glfw.PRESS,0);s['second_queued']=True
 if 'WAIT LANDING' in s['text']:s['queued_visible']=True
 flying='GPU JUMP FLY' in s['text']
 if flying and not s['previous_fly']:s['flight_count']+=1
 s['previous_fly']=flying
 if s['flight_count']>=2 and s['text'].startswith('MANUAL') and 'COMPLETED' in s['text']:
  s['complete']=True;r=s['rect'];rgb=np.empty((r.height,r.width,3),np.uint8)
  demo.mujoco.mjr_readPixels(rgb,None,r,s['ctx']);Image.fromarray(np.flipud(rgb)).save('/home/wmt/wheel_leg_robot_work/wheelleg_warp/results/manual_input_turn_20261003/preview_completed.png')
  glfw.set_window_should_close(w,True)
 if f>1300:glfw.set_window_should_close(w,True)
 swap0(w)
glfw.set_key_callback=callback;glfw.swap_buffers=swap;demo.mujoco.mjr_overlay=overlay;glfw.get_key=key;glfw.get_window_attrib=focus
demo.run_window(.3,1.)
assert s['complete'] and s['queued_visible'] and s['flight_count']==2,s
out=Path('/home/wmt/wheel_leg_robot_work/wheelleg_warp/results/manual_input_turn_20261003')
(out/'window_check.json').write_text(json.dumps(dict(passed=True,A_key_rate_command=.6,main7_then_keypad7=True,queued_during_squat=True,
 queue_visible=True,actual_flights=2,returned_ground_completed=True,frames=s['frame'],final_overlay=s['text'],source_sha256=digest('/home/wmt/wheel_leg_robot_work/wheelleg_warp/manual_demo.py')),indent=2)+'\n')
print('PASS real A-key rate and7/keypad7 duringSQUAT->2flights->completed ground; queue feedback visible')
