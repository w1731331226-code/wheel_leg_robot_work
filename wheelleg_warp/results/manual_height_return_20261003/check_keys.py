import sys
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
import manual_demo as demo
import glfw
original_overlay=demo.mujoco.mjr_overlay;original_swap=glfw.swap_buffers;original_callback=glfw.set_key_callback;original_key=glfw.get_key;original_focus=glfw.get_window_attrib
state=dict(frame=0,callback=None,text='',checkpoint={})
def callback(window,handler):state['callback']=handler;return original_callback(window,handler)
def overlay(*args):
 if args[3].startswith(('MANUAL','TASK','GPU JUMP')):state['text']=args[3]
 return original_overlay(*args)
def key(window,key):
 return glfw.PRESS if key==glfw.KEY_DOWN and 2<=state['frame']<500 else glfw.RELEASE
 def_unused=None
def focus(window,flag):return 1 if flag==glfw.FOCUSED else original_focus(window,flag)
def swap(window):
 state['frame']+=1;f=state['frame']
 if f==510:state['checkpoint']['down']=state['text'];state['callback'](window,glfw.KEY_4,0,glfw.PRESS,0)
 if f==520:state['callback'](window,glfw.KEY_KP_1,0,glfw.PRESS,0)
 if f==550:state['checkpoint']['keypad1']=state['text'];state['callback'](window,glfw.KEY_1,0,glfw.PRESS,0)
 if f==580:
  state['checkpoint']['main1']=state['text'];glfw.set_window_should_close(window,True)
 original_swap(window)
demo.mujoco.mjr_overlay=overlay;glfw.set_key_callback=callback;glfw.swap_buffers=swap;glfw.get_key=key;glfw.get_window_attrib=focus
demo.run_window(.3,1.,'step')
for name,text in state['checkpoint'].items():print('CASE',name,'\n'+text,flush=True)
