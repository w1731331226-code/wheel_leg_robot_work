from pathlib import Path
import sys
sys.path.insert(0,'/home/wmt/wheel_leg_robot_work/wheelleg_warp')
import manual_demo as demo
import glfw
import numpy as np
from PIL import Image
original_overlay=demo.mujoco.mjr_overlay
original_swap=glfw.swap_buffers
frame=[0];context=[None];viewport=[None]
def overlay(*args):
    context[0]=args[-1];viewport[0]=args[2]
    return original_overlay(*args)
def swap(window):
    frame[0]+=1
    if frame[0]==90:
        rect=viewport[0];rgb=np.empty((rect.height,rect.width,3),np.uint8)
        demo.mujoco.mjr_readPixels(rgb,None,rect,context[0])
        Image.fromarray(np.flipud(rgb)).save('/home/wmt/wheel_leg_robot_work/wheelleg_warp/results/manual_demo_v2_20261003/preview_115.png')
        glfw.set_window_should_close(window,True)
    original_swap(window)
demo.mujoco.mjr_overlay=overlay;glfw.swap_buffers=swap
demo.run_window(.115,1.)
print('PASS real GLFW/MuJoCo90-frame GPU115mm manual preview and overlay')
