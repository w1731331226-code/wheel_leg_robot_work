"""回归：真实接触经GPU回读后，原控制器使用的geom1/geom2必须正确。"""
import mujoco
import numpy as np
import warp as wp
from baseline import WarpPhysics

wp.init()
wp.set_device('cuda:0')
model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
<geom name="floor" type="plane" size="1 1 .1"/>
<body pos="0 0 .05"><freejoint/><geom name="wheel" type="sphere" size=".1" mass="1"/></body>
</worldbody></mujoco>''')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
physics = WarpPhysics(model, data)
# 模拟_realloc_con_efc留下的旧字段；测试不能依赖偶然复用的正确内存。
data.contact.geom1[:] = -1
data.contact.geom2[:] = -1
physics.step(model, data)
assert data.ncon > 0
np.testing.assert_array_equal(data.contact.geom1, data.contact.geom[:, 0])
np.testing.assert_array_equal(data.contact.geom2, data.contact.geom[:, 1])
assert all(0 <= int(g) < model.ngeom for g in data.contact.geom.ravel())
print('PASS：GPU接触字段与原CPU控制器兼容')
