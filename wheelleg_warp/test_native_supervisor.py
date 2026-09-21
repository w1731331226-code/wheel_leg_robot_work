"""选模优先级和停滞判定：不得以偏航改善掩盖失败增加。"""
from train_native import significant,selection_key
base=dict(total=32,success_count=31,complete=True,mean_yaw_score_deg=1.)
assert not significant({**base,'success_count':30,'mean_yaw_score_deg':.1},base)
assert significant({**base,'success_count':32,'mean_yaw_score_deg':2.},base)
assert not significant({**base,'mean_yaw_score_deg':.997},base)
assert significant({**base,'mean_yaw_score_deg':.99},base)
assert not significant({**base,'complete':False},base)
assert selection_key({**base,'success_count':30,'mean_yaw_score_deg':.1})>selection_key(base)
print('PASS: success first, meaningful improvement, incomplete rejection')
