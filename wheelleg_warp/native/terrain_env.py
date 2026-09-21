"""Native GPU environment with terrain-v1's uniform geometry bank."""
from native.environment import NativeEnv
from native.terrain import bank


class TerrainEnv(NativeEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,bank_factory=bank,**kwargs)
