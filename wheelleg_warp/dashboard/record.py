"""读取真实训练采样状态渲染；轨迹归档可离线重放，不重新运行策略。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import numpy as np
from PIL import Image
from ppo_env import build_model,Scenario

HERE=Path(__file__).resolve().parent
DATA=HERE/'local_data'
RUN=ROOT/'wheelleg_warp/results/formal_cpu_warp_live_v1_20260921'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


class View:
    def __init__(self,scenario):
        self.model=build_model(Scenario(**scenario));self.data=mujoco.MjData(self.model)
        self.renderer=mujoco.Renderer(self.model,height=360,width=640)
        self.camera=mujoco.MjvCamera();self.camera.distance=1.65;self.camera.azimuth=125;self.camera.elevation=-19

    def image(self,qpos,qvel,ctrl):
        self.data.qpos[:]=qpos;self.data.qvel[:]=qvel;self.data.ctrl[:]=ctrl
        mujoco.mj_forward(self.model,self.data)
        self.camera.lookat[:]=[qpos[0]+.35,qpos[1],.20]
        self.renderer.update_scene(self.data,camera=self.camera)
        return Image.fromarray(self.renderer.render())

    def close(self):self.renderer.close()


def replay(folder,target):
    metadata=read(folder/'metadata.json')
    if not metadata or metadata['status']!='completed':raise ValueError('仅重放完整归档回合')
    protocol=read(Path(metadata['source_run']).parent/'protocol.json')
    if protocol is None:
        protocol=read(folder.parents[3]/'protocol.json') if len(folder.parents)>3 else None
    hashes=metadata.get('model_source_sha256') or (protocol or {}).get('source_sha256',{})
    for name,digest in hashes.items():
        if name.startswith('wheelleg_ppo/') and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('模型/控制源码变化，先恢复归档版本再重放：'+name)
    if not hashes:raise RuntimeError('缺少原始模型源码指纹')
    view=View(metadata['scenario']);images=[]
    try:
        with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
            indices=sorted(set(range(0,len(trace['time']),2))|{len(trace['time'])-1})
            for i in indices:images.append(view.image(trace['qpos'][i],trace['qvel'][i],trace['ctrl'][i]))
        images[0].save(target,format='GIF',save_all=True,append_images=images[1:],duration=40,loop=0,optimize=False)
    finally:view.close()
    return len(images)


def source(backend):
    folders=[p for p in RUN.glob(backend+'_*') if (p/'live/latest.json').exists()]
    if not folders:folders=[p for p in RUN.glob('smoke_'+backend+'_*') if (p/'live/latest.json').exists()]
    return max(folders,key=lambda p:(p/'live/latest.json').stat().st_mtime)/'live' if folders else None


def watch():
    views={};last={};saved={}
    (DATA/'live').mkdir(parents=True,exist_ok=True)
    while True:
        for backend in ('cpu','warp'):
            src=source(backend)
            if not src:continue
            metadata=read(src/'latest.json')
            if not metadata:continue
            key=(str(src),metadata['episode'],metadata['frame'])
            if key==last.get(backend):continue
            try:
                with np.load(src/'latest.npz',allow_pickle=False) as frame:
                    if int(frame['frame'])!=metadata['frame'] or int(frame['episode'])!=metadata['episode']:continue
                    model_key=(str(src),metadata['episode'])
                    if backend not in views or views[backend][0]!=model_key:
                        if backend in views:views[backend][1].close()
                        views[backend]=(model_key,View(metadata['scenario']))
                    image=views[backend][1].image(frame['qpos'],frame['qvel'],frame['ctrl'])
                temporary=DATA/'live'/f'{backend}.tmp';image.save(temporary,format='JPEG',quality=90)
                temporary.replace(DATA/'live'/f'{backend}.jpg')
                write(DATA/'live'/f'{backend}.json',dict(**metadata,rendered_wall_time=time.time(),
                    smoke=src.parent.name.startswith('smoke_'),image=f'/media/live/{backend}.jpg'))
                last[backend]=key
                # 原始NPZ每回合都保留；GIF每约2万采样步抽一个完整回合，控制本地录像体积。
                bucket=(str(src),metadata['sample_steps']//20000)
                if saved.get(backend)!=bucket:
                    complete=[p for p in (src/'episodes').glob('*/trajectory.npz') if read(p.parent/'metadata.json').get('status')=='completed']
                    if complete:
                        episode=max(complete,key=lambda p:p.stat().st_mtime).parent
                        output=DATA/'captures'/backend/(src.parent.name+'_'+episode.name)
                        if not output.exists():
                            output.mkdir(parents=True)
                            import shutil
                            shutil.copy2(episode/'trajectory.npz',output/'trajectory.npz')
                            meta=read(episode/'metadata.json');meta.update(episode_directory=str(episode),smoke=src.parent.name.startswith('smoke_'),sample_steps_observed=metadata['sample_steps'],archive_time=time.time())
                            protocol=read(src.parent.parent/'protocol.json')
                            meta['model_source_sha256']={k:v for k,v in protocol['source_sha256'].items() if k.startswith('wheelleg_ppo/')}
                            meta['mujoco_version']=mujoco.__version__
                            write(output/'metadata.json',meta)
                            replay(episode,output/'animation.tmp')
                            (output/'animation.tmp').replace(output/'animation.gif')
                        saved[backend]=bucket
            except Exception as exc:
                print('渲染错误：',backend,repr(exc),flush=True)
        time.sleep(.1)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path);p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.replay:
        target=args.output or args.replay/'reproduced.gif';print(replay(args.replay,target),'frames',target)
    else:watch()
