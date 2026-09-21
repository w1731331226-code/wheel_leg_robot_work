"""读取真实训练采样状态渲染；轨迹归档可离线重放，不重新运行策略。"""
from collections import deque
import base64,io
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
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
RUN=ROOT/'wheelleg_warp/results/formal_native_1024_20260921'


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)


class View:
    def __init__(self,scenario,native=False):
        factory=build_model
        scenario_type=Scenario
        if native and 'terrain' in scenario:
            from native.terrain import model as factory,TerrainScenario as scenario_type
        elif native:
            from native.models import model as factory
        self.model=factory(scenario_type(**scenario));self.data=mujoco.MjData(self.model)
        self.model.vis.global_.offwidth=960;self.model.vis.global_.offheight=540
        self.model.mat_reflectance[:]=0
        self.model.vis.headlight.ambient[:]=[.3,.3,.3]
        self.direction=1 if scenario['speed']>0 else -1
        self.renderer=mujoco.Renderer(self.model,height=540,width=960)
        self.camera=mujoco.MjvCamera();self.camera.distance=1.65;self.camera.azimuth=125;self.camera.elevation=-19

    def image(self,qpos,qvel,ctrl):
        self.data.qpos[:]=qpos;self.data.qvel[:]=qvel;self.data.ctrl[:]=ctrl
        mujoco.mj_forward(self.model,self.data)
        self.camera.lookat[:]=[qpos[0]+.35*self.direction,qpos[1],.20]
        self.renderer.update_scene(self.data,camera=self.camera)
        return Image.fromarray(self.renderer.render())

    def close(self):self.renderer.close()


def replay(folder,target,webp=False):
    metadata=read(folder/'metadata.json')
    if not metadata or metadata['status']!='completed':raise ValueError('仅重放完整归档回合')
    protocol=read(Path(metadata['source_run']).parent/'protocol.json')
    if protocol is None:
        protocol=read(folder.parents[3]/'protocol.json') if len(folder.parents)>3 else None
    hashes=metadata.get('model_source_sha256') or (protocol or {}).get('source_sha256',{})
    for name,digest in hashes.items():
        if (name.startswith('wheelleg_ppo/') or name.startswith('wheelleg_warp/native/')) and hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('模型/控制源码变化，先恢复归档版本再重放：'+name)
    if not hashes:raise RuntimeError('缺少原始模型源码指纹')
    view=View(metadata['scenario'],native=metadata.get('backend')=='native');images=[]
    try:
        with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
            indices=np.unique(np.minimum(np.searchsorted(trace['time'],np.arange(trace['time'][0],trace['time'][-1]+1e-10,.02)),len(trace['time'])-1))
            for i in indices:images.append(view.image(trace['qpos'][i],trace['qvel'][i],trace['ctrl'][i]))
        temporary=target.with_suffix(target.suffix+'.tmp')
        images[0].save(temporary,format='GIF',save_all=True,append_images=images[1:],duration=20,loop=0,optimize=False)
        temporary.replace(target)
        if webp:
            alternate=target.with_suffix('.webp');temporary=alternate.with_suffix('.webp.tmp')
            images[0].save(temporary,format='WEBP',save_all=True,append_images=images[1:],duration=20,loop=0,quality=85,method=3)
            temporary.replace(alternate)
    finally:view.close()
    return len(images)


def source():
    paths=list(RUN.glob('round_*/live/latest.json'))
    return max(paths,key=lambda p:p.stat().st_mtime).parent if paths else None


def watch():
    pending_frames=deque(maxlen=120);view=None;view_key=None;last_chunk=None;export=None;archived=set();last_scan=0.
    counted=0;fps=0.;window=time.monotonic();last_render=0.;dropped=0
    (DATA/'live').mkdir(parents=True,exist_ok=True)
    while True:
        src=source();metadata=read(src/'latest.json') if src else None
        if metadata:
            key=(str(src),metadata['episode'],metadata['sequence'])
            if key!=last_chunk:
                if pending_frames and pending_frames[-1][0]['environment_index']!=metadata['environment_index']:pending_frames.clear()
                try:
                    with np.load(src/'latest.npz',allow_pickle=False) as chunk:
                        if int(chunk['sequence'])==metadata['sequence'] and int(chunk['episode'])==metadata['episode'] and int(chunk['environment'])==metadata['environment_index']:
                            count=len(chunk['time']);dropped+=max(0,len(pending_frames)+count-pending_frames.maxlen)
                            for i in range(count):
                                pending_frames.append((metadata,{k:chunk[k][i].copy() for k in ('time','qpos','qvel','ctrl')},i))
                            last_chunk=key
                except (FileNotFoundError,ValueError,EOFError):pass
        now=time.monotonic()
        if pending_frames and now-last_render>=1/50:
            meta,frame,index=pending_frames.popleft();key=(meta['source_run'],json.dumps(meta['scenario'],sort_keys=True))
            if key!=view_key:
                if view:view.close()
                view=View(meta['scenario'],native=True);view_key=key
            image=view.image(frame['qpos'],frame['qvel'],frame['ctrl']);buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=88)
            image_bytes=buffer.getvalue();last_render=now;counted+=1
            if last_render-window>=1:fps=counted/(last_render-window);counted=0;window=last_render
            display={**meta,'simulation_seconds':float(frame['time']),'frame':meta['sequence']*8+index,
                'rendered_wall_time':time.time(),'render_fps':fps,'buffered_frames':len(pending_frames),'dropped_display_frames':dropped}
            temporary=DATA/'live/native.jpg.tmp';temporary.write_bytes(image_bytes);temporary.replace(DATA/'live/native.jpg')
            write(DATA/'live/native.json',display)
            # 图像与来源标签处于同一原子包，浏览器解码后同时切换，避免跨回合错配。
            write(DATA/'live/native.frame.json',dict(metadata=display,jpeg=base64.b64encode(image_bytes).decode()))
        if export is not None and export.poll() is not None:export=None
        if now-last_scan>3:
            last_scan=now
            if src:
                for metadata_path in (src/'episodes').glob('*/metadata.json'):
                    meta=read(metadata_path)
                    if not meta or meta['status']!='completed' or str(metadata_path) in archived:continue
                    output=DATA/'captures/native'/(RUN.name+'__'+src.parent.name+'_'+metadata_path.parent.name)
                    output.mkdir(parents=True,exist_ok=True)
                    import shutil
                    shutil.copy2(metadata_path.parent/'trajectory.npz',output/'trajectory.npz')
                    protocol=read(RUN/'protocol.json') or {}
                    meta.update(archive_time=time.time(),model_source_sha256=protocol.get('source_sha256',{}))
                    write(output/'metadata.json',meta);archived.add(str(metadata_path))
            if export is None:
                waiting=[p.parent for p in (DATA/'captures/native').glob('*/metadata.json') if not (p.parent/'animation_50.webp').exists() and not (p.parent/'export_failed.json').exists()]
                if waiting:
                    folder=min(waiting,key=lambda p:p.stat().st_mtime)
                    with (DATA/'export.log').open('a') as log:
                        export=subprocess.Popen([sys.executable,str(Path(__file__)),'--replay',str(folder),'--output',str(folder/'animation_50.gif'),'--webp'],stdout=log,stderr=subprocess.STDOUT)
        time.sleep(max(.001,min(.01,1/50-(time.monotonic()-last_render))) if pending_frames else .01)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path);p.add_argument('--output',type=Path);p.add_argument('--webp',action='store_true');p.add_argument('--run-root',type=Path);p.add_argument('--data-root',type=Path)
    args=p.parse_args()
    if args.run_root:RUN=args.run_root.resolve()
    if args.data_root:DATA=args.data_root.resolve()
    if args.replay:
        target=args.output or args.replay/'reproduced_50.gif'
        try:print(replay(args.replay,target,args.webp),'frames',target,flush=True)
        except Exception as exc:
            write(args.replay/'export_failed.json',dict(error=repr(exc)))
            raise
    else:watch()
