const $=id=>document.getElementById(id),num=n=>new Intl.NumberFormat('zh-CN').format(n??0),terrainNames={legacy:'原左右障碍',ramp:'坡道',cross_slope:'横坡',rough:'粗糙路',step:'全宽台阶',mixed:'粗糙路+左右障碍',rolling_slope:'连续坡段',multi_step:'多级台阶',split_level:'左右异高'};
let mode='live',latestClip=null,etag=null,lastKey=null,activePhase='validating';
let displayCount=0,displayWindow=performance.now(),displayFPS=0,lastDisplayed=0;
const names={validating:'收敛准入验证',ready:'准备就绪',initializing:'初始化本轮场景',training:'训练中',evaluating:'开发集评估',final_evaluation:'独立终评',completed:'已完成',failed:'需要检查',interrupted:'已中断'};
const duration=s=>s==null?'等待有效进度':s<60?`${Math.ceil(s)} 秒`:`${(s/60).toFixed(1)} 分钟`;
function meta(m){
 $('source').textContent=activePhase==='completed'?'FINAL SNAPSHOT · 正式训练最后实帧':m.phase?.startsWith('terrain')?'LIVE · 非结构化地形训练':m.phase==='formal_training'?'LIVE · 正式训练原始帧':m.phase==='preflight_training'?'预检 PPO · 原始训练帧':m.phase==='validation_training'?'准入续训 · 原始训练帧':'工程采集预检';
 $('episode').textContent=`ENV ${m.environment_index} / ${m.environments} · ${terrainNames[m.scenario?.terrain]??'原场景'} · EP ${m.episode} · FRAME ${m.frame}`;
 $('sim').textContent=`仿真 ${m.simulation_seconds.toFixed(3)} s · ${num(m.sample_steps)} 策略步`;
 $('lag').textContent=`源状态延迟 ${Math.max(0,Date.now()/1000-m.wall_time).toFixed(1)} s`;
 $('fps').textContent=`${displayFPS.toFixed(1)} FPS / 400 Hz`;
 $('reward-label').textContent='当前回合累计奖励';$('reward').textContent=Number(m.cumulative_reward??0).toFixed(3);
 $('stream-status').textContent=activePhase==='completed'?'训练已完成 · 最后实帧':m.environment_index===chosenEnvironment?(names[activePhase]??activePhase):`等待环境 ${chosenEnvironment} 的新帧`;
}
async function frames(){
 const tick=performance.now();
 if(mode==='live'){
  try{
   const r=await fetch('/api/frame',{headers:etag?{'If-None-Match':etag}:{},cache:'no-cache'});
   if(r.ok){etag=r.headers.get('ETag');const p=await r.json(),m=p.metadata,key=`${m.source_run}/${m.episode}/${m.frame}`;
    if(key!==lastKey){const image=new Image();image.src='data:image/jpeg;base64,'+p.jpeg;await image.decode();
     if(mode==='live'){lastDisplayed=performance.now();displayCount++;if(lastDisplayed-displayWindow>=1000){displayFPS=displayCount*1000/(lastDisplayed-displayWindow);displayCount=0;displayWindow=lastDisplayed;}$('scene').src=image.src;$('empty').style.display='none';meta(m);lastKey=key;}}
   }
  }catch(e){$('stream-status').textContent='等待图像流';}
 }
 setTimeout(frames,Math.max(0,20-(performance.now()-tick)));
}
$('pause').onclick=()=>{mode=mode==='paused'?'live':'paused';$('pause').textContent=mode==='paused'?'继续真实画面':'暂停详情';$('stream-status').textContent=mode==='paused'?'画面已暂停':names[activePhase];etag=null;lastKey=null;};
$('replay').onclick=()=>{
 if(mode==='replay'){mode='live';etag=null;lastKey=null;$('replay').textContent='50 FPS 回合录像';$('pause').disabled=false;return;}
 if(!latestClip)return;mode='replay';$('scene').src=latestClip.webp;$('source').textContent='REPLAY · 已记录回合，非当前采样';
 $('episode').textContent=`ENV 0 / ${latestClip.environments} · EP ${latestClip.episode}`;$('sim').textContent=`真实轨迹 · 50 FPS · ${latestClip.metrics.duration_s.toFixed(2)} s`;
 $('lag').textContent='历史回合播放';$('reward-label').textContent='录像回合总奖励';$('reward').textContent=Number(latestClip.metrics.episode?.r??0).toFixed(3);$('fps').textContent='50 FPS 录像 / 400 Hz 原始';$('replay').textContent='返回真实画面';$('pause').disabled=true;$('empty').style.display='none';
};
function chart(id,points,percentage){
 const w=620,h=185,L=44,R=18,T=12,B=30,maxX=Math.max(1,...points.map(p=>p.x)),maxY=percentage?100:Math.max(.1,...points.map(p=>p.y))*1.15;
 const X=x=>L+x/maxX*(w-L-R),Y=y=>h-B-y/maxY*(h-T-B);let html='';
 for(let i=0;i<4;i++){const value=maxY*i/3;html+=`<line x1="${L}" x2="${w-R}" y1="${Y(value)}" y2="${Y(value)}" stroke="#283448"/><text x="${L-5}" y="${Y(value)+4}" text-anchor="end" fill="#8593a8" font-size="10">${percentage?value.toFixed(0)+'%':value.toFixed(2)+'°'}</text>`;}
 if(points.length){html+=`<path d="${points.map((p,i)=>`${i?'L':'M'}${X(p.x)},${Y(p.y)}`).join(' ')}" fill="none" stroke="#53dbba" stroke-width="2"/>`;for(const p of points)html+=`<circle cx="${X(p.x)}" cy="${Y(p.y)}" r="3" fill="#53dbba"/><text x="${X(p.x)}" y="180" text-anchor="middle" font-size="10" fill="#8593a8">${p.x===0?'起点':'第'+p.x+'轮'}</text>`;}
 else html+='<text x="300" y="85" text-anchor="middle" fill="#8593a8">等待正式检查点评估</text>';
 $(id).innerHTML=html;
}
function update(s){
 const c=s.current,p=s.protocol,selection=s.selection;activePhase=c.status;chosenEnvironment=c.status==='completed'&&s.live?s.live.environment_index:(s.requested_environment??chosenEnvironment);if(document.activeElement!==$('environment'))$('environment').value=chosenEnvironment;
 const roundSteps=c.round_steps??Math.max(0,(c.new_steps??0)-Math.max(0,(c.round??1)-1)*(p.steps_per_round??0));
 const terrain=p.name?.startsWith('terrain-'),total=p.bootstrap_summary?.total??32;document.querySelector('.nav-label').textContent=terrain?'非结构化地形训练':'原生 GPU 基线';document.querySelector('#learning .panel-title p').textContent=terrain?`固定GPU地形开发集 · ${total}场景 · 原场景另行回归`:'固定 CPU 开发集 · 32场景 · 每轮候选最佳';
 if(s.validation){const v=s.validation;$('notice-body').textContent=`1024准入验证：${v.current??v.phase??'准备中'} · 最近完成 ${num(v.policy_steps)} / ${num(v.target)} 步。下方显示带来源标签的真实采样帧，正式启动后接入正式训练流。`;}else{$('notice-body').textContent='总览包含全部1024个训练世界，点击任意格子查看真实3D详情。详情采集400 Hz，显示目标50 FPS；评估期间可能暂停。';}
 $('phase').textContent=names[c.status]??c.status;$('connection').textContent='本机数据已连接';$('updated').textContent=new Date(s.now*1000).toLocaleTimeString('zh-CN');
 $('round').textContent=`${c.round??0} / ${p.max_rounds??10}`;$('stop-rule').textContent=`停滞 ${selection.stagnant_rounds??0} / ${p.patience??3} 轮`;
 $('eta').textContent={evaluating:'评估中',final_evaluation:'独立终评中',completed:'已完成',initializing:'初始化场景',ready:'等待启动',validating:'准入验证',failed:'已停止'}[c.status]??duration(s.round_remaining_training_seconds);$('rate').textContent=s.rate?num(Math.round(s.rate)):'—';
 $('steps').textContent=`${num(roundSteps)} / ${num(p.steps_per_round??2048000)}`;
 $('round-bar').style.width=`${Math.min(100,roundSteps/(p.steps_per_round??2048000)*100)}%`;
 $('total').textContent=`新增 ${num(c.new_steps)} 步 · 继承 ${num(p.inherited_steps)} 步`;
 if(selection.best){$('best').textContent=`${selection.best.summary.success_count} / ${selection.best.summary.total}`;$('best-yaw').textContent=`Jψ ${selection.best.summary.mean_yaw_score_deg.toFixed(3)}°`;$('best-round').textContent=selection.best.round===0?'最佳：准入检查点':`最佳：第 ${selection.best.round} 轮`;$('best-download').hidden=false;$('norm-download').hidden=false;}
 $('config').textContent=`1024环境 · M3 PPO · 每环境采样 ${p.n_steps??'待选定'} 步 · 本轮已更新 ${c.updates??Math.floor(roundSteps/((p.n_steps??50)*(p.environments??1024)))} 次`;
 const rows=[...(p.bootstrap_summary?[{round:0,summary:p.bootstrap_summary}]:[]),...selection.rounds];
 chart('success-chart',rows.map(r=>({x:r.round,y:r.summary.success_count/r.summary.total*100})),true);
 chart('yaw-chart',rows.filter(r=>r.summary.mean_yaw_score_deg!=null).map(r=>({x:r.round,y:r.summary.mean_yaw_score_deg})),false);
 $('decision').textContent=c.status==='completed'?`已按${c.stop_reason==='plateau'?`连续${p.patience}轮停滞`:`${p.max_rounds}轮预算`}停止，保留开发集最佳检查点。`:c.status==='failed'?'本轮异常已停止，请查看保存的错误记录。':`成功数优先，Jψ次之；每轮重新抽样，停滞${p.patience}轮或最多${p.max_rounds}轮停止。`;
 if(mode==='live'&&performance.now()-lastDisplayed>1500)$('fps').textContent='0 FPS · 等待新物理帧';
 if(mode==='live'&&s.live&&Date.now()/1000-s.live.wall_time>3){$('lag').textContent=`最近源状态 ${Math.round(Date.now()/1000-s.live.wall_time)} 秒前`;$('stream-status').textContent=names[c.status]??c.status;}
 latestClip=s.archives[0]??null;$('replay').disabled=!latestClip;$('archive-count').textContent=`${s.archives.length} 个完整回合`;$('archives').replaceChildren();
 for(const a of s.archives.slice(0,6)){const row=document.createElement('div');row.className='archive';const label=document.createElement('div');label.className='label';label.textContent=`${a.phase?.startsWith('terrain')?'地形第 '+Number(a.source_run.split('/').at(-1).split('_').at(-1))+' 轮':a.phase==='formal_training'?'正式第 '+Number(a.source_run.split('/').at(-1).split('_').at(-1))+' 轮':'采集预检'} · ${terrainNames[a.scenario?.terrain]??'原场景'} · 环境 ${a.environment_index??0} · 回合 ${a.episode}`;row.append(label);for(const [name,url] of [['50FPS WebP',a.webp],['GIF',a.gif],['400Hz轨迹',a.trace],['来源',a.metadata]]){const link=document.createElement('a');link.className='download';link.textContent=name+' ↓';link.href=url;link.download='';row.append(link);}$('archives').append(row);}
 if(overviewData&&Date.now()/1000-overviewData.meta.wall_time>1)$('overview-time').textContent=`${names[c.status]??c.status} · 总览源状态 ${Math.round(Date.now()/1000-overviewData.meta.wall_time)} 秒前 · 不生成替代运动`;
 if(s.final_evaluation){const f=s.final_evaluation.summary??s.final_evaluation.terrain,yaw=f.mean_yaw_score_deg==null?'未完整':f.mean_yaw_score_deg.toFixed(3)+'°';$('final-result').textContent=`独立终评：${f.success_count}/${f.total}成功 · Jψ ${yaw}（未用于选模）`;}
}
async function refresh(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error(r.status);update(await r.json());}catch(e){$('connection').textContent='连接中断，自动重试';}setTimeout(refresh,1500);}
refresh();frames();
let overviewTag=null,overviewPaused=false,overviewData=null,chosenEnvironment=0,gridCols=32;
const canvas=$('worlds'),ctx=canvas.getContext('2d');
async function chooseEnvironment(index){
 if(!Number.isInteger(index)||index<0||index>=(overviewData?.meta.environments??1024))return;
 const r=await fetch('/api/environment',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({environment:index})});
 if(r.ok){chosenEnvironment=index;$('environment').value=index;$('stream-status').textContent=`正在切换环境 ${index}`;mode='live';etag=null;lastKey=null;$('pause').disabled=false;$('pause').textContent='暂停详情';$('replay').textContent='50 FPS 回合录像';if(overviewData)drawWorlds(overviewData);}
}
$('choose').onclick=()=>chooseEnvironment(Number($('environment').value));$('environment').onkeydown=e=>{if(e.key==='Enter')chooseEnvironment(Number(e.target.value));};
canvas.onclick=e=>{const box=canvas.getBoundingClientRect(),rows=Math.ceil((overviewData?.meta.environments??1024)/gridCols);const col=Math.floor((e.clientX-box.left)/box.width*gridCols),row=Math.floor((e.clientY-box.top)/box.height*rows);chooseEnvironment(row*gridCols+col);};
$('pause-overview').onclick=()=>{overviewPaused=!overviewPaused;$('pause-overview').textContent=overviewPaused?'继续总览':'暂停总览';};
function drawWorlds({meta:m,values:v}){
 const width=canvas.clientWidth||1000;gridCols=Math.ceil(Math.sqrt(m.environments));const rows=Math.ceil(m.environments/gridCols),height=Math.min(760,Math.max(220,width*.70));
 const dpr=window.devicePixelRatio||1;if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);canvas.style.height=height+'px';}
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);const cw=width/gridCols,ch=height/rows,scale=Math.min(cw/.65,(ch-5)/.55);
 ctx.lineWidth=.8;ctx.font=Math.min(9,cw/4)+'px sans-serif';
 for(let w=0;w<m.environments;w++){
  const b=w*m.width,x=(w%gridCols)*cw,y=Math.floor(w/gridCols)*ch,reason=v[b+1];
  ctx.fillStyle=reason===0?'#101d29':reason===5&&v[b+2]?'#123c31':'#39291d';ctx.fillRect(x+.5,y+.5,cw-1,ch-1);
  if(w===chosenEnvironment){ctx.strokeStyle='#76edcd';ctx.lineWidth=1.6;ctx.strokeRect(x+1,y+1,cw-2,ch-2);ctx.lineWidth=.8;}
  const root=12+m.root*3,rx=v[b+root],ry=v[b+root+1];
  const project=(px,py,pz)=>[x+cw/2+((px-rx)*.76+(py-ry)*.65)*scale,y+ch-3-pz*scale+((px-rx)*.12-(py-ry)*.14)*scale];
  const at=id=>project(v[b+12+id*3],v[b+13+id*3],v[b+14+id*3]);
  ctx.strokeStyle='#94adc2';ctx.beginPath();for(let body=1;body<m.bodies;body++){const parent=m.parents[body];if(parent<=0)continue;const a=at(parent),c=at(body);ctx.moveTo(...a);ctx.lineTo(...c);}ctx.stroke();
  const rz=v[b+root+2],ax=v[b+3],ay=v[b+6],az=v[b+9],a=project(rx-.14*ax,ry-.14*ay,rz-.14*az),c=project(rx+.14*ax,ry+.14*ay,rz+.14*az);
  ctx.strokeStyle='#78c0ff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...c);ctx.stroke();ctx.lineWidth=.8;
  for(let j=0;j<m.wheels.length;j++){const body=m.wheels[j],point=at(body),rad=m.wheel_radii?.[j]??.05,radius=Math.max(1,rad*scale),axis=12+m.bodies*3+j*3;
   ctx.strokeStyle='#69ddbb';ctx.beginPath();ctx.arc(point[0],point[1],radius,0,Math.PI*2);ctx.stroke();const end=project(v[b+12+body*3]+rad*v[b+axis],v[b+13+body*3]+rad*v[b+axis+1],v[b+14+body*3]+rad*v[b+axis+2]);ctx.beginPath();ctx.moveTo(...point);ctx.lineTo(...end);ctx.stroke();}
  ctx.fillStyle=w===chosenEnvironment?'#bcffee':'#597285';ctx.fillText(String(w),x+2,y+Math.min(9,ch/3));
 }
 $('overview-state').textContent=`${m.environments} / ${m.environments} 个真实世界`;$('overview-time').textContent=`采样 ${num(m.sample_steps)} 步 · 状态距今 ${Math.max(0,Date.now()/1000-m.wall_time).toFixed(1)} s · 点击任意环境查看3D详情`;
 $('environment').max=m.environments-1;
}
async function overviewFrames(){
 if(!overviewPaused){try{const r=await fetch('/api/overview',{headers:overviewTag?{'If-None-Match':overviewTag}:{},cache:'no-cache'});if(r.ok){overviewTag=r.headers.get('ETag');const buffer=await r.arrayBuffer(),length=new DataView(buffer).getUint32(0,true);if(length+4>buffer.byteLength)throw Error('invalid overview');const m=JSON.parse(new TextDecoder().decode(buffer.slice(4,4+length))),v=new Float32Array(buffer,4+length);if(v.length!==m.environments*m.width)throw Error('invalid shape');overviewData={meta:m,values:v};drawWorlds(overviewData);}}catch(e){$('overview-state').textContent='等待真实批量状态';}}
 setTimeout(overviewFrames,100);
}
window.addEventListener('resize',()=>{if(overviewData)drawWorlds(overviewData);});overviewFrames();
