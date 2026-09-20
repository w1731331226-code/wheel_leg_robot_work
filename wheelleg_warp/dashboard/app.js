const $=id=>document.getElementById(id), names={cpu:'CPU · MuJoCo',warp:'GPU · MuJoCo Warp'}, paused={cpu:false,warp:false};
const number=n=>new Intl.NumberFormat('zh-CN').format(n??0);
const duration=s=>s==null?'待估算':s<60?'不足 1 分钟':s<3600?`${Math.round(s/60)} 分钟`:s<86400?`${(s/3600).toFixed(1)} 小时`:`${(s/86400).toFixed(1)} 天`;
const date=t=>new Date(t*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});
const robotIcon='<svg viewBox="0 0 60 55"><path d="M16 36 22 14h16l7 22M22 14l8 20 8-20"/><circle cx="14" cy="42" r="8"/><circle cx="46" cy="42" r="8"/></svg>';
$('robots').innerHTML=['cpu','warp'].map(b=>`<article class="robot ${b}"><div class="robot-head"><div class="robot-title"><span class="backend-icon">${b==='cpu'?'▦':'◈'}</span><div><h3>${names[b]}</h3><small>${b==='cpu'?'原生 CPU 物理':'CUDA 物理 · CPU 控制器'} / M3</small></div></div><span class="status-pill" id="${b}-phase">读取状态</span></div><div class="view"><div class="view-empty" id="${b}-empty">${robotIcon}<span>等待真实训练状态</span></div><img id="${b}-image" src="/live/${b}.mjpg" alt="${names[b]}真实训练环境0的小车画面"><div class="view-top"><span class="live-badge" id="${b}-source">TRAINING STREAM</span><span id="${b}-episode">ENV 0 / 8</span></div><div class="view-bottom"><span id="${b}-sim">仿真时间 —</span><span id="${b}-lag">等待采样</span></div></div><div class="robot-data"><div class="small-grid"><div><span>当前种子 / 策略步</span><strong id="${b}-steps">—</strong></div><div><span>课程阶段</span><strong id="${b}-stage">—</strong></div><div><span>该队列剩余</span><strong id="${b}-eta">—</strong></div></div><div class="bar"><i id="${b}-bar"></i></div><div class="robot-foot"><span id="${b}-budget">0 / 600 万步</span><span id="${b}-score">等待选择集评估</span></div></div><div class="view-actions"><button id="${b}-pause">暂停画面</button><a class="download" id="${b}-download" hidden download>保存最新 GIF ↓</a><small id="${b}-reward">回合奖励 —</small></div></article>`).join('');
for(const b of ['cpu','warp']){
  $(`${b}-image`).onload=()=>$(`${b}-empty`).style.display='none';
  $(`${b}-pause`).onclick=()=>{paused[b]=!paused[b];$(`${b}-pause`).textContent=paused[b]?'继续实时画面':'暂停画面';$(`${b}-image`).src=paused[b]?`/media/live/${b}.jpg?t=${Date.now()}`:`/live/${b}.mjpg?t=${Date.now()}`;};
}
function chart(id,series,success=false){
 const w=620,h=185,l=48,r=16,top=12,bottom=30,colors=['#70adff','#53dbba'];
 const points=series.flatMap(s=>s.points),empty=!points.length;
 const xs=points.map(p=>p.x),ys=points.map(p=>p.y);
 let x0=success?0:Math.min(...xs),x1=Math.max(...xs);if(empty){x0=0;x1=1;}if(x1===x0)x1=x0+1;
 const maxY=success?100:Math.max(2000,...ys)*1.12, X=x=>l+(x-x0)/(x1-x0)*(w-l-r),Y=y=>h-bottom-y/maxY*(h-bottom-top);
 let html='';for(let i=0;i<4;i++){const value=maxY*i/3,y=Y(value);html+=`<line x1="${l}" x2="${w-r}" y1="${y}" y2="${y}" stroke="#283448" stroke-dasharray="3 4"/><text x="${l-8}" y="${y+3}" text-anchor="end" fill="#6e829e" font-size="9">${success?Math.round(value)+'%':value>=10000?(value/10000).toFixed(1)+'万':Math.round(value)}</text>`;}
 for(let j=0;j<series.length;j++){const p=series[j].points;if(!p.length)continue;html+=`<path d="${p.map((p,i)=>`${i?'L':'M'}${X(p.x).toFixed(1)},${Y(p.y).toFixed(1)}`).join(' ')}" stroke="${colors[j]}" stroke-width="2" fill="none"/>`;const last=p[p.length-1];html+=`<circle cx="${X(last.x)}" cy="${Y(last.y)}" r="3" fill="${colors[j]}"/>`;}
 for(let i=0;i<3;i++){const x=x0+(x1-x0)*i/2;html+=`<text x="${X(x)}" y="${h-8}" text-anchor="${i===0?'start':i===2?'end':'middle'}" fill="#6e829e" font-size="9">${success?number(Math.round(x))+'步':new Date(x*1000).toLocaleTimeString('zh-CN',{hour:'2-digit',minute:'2-digit',hour12:false})}</text>`;}
 if(empty)html+=`<text x="335" y="87" text-anchor="middle" fill="#7f94b0" font-size="12">${success?'等待首个完整检查点评估':'正在采集进度历史'}</text>`;
 $(id).innerHTML=html;
}
function update(s){
 if(!s.backends)return;
 $('connection').textContent=s.queue.status==='training'?'双队列观测中':s.queue.status==='completed'?'训练与终评已完成':'已连接本地数据';
 $('connection-dot').style.background='#53dbba';$('updated').textContent=`更新于 ${date(s.now)}`;
 $('total').textContent=`${(s.total_done/s.total_budget*100).toFixed(2)}%`;$('total-bar').style.width=`${s.total_done/s.total_budget*100}%`;
 $('total-caption').textContent=`${number(s.total_done)} / ${number(s.total_budget)} 策略步`;
 $('eta').textContent=duration(s.all_remaining_seconds);
 $('eta-date').textContent=s.all_remaining_seconds==null?'等待有效训练数据':`粗估 ${date(s.now+s.all_remaining_seconds)} · 含终评预留`;
 $('eta-range').textContent=s.all_remaining_seconds==null?'等待双方完成首轮策略更新':`早期区间约 ${duration(s.all_remaining_seconds*.5)}–${duration(s.all_remaining_seconds*2)}`;
 $('eta-date').title='吞吐外推，不是保证完成时间；早期估算误差可达一倍以上。';
 $('rates').textContent=['cpu','warp'].map(b=>s.backends[b].estimate.rate?.toFixed(1)??'—').join(' / ');
 const allArchives=[];
 for(const b of ['cpu','warp']){
  const d=s.backends[b],live=d.live;
  $(`${b}-phase`).textContent={running:'训练进行中',evaluating:'检查点评估中',stopped:'未运行',failed:'需要检查',completed:'已完成'}[d.phase]??d.phase;
  $(`${b}-steps`).textContent=`${d.current.seed} · ${number(d.current.steps)}`;
  $(`${b}-stage`).textContent=d.current.stage?`Stage ${d.current.stage} / 3`:'待开始';
  $(`${b}-eta`).textContent=duration(d.estimate.remaining_seconds);
  $(`${b}-eta`).title=d.estimate.confidence+(d.estimate.range_seconds?`；范围 ${duration(d.estimate.range_seconds[0])}—${duration(d.estimate.range_seconds[1])}`:'');
  $(`${b}-bar`).style.width=`${d.total_steps/d.budget*100}%`;$(`${b}-budget`).textContent=`${number(d.total_steps)} / ${number(d.budget)}`;
  const last=d.selection.at(-1);$(`${b}-score`).textContent=last?`选择集成功率 ${last.success.toFixed(1)}%`:'等待选择集评估';
  if(live){
   $(`${b}-source`).textContent=live.smoke?'预检训练原始画面':'LIVE · 真实训练采样';
   $(`${b}-episode`).textContent=`ENV 0 / 8 · SEED ${live.source_run.split('_').at(-1)} · EP ${live.episode}`;
   $(`${b}-sim`).textContent=`仿真 ${live.simulation_seconds.toFixed(2)} s · 采样 ${number(live.sample_steps)} 步`;
   const lag=Math.max(0,Date.now()/1000-live.wall_time);$(`${b}-lag`).textContent=lag>15?(d.phase==='evaluating'?'评估期间暂停采样':`画面距今 ${Math.round(lag)} 秒`):`最近状态 ${lag.toFixed(1)} 秒前`;
   $(`${b}-reward`).textContent=`回合奖励 ${live.cumulative_reward.toFixed(2)}`;
  }
  const clips=d.archives.filter(x=>x.status==='completed').sort((a,b)=>(b.archive_time??0)-(a.archive_time??0));
  if(clips.length){const a=$(`${b}-download`);a.hidden=false;a.href=clips[0].gif;}
  allArchives.push(...clips.map(x=>({...x,backend:b})));
 }
 chart('progress-chart',['cpu','warp'].map(b=>({points:s.history.map(h=>({x:h.time,y:h[b]}))})));
 chart('success-chart',['cpu','warp'].map(b=>({points:s.backends[b].selection.map(h=>({x:h.step,y:h.success}))})),true);
 allArchives.sort((a,b)=>(b.archive_time??0)-(a.archive_time??0));
 $('archive-count').textContent=`${allArchives.length} 个已生成录像`;
 $('archive-list').replaceChildren();
 for(const a of allArchives.slice(0,6)){
  const row=document.createElement('div');row.className='archive';
  const label=document.createElement('div');label.className='label';label.textContent=`${a.backend==='cpu'?'CPU':'GPU'} · ${a.smoke?'预检':'正式训练'} · 回合 ${a.episode}`;
  const small=document.createElement('small');small.textContent=`${a.frames??'—'} 帧状态 · ${date(a.archive_time??a.finished??a.started)}`;label.append(small);row.append(label);
  for(const [text,url] of [['GIF',a.gif],['轨迹 NPZ',a.trace],['配置',a.metadata]]){const link=document.createElement('a');link.className='download';link.textContent=text+' ↓';link.href=url;link.download='';row.append(link);}
  $('archive-list').append(row);
 }
 if(!allArchives.length)$('archive-list').textContent='等待第一个完整回合；实时画面和原始状态采集会先开始。';
 $('local-path').textContent=`原始轨迹：${s.run_directory}/<backend_seed>/live/episodes/ · GIF：wheelleg_warp/dashboard/local_data/captures/`;
}
async function refresh(){try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error('HTTP '+response.status);update(await response.json());}catch(e){$('connection').textContent='连接中断 · 自动重试';$('connection-dot').style.background='#f46c74';}}
refresh();setInterval(refresh,5000);
