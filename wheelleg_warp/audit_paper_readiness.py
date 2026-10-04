"""Final scope-preserving readiness audit: material for formal writing, not acceptance."""
from pathlib import Path
import json,re,hashlib

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'wheelleg_warp/results/paper_recovery_20261004'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()

def run():
    manuscript=ROOT/'wheelleg_warp/paper.tex';tex=manuscript.read_text()
    required=['研究问题与主张边界','相关工作与实验区别','模型、控制与执行契约','可得观测与完整任务',
        '奖励变换与价值估计边界','完成的有限学习预试验','独立固定模型通道确认协议','已完成的独立固定模型确认',
        '统计单位与可复現性','讨论与适用范围','结论','复现材料入口']
    assert all(r'\section{'+name+'}' in tex for name in required)
    assert r'\begin{abstract}' in tex and r'\appendix' in tex
    stack=[]
    for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',tex):
        if kind=='begin':stack.append(name)
        else:assert stack and stack.pop()==name
    assert not stack and tex.count('{')==tex.count('}')
    refs=set(re.findall(r'\\bibitem\{([^}]+)\}',tex));cites={v for c in re.findall(r'\\cite\{([^}]+)\}',tex) for v in c.split(',')}
    assert cites<=refs and {'residual','action','ng','hprs','hybrid','crrl'}<=refs
    pilot=json.loads((P/'reward_pilot_v1/pair_review.json').read_text());assert pilot['verified'] and pilot['completed_runs']==6 and pilot['completed_seed_pairs']==3
    assert pilot['observed_sampled_budget']==1200000 and pilot['continuation_gate'] is False
    for name,value in pilot['artifact_sha256'].items():assert sha(P/'reward_pilot_v1'/name)==value
    independent=P/'channel_validation_v1';review=json.loads((independent/'independent_review.json').read_text())
    table=json.loads((independent/'table_report.json').read_text());assert review['complete'] and table['complete']
    assert review['verified_case_records']==4472 and review['verified_jobs']==78 and len(review['available_primary_model_effects'])==6
    assert table['regular_primary']['mean_success_difference_pp']==62.5
    assert sha(independent/'reviewed_jobs_snapshot.json')==review['ledger_sha256']
    assert sha(independent/'verified_tables.csv')==table['table_sha256']
    manifest=json.loads((independent/'completed_jobs.json').read_text())
    for r in manifest['records']:assert sha(independent/r['path'])==r['sha256']
    models=['original_1609','potential_1609','original_1610','potential_1610','original_1611','potential_1611']
    expected_rows=[]
    for model in models:
        values=[]
        for group in ['regular','pressure','legacy_regression']:
            for mask in ['all','legs_only']:values.append(review['decoded'][f'{model}_{mask}/{group}']['summary']['success_count'])
        label=model.replace('original_','Raw ').replace('potential_','Phi ')
        expected_rows.append(label+' & '+' & '.join(map(str,values))+r'\\')
    assert all(row in tex for row in expected_rows)
    assert '62.5' in tex and '180/576' in tex and '540/576' in tex and '81/288' in tex and '210/288' in tex and '97/168' in tex and '167/168' in tex
    periods=[]
    for round_id in [90,95,100,105,110]:
        report=P/f'round{round_id}_direction_review.json';cleanup=P/f'round{round_id}_cleanup.json'
        a=json.loads(report.read_text());b=json.loads(cleanup.read_text());assert a['round']==b['round']==round_id
        assert b['files'];periods.append(dict(round=round_id,review_sha256=sha(report),cleanup_sha256=sha(cleanup),items=len(b['files'])))
    figures=[P/'reward_pilot_v1/three_seed_pilot.png',P/'reward_pilot_v1/failure_structure.png',
        P/'reward_pilot_v1/contact_trajectory_1610/lateral_position.png',independent/'independent_channels.png']
    assert all(f.is_file() and f.stat().st_size>0 for f in figures)
    criteria=[
        dict(requirement='可检验主张及相对文献贡献边界',achieved=True,evidence='明确限定系统执行/任务/通道审计；六固定模型独立主效果一致，已核RRL、动作表示、HybridLMC、CRRL、PBRS/HPRS；非新算法/排他性穷尽证明/普遍理论'),
        dict(requirement='正确模型、信息、输出与约束契约',achieved=True,evidence='原2kHz/50Hz/38混合时序/真实链与扭矩/完整success源合同通过，草稿区分设计/物理/任务，无未经核全局状态安全或Markov宣称'),
        dict(requirement='强经典及同信息同预算至少3学习种子',achieved=True,evidence='六matched200k三seed/初始hash/400epochs8000Adam完整，强B0/B1和失败都报告；mask干预不是同权限学习方法优势'),
        dict(requirement='新独立、压力和旧能力证据',achieved=True,evidence='4472/78事前模型场景源冻结，fresh96/pressure48/old28分列，22early保留，未触旧64/3000，旧28非新独立'),
        dict(requirement='受控机制证据',achieved=True,evidence='六固定模型wheel关闭的闭环作用独立复现，设计权衡完整；body漂移仅关联，不称训练失败唯一原因或MC值校准'),
        dict(requirement='足以正式写作的方法、图表、统计、失败和复现材料',achieved=True,evidence='完整中文初稿/方法推导/参数表/开发与独立表/六模型三seed簇/四可复现图/原模型源码hash和周期审计，未以负摘要代替；作者期刊语言后续编辑')]
    result=dict(objective='继续直到可以写论文，每五轮深评方向调整与确认冗余清理',
        ready_to_begin_formal_paper_writing=all(c['achieved'] for c in criteria),publication_or_acceptance_guaranteed=False,
        manuscript_complete_initial_draft=True,scope='Limited simulation empirical audit, not PPO superiority/control innovation/global safety/sim2real result',
        criteria=criteria,periodic_reviews=periods,manuscript_sha256=sha(manuscript),
        core_evidence_sha256={'pilot':sha(P/'reward_pilot_v1/pair_review.json'),'independent':sha(independent/'independent_review.json'),
            'tables':sha(independent/'verified_tables.csv'),'source_contract':sha(independent/'source_contract.json')},
        figures=[dict(path=str(f.relative_to(ROOT)),sha256=sha(f)) for f in figures],
        typesetting_limit='Built-in compile attempt failed because Codex sandbox executable missing, source environment/braces/citations checked but PDF compilation and rendered layout unverified. Compiler repair is external to materials-readiness objective.',
        excluded='Submission, acceptance, real hardware trials, algorithmic firstness, new long training or guaranteed safety are not achieved or asserted',source_sha256=sha(Path(__file__)))
    (P/'paper_readiness_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('READINESS',result['ready_to_begin_formal_paper_writing'],'all6 requirements, periodic90/95/100/105/110, source/tables/models preserved; PDF unverified')

if __name__=='__main__':run()
