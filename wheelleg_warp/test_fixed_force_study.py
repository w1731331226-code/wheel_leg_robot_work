"""Receipt tests for checkpoint timing, world identity and fail-closed admission."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import train_fixed_force_study as study


def run():
    old=study.Ledger.save;oldout=study.OUT;saved=[]
    with TemporaryDirectory() as name:
        directory=Path(name)
        def save(self):
            saved.append(self.model.num_timesteps);(self.directory/f'step_{self.model.num_timesteps}.json').write_text('{"engineering_only":true}')
        study.Ledger.save=save
        try:
            ledger=study.StudyLedger.__new__(study.StudyLedger);ledger.directory=directory;ledger.raw=SimpleNamespace(_joint_guard_stats={'valid_substeps':5});ledger.curriculum=SimpleNamespace(stages=np.array([1,2,3]))
            ledger.model=SimpleNamespace(num_timesteps=0)
            for step in range(5000,200001,5000):ledger.model.num_timesteps=step;ledger.save()
            assert saved==list(range(20000,200001,20000))
            row=json.loads((directory/'step_200000.json').read_text());assert row['prequalification_only'] and not row['engineering_only'] and row['guard']['valid_substeps']==5
            ledger.episodes=[];ledger.locals={'infos':[{},dict(episode={'l':1},physical_safety_passed=True,design_joint_passed=True)]};assert ledger._on_step();assert ledger.episodes[0]['world_index']==1
            ledger.locals={'infos':[dict(episode={'l':1},physical_safety_passed=True,design_joint_passed=False)]}
            try:ledger._on_step()
            except AssertionError:pass
            else:raise AssertionError('Unsafe completed episode accepted')
            assert ledger.episodes[-1]['world_index']==0
            study.OUT=directory;(directory/'proposal.json').write_text('{}')
            try:study.run()
            except FileNotFoundError as error:assert Path(error.filename).name=='source_admission.json'
            else:raise AssertionError('Missing admission allowedrun')
            assert not (directory/'training').exists()
        finally:study.Ledger.save=old;study.OUT=oldout
    print('PASS334 postupdate20k checkpoints/guard-stage ledger/world-index failurekept/missing admission blocksbefore output;0physics/learning')


if __name__=='__main__':run()
