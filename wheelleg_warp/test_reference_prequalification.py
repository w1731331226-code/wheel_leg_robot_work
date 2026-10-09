"""Checkpoint schedule and absent-admission fail closed, without robot/learning."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import train_reference_prequalification as study


def run():
    parent=study.Ledger.save;oldout=study.OUT;saved=[]
    with TemporaryDirectory() as name:
        directory=Path(name)
        def save(self):
            saved.append(self.model.num_timesteps)
            (self.directory/f'step_{self.model.num_timesteps}.json').write_text('{"engineering_only":true}')
        study.Ledger.save=save
        try:
            ledger=study.StudyLedger.__new__(study.StudyLedger);ledger.directory=directory
            ledger.model=SimpleNamespace(num_timesteps=0)
            for step in range(5000,200001,5000):
                ledger.model.num_timesteps=step;ledger.save()
            assert saved==list(range(20000,200001,20000))
            for step in saved:
                record=json.loads((directory/f'step_{step}.json').read_text())
                assert record['prequalification_only'] and not record['engineering_only']
            (directory/'proposal.json').write_text('{}');study.OUT=directory
            try:study.verify()
            except FileNotFoundError as error:assert Path(error.filename).name=='main_source_admission.json'
            else:raise AssertionError('Missing source admission admitted')
        finally:study.Ledger.save=parent;study.OUT=oldout
    print('PASS322 checkpoints20k…200k only,prequalification labels,missing admission blocks;0physics/learning')


if __name__=='__main__':run()
