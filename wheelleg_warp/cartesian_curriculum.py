"""Post-curriculum Cartesian reset; does not admit or construct a training adapter."""
import numpy as np
import warp as wp
from train_height_comparison import CurriculumEnv
from cartesian_pair_kernel import reset_aux


def refresh_after_curriculum(raw,observation,done):
    obs=np.asarray(observation);mask=np.asarray(done)
    if obs.shape!=(raw.num_envs,38) or mask.shape!=(raw.num_envs,) or mask.dtype!=np.bool_:
        raise ValueError('Expected returned public38 and Boolean per-world done mask')
    if not np.isfinite(obs).all():raise ValueError('Nonfinite public reset packet')
    if not mask.any():return
    buffers=raw._cartesian_buffers
    # Parent installs the new bank after raw auto-reset; its returned packet owns the next episode.
    np.testing.assert_array_equal(raw.obs.numpy()[mask],obs[mask])
    buffers['mask'].assign(mask.astype(np.int32))
    wp.launch(reset_aux,raw.num_envs,[buffers['mask'],raw.obs,buffers['latent'],buffers['shadow'],buffers['reference']])


class CartesianCurriculum(CurriculumEnv):
    def step_wait(self):
        result=super().step_wait()
        refresh_after_curriculum(self.venv,result[0],result[2])
        return result
