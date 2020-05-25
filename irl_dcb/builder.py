import torch
from .environment import IRL_Env4LHF
from .utils import load
from .models import LHF_Discriminator_Cond, LHF_Policy_Cond_Small


def build(hparams, is_training, device, catIds, load_path=None):

    # build model
    input_size = 134  # number of belief maps
    task_eye = torch.eye(len(catIds)).to(device)
    discriminator = LHF_Discriminator_Cond(
        hparams.Data.patch_count, len(catIds), task_eye,
        input_size).to(device)
    generator = LHF_Policy_Cond_Small(hparams.Data.patch_count,
                                      len(catIds), task_eye,
                                      input_size).to(device)

    if load_path:
        load('best', generator, 'generator', pkg_dir=load_path)
        global_step = load('best',
                           discriminator,
                           'discriminator',
                           pkg_dir=load_path)
    else:
        global_step = 0

    # build IRL environment
    env_train = IRL_Env4LHF(hparams.Data,
                            max_step=hparams.Data.max_traj_length,
                            mask_size=hparams.Data.IOR_size,
                            status_update_mtd=hparams.Train.stop_criteria,
                            device=device,
                            inhibit_return=True)
    env_valid = IRL_Env4LHF(hparams.Data,
                            max_step=hparams.Data.max_traj_length,
                            mask_size=hparams.Data.IOR_size,
                            status_update_mtd=hparams.Train.stop_criteria,
                            device=device,
                            inhibit_return=True)

    return {
        'env': {
            'train': env_train,
            'valid': env_valid
        },
        'model': {
            'gen': generator,
            'disc': discriminator
        },
        'loaded_step': global_step
    }
