import torch
from . import utils
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import scipy.ndimage as filters
from os.path import join
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        return getattr(self.module, name)


#######################################
#      for IRL models (new state)
#######################################
class RolloutStorage_New(object):
    def __init__(self, trajs_all, shuffle=True):
        self.obs_fovs = torch.cat([traj['state'] for traj in trajs_all])
        self.actions = torch.cat([traj['action'] for traj in trajs_all])
        self.tids = torch.cat([traj['task_id'] for traj in trajs_all])
        self.lprobs = torch.cat([traj['log_prob'] for traj in trajs_all])
        self.returns = torch.cat([traj['return']
                                  for traj in trajs_all]).view(-1)
        advs = torch.stack([traj['advantage'] for traj in trajs_all])
        print(advs.size())
        self.advs = (advs -
                     advs.mean()) / advs.std()  # normalize for stability
        self.advs = self.advs.view(-1)

        self.sample_num = self.obs_fovs.size(0)
        self.shuffle = shuffle

    def get_generator(self, minibatch_size):
        perm = torch.randperm(
            self.sample_num) if self.shuffle else torch.arange(self.sample_num)
        for start_ind in range(0, self.sample_num, minibatch_size):
            ind = perm[start_ind:start_ind + minibatch_size]
            obs_fov_batch = self.obs_fovs[ind]
            actions_batch = self.actions[ind]
            tids_batch = self.tids[ind]
            return_batch = self.returns[ind]
            log_probs_batch = self.lprobs[ind]
            advantage_batch = self.advs[ind]

            yield (
                obs_fov_batch, tids_batch
            ), actions_batch, return_batch, log_probs_batch, advantage_batch


class LHF_IRL(Dataset):
    """
    Image data for training generator
    """

    def __init__(self, DCB_HR_dir, DCB_LR_dir, initial_fix, img_info, annos,
                 pa, catIds):
        self.img_info = img_info
        self.annos = annos
        self.pa = pa
        self.initial_fix = initial_fix
        self.catIds = catIds
        self.LR_dir = DCB_LR_dir
        self.HR_dir = DCB_HR_dir

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, idx):
        cat_name, img_name = self.img_info[idx].split('_')
        feat_name = img_name[:-3] + 'pth.tar'
        lr_path = join(self.LR_dir, cat_name.replace(' ', '_'), feat_name)
        hr_path = join(self.HR_dir, cat_name.replace(' ', '_'), feat_name)
        lr = torch.load(lr_path)
        hr = torch.load(hr_path)
        imgId = cat_name + '_' + img_name

        # update state with initial fixation
        init_fix = self.initial_fix[imgId]
        px, py = init_fix
        px, py = px * lr.size(-1), py * lr.size(-2)
        mask = utils.foveal2mask(px, py, self.pa.fovea_radius, hr.size(-2),
                                 hr.size(-1))
        mask = torch.from_numpy(mask)
        mask = mask.unsqueeze(0).repeat(hr.size(0), 1, 1)
        lr = (1 - mask) * lr + mask * hr

        # history fixation map
        history_map = torch.zeros((hr.size(-2), hr.size(-1)))
        history_map = (1 - mask[0]) * history_map + mask[0] * 1

        # action mask
        action_mask = torch.zeros((self.pa.patch_num[1], self.pa.patch_num[0]),
                                  dtype=torch.uint8)
        px, py = init_fix
        px, py = int(px * self.pa.patch_num[0]), int(py * self.pa.patch_num[1])
        action_mask[py - self.pa.IOR_size:py + self.pa.IOR_size + 1, px -
                    self.pa.IOR_size:px + self.pa.IOR_size + 1] = 1

        # target location label
        coding = utils.multi_hot_coding(self.annos[imgId], self.pa.patch_size,
                                        self.pa.patch_num)
        coding = torch.from_numpy(coding).view(1, -1)

        return {
            'task_id': self.catIds[cat_name],
            'img_name': img_name,
            'cat_name': cat_name,
            'lr_feats': lr,
            'hr_feats': hr,
            'history_map': history_map,
            'init_fix': torch.FloatTensor(init_fix),
            'label_coding': coding,
            'action_mask': action_mask
        }


class LHF_Human_Gaze(Dataset):
    """
    Human gaze data for training discriminator
    """

    def __init__(self,
                 DCB_HR_dir,
                 DCB_LR_dir,
                 fix_labels,
                 annos,
                 pa,
                 catIds,
                 blur_action=False):
        self.pa = pa
        self.fix_labels = fix_labels
        self.annos = annos
        self.catIds = catIds
        self.LR_dir = DCB_LR_dir
        self.HR_dir = DCB_HR_dir
        self.blur_action = blur_action

    def __len__(self):
        return len(self.fix_labels)

    def __getitem__(self, idx):
        # load low- and high-res beliefs
        img_name, cat_name, fixs, action = self.fix_labels[idx]
        feat_name = img_name[:-3] + 'pth.tar'
        lr_path = join(self.LR_dir, cat_name.replace(' ', '_'), feat_name)
        hr_path = join(self.HR_dir, cat_name.replace(' ', '_'), feat_name)
        state = torch.load(lr_path)
        hr = torch.load(hr_path)

        # construct DCB
        remap_ratio = self.pa.im_w / float(hr.size(-1))
        history_map = torch.zeros((hr.size(-2), hr.size(-1)))
        for i in range(len(fixs)):
            px, py = fixs[i]
            px, py = px / remap_ratio, py / remap_ratio
            mask = utils.foveal2mask(px, py, self.pa.fovea_radius, hr.size(-2),
                                     hr.size(-1))
            mask = torch.from_numpy(mask)
            mask = mask.unsqueeze(0).repeat(hr.size(0), 1, 1)
            state = (1 - mask) * state + mask * hr
            history_map = (1 - mask[0]) * history_map + mask[0] * 1

        # create labels
        imgId = cat_name + '_' + img_name
        coding = utils.multi_hot_coding(self.annos[imgId], self.pa.patch_size,
                                        self.pa.patch_num)
        coding = torch.from_numpy(coding / coding.sum()).view(1, -1)

        ret = {
            "task_id": self.catIds[cat_name],
            "true_state": state,
            "true_action": torch.tensor([action], dtype=torch.long),
            'label_coding': coding,
            'history_map': history_map,
            'img_name': img_name,
            'task_name': cat_name
        }
        
        # blur action maps for evaluation
        if self.blur_action:
            action_map = np.zeros(self.pa.patch_count, dtype=np.float32)
            action_map[action] = 1
            action_map = action_map.reshape(self.pa.patch_num[1], -1)
            action_map = filters.gaussian_filter(action_map, sigma=1)
            ret['action_map'] = action_map
        return ret


class RolloutStorage(object):
    def __init__(self, trajs_all, shuffle=True, norm_adv=False):
        self.obs_fovs = torch.cat([traj["curr_states"] for traj in trajs_all])
        self.actions = torch.cat([traj["actions"] for traj in trajs_all])
        self.lprobs = torch.cat([traj['log_probs'] for traj in trajs_all])
        self.tids = torch.cat([traj['task_id'] for traj in trajs_all])
        self.returns = torch.cat([traj['acc_rewards']
                                  for traj in trajs_all]).view(-1)
        self.advs = torch.cat([traj['advantages']
                               for traj in trajs_all]).view(-1)
        if norm_adv:
            self.advs = (self.advs - self.advs.mean()) / (self.advs.std() +
                                                          1e-8)

        self.sample_num = self.obs_fovs.size(0)
        self.shuffle = shuffle

    def get_generator(self, minibatch_size):
        minibatch_size = min(self.sample_num, minibatch_size)
        sampler = BatchSampler(SubsetRandomSampler(range(self.sample_num)),
                               minibatch_size,
                               drop_last=True)
        for ind in sampler:
            obs_fov_batch = self.obs_fovs[ind]
            actions_batch = self.actions[ind]
            tids_batch = self.tids[ind]
            return_batch = self.returns[ind]
            log_probs_batch = self.lprobs[ind]
            advantage_batch = self.advs[ind]

            yield (
                obs_fov_batch, tids_batch
            ), actions_batch, return_batch, log_probs_batch, advantage_batch


class FakeDataRollout(object):
    def __init__(self, trajs_all, minibatch_size, shuffle=True):
        self.GS = torch.cat([traj['curr_states'] for traj in trajs_all])
        self.GA = torch.cat([traj['actions']
                             for traj in trajs_all]).unsqueeze(1)
        self.tids = torch.cat([traj['task_id'] for traj in trajs_all])
        self.GP = torch.exp(
            torch.cat([traj["log_probs"] for traj in trajs_all])).unsqueeze(1)
        # self.GIOR = torch.cat([traj["IORs"]
        #                        for traj in trajs_all]).unsqueeze(1)

        self.sample_num = self.GS.size(0)
        self.shuffle = shuffle
        self.batch_size = min(minibatch_size, self.sample_num)

    def __len__(self):
        return int(self.sample_num // self.batch_size)

    def get_generator(self):
        sampler = BatchSampler(SubsetRandomSampler(range(self.sample_num)),
                               self.batch_size,
                               drop_last=True)
        for ind in sampler:
            GS_batch = self.GS[ind]
            tid_batch = self.tids[ind]
            GA_batch = self.GA[ind]
            GP_batch = self.GP[ind]
            # GIOR_batch = self.GIOR[ind]

            yield GS_batch, GA_batch, GP_batch, tid_batch
