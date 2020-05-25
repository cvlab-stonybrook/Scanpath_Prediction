import numpy as np
import torch
from copy import copy
from torch.distributions import Categorical
import warnings
from .data import LHF_IRL, LHF_Human_Gaze
import os
import re
from shutil import copyfile
warnings.filterwarnings("ignore", category=UserWarning)


def process_data(target_trajs,
                 DCB_HR_dir,
                 DCB_LR_dir,
                 target_annos,
                 hparams,
                 is_testing=False):
    target_init_fixs = {}
    for traj in target_trajs:
        key = traj['task'] + '_' + traj['name']
        target_init_fixs[key] = (traj['X'][0] / hparams.Data.im_w,
                                 traj['Y'][0] / hparams.Data.im_h)
    cat_names = list(np.unique([x['task'] for x in target_trajs]))
    catIds = dict(zip(cat_names, list(range(len(cat_names)))))

    if is_testing:
        # testing fixation data
        test_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        assert len(test_target_trajs) > 0, 'no testing data found!'
        test_task_img_pair = np.unique(
            [traj['task'] + '_' + traj['name'] for traj in test_target_trajs])
        human_mean_cdf, _ = compute_search_cdf(test_target_trajs, target_annos,
                                               hparams.Data.max_traj_length)
        print('target fixation prob (test).:', human_mean_cdf)

        # load image data
        test_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                   test_task_img_pair, target_annos,
                                   hparams.Data, catIds)
        return {
            'catIds': catIds,
            'img_test': test_img_dataset,
            'human_mean_cdf': human_mean_cdf,
            'bbox_annos': target_annos,
            'gt_scanpaths': test_target_trajs
        }

    else:
        # training fixation data
        train_target_trajs = list(
            filter(lambda x: x['split'] == 'train', target_trajs))
        train_task_img_pair = np.unique(
            [traj['task'] + '_' + traj['name'] for traj in train_target_trajs])
        train_fix_labels = preprocess_fixations(
            train_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            truncate_num=hparams.Data.max_traj_length)

        # validation fixation data
        valid_target_trajs = list(
            filter(lambda x: x['split'] == 'test', target_trajs))
        valid_task_img_pair = np.unique(
            [traj['task'] + '_' + traj['name'] for traj in valid_target_trajs])
        human_mean_cdf, _ = compute_search_cdf(valid_target_trajs,
                                               target_annos,
                                               hparams.Data.max_traj_length)
        print('target fixation prob (valid).:', human_mean_cdf)
        valid_fix_labels = preprocess_fixations(
            valid_target_trajs,
            hparams.Data.patch_size,
            hparams.Data.patch_num,
            hparams.Data.im_h,
            hparams.Data.im_w,
            truncate_num=hparams.Data.max_traj_length)

        # load image data
        train_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                    train_task_img_pair, target_annos,
                                    hparams.Data, catIds)
        valid_img_dataset = LHF_IRL(DCB_HR_dir, DCB_LR_dir, target_init_fixs,
                                    valid_task_img_pair, target_annos,
                                    hparams.Data, catIds)

        # load human gaze data
        train_HG_dataset = LHF_Human_Gaze(DCB_HR_dir, DCB_LR_dir,
                                          train_fix_labels, target_annos,
                                          hparams.Data, catIds)
        valid_HG_dataset = LHF_Human_Gaze(DCB_HR_dir,
                                          DCB_LR_dir,
                                          valid_fix_labels,
                                          target_annos,
                                          hparams.Data,
                                          catIds,
                                          blur_action=True)

        return {
            'catIds': catIds,
            'img_train': train_img_dataset,
            'img_valid': valid_img_dataset,
            'gaze_train': train_HG_dataset,
            'gaze_valid': valid_HG_dataset,
            'human_mean_cdf': human_mean_cdf,
            'bbox_annos': target_annos
        }


def cutFixOnTarget(trajs, target_annos):
    task_names = np.unique([traj['task'] for traj in trajs])
    if 'condition' in trajs[0].keys():
        trajs = list(filter(lambda x: x['condition'] == 'present', trajs))
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]
            traj_len = get_num_step2target(traj['X'], traj['Y'], bbox)
            num_steps_task[i] = traj_len
            traj['X'] = traj['X'][:traj_len]
            traj['Y'] = traj['Y'][:traj_len]


def pos_to_action(center_x, center_y, patch_size, patch_num):
    x = center_x // patch_size[0]
    y = center_y // patch_size[1]

    return int(patch_num[0] * y + x)


def action_to_pos(acts, patch_size, patch_num):
    patch_y = acts // patch_num[0]
    patch_x = acts % patch_num[0]

    pixel_x = patch_x * patch_size[0] + patch_size[0] / 2
    pixel_y = patch_y * patch_size[1] + patch_size[1] / 2
    return pixel_x, pixel_y


def select_action(obs, policy, sample_action, action_mask=None,
                  softmask=False):
    probs, values = policy(*obs)
    if sample_action:
        m = Categorical(probs)
        if action_mask is not None:
            # prevent sample previous actions by re-normalizing probs
            probs_new = probs.clone().detach()
            if softmask:
                probs_new = probs_new * action_mask
            else:
                probs_new[action_mask] = 0
            probs_new /= probs_new.sum(dim=1).view(probs_new.size(0), 1)
            m_new = Categorical(probs_new)
            actions = m_new.sample()
        else:
            actions = m.sample()
        log_probs = m.log_prob(actions)
        return actions.view(-1), log_probs, values.view(-1), probs
    else:
        probs_new = probs.clone().detach()
        probs_new[action_mask.view(probs_new.size(0), -1)] = 0
        actions = torch.argmax(probs_new, dim=1)
        return actions.view(-1), None, None, None


def collect_trajs(env,
                  policy,
                  patch_num,
                  max_traj_length,
                  is_eval=False,
                  sample_action=True):

    rewards = []
    obs_fov = env.observe()
    act, log_prob, value, prob = select_action((obs_fov, env.task_ids),
                                               policy,
                                               sample_action,
                                               action_mask=env.action_mask)
    status = [env.status]
    values = [value]
    log_probs = [log_prob]
    SASPs = []

    i = 0
    if is_eval:
        actions = []
        while i < max_traj_length:
            new_obs_fov, curr_status = env.step(act)
            status.append(curr_status)
            actions.append(act)
            obs_fov = new_obs_fov
            act, log_prob, value, prob_new = select_action(
                (obs_fov, env.task_ids),
                policy,
                sample_action,
                action_mask=env.action_mask)
            i = i + 1

        trajs = {
            'status': torch.stack(status),
            'actions': torch.stack(actions)
        }

    else:
        IORs = []
        IORs.append(
            env.action_mask.to(dtype=torch.float).view(env.batch_size, 1,
                                                       patch_num[1], -1))
        while i < max_traj_length and env.status.min() < 1:
            new_obs_fov, curr_status = env.step(act)

            status.append(curr_status)
            SASPs.append((obs_fov, act, new_obs_fov))
            obs_fov = new_obs_fov

            IORs.append(
                env.action_mask.to(dtype=torch.float).view(
                    env.batch_size, 1, patch_num[1], -1))

            act, log_prob, value, prob_new = select_action(
                (obs_fov, env.task_ids),
                policy,
                sample_action,
                action_mask=env.action_mask)
            values.append(value)
            log_probs.append(log_prob)

            rewards.append(torch.zeros(env.batch_size))

            i = i + 1

        S = torch.stack([sasp[0] for sasp in SASPs])
        A = torch.stack([sasp[1] for sasp in SASPs])
        V = torch.stack(values)
        R = torch.stack(rewards)
        LogP = torch.stack(log_probs[:-1])
        status = torch.stack(status[1:])

        bs = len(env.img_names)
        trajs = []

        for i in range(bs):
            ind = (status[:, i] == 1).to(torch.int8).argmax().item() + 1
            if status[:, i].sum() == 0:
                ind = status.size(0)
            trajs.append({
                'curr_states': S[:ind, i],
                'actions': A[:ind, i],
                'values': V[:ind + 1, i],
                'log_probs': LogP[:ind, i],
                'rewards': R[:ind, i],
                'task_id': env.task_ids[i].repeat(ind),
                'img_name': [env.img_names[i]] * ind,
                'length': ind
            })

    return trajs


def compute_return_advantage(rewards, values, gamma, mtd='CRITIC', tau=0.96):
    device = rewards.device
    acc_reward = torch.zeros_like(rewards, dtype=torch.float, device=device)
    acc_reward[-1] = rewards[-1]
    for i in reversed(range(acc_reward.size(0) - 1)):
        acc_reward[i] = rewards[i] + gamma * acc_reward[i + 1]

    # compute advantages
    if mtd == 'MC':  # Monte-Carlo estimation
        advs = acc_reward - values[:-1]
    elif mtd == 'CRITIC':  # critic estimation
        advs = rewards + gamma * values[1:] - values[:-1]
    elif mtd == 'GAE':  # generalized advantage estimation
        delta = rewards + gamma * values[1:] - values[:-1]
        adv = torch.zeros_like(delta, dtype=torch.float, device=device)
        adv[-1] = delta[-1]
        for i in reversed(range(delta.size(0) - 1)):
            adv[i] = delta[i] + gamma * tau * adv[i + 1]
    else:
        raise NotImplementedError

    return acc_reward.squeeze(), advs.squeeze()


def process_trajs(trajs, gamma, mtd='CRITIC', tau=0.96):
    # compute discounted cummulative reward
    device = trajs[0]['log_probs'].device
    avg_return = 0
    for traj in trajs:

        acc_reward = torch.zeros_like(traj['rewards'],
                                      dtype=torch.float,
                                      device=device)
        acc_reward[-1] = traj['rewards'][-1]
        for i in reversed(range(acc_reward.size(0) - 1)):
            acc_reward[i] = traj['rewards'][i] + gamma * acc_reward[i + 1]

        traj['acc_rewards'] = acc_reward
        avg_return += acc_reward[0]

        values = traj['values']
        # compute advantages
        if mtd == 'MC':  # Monte-Carlo estimation
            traj['advantages'] = traj['acc_rewards'] - values[:-1]

        elif mtd == 'CRITIC':  # critic estimation
            traj['advantages'] = traj[
                'rewards'] + gamma * values[1:] - values[:-1]

        elif mtd == 'GAE':  # generalized advantage estimation
            delta = traj['rewards'] + gamma * values[1:] - values[:-1]
            adv = torch.zeros_like(delta, dtype=torch.float, device=device)
            adv[-1] = delta[-1]
            for i in reversed(range(delta.size(0) - 1)):
                adv[i] = delta[i] + gamma * tau * adv[i + 1]
            traj['advantages'] = adv
        else:
            raise NotImplementedError

    return avg_return / len(trajs)


def get_num_step2target(X, Y, bbox):
    on_target_X = np.logical_and(X > bbox[0], X < bbox[0] + bbox[2])
    on_target_Y = np.logical_and(Y > bbox[1], Y < bbox[1] + bbox[3])
    on_target = np.logical_and(on_target_X, on_target_Y)
    if np.sum(on_target) > 0:
        first_on_target_idx = np.argmax(on_target)
        return first_on_target_idx + 1
    else:
        return 1000  # some big enough number


def get_CDF(num_steps, max_step):
    cdf = np.zeros(max_step)
    total = float(len(num_steps))
    for i in range(1, max_step + 1):
        cdf[i - 1] = np.sum(num_steps <= i) / total
    return cdf


def get_num_steps(trajs, target_annos, task_names):
    num_steps = {}
    for task in task_names:
        task_trajs = list(filter(lambda x: x['task'] == task, trajs))
        num_steps_task = np.ones(len(task_trajs), dtype=np.uint8)
        for i, traj in enumerate(task_trajs):
            key = traj['task'] + '_' + traj['name']
            bbox = target_annos[key]
            step_num = get_num_step2target(traj['X'], traj['Y'], bbox)
            num_steps_task[i] = step_num
            traj['X'] = traj['X'][:step_num]
            traj['Y'] = traj['Y'][:step_num]
        num_steps[task] = num_steps_task
    return num_steps


def get_mean_cdf(num_steps, task_names, max_step):
    cdf_tasks = []
    for task in task_names:
        cdf_tasks.append(get_CDF(num_steps[task], max_step))
    return cdf_tasks


def compute_search_cdf(scanpaths, annos, max_step, return_by_task=False):
    # compute search CDF
    task_names = np.unique([traj['task'] for traj in scanpaths])
    num_steps = get_num_steps(scanpaths, annos, task_names)
    cdf_tasks = get_mean_cdf(num_steps, task_names, max_step + 1)
    if return_by_task:
        return dict(zip(task_names, cdf_tasks))
    else:
        mean_cdf = np.mean(cdf_tasks, axis=0)
        std_cdf = np.std(cdf_tasks, axis=0)
        return mean_cdf, std_cdf


def calc_overlap_ratio(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    patch_area = float(patch_size[0] * patch_size[1])
    aoi_ratio = np.zeros((1, patch_num[1], patch_num[0]), dtype=np.float32)

    tl_x, tl_y = bbox[0], bbox[1]
    br_x, br_y = bbox[0] + bbox[2], bbox[1] + bbox[3]
    lx, ux = tl_x // patch_size[0], br_x // patch_size[0]
    ly, uy = tl_y // patch_size[1], br_y // patch_size[1]

    for x in range(lx, ux + 1):
        for y in range(ly, uy + 1):
            patch_tlx, patch_tly = x * patch_size[0], y * patch_size[1]
            patch_brx, patch_bry = patch_tlx + patch_size[
                0], patch_tly + patch_size[1]

            aoi_tlx = tl_x if patch_tlx < tl_x else patch_tlx
            aoi_tly = tl_y if patch_tly < tl_y else patch_tly
            aoi_brx = br_x if patch_brx > br_x else patch_brx
            aoi_bry = br_y if patch_bry > br_y else patch_bry

            aoi_ratio[0, y, x] = max((aoi_brx - aoi_tlx), 0) * max(
                (aoi_bry - aoi_tly), 0) / float(patch_area)

    return aoi_ratio


def foveal2mask(x, y, r, h, w):
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - x)**2 + (Y - y)**2)
    mask = dist <= r
    return mask.astype(np.float32)


def multi_hot_coding(bbox, patch_size, patch_num):
    """
    compute the overlaping ratio of the bbox and each patch (10x16)
    """
    thresh = 0
    aoi_ratio = calc_overlap_ratio(bbox, patch_size, patch_num)
    hot_ind = aoi_ratio > thresh
    while hot_ind.sum() == 0:
        thresh *= 0.8
        hot_ind = aoi_ratio > thresh

    aoi_ratio[hot_ind] = 1
    aoi_ratio[np.logical_not(hot_ind)] = 0

    return aoi_ratio[0]


def actions2scanpaths(actions, patch_num):
    # convert actions to scanpaths
    scanpaths = []
    for traj in actions:
        task_name, img_name, condition, actions = traj
        actions = actions.to(dtype=torch.float32)
        py = (actions // patch_num[0]) / float(patch_num[1])
        px = (actions % patch_num[0]) / float(patch_num[0])
        fixs = torch.stack([px, py])
        fixs = np.concatenate([np.array([[0.5], [0.5]]),
                               fixs.cpu().numpy()],
                              axis=1)
        scanpaths.append({
            'X': fixs[0] * 512,
            'Y': fixs[1] * 320,
            'name': img_name,
            'task': task_name,
            'condition': condition
        })
    return scanpaths


def preprocess_fixations(trajs,
                         patch_size,
                         patch_num,
                         im_h,
                         im_w,
                         truncate_num=-1,
                         need_label=True):
    fix_labels = []
    for traj in trajs:
        traj['X'][0], traj['Y'][0] = 256, 160
        label = pos_to_action(traj['X'][0], traj['Y'][0], patch_size,
                              patch_num)
        tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
        fixs = [(tar_x, tar_y)]
        label_his = [label]
        if truncate_num < 1:
            traj_len = len(traj['X'])
        else:
            traj_len = min(truncate_num, len(traj['X']))

        for i in range(1, traj_len):
            label = pos_to_action(traj['X'][i], traj['Y'][i], patch_size,
                                  patch_num)

            # remove returning fixations (enforce inhibition of return)
            if label in label_his:
                continue
            label_his.append(label)
            fix_label = (traj['name'], traj['task'], copy(fixs), label)

            # discretize fixations
            tar_x, tar_y = action_to_pos(label, patch_size, patch_num)
            fixs.append((tar_x, tar_y))

            fix_labels.append(fix_label)

    return fix_labels


def _file_at_step(step, name):
    return "save_{}_{}k{}.pkg".format(name, int(step // 1000),
                                      int(step % 1000))


def _file_best(name):
    return "trained_{}.pkg".format(name)


def save(global_step,
         model,
         optim,
         name,
         pkg_dir="",
         is_best=False,
         max_checkpoints=None):
    if optim is None:
        raise ValueError("cannot save without optimzier")
    state = {
        "global_step":
        global_step,
        # DataParallel wrap model in attr `module`.
        "model":
        model.module.state_dict()
        if hasattr(model, "module") else model.state_dict(),
        "optim":
        optim.state_dict(),
    }
    save_path = os.path.join(pkg_dir, _file_at_step(global_step, name))
    best_path = os.path.join(pkg_dir, _file_best(name))
    torch.save(state, save_path)
    print("[Checkpoint]: save to {} successfully".format(save_path))

    if is_best:
        copyfile(save_path, best_path)
    if max_checkpoints is not None:
        history = []
        for file_name in os.listdir(pkg_dir):
            if re.search("save_{}_\d*k\d*\.pkg".format(name), file_name):
                digits = file_name.replace("save_{}_".format(name),
                                           "").replace(".pkg", "").split("k")
                number = int(digits[0]) * 1000 + int(digits[1])
                history.append(number)
        history.sort()
        while len(history) > max_checkpoints:
            path = os.path.join(pkg_dir, _file_at_step(history[0]))
            print("[Checkpoint]: remove {} to keep {} checkpoints".format(
                path, max_checkpoints))
            if os.path.exists(path):
                os.remove(path)
            history.pop(0)


def load(step_or_path, model, name, optim=None, pkg_dir="", device=None):
    step = step_or_path
    save_path = None
    if isinstance(step, int):
        save_path = os.path.join(pkg_dir, _file_at_step(step, name))
    if isinstance(step, str):
        if pkg_dir is not None:
            if step == "best":
                save_path = os.path.join(pkg_dir, _file_best(name))
            else:
                save_path = os.path.join(pkg_dir, step)
        else:
            save_path = step
    if save_path is not None and not os.path.exists(save_path):
        print("[Checkpoint]: Failed to find {}".format(save_path))
        return
    if save_path is None:
        print("[Checkpoint]: Cannot load the checkpoint")
        return

    # begin to load
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    model.load_state_dict(state["model"])
    if optim is not None:
        optim.load_state_dict(state["optim"])

    print("[Checkpoint]: Load {} successfully".format(save_path))
    return global_step
