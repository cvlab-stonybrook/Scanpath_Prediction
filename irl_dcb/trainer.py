import os
import torch
import datetime
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .ppo import PPO
from .gail import GAIL
from . import utils
from .data import RolloutStorage, FakeDataRollout


class Trainer(object):
    def __init__(self, model, loaded_step, env, dataset, device, hparams):
        # setup logger
        date = str(datetime.datetime.now())
        date = date[:date.rfind(":")].replace("-", "")\
                                     .replace(":", "")\
                                     .replace(" ", "_")
        self.log_dir = os.path.join(hparams.Train.log_root, "log_" + date)
        self.checkpoints_dir = os.path.join(self.log_dir, "checkpoints")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # write hparams
        hparams.dump(self.log_dir)
        if not os.path.exists(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir)
        self.checkpoint_every = hparams.Train.checkpoint_every
        self.max_checkpoints = hparams.Train.max_checkpoints

        self.loaded_step = loaded_step
        self.env = env['train']
        self.env_valid = env['valid']
        self.generator = model['gen']
        self.discriminator = model['disc']
        self.bbox_annos = dataset['bbox_annos']
        self.human_mean_cdf = dataset['human_mean_cdf']
        self.device = device

        # image dataloader
        self.batch_size = hparams.Train.batch_size
        self.train_img_loader = DataLoader(dataset['img_train'],
                                           batch_size=self.batch_size,
                                           shuffle=True,
                                           num_workers=16)
        self.valid_img_loader = DataLoader(dataset['img_valid'],
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=16)

        # human gaze dataloader
        self.train_HG_loader = DataLoader(dataset['gaze_train'],
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=16)

        # training parameters
        self.gamma = hparams.Train.gamma
        self.adv_est = hparams.Train.adv_est
        self.tau = hparams.Train.tau
        self.max_traj_len = hparams.Data.max_traj_length
        self.n_epoches = hparams.Train.num_epoch
        self.n_steps = hparams.Train.num_step
        self.n_critic = hparams.Train.num_critic
        self.patch_num = hparams.Data.patch_num
        self.eval_every = hparams.Train.evaluate_every
        self.ppo = PPO(self.generator, hparams.PPO.lr,
                       hparams.Train.adam_betas, hparams.PPO.clip_param,
                       hparams.PPO.num_epoch, hparams.PPO.batch_size,
                       hparams.PPO.value_coef, hparams.PPO.entropy_coef)
        self.gail = GAIL(self.discriminator,
                         hparams.Train.gail_milestones,
                         None,
                         device,
                         lr=hparams.Train.gail_lr,
                         betas=hparams.Train.adam_betas)
        self.writer = SummaryWriter(self.log_dir)

    def train(self):
        self.generator.train()
        self.discriminator.train()
        self.global_step = self.loaded_step
        for i_epoch in range(self.n_epoches):
            for i_batch, batch in enumerate(self.train_img_loader):
                # run policy to collect trajactories
                print(
                    "generating state-action pairs to train discriminator...")
                trajs_all = []
                self.env.set_data(batch)
                return_train = 0.0
                for i_step in range(self.n_steps):
                    with torch.no_grad():
                        self.env.reset()
                        trajs = utils.collect_trajs(self.env, self.generator,
                                                    self.patch_num,
                                                    self.max_traj_len)
                        trajs_all.extend(trajs)
                smp_num = np.sum(list(map(lambda x: x['length'], trajs_all)))
                print("[{} {}] Collected {} state-action pairs".format(
                    i_epoch, i_batch, smp_num))

                # train discriminator (reward and value function)
                print("updating discriminator (step={})...".format(
                    self.gail.update_counter))

                fake_data = FakeDataRollout(trajs_all, self.batch_size)
                D_loss, D_real, D_fake = self.gail.update(
                    self.train_HG_loader, fake_data)

                self.writer.add_scalar("discriminator/fake_loss", D_fake,
                                       self.global_step)
                self.writer.add_scalar("discriminator/real_loss", D_real,
                                       self.global_step)
                print("Done updating discriminator!")

                # evaluate generator/policy
                if self.global_step > 0 and \
                   self.global_step % self.eval_every == 0:
                    print("evaluating policy...")

                    # generating scanpaths
                    all_actions = []
                    for i_sample in range(10):
                        for batch in self.valid_img_loader:
                            self.env_valid.set_data(batch)
                            img_names_batch = batch['img_name']
                            cat_names_batch = batch['cat_name']
                            with torch.no_grad():
                                self.env_valid.reset()
                                trajs = utils.collect_trajs(self.env_valid,
                                                            self.generator,
                                                            self.patch_num,
                                                            self.max_traj_len,
                                                            is_eval=True,
                                                            sample_action=True)
                                all_actions.extend([
                                    (cat_names_batch[i], img_names_batch[i],
                                     'present', trajs['actions'][:, i])
                                    for i in range(self.env_valid.batch_size)
                                ])
                    scanpaths = utils.actions2scanpaths(
                        all_actions, self.patch_num)
                    utils.cutFixOnTarget(scanpaths, self.bbox_annos)

                    # search effiency
                    mean_cdf, _ = utils.compute_search_cdf(
                        scanpaths, self.bbox_annos, self.max_traj_len)
                    self.writer.add_scalar('evaluation/TFP_step1', mean_cdf[1],
                                           self.global_step)
                    self.writer.add_scalar('evaluation/TFP_step3', mean_cdf[3],
                                           self.global_step)
                    self.writer.add_scalar('evaluation/TFP_step6', mean_cdf[6],
                                           self.global_step)

                    # probability mismatch
                    sad = np.sum(np.abs(self.human_mean_cdf - mean_cdf))
                    self.writer.add_scalar('evaluation/prob_mismatch', sad,
                                           self.global_step)

                # update generator/policy on every n_critic iter
                if i_batch % self.n_critic == 0:
                    print("updating policy...")
                    # update reward and value
                    with torch.no_grad():
                        for i in range(len(trajs_all)):
                            states = trajs_all[i]["curr_states"]
                            actions = trajs_all[i]["actions"].unsqueeze(1)
                            tids = trajs_all[i]['task_id']
                            rewards = F.logsigmoid(
                                self.discriminator(states, actions, tids))
                            trajs_all[i]["rewards"] = rewards

                    return_train = utils.process_trajs(trajs_all,
                                                       self.gamma,
                                                       mtd=self.adv_est,
                                                       tau=self.tau)
                    self.writer.add_scalar("generator/ppo_return",
                                           return_train, self.global_step)
                    print('average return = {:.3f}'.format(return_train))

                    # update policy
                    rollouts = RolloutStorage(trajs_all,
                                              shuffle=True,
                                              norm_adv=True)
                    loss = self.ppo.update(rollouts)
                    self.writer.add_scalar("generator/ppo_loss", loss,
                                           self.global_step)

                    print("Done updating policy")

                # checkpoints
                if self.global_step % self.checkpoint_every == 0 and \
                   self.global_step > 0:
                    utils.save(global_step=self.global_step,
                               model=self.generator,
                               optim=self.ppo.optimizer,
                               name='generator',
                               pkg_dir=self.checkpoints_dir,
                               is_best=True,
                               max_checkpoints=self.max_checkpoints)
                    utils.save(global_step=self.global_step,
                               model=self.discriminator,
                               optim=self.gail.optimizer,
                               name='discriminator',
                               pkg_dir=self.checkpoints_dir,
                               is_best=True,
                               max_checkpoints=self.max_checkpoints)

                self.global_step += 1

        self.writer.close()
