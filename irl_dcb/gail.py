import torch
from torch import autograd
import torch.optim as optim
import torch.nn.functional as F


class GAIL():
    def __init__(self, discriminator, milestones, state_enc, device, lr,
                 betas):
        self.discriminator = discriminator
        self.state_enc = state_enc
        self.device = device
        self.optimizer = optim.Adam(self.discriminator.parameters(),
                                    lr=lr,
                                    betas=betas)
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=0.1)
        self.update_counter = 0

    def compute_grad_pen(self,
                         expert_states,
                         expert_act,
                         expert_task,
                         policy_states,
                         policy_act,
                         policy_task,
                         type='real',
                         lambda_=5):

        if type == 'mixed':
            expert_actions, expert_tids = self.discriminator.get_one_hot(
                expert_act, expert_task)
            policy_actions, policy_tids = self.discriminator.get_one_hot(
                policy_act, policy_task)

            bs = expert_states.size(0)
            alpha = torch.rand(bs, 1).to(expert_states.device)

            alpha = alpha.view(bs, 1, 1, 1)
            mixup_states = alpha.expand_as(
                expert_states) * expert_states.detach() + (1 - alpha.expand_as(
                    expert_states)) * policy_states.detach()
            alpha = alpha.view(bs, 1)
            mixup_actions = alpha.expand_as(
                expert_actions) * expert_actions.detach() + (
                    1 -
                    alpha.expand_as(expert_actions)) * policy_actions.detach()
            mixup_tids = alpha.expand_as(expert_tids) * expert_tids.detach(
            ) + (1 - alpha.expand_as(expert_tids)) * policy_tids.detach()
        elif type == 'real':
            mixup_states = expert_states.detach()
            mixup_actions = expert_act.detach()
            mixup_tids = expert_task.detach()

        mixup_states.requires_grad = True
        # mixup_actions.requires_grad = True
        # mixup_tids.requires_grad = True

        mixup_data = (mixup_states, mixup_actions, mixup_tids)
        disc = torch.sigmoid(self.discriminator(*mixup_data))
        ones = torch.ones(disc.size()).to(disc.device)
        grad = autograd.grad(outputs=disc,
                             inputs=mixup_states,
                             grad_outputs=ones,
                             create_graph=True,
                             retain_graph=True,
                             only_inputs=True)[0]

        grad_pen = lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()
        return grad_pen

    def update(self,
               true_data_loader,
               fake_data,
               iter_num=1,
               noisy_label_ratio=0):
        running_loss = 0.0
        print_every = fake_data.sample_num // (5 * fake_data.batch_size) + 1
        avg_loss = 0.0
        D_real, D_fake, D_grad = 0.0, 0.0, 0.0
        fake_sample_num = 0
        real_sample_num = 0

        for i_iter in range(iter_num):
            fake_data_generator = iter(fake_data.get_generator())
            for i_batch, true_batch in enumerate(true_data_loader):
                if i_batch == len(fake_data):
                    break
                fake_batch = next(fake_data_generator)

                with torch.no_grad():
                    if self.state_enc is None:
                        real_S = true_batch['true_state'].to(self.device)
                    else:
                        real_S = self.state_enc(true_batch['true_state'].to(
                            self.device))
                    real_A = true_batch['true_action']
                    real_tids = true_batch['task_id']

                fake_S, fake_A, fake_P, fake_tids = fake_batch
                fake_num, real_num = fake_S.size(0), real_S.size(0)
                if fake_num == 0 or real_num == 0:
                    break

                x_real = (real_S, real_A, real_tids)
                x_fake = (fake_S, fake_A, fake_tids)

                real_outputs = self.discriminator(*x_real)
                fake_outputs = self.discriminator(*x_fake)

                real_labels = torch.ones(real_outputs.size()).to(self.device)
                fake_labels = torch.zeros(fake_outputs.size()).to(self.device)

                # randomly flip labels of training data in order to
                # increase training stability
                if noisy_label_ratio > 0:
                    flip_num = int(real_labels.size(0) * noisy_label_ratio)
                    ind = torch.randint(real_labels.size(0), (flip_num, ))
                    real_labels[ind] = 0
                    fake_labels[ind] = 1
                expert_loss = F.binary_cross_entropy_with_logits(
                    real_outputs, real_labels)
                policy_loss = F.binary_cross_entropy_with_logits(
                    fake_outputs, fake_labels)

                gail_loss = expert_loss + policy_loss
                grad_pen = self.compute_grad_pen(*x_real, *x_fake)

                self.optimizer.zero_grad()
                (gail_loss + grad_pen).backward()
                self.optimizer.step()
                self.lr_scheduler.step()

                if i_iter == iter_num - 1:
                    avg_loss += gail_loss.item()
                    real_sample_num += real_num
                    fake_sample_num += fake_num
                    D_real += torch.sum(torch.sigmoid(real_outputs)).item()
                    D_fake += torch.sum(torch.sigmoid(fake_outputs)).item()

                running_loss += gail_loss.item()
                D_grad += grad_pen.item()
                if i_batch % print_every == print_every - 1:
                    print("[{}]: D_loss = {:.3f}, Grad_loss = {:.3f}".format(
                        i_batch + 1, running_loss / print_every,
                        D_grad / print_every))
                    D_grad = 0.0
                    running_loss = 0.0

                self.update_counter += 1

        return (avg_loss / fake_data.sample_num, D_real / real_sample_num,
                D_fake / fake_sample_num)
