import torch.nn as nn
import torch
import torch.nn.functional as F


# conditional version of IRL discriminator (on task) for 20x32
class LHF_Discriminator_Cond(nn.Module):
    def __init__(self, action_num, target_size, task_eye, ch):
        super(LHF_Discriminator_Cond, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(ch + target_size, 128, 3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128 + target_size, 64, 3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64 + target_size, 32, 3, padding=1)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32 + target_size, 1, 1)

        self.task_eye = task_eye

    def get_one_hot(self, tid):
        task_onehot = self.task_eye[tid]
        return task_onehot

    def modulate_features(self, feat_maps, tid_onehot):
        """modulat feature maps using task vector"""
        bs, _, h, w = feat_maps.size()
        task_maps = tid_onehot.expand(bs, tid_onehot.size(1), h, w)
        return torch.cat([feat_maps, task_maps], dim=1)

    def forward(self, x, action, tid):
        """ output probability of x being true data"""
        bs, _, h, w = x.size()
        tid_onehot = self.get_one_hot(tid)
        tid_onehot = tid_onehot.view(bs, tid_onehot.size(1), 1, 1)

        x = self.modulate_features(x, tid_onehot)
        x = torch.relu(self.conv1_bn(self.conv1(x)))
        if h == 80:
            x = self.max_pool(x)
        x = self.modulate_features(x, tid_onehot)
        x = torch.relu(self.conv2_bn(self.conv2(x)))
        if h == 80:
            x = self.max_pool(x)
        x = self.modulate_features(x, tid_onehot)
        x = torch.relu(self.conv3_bn(self.conv3(x)))
        x = self.modulate_features(x, tid_onehot)
        x = self.conv4(x).view(bs, -1)
        if action is None:  # return whole reward map
            return x
        else:
            return x[torch.arange(bs), action.squeeze()]


# conditional version of IRL generator (on task)
class LHF_Policy_Cond_Small(nn.Module):
    def __init__(self, action_num, target_size, task_eye, ch):
        super(LHF_Policy_Cond_Small, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.feat_enc = nn.Conv2d(ch + target_size, 128, 5, padding=2)
        self.feat_enc_bn = nn.BatchNorm2d(128)

        # actor
        self.actor1 = nn.Conv2d(128 + target_size, 64, 3, padding=1)
        self.actor1_bn = nn.BatchNorm2d(64)
        self.actor2 = nn.Conv2d(64 + target_size, 32, 3, padding=1)
        self.actor2_bn = nn.BatchNorm2d(32)
        self.actor3 = nn.Conv2d(32 + target_size, 1, 1)
        
        # critic
        self.critic0 = nn.Conv2d(128 + target_size, 128, 3)
        self.critic0_bn = nn.BatchNorm2d(128)
        self.critic1 = nn.Conv2d(128 + target_size, 256, 3)
        self.critic1_bn = nn.BatchNorm2d(256)
        self.critic2 = nn.Linear(256 + target_size, 64)
        self.critic2_bn = nn.BatchNorm1d(64)
        self.critic3 = nn.Linear(64, 1)

        self.task_eye = task_eye

    def get_one_hot(self, tid):
        task_onehot = self.task_eye[tid]
        return task_onehot

    def modulate_features(self, feat_maps, tid_onehot):
        """modulat feature maps using task vector"""
        bs, _, h, w = feat_maps.size()
        task_maps = tid_onehot.expand(bs, tid_onehot.size(1), h, w)
        return torch.cat([feat_maps, task_maps], dim=1)

    def forward(self, x, tid, act_only=False):
        """ output the action probability"""
        bs, _, h, w = x.size()
        tid_onehot = self.get_one_hot(tid)
        tid_onehot = tid_onehot.view(bs, tid_onehot.size(1), 1, 1)

        x = self.modulate_features(x, tid_onehot)
        x = torch.relu(self.feat_enc_bn(self.feat_enc(x)))        
        x = self.modulate_features(x, tid_onehot)
        act_logits = torch.relu(self.actor1_bn(self.actor1(x)))
        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = torch.relu(self.actor2_bn(self.actor2(act_logits)))
        act_logits = self.modulate_features(act_logits, tid_onehot)
        act_logits = self.actor3(act_logits).view(bs, -1)
        
        act_probs = F.softmax(act_logits, dim=-1)
        if act_only:
            return act_probs, None

        x = self.max_pool(torch.relu(self.critic0_bn(self.critic0(x))))
        x = self.modulate_features(x, tid_onehot)
        x = self.max_pool(torch.relu(self.critic1_bn(self.critic1(x))))
        x = x.view(bs, x.size(1), -1).mean(dim=-1)
        x = torch.cat([x, tid_onehot.squeeze()], dim=1)
        x = torch.relu(self.critic2_bn(self.critic2(x)))
        state_values = self.critic3(x)
        return act_probs, state_values
