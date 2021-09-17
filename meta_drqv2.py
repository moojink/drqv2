# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import meta_utils as utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    """Convolutional encoder for image-based observations."""
    def __init__(self, img_obs_shape, feature_dim):
        super().__init__()

        assert len(img_obs_shape) == 3
        self.num_filters = 32
        if img_obs_shape[1] == 84: # DeepMind control suite images are 84x84
            conv_out_size = 35
        elif img_obs_shape[1] == 128:
            conv_out_size = 57
        else:
            raise ValueError("Unsupported image size.")
        self.repr_dim = self.num_filters * conv_out_size * conv_out_size

        self.convnet = nn.Sequential(nn.Conv2d(img_obs_shape[0], self.num_filters, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                                     nn.ReLU())

        self.trunk = nn.Sequential(nn.Linear(self.repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        out = self.trunk(h)
        return out


class VIB(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim * 2, hidden_dim),
                                   nn.LayerNorm(hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, feature_dim * 2))
        self.apply(utils.weight_init)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, obs_repr_dim, proprio_obs_shape, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(obs_repr_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs_repr, std):
        mu = self.policy(obs_repr)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_repr_dim, proprio_obs_shape, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.Linear(obs_repr_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(obs_repr_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs_repr, action):
        h_action = torch.cat([obs_repr, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, view, img_obs_shape, proprio_obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, add_img_repr_loss, img_repr_loss_weight, use_vib, vib_kl_weight):
        self.view = view
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.feature_dim = feature_dim
        self.add_img_repr_loss = add_img_repr_loss
        self.img_repr_loss_weight = int(float(img_repr_loss_weight)) # weight on L2 loss b/t view 1 and view 3 representations
        self.use_vib = use_vib # variational information bottleneck on view 3 representation
        self.vib_kl_weight = float(vib_kl_weight)

        # models
        if self.view == 'both':
            self.encoder1 = Encoder(img_obs_shape, feature_dim).to(device)
            self.encoder3 = Encoder(img_obs_shape, feature_dim).to(device)
            if self.use_vib:
                self.vib = VIB(feature_dim=feature_dim, hidden_dim=feature_dim*2).to(device)
            obs_repr_dim = feature_dim * 2 + proprio_obs_shape # observation representation dim
        else:
            self.encoder = Encoder(img_obs_shape, feature_dim).to(device)
            obs_repr_dim = feature_dim + proprio_obs_shape # observation representation dim
        self.actor = Actor(obs_repr_dim, proprio_obs_shape, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(obs_repr_dim, proprio_obs_shape, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(obs_repr_dim, proprio_obs_shape, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if self.view == 'both':
            self.encoder_opt1 = torch.optim.Adam(self.encoder1.parameters(), lr=lr)
            self.encoder_opt3 = torch.optim.Adam(self.encoder3.parameters(), lr=lr)
        else:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        if self.view == 'both':
            self.encoder1.train(training)
            self.encoder3.train(training)
        else:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        if self.view == 'both':
            img_obs1, img_obs3, proprio_obs = obs
            img_obs1 = torch.as_tensor(img_obs1, device=self.device)
            img_obs3 = torch.as_tensor(img_obs3, device=self.device)
            encoder_out1 = self.encoder1(img_obs1.unsqueeze(0))
            encoder_out3 = self.encoder3(img_obs3.unsqueeze(0))
            proprio_obs = torch.as_tensor(proprio_obs, dtype=torch.float32, device=self.device)
            proprio_obs = proprio_obs.unsqueeze(0)
            if self.use_vib: # variational information bottleneck
                encoder_out_1_and_3 = torch.cat((encoder_out1.detach(), encoder_out3), dim=-1)
                vib_out = self.vib(encoder_out_1_and_3)
                means, log_stds = torch.split(vib_out, self.feature_dim, dim=1)
                eps = torch.reshape(torch.as_tensor(np.random.randn(*means.shape), dtype=torch.float32, device=self.device), means.shape) # sample from mean 0 std 1 gaussian
                vib_repr = means + eps * torch.exp(log_stds) # reparameterization trick
                obs_out = torch.cat((encoder_out1, vib_repr, proprio_obs), dim=-1)
            else:
                obs_out = torch.cat((encoder_out1, encoder_out3, proprio_obs), dim=-1)
        else:
            img_obs, proprio_obs = obs
            img_obs = torch.as_tensor(img_obs, device=self.device)
            encoder_out = self.encoder(img_obs.unsqueeze(0))
            proprio_obs = torch.as_tensor(proprio_obs, dtype=torch.float32, device=self.device)
            proprio_obs = proprio_obs.unsqueeze(0)
            obs_out = torch.cat((encoder_out, proprio_obs), dim=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs_out, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs_repr, action, reward, discount, next_obs_repr, step, orig_vib_out=None):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs_repr, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs_repr, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs_repr, action)
        if self.view == 'both' and self.add_img_repr_loss: # L2 loss b/t view 1 and view 3 representations
            Q1_target_Q_loss, Q2_target_Q_loss = F.mse_loss(Q1, target_Q), F.mse_loss(Q2, target_Q)
            img_repr1, img_repr3 = obs_repr[:, :self.feature_dim], obs_repr[:, self.feature_dim:self.feature_dim*2]
            img_repr1, img_repr3 = F.normalize(img_repr1, dim=1), F.normalize(img_repr3, dim=1)
            next_img_repr1, next_img_repr3 = next_obs_repr[:, :self.feature_dim], next_obs_repr[:, self.feature_dim:self.feature_dim*2]
            next_img_repr1, next_img_repr3 = F.normalize(next_img_repr1, dim=1), F.normalize(next_img_repr3, dim=1)
            # Stop gradients on view 1 representations. This will cause the gradient to only flow through view 3,
            # pushing view 3 closer to view 1 while leaving view 1 unaffected.
            img_repr1, next_img_repr1 = img_repr1.detach(), next_img_repr1.detach()
            img_repr_loss = F.mse_loss(img_repr1, img_repr3)
            next_img_repr_loss = F.mse_loss(next_img_repr1, next_img_repr3)
            batch_size = obs_repr.detach().shape[0]
            total_img_repr_loss = 1 / batch_size * self.img_repr_loss_weight * (img_repr_loss + next_img_repr_loss)
            critic_loss = Q1_target_Q_loss + Q2_target_Q_loss + total_img_repr_loss
        else:
            critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_vib: # variational information bottleneck
            # Add a KL divergence term to the objective to regularize the view 3 representation.
            #   p: p(z|x), encoder output distribution
            #   q: q(z), variational approximation of p(z), fixed as a standard normal distribution
            p_mean, p_log_std = torch.split(orig_vib_out, self.feature_dim, dim=1)
            p_var = torch.square(torch.exp(p_log_std))
            q_mean, q_var = torch.zeros_like(p_mean, dtype=p_mean.dtype), torch.ones_like(p_var, dtype=p_var.dtype)
            kl = 0.5 * ((q_var / p_var).log() + (p_var + (p_mean - q_mean).pow(2)).div(q_var) - 1)
            batch_size = kl.detach().shape[0]
            kl_loss = self.vib_kl_weight * 1 / batch_size * torch.sum(kl)
            critic_loss += kl_loss

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            if self.view == 'both' and self.add_img_repr_loss:
                metrics['critic_q1_target_q_loss'] = Q1_target_Q_loss.item()
                metrics['critic_q2_target_q_loss'] = Q2_target_Q_loss.item()
                metrics['critic_img_repr_loss'] = img_repr_loss.item()
                metrics['critic_next_img_repr_loss'] = next_img_repr_loss.item()
                metrics['critic_total_img_repr_loss'] = total_img_repr_loss.item()
            if self.view == 'both' and self.use_vib:
                metrics['kl_loss'] = kl_loss.item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        if self.view == 'both':
            self.encoder_opt1.zero_grad(set_to_none=True)
            self.encoder_opt3.zero_grad(set_to_none=True)
        else:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.view == 'both':
            self.encoder_opt1.step()
            self.encoder_opt3.step()
        else:
            self.encoder_opt.step()

        return metrics

    def update_actor(self, obs_repr, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs_repr, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs_repr, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        if self.view == 'both':
            img_obs1, img_obs3, proprio_obs, action, reward, discount, next_img_obs1, next_img_obs3, next_proprio_obs = utils.to_torch(batch, self.device)
            # augment
            img_obs_aug1 = self.aug(img_obs1.float())
            img_obs_aug3 = self.aug(img_obs3.float())
            next_img_obs_aug1 = self.aug(next_img_obs1.float())
            next_img_obs_aug3 = self.aug(next_img_obs3.float())
            # encode
            encoder_out1 = self.encoder1(img_obs_aug1)
            encoder_out3 = self.encoder3(img_obs_aug3)
            with torch.no_grad():
                next_encoder_out1 = self.encoder1(next_img_obs_aug1)
                next_encoder_out3 = self.encoder3(next_img_obs_aug3)
            if self.use_vib: # variational information bottleneck
                detached_encoder_out1 = encoder_out1.detach()
                encoder_out_1_and_3 = torch.cat((detached_encoder_out1, encoder_out3), dim=-1)
                vib_out = self.vib(encoder_out_1_and_3)
                orig_vib_out = torch.clone(vib_out)
                means, log_stds = torch.split(vib_out, self.feature_dim, dim=1)
                eps = torch.reshape(torch.as_tensor(np.random.randn(*means.shape), dtype=torch.float32, device=self.device), means.shape) # sample from mean 0 std 1 gaussian
                vib_repr = means + eps * torch.exp(log_stds) # reparameterization trick

                detached_next_encoder_out1 = next_encoder_out1.detach() # not really necessary b/c already applied torch.no_grad() to this, but leaving it for consistency
                next_encoder_out_1_and_3 = torch.cat((detached_next_encoder_out1, next_encoder_out3), dim=-1)
                next_vib_out = self.vib(next_encoder_out_1_and_3)
                means, log_stds = torch.split(next_vib_out, self.feature_dim, dim=1)
                eps = torch.reshape(torch.as_tensor(np.random.randn(*means.shape), dtype=torch.float32, device=self.device), means.shape) # sample from mean 0 std 1 gaussian
                next_vib_repr = means + eps * torch.exp(log_stds) # reparameterization trick
            obs_repr = torch.cat((encoder_out1, vib_repr, proprio_obs), dim=-1) # obs_repr = observation representation
            next_obs_repr = torch.cat((next_encoder_out1, next_vib_repr, next_proprio_obs), dim=-1)
        else:
            img_obs, proprio_obs, action, reward, discount, next_img_obs, next_proprio_obs = utils.to_torch(batch, self.device)
            # augment
            img_obs_aug = self.aug(img_obs.float())
            next_img_obs_aug = self.aug(next_img_obs.float())
            # encode
            encoder_out = self.encoder(img_obs_aug)
            with torch.no_grad():
                next_encoder_out = self.encoder(next_img_obs_aug)
            obs_repr = torch.cat((encoder_out, proprio_obs), dim=-1) # obs_repr = observation representation
            next_obs_repr = torch.cat((next_encoder_out, next_proprio_obs), dim=-1)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        if self.use_vib:
            metrics.update(self.update_critic(obs_repr, action, reward, discount, next_obs_repr, step, orig_vib_out))
        else:
            metrics.update(self.update_critic(obs_repr, action, reward, discount, next_obs_repr, step))

        # update actor
        metrics.update(self.update_actor(obs_repr.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
