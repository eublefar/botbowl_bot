"""Thanks to https://github.com/Curt-Park/rainbow-is-all-you-need"""
from typing import Dict
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import logging
from .base_agent import BaseAgent
from .replay_buffers.per_buffer import PrioritizedReplayBuffer
from .replay_buffers.replay_buffer import ReplayBuffer
from ..policies.categorical_net import CategoricalNet

logging.basicConfig(level=logging.INFO)


class PytorchRainbowDqn(BaseAgent):
    def __init__(self, **params):
        super().__init__(**params)

        obs_dim = self.observation_space.shape[0]
        action_dim = self.action_space.n

        self.memory = PrioritizedReplayBuffer(
            obs_dim,
            self.memory_size,
            self.batch_size,
            self.buffer_alpha,
            gamma=self.gamma,
        )
        self.memory_n = ReplayBuffer(
            obs_dim,
            self.memory_size,
            self.batch_size,
            n_step=self.n_step,
            gamma=self.gamma,
        )

        self.beta = self.buffer_beta_min
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(
            self.device
        )

        # networks: dqn, dqn_target
        self.dqn = (
            self.Policy(
                obs_dim,
                action_dim,
                atom_size=self.atom_size,
                support=self.support,
                **self.policy_parameters
            )
            .to(self.device)
            .train()
        )
        if not hasattr(self.dqn, "distribution"):
            raise NotImplementedError(
                "Policy does not have distribution(state) -> atoms method"
            )
        self.dqn_target = CategoricalNet(
            obs_dim, action_dim, atom_size=self.atom_size, support=self.support
        ).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        self.loss_per_batch = 0
        self.episode_step = 0
        # mode: train / test
        self.is_test = False

    def act(self, state: np.ndarray, global_step: int) -> np.ndarray:
        """Select an action from the input state."""
        selected_action = self.dqn.act(state)["q_values"].argmax()
        return selected_action.detach().cpu().numpy()

    def memorize(
        self,
        last_ob: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        ob: np.ndarray,
        global_step: int,
    ):
        first_in_seq = self.memory_n.store(last_ob, action, reward, ob, done)
        if first_in_seq:
            self.memory.store(*first_in_seq)

    def update(self, global_step: int):
        if len(self.memory) >= self.batch_size:
            self.update_model()

            self.beta = max(
                self.buffer_beta_min,
                self.beta
                - (self.buffer_beta_max - self.buffer_beta_min)
                * self.buffer_beta_decay,
            )
            self._target_update()
            self.dqn.reset_noise()
            self.dqn_target.reset_noise()

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]
        elementwise_loss = self._compute_dqn_loss(samples)

        samples = self.memory_n.sample_batch_from_idxs(indices)
        gamma = self.gamma ** self.n_step
        n_loss = self._compute_dqn_loss(samples, gamma)
        elementwise_loss += n_loss

        loss = torch.mean(elementwise_loss * weights)
        self.loss_per_batch += loss
        self.episode_step += 1
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)

        for p, v in self.dqn.named_parameters():
            if torch.isnan(v.grad).any():
                logging.warn("param {} nan".format(p))

        self.optimizer.step()

        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps

        self.memory.update_priorities(indices, new_priorities.squeeze())

        return loss.item()

    def metrics(self, episode_num: int):
        if episode_num % 10:
            loss_per_batch_mean = (
                (self.loss_per_batch / self.episode_step).detach().numpy()
                if self.episode_step != 0
                else 0
            )
            self.loss_per_batch = 0
            self.episode_step = 0
            grads = {}
            for i, (name, p) in enumerate(self.dqn.named_parameters()):
                if p.grad is not None:
                    avg = torch.mean(p).detach().numpy()
                    std = torch.std(p).detach().numpy() if p.squeeze().dim() != 0 else 0

                    grads.update({name + "_std": std, name + "_mean": avg})
            return {"loss_per_batch": loss_per_batch_mean, **grads}

    def _compute_dqn_loss(
        self, samples: Dict[str, np.ndarray], gamma: float = None
    ) -> torch.Tensor:
        """Return categorical dqn loss."""
        if gamma is None:
            gamma = self.gamma
        device = self.device  # for shortening the following lines
        state = samples["obs"]
        next_state = samples["next_obs"]
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state)["q_values"].argmax(1)
            next_dist = self.dqn_target.distribution(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.distribution(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_update(self):
        for w, w_target in zip(self.dqn.parameters(), self.dqn_target.parameters()):
            w_target.data = w_target.data * (1 - self.tau) + w.data * self.tau

    @staticmethod
    def get_default_parameters():
        return {
            "memory_size": 2000,
            "batch_size": 32,
            "seed": 777,
            "gamma": 0.99,
            "tau": 0.01,
            "buffer_alpha": 0.6,
            "buffer_beta_max": 0.9,
            "buffer_beta_min": 0.1,
            "buffer_beta_decay": 1 / 2000,
            "prior_eps": 1e-6,
            "k_mixtures": 5,
            "eps": 0.2,
            "atom_size": 51,
            "n_step": 3,
            "v_min": -600,
            "v_max": 300,
        }

    @staticmethod
    def get_default_policy():
        return {
            "class": "gym_loop.policies.categorical_net:CategoricalNet",
            "parameters": {},
        }
