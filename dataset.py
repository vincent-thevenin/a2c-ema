import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from utils import Timer


class ReplayQueue():
    def __init__(self, capacity, use_priorities=False):
        self.capacity = capacity
        self.use_priorities = use_priorities

        self.errors = []
        self.memory = []
        self.probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.log_probs = []

    def push(self, sarst, error, value, log_prob):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            self.errors.pop(0)
            self.values.pop(0)

        self.memory.append(sarst)
        self.errors.append(error)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_sample_weights(self):
        exp =  np.exp(self.errors)
        self.probs = exp / sum(exp)

    def compute_advantages_and_returns(self, gamma, lamb, last_value, done):
        last_gae_lam = 0
        for i in reversed(range(len(self.memory))):
            if i == len(self.memory) - 1:
                next_values = last_value
                next_non_terminal = 1 - done
            else:
                next_values = self.values[i + 1]
                next_non_terminal = 1 - self.memory[i][4]
            delta = self.memory[i][2] + gamma * next_values * next_non_terminal - self.values[i]
            last_gae_lam = delta + gamma * lamb * next_non_terminal * last_gae_lam
            self.advantages.append(last_gae_lam)
        self.advantages = list(reversed(self.advantages))
        self.returns = [self.advantages[i] + self.values[i] for i in range(len(self.advantages))]

    def load(self, path):
        with open(path, 'rb') as f:
            self.memory = pickle.load(f)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.memory, f)

    def clear(self):
        self.errors = []
        self.memory = []
        self.probs = []
        self.values = []
        self.advantages = []
        self.returns = []
        self.log_probs = []


    def update_error(self, idxs, errors):
        for idx, error in zip(idxs, errors):
            self.errors[idx] = error.item()

    def __len__(self):
        # retrieve number of steps
        return len(self.memory)

class ModelDataset(Dataset):
    def __init__(self, replay: ReplayQueue):
        self.replay = replay  # replay is updated outside
        if self.replay.use_priorities:
            self.idx_sampler = lambda replay, idx: np.random.choice(
                len(replay),
                p = replay.probs
            )
        else:
            self.idx_sampler = lambda replay, idx: idx

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx):
        idx = self.idx_sampler(self.replay, idx)

        state, action, reward, next_state, is_terminal = self.replay.memory[idx]

        reward = torch.Tensor([reward])
        is_terminal = torch.Tensor([float(is_terminal)])

        return state, action, reward.unsqueeze(1), next_state, is_terminal.unsqueeze(1), idx

class RolloutDataset(Dataset):
    def __init__(self, replay: ReplayQueue, gamma: float, lambda_gae:float):
        self.replay = replay  # replay is updated outside
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        if self.replay.use_priorities:
            self.idx_sampler = lambda replay, idx: np.random.choice(
                len(replay),
                p = replay.probs
            )
        else:
            self.idx_sampler = lambda replay, idx: idx

    def __len__(self):
        return len(self.replay)

    def __getitem__(self, idx):
        with Timer("Dataset data loading", False):
            idx = self.idx_sampler(self.replay, idx)

            state, action, reward, next_state, is_terminal = self.replay.memory[idx]
            returns = torch.Tensor([self.replay.returns[idx]])
            advantages = torch.Tensor([self.replay.advantages[idx]])
            reward = torch.Tensor([reward])
            log_prob = torch.Tensor([self.replay.log_probs[idx]])

            return state, action, reward, next_state, is_terminal, returns, advantages, idx, log_prob


class CustomDataLoader():
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        self.idx = 0
        if self.shuffle:
            self.shuffled_idxs = np.random.permutation(len(self.dataset))
        return self

    def __next__(self):
        if self.idx >= self.__len__() - 1:
            self.__iter__()
            raise StopIteration
        else:
            if self.shuffle:
                idxs = self.shuffled_idxs[self.idx * self.batch_size:(self.idx + 1) * self.batch_size]
            else:
                idxs = self.idx * self.batch_size + np.arange(min(self.batch_size, len(self.dataset) - self.idx * self.batch_size))
            self.idx += 1

            # make batches
            # state, action, reward, next_state, is_terminal, idx
            r = []
            for idx in idxs:
                for i, rr in enumerate(self.dataset[idx]):
                    if len(r) <= i:
                        r.append([])
                    # check and convert tensor type
                    if not isinstance(rr, torch.Tensor):
                        rr = torch.tensor(rr)
                    r[i].append(rr)
            return tuple(torch.stack(rr) for rr in r)
            

    def __len__(self):
        return len(self.dataset) // self.batch_size + ((len(self.dataset) % self.batch_size) != 0)