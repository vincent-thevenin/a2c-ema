import numpy as np
import pickle
import torch
from torch.utils.data import Dataset


class ReplayQueue():
    def __init__(self, capacity, use_priorities=False):
        self.capacity = capacity
        self.use_priorities = use_priorities

        self.errors = []
        self.memory = []
        self.probs = []

    def push(self, sarst, error):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
            if self.use_priorities:
                self.errors.pop(0)

        self.memory.append(sarst)
        if self.use_priorities:
            self.errors.append(error)
            self.probs = np.exp(self.errors) / sum(np.exp(self.errors))

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

    def update_priorities(self, idxs, errors):
        if self.use_priorities:
            for idx, error in zip(idxs, errors):
                self.errors[idx] = error.item()
            self.probs = np.exp(self.errors) / sum(np.exp(self.errors))

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
        if self.idx * self.batch_size >= len(self.dataset):
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
        return len(self.dataset) // self.batch_size + min(1, len(self.dataset) % self.batch_size)
